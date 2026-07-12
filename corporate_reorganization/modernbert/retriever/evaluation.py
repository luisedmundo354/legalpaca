from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeVar

import torch
import torch.distributed as dist

from .data import CorpusPassage, PassageIndexTable, QueryExample
from .markup import SLOT_TOKEN
from .query_views import QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED, select_query_text


FOLD_GLOBAL_VALIDATION_SCHEMA_VERSION = 1
FOLD_GLOBAL_RESULT_SCHEMA_VERSION = 1
VALIDATION_FORWARD_STEPS = 7
VALIDATION_KS = (1, 5, 10, 20)
VALIDATION_MAX_LEN_QUERY = 4096
VALIDATION_MAX_LEN_PASSAGE = 500
VALIDATION_QUERY_BATCH_CAP = 4
VALIDATION_PASSAGE_BATCH_CAP = 38
VALIDATION_WORLD_SIZE = 4
VALIDATION_QUERY_COUNT = 98
VALIDATION_CASE_COUNTS = (8, 9)
VALIDATION_PASSAGE_COUNTS = (1054, 1055, 1060, 1062)
VALIDATION_PRIMARY_METRIC = "eval_validation_case_macro_set_recall_at_20"
VALIDATION_SECONDARY_METRIC = (
    "eval_validation_case_macro_first_gold_reciprocal_rank_full_ranking"
)
CONTROLLED_VALIDATION_QUERY_VIEWS = (QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED)
INVALID_GLOBAL_POSITION = -1

_QUERY_METRIC_NAMES = tuple(
    [f"hit_at_{k}" for k in VALIDATION_KS]
    + [f"set_recall_at_{k}" for k in VALIDATION_KS]
    + [f"exact_target_recovery_at_{k}" for k in VALIDATION_KS]
    + ["first_gold_reciprocal_rank_full_ranking", "candidate_count"]
)
_PER_QUERY_KEYS = frozenset(
    {
        "query_id",
        "doc_id",
        "gold_count",
        "first_gold_rank",
        *_QUERY_METRIC_NAMES,
    }
)
_PER_CASE_KEYS = frozenset({"doc_id", "query_count", "metrics"})
_AGGREGATE_METRIC_KEYS = frozenset(
    {
        "eval_validation_num_queries",
        "eval_validation_num_cases",
        "eval_validation_num_passages",
        *(
            f"eval_validation_{aggregation}_{metric_name}"
            for aggregation in ("query_micro", "case_macro")
            for metric_name in _QUERY_METRIC_NAMES
        ),
        "eval_validation_query_micro_mrr_full_ranking",
        "eval_validation_case_macro_mrr_full_ranking",
    }
)

_T = TypeVar("_T")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _string_list_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(values)).encode("utf-8")).hexdigest()


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze(value: Any) -> Any:
    """Own a recursively immutable copy of a JSON-shaped value."""

    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    if type(value) is tuple:
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return a recursively independent JSON-shaped copy."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FoldGlobalValidationQuery:
    query_id: str
    doc_id: str
    query_text: str
    gold_passage_indices: tuple[int, ...]


@dataclass(frozen=True)
class FoldGlobalValidationData:
    schema_version: int
    role: str
    query_view: str
    case_ids: tuple[str, ...]
    queries: tuple[FoldGlobalValidationQuery, ...]
    passage_indices: tuple[int, ...]
    passage_doc_ids: tuple[str, ...]
    case_ids_sha256: str
    query_ids_sha256: str
    passage_ids_sha256: str
    contract_sha256: str

    @property
    def query_count(self) -> int:
        return len(self.queries)

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @property
    def passage_count(self) -> int:
        return len(self.passage_indices)


@dataclass(frozen=True)
class FoldGlobalValidationResult:
    schema_version: int
    metrics: Mapping[str, float]
    per_query: tuple[Mapping[str, Any], ...]
    per_case: tuple[Mapping[str, Any], ...]
    ranking_sha256: str
    case_ids_sha256: str
    query_ids_sha256: str
    passage_ids_sha256: str
    validation_contract_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metrics": _thaw(self.metrics),
            "per_query": _thaw(self.per_query),
            "per_case": _thaw(self.per_case),
            "ranking_sha256": self.ranking_sha256,
            "case_ids_sha256": self.case_ids_sha256,
            "query_ids_sha256": self.query_ids_sha256,
            "passage_ids_sha256": self.passage_ids_sha256,
            "validation_contract_sha256": self.validation_contract_sha256,
        }


def _validate_exact_strings(
    values: object,
    *,
    name: str,
    require_unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list or tuple")
    result: list[str] = []
    for index, value in enumerate(values):
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError(f"{name}[{index}] must be a non-empty exact stripped string")
        result.append(value)
    if require_unique and len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(result)


def _validation_contract_payload(
    validation_data: FoldGlobalValidationData,
    passage_index_table: PassageIndexTable,
) -> dict[str, Any]:
    return {
        "schema_version": validation_data.schema_version,
        "role": validation_data.role,
        "query_view": validation_data.query_view,
        "passage_index_sha256": passage_index_table.sha256,
        "case_ids": list(validation_data.case_ids),
        "queries": [
            {
                "query_id": query.query_id,
                "doc_id": query.doc_id,
                "query_text": query.query_text,
                "gold_passage_indices": list(query.gold_passage_indices),
            }
            for query in validation_data.queries
        ],
        "passages": [
            {
                "passage_index": passage_index,
                "passage_id": passage_index_table.id_for_index(passage_index),
                "doc_id": doc_id,
                "text": passage_index_table.text_for_index(passage_index),
            }
            for passage_index, doc_id in zip(
                validation_data.passage_indices,
                validation_data.passage_doc_ids,
            )
        ],
    }


def _validate_validation_data(
    validation_data: object,
    passage_index_table: PassageIndexTable,
) -> FoldGlobalValidationData:
    """Recompute every stored identity rather than trusting dataclass fields."""

    if not isinstance(validation_data, FoldGlobalValidationData):
        raise TypeError("validation_data must be FoldGlobalValidationData")
    if validation_data.schema_version != FOLD_GLOBAL_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Unsupported fold-global validation schema version")
    if validation_data.role != "validation":
        raise ValueError("Fold-global validation data must have role='validation'")
    if validation_data.query_view not in CONTROLLED_VALIDATION_QUERY_VIEWS:
        raise ValueError("Fold-global validation data has an unsupported query view")
    if not isinstance(passage_index_table, PassageIndexTable):
        raise TypeError("passage_index_table must be a PassageIndexTable")

    case_ids = _validate_exact_strings(validation_data.case_ids, name="case_ids")
    if case_ids != tuple(sorted(case_ids)):
        raise ValueError("Fold-global validation case IDs must be lexicographically sorted")
    if validation_data.case_ids_sha256 != _string_list_sha256(case_ids):
        raise ValueError("Fold-global validation case inventory digest changed")
    case_id_set = set(case_ids)

    if type(validation_data.passage_indices) is not tuple or not validation_data.passage_indices:
        raise ValueError("Fold-global validation passage_indices must be a non-empty tuple")
    passage_indices: list[int] = []
    for position, passage_index in enumerate(validation_data.passage_indices):
        if type(passage_index) is not int:
            raise TypeError(f"passage_indices[{position}] must be an exact int")
        passage_index_table.id_for_index(passage_index)
        passage_indices.append(passage_index)
    if passage_indices != sorted(passage_indices) or len(passage_indices) != len(
        set(passage_indices)
    ):
        raise ValueError("Fold-global validation passage indices must be sorted and unique")

    passage_doc_ids = _validate_exact_strings(
        validation_data.passage_doc_ids,
        name="passage_doc_ids",
        require_unique=False,
    )
    if len(passage_doc_ids) != len(passage_indices):
        raise ValueError("Passage indices and passage document IDs must align")
    if set(passage_doc_ids) != case_id_set:
        raise ValueError("Fold-global passages do not cover exactly the validation cases")
    passage_index_set = set(passage_indices)
    passage_doc_id_by_index = dict(zip(passage_indices, passage_doc_ids))
    passage_ids = tuple(
        passage_index_table.id_for_index(passage_index)
        for passage_index in passage_indices
    )
    if validation_data.passage_ids_sha256 != _string_list_sha256(passage_ids):
        raise ValueError("Fold-global validation passage inventory digest changed")

    if type(validation_data.queries) is not tuple or not validation_data.queries:
        raise ValueError("Fold-global validation queries must be a non-empty tuple")
    query_ids: list[str] = []
    query_case_ids: set[str] = set()
    for position, query in enumerate(validation_data.queries):
        if not isinstance(query, FoldGlobalValidationQuery):
            raise TypeError(f"queries[{position}] must be FoldGlobalValidationQuery")
        if (
            type(query.query_id) is not str
            or not query.query_id
            or query.query_id.strip() != query.query_id
        ):
            raise ValueError(f"queries[{position}] has an invalid query_id")
        if type(query.doc_id) is not str or query.doc_id not in case_id_set:
            raise ValueError(f"Query {query.query_id!r} is outside the validation cases")
        if type(query.query_text) is not str or not query.query_text.strip():
            raise ValueError(f"Query {query.query_id!r} has empty text")
        if query.query_text.count(SLOT_TOKEN) != 1:
            raise ValueError(
                f"Query {query.query_id!r} must contain exactly one {SLOT_TOKEN}"
            )
        if type(query.gold_passage_indices) is not tuple or not query.gold_passage_indices:
            raise ValueError(f"Query {query.query_id!r} has no gold passage indices")
        gold_indices: list[int] = []
        for gold_position, passage_index in enumerate(query.gold_passage_indices):
            if type(passage_index) is not int:
                raise TypeError(
                    f"Query {query.query_id!r} gold[{gold_position}] must be an exact int"
                )
            if passage_index not in passage_index_set:
                raise ValueError(
                    f"Query {query.query_id!r} has a gold outside the validation pool"
                )
            if passage_doc_id_by_index[passage_index] != query.doc_id:
                raise ValueError(
                    f"Query {query.query_id!r} has a gold from another case"
                )
            gold_indices.append(passage_index)
        if gold_indices != sorted(gold_indices) or len(gold_indices) != len(set(gold_indices)):
            raise ValueError(
                f"Query {query.query_id!r} gold passage indices must be sorted and unique"
            )
        query_ids.append(query.query_id)
        query_case_ids.add(query.doc_id)
    if query_ids != sorted(query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError("Fold-global validation query IDs must be sorted and unique")
    if query_case_ids != case_id_set:
        raise ValueError("Fold-global queries do not cover exactly the validation cases")
    if validation_data.query_ids_sha256 != _string_list_sha256(query_ids):
        raise ValueError("Fold-global validation query inventory digest changed")

    expected_contract = _sha256_canonical(
        _validation_contract_payload(validation_data, passage_index_table)
    )
    if validation_data.contract_sha256 != expected_contract:
        raise ValueError("Fold-global validation canonical contract digest changed")
    return validation_data


def build_fold_global_validation_data(
    *,
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    passage_index_table: PassageIndexTable,
    validation_case_ids: Sequence[str],
    expected_query_count: int,
    expected_passage_count: int,
    query_view: str,
) -> FoldGlobalValidationData:
    """Build the corrected-v2 validation role from the frozen fold case IDs."""

    if not isinstance(passage_index_table, PassageIndexTable):
        raise TypeError("passage_index_table must be a PassageIndexTable")
    if not isinstance(corpus_by_passage_id, Mapping) or not corpus_by_passage_id:
        raise ValueError("corpus_by_passage_id must be a non-empty mapping")
    if tuple(sorted(corpus_by_passage_id)) != passage_index_table.passage_ids:
        raise ValueError("Corpus keys and the immutable passage index table disagree")
    for passage_index, passage_id in enumerate(passage_index_table.passage_ids):
        passage = corpus_by_passage_id[passage_id]
        if not isinstance(passage, CorpusPassage):
            raise TypeError(f"Corpus record {passage_id!r} must be CorpusPassage")
        if passage.passage_id != passage_id:
            raise ValueError(
                f"Corpus key {passage_id!r} does not match record identity "
                f"{passage.passage_id!r}"
            )
        if (
            type(passage.doc_id) is not str
            or not passage.doc_id
            or passage.doc_id.strip() != passage.doc_id
        ):
            raise ValueError(f"Corpus passage {passage_id!r} has an invalid doc_id")
        if passage.text != passage_index_table.text_for_index(passage_index):
            raise ValueError(
                f"Corpus passage {passage_id!r} text disagrees with the immutable index table"
            )
    if type(expected_query_count) is not int or expected_query_count < 1:
        raise ValueError("expected_query_count must be a positive exact int")
    if type(expected_passage_count) is not int or expected_passage_count < 1:
        raise ValueError("expected_passage_count must be a positive exact int")
    if type(query_view) is not str or query_view not in CONTROLLED_VALIDATION_QUERY_VIEWS:
        raise ValueError(
            f"Controlled validation query_view must be one of "
            f"{CONTROLLED_VALIDATION_QUERY_VIEWS}; got {query_view!r}"
        )

    supplied_case_ids = _validate_exact_strings(validation_case_ids, name="validation_case_ids")
    case_ids = tuple(sorted(supplied_case_ids))
    case_id_set = set(case_ids)

    if not isinstance(all_queries, Sequence) or isinstance(all_queries, (str, bytes)):
        raise TypeError("all_queries must be a sequence of QueryExample records")
    global_query_ids: set[str] = set()
    selected_source_queries: list[QueryExample] = []
    for position, query in enumerate(all_queries):
        if not isinstance(query, QueryExample):
            raise TypeError(f"all_queries[{position}] must be a QueryExample")
        if type(query.query_id) is not str or not query.query_id:
            raise ValueError(f"all_queries[{position}] has an invalid query_id")
        if query.query_id in global_query_ids:
            raise ValueError(f"Duplicate global query_id={query.query_id!r}")
        global_query_ids.add(query.query_id)
        if query.doc_id in case_id_set:
            selected_source_queries.append(query)

    passage_indices = tuple(
        passage_index
        for passage_index, passage_id in enumerate(passage_index_table.passage_ids)
        if corpus_by_passage_id[passage_id].doc_id in case_id_set
    )
    if len(passage_indices) != expected_passage_count:
        raise RuntimeError(
            f"Fold-global validation has {len(passage_indices)} passages; "
            f"expected {expected_passage_count}"
        )
    passage_case_ids = {
        corpus_by_passage_id[passage_index_table.id_for_index(index)].doc_id
        for index in passage_indices
    }
    if passage_case_ids != case_id_set:
        raise RuntimeError(
            "Fold-global passage cases disagree with the validation role: "
            f"actual={sorted(passage_case_ids)}, expected={list(case_ids)}"
        )
    validation_passage_index_set = set(passage_indices)

    selected_source_queries.sort(key=lambda query: query.query_id)
    if len(selected_source_queries) != expected_query_count:
        raise RuntimeError(
            f"Fold-global validation has {len(selected_source_queries)} queries; "
            f"expected {expected_query_count}"
        )
    query_case_ids = {query.doc_id for query in selected_source_queries}
    if query_case_ids != case_id_set:
        raise RuntimeError(
            "Fold-global query cases disagree with the validation role: "
            f"actual={sorted(query_case_ids)}, expected={list(case_ids)}"
        )

    validation_queries: list[FoldGlobalValidationQuery] = []
    for query in selected_source_queries:
        if type(query.positive_passage_ids) is not list or not query.positive_passage_ids:
            raise ValueError(f"Validation query {query.query_id!r} has no gold passages")
        gold_ids = _validate_exact_strings(
            query.positive_passage_ids,
            name=f"query[{query.query_id}].positive_passage_ids",
        )
        gold_indices: list[int] = []
        for passage_id in gold_ids:
            passage_index = passage_index_table.index_for_id(passage_id)
            if passage_index not in validation_passage_index_set:
                raise RuntimeError(
                    f"Gold passage {passage_id!r} for query {query.query_id!r} "
                    "is outside its fold-global validation role"
                )
            passage = corpus_by_passage_id[passage_id]
            if passage.doc_id != query.doc_id:
                raise RuntimeError(
                    f"Gold passage {passage_id!r} belongs to case {passage.doc_id!r}, "
                    f"not query case {query.doc_id!r}"
                )
            gold_indices.append(passage_index)
        query_text = select_query_text(query, query_view=query_view)
        validation_queries.append(
            FoldGlobalValidationQuery(
                query_id=query.query_id,
                doc_id=query.doc_id,
                query_text=query_text,
                gold_passage_indices=tuple(sorted(gold_indices)),
            )
        )

    query_ids = tuple(query.query_id for query in validation_queries)
    passage_ids = tuple(
        passage_index_table.id_for_index(passage_index) for passage_index in passage_indices
    )
    passage_doc_ids = tuple(
        corpus_by_passage_id[passage_id].doc_id for passage_id in passage_ids
    )
    validation_data = FoldGlobalValidationData(
        schema_version=FOLD_GLOBAL_VALIDATION_SCHEMA_VERSION,
        role="validation",
        query_view=query_view,
        case_ids=case_ids,
        queries=tuple(validation_queries),
        passage_indices=passage_indices,
        passage_doc_ids=passage_doc_ids,
        case_ids_sha256=_string_list_sha256(case_ids),
        query_ids_sha256=_string_list_sha256(query_ids),
        passage_ids_sha256=_string_list_sha256(passage_ids),
        contract_sha256="",
    )
    validation_data = replace(
        validation_data,
        contract_sha256=_sha256_canonical(
            _validation_contract_payload(validation_data, passage_index_table)
        ),
    )
    return _validate_validation_data(validation_data, passage_index_table)


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"Cannot average an empty metric collection for {name}")
    result = math.fsum(float(value) for value in values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite aggregate metric {name}={result}")
    return result


def _ranking_for_scores(scores: torch.Tensor, passage_ids: Sequence[str]) -> list[int]:
    if scores.ndim != 1 or scores.shape[0] != len(passage_ids):
        raise ValueError("One query score row must align with every candidate passage")
    score_values = [float(value) for value in scores.tolist()]
    if any(not math.isfinite(value) for value in score_values):
        raise FloatingPointError("A validation score row contains a non-finite value")
    return sorted(
        range(len(passage_ids)),
        key=lambda position: (-score_values[position], passage_ids[position]),
    )


def compute_fold_global_metrics_from_embeddings(
    *,
    query_embeddings: torch.Tensor,
    passage_embeddings: torch.Tensor,
    validation_data: FoldGlobalValidationData,
    passage_index_table: PassageIndexTable,
    ks: Sequence[int] = VALIDATION_KS,
) -> FoldGlobalValidationResult:
    """Compute complete stable fold-global rankings and locked metrics."""

    validation_data = _validate_validation_data(
        validation_data,
        passage_index_table,
    )
    normalized_ks = tuple(ks)
    if normalized_ks != VALIDATION_KS or any(
        type(value) is not int for value in normalized_ks
    ):
        raise ValueError(f"Validation k values must be exactly {VALIDATION_KS}")
    for name, tensor, rows in (
        ("query_embeddings", query_embeddings, validation_data.query_count),
        ("passage_embeddings", passage_embeddings, validation_data.passage_count),
    ):
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a rank-2 floating tensor")
        if tensor.shape[0] != rows or tensor.shape[1] < 1:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}; expected {rows} rows")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains non-finite values")
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Query and passage embedding dimensions disagree")

    query_matrix = query_embeddings.detach().to(device="cpu", dtype=torch.float32)
    passage_matrix = passage_embeddings.detach().to(device="cpu", dtype=torch.float32)
    scores = query_matrix @ passage_matrix.T
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Fold-global score matrix contains non-finite values")

    passage_ids = tuple(
        passage_index_table.id_for_index(index) for index in validation_data.passage_indices
    )
    local_position_by_corpus_index = {
        corpus_index: position
        for position, corpus_index in enumerate(validation_data.passage_indices)
    }
    metric_names = _QUERY_METRIC_NAMES

    ranking_digest = hashlib.sha256()
    per_query: list[dict[str, Any]] = []
    for query_position, query in enumerate(validation_data.queries):
        gold_local_positions = {
            local_position_by_corpus_index[corpus_index]
            for corpus_index in query.gold_passage_indices
        }
        ranking = _ranking_for_scores(scores[query_position], passage_ids)
        first_gold_rank = next(
            (
                rank
                for rank, passage_position in enumerate(ranking, start=1)
                if passage_position in gold_local_positions
            ),
            0,
        )
        if first_gold_rank < 1:
            raise RuntimeError(f"Validation ranking contains no gold for query {query.query_id!r}")
        ranked_passage_ids = [passage_ids[position] for position in ranking]
        ranking_digest.update(
            (
                _canonical_json(
                    {"query_id": query.query_id, "ranked_passage_ids": ranked_passage_ids}
                )
                + "\n"
            ).encode("utf-8")
        )

        record: dict[str, Any] = {
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "gold_count": len(gold_local_positions),
            "first_gold_rank": first_gold_rank,
            "first_gold_reciprocal_rank_full_ranking": 1.0 / first_gold_rank,
            "candidate_count": float(validation_data.passage_count),
        }
        for k in normalized_ks:
            top_k = set(ranking[:k])
            recovered = len(top_k & gold_local_positions)
            record[f"hit_at_{k}"] = 1.0 if recovered > 0 else 0.0
            record[f"set_recall_at_{k}"] = recovered / len(gold_local_positions)
            record[f"exact_target_recovery_at_{k}"] = (
                1.0 if gold_local_positions.issubset(top_k) else 0.0
            )
        per_query.append(record)

    per_case: list[dict[str, Any]] = []
    for case_id in validation_data.case_ids:
        case_rows = [record for record in per_query if record["doc_id"] == case_id]
        if not case_rows:
            raise RuntimeError(f"Validation case {case_id!r} has no query metrics")
        case_metrics = {
            metric_name: _mean(
                [float(record[metric_name]) for record in case_rows],
                name=f"case[{case_id}].{metric_name}",
            )
            for metric_name in metric_names
        }
        per_case.append(
            {
                "doc_id": case_id,
                "query_count": len(case_rows),
                "metrics": case_metrics,
            }
        )

    metrics: dict[str, float] = {
        "eval_validation_num_queries": float(validation_data.query_count),
        "eval_validation_num_cases": float(validation_data.case_count),
        "eval_validation_num_passages": float(validation_data.passage_count),
    }
    for metric_name in metric_names:
        metrics[f"eval_validation_query_micro_{metric_name}"] = _mean(
            [float(record[metric_name]) for record in per_query],
            name=f"query_micro.{metric_name}",
        )
        metrics[f"eval_validation_case_macro_{metric_name}"] = _mean(
            [float(record["metrics"][metric_name]) for record in per_case],
            name=f"case_macro.{metric_name}",
        )
    metrics["eval_validation_query_micro_mrr_full_ranking"] = metrics[
        "eval_validation_query_micro_first_gold_reciprocal_rank_full_ranking"
    ]
    metrics["eval_validation_case_macro_mrr_full_ranking"] = metrics[
        VALIDATION_SECONDARY_METRIC
    ]
    if VALIDATION_PRIMARY_METRIC not in metrics or VALIDATION_SECONDARY_METRIC not in metrics:
        raise RuntimeError("Fold-global validation did not produce both selection metrics")
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Fold-global validation produced a non-finite aggregate metric")

    return _result_from_payload(
        {
            "schema_version": FOLD_GLOBAL_RESULT_SCHEMA_VERSION,
            "metrics": metrics,
            "per_query": per_query,
            "per_case": per_case,
            "ranking_sha256": ranking_digest.hexdigest(),
            "case_ids_sha256": validation_data.case_ids_sha256,
            "query_ids_sha256": validation_data.query_ids_sha256,
            "passage_ids_sha256": validation_data.passage_ids_sha256,
            "validation_contract_sha256": validation_data.contract_sha256,
        },
        validation_data,
        passage_index_table,
    )


def _balanced_nonempty_chunks(
    values: tuple[int, ...], *, chunk_count: int
) -> tuple[tuple[int, ...], ...]:
    if type(chunk_count) is not int or chunk_count < 1:
        raise ValueError("chunk_count must be a positive exact int")
    if len(values) < chunk_count:
        raise ValueError(
            f"Cannot split {len(values)} validation rows into {chunk_count} non-empty chunks"
        )
    base_size, remainder = divmod(len(values), chunk_count)
    chunks: list[tuple[int, ...]] = []
    start = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < remainder else 0)
        stop = start + size
        chunks.append(values[start:stop])
        start = stop
    if start != len(values) or any(not chunk for chunk in chunks):
        raise RuntimeError("Balanced validation chunk construction failed")
    return tuple(chunks)


def _tokenize_validation_texts(
    tokenizer,
    texts: list[str],
    *,
    truncation_side: str,
    max_length: int,
    require_slot: bool,
) -> Mapping[str, torch.Tensor]:
    if not texts or any(type(text) is not str or not text for text in texts):
        raise ValueError("Validation tokenizer inputs must be non-empty exact strings")
    if truncation_side not in ("left", "right"):
        raise ValueError("Validation truncation_side must be left or right")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Validation max_length must be a positive exact int")
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = original_side
    if not isinstance(tokens, Mapping):
        raise TypeError("Validation tokenizer did not return a mapping")
    input_ids = tokens.get("input_ids")
    attention_mask = tokens.get("attention_mask")
    if (
        not torch.is_tensor(input_ids)
        or input_ids.dtype != torch.long
        or input_ids.ndim != 2
        or input_ids.shape[0] != len(texts)
        or not torch.is_tensor(attention_mask)
        or attention_mask.dtype != torch.long
        or attention_mask.shape != input_ids.shape
    ):
        raise TypeError("Validation tokenizer returned malformed input tensors")
    if require_slot:
        slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
        if slot_token_id == tokenizer.unk_token_id:
            raise ValueError(f"{SLOT_TOKEN} is absent from the validation tokenizer")
        slot_counts = input_ids.eq(slot_token_id).sum(dim=1)
        if not slot_counts.eq(1).all():
            bad_rows = slot_counts.ne(1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Every tokenized validation query must contain exactly one {SLOT_TOKEN}; "
                f"bad_rows={bad_rows}"
            )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _forward_validation_pair(
    model,
    *,
    query_tokens: Mapping[str, torch.Tensor],
    passage_tokens: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        query_input_ids=query_tokens["input_ids"].to(device),
        query_attention_mask=query_tokens["attention_mask"].to(device),
        passage_input_ids=passage_tokens["input_ids"].to(device),
        passage_attention_mask=passage_tokens["attention_mask"].to(device),
    )
    if type(outputs) is not dict or set(outputs) != {
        "query_embeddings",
        "passage_embeddings",
    }:
        raise TypeError(
            "Fold-global validation requires exactly query_embeddings and passage_embeddings"
        )
    query_embeddings = outputs["query_embeddings"]
    passage_embeddings = outputs["passage_embeddings"]
    if (
        not torch.is_tensor(query_embeddings)
        or query_embeddings.ndim != 2
        or not query_embeddings.is_floating_point()
        or not torch.is_tensor(passage_embeddings)
        or passage_embeddings.ndim != 2
        or not passage_embeddings.is_floating_point()
        or query_embeddings.shape[1] != passage_embeddings.shape[1]
    ):
        raise TypeError("Fold-global validation forward returned malformed embeddings")
    if not torch.isfinite(query_embeddings).all() or not torch.isfinite(
        passage_embeddings
    ).all():
        raise FloatingPointError("Fold-global validation forward returned non-finite embeddings")
    return query_embeddings, passage_embeddings


def _coordinated_local_call(
    context: str,
    operation: Callable[[], _T],
    *,
    group=None,
) -> _T:
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    value = None
    try:
        value = operation()
        local_status: dict[str, Any] = {"ok": True, "rank": rank}
    except BaseException as error:
        local_status = {
            "ok": False,
            "rank": rank,
            "context": context,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    gathered_status: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_status, local_status, group=group)
    failures = [
        status
        for status in gathered_status
        if type(status) is not dict or status.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"{context} failed collectively: {failures}")
    return value


def _all_gather_positioned_embeddings(
    local_embeddings: torch.Tensor,
    local_positions: tuple[int, ...],
    *,
    total_count: int,
    group=None,
) -> torch.Tensor:
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    def validate_local_embeddings() -> None:
        expected_local_positions = tuple(range(rank, total_count, world_size))
        if local_positions != expected_local_positions:
            raise RuntimeError(
                f"Rank {rank} validation positions changed: "
                f"actual={local_positions}, expected={expected_local_positions}"
            )
        if (
            not torch.is_tensor(local_embeddings)
            or local_embeddings.ndim != 2
            or not local_embeddings.is_floating_point()
            or local_embeddings.shape[0] != len(local_positions)
            or local_embeddings.shape[1] < 1
        ):
            raise TypeError("Local validation embeddings do not align with their positions")
        if not torch.isfinite(local_embeddings).all():
            raise FloatingPointError("Local validation embeddings contain non-finite values")

    _coordinated_local_call(
        "Pre-gather validation embedding check",
        validate_local_embeddings,
        group=group,
    )

    device = local_embeddings.device
    local_hidden = torch.tensor([local_embeddings.shape[1]], dtype=torch.long, device=device)
    hidden_sizes = [torch.empty_like(local_hidden) for _ in range(world_size)]
    dist.all_gather(hidden_sizes, local_hidden, group=group)
    if any(int(size.item()) != local_embeddings.shape[1] for size in hidden_sizes):
        raise RuntimeError("Validation embedding dimensions differ across ranks")

    max_rows = (total_count + world_size - 1) // world_size
    padded_positions = torch.full(
        (max_rows,), INVALID_GLOBAL_POSITION, dtype=torch.long, device=device
    )
    padded_positions[: len(local_positions)] = torch.tensor(
        local_positions, dtype=torch.long, device=device
    )
    padded_embeddings = local_embeddings.new_zeros((max_rows, local_embeddings.shape[1]))
    padded_embeddings[: len(local_positions)] = local_embeddings

    gathered_positions = [torch.empty_like(padded_positions) for _ in range(world_size)]
    gathered_embeddings = [torch.empty_like(padded_embeddings) for _ in range(world_size)]
    dist.all_gather(gathered_positions, padded_positions, group=group)
    dist.all_gather(gathered_embeddings, padded_embeddings, group=group)

    full_embeddings = local_embeddings.new_empty((total_count, local_embeddings.shape[1]))
    seen_positions: set[int] = set()
    for source_rank in range(world_size):
        expected_positions = tuple(range(source_rank, total_count, world_size))
        count = len(expected_positions)
        actual_positions = tuple(
            int(value) for value in gathered_positions[source_rank][:count].tolist()
        )
        if actual_positions != expected_positions:
            raise RuntimeError(
                f"Gathered validation positions for rank {source_rank} changed: "
                f"actual={actual_positions}, expected={expected_positions}"
            )
        if not gathered_positions[source_rank][count:].eq(INVALID_GLOBAL_POSITION).all():
            raise RuntimeError(f"Rank {source_rank} validation position padding is malformed")
        if not gathered_embeddings[source_rank][count:].eq(0).all():
            raise RuntimeError(f"Rank {source_rank} validation embedding padding is nonzero")
        for row, global_position in enumerate(expected_positions):
            if global_position in seen_positions:
                raise RuntimeError(f"Duplicate gathered validation position {global_position}")
            seen_positions.add(global_position)
            full_embeddings[global_position] = gathered_embeddings[source_rank][row]
    if seen_positions != set(range(total_count)):
        raise RuntimeError("Gathered validation embeddings have incomplete global coverage")
    if not torch.isfinite(full_embeddings).all():
        raise FloatingPointError("Gathered validation embeddings contain non-finite values")
    return full_embeddings


def _result_from_payload(
    payload: object,
    validation_data: FoldGlobalValidationData,
    passage_index_table: PassageIndexTable,
) -> FoldGlobalValidationResult:
    validation_data = _validate_validation_data(
        validation_data,
        passage_index_table,
    )
    expected_keys = {
        "schema_version",
        "metrics",
        "per_query",
        "per_case",
        "ranking_sha256",
        "case_ids_sha256",
        "query_ids_sha256",
        "passage_ids_sha256",
        "validation_contract_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise RuntimeError("Broadcast fold-global validation result has an invalid schema")
    if payload["schema_version"] != FOLD_GLOBAL_RESULT_SCHEMA_VERSION:
        raise RuntimeError("Broadcast fold-global result schema version changed")
    for digest_name in (
        "ranking_sha256",
        "case_ids_sha256",
        "query_ids_sha256",
        "passage_ids_sha256",
        "validation_contract_sha256",
    ):
        if not _is_sha256(payload[digest_name]):
            raise RuntimeError(f"Broadcast {digest_name} is not a lowercase SHA-256")
    if payload["case_ids_sha256"] != validation_data.case_ids_sha256:
        raise RuntimeError("Broadcast validation case inventory digest changed")
    if payload["query_ids_sha256"] != validation_data.query_ids_sha256:
        raise RuntimeError("Broadcast validation query inventory digest changed")
    if payload["passage_ids_sha256"] != validation_data.passage_ids_sha256:
        raise RuntimeError("Broadcast validation passage inventory digest changed")
    if payload["validation_contract_sha256"] != validation_data.contract_sha256:
        raise RuntimeError("Broadcast validation contract digest changed")

    raw_per_query = payload["per_query"]
    raw_per_case = payload["per_case"]
    if type(raw_per_query) is not list or len(raw_per_query) != validation_data.query_count:
        raise RuntimeError("Broadcast per-query validation coverage changed")
    if type(raw_per_case) is not list or len(raw_per_case) != validation_data.case_count:
        raise RuntimeError("Broadcast per-case validation coverage changed")

    def finite_number(value: object, *, name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise RuntimeError(f"Broadcast {name} must be one finite number")
        return float(value)

    per_query: list[dict[str, Any]] = []
    for position, (raw_record, expected_query) in enumerate(
        zip(raw_per_query, validation_data.queries)
    ):
        if type(raw_record) is not dict or set(raw_record) != _PER_QUERY_KEYS:
            raise RuntimeError(f"Broadcast per-query record {position} has an invalid schema")
        if raw_record["query_id"] != expected_query.query_id:
            raise RuntimeError("Broadcast per-query IDs or order changed")
        if raw_record["doc_id"] != expected_query.doc_id:
            raise RuntimeError("Broadcast per-query document IDs changed")
        gold_count = raw_record["gold_count"]
        first_gold_rank = raw_record["first_gold_rank"]
        if type(gold_count) is not int or gold_count != len(expected_query.gold_passage_indices):
            raise RuntimeError(f"Broadcast query {expected_query.query_id!r} gold count changed")
        if (
            type(first_gold_rank) is not int
            or first_gold_rank < 1
            or first_gold_rank > validation_data.passage_count
        ):
            raise RuntimeError(
                f"Broadcast query {expected_query.query_id!r} first-gold rank is invalid"
            )
        candidate_count = finite_number(
            raw_record["candidate_count"],
            name=f"query[{expected_query.query_id}].candidate_count",
        )
        if candidate_count != float(validation_data.passage_count):
            raise RuntimeError("Broadcast fold-global candidate count changed")
        reciprocal_rank = finite_number(
            raw_record["first_gold_reciprocal_rank_full_ranking"],
            name=(
                f"query[{expected_query.query_id}]."
                "first_gold_reciprocal_rank_full_ranking"
            ),
        )
        if reciprocal_rank != 1.0 / first_gold_rank:
            raise RuntimeError("Broadcast full-ranking reciprocal rank is inconsistent")

        normalized: dict[str, Any] = {
            "query_id": expected_query.query_id,
            "doc_id": expected_query.doc_id,
            "gold_count": gold_count,
            "first_gold_rank": first_gold_rank,
            "first_gold_reciprocal_rank_full_ranking": reciprocal_rank,
            "candidate_count": candidate_count,
        }
        recovered_by_k: dict[int, int] = {}
        for k in VALIDATION_KS:
            hit = finite_number(
                raw_record[f"hit_at_{k}"],
                name=f"query[{expected_query.query_id}].hit_at_{k}",
            )
            set_recall = finite_number(
                raw_record[f"set_recall_at_{k}"],
                name=f"query[{expected_query.query_id}].set_recall_at_{k}",
            )
            exact_recovery = finite_number(
                raw_record[f"exact_target_recovery_at_{k}"],
                name=f"query[{expected_query.query_id}].exact_target_recovery_at_{k}",
            )
            matching_recovered = [
                recovered
                for recovered in range(gold_count + 1)
                if set_recall == recovered / gold_count
            ]
            if len(matching_recovered) != 1:
                raise RuntimeError("Broadcast set recall is not a valid gold-set fraction")
            recovered = matching_recovered[0]
            expected_hit = 1.0 if first_gold_rank <= k else 0.0
            if hit not in (0.0, 1.0) or hit != expected_hit or hit != (1.0 if recovered else 0.0):
                raise RuntimeError("Broadcast Hit@k is inconsistent with the ranking")
            expected_exact = 1.0 if recovered == gold_count else 0.0
            if exact_recovery not in (0.0, 1.0) or exact_recovery != expected_exact:
                raise RuntimeError("Broadcast exact-target recovery is inconsistent")
            normalized[f"hit_at_{k}"] = hit
            normalized[f"set_recall_at_{k}"] = set_recall
            normalized[f"exact_target_recovery_at_{k}"] = exact_recovery
            recovered_by_k[k] = recovered
        if list(recovered_by_k.values()) != sorted(recovered_by_k.values()):
            raise RuntimeError("Broadcast gold-set recovery decreases as k increases")
        per_query.append(normalized)

    per_case: list[dict[str, Any]] = []
    for position, (raw_record, case_id) in enumerate(
        zip(raw_per_case, validation_data.case_ids)
    ):
        if type(raw_record) is not dict or set(raw_record) != _PER_CASE_KEYS:
            raise RuntimeError(f"Broadcast per-case record {position} has an invalid schema")
        if raw_record["doc_id"] != case_id:
            raise RuntimeError("Broadcast per-case IDs or order changed")
        case_rows = [record for record in per_query if record["doc_id"] == case_id]
        if (
            type(raw_record["query_count"]) is not int
            or raw_record["query_count"] != len(case_rows)
        ):
            raise RuntimeError(f"Broadcast case {case_id!r} query count changed")
        raw_case_metrics = raw_record["metrics"]
        if type(raw_case_metrics) is not dict or set(raw_case_metrics) != set(
            _QUERY_METRIC_NAMES
        ):
            raise RuntimeError(f"Broadcast case {case_id!r} metrics have an invalid schema")
        case_metrics: dict[str, float] = {}
        for metric_name in _QUERY_METRIC_NAMES:
            actual = finite_number(
                raw_case_metrics[metric_name],
                name=f"case[{case_id}].{metric_name}",
            )
            expected = _mean(
                [float(record[metric_name]) for record in case_rows],
                name=f"case[{case_id}].{metric_name}",
            )
            if actual != expected:
                raise RuntimeError(f"Broadcast case {case_id!r} aggregate changed")
            case_metrics[metric_name] = actual
        per_case.append(
            {
                "doc_id": case_id,
                "query_count": len(case_rows),
                "metrics": case_metrics,
            }
        )

    raw_metrics = payload["metrics"]
    if type(raw_metrics) is not dict or set(raw_metrics) != set(_AGGREGATE_METRIC_KEYS):
        raise RuntimeError("Broadcast fold-global aggregate metrics have an invalid schema")
    expected_metrics: dict[str, float] = {
        "eval_validation_num_queries": float(validation_data.query_count),
        "eval_validation_num_cases": float(validation_data.case_count),
        "eval_validation_num_passages": float(validation_data.passage_count),
    }
    for metric_name in _QUERY_METRIC_NAMES:
        expected_metrics[f"eval_validation_query_micro_{metric_name}"] = _mean(
            [float(record[metric_name]) for record in per_query],
            name=f"query_micro.{metric_name}",
        )
        expected_metrics[f"eval_validation_case_macro_{metric_name}"] = _mean(
            [float(record["metrics"][metric_name]) for record in per_case],
            name=f"case_macro.{metric_name}",
        )
    expected_metrics["eval_validation_query_micro_mrr_full_ranking"] = expected_metrics[
        "eval_validation_query_micro_first_gold_reciprocal_rank_full_ranking"
    ]
    expected_metrics["eval_validation_case_macro_mrr_full_ranking"] = expected_metrics[
        VALIDATION_SECONDARY_METRIC
    ]
    metrics: dict[str, float] = {}
    for metric_name, expected in expected_metrics.items():
        actual = finite_number(raw_metrics[metric_name], name=metric_name)
        if actual != expected:
            raise RuntimeError(f"Broadcast aggregate metric {metric_name!r} changed")
        metrics[metric_name] = actual

    return FoldGlobalValidationResult(
        schema_version=payload["schema_version"],
        metrics=_freeze(metrics),
        per_query=tuple(_freeze(record) for record in per_query),
        per_case=tuple(_freeze(record) for record in per_case),
        ranking_sha256=payload["ranking_sha256"],
        case_ids_sha256=payload["case_ids_sha256"],
        query_ids_sha256=payload["query_ids_sha256"],
        passage_ids_sha256=payload["passage_ids_sha256"],
        validation_contract_sha256=payload["validation_contract_sha256"],
    )


def evaluate_fold_global_distributed(
    model,
    tokenizer,
    *,
    validation_data: FoldGlobalValidationData,
    passage_index_table: PassageIndexTable,
    max_len_query: int,
    max_len_passage: int,
    forward_steps: int = VALIDATION_FORWARD_STEPS,
    group=None,
) -> FoldGlobalValidationResult:
    """Encode exact rank shards, compute on rank zero, and broadcast one result."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed fold-global validation requires an initialized process group"
        )
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    if world_size < 2:
        raise RuntimeError("Distributed fold-global validation requires world_size >= 2")
    def validate_runtime_contract() -> str:
        _validate_validation_data(validation_data, passage_index_table)
        if world_size != VALIDATION_WORLD_SIZE:
            raise RuntimeError(
                f"Controlled validation requires world_size={VALIDATION_WORLD_SIZE}; "
                f"got {world_size}"
            )
        if forward_steps != VALIDATION_FORWARD_STEPS or type(forward_steps) is not int:
            raise ValueError(
                f"Controlled validation requires exactly {VALIDATION_FORWARD_STEPS} forwards"
            )
        if max_len_query != VALIDATION_MAX_LEN_QUERY or type(max_len_query) is not int:
            raise ValueError(
                f"Controlled validation max query length must be {VALIDATION_MAX_LEN_QUERY}"
            )
        if max_len_passage != VALIDATION_MAX_LEN_PASSAGE or type(max_len_passage) is not int:
            raise ValueError(
                f"Controlled validation max passage length must be "
                f"{VALIDATION_MAX_LEN_PASSAGE}"
            )
        if validation_data.query_count != VALIDATION_QUERY_COUNT:
            raise RuntimeError(
                f"Controlled validation requires {VALIDATION_QUERY_COUNT} queries; "
                f"got {validation_data.query_count}"
            )
        if validation_data.case_count not in VALIDATION_CASE_COUNTS:
            raise RuntimeError(
                f"Controlled validation case count must be one of "
                f"{VALIDATION_CASE_COUNTS}; got {validation_data.case_count}"
            )
        if validation_data.passage_count not in VALIDATION_PASSAGE_COUNTS:
            raise RuntimeError(
                f"Controlled validation passage count must be one of "
                f"{VALIDATION_PASSAGE_COUNTS}; got {validation_data.passage_count}"
            )
        if len(passage_index_table) != 5_286:
            raise RuntimeError("Controlled validation requires the 5,286-passage corpus index")
        runtime_payload = {
            "validation_contract_sha256": validation_data.contract_sha256,
            "passage_index_sha256": passage_index_table.sha256,
            "max_len_query": max_len_query,
            "max_len_passage": max_len_passage,
            "forward_steps": forward_steps,
            "world_size": world_size,
            "query_batch_cap": VALIDATION_QUERY_BATCH_CAP,
            "passage_batch_cap": VALIDATION_PASSAGE_BATCH_CAP,
            "ks": list(VALIDATION_KS),
            "sharding": "sorted_global_position_mod_world_size_v1",
            "scoring": "cpu_float32_v1",
            "ranking": "score_desc_passage_id_asc_v1",
        }
        return hashlib.sha256(_canonical_json(runtime_payload).encode("utf-8")).hexdigest()

    runtime_contract_sha256 = _coordinated_local_call(
        "Fold-global validation runtime preflight",
        validate_runtime_contract,
        group=group,
    )
    gathered_runtime_contracts: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(
        gathered_runtime_contracts,
        runtime_contract_sha256,
        group=group,
    )
    if gathered_runtime_contracts != [runtime_contract_sha256] * world_size:
        raise RuntimeError(
            "Fold-global validation ranks received different data/runtime contracts: "
            f"{gathered_runtime_contracts}"
        )

    def build_local_schedule():
        local_query_positions = tuple(range(rank, validation_data.query_count, world_size))
        local_passage_positions = tuple(range(rank, validation_data.passage_count, world_size))
        local_query_chunks = _balanced_nonempty_chunks(
            local_query_positions, chunk_count=forward_steps
        )
        local_passage_chunks = _balanced_nonempty_chunks(
            local_passage_positions, chunk_count=forward_steps
        )
        if max(len(chunk) for chunk in local_query_chunks) > VALIDATION_QUERY_BATCH_CAP:
            raise RuntimeError(
                "Fold-global validation query chunk exceeds its frozen batch cap"
            )
        if max(len(chunk) for chunk in local_passage_chunks) > VALIDATION_PASSAGE_BATCH_CAP:
            raise RuntimeError(
                "Fold-global validation passage chunk exceeds its frozen batch cap"
            )
        return (
            local_query_positions,
            local_passage_positions,
            local_query_chunks,
            local_passage_chunks,
        )

    query_positions, passage_positions, query_chunks, passage_chunks = (
        _coordinated_local_call(
            "Fold-global validation shard schedule",
            build_local_schedule,
            group=group,
        )
    )

    def inspect_model():
        local_retriever = model.module if hasattr(model, "module") else model
        try:
            local_device = next(local_retriever.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise TypeError("Fold-global validation model must expose parameters") from error
        local_was_training = bool(local_retriever.training)
        if not hasattr(model, "training") or bool(model.training) != local_was_training:
            raise RuntimeError("Validation engine and retriever training modes disagree")
        return local_device, local_was_training

    device, was_training = _coordinated_local_call(
        "Fold-global validation model preflight",
        inspect_model,
        group=group,
    )
    local_query_embeddings: list[torch.Tensor] = []
    local_passage_embeddings: list[torch.Tensor] = []
    try:
        _coordinated_local_call(
            "Fold-global validation enter evaluation mode",
            lambda: model.eval(),
            group=group,
        )
        with torch.no_grad():
            for query_chunk, passage_chunk in zip(query_chunks, passage_chunks):
                def tokenize_pair():
                    query_texts = [
                        validation_data.queries[position].query_text
                        for position in query_chunk
                    ]
                    passage_texts = [
                        passage_index_table.text_for_index(
                            validation_data.passage_indices[position]
                        )
                        for position in passage_chunk
                    ]
                    return (
                        _tokenize_validation_texts(
                            tokenizer,
                            query_texts,
                            truncation_side="left",
                            max_length=max_len_query,
                            require_slot=True,
                        ),
                        _tokenize_validation_texts(
                            tokenizer,
                            passage_texts,
                            truncation_side="right",
                            max_length=max_len_passage,
                            require_slot=False,
                        ),
                    )

                query_tokens, passage_tokens = _coordinated_local_call(
                    "Fold-global validation tokenization preflight",
                    tokenize_pair,
                    group=group,
                )

                def forward_and_validate():
                    query_vectors, passage_vectors = _forward_validation_pair(
                        model,
                        query_tokens=query_tokens,
                        passage_tokens=passage_tokens,
                        device=device,
                    )
                    if query_vectors.shape[0] != len(query_chunk):
                        raise RuntimeError("Validation forward changed the query row count")
                    if passage_vectors.shape[0] != len(passage_chunk):
                        raise RuntimeError("Validation forward changed the passage row count")
                    return query_vectors, passage_vectors

                query_vectors, passage_vectors = _coordinated_local_call(
                    "Fold-global validation top-level forward",
                    forward_and_validate,
                    group=group,
                )
                local_query_embeddings.append(query_vectors.detach())
                local_passage_embeddings.append(passage_vectors.detach())
    finally:
        _coordinated_local_call(
            "Fold-global validation restore model mode",
            lambda: model.train(was_training),
            group=group,
        )

    local_query_matrix, local_passage_matrix = _coordinated_local_call(
        "Fold-global validation local embedding concatenation",
        lambda: (
            torch.cat(local_query_embeddings, dim=0),
            torch.cat(local_passage_embeddings, dim=0),
        ),
        group=group,
    )
    query_embeddings = _all_gather_positioned_embeddings(
        local_query_matrix,
        query_positions,
        total_count=validation_data.query_count,
        group=group,
    )
    passage_embeddings = _all_gather_positioned_embeddings(
        local_passage_matrix,
        passage_positions,
        total_count=validation_data.passage_count,
        group=group,
    )

    status: list[object] = [None]
    if rank == 0:
        try:
            result = compute_fold_global_metrics_from_embeddings(
                query_embeddings=query_embeddings,
                passage_embeddings=passage_embeddings,
                validation_data=validation_data,
                passage_index_table=passage_index_table,
            )
            status[0] = {"ok": True, "result": result.to_payload()}
        except BaseException as error:
            status[0] = {
                "ok": False,
                "context": "Fold-global validation metric computation",
                "error_type": type(error).__name__,
                "message": str(error),
            }
    source_rank = 0 if group is None else dist.get_global_rank(group, 0)
    dist.broadcast_object_list(status, src=source_rank, group=group)
    payload = status[0]
    if type(payload) is not dict or type(payload.get("ok")) is not bool:
        raise RuntimeError("Fold-global validation returned a malformed collective status")
    if payload["ok"] is not True:
        raise RuntimeError(
            f"{payload.get('context')} failed on rank 0: "
            f"{payload.get('error_type')}: {payload.get('message')}"
        )
    return _result_from_payload(
        payload.get("result"),
        validation_data,
        passage_index_table,
    )
