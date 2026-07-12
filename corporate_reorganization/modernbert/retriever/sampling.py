from __future__ import annotations

import hashlib
import json
import operator
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .data import CorpusPassage, PassageIndexTable, QueryExample
from .query_views import QUERY_VIEW_STRUCTURED, normalize_query_view, select_query_text


SAMPLER_LOCAL_UNIQUE = "local_unique"
SAMPLER_GLOBAL_UNIFORM = "global_uniform"
SUPPORTED_CONTROLLED_SAMPLERS: Tuple[str, ...] = (
    SAMPLER_LOCAL_UNIQUE,
    SAMPLER_GLOBAL_UNIFORM,
)

MAX_EXPLICIT_POSITIVES = 4
NUM_EXPLICIT_NEGATIVES = 60
NUM_LOCAL_SAME_CASE_NEGATIVES = 40
NUM_LOCAL_OTHER_CASE_NEGATIVES = 20

SELECTION_ALGORITHM = "sha256_digest_ranking_v1"
TRACE_SCHEMA_VERSION = 1

_POSITIVE_COMPONENT = "explicit_positive"
_LOCAL_SAME_COMPONENT = "local_same_case_negative"
_LOCAL_OTHER_COMPONENT = "local_other_case_negative"
_GLOBAL_COMPONENT = "global_negative"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _require_non_empty_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


def _require_passage_id_list(name: str, value: object, *, allow_empty: bool = False) -> List[str]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON list, not {type(value).__name__}")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    for index, passage_id in enumerate(value):
        _require_non_empty_string(f"{name}[{index}]", passage_id)
    return value


def _selection_digest(
    *,
    experiment_seed: int,
    epoch: int,
    query_id: str,
    component: str,
    passage_id: str,
) -> bytes:
    framed_key = {
        "algorithm": SELECTION_ALGORITHM,
        "component": component,
        "epoch": epoch,
        "experiment_seed": experiment_seed,
        "passage_id": passage_id,
        "query_id": query_id,
    }
    return hashlib.sha256(_canonical_json_bytes(framed_key)).digest()


def digest_ranked_sample(
    passage_ids: Sequence[str],
    k: int,
    *,
    experiment_seed: int,
    epoch: int,
    query_id: str,
    component: str,
) -> List[str]:
    """Select a passage-uniform, deterministic sample without replacement."""

    _require_exact_int("k", k, minimum=0)
    _require_exact_int("experiment_seed", experiment_seed, minimum=0)
    _require_exact_int("epoch", epoch, minimum=0)
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query_id must be a non-empty string")
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")

    pool = list(passage_ids)
    if any(not isinstance(passage_id, str) or not passage_id for passage_id in pool):
        raise ValueError("Every sampled passage ID must be a non-empty string")
    if len(pool) != len(set(pool)):
        raise ValueError(f"Sampling pool for component={component!r} contains duplicate passage IDs")
    if k > len(pool):
        raise ValueError(
            f"Sampling pool for component={component!r} has {len(pool)} eligible passages; "
            f"exact quota is {k} and replacement is forbidden"
        )

    ranked = sorted(
        pool,
        key=lambda passage_id: (
            _selection_digest(
                experiment_seed=experiment_seed,
                epoch=epoch,
                query_id=query_id,
                component=component,
                passage_id=passage_id,
            ),
            passage_id,
        ),
    )
    return ranked[:k]


def sampling_trace_checksum(trace_without_checksum: Mapping[str, Any]) -> str:
    if "trace_sha256" in trace_without_checksum:
        raise ValueError("trace_without_checksum must not contain trace_sha256")
    return hashlib.sha256(_canonical_json_bytes(trace_without_checksum)).hexdigest()


def validate_sampling_trace(trace: Mapping[str, Any]) -> None:
    if type(trace) is not dict:
        raise TypeError(f"Sampling trace must be a JSON object, not {type(trace).__name__}")
    expected_keys = {
        "schema_version",
        "selection_algorithm",
        "sampler",
        "experiment_seed",
        "epoch",
        "query_id",
        "doc_id",
        "positive_passage_ids",
        "selected_positive_passage_ids",
        "negative_passage_ids_by_stratum",
        "eligible_pool_sizes_by_stratum",
        "candidate_passage_ids",
        "trace_sha256",
    }
    if set(trace) != expected_keys:
        raise ValueError(
            "Sampling trace fields do not match schema: "
            f"missing={sorted(expected_keys - set(trace))}, extra={sorted(set(trace) - expected_keys)}"
        )
    schema_version = _require_exact_int("trace schema_version", trace["schema_version"], minimum=1)
    if schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported sampling trace schema_version={trace['schema_version']!r}")
    selection_algorithm = _require_non_empty_string(
        "trace selection_algorithm",
        trace["selection_algorithm"],
    )
    if selection_algorithm != SELECTION_ALGORITHM:
        raise ValueError(f"Unsupported selection_algorithm={trace['selection_algorithm']!r}")

    sampler = _require_non_empty_string("trace sampler", trace["sampler"])
    if sampler not in SUPPORTED_CONTROLLED_SAMPLERS:
        raise ValueError(f"Unsupported controlled sampler in trace: {sampler!r}")
    _require_exact_int("trace experiment_seed", trace["experiment_seed"], minimum=0)
    _require_exact_int("trace epoch", trace["epoch"], minimum=0)
    _require_non_empty_string("trace query_id", trace["query_id"])
    _require_non_empty_string("trace doc_id", trace["doc_id"])

    recorded_checksum = trace["trace_sha256"]
    if type(recorded_checksum) is not str:
        raise TypeError(
            f"trace_sha256 must be a string, not {type(recorded_checksum).__name__}"
        )
    if len(recorded_checksum) != 64 or any(
        character not in "0123456789abcdef" for character in recorded_checksum
    ):
        raise ValueError("trace_sha256 must be exactly 64 lowercase hexadecimal characters")

    payload = dict(trace)
    payload.pop("trace_sha256")
    expected_checksum = sampling_trace_checksum(payload)
    if recorded_checksum != expected_checksum:
        raise ValueError(
            f"Sampling trace checksum mismatch: recorded={recorded_checksum!r}, "
            f"expected={expected_checksum!r}"
        )

    positives = _require_passage_id_list(
        "trace positive_passage_ids",
        trace["positive_passage_ids"],
    )
    selected_positives = _require_passage_id_list(
        "trace selected_positive_passage_ids",
        trace["selected_positive_passage_ids"],
    )
    negative_by_stratum = trace["negative_passage_ids_by_stratum"]
    pool_sizes = trace["eligible_pool_sizes_by_stratum"]
    candidates = _require_passage_id_list(
        "trace candidate_passage_ids",
        trace["candidate_passage_ids"],
    )
    if type(negative_by_stratum) is not dict:
        raise TypeError(
            "trace negative_passage_ids_by_stratum must be a JSON object, "
            f"not {type(negative_by_stratum).__name__}"
        )
    if type(pool_sizes) is not dict:
        raise TypeError(
            "trace eligible_pool_sizes_by_stratum must be a JSON object, "
            f"not {type(pool_sizes).__name__}"
        )
    if not positives or len(positives) != len(set(positives)):
        raise ValueError("Trace positive_passage_ids must be non-empty and unique")
    if not (1 <= len(selected_positives) <= MAX_EXPLICIT_POSITIVES):
        raise ValueError("Trace must contain between one and four explicit positives")
    expected_positive_count = min(len(positives), MAX_EXPLICIT_POSITIVES)
    if len(selected_positives) != expected_positive_count:
        raise ValueError(
            f"Trace has {len(selected_positives)} explicit positives; expected {expected_positive_count}"
        )
    if not set(selected_positives).issubset(positives):
        raise ValueError("Trace selected positives are not a subset of all query positives")

    if sampler == SAMPLER_LOCAL_UNIQUE:
        expected_strata = ("same_case", "other_case")
        expected_quotas = {
            "same_case": NUM_LOCAL_SAME_CASE_NEGATIVES,
            "other_case": NUM_LOCAL_OTHER_CASE_NEGATIVES,
        }
    else:
        expected_strata = ("global",)
        expected_quotas = {"global": NUM_EXPLICIT_NEGATIVES}
    if set(negative_by_stratum) != set(expected_strata) or set(pool_sizes) != set(expected_strata):
        raise ValueError(f"Trace strata do not match sampler={sampler}")

    negatives: List[str] = []
    for stratum in expected_strata:
        stratum_passages = _require_passage_id_list(
            f"trace negative_passage_ids_by_stratum[{stratum!r}]",
            negative_by_stratum[stratum],
        )
        quota = expected_quotas[stratum]
        if len(stratum_passages) != quota:
            raise ValueError(
                f"Trace stratum={stratum!r} has {len(stratum_passages)} passages; expected {quota}"
            )
        pool_size = _require_exact_int(
            f"trace eligible_pool_sizes_by_stratum[{stratum!r}]",
            pool_sizes[stratum],
            minimum=0,
        )
        if pool_size < quota:
            raise ValueError(f"Trace pool size for stratum={stratum!r} cannot satisfy quota={quota}")
        negatives.extend(stratum_passages)

    if len(negatives) != NUM_EXPLICIT_NEGATIVES or len(negatives) != len(set(negatives)):
        raise ValueError("Trace must contain exactly 60 unique negatives")
    if set(negatives).intersection(positives):
        raise ValueError("Trace contains a current-query positive in its negative strata")
    if candidates != [*selected_positives, *negatives]:
        raise ValueError("Trace candidate order does not equal positives followed by declared negative strata")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Trace candidate_passage_ids are not unique")


class ControlledRetrievalTrainDataset:
    """Strict controlled sampler for the ARR cross-validation experiment."""

    def __init__(
        self,
        queries: Sequence[QueryExample],
        corpus_by_passage_id: Mapping[str, CorpusPassage],
        candidates_by_case: Mapping[str, Sequence[str]],
        training_doc_ids: Sequence[str],
        *,
        passage_index_table: PassageIndexTable,
        sampler: str,
        experiment_seed: int,
        query_view: str = QUERY_VIEW_STRUCTURED,
    ) -> None:
        if sampler not in SUPPORTED_CONTROLLED_SAMPLERS:
            raise ValueError(
                f"Unsupported controlled sampler={sampler!r}; expected one of {SUPPORTED_CONTROLLED_SAMPLERS}"
            )
        self.sampler = sampler
        self.experiment_seed = _require_exact_int("experiment_seed", experiment_seed, minimum=0)
        self.query_view = normalize_query_view(query_view)
        self.epoch = 0

        if not isinstance(passage_index_table, PassageIndexTable):
            raise TypeError("passage_index_table must be a PassageIndexTable")
        self.passage_index_table = passage_index_table

        self.queries = list(queries)
        if not self.queries:
            raise ValueError("Controlled training queries must not be empty")
        query_ids = [query.query_id for query in self.queries]
        if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
            raise ValueError("Every query_id must be a non-empty string")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Controlled training queries contain duplicate query IDs")

        self.training_doc_ids = tuple(training_doc_ids)
        if not self.training_doc_ids:
            raise ValueError("training_doc_ids must not be empty")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in self.training_doc_ids):
            raise ValueError("Every training doc_id must be a non-empty string")
        if len(self.training_doc_ids) != len(set(self.training_doc_ids)):
            raise ValueError("training_doc_ids contains duplicates")
        self._training_doc_id_set = set(self.training_doc_ids)

        self.corpus_by_passage_id = dict(corpus_by_passage_id)
        if not self.corpus_by_passage_id:
            raise ValueError("corpus_by_passage_id must not be empty")
        for passage_id, passage in self.corpus_by_passage_id.items():
            if passage_id != passage.passage_id:
                raise ValueError(
                    f"Corpus key {passage_id!r} does not match record passage_id={passage.passage_id!r}"
                )
        if set(self.passage_index_table.passage_ids) != set(self.corpus_by_passage_id):
            raise ValueError("Passage index table does not cover exactly the controlled corpus")
        for passage_id, passage in self.corpus_by_passage_id.items():
            passage_index = self.passage_index_table.index_for_id(passage_id)
            if self.passage_index_table.text_for_index(passage_index) != passage.text:
                raise ValueError(
                    f"Passage index table text disagrees with controlled corpus for {passage_id!r}"
                )

        corpus_ids_by_doc_id: Dict[str, set[str]] = {}
        for passage_id, passage in self.corpus_by_passage_id.items():
            corpus_ids_by_doc_id.setdefault(passage.doc_id, set()).add(passage_id)

        pools: Dict[str, Tuple[str, ...]] = {}
        seen_training_passage_ids: set[str] = set()
        for doc_id in self.training_doc_ids:
            if doc_id not in candidates_by_case:
                raise ValueError(f"Missing candidate pool for training doc_id={doc_id}")
            raw_pool = list(candidates_by_case[doc_id])
            if not raw_pool:
                raise ValueError(f"Candidate pool is empty for training doc_id={doc_id}")
            if any(not isinstance(passage_id, str) or not passage_id for passage_id in raw_pool):
                raise ValueError(f"Candidate pool for doc_id={doc_id} contains an invalid passage ID")
            if len(raw_pool) != len(set(raw_pool)):
                raise ValueError(f"Candidate pool for doc_id={doc_id} contains duplicate passage IDs")

            actual_ids = set(raw_pool)
            expected_ids = corpus_ids_by_doc_id.get(doc_id, set())
            if actual_ids != expected_ids:
                raise ValueError(
                    f"Candidate pool for doc_id={doc_id} is not the complete case corpus: "
                    f"missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}"
                )
            duplicate_across_cases = seen_training_passage_ids.intersection(actual_ids)
            if duplicate_across_cases:
                raise ValueError(
                    "Training candidate pools overlap across cases: "
                    f"{sorted(duplicate_across_cases)}"
                )
            seen_training_passage_ids.update(actual_ids)
            pools[doc_id] = tuple(sorted(actual_ids))
        self._candidate_ids_by_doc_id = pools
        self._all_training_passage_ids = tuple(sorted(seen_training_passage_ids))
        self._other_case_passage_ids_by_doc_id = {
            doc_id: tuple(
                passage_id
                for passage_id in self._all_training_passage_ids
                if self.corpus_by_passage_id[passage_id].doc_id != doc_id
            )
            for doc_id in self.training_doc_ids
        }

        for query in self.queries:
            self._validate_query(query)
            self._validate_query_quotas(query)
            select_query_text(query, query_view=self.query_view)

    def _validate_query(self, query: QueryExample) -> None:
        if query.doc_id not in self._training_doc_id_set:
            raise ValueError(
                f"Query {query.query_id} belongs to non-training doc_id={query.doc_id}; "
                "filter queries to the frozen training folds"
            )
        positives = list(query.positive_passage_ids)
        if not positives:
            raise ValueError(f"Query has no positives: {query.query_id}")
        if any(not isinstance(passage_id, str) or not passage_id for passage_id in positives):
            raise ValueError(f"Query {query.query_id} contains an invalid positive passage ID")
        if len(positives) != len(set(positives)):
            raise ValueError(f"Query {query.query_id} contains duplicate positive passage IDs")
        case_pool = set(self._candidate_ids_by_doc_id[query.doc_id])
        missing = set(positives) - case_pool
        if missing:
            raise ValueError(
                f"Query {query.query_id} has positives outside its complete case pool: {sorted(missing)}"
            )

    def _eligible_pools(self, query: QueryExample) -> Dict[str, Tuple[str, ...]]:
        positive_ids = set(query.positive_passage_ids)
        if self.sampler == SAMPLER_LOCAL_UNIQUE:
            return {
                "same_case": tuple(
                    passage_id
                    for passage_id in self._candidate_ids_by_doc_id[query.doc_id]
                    if passage_id not in positive_ids
                ),
                "other_case": tuple(
                    passage_id
                    for passage_id in self._other_case_passage_ids_by_doc_id[query.doc_id]
                    if passage_id not in positive_ids
                ),
            }
        return {
            "global": tuple(
                passage_id
                for passage_id in self._all_training_passage_ids
                if passage_id not in positive_ids
            )
        }

    def _validate_query_quotas(self, query: QueryExample) -> None:
        pools = self._eligible_pools(query)
        quotas = (
            {"same_case": NUM_LOCAL_SAME_CASE_NEGATIVES, "other_case": NUM_LOCAL_OTHER_CASE_NEGATIVES}
            if self.sampler == SAMPLER_LOCAL_UNIQUE
            else {"global": NUM_EXPLICIT_NEGATIVES}
        )
        for stratum, quota in quotas.items():
            if len(pools[stratum]) < quota:
                raise ValueError(
                    f"Query {query.query_id} has {len(pools[stratum])} eligible {stratum} negatives; "
                    f"exact quota is {quota} and replacement is forbidden"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _require_exact_int("epoch", epoch, minimum=0)

    def __len__(self) -> int:
        return len(self.queries)

    def _sample_query(self, query: QueryExample) -> Dict[str, Any]:
        positives = sorted(query.positive_passage_ids)
        if len(positives) <= MAX_EXPLICIT_POSITIVES:
            selected_positives = positives
        else:
            selected_positives = digest_ranked_sample(
                positives,
                MAX_EXPLICIT_POSITIVES,
                experiment_seed=self.experiment_seed,
                epoch=self.epoch,
                query_id=query.query_id,
                component=_POSITIVE_COMPONENT,
            )

        eligible_pools = self._eligible_pools(query)
        if self.sampler == SAMPLER_LOCAL_UNIQUE:
            negative_by_stratum = {
                "same_case": digest_ranked_sample(
                    eligible_pools["same_case"],
                    NUM_LOCAL_SAME_CASE_NEGATIVES,
                    experiment_seed=self.experiment_seed,
                    epoch=self.epoch,
                    query_id=query.query_id,
                    component=_LOCAL_SAME_COMPONENT,
                ),
                "other_case": digest_ranked_sample(
                    eligible_pools["other_case"],
                    NUM_LOCAL_OTHER_CASE_NEGATIVES,
                    experiment_seed=self.experiment_seed,
                    epoch=self.epoch,
                    query_id=query.query_id,
                    component=_LOCAL_OTHER_COMPONENT,
                ),
            }
            negative_ids = [
                *negative_by_stratum["same_case"],
                *negative_by_stratum["other_case"],
            ]
        else:
            negative_by_stratum = {
                "global": digest_ranked_sample(
                    eligible_pools["global"],
                    NUM_EXPLICIT_NEGATIVES,
                    experiment_seed=self.experiment_seed,
                    epoch=self.epoch,
                    query_id=query.query_id,
                    component=_GLOBAL_COMPONENT,
                )
            }
            negative_ids = list(negative_by_stratum["global"])

        candidate_ids = [*selected_positives, *negative_ids]
        trace_payload: Dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "selection_algorithm": SELECTION_ALGORITHM,
            "sampler": self.sampler,
            "experiment_seed": self.experiment_seed,
            "epoch": self.epoch,
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "positive_passage_ids": positives,
            "selected_positive_passage_ids": selected_positives,
            "negative_passage_ids_by_stratum": negative_by_stratum,
            "eligible_pool_sizes_by_stratum": {
                stratum: len(pool) for stratum, pool in eligible_pools.items()
            },
            "candidate_passage_ids": candidate_ids,
        }
        trace = {
            **trace_payload,
            "trace_sha256": sampling_trace_checksum(trace_payload),
        }
        validate_sampling_trace(trace)
        return {
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "query_text": select_query_text(query, query_view=self.query_view),
            "positive_passage_indices": self.passage_index_table.indices_for_ids(positives),
            "candidate_passage_indices": self.passage_index_table.indices_for_ids(candidate_ids),
            "sampling_trace": trace,
            "sampling_trace_sha256": trace["trace_sha256"],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, bool):
            raise TypeError("Dataset index must be an integer, not bool")
        try:
            index = operator.index(idx)
        except TypeError as exc:
            raise TypeError(f"Dataset index must be an integer; got {type(idx).__name__}") from exc
        if index < 0 or index >= len(self.queries):
            raise IndexError(f"Dataset index out of range: {index}")
        return self._sample_query(self.queries[index])
