"""Sealed final evaluation for the corrected legacy-style diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from .data import CorpusPassage, PassageIndexTable, QueryExample
from .evaluation import (
    CanonicalEvaluationData,
    CanonicalRetrievalResult,
    _all_gather_positioned_embeddings,
    _balanced_nonempty_chunks,
    _coordinated_local_call,
    _forward_validation_pair,
    _tokenize_validation_texts,
    build_canonical_evaluation_data,
    canonical_result_from_payload,
    compute_canonical_retrieval_result_from_scores,
)
from .query_views import QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED, select_query_text
from .regimes import (
    REGIME_FOLD_GLOBAL,
    REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
    REGIME_SAME_CASE_FULL,
    REGIME_SAME_CASE_LEGACY,
)


CORRECTED_LEGACY_TEST_REGIMES = (
    REGIME_SAME_CASE_LEGACY,
    REGIME_SAME_CASE_FULL,
    REGIME_FOLD_GLOBAL,
    REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
)
CORRECTED_LEGACY_TEST_CASES = 4
CORRECTED_LEGACY_TEST_QUERIES = 40
CORRECTED_LEGACY_TEST_PASSAGES = 581
CORRECTED_LEGACY_TEST_FORWARD_STEPS = 4
CORRECTED_LEGACY_WORLD_SIZE = 4
CORRECTED_LEGACY_MAX_QUERY_TOKENS = 4_096
CORRECTED_LEGACY_MAX_PASSAGE_TOKENS = 500
CORRECTED_LEGACY_QUERY_BATCH_CAP = 4
CORRECTED_LEGACY_PASSAGE_BATCH_CAP = 38
CORRECTED_LEGACY_VALIDATION_CASES = 4
CORRECTED_LEGACY_VALIDATION_QUERIES = 32
CORRECTED_LEGACY_VALIDATION_PASSAGES = 398
CORRECTED_LEGACY_VALIDATION_FORWARD_STEPS = 3


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorrectedLegacyTestData:
    schema_version: int
    query_view: str
    case_ids: tuple[str, ...]
    queries: tuple[QueryExample, ...]
    query_texts: tuple[str, ...]
    passage_indices: tuple[int, ...]
    evaluation_data_by_regime: Mapping[str, CanonicalEvaluationData]
    contract_sha256: str


@dataclass(frozen=True)
class CorrectedLegacyValidationEvidenceData:
    schema_version: int
    query_view: str
    case_ids: tuple[str, ...]
    queries: tuple[QueryExample, ...]
    query_texts: tuple[str, ...]
    passage_indices: tuple[int, ...]
    evaluation_data: CanonicalEvaluationData
    contract_sha256: str


def _validation_evidence_contract_payload(
    value: CorrectedLegacyValidationEvidenceData,
    passage_index: PassageIndexTable,
) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "query_view": value.query_view,
        "case_ids": list(value.case_ids),
        "queries": [
            {
                "query_id": query.query_id,
                "doc_id": query.doc_id,
                "query_text": text,
            }
            for query, text in zip(value.queries, value.query_texts)
        ],
        "passage_index_sha256": passage_index.sha256,
        "passage_indices": list(value.passage_indices),
        "evaluation_contract_sha256": value.evaluation_data.contract_sha256,
    }


def _validate_validation_evidence_data(
    value: object,
    passage_index: PassageIndexTable,
) -> CorrectedLegacyValidationEvidenceData:
    if not isinstance(value, CorrectedLegacyValidationEvidenceData):
        raise TypeError("Corrected legacy validation evidence has the wrong type")
    if value.schema_version != 1:
        raise ValueError("Corrected legacy validation evidence schema changed")
    if value.query_view not in (QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED):
        raise ValueError("Corrected legacy validation query view changed")
    if (
        type(value.case_ids) is not tuple
        or len(value.case_ids) != CORRECTED_LEGACY_VALIDATION_CASES
        or value.case_ids != tuple(sorted(value.case_ids))
        or len(value.case_ids) != len(set(value.case_ids))
    ):
        raise ValueError("Corrected legacy validation cases changed")
    if type(value.queries) is not tuple or len(value.queries) != CORRECTED_LEGACY_VALIDATION_QUERIES:
        raise ValueError("Corrected legacy validation query count changed")
    query_ids = tuple(query.query_id for query in value.queries)
    if (
        any(not isinstance(query, QueryExample) for query in value.queries)
        or query_ids != tuple(sorted(query_ids))
        or len(query_ids) != len(set(query_ids))
        or {query.doc_id for query in value.queries} != set(value.case_ids)
    ):
        raise ValueError("Corrected legacy validation query inventory changed")
    expected_texts = tuple(
        select_query_text(query, query_view=value.query_view) for query in value.queries
    )
    if value.query_texts != expected_texts:
        raise ValueError("Corrected legacy validation query texts changed")
    if (
        type(value.passage_indices) is not tuple
        or len(value.passage_indices) != CORRECTED_LEGACY_VALIDATION_PASSAGES
        or value.passage_indices != tuple(sorted(value.passage_indices))
        or len(value.passage_indices) != len(set(value.passage_indices))
    ):
        raise ValueError("Corrected legacy validation passages changed")
    passage_ids = tuple(passage_index.id_for_index(index) for index in value.passage_indices)
    evaluation_data = value.evaluation_data
    if (
        not isinstance(evaluation_data, CanonicalEvaluationData)
        or evaluation_data.role != "validation"
        or evaluation_data.regime_name != REGIME_FOLD_GLOBAL
        or evaluation_data.case_ids != value.case_ids
        or tuple(query.query_id for query in evaluation_data.queries) != query_ids
        or evaluation_data.passage_ids != passage_ids
    ):
        raise ValueError("Corrected legacy canonical validation contract changed")
    if value.contract_sha256 != _sha256_payload(
        _validation_evidence_contract_payload(value, passage_index)
    ):
        raise ValueError("Corrected legacy validation evidence digest changed")
    return value


def _contract_payload(value: CorrectedLegacyTestData, passage_index: PassageIndexTable) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "query_view": value.query_view,
        "case_ids": list(value.case_ids),
        "queries": [
            {
                "query_id": query.query_id,
                "doc_id": query.doc_id,
                "query_text": query_text,
            }
            for query, query_text in zip(value.queries, value.query_texts)
        ],
        "passage_index_sha256": passage_index.sha256,
        "passage_indices": list(value.passage_indices),
        "evaluation_contracts": {
            regime: value.evaluation_data_by_regime[regime].contract_sha256
            for regime in CORRECTED_LEGACY_TEST_REGIMES
        },
    }


def _validate_test_data(
    value: object,
    passage_index: PassageIndexTable,
) -> CorrectedLegacyTestData:
    if not isinstance(value, CorrectedLegacyTestData):
        raise TypeError("Corrected legacy test data has the wrong type")
    if value.schema_version != 1:
        raise ValueError("Corrected legacy test-data schema changed")
    if value.query_view not in (QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED):
        raise ValueError("Corrected legacy test query view changed")
    if (
        type(value.case_ids) is not tuple
        or len(value.case_ids) != CORRECTED_LEGACY_TEST_CASES
        or value.case_ids != tuple(sorted(value.case_ids))
        or len(value.case_ids) != len(set(value.case_ids))
    ):
        raise ValueError("Corrected legacy test cases changed")
    if type(value.queries) is not tuple or len(value.queries) != CORRECTED_LEGACY_TEST_QUERIES:
        raise ValueError("Corrected legacy test query count changed")
    query_ids = [query.query_id for query in value.queries]
    if (
        any(not isinstance(query, QueryExample) for query in value.queries)
        or query_ids != sorted(query_ids)
        or len(query_ids) != len(set(query_ids))
        or {query.doc_id for query in value.queries} != set(value.case_ids)
    ):
        raise ValueError("Corrected legacy test query inventory changed")
    if (
        type(value.query_texts) is not tuple
        or len(value.query_texts) != len(value.queries)
        or any(type(text) is not str or not text for text in value.query_texts)
        or tuple(
            select_query_text(query, query_view=value.query_view)
            for query in value.queries
        )
        != value.query_texts
    ):
        raise ValueError("Corrected legacy test query texts changed")
    if (
        type(value.passage_indices) is not tuple
        or len(value.passage_indices) != CORRECTED_LEGACY_TEST_PASSAGES
        or value.passage_indices != tuple(sorted(value.passage_indices))
        or len(value.passage_indices) != len(set(value.passage_indices))
    ):
        raise ValueError("Corrected legacy test passage inventory changed")
    passage_ids = tuple(passage_index.id_for_index(index) for index in value.passage_indices)
    if type(value.evaluation_data_by_regime) is not MappingProxyType:
        raise TypeError("Corrected legacy regime contracts must be immutable")
    if tuple(value.evaluation_data_by_regime) != CORRECTED_LEGACY_TEST_REGIMES:
        raise ValueError("Corrected legacy test regimes or order changed")
    common_query_ids: tuple[str, ...] | None = None
    common_passage_ids: tuple[str, ...] | None = None
    for regime in CORRECTED_LEGACY_TEST_REGIMES:
        evaluation_data = value.evaluation_data_by_regime[regime]
        if (
            not isinstance(evaluation_data, CanonicalEvaluationData)
            or evaluation_data.role != "test"
            or evaluation_data.regime_name != regime
            or evaluation_data.case_ids != value.case_ids
            or evaluation_data.query_count != CORRECTED_LEGACY_TEST_QUERIES
            or evaluation_data.passage_count != CORRECTED_LEGACY_TEST_PASSAGES
        ):
            raise ValueError(f"Corrected legacy {regime} evaluation contract changed")
        regime_query_ids = tuple(query.query_id for query in evaluation_data.queries)
        common_query_ids = regime_query_ids if common_query_ids is None else common_query_ids
        common_passage_ids = (
            evaluation_data.passage_ids
            if common_passage_ids is None
            else common_passage_ids
        )
        if regime_query_ids != common_query_ids or evaluation_data.passage_ids != common_passage_ids:
            raise ValueError("Corrected legacy regime source inventories disagree")
    if tuple(query_ids) != common_query_ids or passage_ids != common_passage_ids:
        raise ValueError("Corrected legacy encoder and evaluator inventories disagree")
    if value.contract_sha256 != _sha256_payload(_contract_payload(value, passage_index)):
        raise ValueError("Corrected legacy test-data contract digest changed")
    return value


def build_corrected_legacy_test_data(
    *,
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    passage_index_table: PassageIndexTable,
    test_case_ids: Sequence[str],
    query_view: str,
) -> CorrectedLegacyTestData:
    if query_view not in (QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED):
        raise ValueError("Corrected legacy query view must be flat_masked or structured")
    if not isinstance(passage_index_table, PassageIndexTable) or len(passage_index_table) != 5_286:
        raise ValueError("Corrected legacy evaluation requires the exact 5,286-passage index")
    case_ids = tuple(sorted(test_case_ids))
    if len(case_ids) != CORRECTED_LEGACY_TEST_CASES or len(case_ids) != len(set(case_ids)):
        raise ValueError("Corrected legacy test membership must contain four unique cases")
    queries = tuple(sorted(
        (query for query in all_queries if query.doc_id in set(case_ids)),
        key=lambda query: query.query_id,
    ))
    if len(queries) != CORRECTED_LEGACY_TEST_QUERIES:
        raise RuntimeError(
            f"Corrected legacy test membership has {len(queries)} queries; expected 40"
        )
    passage_indices = tuple(
        index
        for index, passage_id in enumerate(passage_index_table.passage_ids)
        if corpus_by_passage_id[passage_id].doc_id in set(case_ids)
    )
    if len(passage_indices) != CORRECTED_LEGACY_TEST_PASSAGES:
        raise RuntimeError(
            f"Corrected legacy test membership has {len(passage_indices)} passages; expected 581"
        )
    evaluation_data = {
        regime: build_canonical_evaluation_data(
            all_queries=all_queries,
            corpus_by_passage_id=corpus_by_passage_id,
            evaluated_case_ids=case_ids,
            role="test",
            regime_name=regime,
        )
        for regime in CORRECTED_LEGACY_TEST_REGIMES
    }
    value = CorrectedLegacyTestData(
        schema_version=1,
        query_view=query_view,
        case_ids=case_ids,
        queries=queries,
        query_texts=tuple(select_query_text(query, query_view=query_view) for query in queries),
        passage_indices=passage_indices,
        evaluation_data_by_regime=MappingProxyType(evaluation_data),
        contract_sha256="",
    )
    object.__setattr__(
        value,
        "contract_sha256",
        _sha256_payload(_contract_payload(value, passage_index_table)),
    )
    return _validate_test_data(value, passage_index_table)


def build_corrected_legacy_validation_evidence_data(
    *,
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    passage_index_table: PassageIndexTable,
    validation_case_ids: Sequence[str],
    query_view: str,
) -> CorrectedLegacyValidationEvidenceData:
    if query_view not in (QUERY_VIEW_FLAT_MASKED, QUERY_VIEW_STRUCTURED):
        raise ValueError("Corrected legacy validation view must be flat_masked or structured")
    if not isinstance(passage_index_table, PassageIndexTable) or len(passage_index_table) != 5_286:
        raise ValueError("Corrected legacy validation requires the exact passage index")
    case_ids = tuple(sorted(validation_case_ids))
    if len(case_ids) != 4 or len(case_ids) != len(set(case_ids)):
        raise ValueError("Corrected legacy validation membership must contain four cases")
    case_set = set(case_ids)
    queries = tuple(
        sorted(
            (query for query in all_queries if query.doc_id in case_set),
            key=lambda query: query.query_id,
        )
    )
    if len(queries) != CORRECTED_LEGACY_VALIDATION_QUERIES:
        raise RuntimeError("Corrected legacy validation membership must contain 32 queries")
    passage_indices = tuple(
        index
        for index, passage_id in enumerate(passage_index_table.passage_ids)
        if corpus_by_passage_id[passage_id].doc_id in case_set
    )
    if len(passage_indices) != CORRECTED_LEGACY_VALIDATION_PASSAGES:
        raise RuntimeError("Corrected legacy validation membership must contain 398 passages")
    evaluation_data = build_canonical_evaluation_data(
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        evaluated_case_ids=case_ids,
        role="validation",
        regime_name=REGIME_FOLD_GLOBAL,
    )
    value = CorrectedLegacyValidationEvidenceData(
        schema_version=1,
        query_view=query_view,
        case_ids=case_ids,
        queries=queries,
        query_texts=tuple(select_query_text(query, query_view=query_view) for query in queries),
        passage_indices=passage_indices,
        evaluation_data=evaluation_data,
        contract_sha256="",
    )
    object.__setattr__(
        value,
        "contract_sha256",
        _sha256_payload(_validation_evidence_contract_payload(value, passage_index_table)),
    )
    return _validate_validation_evidence_data(value, passage_index_table)


def _encode_role_distributed(
    model,
    tokenizer,
    *,
    query_texts: tuple[str, ...],
    passage_indices: tuple[int, ...],
    passage_index_table: PassageIndexTable,
    contract_sha256: str,
    contract_name: str,
    expected_query_count: int,
    expected_passage_count: int,
    forward_steps: int,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(f"{contract_name} requires distributed initialization")
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    if world_size != CORRECTED_LEGACY_WORLD_SIZE:
        raise RuntimeError(f"{contract_name} requires exactly four ranks")
    if (
        type(query_texts) is not tuple
        or len(query_texts) != expected_query_count
        or any(type(text) is not str or not text for text in query_texts)
        or type(passage_indices) is not tuple
        or len(passage_indices) != expected_passage_count
        or passage_indices != tuple(sorted(passage_indices))
        or len(passage_indices) != len(set(passage_indices))
        or type(contract_sha256) is not str
        or len(contract_sha256) != 64
    ):
        raise ValueError(f"{contract_name} encoder inventory changed")
    gathered_contracts: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_contracts, contract_sha256, group=group)
    if gathered_contracts != [contract_sha256] * world_size:
        raise RuntimeError(f"{contract_name} contracts differ across ranks")

    query_positions = tuple(range(rank, expected_query_count, world_size))
    passage_positions = tuple(range(rank, expected_passage_count, world_size))
    query_chunks = _balanced_nonempty_chunks(query_positions, chunk_count=forward_steps)
    passage_chunks = _balanced_nonempty_chunks(passage_positions, chunk_count=forward_steps)
    if max(map(len, query_chunks)) > CORRECTED_LEGACY_QUERY_BATCH_CAP:
        raise RuntimeError(f"{contract_name} query schedule exceeds its batch cap")
    if max(map(len, passage_chunks)) > CORRECTED_LEGACY_PASSAGE_BATCH_CAP:
        raise RuntimeError(f"{contract_name} passage schedule exceeds its batch cap")

    def inspect_model():
        retriever = model.module if hasattr(model, "module") else model
        try:
            device = next(retriever.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise TypeError(f"{contract_name} model must expose parameters") from error
        was_training = bool(retriever.training)
        if not hasattr(model, "training") or bool(model.training) != was_training:
            raise RuntimeError(f"{contract_name} engine/model modes disagree")
        return device, was_training

    device, was_training = _coordinated_local_call(
        f"{contract_name} model preflight",
        inspect_model,
        group=group,
    )
    local_queries: list[torch.Tensor] = []
    local_passages: list[torch.Tensor] = []
    try:
        _coordinated_local_call(
            f"{contract_name} enter evaluation mode",
            lambda: model.eval(),
            group=group,
        )
        with torch.no_grad():
            for query_chunk, passage_chunk in zip(query_chunks, passage_chunks):
                query_tokens, passage_tokens = _coordinated_local_call(
                    f"{contract_name} tokenization",
                    lambda: (
                        _tokenize_validation_texts(
                            tokenizer,
                            [query_texts[position] for position in query_chunk],
                            truncation_side="left",
                            max_length=CORRECTED_LEGACY_MAX_QUERY_TOKENS,
                            require_slot=True,
                        ),
                        _tokenize_validation_texts(
                            tokenizer,
                            [
                                passage_index_table.text_for_index(
                                    passage_indices[position]
                                )
                                for position in passage_chunk
                            ],
                            truncation_side="right",
                            max_length=CORRECTED_LEGACY_MAX_PASSAGE_TOKENS,
                            require_slot=False,
                        ),
                    ),
                    group=group,
                )
                query_vectors, passage_vectors = _coordinated_local_call(
                    f"{contract_name} top-level forward",
                    lambda: _forward_validation_pair(
                        model,
                        query_tokens=query_tokens,
                        passage_tokens=passage_tokens,
                        device=device,
                    ),
                    group=group,
                )
                if query_vectors.shape[0] != len(query_chunk):
                    raise RuntimeError(f"{contract_name} query forward count changed")
                if passage_vectors.shape[0] != len(passage_chunk):
                    raise RuntimeError(f"{contract_name} passage forward count changed")
                local_queries.append(query_vectors.detach())
                local_passages.append(passage_vectors.detach())
    finally:
        _coordinated_local_call(
            f"{contract_name} restore model mode",
            lambda: model.train(was_training),
            group=group,
        )
    return (
        _all_gather_positioned_embeddings(
            torch.cat(local_queries, dim=0),
            query_positions,
            total_count=expected_query_count,
            group=group,
        ),
        _all_gather_positioned_embeddings(
            torch.cat(local_passages, dim=0),
            passage_positions,
            total_count=expected_passage_count,
            group=group,
        ),
    )


def evaluate_corrected_legacy_validation_evidence_distributed(
    model,
    tokenizer,
    *,
    validation_data: CorrectedLegacyValidationEvidenceData,
    passage_index_table: PassageIndexTable,
    group=None,
) -> CanonicalRetrievalResult:
    """Return one replayable complete fold-global validation result."""

    validation_data = _validate_validation_evidence_data(
        validation_data,
        passage_index_table,
    )
    query_embeddings, passage_embeddings = _encode_role_distributed(
        model,
        tokenizer,
        query_texts=validation_data.query_texts,
        passage_indices=validation_data.passage_indices,
        passage_index_table=passage_index_table,
        contract_sha256=validation_data.contract_sha256,
        contract_name="Corrected legacy validation evidence",
        expected_query_count=CORRECTED_LEGACY_VALIDATION_QUERIES,
        expected_passage_count=CORRECTED_LEGACY_VALIDATION_PASSAGES,
        forward_steps=CORRECTED_LEGACY_VALIDATION_FORWARD_STEPS,
        group=group,
    )
    rank = dist.get_rank(group=group)
    collective: list[object] = [None]
    if rank == 0:
        try:
            scores = (
                query_embeddings.detach().to(device="cpu", dtype=torch.float32)
                @ passage_embeddings.detach().to(device="cpu", dtype=torch.float32).T
            )
            result = compute_canonical_retrieval_result_from_scores(
                scores=scores,
                evaluation_data=validation_data.evaluation_data,
            )
            collective[0] = {"ok": True, "result": result.to_payload()}
        except BaseException as error:
            collective[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "message": str(error),
            }
    source_rank = 0 if group is None else dist.get_global_rank(group, 0)
    dist.broadcast_object_list(collective, src=source_rank, group=group)
    status = collective[0]
    if type(status) is not dict or status.get("ok") is not True:
        raise RuntimeError(
            "Corrected legacy validation evidence failed: "
            f"{status.get('error_type') if type(status) is dict else type(status).__name__}: "
            f"{status.get('message') if type(status) is dict else status}"
        )
    return canonical_result_from_payload(
        status.get("result"),
        validation_data.evaluation_data,
    )


def evaluate_corrected_legacy_test_distributed(
    model,
    tokenizer,
    *,
    test_data: CorrectedLegacyTestData,
    passage_index_table: PassageIndexTable,
    group=None,
) -> Mapping[str, CanonicalRetrievalResult]:
    """Encode the exact corrected test role on all ranks and broadcast four results."""

    test_data = _validate_test_data(test_data, passage_index_table)
    rank = dist.get_rank(group=group)
    query_embeddings, passage_embeddings = _encode_role_distributed(
        model,
        tokenizer,
        query_texts=test_data.query_texts,
        passage_indices=test_data.passage_indices,
        passage_index_table=passage_index_table,
        contract_sha256=test_data.contract_sha256,
        contract_name="Corrected legacy test evaluation",
        expected_query_count=CORRECTED_LEGACY_TEST_QUERIES,
        expected_passage_count=CORRECTED_LEGACY_TEST_PASSAGES,
        forward_steps=CORRECTED_LEGACY_TEST_FORWARD_STEPS,
        group=group,
    )

    collective: list[object] = [None]
    if rank == 0:
        try:
            scores = (
                query_embeddings.detach().to(device="cpu", dtype=torch.float32)
                @ passage_embeddings.detach().to(device="cpu", dtype=torch.float32).T
            )
            results = {
                regime: compute_canonical_retrieval_result_from_scores(
                    scores=scores,
                    evaluation_data=test_data.evaluation_data_by_regime[regime],
                ).to_payload()
                for regime in CORRECTED_LEGACY_TEST_REGIMES
            }
            source_hashes = {value["source_ranking_sha256"] for value in results.values()}
            if len(source_hashes) != 1:
                raise RuntimeError("Corrected legacy regimes changed the source ranking")
            collective[0] = {"ok": True, "results": results}
        except BaseException as error:
            collective[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "message": str(error),
            }
    source_rank = 0 if group is None else dist.get_global_rank(group, 0)
    dist.broadcast_object_list(collective, src=source_rank, group=group)
    status = collective[0]
    if type(status) is not dict or status.get("ok") is not True:
        raise RuntimeError(
            "Corrected legacy test result computation failed: "
            f"{status.get('error_type') if type(status) is dict else type(status).__name__}: "
            f"{status.get('message') if type(status) is dict else status}"
        )
    raw_results = status.get("results")
    if type(raw_results) is not dict or tuple(raw_results) != CORRECTED_LEGACY_TEST_REGIMES:
        raise RuntimeError("Corrected legacy broadcast result regimes changed")
    validated = {
        regime: canonical_result_from_payload(
            raw_results[regime],
            test_data.evaluation_data_by_regime[regime],
        )
        for regime in CORRECTED_LEGACY_TEST_REGIMES
    }
    if len({result.source_ranking_sha256 for result in validated.values()}) != 1:
        raise RuntimeError("Validated corrected legacy source rankings differ by regime")
    return MappingProxyType(validated)


__all__ = [
    "CORRECTED_LEGACY_TEST_REGIMES",
    "CorrectedLegacyTestData",
    "CorrectedLegacyValidationEvidenceData",
    "build_corrected_legacy_test_data",
    "build_corrected_legacy_validation_evidence_data",
    "evaluate_corrected_legacy_test_distributed",
    "evaluate_corrected_legacy_validation_evidence_distributed",
]
