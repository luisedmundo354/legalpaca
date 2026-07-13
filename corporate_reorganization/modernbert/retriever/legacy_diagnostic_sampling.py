from __future__ import annotations

import hashlib
import heapq
import json
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .data import PassageIndexTable
from .legacy_diagnostic_data import CorrectedLegacyData, CorrectedLegacyQuery
from .query_views import QUERY_VIEW_STRUCTURED, normalize_query_view, select_query_text


MAX_SELECTED_POSITIVES: Final[int] = 4
BASE_SAME_CASE_DRAWS: Final[int] = 56
OTHER_CASE_BACKGROUND_DRAWS: Final[int] = 4
CANDIDATE_OCCURRENCES_PER_QUERY: Final[int] = 64

SELECTION_ALGORITHM: Final[str] = "sha256_corrected_legacy_occurrences_v1"
TRACE_SCHEMA_VERSION: Final[int] = 1

ROLE_POSITIVE: Final[str] = "positive"
ROLE_SAME_CASE: Final[str] = "same_case_negative"
ROLE_OTHER_BACKGROUND: Final[str] = "other_training_case_background"
ROLES: Final[tuple[str, ...]] = (
    ROLE_POSITIVE,
    ROLE_SAME_CASE,
    ROLE_OTHER_BACKGROUND,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_int(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _require_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string, not {type(value).__name__}")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and whitespace-trimmed")
    return value


def _pool_sha256(passage_ids: Sequence[str]) -> str:
    pool = tuple(sorted(passage_ids))
    if len(pool) != len(set(pool)):
        raise ValueError("Cannot checksum a sampling pool containing duplicate passage IDs")
    return hashlib.sha256(_canonical_json_bytes(list(pool))).hexdigest()


def _selection_digest(
    *,
    experiment_seed: int,
    epoch: int,
    query_id: str,
    role: str,
    passage_id: str,
    replacement_ordinal: int | None,
) -> bytes:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "algorithm": SELECTION_ALGORITHM,
                "epoch": epoch,
                "experiment_seed": experiment_seed,
                "passage_id": passage_id,
                "query_id": query_id,
                "replacement_ordinal": replacement_ordinal,
                "role": role,
            }
        )
    ).digest()


def _sample_occurrences(
    passage_ids: Sequence[str],
    count: int,
    *,
    experiment_seed: int,
    epoch: int,
    query_id: str,
    role: str,
) -> list[tuple[str, bool, str]]:
    _require_exact_int("count", count, minimum=0)
    pool = tuple(passage_ids)
    if not pool and count:
        raise ValueError(f"Sampling pool for role={role!r} is empty but count={count}")
    if any(type(passage_id) is not str or not passage_id for passage_id in pool):
        raise ValueError(f"Sampling pool for role={role!r} contains an invalid passage ID")
    if len(pool) != len(set(pool)):
        raise ValueError(f"Sampling pool for role={role!r} contains duplicate passage IDs")

    selected: list[tuple[str, bool, str]] = []
    if count <= len(pool):
        ranked = heapq.nsmallest(
            count,
            pool,
            key=lambda passage_id: (
                _selection_digest(
                    experiment_seed=experiment_seed,
                    epoch=epoch,
                    query_id=query_id,
                    role=role,
                    passage_id=passage_id,
                    replacement_ordinal=None,
                ),
                passage_id,
            ),
        )
        for passage_id in ranked:
            digest = _selection_digest(
                experiment_seed=experiment_seed,
                epoch=epoch,
                query_id=query_id,
                role=role,
                passage_id=passage_id,
                replacement_ordinal=None,
            )
            selected.append((passage_id, False, digest.hex()))
    else:
        # Preserve the intended legacy sampling mode: when the pool is too small,
        # the whole k-draw operation is with replacement (not a unique prefix plus
        # a replacement-only deficit). Each draw has an independent SHA-256 key.
        for role_ordinal in range(count):
            passage_id = min(
                pool,
                key=lambda passage_id: (
                    _selection_digest(
                        experiment_seed=experiment_seed,
                        epoch=epoch,
                        query_id=query_id,
                        role=role,
                        passage_id=passage_id,
                        replacement_ordinal=role_ordinal,
                    ),
                    passage_id,
                ),
            )
            digest = _selection_digest(
                experiment_seed=experiment_seed,
                epoch=epoch,
                query_id=query_id,
                role=role,
                passage_id=passage_id,
                replacement_ordinal=role_ordinal,
            )
            selected.append((passage_id, True, digest.hex()))

    if len(selected) != count:
        raise RuntimeError(f"Internal {role} selection count mismatch")
    if count <= len(pool) and any(with_replacement for _, with_replacement, _ in selected):
        raise RuntimeError(f"Internal {role} sampler used replacement unnecessarily")
    return selected


def legacy_diagnostic_trace_checksum(trace_without_checksum: Mapping[str, Any]) -> str:
    if "trace_sha256" in trace_without_checksum:
        raise ValueError("trace_without_checksum must not contain trace_sha256")
    return hashlib.sha256(_canonical_json_bytes(trace_without_checksum)).hexdigest()


def validate_legacy_diagnostic_trace(trace: Mapping[str, Any]) -> None:
    if type(trace) is not dict:
        raise TypeError("Corrected-legacy trace must be an exact JSON object")
    expected_keys = {
        "schema_version",
        "selection_algorithm",
        "experiment_seed",
        "epoch",
        "query_id",
        "doc_id",
        "membership_sha256",
        "all_gold_passage_ids",
        "selected_positive_passage_ids",
        "case_wide_gold_sha256",
        "eligible_pool_sizes_by_role",
        "eligible_pool_sha256_by_role",
        "quota_by_role",
        "occurrences",
        "unique_candidate_passage_ids",
        "multiplicity_by_unique_candidate",
        "replacement_count_by_role",
        "trace_sha256",
    }
    if set(trace) != expected_keys:
        raise ValueError(
            "Corrected-legacy trace schema changed: "
            f"missing={sorted(expected_keys - set(trace))}, extra={sorted(set(trace) - expected_keys)}"
        )
    if trace["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported corrected-legacy trace schema={trace['schema_version']!r}")
    if trace["selection_algorithm"] != SELECTION_ALGORITHM:
        raise ValueError("Corrected-legacy trace selection algorithm changed")
    _require_exact_int("trace experiment_seed", trace["experiment_seed"], minimum=0)
    _require_exact_int("trace epoch", trace["epoch"], minimum=0)
    _require_string("trace query_id", trace["query_id"])
    doc_id = _require_string("trace doc_id", trace["doc_id"])
    _require_string("trace membership_sha256", trace["membership_sha256"])
    _require_string("trace case_wide_gold_sha256", trace["case_wide_gold_sha256"])

    all_gold = trace["all_gold_passage_ids"]
    selected_positives = trace["selected_positive_passage_ids"]
    if type(all_gold) is not list or not all_gold or len(all_gold) != len(set(all_gold)):
        raise ValueError("Trace all_gold_passage_ids must be a non-empty unique JSON list")
    if type(selected_positives) is not list or not selected_positives:
        raise ValueError("Trace selected_positive_passage_ids must be a non-empty JSON list")
    if len(selected_positives) > MAX_SELECTED_POSITIVES or len(selected_positives) != len(
        set(selected_positives)
    ):
        raise ValueError("Trace selected positives must contain at most four unique IDs")
    if not set(selected_positives).issubset(all_gold):
        raise ValueError("Trace selected positives are not a subset of all gold IDs")

    expected_positive_count = min(len(all_gold), MAX_SELECTED_POSITIVES)
    expected_quotas = {
        ROLE_POSITIVE: expected_positive_count,
        ROLE_SAME_CASE: BASE_SAME_CASE_DRAWS
        + (MAX_SELECTED_POSITIVES - expected_positive_count),
        ROLE_OTHER_BACKGROUND: OTHER_CASE_BACKGROUND_DRAWS,
    }
    quotas = trace["quota_by_role"]
    pool_sizes = trace["eligible_pool_sizes_by_role"]
    pool_hashes = trace["eligible_pool_sha256_by_role"]
    replacement_counts = trace["replacement_count_by_role"]
    for name, value in (
        ("quota_by_role", quotas),
        ("eligible_pool_sizes_by_role", pool_sizes),
        ("eligible_pool_sha256_by_role", pool_hashes),
        ("replacement_count_by_role", replacement_counts),
    ):
        if type(value) is not dict or set(value) != set(ROLES):
            raise ValueError(f"Trace {name} must contain exactly the three corrected-legacy roles")
    if quotas != expected_quotas:
        raise ValueError(f"Trace role quotas changed: {quotas!r} != {expected_quotas!r}")
    for role in ROLES:
        _require_exact_int(f"trace pool size {role}", pool_sizes[role], minimum=1)
        _require_string(f"trace pool checksum {role}", pool_hashes[role])
        _require_exact_int(f"trace replacement count {role}", replacement_counts[role], minimum=0)

    occurrences = trace["occurrences"]
    if type(occurrences) is not list or len(occurrences) != CANDIDATE_OCCURRENCES_PER_QUERY:
        raise ValueError("Trace must contain exactly 64 candidate occurrences")
    expected_occurrence_keys = {
        "candidate_position",
        "role",
        "role_ordinal",
        "passage_id",
        "source_doc_id",
        "with_replacement",
        "selection_sha256",
    }
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for position, occurrence in enumerate(occurrences):
        if type(occurrence) is not dict or set(occurrence) != expected_occurrence_keys:
            raise ValueError(f"Trace occurrence {position} schema changed")
        if occurrence["candidate_position"] != position:
            raise ValueError("Trace candidate positions must be contiguous and ordered")
        role = occurrence["role"]
        if role not in by_role:
            raise ValueError(f"Trace occurrence has unknown role={role!r}")
        if occurrence["role_ordinal"] != len(by_role[role]):
            raise ValueError(f"Trace role ordinals are not contiguous for role={role}")
        _require_string(f"occurrence {position} passage_id", occurrence["passage_id"])
        source_doc_id = _require_string(
            f"occurrence {position} source_doc_id", occurrence["source_doc_id"]
        )
        if type(occurrence["with_replacement"]) is not bool:
            raise TypeError("Trace occurrence with_replacement must be an exact bool")
        _require_string(f"occurrence {position} selection_sha256", occurrence["selection_sha256"])
        if role in (ROLE_POSITIVE, ROLE_SAME_CASE) and source_doc_id != doc_id:
            raise ValueError(f"Trace {role} occurrence must come from the query case")
        if role == ROLE_OTHER_BACKGROUND and source_doc_id == doc_id:
            raise ValueError("Trace other-case Background Facts occurrence came from the query case")
        by_role[role].append(occurrence)

    for role, role_occurrences in by_role.items():
        if len(role_occurrences) != quotas[role]:
            raise ValueError(f"Trace occurrence count for role={role} does not match its quota")
        expected_replacements = quotas[role] if quotas[role] > pool_sizes[role] else 0
        replacement_flags = [item["with_replacement"] for item in role_occurrences]
        expected_flags = [bool(expected_replacements)] * quotas[role]
        if replacement_flags != expected_flags:
            raise ValueError(f"Trace replacement flags are not minimal for role={role}")
        if replacement_counts[role] != expected_replacements:
            raise ValueError(f"Trace replacement count is not minimal for role={role}")

    if [item["passage_id"] for item in by_role[ROLE_POSITIVE]] != selected_positives:
        raise ValueError("Trace positive occurrences differ from selected_positive_passage_ids")

    candidate_ids = [item["passage_id"] for item in occurrences]
    unique_ids = trace["unique_candidate_passage_ids"]
    multiplicities = trace["multiplicity_by_unique_candidate"]
    expected_unique = sorted(set(candidate_ids))
    if unique_ids != expected_unique:
        raise ValueError("Trace unique candidate IDs must be canonically sorted")
    if type(multiplicities) is not list or len(multiplicities) != len(unique_ids):
        raise ValueError("Trace multiplicities must align with unique candidate IDs")
    counts = Counter(candidate_ids)
    if multiplicities != [counts[passage_id] for passage_id in unique_ids]:
        raise ValueError("Trace multiplicities do not reconstruct the 64 occurrences")
    if sum(multiplicities) != CANDIDATE_OCCURRENCES_PER_QUERY:
        raise ValueError("Trace candidate multiplicities do not sum to 64")

    stored_checksum = _require_string("trace trace_sha256", trace["trace_sha256"])
    payload = {key: value for key, value in trace.items() if key != "trace_sha256"}
    if stored_checksum != legacy_diagnostic_trace_checksum(payload):
        raise ValueError("Corrected-legacy trace SHA-256 mismatch")


class CorrectedLegacyDiagnosticDataset:
    """Corrected 64-occurrence sampler, isolated from reconstructed March code."""

    def __init__(
        self,
        data: CorrectedLegacyData,
        *,
        experiment_seed: int,
        query_view: str = QUERY_VIEW_STRUCTURED,
    ) -> None:
        if not isinstance(data, CorrectedLegacyData):
            raise TypeError("data must be CorrectedLegacyData")
        self.data = data
        self.queries = data.queries_by_split["train"]
        if len(self.queries) != 418:
            raise ValueError("Corrected legacy diagnostic requires exactly 418 train queries")
        self.experiment_seed = _require_exact_int(
            "experiment_seed", experiment_seed, minimum=0
        )
        self.query_view = normalize_query_view(query_view)
        self.epoch = 0
        self.passage_index_table = PassageIndexTable(data.corpus_by_passage_id)

    def __len__(self) -> int:
        return len(self.queries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _require_exact_int("epoch", epoch, minimum=0)

    def _sample_query(self, query: CorrectedLegacyQuery) -> dict[str, Any]:
        all_gold = tuple(sorted(query.positive_passage_ids))
        selected_positive_occurrences = _sample_occurrences(
            all_gold,
            min(len(all_gold), MAX_SELECTED_POSITIVES),
            experiment_seed=self.experiment_seed,
            epoch=self.epoch,
            query_id=query.query_id,
            role=ROLE_POSITIVE,
        )

        case_gold = self.data.gold_passage_ids_by_case[query.doc_id]
        same_case_pool = tuple(
            passage_id
            for passage_id in self.data.candidate_passage_ids_by_case[query.doc_id]
            if passage_id not in case_gold
        )
        same_case_count = BASE_SAME_CASE_DRAWS + (
            MAX_SELECTED_POSITIVES - len(selected_positive_occurrences)
        )
        same_case_occurrences = _sample_occurrences(
            same_case_pool,
            same_case_count,
            experiment_seed=self.experiment_seed,
            epoch=self.epoch,
            query_id=query.query_id,
            role=ROLE_SAME_CASE,
        )

        other_background_pool = tuple(
            passage_id
            for passage_id in self.data.training_background_passage_ids
            if self.data.corpus_by_passage_id[passage_id].doc_id != query.doc_id
        )
        other_background_occurrences = _sample_occurrences(
            other_background_pool,
            OTHER_CASE_BACKGROUND_DRAWS,
            experiment_seed=self.experiment_seed,
            epoch=self.epoch,
            query_id=query.query_id,
            role=ROLE_OTHER_BACKGROUND,
        )

        selected_by_role = {
            ROLE_POSITIVE: selected_positive_occurrences,
            ROLE_SAME_CASE: same_case_occurrences,
            ROLE_OTHER_BACKGROUND: other_background_occurrences,
        }
        pool_by_role = {
            ROLE_POSITIVE: all_gold,
            ROLE_SAME_CASE: same_case_pool,
            ROLE_OTHER_BACKGROUND: other_background_pool,
        }
        occurrences: list[dict[str, Any]] = []
        for role in ROLES:
            for role_ordinal, (passage_id, with_replacement, selection_sha256) in enumerate(
                selected_by_role[role]
            ):
                occurrences.append(
                    {
                        "candidate_position": len(occurrences),
                        "role": role,
                        "role_ordinal": role_ordinal,
                        "passage_id": passage_id,
                        "source_doc_id": self.data.corpus_by_passage_id[passage_id].doc_id,
                        "with_replacement": with_replacement,
                        "selection_sha256": selection_sha256,
                    }
                )
        if len(occurrences) != CANDIDATE_OCCURRENCES_PER_QUERY:
            raise RuntimeError("Corrected-legacy sampler did not produce exactly 64 occurrences")

        candidate_ids = [item["passage_id"] for item in occurrences]
        unique_candidate_ids = sorted(set(candidate_ids))
        counts = Counter(candidate_ids)
        payload: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "selection_algorithm": SELECTION_ALGORITHM,
            "experiment_seed": self.experiment_seed,
            "epoch": self.epoch,
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "membership_sha256": self.data.membership_sha256,
            "all_gold_passage_ids": list(all_gold),
            "selected_positive_passage_ids": [
                passage_id for passage_id, _, _ in selected_positive_occurrences
            ],
            "case_wide_gold_sha256": _pool_sha256(tuple(case_gold)),
            "eligible_pool_sizes_by_role": {
                role: len(pool_by_role[role]) for role in ROLES
            },
            "eligible_pool_sha256_by_role": {
                role: _pool_sha256(pool_by_role[role]) for role in ROLES
            },
            "quota_by_role": {
                role: len(selected_by_role[role]) for role in ROLES
            },
            "occurrences": occurrences,
            "unique_candidate_passage_ids": unique_candidate_ids,
            "multiplicity_by_unique_candidate": [
                counts[passage_id] for passage_id in unique_candidate_ids
            ],
            "replacement_count_by_role": {
                role: sum(
                    int(with_replacement)
                    for _, with_replacement, _ in selected_by_role[role]
                )
                for role in ROLES
            },
        }
        trace = {**payload, "trace_sha256": legacy_diagnostic_trace_checksum(payload)}
        validate_legacy_diagnostic_trace(trace)

        return {
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "query_text": select_query_text(query, query_view=self.query_view),
            "positive_passage_indices": self.passage_index_table.indices_for_ids(list(all_gold)),
            "candidate_passage_occurrence_indices": self.passage_index_table.indices_for_ids(
                candidate_ids
            ),
            "unique_candidate_passage_indices": self.passage_index_table.indices_for_ids(
                unique_candidate_ids
            ),
            "candidate_multiplicities": [counts[passage_id] for passage_id in unique_candidate_ids],
            "sampling_trace": trace,
            "sampling_trace_sha256": trace["trace_sha256"],
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if isinstance(idx, bool):
            raise TypeError("Dataset index must be an integer, not bool")
        try:
            index = operator.index(idx)
        except TypeError as exc:
            raise TypeError(f"Dataset index must be an integer; got {type(idx).__name__}") from exc
        if index < 0 or index >= len(self.queries):
            raise IndexError(f"Dataset index out of range: {index}")
        return self._sample_query(self.queries[index])
