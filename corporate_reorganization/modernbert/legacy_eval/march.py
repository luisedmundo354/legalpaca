"""Hash-gated, read-only replay of the reconstructed March 2026 archive.

This module verifies rankings that already exist.  It deliberately contains no
model loader, S3 client, scorer, or artifact writer, and it does not establish
model-to-ranking reproducibility.  The historical names and arithmetic below
are retained only inside ``reconstructed_march_2026``.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple


RECONSTRUCTED_MARCH_NAMESPACE = "reconstructed_march_2026"

FROZEN_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "config.json": "f54aa4259d7502107004eba16651727fae69201394f53c29054671a6111526fc",
        "results.json": "410c6f5b4dfe11cb57caa9aca31f95eec0657bc969c297128a2e4e2f9ae6f133",
        "runs/rankings.jsonl": "cea4caf334e05de390b0eab7abcb438cd97f6d461fbb73b9febd59d38015b6fd",
        "corpus.jsonl": "7a52531a58d3eda19a8363516bf4c5c34687c1211f94a4e89638d2eab08f7a8f",
        "queries/test.jsonl": "d412eedd00aff5cc2f3b3ffb49af8b0f5a144a850c559e9dfeac092b0d1122bb",
        "pools/candidates_by_case.json": "6ce68bae370676d6b40b84862dbe1f6f70e5168b72ba8fe925812db7dc5d7e2e",
        "splits/test_cases.txt": "ed59e5830650a1947b86e986db0820ca888589dc54f132d6d0620efa5e3f37c7",
    }
)

LEGACY_REGIME_SEMANTICS: Mapping[str, str] = MappingProxyType(
    {
        "same_case_legacy": "same_case_legacy",
        "same_case_full": "same_case_full",
        "global_split": "fold_global",
    }
)

_ARCHIVE_FILES = {
    "config.json": "config.json",
    "results.json": "results.json",
    "runs/rankings.jsonl": "runs/rankings.jsonl",
}
_PROCESSED_FILES = {
    "corpus.jsonl": "corpus.jsonl",
    "queries/test.jsonl": "queries/test.jsonl",
    "pools/candidates_by_case.json": "pools/candidates_by_case.json",
    "splits/test_cases.txt": "splits/test_cases.txt",
}

_EXPECTED_SYSTEMS: Tuple[Tuple[str, str, str], ...] = (
    ("bm25_flat", "bm25_pyserini", "flat_plain"),
    ("dense_open_flat", "open_dense", "flat_plain"),
    ("base_modernbert_flat", "modernbert_base", "flat_masked"),
    ("fine_tuned_flat", "modernbert_artifact", "flat_masked"),
    ("fine_tuned_structured", "modernbert_artifact", "structured"),
)
_EXPECTED_REGIMES = tuple(LEGACY_REGIME_SEMANTICS)
_EXPECTED_KS = (1, 5, 10, 20)
_EXPECTED_QUERY_COUNT = 40
_EXPECTED_ROW_COUNT = 600
_EXPECTED_CELL_COUNT = 15
_EXPECTED_NUMERIC_VALUE_COUNT = 3300
_EXPECTED_STABLE_REORDERED_ROWS = 386

_RANKING_KEYS = (
    "system",
    "system_type",
    "query_view",
    "regime",
    "query_id",
    "doc_id",
    "query_text",
    "candidate_pool_size",
    "gold_passage_ids",
    "ranked_candidates",
)
_RANKED_CANDIDATE_KEYS = ("rank", "passage_id", "score")
_QUERY_KEYS = (
    "query_id",
    "doc_id",
    "motion_root_id",
    "mask_parent_id",
    "query_text",
    "flat_query_text_plain",
    "flat_query_text_masked",
    "positive_passage_ids",
    "positive_labels",
)
_CORPUS_KEYS = (
    "passage_id",
    "doc_id",
    "label",
    "text",
    "start",
    "end",
    "source_node_id",
    "is_implicit",
    "order",
)
_CONFIG_KEYS = (
    "timestamp",
    "processed_dir",
    "output_dir",
    "split",
    "k_values",
    "max_len_query",
    "max_len_passage",
    "query_batch_size",
    "passage_batch_size",
    "random_seed",
    "split_doc_ids",
    "split_passage_count",
    "systems",
    "regimes",
    "data_sha256",
)
_SYSTEM_CONFIG_KEYS = (
    "name",
    "system_type",
    "query_view",
    "model_name_or_path",
    "model_dir",
    "model_s3_uri",
    "work_dir",
    "temperature",
    "query_prefix",
    "passage_prefix",
    "bm25_k1",
    "bm25_b",
    "cohere_model_name",
    "cohere_api_key_env",
    "cohere_output_dimension",
)
_METRIC_KEYS = tuple(
    key
    for k in _EXPECTED_KS
    for key in (
        f"recall_at_{k}",
        f"mrr_at_{k}",
        f"set_recall_at_{k}",
        f"exact_set_match_at_{k}",
    )
) + ("num_gold", "first_hit_rank", "candidate_pool_size")
_GLOBAL_METRIC_KEYS = _METRIC_KEYS + ("num_queries",)
_BREAKDOWN_NAMES = (
    "by_missing_type",
    "by_gold_label_composition",
    "by_num_positives",
    "by_has_implicit",
)


class MarchReplayError(RuntimeError):
    """The immutable March evidence failed a replay invariant."""


@dataclass(frozen=True)
class MarchReplayResult:
    """Immutable proof summary and recomputed historical result cells."""

    namespace: str
    sha256_by_input: Mapping[str, str]
    systems: Tuple[str, ...]
    legacy_regime_labels: Tuple[str, ...]
    semantic_regime_by_legacy_label: Mapping[str, str]
    query_count: int
    ranking_row_count: int
    result_cell_count: int
    numeric_values_verified: int
    stable_tie_rows_reordered: int
    stable_tie_metric_changes: int
    replayed_regimes: Mapping[str, Any]


@dataclass(frozen=True)
class _Query:
    query_id: str
    doc_id: str
    query_text: str
    flat_query_text_plain: str
    flat_query_text_masked: str
    positive_passage_ids: Tuple[str, ...]
    positive_labels: Tuple[str, ...]


def replay_reconstructed_march(
    *,
    archive_dir: Path,
    processed_dir: Path,
) -> MarchReplayResult:
    """Verify and replay the exact reconstructed March archive without writes.

    ``archive_dir`` must contain the frozen config, results, and full rankings;
    ``processed_dir`` must contain the frozen legacy corpus, test queries, case
    pool, and test split.  Every file is read once and checked by byte SHA-256
    before any JSON is trusted.
    """

    archive_dir = Path(archive_dir)
    processed_dir = Path(processed_dir)
    frozen_bytes: Dict[str, bytes] = {}
    for logical_name, relative_name in _ARCHIVE_FILES.items():
        frozen_bytes[logical_name] = _read_frozen_file(
            archive_dir / relative_name,
            logical_name=logical_name,
        )
    for logical_name, relative_name in _PROCESSED_FILES.items():
        frozen_bytes[logical_name] = _read_frozen_file(
            processed_dir / relative_name,
            logical_name=logical_name,
        )
    return _replay_hash_verified_bytes(frozen_bytes)


def _read_frozen_file(path: Path, *, logical_name: str) -> bytes:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise MarchReplayError(
            f"Missing frozen March input {logical_name!r}: {path}"
        ) from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise MarchReplayError(
            f"Frozen March input {logical_name!r} must be a regular, non-symlink file: {path}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MarchReplayError(
            f"Could not read frozen March input {logical_name!r}: {path}: {exc}"
        ) from exc
    actual = hashlib.sha256(payload).hexdigest()
    expected = FROZEN_SHA256[logical_name]
    if actual != expected:
        raise MarchReplayError(
            f"SHA-256 mismatch for frozen March input {logical_name!r}: "
            f"expected {expected}, got {actual}"
        )
    return payload


def _replay_hash_verified_bytes(frozen_bytes: Mapping[str, bytes]) -> MarchReplayResult:
    """Replay bytes whose hashes were already checked by the public entry point."""

    if set(frozen_bytes) != set(FROZEN_SHA256):
        missing = sorted(set(FROZEN_SHA256) - set(frozen_bytes))
        extra = sorted(set(frozen_bytes) - set(FROZEN_SHA256))
        raise MarchReplayError(
            f"Hash-verified input inventory mismatch: missing={missing}, extra={extra}"
        )

    config = _load_json_bytes(frozen_bytes["config.json"], "config.json")
    results = _load_json_bytes(frozen_bytes["results.json"], "results.json")
    corpus_rows = _load_jsonl_bytes(frozen_bytes["corpus.jsonl"], "corpus.jsonl")
    query_rows = _load_jsonl_bytes(
        frozen_bytes["queries/test.jsonl"], "queries/test.jsonl"
    )
    ranking_rows = _load_jsonl_bytes(
        frozen_bytes["runs/rankings.jsonl"], "runs/rankings.jsonl"
    )
    pools = _load_json_bytes(
        frozen_bytes["pools/candidates_by_case.json"],
        "pools/candidates_by_case.json",
    )
    split_doc_ids = _load_split_bytes(
        frozen_bytes["splits/test_cases.txt"], "splits/test_cases.txt"
    )

    system_specs = _validate_config(config, split_doc_ids=split_doc_ids)
    corpus_ids_by_doc = _validate_corpus(corpus_rows)
    pools_by_doc = _validate_pools(pools, corpus_ids_by_doc=corpus_ids_by_doc)
    queries = _validate_queries(
        query_rows,
        split_doc_ids=split_doc_ids,
        corpus_ids_by_doc=corpus_ids_by_doc,
    )
    if len(queries) != _EXPECTED_QUERY_COUNT:
        raise MarchReplayError(
            f"Expected exactly {_EXPECTED_QUERY_COUNT} March queries, got {len(queries)}"
        )

    expected_candidates = _build_expected_candidates(
        queries=queries,
        split_doc_ids=split_doc_ids,
        corpus_ids_by_doc=corpus_ids_by_doc,
        pools_by_doc=pools_by_doc,
    )
    row_metrics, stable_reordered_rows = _validate_and_score_rankings(
        ranking_rows=ranking_rows,
        queries=queries,
        system_specs=system_specs,
        expected_candidates=expected_candidates,
    )
    replayed_regimes = _build_replayed_regimes(
        queries=queries,
        row_metrics=row_metrics,
        system_specs=system_specs,
    )
    numeric_values = _validate_results(
        results=results,
        config=config,
        replayed_regimes=replayed_regimes,
        system_specs=system_specs,
    )

    if stable_reordered_rows != _EXPECTED_STABLE_REORDERED_ROWS:
        raise MarchReplayError(
            "Stable tie normalization changed the order of "
            f"{stable_reordered_rows} ranking rows; expected exactly "
            f"{_EXPECTED_STABLE_REORDERED_ROWS}"
        )
    if numeric_values != _EXPECTED_NUMERIC_VALUE_COUNT:
        raise MarchReplayError(
            f"Replayed {numeric_values} stored numeric result values; expected "
            f"{_EXPECTED_NUMERIC_VALUE_COUNT}"
        )

    return MarchReplayResult(
        namespace=RECONSTRUCTED_MARCH_NAMESPACE,
        sha256_by_input=FROZEN_SHA256,
        systems=tuple(name for name, _, _ in system_specs),
        legacy_regime_labels=_EXPECTED_REGIMES,
        semantic_regime_by_legacy_label=LEGACY_REGIME_SEMANTICS,
        query_count=len(queries),
        ranking_row_count=len(ranking_rows),
        result_cell_count=_EXPECTED_CELL_COUNT,
        numeric_values_verified=numeric_values,
        stable_tie_rows_reordered=stable_reordered_rows,
        stable_tie_metric_changes=0,
        replayed_regimes=_deep_freeze(replayed_regimes),
    )


def _load_json_bytes(payload: bytes, context: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MarchReplayError(f"{context} is not valid UTF-8: {exc}") from exc
    return _strict_json_loads(text, context)


def _load_jsonl_bytes(payload: bytes, context: str) -> List[Mapping[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MarchReplayError(f"{context} is not valid UTF-8: {exc}") from exc
    if not text.endswith("\n"):
        raise MarchReplayError(f"{context} must end with exactly a complete JSONL row")
    lines = text.splitlines()
    if any(not line for line in lines):
        raise MarchReplayError(f"{context} contains a blank JSONL row")
    rows: List[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        value = _strict_json_loads(line, f"{context}:{line_number}")
        if type(value) is not dict:
            raise MarchReplayError(
                f"{context}:{line_number} must contain one JSON object"
            )
        rows.append(value)
    return rows


def _strict_json_loads(text: str, context: str) -> Any:
    def reject_constant(value: str) -> None:
        raise MarchReplayError(f"{context} contains non-finite JSON number {value!r}")

    def unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise MarchReplayError(f"{context} contains duplicate JSON key {key!r}")
            obj[key] = value
        return obj

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except MarchReplayError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise MarchReplayError(f"Malformed JSON in {context}: {exc}") from exc


def _load_split_bytes(payload: bytes, context: str) -> Tuple[str, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MarchReplayError(f"{context} is not valid UTF-8: {exc}") from exc
    if not text.endswith("\n"):
        raise MarchReplayError(f"{context} must end in a newline")
    lines = text.splitlines()
    if not lines or any(not line or line.strip() != line for line in lines):
        raise MarchReplayError(f"{context} contains an empty or non-canonical case ID")
    if len(set(lines)) != len(lines):
        raise MarchReplayError(f"{context} contains a duplicate case ID")
    return tuple(lines)


def _validate_config(
    config: Any,
    *,
    split_doc_ids: Tuple[str, ...],
) -> Tuple[Tuple[str, str, str], ...]:
    _require_mapping(config, "config.json")
    _require_exact_keys(config, _CONFIG_KEYS, "config.json")
    if config["split"] != "test":
        raise MarchReplayError("config.json split must remain the legacy label 'test'")
    if tuple(config["k_values"]) != _EXPECTED_KS:
        raise MarchReplayError(
            f"config.json k_values must be exactly {list(_EXPECTED_KS)}"
        )
    if tuple(config["regimes"]) != _EXPECTED_REGIMES:
        raise MarchReplayError(
            f"config.json regimes must remain the three legacy labels {_EXPECTED_REGIMES}"
        )
    if tuple(config["split_doc_ids"]) != split_doc_ids:
        raise MarchReplayError(
            "config.json split_doc_ids do not exactly match splits/test_cases.txt"
        )
    if type(config["split_passage_count"]) is not int or config["split_passage_count"] != 581:
        raise MarchReplayError("config.json split_passage_count must be exactly 581")
    data_hashes = config["data_sha256"]
    _require_mapping(data_hashes, "config.json data_sha256")
    expected_data_hashes = {
        key: FROZEN_SHA256[key]
        for key in (
            "corpus.jsonl",
            "queries/test.jsonl",
            "pools/candidates_by_case.json",
            "splits/test_cases.txt",
        )
    }
    if data_hashes != expected_data_hashes:
        raise MarchReplayError(
            "config.json data_sha256 does not exactly match the frozen data hashes"
        )

    systems = config["systems"]
    if type(systems) is not list or len(systems) != len(_EXPECTED_SYSTEMS):
        raise MarchReplayError("config.json must contain exactly five legacy systems")
    observed_specs: List[Tuple[str, str, str]] = []
    for index, system in enumerate(systems):
        context = f"config.json systems[{index}]"
        _require_mapping(system, context)
        _require_exact_keys(system, _SYSTEM_CONFIG_KEYS, context)
        observed_specs.append(
            (system["name"], system["system_type"], system["query_view"])
        )
    if tuple(observed_specs) != _EXPECTED_SYSTEMS:
        raise MarchReplayError(
            f"Legacy system specifications changed: expected {_EXPECTED_SYSTEMS}, "
            f"got {tuple(observed_specs)}"
        )
    return tuple(observed_specs)


def _validate_corpus(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Tuple[str, ...]]:
    if len(rows) != 5286:
        raise MarchReplayError(f"Expected exactly 5,286 corpus rows, got {len(rows)}")
    ids_by_doc: Dict[str, List[str]] = {}
    seen_ids = set()
    orders_by_doc: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        context = f"corpus.jsonl:{index + 1}"
        _require_exact_keys(row, _CORPUS_KEYS, context)
        passage_id = _require_string(row["passage_id"], f"{context} passage_id")
        doc_id = _require_string(row["doc_id"], f"{context} doc_id")
        _require_string(row["label"], f"{context} label")
        _require_string(row["text"], f"{context} text", allow_empty=False)
        if passage_id in seen_ids:
            raise MarchReplayError(f"Duplicate corpus passage_id {passage_id!r}")
        if not passage_id.startswith(f"{doc_id}::"):
            raise MarchReplayError(
                f"{context} passage_id {passage_id!r} does not belong to doc_id {doc_id!r}"
            )
        seen_ids.add(passage_id)
        if type(row["start"]) is not int or type(row["end"]) is not int:
            raise MarchReplayError(f"{context} start/end must be exact integers")
        if row["start"] < 0 or row["end"] <= row["start"]:
            raise MarchReplayError(f"{context} has invalid source offsets")
        if row["source_node_id"] is not None and type(row["source_node_id"]) is not str:
            raise MarchReplayError(f"{context} source_node_id must be a string or null")
        if type(row["is_implicit"]) is not bool:
            raise MarchReplayError(f"{context} is_implicit must be a boolean")
        if type(row["order"]) is not int or row["order"] < 0:
            raise MarchReplayError(f"{context} order must be a non-negative integer")
        ids_by_doc.setdefault(doc_id, []).append(passage_id)
        orders_by_doc.setdefault(doc_id, []).append(row["order"])
    for doc_id, orders in orders_by_doc.items():
        if orders != list(range(len(orders))):
            raise MarchReplayError(
                f"Corpus order for case {doc_id!r} is not complete and sequential"
            )
    return {doc_id: tuple(passage_ids) for doc_id, passage_ids in ids_by_doc.items()}


def _validate_pools(
    pools: Any,
    *,
    corpus_ids_by_doc: Mapping[str, Tuple[str, ...]],
) -> Dict[str, Tuple[str, ...]]:
    _require_mapping(pools, "pools/candidates_by_case.json")
    if tuple(pools) != tuple(corpus_ids_by_doc):
        raise MarchReplayError("Candidate-pool case inventory/order does not match corpus")
    normalized: Dict[str, Tuple[str, ...]] = {}
    for doc_id, passage_ids in pools.items():
        if type(doc_id) is not str or type(passage_ids) is not list:
            raise MarchReplayError(
                "Candidate-pool entries must map exact string case IDs to JSON lists"
            )
        if any(type(passage_id) is not str for passage_id in passage_ids):
            raise MarchReplayError(f"Candidate pool for case {doc_id!r} has a non-string ID")
        if len(passage_ids) != len(set(passage_ids)):
            raise MarchReplayError(f"Candidate pool for case {doc_id!r} has duplicate IDs")
        if tuple(passage_ids) != corpus_ids_by_doc[doc_id]:
            raise MarchReplayError(
                f"Candidate pool for case {doc_id!r} does not exactly equal its corpus passages"
            )
        normalized[doc_id] = tuple(passage_ids)
    return normalized


def _validate_queries(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_doc_ids: Tuple[str, ...],
    corpus_ids_by_doc: Mapping[str, Tuple[str, ...]],
) -> Tuple[_Query, ...]:
    seen_ids = set()
    split_set = set(split_doc_ids)
    queries: List[_Query] = []
    for index, row in enumerate(rows):
        context = f"queries/test.jsonl:{index + 1}"
        _require_exact_keys(row, _QUERY_KEYS, context)
        for key in (
            "query_id",
            "doc_id",
            "motion_root_id",
            "mask_parent_id",
            "query_text",
            "flat_query_text_plain",
            "flat_query_text_masked",
        ):
            _require_string(row[key], f"{context} {key}", allow_empty=False)
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        if query_id in seen_ids:
            raise MarchReplayError(f"Duplicate legacy query_id {query_id!r}")
        if doc_id not in split_set or not query_id.startswith(f"{doc_id}::"):
            raise MarchReplayError(
                f"{context} query/doc identity is outside the frozen test split"
            )
        seen_ids.add(query_id)
        positive_ids = _require_string_list(
            row["positive_passage_ids"],
            f"{context} positive_passage_ids",
            nonempty=True,
            unique=True,
        )
        positive_labels = _require_string_list(
            row["positive_labels"],
            f"{context} positive_labels",
            nonempty=True,
            unique=False,
        )
        case_passages = set(corpus_ids_by_doc[doc_id])
        if not set(positive_ids).issubset(case_passages):
            raise MarchReplayError(f"{context} contains gold IDs outside query case {doc_id!r}")
        queries.append(
            _Query(
                query_id=query_id,
                doc_id=doc_id,
                query_text=row["query_text"],
                flat_query_text_plain=row["flat_query_text_plain"],
                flat_query_text_masked=row["flat_query_text_masked"],
                positive_passage_ids=positive_ids,
                positive_labels=positive_labels,
            )
        )
    return tuple(queries)


def _build_expected_candidates(
    *,
    queries: Sequence[_Query],
    split_doc_ids: Sequence[str],
    corpus_ids_by_doc: Mapping[str, Tuple[str, ...]],
    pools_by_doc: Mapping[str, Tuple[str, ...]],
) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    global_ids = tuple(
        passage_id
        for doc_id in corpus_ids_by_doc
        if doc_id in set(split_doc_ids)
        for passage_id in corpus_ids_by_doc[doc_id]
    )
    if len(global_ids) != 581:
        raise MarchReplayError(
            f"Frozen test role must contain exactly 581 passages, got {len(global_ids)}"
        )
    all_gold_by_doc: Dict[str, set[str]] = {}
    for query in queries:
        all_gold_by_doc.setdefault(query.doc_id, set()).update(query.positive_passage_ids)

    expected: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for query in queries:
        full = pools_by_doc[query.doc_id]
        current_gold = set(query.positive_passage_ids)
        excluded = all_gold_by_doc[query.doc_id] - current_gold
        expected[("same_case_legacy", query.query_id)] = tuple(
            passage_id for passage_id in full if passage_id not in excluded
        )
        expected[("same_case_full", query.query_id)] = full
        expected[("global_split", query.query_id)] = global_ids
    return expected


def _validate_and_score_rankings(
    *,
    ranking_rows: Sequence[Mapping[str, Any]],
    queries: Sequence[_Query],
    system_specs: Sequence[Tuple[str, str, str]],
    expected_candidates: Mapping[Tuple[str, str], Tuple[str, ...]],
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, float]]], int]:
    if len(ranking_rows) != _EXPECTED_ROW_COUNT:
        raise MarchReplayError(
            f"Expected exactly {_EXPECTED_ROW_COUNT} ranking rows "
            f"(5 systems x 3 regimes x 40 queries), got {len(ranking_rows)}"
        )
    expected_order = [
        (system_name, system_type, query_view, regime, query)
        for system_name, system_type, query_view in system_specs
        for regime in _EXPECTED_REGIMES
        for query in queries
    ]
    if len(expected_order) != _EXPECTED_ROW_COUNT:
        raise MarchReplayError("Internal March row inventory is not exactly 600")

    metrics_by_cell: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    seen_identities = set()
    stable_reordered_rows = 0
    for index, (row, expected_identity) in enumerate(zip(ranking_rows, expected_order)):
        system_name, system_type, query_view, regime, query = expected_identity
        context = f"runs/rankings.jsonl:{index + 1}"
        _validate_ranking_record_shape(row, context)
        actual_identity = (
            row["system"],
            row["system_type"],
            row["query_view"],
            row["regime"],
            row["query_id"],
        )
        expected_row_identity = (
            system_name,
            system_type,
            query_view,
            regime,
            query.query_id,
        )
        if actual_identity != expected_row_identity:
            raise MarchReplayError(
                f"{context} identity/order mismatch: expected {expected_row_identity}, "
                f"got {actual_identity}"
            )
        if actual_identity in seen_identities:
            raise MarchReplayError(f"{context} duplicates ranking identity {actual_identity}")
        seen_identities.add(actual_identity)
        if row["doc_id"] != query.doc_id:
            raise MarchReplayError(f"{context} doc_id does not match the frozen query")
        expected_query_text = {
            "structured": query.query_text,
            "flat_plain": query.flat_query_text_plain,
            "flat_masked": query.flat_query_text_masked,
        }[query_view]
        if row["query_text"] != expected_query_text:
            raise MarchReplayError(
                f"{context} query_text does not match query_view={query_view!r}"
            )
        if tuple(row["gold_passage_ids"]) != query.positive_passage_ids:
            raise MarchReplayError(f"{context} gold_passage_ids changed")

        ranked = row["ranked_candidates"]
        ranked_ids = tuple(candidate["passage_id"] for candidate in ranked)
        expected_ids = expected_candidates[(regime, query.query_id)]
        if row["candidate_pool_size"] != len(expected_ids):
            raise MarchReplayError(
                f"{context} candidate_pool_size={row['candidate_pool_size']} does not "
                f"match exact {regime} pool size {len(expected_ids)}"
            )
        if len(ranked_ids) != len(expected_ids) or set(ranked_ids) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(ranked_ids))
            extra = sorted(set(ranked_ids) - set(expected_ids))
            raise MarchReplayError(
                f"{context} is not the complete exact {regime} candidate set: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if not set(query.positive_passage_ids).issubset(ranked_ids):
            raise MarchReplayError(f"{context} does not contain every gold passage")

        original_metrics = _compute_query_metrics(
            retrieved_passage_ids=ranked_ids,
            gold_passage_ids=query.positive_passage_ids,
        )
        original_metrics["candidate_pool_size"] = float(len(expected_ids))
        metrics_by_cell.setdefault((regime, system_name), []).append(original_metrics)

        stable_ranked = sorted(
            ranked,
            key=lambda candidate: (-candidate["score"], candidate["passage_id"]),
        )
        stable_ids = tuple(candidate["passage_id"] for candidate in stable_ranked)
        if stable_ids != ranked_ids:
            stable_reordered_rows += 1
        stable_metrics = _compute_query_metrics(
            retrieved_passage_ids=stable_ids,
            gold_passage_ids=query.positive_passage_ids,
        )
        for metric_name in _METRIC_KEYS[:-1]:
            if stable_metrics[metric_name] != original_metrics[metric_name]:
                raise MarchReplayError(
                    f"{context} stable tie normalization changes historical metric "
                    f"{metric_name}: {original_metrics[metric_name]} -> "
                    f"{stable_metrics[metric_name]}"
                )
        original_rank = original_metrics["first_hit_rank"]
        stable_rank = stable_metrics["first_hit_rank"]
        original_full_rr = 1.0 / original_rank
        stable_full_rr = 1.0 / stable_rank
        if original_full_rr != stable_full_rr:
            raise MarchReplayError(
                f"{context} stable tie normalization changes full first-gold RR"
            )

    if len(seen_identities) != _EXPECTED_ROW_COUNT:
        raise MarchReplayError("Ranking identities are not exactly the complete 600-row grid")
    return metrics_by_cell, stable_reordered_rows


def _validate_ranking_record_shape(row: Mapping[str, Any], context: str) -> None:
    _require_mapping(row, context)
    _require_exact_keys(row, _RANKING_KEYS, context)
    for key in (
        "system",
        "system_type",
        "query_view",
        "regime",
        "query_id",
        "doc_id",
        "query_text",
    ):
        _require_string(row[key], f"{context} {key}", allow_empty=False)
    if type(row["candidate_pool_size"]) is not int or row["candidate_pool_size"] <= 0:
        raise MarchReplayError(f"{context} candidate_pool_size must be a positive integer")
    _require_string_list(
        row["gold_passage_ids"],
        f"{context} gold_passage_ids",
        nonempty=True,
        unique=True,
    )
    ranked = row["ranked_candidates"]
    if type(ranked) is not list or len(ranked) != row["candidate_pool_size"]:
        raise MarchReplayError(
            f"{context} ranked_candidates must be a complete list of candidate_pool_size rows"
        )
    seen_ids = set()
    previous_score = math.inf
    for index, candidate in enumerate(ranked, start=1):
        candidate_context = f"{context} ranked_candidates[{index - 1}]"
        _require_mapping(candidate, candidate_context)
        _require_exact_keys(candidate, _RANKED_CANDIDATE_KEYS, candidate_context)
        if type(candidate["rank"]) is not int or candidate["rank"] != index:
            raise MarchReplayError(
                f"{candidate_context} rank must be the sequential integer {index}"
            )
        passage_id = _require_string(
            candidate["passage_id"], f"{candidate_context} passage_id", allow_empty=False
        )
        if passage_id in seen_ids:
            raise MarchReplayError(f"{context} has duplicate ranked passage_id {passage_id!r}")
        seen_ids.add(passage_id)
        score = candidate["score"]
        if type(score) is not float or not math.isfinite(score):
            raise MarchReplayError(
                f"{candidate_context} score must be an exact finite JSON float"
            )
        if score > previous_score:
            raise MarchReplayError(f"{context} scores are not in descending order")
        previous_score = score


def _build_replayed_regimes(
    *,
    queries: Sequence[_Query],
    row_metrics: Mapping[Tuple[str, str], Sequence[Dict[str, float]]],
    system_specs: Sequence[Tuple[str, str, str]],
) -> Dict[str, Any]:
    replayed: Dict[str, Any] = {}
    for regime in _EXPECTED_REGIMES:
        systems: Dict[str, Any] = {}
        for system_name, _, _ in system_specs:
            metrics = list(row_metrics[(regime, system_name)])
            if len(metrics) != len(queries):
                raise MarchReplayError(
                    f"Historical cell {regime}/{system_name} does not contain 40 queries"
                )
            global_metrics = _aggregate_metrics_sequential(metrics)
            global_metrics["num_queries"] = float(len(metrics))
            systems[system_name] = {
                "global": global_metrics,
                "breakdowns": _compute_breakdowns(queries=queries, metrics=metrics),
            }
        replayed[regime] = {"systems": systems}
    return replayed


def _compute_query_metrics(
    *,
    retrieved_passage_ids: Sequence[str],
    gold_passage_ids: Sequence[str],
) -> Dict[str, float]:
    gold_set = set(gold_passage_ids)
    if not gold_set:
        raise MarchReplayError("A reconstructed March query has no gold passages")
    hit_rank = 0
    for index, passage_id in enumerate(retrieved_passage_ids, start=1):
        if passage_id in gold_set:
            hit_rank = index
            break
    if hit_rank == 0:
        raise MarchReplayError("A complete reconstructed March ranking contains no gold")

    metrics: Dict[str, float] = {}
    num_gold = float(len(gold_set))
    for k in _EXPECTED_KS:
        topk_set = set(retrieved_passage_ids[:k])
        any_hit = hit_rank <= k
        metrics[f"recall_at_{k}"] = 1.0 if any_hit else 0.0
        metrics[f"mrr_at_{k}"] = 1.0 / float(hit_rank) if any_hit else 0.0
        metrics[f"set_recall_at_{k}"] = float(len(topk_set & gold_set)) / num_gold
        metrics[f"exact_set_match_at_{k}"] = (
            1.0 if gold_set.issubset(topk_set) else 0.0
        )
    metrics["num_gold"] = num_gold
    metrics["first_hit_rank"] = float(hit_rank)
    return metrics


def _aggregate_metrics_sequential(
    rows: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    if not rows:
        raise MarchReplayError("Cannot aggregate an empty historical metric group")
    sums: Dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {key: value / float(len(rows)) for key, value in sums.items()}


def _compute_breakdowns(
    *,
    queries: Sequence[_Query],
    metrics: Sequence[Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if len(queries) != len(metrics):
        raise MarchReplayError("Breakdown queries and metrics are not aligned")
    groups: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "by_missing_type": {},
        "by_gold_label_composition": {},
        "by_num_positives": {},
        "by_has_implicit": {},
    }
    for query, query_metrics in zip(queries, metrics):
        buckets = {
            "by_missing_type": (
                "missing_conclusion"
                if "MISSING=CONCLUSION" in query.query_id
                else "missing_premise_group"
            ),
            "by_gold_label_composition": _gold_label_bucket(query.positive_labels),
            "by_num_positives": _positive_count_bucket(
                len(set(query.positive_passage_ids))
            ),
            "by_has_implicit": (
                "has_implicit" if "[IMPLICIT]" in query.query_text else "no_implicit"
            ),
        }
        for group_name, bucket_name in buckets.items():
            groups[group_name].setdefault(bucket_name, []).append(query_metrics)

    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_name, buckets in groups.items():
        output[group_name] = {}
        for bucket_name, rows in buckets.items():
            aggregate = _aggregate_metrics_sequential(rows)
            aggregate["num_queries"] = float(len(rows))
            output[group_name][bucket_name] = aggregate
    return output


def _gold_label_bucket(labels: Sequence[str]) -> str:
    label_set = {str(label).strip() for label in labels if str(label).strip()}
    if not label_set:
        return "unknown"
    if len(label_set) == 1:
        only = next(iter(label_set)).lower()
        if only in {"rule", "analysis", "conclusion"}:
            return f"only_{only}"
        return f"only_{only}"
    return "mixed"


def _positive_count_bucket(num_gold: int) -> str:
    if num_gold <= 1:
        return "1"
    if num_gold <= 3:
        return "2_3"
    return "4_plus"


def _validate_results(
    *,
    results: Any,
    config: Mapping[str, Any],
    replayed_regimes: Mapping[str, Any],
    system_specs: Sequence[Tuple[str, str, str]],
) -> int:
    _require_mapping(results, "results.json")
    _require_exact_keys(results, ("config", "systems", "regimes"), "results.json")
    if results["config"] != config:
        raise MarchReplayError("results.json config is not exactly config.json")
    result_systems = results["systems"]
    _require_mapping(result_systems, "results.json systems")
    expected_system_names = tuple(name for name, _, _ in system_specs)
    if tuple(result_systems) != expected_system_names:
        raise MarchReplayError("results.json system inventory/order changed")
    for name, system_type, query_view in system_specs:
        system = result_systems[name]
        _require_mapping(system, f"results.json systems/{name}")
        _require_exact_keys(
            system,
            ("system_type", "query_view", "metadata"),
            f"results.json systems/{name}",
        )
        if (system["system_type"], system["query_view"]) != (system_type, query_view):
            raise MarchReplayError(f"results.json specification changed for system {name!r}")
        _require_mapping(system["metadata"], f"results.json systems/{name}/metadata")

    stored_regimes = results["regimes"]
    _require_mapping(stored_regimes, "results.json regimes")
    if tuple(stored_regimes) != _EXPECTED_REGIMES:
        raise MarchReplayError(
            "results.json must retain global_split as the historical output label"
        )

    numeric_values = 0
    cells = 0
    for regime in _EXPECTED_REGIMES:
        regime_payload = stored_regimes[regime]
        _require_mapping(regime_payload, f"results.json regimes/{regime}")
        _require_exact_keys(
            regime_payload,
            ("systems",),
            f"results.json regimes/{regime}",
        )
        stored_systems = regime_payload["systems"]
        _require_mapping(stored_systems, f"results.json regimes/{regime}/systems")
        if tuple(stored_systems) != expected_system_names:
            raise MarchReplayError(
                f"results.json system inventory/order changed for regime {regime!r}"
            )
        for system_name in expected_system_names:
            stored_cell = stored_systems[system_name]
            replayed_cell = replayed_regimes[regime]["systems"][system_name]
            _validate_result_cell_schema(
                stored_cell,
                f"results.json regimes/{regime}/systems/{system_name}",
            )
            numeric_values += _compare_replayed_value(
                replayed_cell,
                stored_cell,
                f"results.json regimes/{regime}/systems/{system_name}",
            )
            cells += 1
    if cells != _EXPECTED_CELL_COUNT:
        raise MarchReplayError(f"Replayed {cells} result cells, expected 15")
    return numeric_values


def _validate_result_cell_schema(cell: Any, context: str) -> None:
    _require_mapping(cell, context)
    _require_exact_keys(cell, ("global", "breakdowns"), context)
    global_metrics = cell["global"]
    _require_mapping(global_metrics, f"{context}/global")
    _require_exact_keys(global_metrics, _GLOBAL_METRIC_KEYS, f"{context}/global")
    breakdowns = cell["breakdowns"]
    _require_mapping(breakdowns, f"{context}/breakdowns")
    _require_exact_keys(breakdowns, _BREAKDOWN_NAMES, f"{context}/breakdowns")
    for group_name, buckets in breakdowns.items():
        _require_mapping(buckets, f"{context}/breakdowns/{group_name}")
        if not buckets:
            raise MarchReplayError(f"{context}/breakdowns/{group_name} is empty")
        for bucket_name, metrics in buckets.items():
            _require_string(bucket_name, f"{context} bucket name", allow_empty=False)
            _require_mapping(metrics, f"{context}/breakdowns/{group_name}/{bucket_name}")
            _require_exact_keys(
                metrics,
                _GLOBAL_METRIC_KEYS,
                f"{context}/breakdowns/{group_name}/{bucket_name}",
            )


def _compare_replayed_value(actual: Any, expected: Any, context: str) -> int:
    if type(expected) is dict:
        if type(actual) is not dict or tuple(actual) != tuple(expected):
            raise MarchReplayError(f"{context} key inventory/order does not replay exactly")
        return sum(
            _compare_replayed_value(actual[key], expected[key], f"{context}/{key}")
            for key in expected
        )
    if type(expected) is float:
        if type(actual) is not float or not math.isfinite(expected) or actual != expected:
            raise MarchReplayError(
                f"{context} does not replay exactly: stored={expected!r}, replayed={actual!r}"
            )
        return 1
    raise MarchReplayError(
        f"{context} has unexpected non-float stored result value {expected!r}"
    )


def _require_mapping(value: Any, context: str) -> None:
    if type(value) is not dict:
        raise MarchReplayError(f"{context} must be an exact JSON object")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected_keys: Sequence[str],
    context: str,
) -> None:
    actual = tuple(value)
    expected = tuple(expected_keys)
    if actual != expected:
        raise MarchReplayError(
            f"{context} schema keys/order changed: expected={expected}, got={actual}"
        )


def _require_string(value: Any, context: str, *, allow_empty: bool = True) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise MarchReplayError(f"{context} must be an exact{' non-empty' if not allow_empty else ''} string")
    return value


def _require_string_list(
    value: Any,
    context: str,
    *,
    nonempty: bool,
    unique: bool,
) -> Tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise MarchReplayError(f"{context} must be a JSON list of exact strings")
    if nonempty and not value:
        raise MarchReplayError(f"{context} must not be empty")
    if unique and len(value) != len(set(value)):
        raise MarchReplayError(f"{context} must not contain duplicates")
    return tuple(value)


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    return value
