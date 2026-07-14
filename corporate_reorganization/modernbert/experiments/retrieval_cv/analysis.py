"""Strict case-first statistical analysis for the retrieval-CV study.

The scientific unit is the held-out case.  This module streams the canonical
complete rankings, independently recomputes every query metric, aggregates
queries within case and matched seeds within case, and evaluates only the five
contrasts frozen in ``configs/experiment.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import aggregate


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_PROTOCOL = "arr_retrieval_case_first_analysis_v1"
BOOTSTRAP_PROTOCOL = "python_random_mt19937_paired_percentile_v1"
PERCENTILE_PROTOCOL = "linear_order_statistic_n_minus_one_v1"
ANALYSIS_SEED = 17
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
SEEDS = (17, 29, 43)
QUERY_VIEWS = ("flat_masked", "structured")
SAMPLERS = ("global_uniform", "local_unique")
PRIMARY_REGIME = "fold_global"
CONTEXT_EXCLUDED_REGIME = "fold_global_context_excluded"
PRIMARY_METRIC = "hit_at_20"
KS = (1, 5, 10, 20)
METRIC_NAMES = tuple(
    [f"hit_at_{k}" for k in KS]
    + [f"set_recall_at_{k}" for k in KS]
    + [f"exact_target_recovery_at_{k}" for k in KS]
    + ["first_gold_reciprocal_rank_full_ranking", "candidate_count"]
)
EXPECTED_ACQUIRED_FILES = (
    "evaluation/artifact_manifest.json",
    "evaluation/evaluation_config.json",
    "evaluation/rankings.jsonl",
    "evaluation/results.json",
    "evidence/artifact_manifest.json",
    "evidence/materialization_receipt.json",
)
EXPECTED_DATASET_INPUT_FILES = (
    "cases.jsonl",
    "corpus.jsonl",
    "pools/candidates_by_case.json",
    "pools/candidates_global.json",
    "queries/all.jsonl",
)
_CONTROLLED_SYSTEM = re.compile(
    r"(?P<query_view>flat_masked|structured)_"
    r"(?P<sampler>global_uniform|local_unique)_seed(?P<seed>17|29|43)"
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_TERMINAL_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "job_name",
    "job_arn",
    "status",
    "failure_reason",
    "exit_message",
    "processing_start_time",
    "processing_end_time",
    "processing_time_microseconds",
    "request_sha256",
    "preflight_receipt_sha256",
    "submission_receipt_sha256",
    "receipt_sha256",
}
_ACQUISITION_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "output_prefix",
    "terminal_receipt_sha256",
    "control_bundle_receipt_sha256",
    "evaluation_artifact_manifest_sha256",
    "materialization_artifact_manifest_sha256",
    "remote_objects",
    "files",
    "receipt_sha256",
}
_REMOTE_OBJECT_KEYS = {
    "bucket",
    "key",
    "version_id",
    "etag",
    "size",
    "sha256",
    "encryption",
}
_RANKING_ROW_KEYS = {
    "schema_version",
    "system_id",
    "system_type",
    "query_view",
    "regime_name",
    "query_index",
    "query_id",
    "per_query",
    "source_ranking",
    "ranking",
}
_PER_QUERY_KEYS = {
    "query_id",
    "doc_id",
    "gold_passage_ids",
    "visible_passage_ids",
    "gold_count",
    "first_gold_rank",
    *METRIC_NAMES,
}
_RANKING_KEYS = {
    "query_id",
    "doc_id",
    "candidate_count",
    "ranking_sha256",
    "parent_ranking_sha256",
    "ranked_candidates",
}
_CANDIDATE_KEYS = {"rank", "passage_id", "score"}


@dataclass(frozen=True)
class ContrastDefinition:
    contrast_id: str
    label: str
    weights: tuple[tuple[str, str, int], ...]


CONTRASTS = (
    ContrastDefinition(
        "flat_global_minus_local",
        "Flat: global-uniform minus local-unique",
        (("flat_masked", "global_uniform", 1), ("flat_masked", "local_unique", -1)),
    ),
    ContrastDefinition(
        "structured_global_minus_local",
        "Structured: global-uniform minus local-unique",
        (("structured", "global_uniform", 1), ("structured", "local_unique", -1)),
    ),
    ContrastDefinition(
        "structured_minus_flat_local",
        "Local-unique: structured minus flat",
        (("structured", "local_unique", 1), ("flat_masked", "local_unique", -1)),
    ),
    ContrastDefinition(
        "structured_minus_flat_global",
        "Global-uniform: structured minus flat",
        (("structured", "global_uniform", 1), ("flat_masked", "global_uniform", -1)),
    ),
    ContrastDefinition(
        "difference_in_structural_effects",
        "Difference in structural effects (global minus local)",
        (
            ("structured", "global_uniform", 1),
            ("flat_masked", "global_uniform", -1),
            ("structured", "local_unique", -1),
            ("flat_masked", "local_unique", 1),
        ),
    ),
)


@dataclass(frozen=True)
class AnalysisBundle:
    experiment_config: Mapping[str, Any]
    fold_manifest: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    experiment_config_bytes: bytes
    fold_manifest_bytes: bytes
    dataset_manifest_bytes: bytes
    dataset_input_files: tuple[tuple[str, bytes], ...]
    terminal_receipts: tuple[Mapping[str, Any], ...]
    acquisition_receipts: tuple[Mapping[str, Any], ...]
    evaluation_index: Mapping[str, Any]
    jobs: tuple[Mapping[str, Any], ...]
    rankings: tuple[Mapping[str, Any], ...]
    fold_load: tuple[Mapping[str, Any], ...]
    query_metrics: tuple[Mapping[str, Any], ...]
    case_metrics: tuple[Mapping[str, Any], ...]
    system_summary: tuple[Mapping[str, Any], ...]
    cell_case_metrics: tuple[Mapping[str, Any], ...]
    cell_summary: tuple[Mapping[str, Any], ...]
    seed_summary: tuple[Mapping[str, Any], ...]
    contrasts: tuple[Mapping[str, Any], ...]
    per_case_primary: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required analysis input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Analysis input is not valid JSON: {path}") from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError(f"Analysis input is not one canonical JSON object: {path}")
    return value


def _load_exact_hashed_object(path: Path) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required frozen input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Frozen input is not valid JSON: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"Frozen input must contain one JSON object: {path}")
    return value, raw


def _load_dataset_input_files(
    dataset_dir: Path,
    dataset_manifest: Mapping[str, Any],
) -> tuple[tuple[str, bytes], ...]:
    """Load the complete corrected dataset exactly as bound by its manifest."""

    dataset_dir = Path(dataset_dir)
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError(f"Corrected dataset must be one real directory: {dataset_dir}")
    expected_inventory = {"dataset_manifest.json", *EXPECTED_DATASET_INPUT_FILES}
    observed_inventory: set[str] = set()
    for path in dataset_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Corrected dataset contains a symlink: {path}")
        if path.is_file():
            observed_inventory.add(path.relative_to(dataset_dir).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Corrected dataset contains a non-file entry: {path}")
    if observed_inventory != expected_inventory:
        raise ValueError("Corrected dataset file inventory changed")

    records = dataset_manifest.get("output_files")
    if type(records) is not dict or set(records) != set(EXPECTED_DATASET_INPUT_FILES):
        raise ValueError("Corrected dataset manifest output inventory changed")
    loaded: list[tuple[str, bytes]] = []
    for relative_name in EXPECTED_DATASET_INPUT_FILES:
        record = records[relative_name]
        if (
            type(record) is not dict
            or set(record) != {"bytes", "records", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 1
            or type(record["records"]) is not int
            or record["records"] < 1
            or type(record["sha256"]) is not str
            or _LOWER_SHA256.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Corrected dataset manifest record changed: {relative_name}")
        path = dataset_dir / Path(relative_name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Corrected dataset input is missing or unsafe: {relative_name}")
        payload = path.read_bytes()
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Corrected dataset input identity changed: {relative_name}")
        loaded.append((relative_name, payload))
    return tuple(loaded)


def _load_canonical_receipt(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required analysis receipt is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Analysis receipt is not valid JSON: {path}") from error
    if type(value) is not dict or raw not in {
        _canonical_bytes(value),
        _pretty_canonical_bytes(value),
    }:
        raise ValueError(f"Analysis receipt is not in an accepted canonical encoding: {path}")
    return value


def _document_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    actual = _require_sha256(value.get("receipt_sha256"), name=f"{name}.receipt_sha256")
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if actual != _document_sha256(payload):
        raise ValueError(f"{name} self-hash changed")


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_strings(values: object, *, name: str, sorted_values: bool = False) -> tuple[str, ...]:
    if type(values) is not list or not values:
        raise ValueError(f"{name} must be one non-empty list")
    result: list[str] = []
    for position, value in enumerate(values):
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError(f"{name}[{position}] must be one exact non-empty string")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates")
    if sorted_values and result != sorted(result):
        raise ValueError(f"{name} must be lexicographically sorted")
    return tuple(result)


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"Cannot average empty values for {name}")
    result = math.fsum(float(value) for value in values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite mean for {name}: {result}")
    return result


def _controlled_identity(system_id: str) -> tuple[str | None, str | None, int | None]:
    match = _CONTROLLED_SYSTEM.fullmatch(system_id)
    if match is None:
        return None, None, None
    return match["query_view"], match["sampler"], int(match["seed"])


def _validate_acquisition_chain(
    acquisition_dirs: Sequence[Path],
    terminal_receipts: Sequence[Path],
) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if len(acquisition_dirs) != 5 or len(terminal_receipts) != 5:
        raise ValueError("Analysis requires exactly five acquisition directories and terminal receipts")

    terminal_by_fold: dict[int, dict[str, Any]] = {}
    terminal_path_by_fold: dict[int, Path] = {}
    for raw_path in terminal_receipts:
        path = Path(raw_path)
        value = _load_canonical_receipt(path)
        fold = value.get("outer_fold")
        if type(fold) is not int or fold not in range(5) or fold in terminal_by_fold:
            raise ValueError("Terminal receipts must cover outer folds 0..4 exactly once")
        if (
            set(value) != _TERMINAL_KEYS
            or value.get("schema_version") != 1
            or value.get("protocol") != "retrieval_cv_fold_evaluation_terminal_v1"
            or value.get("status") != "Completed"
            or value.get("failure_reason") is not None
            or value.get("exit_message") is not None
            or type(value.get("processing_time_microseconds")) is not int
            or value["processing_time_microseconds"] <= 0
            or type(value.get("processing_start_time")) is not str
            or type(value.get("processing_end_time")) is not str
            or type(value.get("job_name")) is not str
            or re.fullmatch(rf"arr-ret-cv1-f{fold}-evaluate-a3-r[1-9][0-9]*", value["job_name"])
            is None
            or type(value.get("job_arn")) is not str
            or not value["job_arn"].endswith(f"/{value['job_name']}")
        ):
            raise ValueError(f"Fold {fold} terminal receipt is not one clean completion")
        for key in (
            "request_sha256",
            "preflight_receipt_sha256",
            "submission_receipt_sha256",
            "receipt_sha256",
        ):
            _require_sha256(value.get(key), name=f"terminal[{fold}].{key}")
        _validate_self_hash(value, name=f"terminal[{fold}]")
        terminal_by_fold[fold] = value
        terminal_path_by_fold[fold] = path

    acquisition_by_fold: dict[int, dict[str, Any]] = {}
    acquisition_dir_by_fold: dict[int, Path] = {}
    for raw_dir in acquisition_dirs:
        directory = Path(raw_dir)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Acquisition directory must be a real directory: {directory}")
        receipt = _load_canonical_receipt(directory / "acquisition_receipt.json")
        fold = receipt.get("outer_fold")
        if type(fold) is not int or fold not in range(5) or fold in acquisition_by_fold:
            raise ValueError("Acquisition receipts must cover outer folds 0..4 exactly once")
        if (
            set(receipt) != _ACQUISITION_KEYS
            or receipt.get("schema_version") != 1
            or receipt.get("protocol") != "retrieval_cv_fold_evaluation_acquisition_v1"
            or type(receipt.get("output_prefix")) is not str
            or not receipt["output_prefix"].startswith(
                f"arr-retrieval-cv/evaluation-a3/fold-{fold}/"
            )
        ):
            raise ValueError(f"Fold {fold} acquisition receipt schema changed")
        for key in (
            "terminal_receipt_sha256",
            "control_bundle_receipt_sha256",
            "evaluation_artifact_manifest_sha256",
            "materialization_artifact_manifest_sha256",
            "receipt_sha256",
        ):
            _require_sha256(receipt.get(key), name=f"acquisition[{fold}].{key}")
        _validate_self_hash(receipt, name=f"acquisition[{fold}]")
        terminal = terminal_by_fold[fold]
        if receipt.get("terminal_receipt_sha256") != _document_sha256(terminal):
            raise ValueError(f"Fold {fold} acquisition is not bound to its terminal receipt")
        file_records = receipt.get("files")
        if type(file_records) is not list or tuple(
            record.get("path") if type(record) is dict else None for record in file_records
        ) != EXPECTED_ACQUIRED_FILES:
            raise ValueError(f"Fold {fold} acquisition file inventory changed")
        for position, record in enumerate(file_records):
            if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
                raise ValueError(f"Fold {fold} acquisition file[{position}] schema changed")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Fold {fold} acquisition contains an unsafe relative path")
            path = directory / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Fold {fold} acquired file is missing or unsafe: {relative}")
            if (
                type(record["size"]) is not int
                or record["size"] < 1
                or path.stat().st_size != record["size"]
                or _require_sha256(record["sha256"], name=f"acquisition[{fold}].sha256")
                != _sha256_file(path)
            ):
                raise ValueError(f"Fold {fold} acquired file identity changed: {relative}")
        remote_objects = receipt.get("remote_objects")
        if type(remote_objects) is not list or len(remote_objects) != len(file_records):
            raise ValueError(f"Fold {fold} remote acquisition inventory changed")
        for file_record, remote in zip(file_records, remote_objects):
            if (
                type(remote) is not dict
                or set(remote) != _REMOTE_OBJECT_KEYS
                or type(remote.get("bucket")) is not str
                or not remote["bucket"]
                or type(remote.get("key")) is not str
                or remote["key"]
                != f"{receipt['output_prefix']}{file_record['path']}"
                or type(remote.get("version_id")) is not str
                or not remote["version_id"]
                or type(remote.get("etag")) is not str
                or not remote["etag"]
                or remote.get("size") != file_record["size"]
                or remote.get("sha256") != file_record["sha256"]
                or type(remote.get("encryption")) is not dict
            ):
                raise ValueError(f"Fold {fold} remote acquisition object changed")
        acquisition_by_fold[fold] = receipt
        acquisition_dir_by_fold[fold] = directory

    if set(terminal_by_fold) != set(range(5)) or set(acquisition_by_fold) != set(range(5)):
        raise ValueError("Analysis provenance does not cover all five folds")
    ordered_dirs = tuple(acquisition_dir_by_fold[fold] for fold in range(5))
    ordered_terminals = tuple(terminal_by_fold[fold] for fold in range(5))
    ordered_acquisitions = tuple(acquisition_by_fold[fold] for fold in range(5))
    for fold, directory in enumerate(ordered_dirs):
        evaluation_dir = directory / "evaluation"
        if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
            raise ValueError(f"Fold {fold} evaluation directory is missing or unsafe")
        if terminal_path_by_fold[fold].is_symlink():
            raise ValueError(f"Fold {fold} terminal receipt path became a symlink")
    return ordered_dirs, ordered_terminals, ordered_acquisitions


def _recompute_query_metrics(per_query: Mapping[str, Any], ranking: Mapping[str, Any]) -> dict[str, Any]:
    if type(per_query) is not dict or set(per_query) != _PER_QUERY_KEYS:
        raise ValueError("Canonical per-query metric schema changed")
    if type(ranking) is not dict or set(ranking) != _RANKING_KEYS:
        raise ValueError("Canonical ranking schema changed")
    if ranking["query_id"] != per_query["query_id"] or ranking["doc_id"] != per_query["doc_id"]:
        raise ValueError("Per-query and ranking identities disagree")
    gold_ids = _exact_strings(
        per_query["gold_passage_ids"],
        name=f"query[{per_query['query_id']}].gold_passage_ids",
        sorted_values=True,
    )
    _exact_strings(
        per_query["visible_passage_ids"],
        name=f"query[{per_query['query_id']}].visible_passage_ids",
        sorted_values=True,
    ) if per_query["visible_passage_ids"] else ()
    candidates = ranking["ranked_candidates"]
    if type(candidates) is not list or not candidates:
        raise ValueError("Canonical ranking must contain candidates")
    passage_ids: list[str] = []
    for expected_rank, candidate in enumerate(candidates, start=1):
        if type(candidate) is not dict or set(candidate) != _CANDIDATE_KEYS:
            raise ValueError("Canonical ranked-candidate schema changed")
        if candidate["rank"] != expected_rank:
            raise ValueError("Canonical candidate ranks are not complete and contiguous")
        passage_id = candidate["passage_id"]
        score = candidate["score"]
        if type(passage_id) is not str or not passage_id or passage_id.strip() != passage_id:
            raise ValueError("Canonical ranking contains an invalid passage ID")
        if type(score) not in {int, float} or not math.isfinite(float(score)):
            raise ValueError("Canonical ranking contains a non-finite score")
        passage_ids.append(passage_id)
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Canonical ranking contains duplicate passages")
    if ranking["candidate_count"] != len(passage_ids):
        raise ValueError("Canonical ranking candidate_count changed")
    gold_set = set(gold_ids)
    if not gold_set.issubset(passage_ids):
        raise ValueError("Canonical ranking dropped at least one gold passage")
    first_gold_rank = next(
        rank for rank, passage_id in enumerate(passage_ids, start=1) if passage_id in gold_set
    )
    computed: dict[str, Any] = {
        "gold_count": len(gold_ids),
        "first_gold_rank": first_gold_rank,
        "first_gold_reciprocal_rank_full_ranking": 1.0 / first_gold_rank,
        "candidate_count": len(passage_ids),
    }
    for k in KS:
        recovered = len(set(passage_ids[:k]) & gold_set)
        computed[f"hit_at_{k}"] = 1.0 if recovered else 0.0
        computed[f"set_recall_at_{k}"] = recovered / len(gold_set)
        computed[f"exact_target_recovery_at_{k}"] = (
            1.0 if gold_set.issubset(passage_ids[:k]) else 0.0
        )
    for key, value in computed.items():
        if per_query[key] != value:
            raise ValueError(
                f"Stored query metric disagrees with raw ranking: {per_query['query_id']} {key}"
            )
    return computed


def _stream_fold_query_metrics(
    evaluation_dir: Path,
    *,
    outer_fold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = _load_canonical_object(evaluation_dir / "evaluation_config.json")
    identity = config.get("identity")
    if type(identity) is not dict or identity.get("outer_fold") != outer_fold or identity.get("role") != "test":
        raise ValueError(f"Fold {outer_fold} evaluation identity changed")
    case_ids = _exact_strings(config.get("case_ids"), name=f"fold[{outer_fold}].case_ids", sorted_values=True)
    query_ids = _exact_strings(config.get("query_ids"), name=f"fold[{outer_fold}].query_ids", sorted_values=True)
    systems = config.get("systems")
    regimes = config.get("regimes")
    if type(systems) is not list or type(regimes) is not list:
        raise ValueError(f"Fold {outer_fold} system/regime inventory changed")
    system_by_id: dict[str, dict[str, Any]] = {}
    for system in systems:
        if type(system) is not dict or type(system.get("system_id")) is not str:
            raise ValueError(f"Fold {outer_fold} contains malformed system metadata")
        system_id = system["system_id"]
        if system_id in system_by_id:
            raise ValueError(f"Fold {outer_fold} contains duplicate systems")
        system_by_id[system_id] = system
    regime_names = tuple(regime.get("regime_name") for regime in regimes if type(regime) is dict)
    if len(regime_names) != len(regimes) or len(regime_names) != len(set(regime_names)):
        raise ValueError(f"Fold {outer_fold} contains malformed regimes")

    rows: list[dict[str, Any]] = []
    coverage: set[tuple[str, str, str]] = set()
    rankings_path = evaluation_dir / "rankings.jsonl"
    if rankings_path.is_symlink() or not rankings_path.is_file():
        raise ValueError(f"Fold {outer_fold} rankings are missing or unsafe")
    rankings_digest = hashlib.sha256()
    with rankings_path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            rankings_digest.update(raw)
            if not raw.endswith(b"\n"):
                raise ValueError(f"Fold {outer_fold} ranking line {line_number} lacks newline")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Fold {outer_fold} ranking line {line_number} is invalid JSON"
                ) from error
            if type(row) is not dict or set(row) != _RANKING_ROW_KEYS or raw != _canonical_bytes(row):
                raise ValueError(f"Fold {outer_fold} ranking line {line_number} is not canonical")
            system_id = row["system_id"]
            regime_name = row["regime_name"]
            query_id = row["query_id"]
            if system_id not in system_by_id or regime_name not in regime_names or query_id not in query_ids:
                raise ValueError(f"Fold {outer_fold} ranking row left its frozen inventory")
            system = system_by_id[system_id]
            if (
                row["system_type"] != system.get("system_type")
                or row["query_view"] != system.get("query_view")
                or row["per_query"].get("query_id") != query_id
                or row["query_index"] != query_ids.index(query_id)
            ):
                raise ValueError(f"Fold {outer_fold} ranking row metadata changed")
            key = (system_id, regime_name, query_id)
            if key in coverage:
                raise ValueError(f"Fold {outer_fold} contains a duplicate ranking row")
            coverage.add(key)
            computed = _recompute_query_metrics(row["per_query"], row["ranking"])
            case_id = row["per_query"]["doc_id"]
            if case_id not in case_ids:
                raise ValueError(f"Fold {outer_fold} query is outside its test cases")
            controlled_view, sampler, seed = _controlled_identity(system_id)
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "case_id": case_id,
                    "query_id": query_id,
                    "system_id": system_id,
                    "system_type": row["system_type"],
                    "query_view": row["query_view"],
                    "regime_name": regime_name,
                    "controlled_query_view": controlled_view,
                    "sampler": sampler,
                    "seed": seed,
                    **{metric: computed[metric] for metric in METRIC_NAMES},
                }
            )
    expected_coverage = {
        (system_id, regime_name, query_id)
        for system_id in system_by_id
        for regime_name in regime_names
        for query_id in query_ids
    }
    if coverage != expected_coverage:
        raise ValueError(f"Fold {outer_fold} ranking coverage is incomplete")
    ranking_record = {
        "outer_fold": outer_fold,
        "logical_path": f"fold-{outer_fold}/evaluation/rankings.jsonl",
        "size": rankings_path.stat().st_size,
        "sha256": rankings_digest.hexdigest(),
        "rows": len(rows),
        "systems": len(system_by_id),
        "regimes": len(regime_names),
        "queries": len(query_ids),
    }
    fold_load = {
        "outer_fold": outer_fold,
        "case_count": len(case_ids),
        "query_count": len(query_ids),
        "passage_count": len(config.get("passage_ids", [])),
        "ranking_row_count": len(rows),
    }
    return rows, ranking_record, fold_load


def _group_mean_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_names: Sequence[str],
    count_name: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in key_names)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        record = dict(zip(key_names, key))
        record[count_name] = len(group)
        for metric in METRIC_NAMES:
            record[metric] = _mean(
                [float(row[metric]) for row in group],
                name=f"group[{key}].{metric}",
            )
        output.append(record)
    return output


def _aggregate_query_rows(query_rows: Sequence[Mapping[str, Any]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    case_rows = _group_mean_rows(
        query_rows,
        key_names=(
            "outer_fold",
            "case_id",
            "system_id",
            "system_type",
            "query_view",
            "regime_name",
            "controlled_query_view",
            "sampler",
            "seed",
        ),
        count_name="query_count",
    )
    if len({row["case_id"] for row in case_rows}) != 42:
        raise ValueError("Case aggregation must cover exactly 42 held-out cases")

    system_case_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    query_system_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in case_rows:
        system_case_groups[(row["system_id"], row["regime_name"])].append(row)
    for row in query_rows:
        query_system_groups[(row["system_id"], row["regime_name"])].append(row)
    system_summary: list[dict[str, Any]] = []
    for key in sorted(system_case_groups):
        case_group = system_case_groups[key]
        query_group = query_system_groups[key]
        if len(case_group) != 42 or len(query_group) != 490:
            raise ValueError(f"System/regime {key} does not cover 42 cases and 490 queries")
        first = case_group[0]
        record = {
            "system_id": key[0],
            "regime_name": key[1],
            "system_type": first["system_type"],
            "query_view": first["query_view"],
            "controlled_query_view": first["controlled_query_view"],
            "sampler": first["sampler"],
            "seed": first["seed"],
            "case_count": len(case_group),
            "query_count": len(query_group),
        }
        for metric in METRIC_NAMES:
            record[f"case_macro_{metric}"] = _mean(
                [float(row[metric]) for row in case_group],
                name=f"system[{key}].case_macro.{metric}",
            )
            record[f"query_micro_{metric}"] = _mean(
                [float(row[metric]) for row in query_group],
                name=f"system[{key}].query_micro.{metric}",
            )
        system_summary.append(record)

    controlled_case_rows = [row for row in case_rows if row["seed"] in SEEDS]
    expected_controlled_case_rows = 42 * 4 * len(SEEDS) * 4
    if len(controlled_case_rows) != expected_controlled_case_rows:
        raise ValueError("Controlled case rows do not cover every case/cell/seed/regime")
    cell_case_rows = _group_mean_rows(
        controlled_case_rows,
        key_names=("outer_fold", "case_id", "regime_name", "controlled_query_view", "sampler"),
        count_name="seed_count",
    )
    if any(row["seed_count"] != len(SEEDS) for row in cell_case_rows):
        raise ValueError("Every case-level controlled cell must average exactly three seeds")
    cell_summary = _group_mean_rows(
        cell_case_rows,
        key_names=("regime_name", "controlled_query_view", "sampler"),
        count_name="case_count",
    )
    if any(row["case_count"] != 42 for row in cell_summary):
        raise ValueError("Every seed-aggregated cell must cover exactly 42 cases")
    seed_summary = _group_mean_rows(
        controlled_case_rows,
        key_names=("regime_name", "controlled_query_view", "sampler", "seed"),
        count_name="case_count",
    )
    if any(row["case_count"] != 42 for row in seed_summary):
        raise ValueError("Every seed-specific cell must cover exactly 42 cases")
    seed_summary_by_cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_summary:
        seed_summary_by_cell[
            (row["regime_name"], row["controlled_query_view"], row["sampler"])
        ].append(row)
    for row in cell_summary:
        key = (row["regime_name"], row["controlled_query_view"], row["sampler"])
        seed_rows = sorted(seed_summary_by_cell[key], key=lambda item: item["seed"])
        if [seed_row["seed"] for seed_row in seed_rows] != list(SEEDS):
            raise ValueError("Cell seed-variability summary is incomplete")
        for metric in METRIC_NAMES:
            row[f"seed_sd_{metric}"] = statistics.stdev(
                float(seed_row[metric]) for seed_row in seed_rows
            )
    return case_rows, system_summary, controlled_case_rows, cell_case_rows, cell_summary, seed_summary


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile inputs are invalid")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(sorted_values[upper]) * fraction


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(float(value) for value in values)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return _linear_percentile(ordered, alpha), _linear_percentile(ordered, 1.0 - alpha)


def _paired_case_bootstrap(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 42:
        raise ValueError("Primary paired bootstrap requires exactly 42 case values")
    rng = random.Random(ANALYSIS_SEED)
    estimates = [
        math.fsum(float(values[rng.randrange(len(values))]) for _ in values) / len(values)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile_interval(estimates)


def _hierarchical_case_seed_bootstrap(
    values_by_case_seed: Sequence[Sequence[float]],
) -> tuple[float, float]:
    if len(values_by_case_seed) != 42 or any(len(values) != len(SEEDS) for values in values_by_case_seed):
        raise ValueError("Hierarchical bootstrap requires a complete 42-case by 3-seed matrix")
    rng = random.Random(ANALYSIS_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        case_values: list[float] = []
        for _ in values_by_case_seed:
            seed_values = values_by_case_seed[rng.randrange(len(values_by_case_seed))]
            case_values.append(
                math.fsum(float(seed_values[rng.randrange(len(SEEDS))]) for _ in SEEDS)
                / len(SEEDS)
            )
        estimates.append(math.fsum(case_values) / len(case_values))
    return _percentile_interval(estimates)


def _claim_status(point: float, lower: float, upper: float) -> str:
    if point > 0.0 and lower > 0.0:
        return "positive_supported"
    if point < 0.0 and upper < 0.0:
        return "negative_supported"
    if lower <= 0.0 <= upper:
        return "uncertain_crosses_zero"
    return "direction_inconsistent"


def _build_primary_contrasts(
    controlled_case_rows: Sequence[Mapping[str, Any]],
    cell_case_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_values: dict[tuple[str, int, str, str], float] = {}
    for row in controlled_case_rows:
        if row["regime_name"] == PRIMARY_REGIME:
            key = (
                row["case_id"],
                row["seed"],
                row["controlled_query_view"],
                row["sampler"],
            )
            if key in seed_values:
                raise ValueError("Duplicate controlled case/seed cell")
            seed_values[key] = float(row[PRIMARY_METRIC])
    cell_values: dict[tuple[str, str, str], float] = {}
    for row in cell_case_rows:
        if row["regime_name"] == PRIMARY_REGIME:
            key = (row["case_id"], row["controlled_query_view"], row["sampler"])
            if key in cell_values:
                raise ValueError("Duplicate seed-aggregated case cell")
            cell_values[key] = float(row[PRIMARY_METRIC])
    case_ids = sorted({key[0] for key in cell_values})
    if len(case_ids) != 42:
        raise ValueError("Primary contrast construction requires all 42 cases")

    contrasts: list[dict[str, Any]] = []
    per_case: dict[str, dict[str, Any]] = {
        case_id: {"case_id": case_id} for case_id in case_ids
    }
    for definition in CONTRASTS:
        case_contrasts = [
            math.fsum(
                weight * cell_values[(case_id, query_view, sampler)]
                for query_view, sampler, weight in definition.weights
            )
            for case_id in case_ids
        ]
        seed_contrasts_by_seed: dict[int, list[float]] = {}
        for seed in SEEDS:
            seed_contrasts_by_seed[seed] = [
                math.fsum(
                    weight * seed_values[(case_id, seed, query_view, sampler)]
                    for query_view, sampler, weight in definition.weights
                )
                for case_id in case_ids
            ]
        case_seed_matrix = [
            [seed_contrasts_by_seed[seed][position] for seed in SEEDS]
            for position in range(len(case_ids))
        ]
        point = _mean(case_contrasts, name=f"contrast[{definition.contrast_id}]")
        lower, upper = _paired_case_bootstrap(case_contrasts)
        hierarchical_lower, hierarchical_upper = _hierarchical_case_seed_bootstrap(
            case_seed_matrix
        )
        seed_estimates = {
            seed: _mean(
                seed_contrasts_by_seed[seed],
                name=f"contrast[{definition.contrast_id}].seed[{seed}]",
            )
            for seed in SEEDS
        }
        record = {
            "contrast_id": definition.contrast_id,
            "label": definition.label,
            "regime_name": PRIMARY_REGIME,
            "metric_name": PRIMARY_METRIC,
            "case_count": len(case_ids),
            "seed_count": len(SEEDS),
            "estimate": point,
            "case_bootstrap_lower": lower,
            "case_bootstrap_upper": upper,
            "hierarchical_lower": hierarchical_lower,
            "hierarchical_upper": hierarchical_upper,
            "seed_17_estimate": seed_estimates[17],
            "seed_29_estimate": seed_estimates[29],
            "seed_43_estimate": seed_estimates[43],
            "seed_sd": statistics.stdev(seed_estimates.values()),
            "claim_status": _claim_status(point, lower, upper),
        }
        contrasts.append(record)
        for case_id, value in zip(case_ids, case_contrasts):
            per_case[case_id][definition.contrast_id] = value
    return contrasts, [per_case[case_id] for case_id in case_ids]


def _jobs_and_rankings(
    acquisition_dirs: Sequence[Path],
    terminals: Sequence[Mapping[str, Any]],
    acquisitions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for fold, (directory, terminal, acquisition) in enumerate(
        zip(acquisition_dirs, terminals, acquisitions)
    ):
        jobs.append(
            {
                "outer_fold": fold,
                "job_name": terminal["job_name"],
                "job_arn": terminal["job_arn"],
                "status": terminal["status"],
                "processing_start_time": terminal["processing_start_time"],
                "processing_end_time": terminal["processing_end_time"],
                "processing_time_microseconds": terminal["processing_time_microseconds"],
                "request_sha256": terminal["request_sha256"],
                "submission_receipt_sha256": terminal["submission_receipt_sha256"],
                "terminal_receipt_sha256": terminal["receipt_sha256"],
                "acquisition_receipt_sha256": acquisition["receipt_sha256"],
                "output_prefix": acquisition["output_prefix"],
            }
        )
        file_record = next(
            record for record in acquisition["files"] if record["path"] == "evaluation/rankings.jsonl"
        )
        remote_record = next(
            record
            for record in acquisition["remote_objects"]
            if record["key"].endswith("/evaluation/rankings.jsonl")
        )
        if file_record["sha256"] != remote_record["sha256"]:
            raise ValueError(f"Fold {fold} local/remote ranking identities disagree")
        rankings.append(
            {
                "outer_fold": fold,
                "logical_path": f"fold-{fold}/evaluation/rankings.jsonl",
                "size": file_record["size"],
                "sha256": file_record["sha256"],
                "s3_bucket": remote_record["bucket"],
                "s3_key": remote_record["key"],
                "s3_version_id": remote_record["version_id"],
            }
        )
    return jobs, rankings


def _validate_locked_analysis_config(experiment_config: Mapping[str, Any]) -> None:
    expected_analysis = {
        "bootstrap": {
            "analysis_seed": ANALYSIS_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "case",
            "method": "paired_percentile",
        },
        "primary_aggregation": [
            "average queries within case",
            "average matched seeds within case",
            "average 42 cases",
        ],
        "primary_claim_rule": (
            "positive mean and paired case-bootstrap interval wholly above zero"
        ),
        "prespecified_contrasts": [
            "flat: global_uniform - local_unique",
            "structured: global_uniform - local_unique",
            "local_unique: structured - flat",
            "global_uniform: structured - flat",
            "(structured|global_uniform - flat_masked|global_uniform) - "
            "(structured|local_unique - flat_masked|local_unique)",
        ],
    }
    if experiment_config.get("analysis") != expected_analysis:
        raise ValueError("Frozen Step-12 analysis configuration changed")
    evaluation = experiment_config.get("evaluation")
    if (
        type(evaluation) is not dict
        or evaluation.get("primary_candidate_regime") != PRIMARY_REGIME
        or evaluation.get("primary_evaluation_role") != "test"
        or evaluation.get("primary_endpoint") != "case_macro_hit_at_20"
        or evaluation.get("robustness_regime", {}).get("candidate_regime")
        != CONTEXT_EXCLUDED_REGIME
    ):
        raise ValueError("Frozen primary endpoint or robustness regime changed")


def build_analysis_bundle(
    *,
    acquisition_dirs: Sequence[Path],
    terminal_receipts: Sequence[Path],
    dataset_dir: Path,
    fold_manifest_path: Path,
    experiment_config_path: Path,
) -> AnalysisBundle:
    """Build the complete Step-12 analysis after strict five-fold readback."""

    experiment_config_path = Path(experiment_config_path)
    fold_manifest_path = Path(fold_manifest_path)
    dataset_dir = Path(dataset_dir)
    experiment_config, experiment_config_bytes = _load_exact_hashed_object(
        experiment_config_path
    )
    fold_manifest, fold_manifest_bytes = _load_exact_hashed_object(fold_manifest_path)
    dataset_manifest, dataset_manifest_bytes = _load_exact_hashed_object(
        dataset_dir / "dataset_manifest.json"
    )
    for path, expected, name in (
        (
            experiment_config_path,
            aggregate.EXPECTED_EXPERIMENT_CONFIG_SHA256,
            "experiment config",
        ),
        (fold_manifest_path, aggregate.EXPECTED_FOLD_MANIFEST_SHA256, "fold manifest"),
        (
            dataset_dir / "dataset_manifest.json",
            aggregate.EXPECTED_DATASET_MANIFEST_SHA256,
            "dataset manifest",
        ),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"Frozen {name} SHA-256 changed")
    dataset_input_files = _load_dataset_input_files(dataset_dir, dataset_manifest)
    _validate_locked_analysis_config(experiment_config)

    ordered_dirs, terminals, acquisitions = _validate_acquisition_chain(
        acquisition_dirs,
        terminal_receipts,
    )
    evaluation_dirs = tuple(directory / "evaluation" for directory in ordered_dirs)
    evaluation_index = aggregate.build_evaluation_index(
        evaluation_dirs,
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
    )
    jobs, ranking_provenance = _jobs_and_rankings(ordered_dirs, terminals, acquisitions)

    query_rows: list[dict[str, Any]] = []
    fold_load: list[dict[str, Any]] = []
    streamed_rankings: list[dict[str, Any]] = []
    index_records = evaluation_index.get("folds")
    if type(index_records) is not list or len(index_records) != 5:
        raise ValueError("Strict evaluation index fold inventory changed")
    for fold, evaluation_dir in enumerate(evaluation_dirs):
        fold_rows, ranking_record, load_record = _stream_fold_query_metrics(
            evaluation_dir,
            outer_fold=fold,
        )
        provenance = ranking_provenance[fold]
        if (
            ranking_record["size"] != provenance["size"]
            or ranking_record["sha256"] != provenance["sha256"]
        ):
            raise ValueError(f"Fold {fold} streamed ranking identity changed")
        index_record = index_records[fold]
        if (
            type(index_record) is not dict
            or index_record.get("outer_fold") != fold
            or load_record["case_count"] != index_record.get("case_count")
            or load_record["query_count"] != index_record.get("query_count")
            or load_record["passage_count"] != index_record.get("passage_count")
            or load_record["ranking_row_count"]
            != index_record.get("ranking_row_count")
        ):
            raise ValueError(f"Fold {fold} streamed load disagrees with strict index")
        streamed_rankings.append({**provenance, **ranking_record})
        query_rows.extend(fold_rows)
        fold_load.append(load_record)
    if len(query_rows) != 5 * 15 * 4 * 98:
        raise ValueError("Five-fold query metric row count changed")
    if {
        "cases": sum(record["case_count"] for record in fold_load),
        "queries": sum(record["query_count"] for record in fold_load),
        "passages": sum(record["passage_count"] for record in fold_load),
    } != {"cases": 42, "queries": 490, "passages": 5_286}:
        raise ValueError("Five-fold streamed workload totals changed")
    query_rows.sort(
        key=lambda row: (
            row["outer_fold"],
            row["system_id"],
            row["regime_name"],
            row["query_id"],
        )
    )
    (
        case_rows,
        system_summary,
        controlled_case_rows,
        cell_case_rows,
        cell_summary,
        seed_summary,
    ) = _aggregate_query_rows(query_rows)
    contrasts, per_case_primary = _build_primary_contrasts(
        controlled_case_rows,
        cell_case_rows,
    )

    primary_cells = [
        {
            "query_view": row["controlled_query_view"],
            "sampler": row["sampler"],
            "estimate": row[PRIMARY_METRIC],
            "seed_sd": row[f"seed_sd_{PRIMARY_METRIC}"],
        }
        for row in cell_summary
        if row["regime_name"] == PRIMARY_REGIME
    ]
    context_sensitivity = []
    cell_lookup = {
        (row["regime_name"], row["controlled_query_view"], row["sampler"]): row
        for row in cell_summary
    }
    for query_view in QUERY_VIEWS:
        for sampler in SAMPLERS:
            global_value = cell_lookup[(PRIMARY_REGIME, query_view, sampler)][PRIMARY_METRIC]
            excluded_value = cell_lookup[
                (CONTEXT_EXCLUDED_REGIME, query_view, sampler)
            ][PRIMARY_METRIC]
            context_sensitivity.append(
                {
                    "query_view": query_view,
                    "sampler": sampler,
                    "fold_global": global_value,
                    "fold_global_context_excluded": excluded_value,
                    "difference": excluded_value - global_value,
                }
            )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "protocol": ANALYSIS_PROTOCOL,
        "experiment_id": "arr_retrieval_cv_v1",
        "counts": {
            "folds": 5,
            "cases": 42,
            "queries": 490,
            "passages": 5_286,
            "systems": 15,
            "controlled_systems": 12,
            "query_metric_rows": len(query_rows),
            "case_metric_rows": len(case_rows),
        },
        "aggregation_order": [
            "recompute query metrics from complete raw rankings",
            "average queries within held-out case",
            "average matched seeds 17/29/43 within case",
            "average 42 held-out cases",
        ],
        "primary_endpoint": {
            "regime_name": PRIMARY_REGIME,
            "metric_name": PRIMARY_METRIC,
            "cells": primary_cells,
        },
        "bootstrap": {
            "analysis_seed": ANALYSIS_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": CONFIDENCE_LEVEL,
            "primary_protocol": BOOTSTRAP_PROTOCOL,
            "percentile_protocol": PERCENTILE_PROTOCOL,
            "conditional_on_seeds": list(SEEDS),
            "hierarchical_sensitivity": "paired case and matched-seed resampling",
        },
        "software": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "stdlib_only_statistics": True,
        },
        "prespecified_contrasts": contrasts,
        "context_excluded_sensitivity": context_sensitivity,
        "reporting_rules": {
            "positive_claim": "estimate > 0 and paired case-bootstrap interval wholly > 0",
            "crosses_zero": "report as uncertain",
            "seed_variability": "sample SD of three seed-specific case-macro contrast estimates",
        },
        "data_boundary": {
            "controlled": "corrected 490-query dataset; all 42 cases yield queries",
            "legacy": "frozen March 471-query dataset; excluded from controlled aggregate",
            "parser_correction": "case 42 left-directed support edge into the final Conclusion is no longer reversed",
            "evidence_copy": (
                "the complete corrected dataset is copied byte-for-byte into the "
                "analysis bundle after manifest hash validation"
            ),
        },
    }
    return AnalysisBundle(
        experiment_config=experiment_config,
        fold_manifest=fold_manifest,
        dataset_manifest=dataset_manifest,
        experiment_config_bytes=experiment_config_bytes,
        fold_manifest_bytes=fold_manifest_bytes,
        dataset_manifest_bytes=dataset_manifest_bytes,
        dataset_input_files=dataset_input_files,
        terminal_receipts=tuple(terminals),
        acquisition_receipts=tuple(acquisitions),
        evaluation_index=evaluation_index,
        jobs=tuple(jobs),
        rankings=tuple(streamed_rankings),
        fold_load=tuple(fold_load),
        query_metrics=tuple(query_rows),
        case_metrics=tuple(case_rows),
        system_summary=tuple(system_summary),
        cell_case_metrics=tuple(cell_case_rows),
        cell_summary=tuple(cell_summary),
        seed_summary=tuple(seed_summary),
        contrasts=tuple(contrasts),
        per_case_primary=tuple(per_case_primary),
        summary=summary,
    )
