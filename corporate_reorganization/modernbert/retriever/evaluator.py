from __future__ import annotations

import ctypes
import errno
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .data import CorpusPassage, QueryExample
from .data import PassageIndexTable, load_corpus, load_queries
from .evaluation import (
    CANONICAL_EVALUATION_KS,
    CanonicalRetrievalResult,
    build_canonical_evaluation_data,
    canonical_result_from_payload,
    compute_canonical_retrieval_result_from_scores,
)
from .query_views import normalize_query_view
from .query_views import select_query_text
from .rankers import (
    RANKER_SCORE_PROTOCOL,
    score_loaded_dual_encoder,
    validate_complete_score_matrix,
)
from .regimes import CANONICAL_CANDIDATE_REGIMES
from .provenance import (
    EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_EXPERIMENT_CONFIG_SHA256,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
)
from .staged_data import validate_staged_dataset_and_fold


EVALUATION_BUNDLE_SCHEMA_VERSION = 1
EVALUATION_BUNDLE_PROTOCOL = "canonical_complete_rankings_v1"
EVALUATION_PLAN_SCHEMA_VERSION = 1
LOCAL_BINDINGS_SCHEMA_VERSION = 1
CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE = "controlled_dual_encoder_artifact"
COMPLETE_EVALUATION_PLAN_SCHEMA_VERSION = 2
COMPLETE_LOCAL_BINDINGS_SCHEMA_VERSION = 2
BM25_SYSTEM_TYPE = "bm25_pyserini"
E5_SYSTEM_TYPE = "e5_base_v2"
FIXED_BASE_SYSTEM_TYPE = "fixed_untrained_modernbert_artifact"
EXPECTED_BASELINE_CONFIG_SHA256 = (
    "714b8c18e9e32130ebf3358a72d9c6aceceeb1e14ee0d76270306d901b81f33a"
)
EXPECTED_E5_PACK_MANIFEST_SHA256 = (
    "9875bd57c23a7e390c85d2a4b1b3aab7415597c0223c2fed621e613d4dfded10"
)
EXPECTED_E5_PACK_INVENTORY_SHA256 = (
    "9cfe6cbd83c60a686751c82d1c811612a27eb5a04d835a1a600335081f5b1edf"
)
EXPECTED_FIXED_BASE_ARTIFACT_MANIFEST_SHA256 = (
    "ccff3fa4c141290ef9383992a4d3de2b8cfa5e50d02c4cd06e3fe52e92d0202b"
)
EXPECTED_FIXED_BASE_MODEL_SHA256 = (
    "a2822fd04d0ba9b5df5289d9384e89740d113ddd68810a8d05ba6dbefbc33300"
)
EXPECTED_FIXED_BASE_NEW_ROWS_SHA256 = (
    "6dba50931329f2bea4618616ba222440488b776dd1216a2a61279f83f9e9a26b"
)
EXPECTED_FIXED_BASE_STATE_KEYS_SHA256 = (
    "d715c23e469ddfad4e731db3c01f30ef8b7fc1a6e7117fc37915d845d20386a9"
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVALUATION_IMAGE_URI = re.compile(
    r"[0-9]{12}\.dkr\.ecr\.us-east-1\.amazonaws\.com/"
    r"arr-retrieval-eval@sha256:[0-9a-f]{64}"
)
PROCESSING_JOB_CONFIG_PATH = Path("/opt/ml/config/processingjobconfig.json")


def _evaluation_protocols() -> dict[str, str]:
    return {
        "score_matrix": RANKER_SCORE_PROTOCOL,
        "ranking": "score_desc_passage_id_asc_v1",
        "context_filter": "visible_nongold_filter_without_rescoring_v1",
        "aggregation": "query_within_case_then_case_macro_math_fsum_v1",
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and _LOWER_SHA256.fullmatch(value) is not None


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    if type(value) is tuple:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _exact_strings(values: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list or tuple")
    result: list[str] = []
    for position, value in enumerate(values):
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError(f"{name}[{position}] must be a non-empty exact string")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates")
    if result != sorted(result):
        raise ValueError(f"{name} must be lexicographically sorted")
    return tuple(result)


def _validate_scientific_json(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    normalized = json.loads(_canonical_json(dict(value)))

    def inspect(item: object, *, path: str) -> None:
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str or not key or key.strip() != key:
                    raise ValueError(f"{path} contains an invalid key")
                inspect(child, path=f"{path}.{key}")
        elif type(item) is list:
            for position, child in enumerate(item):
                inspect(child, path=f"{path}[{position}]")
        elif type(item) is str:
            if item.startswith(("/", "file://")):
                raise ValueError(f"{path} contains a local absolute path")
        elif item is None or type(item) in {bool, int, float}:
            return
        else:
            raise TypeError(f"{path} contains a non-JSON value")

    inspect(normalized, path=name)
    return normalized


@dataclass(frozen=True)
class EvaluationIdentity:
    experiment_id: str
    outer_fold: int
    role: str
    evaluation_plan_sha256: str
    experiment_config_sha256: str
    dataset_manifest_sha256: str
    fold_manifest_sha256: str
    passage_index_sha256: str

    def to_payload(self) -> dict[str, Any]:
        if (
            type(self.experiment_id) is not str
            or not self.experiment_id
            or self.experiment_id.strip() != self.experiment_id
        ):
            raise ValueError("experiment_id must be a non-empty exact string")
        if type(self.outer_fold) is not int or self.outer_fold not in range(5):
            raise ValueError("outer_fold must be an exact integer in 0..4")
        if self.role not in {"validation", "test"}:
            raise ValueError("role must be exactly 'validation' or 'test'")
        payload = {
            "experiment_id": self.experiment_id,
            "outer_fold": self.outer_fold,
            "role": self.role,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "experiment_config_sha256": self.experiment_config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "passage_index_sha256": self.passage_index_sha256,
        }
        for field_name in (
            "evaluation_plan_sha256",
            "experiment_config_sha256",
            "dataset_manifest_sha256",
            "fold_manifest_sha256",
            "passage_index_sha256",
        ):
            if not _is_sha256(payload[field_name]):
                raise ValueError(f"{field_name} must be lowercase SHA-256")
        return payload


@dataclass(frozen=True, eq=False)
class SystemScoreInput:
    system_id: str
    system_type: str
    query_view: str
    model_identity: Mapping[str, Any]
    scores: Any


@dataclass(frozen=True)
class CanonicalEvaluationBundle:
    config: Mapping[str, Any]
    result_records: tuple[Mapping[str, Any], ...]


def _system_metadata(system: SystemScoreInput) -> dict[str, Any]:
    if type(system.system_id) is not str or not system.system_id or (
        system.system_id.strip() != system.system_id
    ):
        raise ValueError("system_id must be a non-empty exact string")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in system.system_id):
        raise ValueError(f"system_id has unsupported characters: {system.system_id!r}")
    if (
        type(system.system_type) is not str
        or not system.system_type
        or system.system_type.strip() != system.system_type
    ):
        raise ValueError("system_type must be a non-empty exact string")
    query_view = normalize_query_view(system.query_view)
    if type(system.query_view) is not str or system.query_view != query_view:
        raise ValueError("query_view must use its exact canonical spelling")
    return {
        "system_id": system.system_id,
        "system_type": system.system_type,
        "query_view": query_view,
        "model_identity": _validate_scientific_json(
            system.model_identity,
            name=f"system[{system.system_id}].model_identity",
        ),
    }


def build_canonical_evaluation_bundle(
    *,
    identity: EvaluationIdentity,
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    evaluated_case_ids: Sequence[str],
    systems: Sequence[SystemScoreInput],
    runtime_identity: Mapping[str, Any],
    torch_module,
) -> CanonicalEvaluationBundle:
    """Evaluate complete aligned score matrices under all four locked regimes."""

    identity_payload = identity.to_payload()
    runtime_payload = _validate_scientific_json(
        runtime_identity,
        name="runtime_identity",
    )
    case_ids = _exact_strings(evaluated_case_ids, name="evaluated_case_ids")
    if not isinstance(systems, Sequence) or isinstance(systems, (str, bytes)) or not systems:
        raise ValueError("systems must be a non-empty sequence")
    system_list = list(systems)
    if any(not isinstance(system, SystemScoreInput) for system in system_list):
        raise TypeError("Every system must be SystemScoreInput")
    metadata_by_system_id: dict[str, dict[str, Any]] = {}
    for system in system_list:
        metadata = _system_metadata(system)
        if metadata["system_id"] in metadata_by_system_id:
            raise ValueError(f"Duplicate evaluation system_id={metadata['system_id']!r}")
        metadata_by_system_id[metadata["system_id"]] = metadata
    if [system.system_id for system in system_list] != sorted(metadata_by_system_id):
        raise ValueError("Evaluation systems must be supplied in system_id order")

    data_by_regime = {
        regime_name: build_canonical_evaluation_data(
            all_queries=all_queries,
            corpus_by_passage_id=corpus_by_passage_id,
            evaluated_case_ids=case_ids,
            role=identity.role,
            regime_name=regime_name,
        )
        for regime_name in CANONICAL_CANDIDATE_REGIMES
    }
    fold_global_data = data_by_regime["fold_global"]
    query_ids = tuple(query.query_id for query in fold_global_data.queries)
    passage_ids = fold_global_data.passage_ids
    for regime_name, evaluation_data in data_by_regime.items():
        if (
            tuple(query.query_id for query in evaluation_data.queries) != query_ids
            or evaluation_data.passage_ids != passage_ids
            or evaluation_data.case_ids != fold_global_data.case_ids
        ):
            raise RuntimeError(f"Canonical regime {regime_name!r} changed role inventories")

    result_records: list[dict[str, Any]] = []
    for system in system_list:
        metadata = metadata_by_system_id[system.system_id]
        scores = validate_complete_score_matrix(
            system.scores,
            query_ids=query_ids,
            passage_ids=passage_ids,
            torch_module=torch_module,
        )
        source_ranking_sha256: str | None = None
        for regime_name in CANONICAL_CANDIDATE_REGIMES:
            result = compute_canonical_retrieval_result_from_scores(
                scores=scores,
                evaluation_data=data_by_regime[regime_name],
            )
            if source_ranking_sha256 is None:
                source_ranking_sha256 = result.source_ranking_sha256
            elif result.source_ranking_sha256 != source_ranking_sha256:
                raise RuntimeError("Canonical regimes did not share one source ranking")
            result_records.append(
                {
                    **metadata,
                    "regime_name": regime_name,
                    "result": result.to_payload(),
                }
            )

    config = {
        "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
        "bundle_protocol": EVALUATION_BUNDLE_PROTOCOL,
        "identity": identity_payload,
        "runtime_identity": runtime_payload,
        "protocols": _evaluation_protocols(),
        "ks": list(CANONICAL_EVALUATION_KS),
        "case_ids": list(fold_global_data.case_ids),
        "case_ids_sha256": fold_global_data.case_ids_sha256,
        "query_ids": list(query_ids),
        "query_ids_sha256": fold_global_data.query_ids_sha256,
        "passage_ids": list(passage_ids),
        "passage_ids_sha256": fold_global_data.passage_ids_sha256,
        "regimes": [
            {
                "regime_name": regime_name,
                "candidate_pools_sha256": data_by_regime[regime_name].candidate_pools_sha256,
                "evaluation_contract_sha256": data_by_regime[regime_name].contract_sha256,
            }
            for regime_name in CANONICAL_CANDIDATE_REGIMES
        ],
        "systems": [metadata_by_system_id[system.system_id] for system in system_list],
    }
    return CanonicalEvaluationBundle(
        config=_freeze_json(config),
        result_records=tuple(_freeze_json(record) for record in result_records),
    )


def _bundle_output_payloads(bundle: CanonicalEvaluationBundle) -> dict[str, bytes]:
    config = _thaw_json(bundle.config)
    result_records = [_thaw_json(record) for record in bundle.result_records]
    expected_record_keys = {
        "system_id",
        "system_type",
        "query_view",
        "model_identity",
        "regime_name",
        "result",
    }
    ranking_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for record in result_records:
        if type(record) is not dict or set(record) != expected_record_keys:
            raise RuntimeError("Evaluation bundle result record schema changed")
        result = record["result"]
        if type(result) is not dict:
            raise RuntimeError("Evaluation bundle contains a malformed canonical result")
        per_query = result["per_query"]
        source_rankings = result["source_rankings"]
        rankings = result["rankings"]
        if not (
            type(per_query) is list
            and type(source_rankings) is list
            and type(rankings) is list
            and len(per_query) == len(source_rankings) == len(rankings)
        ):
            raise RuntimeError("Canonical result query/ranking rows do not align")
        for query_index, (query_metrics, source_ranking, ranking) in enumerate(
            zip(per_query, source_rankings, rankings)
        ):
            query_id = query_metrics.get("query_id") if type(query_metrics) is dict else None
            if (
                type(query_id) is not str
                or source_ranking.get("query_id") != query_id
                or ranking.get("query_id") != query_id
            ):
                raise RuntimeError("Canonical ranking row query identities changed")
            ranking_rows.append(
                {
                    "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
                    "system_id": record["system_id"],
                    "system_type": record["system_type"],
                    "query_view": record["query_view"],
                    "regime_name": record["regime_name"],
                    "query_index": query_index,
                    "query_id": query_id,
                    "per_query": query_metrics,
                    "source_ranking": source_ranking,
                    "ranking": ranking,
                }
            )
        result_summary = {
            key: value
            for key, value in result.items()
            if key not in {"per_query", "source_rankings", "rankings"}
        }
        summaries.append(
            {
                **{key: record[key] for key in expected_record_keys if key != "result"},
                "result": result_summary,
            }
        )

    rankings_bytes = b"".join(
        (_canonical_json(row) + "\n").encode("utf-8") for row in ranking_rows
    )
    results_payload = {
        "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
        "bundle_protocol": EVALUATION_BUNDLE_PROTOCOL,
        "result_records": summaries,
    }
    return {
        "evaluation_config.json": _canonical_bytes(config),
        "rankings.jsonl": rankings_bytes,
        "results.json": _canonical_bytes(results_payload),
    }


def _write_new_file(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_to_absent(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "Atomic no-replace publication requires Linux renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"Refusing to replace evaluation output: {target}",
            str(target),
        )
    raise OSError(
        error_number,
        f"Atomic no-replace evaluation publication failed: {source} -> {target}",
    )


def publish_canonical_evaluation_bundle(
    bundle: CanonicalEvaluationBundle,
    *,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Atomically publish deterministic bytes with the manifest as commit marker."""

    if not isinstance(bundle, CanonicalEvaluationBundle):
        raise TypeError("bundle must be CanonicalEvaluationBundle")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Evaluation output must be absent: {output_dir}")
    parent = output_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"Evaluation output parent must be a real directory: {parent}")
    incomplete = parent / f".{output_dir.name}.incomplete"
    if incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError(f"Stale incomplete evaluation output exists: {incomplete}")

    payloads = _bundle_output_payloads(bundle)
    incomplete.mkdir()
    manifest_path = incomplete / "artifact_manifest.json"
    published = False
    try:
        file_records: list[dict[str, Any]] = []
        for relative_name in sorted(payloads):
            payload = payloads[relative_name]
            _write_new_file(incomplete / relative_name, payload)
            file_records.append(
                {
                    "path": relative_name,
                    "size": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
        manifest = {
            "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
            "bundle_protocol": EVALUATION_BUNDLE_PROTOCOL,
            "commit_marker": True,
            "files": file_records,
        }
        _fsync_directory(incomplete)
        _write_new_file(manifest_path, _canonical_bytes(manifest))
        _fsync_directory(incomplete)
        _rename_directory_to_absent(incomplete, output_dir)
        published = True
        _fsync_directory(parent)

        actual_names = sorted(entry.name for entry in output_dir.iterdir())
        expected_names = sorted([*payloads, "artifact_manifest.json"])
        if actual_names != expected_names:
            raise RuntimeError("Published evaluation inventory changed")
        for record in file_records:
            path = output_dir / record["path"]
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Published evaluation contains a non-regular artifact")
            if path.stat().st_size != record["size"] or _sha256_file(path) != record["sha256"]:
                raise RuntimeError("Published evaluation artifact failed exact readback")
        if json.loads((output_dir / "artifact_manifest.json").read_bytes()) != manifest:
            raise RuntimeError("Published evaluation commit marker changed on readback")
        return _freeze_json(
            {
                "output_name": output_dir.name,
                "artifact_manifest_sha256": _sha256_file(
                    output_dir / "artifact_manifest.json"
                ),
                "files": file_records,
            }
        )
    except BaseException:
        current_root = output_dir if published else incomplete
        current_manifest = current_root / "artifact_manifest.json"
        if current_manifest.exists() or current_manifest.is_symlink():
            current_manifest.unlink()
            _fsync_directory(current_root)
        if published and output_dir.exists() and not incomplete.exists():
            _rename_directory_to_absent(output_dir, incomplete)
            _fsync_directory(parent)
        raise


def _retract_published_evaluation(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    parent = output_dir.parent
    incomplete = parent / f".{output_dir.name}.incomplete"
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError(f"Cannot retract unsafe evaluation output: {output_dir}")
    manifest_path = output_dir / "artifact_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest_path.unlink()
        _fsync_directory(output_dir)
    if os.path.lexists(incomplete):
        raise RuntimeError(f"Cannot retract over an existing incomplete output: {incomplete}")
    _rename_directory_to_absent(output_dir, incomplete)
    _fsync_directory(parent)


def publish_and_validate_canonical_evaluation_bundle(
    bundle: CanonicalEvaluationBundle,
    *,
    output_dir: Path,
    identity: EvaluationIdentity,
    runtime_identity: Mapping[str, Any],
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    expected_system_contract: Sequence[tuple[str, str, str]] | None = None,
) -> Mapping[str, Any]:
    """Publish once and retract the success marker on any scientific readback failure."""

    publication = publish_canonical_evaluation_bundle(bundle, output_dir=output_dir)
    try:
        validate_published_evaluation_bundle(
            output_dir=output_dir,
            identity=identity,
            runtime_identity=runtime_identity,
            all_queries=all_queries,
            corpus_by_passage_id=corpus_by_passage_id,
            expected_system_contract=expected_system_contract,
        )
    except BaseException:
        _retract_published_evaluation(output_dir)
        raise
    return publication


def _load_canonical_file(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evaluation artifact must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Evaluation artifact is not valid JSON: {path}") from error
    if raw != _canonical_bytes(value):
        raise ValueError(f"Evaluation JSON bytes are not canonical: {path}")
    return value


def validate_published_evaluation_bundle(
    *,
    output_dir: Path,
    identity: EvaluationIdentity,
    runtime_identity: Mapping[str, Any],
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    expected_system_contract: Sequence[tuple[str, str, str]] | None = None,
) -> tuple[CanonicalRetrievalResult, ...]:
    """Read complete rankings and independently reconstruct every stored result."""

    output_dir = Path(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"Evaluation bundle must be a real directory: {output_dir}")
    expected_names = {
        "artifact_manifest.json",
        "evaluation_config.json",
        "rankings.jsonl",
        "results.json",
    }
    actual_names = {entry.name for entry in output_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("Evaluation bundle inventory changed")
    manifest = _load_canonical_file(output_dir / "artifact_manifest.json")
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "bundle_protocol", "commit_marker", "files"}
        or manifest["schema_version"] != EVALUATION_BUNDLE_SCHEMA_VERSION
        or manifest["bundle_protocol"] != EVALUATION_BUNDLE_PROTOCOL
        or manifest["commit_marker"] is not True
    ):
        raise ValueError("Evaluation artifact manifest schema changed")
    file_records = manifest["files"]
    if type(file_records) is not list or [record.get("path") for record in file_records] != [
        "evaluation_config.json",
        "rankings.jsonl",
        "results.json",
    ]:
        raise ValueError("Evaluation artifact manifest file order/inventory changed")
    for record in file_records:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Evaluation artifact manifest file record is malformed")
        path = output_dir / record["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Evaluation artifact is missing or unsafe")
        if (
            type(record["size"]) is not int
            or record["size"] < 1
            or path.stat().st_size != record["size"]
            or not _is_sha256(record["sha256"])
            or _sha256_file(path) != record["sha256"]
        ):
            raise ValueError("Evaluation artifact size/hash changed")

    config = _load_canonical_file(output_dir / "evaluation_config.json")
    expected_runtime_identity = _validate_scientific_json(
        runtime_identity,
        name="runtime_identity",
    )
    if (
        type(config) is not dict
        or config.get("identity") != identity.to_payload()
        or config.get("runtime_identity") != expected_runtime_identity
    ):
        raise ValueError("Evaluation configuration identity changed")
    expected_config_keys = {
        "schema_version",
        "bundle_protocol",
        "identity",
        "runtime_identity",
        "protocols",
        "ks",
        "case_ids",
        "case_ids_sha256",
        "query_ids",
        "query_ids_sha256",
        "passage_ids",
        "passage_ids_sha256",
        "regimes",
        "systems",
    }
    if set(config) != expected_config_keys:
        raise ValueError("Evaluation configuration schema changed")
    case_ids = _exact_strings(config["case_ids"], name="config.case_ids")
    data_by_regime = {
        regime_name: build_canonical_evaluation_data(
            all_queries=all_queries,
            corpus_by_passage_id=corpus_by_passage_id,
            evaluated_case_ids=case_ids,
            role=identity.role,
            regime_name=regime_name,
        )
        for regime_name in CANONICAL_CANDIDATE_REGIMES
    }
    fold_global_data = data_by_regime["fold_global"]
    if (
        config["schema_version"] != EVALUATION_BUNDLE_SCHEMA_VERSION
        or config["bundle_protocol"] != EVALUATION_BUNDLE_PROTOCOL
        or config["protocols"] != _evaluation_protocols()
        or config["ks"] != list(CANONICAL_EVALUATION_KS)
        or config["case_ids_sha256"] != fold_global_data.case_ids_sha256
        or config["query_ids"] != [query.query_id for query in fold_global_data.queries]
        or config["query_ids_sha256"] != fold_global_data.query_ids_sha256
        or config["passage_ids"] != list(fold_global_data.passage_ids)
        or config["passage_ids_sha256"] != fold_global_data.passage_ids_sha256
    ):
        raise ValueError("Evaluation configuration inventories changed")
    expected_regimes = [
        {
            "regime_name": regime_name,
            "candidate_pools_sha256": data_by_regime[regime_name].candidate_pools_sha256,
            "evaluation_contract_sha256": data_by_regime[regime_name].contract_sha256,
        }
        for regime_name in CANONICAL_CANDIDATE_REGIMES
    ]
    if config["regimes"] != expected_regimes:
        raise ValueError("Evaluation configuration regime contracts changed")

    results_payload = _load_canonical_file(output_dir / "results.json")
    if (
        type(results_payload) is not dict
        or set(results_payload) != {"schema_version", "bundle_protocol", "result_records"}
        or results_payload["schema_version"] != EVALUATION_BUNDLE_SCHEMA_VERSION
        or results_payload["bundle_protocol"] != EVALUATION_BUNDLE_PROTOCOL
        or type(results_payload["result_records"]) is not list
    ):
        raise ValueError("Evaluation results schema changed")
    summary_records = results_payload["result_records"]

    rankings_path = output_dir / "rankings.jsonl"
    raw_rankings = rankings_path.read_bytes()
    if not raw_rankings or not raw_rankings.endswith(b"\n"):
        raise ValueError("Evaluation rankings JSONL is empty or unterminated")
    ranking_rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_rankings.splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid rankings JSONL line {line_number}") from error
        if raw_line.decode("utf-8") != _canonical_json(row):
            raise ValueError(f"Rankings JSONL line {line_number} is not canonical")
        if type(row) is not dict:
            raise ValueError(f"Rankings JSONL line {line_number} is not an object")
        ranking_rows.append(row)

    expected_systems = config["systems"]
    if type(expected_systems) is not list or not expected_systems:
        raise ValueError("Evaluation configuration system inventory is empty")
    if expected_system_contract is not None:
        contract = tuple(expected_system_contract)
        if (
            not contract
            or any(
                type(record) is not tuple
                or len(record) != 3
                or any(type(value) is not str or not value for value in record)
                for record in contract
            )
        ):
            raise ValueError("Expected evaluation system contract is malformed")
        actual_contract = tuple(
            (
                system.get("system_id"),
                system.get("system_type"),
                system.get("query_view"),
            )
            for system in expected_systems
        )
        if actual_contract != contract:
            raise ValueError("Evaluation bundle left its expected system contract")
    expected_pairs = [
        (system["system_id"], regime_name)
        for system in expected_systems
        for regime_name in CANONICAL_CANDIDATE_REGIMES
    ]
    actual_pairs = [
        (record.get("system_id"), record.get("regime_name")) for record in summary_records
    ]
    if actual_pairs != expected_pairs:
        raise ValueError("Evaluation result system/regime order or coverage changed")

    query_count = fold_global_data.query_count
    if len(ranking_rows) != len(expected_pairs) * query_count:
        raise ValueError("Evaluation rankings do not cover every system/regime/query")
    reconstructed: list[CanonicalRetrievalResult] = []
    source_ranking_sha256_by_system: dict[str, str] = {}
    row_offset = 0
    for summary, pair in zip(summary_records, expected_pairs):
        expected_summary_keys = {
            "system_id",
            "system_type",
            "query_view",
            "model_identity",
            "regime_name",
            "result",
        }
        if type(summary) is not dict or set(summary) != expected_summary_keys:
            raise ValueError("Evaluation result summary schema changed")
        system_id, regime_name = pair
        system_config = next(
            system for system in expected_systems if system["system_id"] == system_id
        )
        if (
            {key: summary[key] for key in ("system_id", "system_type", "query_view", "model_identity")}
            != system_config
            or summary["regime_name"] != regime_name
        ):
            raise ValueError("Evaluation result summary system identity changed")
        rows = ranking_rows[row_offset : row_offset + query_count]
        row_offset += query_count
        per_query: list[object] = []
        source_rankings: list[object] = []
        rankings: list[object] = []
        for query_index, row in enumerate(rows):
            expected_row_keys = {
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
            if type(row) is not dict or set(row) != expected_row_keys:
                raise ValueError("Evaluation ranking row schema changed")
            expected_query_id = data_by_regime[regime_name].queries[query_index].query_id
            if (
                row["schema_version"] != EVALUATION_BUNDLE_SCHEMA_VERSION
                or row["system_id"] != system_id
                or row["system_type"] != system_config["system_type"]
                or row["query_view"] != system_config["query_view"]
                or row["regime_name"] != regime_name
                or row["query_index"] != query_index
                or row["query_id"] != expected_query_id
            ):
                raise ValueError("Evaluation ranking row identity/order changed")
            per_query.append(row["per_query"])
            source_rankings.append(row["source_ranking"])
            rankings.append(row["ranking"])
        result_summary = summary["result"]
        if type(result_summary) is not dict:
            raise ValueError("Evaluation canonical result summary is malformed")
        full_payload = {
            **result_summary,
            "per_query": per_query,
            "source_rankings": source_rankings,
            "rankings": rankings,
        }
        result = canonical_result_from_payload(
            full_payload,
            data_by_regime[regime_name],
        )
        previous_source_sha256 = source_ranking_sha256_by_system.get(system_id)
        if previous_source_sha256 is None:
            source_ranking_sha256_by_system[system_id] = result.source_ranking_sha256
        elif result.source_ranking_sha256 != previous_source_sha256:
            raise ValueError(
                f"System {system_id!r} regimes do not share one complete source ranking"
            )
        reconstructed.append(result)
    return tuple(reconstructed)


def _parse_exact_json_object(
    raw: bytes,
    *,
    path: Path,
    name: str,
    canonical: bool,
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    if canonical and raw != _canonical_bytes(value):
        raise ValueError(f"{name} must use canonical deterministic JSON bytes")
    return value


def _load_exact_json_file(path: Path, *, name: str, canonical: bool) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    return _parse_exact_json_object(
        path.read_bytes(),
        path=path,
        name=name,
        canonical=canonical,
    )


def _load_exact_json_file_with_sha256(
    path: Path,
    *,
    name: str,
    canonical: bool,
) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    value = _parse_exact_json_object(
        raw,
        path=path,
        name=name,
        canonical=canonical,
    )
    return value, _sha256_bytes(raw)


def _validate_processing_job_image_and_command(
    *,
    config_path: Path,
    expected_image_uri: str,
    expected_entrypoint: Sequence[str],
    evaluation_plan_path: Path,
    local_bindings_path: Path,
    output_dir: Path,
    device: str,
) -> str:
    config = _load_exact_json_file(
        Path(config_path),
        name="SageMaker processing job configuration",
        canonical=False,
    )
    application = config.get("AppSpecification")
    if type(application) is not dict or set(application) != {
        "ImageUri",
        "ContainerEntrypoint",
        "ContainerArguments",
    }:
        raise ValueError("SageMaker AppSpecification schema changed")
    entrypoint = list(expected_entrypoint)
    if (
        not entrypoint
        or any(type(value) is not str or not value for value in entrypoint)
        or application["ImageUri"] != expected_image_uri
        or type(application["ImageUri"]) is not str
        or _EVALUATION_IMAGE_URI.fullmatch(application["ImageUri"]) is None
        or application["ContainerEntrypoint"] != entrypoint
    ):
        raise RuntimeError("SageMaker Processing image or entrypoint changed")
    expected_arguments = [
        "--evaluation-plan",
        str(evaluation_plan_path),
        "--local-bindings",
        str(local_bindings_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    if application["ContainerArguments"] != expected_arguments:
        raise RuntimeError("SageMaker Processing container arguments changed")
    return application["ImageUri"]


def _runtime_identity_for_controlled_artifacts(*, device: str, runtime) -> dict[str, str]:
    torch_module = runtime.torch_module
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    return {
        "device": device,
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda": str(cuda_version),
    }


def _validate_controlled_evaluation_plan(
    plan: dict[str, Any],
    *,
    evaluation_plan_sha256: str,
) -> tuple[EvaluationIdentity, tuple[str, ...], list[dict[str, Any]]]:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "outer_fold",
        "role",
        "experiment_config_sha256",
        "dataset_manifest_sha256",
        "fold_manifest_sha256",
        "passage_index_sha256",
        "case_ids",
        "query_count",
        "passage_count",
        "max_len_query",
        "max_len_passage",
        "query_batch_size",
        "passage_batch_size",
        "runtime_identity",
        "systems",
    }
    if set(plan) != expected_keys:
        raise ValueError(
            "Evaluation plan schema changed: "
            f"missing={sorted(expected_keys - set(plan))}, "
            f"extra={sorted(set(plan) - expected_keys)}"
        )
    if plan["schema_version"] != EVALUATION_PLAN_SCHEMA_VERSION or type(
        plan["schema_version"]
    ) is not int:
        raise ValueError("Evaluation plan schema_version must be exact integer 1")
    identity = EvaluationIdentity(
        experiment_id=plan["experiment_id"],
        outer_fold=plan["outer_fold"],
        role=plan["role"],
        evaluation_plan_sha256=evaluation_plan_sha256,
        experiment_config_sha256=plan["experiment_config_sha256"],
        dataset_manifest_sha256=plan["dataset_manifest_sha256"],
        fold_manifest_sha256=plan["fold_manifest_sha256"],
        passage_index_sha256=plan["passage_index_sha256"],
    )
    identity.to_payload()
    expected_identity = {
        "experiment_id": "arr_retrieval_cv_v1",
        "experiment_config_sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
    }
    for field_name, expected in expected_identity.items():
        if getattr(identity, field_name) != expected:
            raise ValueError(
                f"Evaluation plan {field_name} left the frozen controlled study"
            )
    case_ids = _exact_strings(plan["case_ids"], name="plan.case_ids")
    if type(plan["query_count"]) is not int or plan["query_count"] < 1:
        raise ValueError("Evaluation plan query_count must be a positive exact integer")
    if type(plan["passage_count"]) is not int or plan["passage_count"] < 1:
        raise ValueError("Evaluation plan passage_count must be a positive exact integer")
    if plan["max_len_query"] != 4_096 or type(plan["max_len_query"]) is not int:
        raise ValueError("Controlled final max_len_query must be exact integer 4096")
    if plan["max_len_passage"] != 500 or type(plan["max_len_passage"]) is not int:
        raise ValueError("Controlled final max_len_passage must be exact integer 500")
    for name in ("query_batch_size", "passage_batch_size"):
        if type(plan[name]) is not int or plan[name] < 1:
            raise ValueError(f"Evaluation plan {name} must be a positive exact integer")
    _validate_scientific_json(plan["runtime_identity"], name="plan.runtime_identity")
    systems = plan["systems"]
    if type(systems) is not list or not systems:
        raise ValueError("Evaluation plan systems must be a non-empty list")
    expected_system_keys = {
        "system_id",
        "system_type",
        "query_view",
        "artifact_expectation",
    }
    normalized_systems: list[dict[str, Any]] = []
    system_ids: list[str] = []
    expectation_keys = {
        "artifact_manifest_sha256",
        "experiment_id",
        "outer_fold",
        "query_view",
        "sampler",
        "experiment_seed",
        "dataset_manifest_sha256",
        "fold_manifest_sha256",
        "passage_index_sha256",
        "model_artifact_protocol",
    }
    from .artifacts import ControlledArtifactExpectation

    for position, system in enumerate(systems):
        if type(system) is not dict or set(system) != expected_system_keys:
            raise ValueError(f"Evaluation plan system {position} has an invalid schema")
        system_id = system["system_id"]
        if (
            type(system_id) is not str
            or not system_id
            or system_id.strip() != system_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in system_id
            )
        ):
            raise ValueError(f"Evaluation plan system {position} has an invalid system_id")
        if system["system_type"] != CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE:
            raise ValueError(
                "Step 7 local runner supports only controlled dual-encoder artifacts; "
                "Step 8 adds pinned baseline rankers"
            )
        query_view = normalize_query_view(system["query_view"])
        if type(system["query_view"]) is not str or system["query_view"] != query_view:
            raise ValueError(
                f"Evaluation plan system {system_id!r} query_view is not canonical"
            )
        expectation = system["artifact_expectation"]
        if type(expectation) is not dict or set(expectation) != expectation_keys:
            raise ValueError(f"System {system_id!r} artifact expectation schema changed")
        if (
            expectation["experiment_id"] != identity.experiment_id
            or expectation["outer_fold"] != identity.outer_fold
            or expectation["query_view"] != query_view
            or expectation["dataset_manifest_sha256"] != identity.dataset_manifest_sha256
            or expectation["fold_manifest_sha256"] != identity.fold_manifest_sha256
            or expectation["passage_index_sha256"] != identity.passage_index_sha256
        ):
            raise ValueError(f"System {system_id!r} artifact expectation left the plan identity")
        ControlledArtifactExpectation(**expectation)
        system_ids.append(system_id)
        normalized_systems.append(
            {
                "system_id": system_id,
                "system_type": system["system_type"],
                "query_view": query_view,
                "artifact_expectation": dict(expectation),
            }
        )
    if system_ids != sorted(system_ids) or len(system_ids) != len(set(system_ids)):
        raise ValueError("Evaluation plan systems must be unique and sorted by system_id")
    return identity, case_ids, normalized_systems


def _validate_complete_evaluation_plan(
    plan: dict[str, Any],
    *,
    evaluation_plan_sha256: str,
) -> tuple[EvaluationIdentity, tuple[str, ...], list[dict[str, Any]]]:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "outer_fold",
        "role",
        "experiment_config_sha256",
        "dataset_manifest_sha256",
        "fold_manifest_sha256",
        "passage_index_sha256",
        "baseline_config_sha256",
        "image_contract_sha256",
        "image_uri",
        "case_ids",
        "query_count",
        "passage_count",
        "controlled_max_len_query",
        "controlled_max_len_passage",
        "e5_max_len_passage",
        "query_batch_size",
        "passage_batch_size",
        "runtime_identity",
        "systems",
    }
    if set(plan) != expected_keys:
        raise ValueError(
            "Complete evaluation plan schema changed: "
            f"missing={sorted(expected_keys - set(plan))}, "
            f"extra={sorted(set(plan) - expected_keys)}"
        )
    if (
        plan["schema_version"] != COMPLETE_EVALUATION_PLAN_SCHEMA_VERSION
        or type(plan["schema_version"]) is not int
    ):
        raise ValueError("Complete evaluation plan schema_version must be exact integer 2")
    identity = EvaluationIdentity(
        experiment_id=plan["experiment_id"],
        outer_fold=plan["outer_fold"],
        role=plan["role"],
        evaluation_plan_sha256=evaluation_plan_sha256,
        experiment_config_sha256=plan["experiment_config_sha256"],
        dataset_manifest_sha256=plan["dataset_manifest_sha256"],
        fold_manifest_sha256=plan["fold_manifest_sha256"],
        passage_index_sha256=plan["passage_index_sha256"],
    )
    identity.to_payload()
    expected_identity = {
        "experiment_id": "arr_retrieval_cv_v1",
        "experiment_config_sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
    }
    for name, expected in expected_identity.items():
        if getattr(identity, name) != expected:
            raise ValueError(f"Complete evaluation plan {name} left the frozen study")
    if plan["baseline_config_sha256"] != EXPECTED_BASELINE_CONFIG_SHA256:
        raise ValueError("Complete evaluation plan baseline configuration changed")
    if plan["image_contract_sha256"] != (
        "c0dba1f1a2387bce425b6c33f83e5035d3904ccb62de0e4f1422602ead0cbca8"
    ):
        raise ValueError("Complete evaluation plan image contract changed")
    if (
        type(plan["image_uri"]) is not str
        or _EVALUATION_IMAGE_URI.fullmatch(plan["image_uri"]) is None
    ):
        raise ValueError("Complete evaluation plan image_uri must be one us-east-1 ECR digest")
    case_ids = _exact_strings(plan["case_ids"], name="plan.case_ids")
    for name in ("query_count", "passage_count"):
        if type(plan[name]) is not int or plan[name] < 1:
            raise ValueError(f"Complete evaluation plan {name} must be a positive exact integer")
    exact_lengths = {
        "controlled_max_len_query": 4_096,
        "controlled_max_len_passage": 500,
        "e5_max_len_passage": 500,
        "query_batch_size": 4,
        "passage_batch_size": 38,
    }
    for name, expected in exact_lengths.items():
        if plan[name] != expected or type(plan[name]) is not int:
            raise ValueError(f"Complete evaluation plan {name} changed")
    _validate_scientific_json(plan["runtime_identity"], name="plan.runtime_identity")

    systems = plan["systems"]
    if type(systems) is not list or len(systems) != 15:
        raise ValueError("Complete evaluation plan must contain exactly 15 systems")
    normalized: list[dict[str, Any]] = []
    system_ids: list[str] = []
    controlled_cells: set[tuple[str, str, int]] = set()
    baseline_types: set[str] = set()
    from .artifacts import ControlledArtifactExpectation
    from .baseline_artifacts import (
        E5_MODEL_ID,
        E5_REVISION,
        E5_SNAPSHOT_MANIFEST_SHA256,
        E5_SNAPSHOT_TREE_SHA256,
        FIXED_BASE_ARTIFACT_PROTOCOL,
        FIXED_BASE_SEED,
        MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
        MODERNBERT_SNAPSHOT_TREE_SHA256,
    )
    from .bm25 import (
        ANSERINI_JAR_SHA256,
        BM25_B,
        BM25_K1,
        BM25_RUNTIME_PROTOCOL,
        PYJNIUS_VERSION,
        PYSERINI_VERSION,
    )
    from .e5_pack_artifact import E5_PACK_ARTIFACT_PROTOCOL
    from .query_packing import FOCUS_PRESERVING_PACK_PROTOCOL

    for position, system in enumerate(systems):
        if type(system) is not dict or set(system) != {
            "system_id",
            "system_type",
            "query_view",
            "expectation",
        }:
            raise ValueError(f"Complete evaluation system {position} has an invalid schema")
        system_id = system["system_id"]
        if (
            type(system_id) is not str
            or not system_id
            or system_id.strip() != system_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in system_id
            )
        ):
            raise ValueError(f"Complete evaluation system {position} has an invalid system_id")
        query_view = normalize_query_view(system["query_view"])
        if query_view != system["query_view"]:
            raise ValueError(f"System {system_id!r} query_view is not canonical")
        system_type = system["system_type"]
        expectation = system["expectation"]
        if type(expectation) is not dict:
            raise ValueError(f"System {system_id!r} expectation must be an exact object")
        if system_type == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE:
            expected_keys = {
                "artifact_manifest_sha256",
                "experiment_id",
                "outer_fold",
                "query_view",
                "sampler",
                "experiment_seed",
                "dataset_manifest_sha256",
                "fold_manifest_sha256",
                "passage_index_sha256",
                "model_artifact_protocol",
            }
            if set(expectation) != expected_keys:
                raise ValueError(f"Controlled system {system_id!r} expectation schema changed")
            artifact_expectation = ControlledArtifactExpectation(**expectation)
            expected_system_id = (
                f"{artifact_expectation.query_view}_"
                f"{artifact_expectation.sampler}_seed"
                f"{artifact_expectation.experiment_seed}"
            )
            if (
                system_id != expected_system_id
                or artifact_expectation.outer_fold != identity.outer_fold
                or artifact_expectation.query_view != query_view
                or artifact_expectation.dataset_manifest_sha256
                != identity.dataset_manifest_sha256
                or artifact_expectation.fold_manifest_sha256
                != identity.fold_manifest_sha256
                or artifact_expectation.passage_index_sha256
                != identity.passage_index_sha256
            ):
                raise ValueError(f"Controlled system {system_id!r} left the plan identity")
            controlled_cells.add(
                (
                    artifact_expectation.query_view,
                    artifact_expectation.sampler,
                    artifact_expectation.experiment_seed,
                )
            )
        elif system_type == BM25_SYSTEM_TYPE:
            expected = {
                "baseline_config_sha256": EXPECTED_BASELINE_CONFIG_SHA256,
                "runtime_protocol": BM25_RUNTIME_PROTOCOL,
                "pyserini_version": PYSERINI_VERSION,
                "pyjnius_version": PYJNIUS_VERSION,
                "anserini_jar_sha256": ANSERINI_JAR_SHA256,
                "k1": BM25_K1,
                "b": BM25_B,
            }
            if expectation != expected or query_view != "flat_plain" or system_id != "bm25_flat_plain":
                raise ValueError("BM25 system contract changed")
            baseline_types.add(system_type)
        elif system_type == E5_SYSTEM_TYPE:
            expected = {
                "baseline_config_sha256": EXPECTED_BASELINE_CONFIG_SHA256,
                "model_id": E5_MODEL_ID,
                "revision": E5_REVISION,
                "snapshot_manifest_sha256": E5_SNAPSHOT_MANIFEST_SHA256,
                "snapshot_tree_sha256": E5_SNAPSHOT_TREE_SHA256,
                "pack_artifact_protocol": E5_PACK_ARTIFACT_PROTOCOL,
                "pack_manifest_sha256": EXPECTED_E5_PACK_MANIFEST_SHA256,
                "packed_query_inventory_sha256": EXPECTED_E5_PACK_INVENTORY_SHA256,
                "packing_protocol": FOCUS_PRESERVING_PACK_PROTOCOL,
                "weight_dtype": "float32",
                "attention_implementation": "eager",
                "pooling": "attention_masked_mean_then_l2_normalize_v1",
                "max_positions": 512,
                "max_passage_tokens": 500,
                "passage_truncation": "right",
                "token_type_ids": "explicit_all_zero",
            }
            if (
                expectation != expected
                or query_view != "flat_plain"
                or system_id != "e5_base_v2_flat_plain"
            ):
                raise ValueError("E5 system contract changed")
            baseline_types.add(system_type)
        elif system_type == FIXED_BASE_SYSTEM_TYPE:
            expected = {
                "baseline_config_sha256": EXPECTED_BASELINE_CONFIG_SHA256,
                "artifact_manifest_sha256": EXPECTED_FIXED_BASE_ARTIFACT_MANIFEST_SHA256,
                "model_artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
                "fixed_initialization_seed": FIXED_BASE_SEED,
                "model_sha256": EXPECTED_FIXED_BASE_MODEL_SHA256,
                "new_embedding_rows_sha256": EXPECTED_FIXED_BASE_NEW_ROWS_SHA256,
                "state_key_sha256": EXPECTED_FIXED_BASE_STATE_KEYS_SHA256,
                "snapshot_manifest_sha256": MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
                "snapshot_tree_sha256": MODERNBERT_SNAPSHOT_TREE_SHA256,
                "weight_dtype": "bfloat16",
            }
            if (
                expectation != expected
                or query_view != "flat_masked"
                or system_id != "modernbert_base_flat_masked"
            ):
                raise ValueError("Fixed-base ModernBERT system contract changed")
            baseline_types.add(system_type)
        else:
            raise ValueError(f"Unsupported complete evaluation system_type={system_type!r}")
        system_ids.append(system_id)
        normalized.append(
            {
                "system_id": system_id,
                "system_type": system_type,
                "query_view": query_view,
                "expectation": dict(expectation),
            }
        )
    if system_ids != sorted(system_ids) or len(system_ids) != len(set(system_ids)):
        raise ValueError("Complete evaluation systems must be unique and sorted by system_id")
    expected_cells = {
        (query_view, sampler, seed)
        for query_view in ("flat_masked", "structured")
        for sampler in ("local_unique", "global_uniform")
        for seed in (17, 29, 43)
    }
    if controlled_cells != expected_cells:
        raise ValueError("Complete evaluation plan does not cover the exact 12 controlled cells")
    if baseline_types != {BM25_SYSTEM_TYPE, E5_SYSTEM_TYPE, FIXED_BASE_SYSTEM_TYPE}:
        raise ValueError("Complete evaluation plan baseline inventory changed")
    return identity, case_ids, normalized


def _validate_complete_local_bindings(
    bindings: dict[str, Any],
    *,
    system_plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "dataset_dir",
        "fold_manifest_path",
        "experiment_config_path",
        "baseline_config_path",
        "image_contract_path",
        "bm25_scratch_dir",
        "systems",
    }
    if set(bindings) != expected_keys:
        raise ValueError("Complete local bindings schema changed")
    if (
        bindings["schema_version"] != COMPLETE_LOCAL_BINDINGS_SCHEMA_VERSION
        or type(bindings["schema_version"]) is not int
    ):
        raise ValueError("Complete local bindings schema_version must be exact integer 2")
    paths: dict[str, Path] = {}
    for name in (
        "dataset_dir",
        "fold_manifest_path",
        "experiment_config_path",
        "baseline_config_path",
        "image_contract_path",
        "bm25_scratch_dir",
    ):
        value = bindings[name]
        if type(value) is not str or not value:
            raise ValueError(f"Complete local binding {name} must be an explicit path")
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"Complete local binding {name} must be absolute")
        paths[name] = path
    raw_systems = bindings["systems"]
    if type(raw_systems) is not list or len(raw_systems) != len(system_plans):
        raise ValueError("Complete local system bindings have wrong coverage")
    by_id: dict[str, dict[str, Any]] = {}
    plan_by_id = {plan["system_id"]: plan for plan in system_plans}
    for position, record in enumerate(raw_systems):
        if type(record) is not dict or "system_id" not in record:
            raise ValueError(f"Complete local system binding {position} is invalid")
        system_id = record["system_id"]
        if system_id not in plan_by_id or system_id in by_id:
            raise ValueError(f"Complete local system binding ID is invalid: {system_id!r}")
        system_type = plan_by_id[system_id]["system_type"]
        if system_type in {CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE, FIXED_BASE_SYSTEM_TYPE}:
            expected_record_keys = {"system_id", "artifact_dir"}
            path_fields = ("artifact_dir",)
        elif system_type == E5_SYSTEM_TYPE:
            expected_record_keys = {
                "system_id",
                "snapshot_dir",
                "snapshot_manifest_path",
                "pack_artifact_dir",
            }
            path_fields = ("snapshot_dir", "snapshot_manifest_path", "pack_artifact_dir")
        else:
            expected_record_keys = {"system_id"}
            path_fields = ()
        if set(record) != expected_record_keys:
            raise ValueError(f"Complete local binding schema changed for {system_id!r}")
        normalized_record = {"system_id": system_id}
        for name in path_fields:
            if type(record[name]) is not str or not Path(record[name]).is_absolute():
                raise ValueError(f"Complete local binding {system_id}.{name} must be absolute")
            normalized_record[name] = Path(record[name])
        by_id[system_id] = normalized_record
    if list(by_id) != [plan["system_id"] for plan in system_plans]:
        raise ValueError("Complete local system binding order changed")

    scratch_dir = paths["bm25_scratch_dir"]
    scratch_parent = scratch_dir.parent
    if scratch_dir.exists() or scratch_dir.is_symlink():
        raise FileExistsError("Bound BM25 scratch path must be absent before evaluation")
    if scratch_parent.is_symlink() or not scratch_parent.is_dir():
        raise ValueError("Bound BM25 scratch parent must be a real directory")
    protected_roots = [
        paths["dataset_dir"],
        paths["fold_manifest_path"].parent,
        paths["experiment_config_path"].parent,
        paths["baseline_config_path"].parent,
        paths["image_contract_path"].parent,
    ]
    for record in by_id.values():
        if "artifact_dir" in record:
            protected_roots.append(record["artifact_dir"])
        if "snapshot_dir" in record:
            protected_roots.extend(
                (
                    record["snapshot_dir"],
                    record["snapshot_manifest_path"].parent,
                    record["pack_artifact_dir"],
                )
            )
    normalized_scratch = scratch_dir.resolve(strict=False)
    normalized_roots = tuple(
        sorted(
            {path.resolve(strict=False) for path in protected_roots},
            key=lambda path: path.as_posix(),
        )
    )
    if any(
        normalized_scratch == root
        or normalized_scratch.is_relative_to(root)
        or root.is_relative_to(normalized_scratch)
        for root in normalized_roots
    ):
        raise ValueError("Bound BM25 scratch path overlaps an immutable input root")
    return {
        **paths,
        "systems": by_id,
        "protected_input_roots": normalized_roots,
    }


def _preflight_complete_output(
    output_dir: Path,
    *,
    scratch_dir: Path,
    protected_input_roots: Sequence[Path],
) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        raise ValueError("Complete evaluation output must be an absolute path")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Complete evaluation output must be absent: {output_dir}")
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("Complete evaluation output parent must be a real directory")
    incomplete = parent / f".{output_dir.name}.incomplete"
    if incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError(f"Stale incomplete evaluation output exists: {incomplete}")
    normalized_output = output_dir.resolve(strict=False)
    normalized_scratch = Path(scratch_dir).resolve(strict=False)
    protected = tuple(Path(path).resolve(strict=False) for path in protected_input_roots)
    if (
        normalized_output == normalized_scratch
        or normalized_output.is_relative_to(normalized_scratch)
        or normalized_scratch.is_relative_to(normalized_output)
        or any(
            normalized_output == root
            or normalized_output.is_relative_to(root)
            or root.is_relative_to(normalized_output)
            for root in protected
        )
    ):
        raise ValueError("Complete evaluation output overlaps scratch or an input root")
    return output_dir


def _validate_local_bindings(
    bindings: dict[str, Any],
    *,
    system_ids: Sequence[str],
) -> tuple[Path, Path, Path, dict[str, Path]]:
    expected_keys = {
        "schema_version",
        "dataset_dir",
        "fold_manifest_path",
        "experiment_config_path",
        "systems",
    }
    if set(bindings) != expected_keys:
        raise ValueError("Local evaluation bindings schema changed")
    if bindings["schema_version"] != LOCAL_BINDINGS_SCHEMA_VERSION or type(
        bindings["schema_version"]
    ) is not int:
        raise ValueError("Local bindings schema_version must be exact integer 1")
    for name in ("dataset_dir", "fold_manifest_path", "experiment_config_path"):
        if type(bindings[name]) is not str or not bindings[name]:
            raise ValueError(f"Local binding {name} must be an explicit path string")
    dataset_dir = Path(bindings["dataset_dir"])
    fold_manifest_path = Path(bindings["fold_manifest_path"])
    experiment_config_path = Path(bindings["experiment_config_path"])
    if not dataset_dir.is_absolute() or not fold_manifest_path.is_absolute() or not (
        experiment_config_path.is_absolute()
    ):
        raise ValueError("Every local evaluation binding path must be absolute")
    raw_systems = bindings["systems"]
    if type(raw_systems) is not list:
        raise ValueError("Local system bindings must be a list")
    artifact_dir_by_system_id: dict[str, Path] = {}
    for position, record in enumerate(raw_systems):
        if type(record) is not dict or set(record) != {"system_id", "artifact_dir"}:
            raise ValueError(f"Local system binding {position} has an invalid schema")
        system_id = record["system_id"]
        artifact_dir = record["artifact_dir"]
        if type(system_id) is not str or type(artifact_dir) is not str:
            raise ValueError("Local system binding values must be exact strings")
        if system_id in artifact_dir_by_system_id:
            raise ValueError(f"Duplicate local system binding={system_id!r}")
        path = Path(artifact_dir)
        if not path.is_absolute():
            raise ValueError("Local artifact directories must be absolute")
        artifact_dir_by_system_id[system_id] = path
    if list(artifact_dir_by_system_id) != list(system_ids):
        raise ValueError("Local system binding order/coverage changed")
    return (
        dataset_dir,
        fold_manifest_path,
        experiment_config_path,
        artifact_dir_by_system_id,
    )


def _validate_controlled_artifact_uniqueness(artifacts: Sequence[Any]) -> None:
    if not artifacts:
        raise ValueError("Controlled artifact preflight cannot be empty")
    artifact_manifest_hashes: set[str] = set()
    training_cells: set[tuple[str, str, int]] = set()
    for artifact in artifacts:
        identity = artifact.identity
        manifest_hash = identity.artifact_manifest_sha256
        training_cell = (
            identity.query_view,
            identity.sampler,
            identity.experiment_seed,
        )
        if manifest_hash in artifact_manifest_hashes:
            raise ValueError("Evaluation plan aliases one controlled artifact more than once")
        if training_cell in training_cells:
            raise ValueError(
                "Evaluation plan contains duplicate controlled training cells: "
                f"{training_cell}"
            )
        artifact_manifest_hashes.add(manifest_hash)
        training_cells.add(training_cell)


def run_local_controlled_evaluation_plan(
    *,
    evaluation_plan_path: Path,
    local_bindings_path: Path,
    output_dir: Path,
    device: str,
) -> Mapping[str, Any]:
    """Run exact local controlled artifacts; AWS staging and baselines are later layers."""

    evaluation_plan_path = Path(evaluation_plan_path)
    local_bindings_path = Path(local_bindings_path)
    plan, plan_sha256 = _load_exact_json_file_with_sha256(
        evaluation_plan_path,
        name="evaluation plan",
        canonical=True,
    )
    identity, case_ids, system_plans = _validate_controlled_evaluation_plan(
        plan,
        evaluation_plan_sha256=plan_sha256,
    )
    bindings = _load_exact_json_file(
        local_bindings_path,
        name="local bindings",
        canonical=True,
    )
    (
        dataset_dir,
        fold_manifest_path,
        experiment_config_path,
        artifact_dir_by_system_id,
    ) = _validate_local_bindings(
        bindings,
        system_ids=[system["system_id"] for system in system_plans],
    )
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError(f"Bound dataset directory must be real: {dataset_dir}")
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    if (
        dataset_manifest_path.is_symlink()
        or not dataset_manifest_path.is_file()
        or _sha256_file(dataset_manifest_path) != identity.dataset_manifest_sha256
    ):
        raise ValueError("Bound corrected dataset manifest hash changed")
    if (
        fold_manifest_path.is_symlink()
        or not fold_manifest_path.is_file()
        or _sha256_file(fold_manifest_path) != identity.fold_manifest_sha256
    ):
        raise ValueError("Bound fold manifest hash changed")
    if (
        experiment_config_path.is_symlink()
        or not experiment_config_path.is_file()
        or _sha256_file(experiment_config_path) != identity.experiment_config_sha256
    ):
        raise ValueError("Bound experiment configuration hash changed")

    fold_manifest = validate_staged_dataset_and_fold(
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
        expected_dataset_manifest_sha256=identity.dataset_manifest_sha256,
        expected_fold_manifest_sha256=identity.fold_manifest_sha256,
        expected_dataset_manifest_logical_path=(
            EXPECTED_DATASET_MANIFEST_LOGICAL_PATH
        ),
    )
    rotation = fold_manifest["rotations"][identity.outer_fold]
    role_record = rotation[identity.role]
    if (
        rotation["outer_fold"] != identity.outer_fold
        or role_record["case_ids"] != list(case_ids)
        or role_record["queries"] != plan["query_count"]
        or role_record["passages"] != plan["passage_count"]
    ):
        raise ValueError("Evaluation plan role inventory disagrees with the frozen fold")

    corpus_by_passage_id = load_corpus(dataset_dir)
    all_queries = load_queries(dataset_dir, "all")
    if len(corpus_by_passage_id) != 5_286 or len(all_queries) != 490:
        raise RuntimeError("Controlled evaluator requires the exact 5,286/490 corrected data")
    passage_index_table = PassageIndexTable(corpus_by_passage_id)
    if passage_index_table.sha256 != identity.passage_index_sha256:
        raise RuntimeError("Bound corrected corpus passage-index hash changed")
    fold_global_data = build_canonical_evaluation_data(
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        evaluated_case_ids=case_ids,
        role=identity.role,
        regime_name="fold_global",
    )
    if (
        fold_global_data.query_count != plan["query_count"]
        or fold_global_data.passage_count != plan["passage_count"]
    ):
        raise RuntimeError("Controlled evaluation data counts changed")

    from .artifacts import (
        ControlledArtifactExpectation,
        import_pinned_artifact_runtime,
        load_controlled_retriever,
        validate_controlled_artifact,
    )

    runtime = import_pinned_artifact_runtime()
    runtime_identity = _runtime_identity_for_controlled_artifacts(
        device=device,
        runtime=runtime,
    )
    if plan["runtime_identity"] != runtime_identity:
        raise RuntimeError(
            "Actual evaluation runtime differs from the immutable plan: "
            f"actual={runtime_identity}, expected={plan['runtime_identity']}"
        )
    source_query_by_id = {query.query_id: query for query in all_queries}
    query_ids = tuple(query.query_id for query in fold_global_data.queries)
    passage_ids = fold_global_data.passage_ids
    passage_texts = tuple(corpus_by_passage_id[passage_id].text for passage_id in passage_ids)
    validated_artifact_by_system_id = {}
    for system_plan in system_plans:
        expectation = ControlledArtifactExpectation(
            **system_plan["artifact_expectation"]
        )
        artifact = validate_controlled_artifact(
            artifact_dir_by_system_id[system_plan["system_id"]],
            expectation=expectation,
        )
        validated_artifact_by_system_id[system_plan["system_id"]] = artifact
    _validate_controlled_artifact_uniqueness(
        tuple(validated_artifact_by_system_id.values())
    )

    system_scores: list[SystemScoreInput] = []
    for system_plan in system_plans:
        artifact = validated_artifact_by_system_id[system_plan["system_id"]]
        loaded = load_controlled_retriever(
            artifact,
            device=device,
            runtime=runtime,
        )
        query_texts = tuple(
            select_query_text(
                source_query_by_id[query_id],
                query_view=system_plan["query_view"],
            )
            for query_id in query_ids
        )
        scores = score_loaded_dual_encoder(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            query_ids=query_ids,
            query_texts=query_texts,
            passage_ids=passage_ids,
            passage_texts=passage_texts,
            slot_token_id=artifact.slot_token_id,
            query_batch_size=plan["query_batch_size"],
            passage_batch_size=plan["passage_batch_size"],
            max_len_query=plan["max_len_query"],
            max_len_passage=plan["max_len_passage"],
            device=device,
            torch_module=runtime.torch_module,
        )
        system_scores.append(
            SystemScoreInput(
                system_id=system_plan["system_id"],
                system_type=system_plan["system_type"],
                query_view=system_plan["query_view"],
                model_identity=asdict(artifact.identity),
                scores=scores,
            )
        )
        del loaded
        del artifact
        gc.collect()
        if runtime.torch_module.cuda.is_available():
            runtime.torch_module.cuda.empty_cache()

    bundle = build_canonical_evaluation_bundle(
        identity=identity,
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        evaluated_case_ids=case_ids,
        systems=tuple(system_scores),
        runtime_identity=runtime_identity,
        torch_module=runtime.torch_module,
    )
    return publish_and_validate_canonical_evaluation_bundle(
        bundle,
        output_dir=Path(output_dir),
        identity=identity,
        runtime_identity=runtime_identity,
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        expected_system_contract=tuple(
            (
                system_plan["system_id"],
                system_plan["system_type"],
                system_plan["query_view"],
            )
            for system_plan in system_plans
        ),
    )


def run_complete_evaluation_plan(
    *,
    evaluation_plan_path: Path,
    local_bindings_path: Path,
    output_dir: Path,
    device: str,
) -> Mapping[str, Any]:
    """Run the exact 12 controlled systems and three frozen baselines serially."""

    evaluation_plan_path = Path(evaluation_plan_path)
    local_bindings_path = Path(local_bindings_path)
    if not evaluation_plan_path.is_absolute() or not local_bindings_path.is_absolute():
        raise ValueError("Complete evaluation plan and bindings paths must be absolute")
    plan, plan_sha256 = _load_exact_json_file_with_sha256(
        evaluation_plan_path,
        name="complete evaluation plan",
        canonical=True,
    )
    identity, case_ids, system_plans = _validate_complete_evaluation_plan(
        plan,
        evaluation_plan_sha256=plan_sha256,
    )
    bindings = _load_exact_json_file(
        local_bindings_path,
        name="complete local bindings",
        canonical=True,
    )
    bound = _validate_complete_local_bindings(bindings, system_plans=system_plans)
    output_dir = _preflight_complete_output(
        Path(output_dir),
        scratch_dir=bound["bm25_scratch_dir"],
        protected_input_roots=bound["protected_input_roots"],
    )
    fixed_inputs = {
        "dataset_manifest": (
            bound["dataset_dir"] / "dataset_manifest.json",
            identity.dataset_manifest_sha256,
        ),
        "fold_manifest": (bound["fold_manifest_path"], identity.fold_manifest_sha256),
        "experiment_config": (
            bound["experiment_config_path"],
            identity.experiment_config_sha256,
        ),
        "baseline_config": (
            bound["baseline_config_path"],
            EXPECTED_BASELINE_CONFIG_SHA256,
        ),
        "image_contract": (
            bound["image_contract_path"],
            plan["image_contract_sha256"],
        ),
    }
    for name, (path, expected_sha256) in fixed_inputs.items():
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"Bound complete-evaluation {name} bytes changed")
    if bound["dataset_dir"].is_symlink() or not bound["dataset_dir"].is_dir():
        raise ValueError("Bound complete-evaluation dataset directory is not real")
    image_contract = _load_exact_json_file(
        bound["image_contract_path"],
        name="Processing image contract",
        canonical=True,
    )
    service_image_uri = _validate_processing_job_image_and_command(
        config_path=PROCESSING_JOB_CONFIG_PATH,
        expected_image_uri=plan["image_uri"],
        expected_entrypoint=image_contract["entrypoint"],
        evaluation_plan_path=evaluation_plan_path,
        local_bindings_path=local_bindings_path,
        output_dir=output_dir,
        device=device,
    )

    fold_manifest = validate_staged_dataset_and_fold(
        dataset_dir=bound["dataset_dir"],
        fold_manifest_path=bound["fold_manifest_path"],
        expected_dataset_manifest_sha256=identity.dataset_manifest_sha256,
        expected_fold_manifest_sha256=identity.fold_manifest_sha256,
        expected_dataset_manifest_logical_path=EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
    )
    rotation = fold_manifest["rotations"][identity.outer_fold]
    role_record = rotation[identity.role]
    if (
        rotation["outer_fold"] != identity.outer_fold
        or role_record["case_ids"] != list(case_ids)
        or role_record["queries"] != plan["query_count"]
        or role_record["passages"] != plan["passage_count"]
    ):
        raise ValueError("Complete evaluation plan role inventory disagrees with the frozen fold")

    corpus_by_passage_id = load_corpus(bound["dataset_dir"])
    all_queries = load_queries(bound["dataset_dir"], "all")
    if len(corpus_by_passage_id) != 5_286 or len(all_queries) != 490:
        raise RuntimeError("Complete evaluator requires exact 5,286/490 corrected data")
    passage_index_table = PassageIndexTable(corpus_by_passage_id)
    if passage_index_table.sha256 != identity.passage_index_sha256:
        raise RuntimeError("Complete evaluator passage-index hash changed")
    fold_global_data = build_canonical_evaluation_data(
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        evaluated_case_ids=case_ids,
        role=identity.role,
        regime_name="fold_global",
    )
    if (
        fold_global_data.query_count != plan["query_count"]
        or fold_global_data.passage_count != plan["passage_count"]
    ):
        raise RuntimeError("Complete evaluation fold-global counts changed")

    from processing_eval.image_smoke import validate_image_runtime

    image_runtime_identity = validate_image_runtime(bound["image_contract_path"])
    runtime_identity = {
        "device": device,
        "image_uri": service_image_uri,
        **image_runtime_identity,
    }
    if plan["runtime_identity"] != runtime_identity:
        raise RuntimeError(
            "Actual Processing runtime differs from the immutable plan: "
            f"actual={runtime_identity}, expected={plan['runtime_identity']}"
        )

    from .artifacts import (
        ControlledArtifactExpectation,
        import_pinned_artifact_runtime,
        load_controlled_retriever,
        validate_controlled_artifact,
    )
    from .baseline_artifacts import (
        E5_MODEL_ID,
        E5_REVISION,
        E5_SNAPSHOT_MANIFEST_SHA256,
        E5_SNAPSHOT_TREE_SHA256,
        FixedBaseArtifactExpectation,
        load_e5_encoder,
        load_fixed_base_retriever,
        validate_fixed_base_artifact,
        validate_snapshot,
    )
    from .bm25 import BM25_B, BM25_K1, build_and_score_bm25
    from .e5_pack_artifact import validate_e5_pack_artifact
    from .rankers import score_loaded_e5_encoder

    runtime = import_pinned_artifact_runtime()
    source_query_by_id = {query.query_id: query for query in all_queries}
    query_ids = tuple(query.query_id for query in fold_global_data.queries)
    passage_ids = fold_global_data.passage_ids
    passage_texts = tuple(
        corpus_by_passage_id[passage_id].text for passage_id in passage_ids
    )
    plan_by_type = {
        system_type: [plan for plan in system_plans if plan["system_type"] == system_type]
        for system_type in (
            BM25_SYSTEM_TYPE,
            E5_SYSTEM_TYPE,
            FIXED_BASE_SYSTEM_TYPE,
            CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
        )
    }
    bm25_plan = plan_by_type[BM25_SYSTEM_TYPE][0]
    e5_plan = plan_by_type[E5_SYSTEM_TYPE][0]
    fixed_base_plan = plan_by_type[FIXED_BASE_SYSTEM_TYPE][0]
    local_systems = bound["systems"]

    validated_controlled = {}
    for system_plan in plan_by_type[CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE]:
        expectation = ControlledArtifactExpectation(**system_plan["expectation"])
        validated_controlled[system_plan["system_id"]] = validate_controlled_artifact(
            local_systems[system_plan["system_id"]]["artifact_dir"],
            expectation=expectation,
        )
    _validate_controlled_artifact_uniqueness(tuple(validated_controlled.values()))
    fixed_expectation = FixedBaseArtifactExpectation(
        artifact_manifest_sha256=(
            fixed_base_plan["expectation"]["artifact_manifest_sha256"]
        ),
        baseline_config_sha256=plan["baseline_config_sha256"],
    )
    validated_fixed_base = validate_fixed_base_artifact(
        local_systems[fixed_base_plan["system_id"]]["artifact_dir"],
        expectation=fixed_expectation,
    )

    e5_binding = local_systems[e5_plan["system_id"]]
    e5_snapshot_identity = validate_snapshot(
        snapshot_dir=e5_binding["snapshot_dir"],
        manifest_path=e5_binding["snapshot_manifest_path"],
        expected_manifest_sha256=E5_SNAPSHOT_MANIFEST_SHA256,
        expected_model_id=E5_MODEL_ID,
        expected_revision=E5_REVISION,
        expected_tree_sha256=E5_SNAPSHOT_TREE_SHA256,
    )
    e5_tokenizer = runtime.auto_tokenizer_class.from_pretrained(
        str(e5_binding["snapshot_dir"]),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    validated_pack = validate_e5_pack_artifact(
        e5_binding["pack_artifact_dir"],
        expected_manifest_sha256=e5_plan["expectation"]["pack_manifest_sha256"],
        queries=all_queries,
        tokenizer=e5_tokenizer,
    )
    if (
        validated_pack.packed_query_inventory_sha256
        != e5_plan["expectation"]["packed_query_inventory_sha256"]
    ):
        raise RuntimeError("Validated E5 packed-query inventory changed")

    score_by_system_id: dict[str, Any] = {}
    model_identity_by_system_id: dict[str, dict[str, Any]] = {}

    bm25_query_texts = tuple(
        select_query_text(source_query_by_id[query_id], query_view="flat_plain")
        for query_id in query_ids
    )
    score_by_system_id[bm25_plan["system_id"]] = build_and_score_bm25(
        query_ids=query_ids,
        query_texts=bm25_query_texts,
        passage_ids=passage_ids,
        passage_texts=passage_texts,
        scratch_dir=bound["bm25_scratch_dir"],
        torch_module=runtime.torch_module,
    )
    model_identity_by_system_id[bm25_plan["system_id"]] = {
        **bm25_plan["expectation"],
        "query_view": "flat_plain",
        "zero_tail": "float32_zero_then_canonical_stable_rank_v1",
        "index_document_count": len(passage_ids),
        "k1": BM25_K1,
        "b": BM25_B,
    }

    packed_by_query_id = {
        packed.query_id: packed for packed in validated_pack.packed_queries
    }
    loaded_e5 = load_e5_encoder(
        snapshot_dir=e5_binding["snapshot_dir"],
        manifest_path=e5_binding["snapshot_manifest_path"],
        device=device,
        runtime=runtime,
    )
    if loaded_e5.snapshot_identity != e5_snapshot_identity:
        raise RuntimeError("Loaded E5 snapshot differs from complete preflight")
    score_by_system_id[e5_plan["system_id"]] = score_loaded_e5_encoder(
        model=loaded_e5.model,
        tokenizer=loaded_e5.tokenizer,
        query_ids=query_ids,
        packed_query_input_ids=tuple(
            packed_by_query_id[query_id].input_ids for query_id in query_ids
        ),
        passage_ids=passage_ids,
        passage_texts=passage_texts,
        passage_prefix="passage: ",
        query_batch_size=plan["query_batch_size"],
        passage_batch_size=plan["passage_batch_size"],
        max_len_passage=plan["e5_max_len_passage"],
        device=device,
        torch_module=runtime.torch_module,
    )
    model_identity_by_system_id[e5_plan["system_id"]] = {
        **e5_plan["expectation"],
        "snapshot_model_sha256": next(
            sha256
            for path, _, sha256 in loaded_e5.snapshot_identity.files
            if path == "model.safetensors"
        ),
    }
    del loaded_e5
    del e5_tokenizer
    gc.collect()
    if runtime.torch_module.cuda.is_available():
        runtime.torch_module.cuda.empty_cache()

    loaded_fixed_base = load_fixed_base_retriever(
        validated_fixed_base,
        device=device,
        runtime=runtime,
    )
    fixed_query_texts = tuple(
        select_query_text(source_query_by_id[query_id], query_view="flat_masked")
        for query_id in query_ids
    )
    score_by_system_id[fixed_base_plan["system_id"]] = score_loaded_dual_encoder(
        model=loaded_fixed_base.model,
        tokenizer=loaded_fixed_base.tokenizer,
        query_ids=query_ids,
        query_texts=fixed_query_texts,
        passage_ids=passage_ids,
        passage_texts=passage_texts,
        slot_token_id=validated_fixed_base.slot_token_id,
        query_batch_size=plan["query_batch_size"],
        passage_batch_size=plan["passage_batch_size"],
        max_len_query=plan["controlled_max_len_query"],
        max_len_passage=plan["controlled_max_len_passage"],
        device=device,
        torch_module=runtime.torch_module,
    )
    model_identity_by_system_id[fixed_base_plan["system_id"]] = {
        **fixed_base_plan["expectation"],
        "artifact_manifest_sha256": validated_fixed_base.manifest_sha256,
    }
    del loaded_fixed_base
    del validated_fixed_base
    gc.collect()
    if runtime.torch_module.cuda.is_available():
        runtime.torch_module.cuda.empty_cache()

    for system_plan in plan_by_type[CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE]:
        artifact = validated_controlled[system_plan["system_id"]]
        loaded = load_controlled_retriever(artifact, device=device, runtime=runtime)
        query_texts = tuple(
            select_query_text(
                source_query_by_id[query_id],
                query_view=system_plan["query_view"],
            )
            for query_id in query_ids
        )
        score_by_system_id[system_plan["system_id"]] = score_loaded_dual_encoder(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            query_ids=query_ids,
            query_texts=query_texts,
            passage_ids=passage_ids,
            passage_texts=passage_texts,
            slot_token_id=artifact.slot_token_id,
            query_batch_size=plan["query_batch_size"],
            passage_batch_size=plan["passage_batch_size"],
            max_len_query=plan["controlled_max_len_query"],
            max_len_passage=plan["controlled_max_len_passage"],
            device=device,
            torch_module=runtime.torch_module,
        )
        model_identity_by_system_id[system_plan["system_id"]] = asdict(
            artifact.identity
        )
        del loaded
        gc.collect()
        if runtime.torch_module.cuda.is_available():
            runtime.torch_module.cuda.empty_cache()

    expected_ids = [system_plan["system_id"] for system_plan in system_plans]
    if set(score_by_system_id) != set(expected_ids) or set(model_identity_by_system_id) != set(
        expected_ids
    ):
        raise RuntimeError("Complete evaluator did not score every planned system exactly once")
    system_scores = tuple(
        SystemScoreInput(
            system_id=system_plan["system_id"],
            system_type=system_plan["system_type"],
            query_view=system_plan["query_view"],
            model_identity=model_identity_by_system_id[system_plan["system_id"]],
            scores=score_by_system_id[system_plan["system_id"]],
        )
        for system_plan in system_plans
    )
    bundle = build_canonical_evaluation_bundle(
        identity=identity,
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        evaluated_case_ids=case_ids,
        systems=system_scores,
        runtime_identity=runtime_identity,
        torch_module=runtime.torch_module,
    )
    return publish_and_validate_canonical_evaluation_bundle(
        bundle,
        output_dir=output_dir,
        identity=identity,
        runtime_identity=runtime_identity,
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        expected_system_contract=tuple(
            (
                system_plan["system_id"],
                system_plan["system_type"],
                system_plan["query_view"],
            )
            for system_plan in system_plans
        ),
    )
