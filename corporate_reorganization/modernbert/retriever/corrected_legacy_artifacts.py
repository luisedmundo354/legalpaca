"""Strict readback validator for corrected legacy-style diagnostic artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.retrieval_cv.corrected_legacy_config import (
    CORRECTED_LEGACY_CONFIG_SHA256,
    load_corrected_legacy_config,
)
from .corrected_legacy_evaluation import (
    CORRECTED_LEGACY_TEST_REGIMES,
    build_corrected_legacy_test_data,
    build_corrected_legacy_validation_evidence_data,
)
from .artifacts import CONTROLLED_TOKENIZER_SIZE
from .batching import DUMMY_QUERY_INDEX
from .data import PassageIndexTable, load_corpus, load_queries
from .determinism_artifacts import _canonical_bf16_safetensors_identity
from .evaluation import canonical_result_from_payload
from .legacy_diagnostic_batching import CorrectedLegacyQueryBatchSampler
from .legacy_diagnostic_data import load_corrected_legacy_data
from .legacy_diagnostic_sampling import (
    CorrectedLegacyDiagnosticDataset,
    validate_legacy_diagnostic_trace,
)
from .markup import SLOT_TOKEN
from .provenance import (
    EXPECTED_BASE_TRAINING_IMAGE,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_DERIVED_TRAINING_IMAGE,
    EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
    EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
)
from .staged_data import validate_staged_dataset


EXPECTED_TOP_LEVEL = {
    "artifact_manifest.json",
    "candidate_traces",
    "corrected_legacy_run.json",
    "encoder_config",
    "evaluation",
    "inputs",
    "model.safetensors",
    "setting_explanation.md",
    "tokenizer",
    "validation",
    "wrapper_config.json",
}

EXPECTED_TOKENIZER_FILES = {
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
EXPECTED_ENCODER_CONFIG_FILES = {"config.json"}


@dataclass(frozen=True)
class CorrectedLegacyArtifactExpectation:
    artifact_manifest_sha256: str
    run_id: str
    query_view: str
    training_plan_sha256: str
    training_staging_receipt_sha256: str
    training_request_payload_sha256: str
    source_bundle_name: str
    source_bundle_size: int
    source_bundle_sha256: str
    source_bundle_inventory_sha256: str
    source_bundle_commit_epoch: int

    def __post_init__(self) -> None:
        for field in (
            "artifact_manifest_sha256",
            "training_plan_sha256",
            "training_staging_receipt_sha256",
            "training_request_payload_sha256",
            "source_bundle_sha256",
            "source_bundle_inventory_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"Corrected legacy expectation {field} is not lowercase SHA-256")
        expected_coordinates = {
            "corrected-legacy-flat": "flat_masked",
            "corrected-legacy-structured": "structured",
        }
        if expected_coordinates.get(self.run_id) != self.query_view:
            raise ValueError("Corrected legacy expectation run/view coordinates changed")
        if (
            self.source_bundle_name != f"source-{self.source_bundle_sha256}.tar.gz"
            or type(self.source_bundle_size) is not int
            or self.source_bundle_size < 1
            or type(self.source_bundle_commit_epoch) is not int
            or self.source_bundle_commit_epoch < 1
        ):
            raise ValueError("Corrected legacy expected source-bundle identity is invalid")

    def source_bundle_payload(self) -> dict[str, Any]:
        return {
            "commit_epoch": self.source_bundle_commit_epoch,
            "inventory_sha256": self.source_bundle_inventory_sha256,
            "name": self.source_bundle_name,
            "sha256": self.source_bundle_sha256,
            "size": self.source_bundle_size,
        }


@dataclass(frozen=True)
class CorrectedLegacyArtifact:
    root: Path
    artifact_manifest_sha256: str
    run_id: str
    model_sha256: str
    query_view: str
    trace_merged_sha256: str
    validation_records_sha256: str
    evaluation_results_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _load_json(path: Path, *, canonical_indent: int | None = 2) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"Corrected legacy JSON must be a non-empty regular file: {path}")
    raw = path.read_text(encoding="utf-8")
    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate corrected legacy JSON key: {key}")
            result[key] = item
        return result

    value = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON value {token}")
        ),
    )
    if type(value) is not dict:
        raise TypeError(f"Corrected legacy JSON must contain one object: {path}")
    if raw != _canonical_json(value, indent=canonical_indent) + "\n":
        raise ValueError(f"Corrected legacy JSON is not canonical: {path}")
    return value


def _load_bound_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"{name} must be a non-empty regular JSON file")

    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate {name} JSON key: {key}")
            result[key] = item
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite {name} JSON value {token}")
        ),
    )
    if type(value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _validate_tokenizer_payload(root: Path, *, slot_token_id: int) -> None:
    payload = _load_bound_json_object(root / "tokenizer/tokenizer.json", name="tokenizer")
    model = payload.get("model")
    added_tokens = payload.get("added_tokens")
    if type(model) is not dict or type(model.get("vocab")) is not dict:
        raise ValueError("Corrected legacy tokenizer model vocabulary is malformed")
    if type(added_tokens) is not list:
        raise ValueError("Corrected legacy tokenizer added-token inventory is malformed")
    id_by_token: dict[str, int] = {}
    token_by_id: dict[int, str] = {}

    def add_token(token: object, token_id: object) -> None:
        if (
            type(token) is not str
            or not token
            or type(token_id) is not int
            or token_id < 0
        ):
            raise ValueError("Corrected legacy tokenizer contains an invalid token/ID pair")
        if token in id_by_token and id_by_token[token] != token_id:
            raise ValueError("Corrected legacy tokenizer maps one token to multiple IDs")
        if token_id in token_by_id and token_by_id[token_id] != token:
            raise ValueError("Corrected legacy tokenizer maps one ID to multiple tokens")
        id_by_token[token] = token_id
        token_by_id[token_id] = token

    for token, token_id in model["vocab"].items():
        add_token(token, token_id)
    for record in added_tokens:
        if type(record) is not dict or "content" not in record or "id" not in record:
            raise ValueError("Corrected legacy tokenizer added-token record is malformed")
        add_token(record["content"], record["id"])
    if set(token_by_id) != set(range(CONTROLLED_TOKENIZER_SIZE)):
        raise ValueError("Corrected legacy tokenizer ID inventory changed")
    if id_by_token.get(SLOT_TOKEN) != slot_token_id:
        raise ValueError("Corrected legacy tokenizer and wrapper slot-token IDs disagree")


def _validate_tree(root: Path) -> None:
    root = Path(os.path.abspath(root))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Corrected legacy artifact root must be a real directory")
    if {entry.name for entry in root.iterdir()} != EXPECTED_TOP_LEVEL:
        raise ValueError("Corrected legacy artifact top-level inventory changed")
    for entry in root.rglob("*"):
        metadata = os.lstat(entry)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Corrected legacy artifact contains symlink: {entry}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_size < 1:
                raise ValueError(f"Corrected legacy artifact contains empty file: {entry}")
            if entry.name.startswith(".") or entry.name.endswith((".tmp", ".partial", ".incomplete")):
                raise ValueError(f"Corrected legacy artifact contains temporary file: {entry}")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Corrected legacy artifact contains non-file entry: {entry}")


def _validate_file_record(
    root: Path,
    value: object,
    *,
    expected_path: str,
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"path", "size", "sha256"}:
        raise ValueError(f"{name} must be one exact file record")
    if value["path"] != expected_path:
        raise ValueError(f"{name} path changed")
    path = root / expected_path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} target is not a regular file")
    if (
        type(value["size"]) is not int
        or value["size"] != path.stat().st_size
        or type(value["sha256"]) is not str
        or value["sha256"] != _sha256(path)
    ):
        raise ValueError(f"{name} size or SHA-256 changed")
    return value


def _validate_directory_record(
    root: Path,
    value: object,
    *,
    name: str,
    expected_files: set[str],
) -> None:
    if type(value) is not dict or set(value) != {"path", "files"}:
        raise ValueError(f"{name} directory record schema changed")
    directory = root / value["path"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{name} directory is missing")
    files = value["files"]
    if type(files) is not list or not files:
        raise ValueError(f"{name} directory inventory is empty")
    actual = [
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    if set(actual) != expected_files:
        raise ValueError(f"{name} exact file inventory changed")
    if [record.get("path") for record in files if type(record) is dict] != actual:
        raise ValueError(f"{name} directory inventory changed")
    for record in files:
        _validate_file_record(directory, record, expected_path=record["path"], name=name)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_trace_runtime_positions(
    dataset: CorrectedLegacyDiagnosticDataset,
    *,
    epoch: int,
) -> dict[str, tuple[int, int, int]]:
    query_ids = tuple(query.query_id for query in dataset.queries)
    batcher = CorrectedLegacyQueryBatchSampler(
        query_ids,
        experiment_seed=17,
        world_size=4,
        per_device_batch_size=4,
    )
    batcher.set_epoch(epoch)
    raw_batches = batcher.batches()
    positions: dict[str, tuple[int, int, int]] = {}
    for prepared_microbatch_index in range(27):
        for rank in range(4):
            raw_batch = raw_batches[prepared_microbatch_index * 4 + rank]
            real_indices = [
                index for index in raw_batch if index != DUMMY_QUERY_INDEX
            ]
            for local_row, query_index in enumerate(real_indices):
                query_id = query_ids[query_index]
                if query_id in positions:
                    raise RuntimeError("Corrected legacy deterministic batch replay duplicated a query")
                positions[query_id] = (rank, prepared_microbatch_index, local_row)
    if set(positions) != set(query_ids):
        raise RuntimeError("Corrected legacy deterministic batch replay lost a query")
    return positions


def _validate_trace_bundle(
    root: Path,
    *,
    dataset: CorrectedLegacyDiagnosticDataset,
) -> tuple[str, int]:
    expected_query_ids = {query.query_id for query in dataset.queries}
    query_index_by_id = {
        query.query_id: index for index, query in enumerate(dataset.queries)
    }
    directory = root / "candidate_traces"
    if {path.name for path in directory.iterdir()} != {
        "manifest.json",
        "sampling_traces.jsonl",
        *{f"rank-{rank:05d}.jsonl" for rank in range(4)},
    }:
        raise ValueError("Corrected legacy trace directory inventory changed")
    manifest = _load_json(directory / "manifest.json")
    if set(manifest) != {
        "schema_version", "record_count", "epochs", "queries_per_epoch",
        "merge_order", "shards", "merged",
    }:
        raise ValueError("Corrected legacy trace manifest schema changed")
    if (
        manifest["schema_version"] != 1
        or manifest["record_count"] != 8_360
        or manifest["epochs"] != 20
        or manifest["queries_per_epoch"] != 418
        or manifest["merge_order"] != ["epoch", "query_id"]
    ):
        raise ValueError("Corrected legacy trace manifest constants changed")
    shards = manifest["shards"]
    if type(shards) is not list or len(shards) != 4:
        raise ValueError("Corrected legacy trace manifest must contain four shards")
    shard_values: list[dict[str, Any]] = []
    for rank, record in enumerate(shards):
        if type(record) is not dict or set(record) != {
            "rank", "record_count", "query_counts_by_epoch", "sha256", "path", "size"
        }:
            raise ValueError("Corrected legacy trace shard record schema changed")
        expected_per_epoch = 106 if rank == 0 else 104
        if (
            record["rank"] != rank
            or record["record_count"] != expected_per_epoch * 20
            or record["query_counts_by_epoch"] != [expected_per_epoch] * 20
        ):
            raise ValueError("Corrected legacy trace shard coverage changed")
        shard_file = _validate_file_record(
            directory,
            {key: record[key] for key in ("path", "size", "sha256")},
            expected_path=f"rank-{rank:05d}.jsonl",
            name=f"trace shard {rank}",
        )
        counts = [0] * 20
        with (directory / shard_file["path"]).open("r", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if line != _canonical_json(value) + "\n":
                    raise ValueError("Corrected legacy trace shard row is not canonical")
                if type(value) is not dict or value.get("rank") != rank:
                    raise ValueError("Corrected legacy trace shard row rank changed")
                trace = value.get("trace")
                validate_legacy_diagnostic_trace(trace)
                counts[trace["epoch"]] += 1
                shard_values.append(value)
        if counts != [expected_per_epoch] * 20:
            raise ValueError("Corrected legacy trace shard rows disagree with metadata")
    merged_record = _validate_file_record(
        directory,
        manifest["merged"],
        expected_path="sampling_traces.jsonl",
        name="merged corrected legacy traces",
    )
    merged_path = directory / "sampling_traces.jsonl"
    coverage = {epoch: set() for epoch in range(20)}
    previous: tuple[int, str] | None = None
    replay_epoch: int | None = None
    replay_positions: dict[str, tuple[int, int, int]] = {}
    count = 0
    merged_values: list[dict[str, Any]] = []
    with merged_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError("Corrected legacy merged trace lacks final newlines")
            record = json.loads(line)
            if line != _canonical_json(record) + "\n":
                raise ValueError("Corrected legacy merged trace row is not canonical")
            if type(record) is not dict or set(record) != {
                "rank", "prepared_microbatch_index", "local_row", "trace"
            }:
                raise ValueError("Corrected legacy merged trace schema changed")
            if (
                type(record["rank"]) is not int
                or record["rank"] not in range(4)
                or type(record["prepared_microbatch_index"]) is not int
                or record["prepared_microbatch_index"] not in range(27)
                or type(record["local_row"]) is not int
                or record["local_row"] not in range(4)
            ):
                raise ValueError("Corrected legacy trace runtime position changed")
            trace = record["trace"]
            validate_legacy_diagnostic_trace(trace)
            key = (trace["epoch"], trace["query_id"])
            if previous is not None and key <= previous:
                raise ValueError("Corrected legacy merged traces are not strictly ordered")
            previous = key
            if trace["query_id"] in coverage[trace["epoch"]]:
                raise ValueError("Corrected legacy merged traces duplicate a query")
            coverage[trace["epoch"]].add(trace["query_id"])
            if trace["epoch"] != replay_epoch:
                replay_epoch = trace["epoch"]
                dataset.set_epoch(replay_epoch)
                replay_positions = _expected_trace_runtime_positions(
                    dataset,
                    epoch=replay_epoch,
                )
            expected_position = replay_positions[trace["query_id"]]
            actual_position = (
                record["rank"],
                record["prepared_microbatch_index"],
                record["local_row"],
            )
            if actual_position != expected_position:
                raise ValueError(
                    "Corrected legacy trace runtime position disagrees with deterministic "
                    "batch replay"
                )
            expected_trace = dataset[query_index_by_id[trace["query_id"]]][
                "sampling_trace"
            ]
            if trace != expected_trace:
                raise ValueError(
                    "Corrected legacy trace disagrees with deterministic sampling replay"
                )
            merged_values.append(record)
            count += 1
    if count != 8_360 or any(values != expected_query_ids for values in coverage.values()):
        raise ValueError("Corrected legacy merged traces have incomplete query/epoch coverage")
    expected_merged = sorted(
        shard_values,
        key=lambda value: (value["trace"]["epoch"], value["trace"]["query_id"]),
    )
    if merged_values != expected_merged:
        raise ValueError("Corrected legacy merged traces differ from rank shards")
    return merged_record["sha256"], count


def _validate_validation_bundle(
    root: Path,
    *,
    evaluation_data,
) -> str:
    directory = root / "validation"
    if {path.name for path in directory.iterdir()} != {
        "manifest.json",
        *{f"epoch-{epoch:03d}.json" for epoch in range(1, 21)},
    }:
        raise ValueError("Corrected legacy validation directory inventory changed")
    manifest = _load_json(directory / "manifest.json")
    if set(manifest) != {
        "schema_version", "epochs", "global_steps", "model_selection",
        "files", "records_sha256",
    }:
        raise ValueError("Corrected legacy validation manifest schema changed")
    if (
        manifest["schema_version"] != 1
        or manifest["epochs"] != 20
        or manifest["global_steps"] != list(range(4, 81, 4))
        or manifest["model_selection"] != "none_final_epoch_only"
    ):
        raise ValueError("Corrected legacy validation chronology changed")
    files = manifest["files"]
    if type(files) is not list or len(files) != 20:
        raise ValueError("Corrected legacy validation must contain 20 files")
    records = []
    for epoch, file_record in enumerate(files, start=1):
        expected_name = f"epoch-{epoch:03d}.json"
        _validate_file_record(directory, file_record, expected_path=expected_name, name="validation")
        record = _load_json(directory / expected_name)
        if set(record) != {
            "schema_version", "epoch", "global_step", "validation_result", "record_sha256"
        }:
            raise ValueError("Corrected legacy validation record schema changed")
        if record["epoch"] != epoch or record["global_step"] != epoch * 4:
            raise ValueError("Corrected legacy validation record chronology changed")
        checksum_payload = {key: value for key, value in record.items() if key != "record_sha256"}
        if record["record_sha256"] != hashlib.sha256(
            _canonical_json(checksum_payload).encode("utf-8")
        ).hexdigest():
            raise ValueError("Corrected legacy validation record checksum changed")
        try:
            canonical_result_from_payload(record["validation_result"], evaluation_data)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"Corrected legacy validation epoch {epoch} failed canonical replay"
            ) from error
        records.append(record)
    expected_records_sha = hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()
    if manifest["records_sha256"] != expected_records_sha:
        raise ValueError("Corrected legacy validation history checksum changed")
    return expected_records_sha


def _validate_evaluation_bundle(root: Path, *, test_data, query_view: str, model_sha256: str) -> str:
    directory = root / "evaluation"
    if {path.name for path in directory.iterdir()} != {
        "artifact_manifest.json", "evaluation_config.json", "results.json", "rankings.jsonl"
    }:
        raise ValueError("Corrected legacy evaluation directory inventory changed")
    manifest = _load_json(directory / "artifact_manifest.json")
    if set(manifest) != {"schema_version", "evaluation_config", "results", "rankings"}:
        raise ValueError("Corrected legacy evaluation manifest schema changed")
    if manifest["schema_version"] != 1:
        raise ValueError("Corrected legacy evaluation schema changed")
    for field, path in (
        ("evaluation_config", "evaluation_config.json"),
        ("results", "results.json"),
        ("rankings", "rankings.jsonl"),
    ):
        _validate_file_record(directory, manifest[field], expected_path=path, name=field)
    config = _load_json(directory / "evaluation_config.json")
    expected_config_keys = {
        "schema_version",
        "evaluation_type",
        "query_view",
        "system_count",
        "regimes",
        "query_rankings",
        "test_contract_sha256",
        "final_model",
    }
    if (
        set(config) != expected_config_keys
        or config["schema_version"] != 1
        or config["evaluation_type"] != "corrected_legacy_diagnostic_test"
        or config["query_view"] != query_view
        or config["system_count"] != 1
        or config["regimes"] != list(CORRECTED_LEGACY_TEST_REGIMES)
        or config["query_rankings"] != 160
        or config["test_contract_sha256"] != test_data.contract_sha256
        or config["final_model"]
        != {"path": "model.safetensors", "size": (root / "model.safetensors").stat().st_size,
            "sha256": model_sha256}
    ):
        raise ValueError("Corrected legacy evaluation configuration changed")
    results_payload = _load_json(directory / "results.json")
    if set(results_payload) != {"schema_version", "results"} or results_payload["schema_version"] != 1:
        raise ValueError("Corrected legacy evaluation results schema changed")
    if type(results_payload["results"]) is not dict or set(results_payload["results"]) != set(
        CORRECTED_LEGACY_TEST_REGIMES
    ):
        raise ValueError("Corrected legacy evaluation result regimes changed")
    expected_rows = []
    source_hashes = set()
    for regime in CORRECTED_LEGACY_TEST_REGIMES:
        try:
            result = canonical_result_from_payload(
                results_payload["results"][regime],
                test_data.evaluation_data_by_regime[regime],
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"Corrected legacy test regime {regime} failed canonical replay"
            ) from error
        source_hashes.add(result.source_ranking_sha256)
        expected_rows.extend(
            {"regime_name": regime, **ranking}
            for ranking in result.to_payload()["rankings"]
        )
    if len(source_hashes) != 1 or len(expected_rows) != 160:
        raise ValueError("Corrected legacy evaluation source ranking or coverage changed")
    actual_rows = []
    with (directory / "rankings.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if line != _canonical_json(value) + "\n":
                raise ValueError("Corrected legacy ranking row is not canonical")
            actual_rows.append(value)
    if actual_rows != expected_rows:
        raise ValueError("Corrected legacy rankings JSONL disagrees with replayed results")
    return _sha256(directory / "results.json")


def _validate_input_bundle(root: Path, loaded_config) -> None:
    directory = root / "inputs"
    expected_paths = {
        "manifest.json",
        "settings.json",
        *{f"data/{path}" for path in (
            "cases.jsonl", "corpus.jsonl", "dataset_manifest.json",
            "pools/candidates_by_case.json", "pools/candidates_global.json",
            "queries/all.jsonl",
        )},
        *{
            f"corrected_legacy_membership/{role}_cases.txt"
            for role in ("train", "validation", "test")
        },
        *{
            f"subsets/{role}_{kind}.jsonl"
            for role in ("train", "validation", "test")
            for kind in ("queries", "corpus")
        },
    }
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("Corrected legacy copied-input inventory changed")
    manifest = _load_json(directory / "manifest.json")
    if set(manifest) != {
        "schema_version", "config_sha256", "data", "membership", "subsets", "settings"
    }:
        raise ValueError("Corrected legacy input manifest schema changed")
    if manifest["schema_version"] != 1 or manifest["config_sha256"] != CORRECTED_LEGACY_CONFIG_SHA256:
        raise ValueError("Corrected legacy input manifest identity changed")
    expected_data = [
        "data/cases.jsonl",
        "data/corpus.jsonl",
        "data/dataset_manifest.json",
        "data/pools/candidates_by_case.json",
        "data/pools/candidates_global.json",
        "data/queries/all.jsonl",
    ]
    if type(manifest["data"]) is not list or [record.get("path") for record in manifest["data"]] != expected_data:
        raise ValueError("Corrected legacy input data records changed")
    for record, path in zip(manifest["data"], expected_data):
        _validate_file_record(directory, record, expected_path=path, name="copied data")
    if type(manifest["membership"]) is not list or len(manifest["membership"]) != 3:
        raise ValueError("Corrected legacy membership input records changed")
    for role, record in zip(("train", "validation", "test"), manifest["membership"]):
        if type(record) is not dict or set(record) != {"role", "path", "size", "sha256"}:
            raise ValueError("Corrected legacy membership record schema changed")
        if record["role"] != role:
            raise ValueError("Corrected legacy membership record order changed")
        _validate_file_record(
            directory,
            {key: record[key] for key in ("path", "size", "sha256")},
            expected_path=f"corrected_legacy_membership/{role}_cases.txt",
            name=f"{role} membership",
        )
    if type(manifest["subsets"]) is not list or len(manifest["subsets"]) != 6:
        raise ValueError("Corrected legacy subset input records changed")
    expected_subset_pairs = [
        (role, kind)
        for role in ("train", "validation", "test")
        for kind in ("queries", "corpus")
    ]
    for record, (role, kind) in zip(manifest["subsets"], expected_subset_pairs):
        if type(record) is not dict or set(record) != {
            "role", "kind", "count", "path", "size", "sha256"
        }:
            raise ValueError("Corrected legacy subset record schema changed")
        config_role = loaded_config.value["membership"][role]
        expected_count = config_role["query_count" if kind == "queries" else "passage_count"]
        expected_sha = config_role[
            "query_subset_sha256" if kind == "queries" else "passage_subset_sha256"
        ]
        expected_path = f"subsets/{role}_{kind}.jsonl"
        if (
            record["role"] != role
            or record["kind"] != kind
            or record["count"] != expected_count
            or record["sha256"] != expected_sha
        ):
            raise ValueError("Corrected legacy subset record identity changed")
        _validate_file_record(
            directory,
            {key: record[key] for key in ("path", "size", "sha256")},
            expected_path=expected_path,
            name=f"{role} {kind} subset",
        )
    _validate_file_record(
        directory,
        manifest["settings"],
        expected_path="settings.json",
        name="corrected legacy settings",
    )


def validate_corrected_legacy_artifact(
    root: Path,
    *,
    expectation: CorrectedLegacyArtifactExpectation,
) -> CorrectedLegacyArtifact:
    if not isinstance(expectation, CorrectedLegacyArtifactExpectation):
        raise TypeError("Corrected legacy artifact validation requires an external expectation")
    root = Path(os.path.abspath(root))
    _validate_tree(root)
    if _sha256(root / "artifact_manifest.json") != expectation.artifact_manifest_sha256:
        raise ValueError("Corrected legacy artifact commit-marker digest changed")
    artifact_manifest = _load_json(root / "artifact_manifest.json")
    expected_manifest_keys = {
        "schema_version", "commit_marker", "artifact_type", "run", "model",
        "tokenizer", "encoder_config", "wrapper", "setting_explanation",
        "inputs_manifest", "trace_manifest", "validation_manifest", "evaluation_manifest",
    }
    if type(artifact_manifest) is not dict or set(artifact_manifest) != expected_manifest_keys:
        raise ValueError("Corrected legacy artifact commit-marker schema changed")
    if (
        artifact_manifest["schema_version"] != 1
        or artifact_manifest["commit_marker"] is not True
        or artifact_manifest["artifact_type"] != "corrected_legacy_diagnostic_retriever"
    ):
        raise ValueError("Corrected legacy artifact identity changed")
    _validate_file_record(root, artifact_manifest["run"], expected_path="corrected_legacy_run.json", name="run")
    model_record = _validate_file_record(root, artifact_manifest["model"], expected_path="model.safetensors", name="model")
    _validate_file_record(root, artifact_manifest["wrapper"], expected_path="wrapper_config.json", name="wrapper")
    _validate_file_record(root, artifact_manifest["setting_explanation"], expected_path="setting_explanation.md", name="explanation")
    _validate_file_record(root, artifact_manifest["inputs_manifest"], expected_path="inputs/manifest.json", name="inputs")
    _validate_file_record(root, artifact_manifest["trace_manifest"], expected_path="candidate_traces/manifest.json", name="traces")
    _validate_file_record(root, artifact_manifest["validation_manifest"], expected_path="validation/manifest.json", name="validation")
    _validate_file_record(root, artifact_manifest["evaluation_manifest"], expected_path="evaluation/artifact_manifest.json", name="evaluation")
    _validate_directory_record(
        root,
        artifact_manifest["tokenizer"],
        name="tokenizer",
        expected_files=EXPECTED_TOKENIZER_FILES,
    )
    _validate_directory_record(
        root,
        artifact_manifest["encoder_config"],
        name="encoder config",
        expected_files=EXPECTED_ENCODER_CONFIG_FILES,
    )

    model_identity = _canonical_bf16_safetensors_identity(
        root / "model.safetensors",
        expected_metadata={
            "format": "pt",
            "source": "active_engine_epoch_20_zero3_gathered_state",
            "weight_dtype": "bfloat16",
        },
    )
    if model_identity["tensor_count"] != 134 or model_record["sha256"] != _sha256(root / "model.safetensors"):
        raise ValueError("Corrected legacy model tensor inventory changed")
    wrapper = _load_json(root / "wrapper_config.json")
    expected_wrapper_keys = {
        "schema_version",
        "architecture",
        "artifact_type",
        "query_view",
        "slot_token",
        "slot_token_id",
        "temperature",
        "tokenizer_size",
        "weight_dtype",
        "model_artifact_protocol",
    }
    query_view = wrapper.get("query_view")
    if (
        set(wrapper) != expected_wrapper_keys
        or wrapper.get("schema_version") != 1
        or wrapper.get("architecture") != "DualEncoderRetriever"
        or wrapper.get("artifact_type") != "corrected_legacy_diagnostic_retriever"
        or query_view != expectation.query_view
        or wrapper.get("slot_token") != SLOT_TOKEN
        or type(wrapper.get("slot_token_id")) is not int
        or wrapper["slot_token_id"] not in range(CONTROLLED_TOKENIZER_SIZE)
        or wrapper.get("weight_dtype") != "bfloat16"
        or wrapper.get("temperature") != 0.07
        or wrapper.get("tokenizer_size") != CONTROLLED_TOKENIZER_SIZE
    ):
        raise ValueError("Corrected legacy wrapper contract changed")
    _validate_tokenizer_payload(root, slot_token_id=wrapper["slot_token_id"])

    validate_staged_dataset(
        dataset_dir=root / "inputs/data",
        expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
    )
    loaded_config = load_corrected_legacy_config(
        root / "inputs/settings.json",
        expected_sha256=CORRECTED_LEGACY_CONFIG_SHA256,
        dataset_dir=root / "inputs/data",
    )
    _validate_input_bundle(root, loaded_config)
    if wrapper["model_artifact_protocol"] != loaded_config.value["training"]["final_model"]:
        raise ValueError("Corrected legacy wrapper model-artifact protocol changed")
    encoder_config = _load_json(root / "encoder_config/config.json")
    expected_encoder_values = {
        "model_type": "modernbert",
        "vocab_size": CONTROLLED_TOKENIZER_SIZE,
        "deterministic_flash_attn": True,
        "reference_compile": False,
        "torch_dtype": "float32",
    }
    if any(
        type(encoder_config.get(field)) is not type(expected)
        or encoder_config.get(field) != expected
        for field, expected in expected_encoder_values.items()
    ):
        raise ValueError("Corrected legacy encoder configuration changed")
    corpus = load_corpus(root / "inputs/data")
    queries = load_queries(root / "inputs/data", "all")
    passage_index = PassageIndexTable(corpus)
    if passage_index.sha256 != EXPECTED_PASSAGE_INDEX_SHA256:
        raise ValueError("Corrected legacy copied passage index changed")
    validation_data = build_corrected_legacy_validation_evidence_data(
        all_queries=queries,
        corpus_by_passage_id=corpus,
        passage_index_table=passage_index,
        validation_case_ids=loaded_config.memberships.validation,
        query_view=query_view,
    )
    test_data = build_corrected_legacy_test_data(
        all_queries=queries,
        corpus_by_passage_id=corpus,
        passage_index_table=passage_index,
        test_case_ids=loaded_config.memberships.test,
        query_view=query_view,
    )
    corrected_data = load_corrected_legacy_data(root / "inputs/data")
    trace_dataset = CorrectedLegacyDiagnosticDataset(
        corrected_data,
        experiment_seed=17,
        query_view=query_view,
    )
    trace_sha, _ = _validate_trace_bundle(root, dataset=trace_dataset)
    validation_sha = _validate_validation_bundle(
        root,
        evaluation_data=validation_data.evaluation_data,
    )
    evaluation_sha = _validate_evaluation_bundle(
        root,
        test_data=test_data,
        query_view=query_view,
        model_sha256=model_record["sha256"],
    )

    run = _load_json(root / "corrected_legacy_run.json")
    expected_run_keys = {
        "schema_version",
        "diagnostic_id",
        "label",
        "run_id",
        "run_kind",
        "query_view",
        "seed",
        "schedule",
        "runtime_versions",
        "training_image",
        "training_base_image",
        "training_image_runtime_inventory_sha256",
        "training_launch_provenance",
        "snapshot",
        "config_sha256",
        "passage_index_sha256",
        "validation_records",
        "candidate_traces",
        "final_model",
        "inputs_manifest_sha256",
        "validation_manifest_sha256",
        "evaluation_manifest_sha256",
        "reporting_boundary",
    }
    if (
        set(run) != expected_run_keys
        or run["schema_version"] != 1
        or run["diagnostic_id"] != loaded_config.value["diagnostic_id"]
        or run["run_id"] != expectation.run_id
        or run["run_kind"] != "corrected_legacy_diagnostic"
        or run["label"] != loaded_config.value["label"]
        or run["query_view"] != query_view
        or run["seed"] != 17
        or run["schedule"] != {"epochs": 20, "updates_per_epoch": 4, "total_updates": 80}
        or run["runtime_versions"] != EXPECTED_RUNTIME_VERSIONS
        or run["training_image"] != EXPECTED_DERIVED_TRAINING_IMAGE
        or run["training_base_image"] != EXPECTED_BASE_TRAINING_IMAGE
        or run["training_image_runtime_inventory_sha256"]
        != EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        or run["snapshot"]
        != {
            "manifest_sha256": EXPECTED_SNAPSHOT_MANIFEST_SHA256,
            "tree_sha256": EXPECTED_SNAPSHOT_TREE_SHA256,
        }
        or run["config_sha256"] != CORRECTED_LEGACY_CONFIG_SHA256
        or run["passage_index_sha256"] != EXPECTED_PASSAGE_INDEX_SHA256
        or run["validation_records"] != 20
        or run["candidate_traces"]
        != {
            "record_count": 8_360,
            "manifest_sha256": _sha256(root / "candidate_traces/manifest.json"),
        }
        or run["final_model"] != {**model_record, "tensor_count": 134}
        or run["inputs_manifest_sha256"] != _sha256(root / "inputs/manifest.json")
        or run["validation_manifest_sha256"] != _sha256(root / "validation/manifest.json")
        or run["evaluation_manifest_sha256"]
        != _sha256(root / "evaluation/artifact_manifest.json")
        or run["reporting_boundary"] != loaded_config.value["reporting_boundary"]
    ):
        raise ValueError("Corrected legacy run record changed")
    provenance = run["training_launch_provenance"]
    expected_provenance_keys = {
        "bootstrap_protocol",
        "source_bundle",
        "training_image_contract_sha256",
        "training_plan_sha256",
        "training_request_payload_sha256",
        "training_run_id",
        "training_staging_receipt_sha256",
    }
    source = provenance.get("source_bundle") if type(provenance) is dict else None
    if (
        type(provenance) is not dict
        or set(provenance) != expected_provenance_keys
        or provenance["bootstrap_protocol"] != EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL
        or provenance["training_image_contract_sha256"]
        != EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256
        or provenance["training_plan_sha256"] != expectation.training_plan_sha256
        or provenance["training_staging_receipt_sha256"]
        != expectation.training_staging_receipt_sha256
        or provenance["training_request_payload_sha256"]
        != expectation.training_request_payload_sha256
        or provenance["training_run_id"] != expectation.run_id
        or type(source) is not dict
        or set(source) != {"commit_epoch", "inventory_sha256", "name", "sha256", "size"}
        or type(source["commit_epoch"]) is not int
        or source["commit_epoch"] < 1
        or type(source["size"]) is not int
        or source["size"] < 1
        or not _is_sha256(source["sha256"])
        or not _is_sha256(source["inventory_sha256"])
        or source != expectation.source_bundle_payload()
    ):
        raise ValueError("Corrected legacy training launch provenance changed")
    expected_explanation = (
        "# Corrected legacy-style diagnostic\n\n"
        + loaded_config.value["setting_explanation"]
        + "\n\nThe job completed 20 full epochs and exported the active epoch-20 model. "
        "No best-epoch selection or checkpoint reload was performed.\n"
    )
    if (root / "setting_explanation.md").read_text(encoding="utf-8") != expected_explanation:
        raise ValueError("Corrected legacy setting explanation changed")
    return CorrectedLegacyArtifact(
        root=root,
        artifact_manifest_sha256=_sha256(root / "artifact_manifest.json"),
        run_id=expectation.run_id,
        model_sha256=model_record["sha256"],
        query_view=query_view,
        trace_merged_sha256=trace_sha,
        validation_records_sha256=validation_sha,
        evaluation_results_sha256=evaluation_sha,
    )


__all__ = [
    "CorrectedLegacyArtifact",
    "CorrectedLegacyArtifactExpectation",
    "validate_corrected_legacy_artifact",
]
