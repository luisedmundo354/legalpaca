from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .provenance import (
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_DATASET_OUTPUT_SHA256,
    EXPECTED_DEEPSPEED_CONFIG_SHA256,
    EXPECTED_EXPERIMENT_CONFIG_SHA256,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_FOLD_ROTATION_SHA256_BY_OUTER_FOLD,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_IMAGE,
    EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD,
    EXPECTED_VALIDATION_IDENTITY_BY_CELL,
)


CONTROLLED_ARTIFACT_SCHEMA_VERSION = 1
CONTROLLED_ARTIFACT_PROTOCOL = (
    "fresh_best_engine_zero3_gathered_bf16_safetensors_v1"
)
CONTROLLED_EXPERIMENT_ID = "arr_retrieval_cv_v1"
CONTROLLED_QUERY_VIEWS = ("structured", "flat_masked")
CONTROLLED_SAMPLERS = ("local_unique", "global_uniform")
CONTROLLED_SEEDS = (17, 29, 43)
CONTROLLED_WORLD_SIZE = 4
CONTROLLED_EPOCHS = 20
CONTROLLED_TOKENIZER_SIZE = 50_386
CONTROLLED_MODEL_STATE_COUNT = 134
CONTROLLED_ATTENTION_MODULE_COUNT = 22
CONTROLLED_TEMPERATURE = 0.07
CONTROLLED_SLOT_TOKEN = "[MASK]"

_ARTIFACT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "commit_marker",
        "controlled_run",
        "model",
        "tokenizer",
        "encoder_config",
        "wrapper_config",
        "candidate_trace_manifest",
        "validation_manifest",
        "retained_checkpoints",
    }
)
_CONTROLLED_RUN_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "outer_fold",
        "query_view",
        "sampler",
        "experiment_seed",
        "runtime_versions",
        "training_image",
        "experiment_config",
        "deepspeed_config",
        "dataset",
        "folds",
        "snapshot",
        "passage_index",
        "validation_data",
        "candidate_traces",
        "validation_history",
        "best_checkpoint_reload",
        "final_model",
        "tokenizer",
        "encoder_config",
        "wrapper_config",
        "retained_checkpoints",
    }
)
_WRAPPER_KEYS = frozenset(
    {
        "schema_version",
        "architecture",
        "slot_token",
        "slot_token_id",
        "temperature",
        "tokenizer_size",
        "weight_dtype",
        "model_artifact_protocol",
    }
)
_FILE_RECORD_KEYS = frozenset({"path", "size", "sha256"})
_DIRECTORY_RECORD_KEYS = frozenset({"path", "files"})
_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "epoch",
        "global_step",
        "checkpoint_dir",
        "deepspeed_tag",
        "primary_metric",
        "secondary_metric",
        "ranking_sha256",
    }
)


@dataclass(frozen=True)
class ControlledArtifactExpectation:
    artifact_manifest_sha256: str
    experiment_id: str
    outer_fold: int
    query_view: str
    sampler: str
    experiment_seed: int
    dataset_manifest_sha256: str
    fold_manifest_sha256: str
    passage_index_sha256: str
    model_artifact_protocol: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.artifact_manifest_sha256,
            name="expectation.artifact_manifest_sha256",
        )
        if type(self.experiment_id) is not str or self.experiment_id != CONTROLLED_EXPERIMENT_ID:
            raise ValueError(
                f"Controlled experiment_id must be {CONTROLLED_EXPERIMENT_ID!r}"
            )
        if type(self.outer_fold) is not int or self.outer_fold not in range(5):
            raise ValueError("Controlled outer_fold must be an exact integer from 0 through 4")
        if type(self.query_view) is not str or self.query_view not in CONTROLLED_QUERY_VIEWS:
            raise ValueError(
                f"Controlled query_view must be one of {CONTROLLED_QUERY_VIEWS}"
            )
        if type(self.sampler) is not str or self.sampler not in CONTROLLED_SAMPLERS:
            raise ValueError(f"Controlled sampler must be one of {CONTROLLED_SAMPLERS}")
        if type(self.experiment_seed) is not int or self.experiment_seed not in CONTROLLED_SEEDS:
            raise ValueError(f"Controlled experiment_seed must be one of {CONTROLLED_SEEDS}")
        for name, value in (
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("fold_manifest_sha256", self.fold_manifest_sha256),
            ("passage_index_sha256", self.passage_index_sha256),
        ):
            _require_sha256(value, name=f"expectation.{name}")
        expected_study_hashes = {
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
            "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
        }
        for name, expected in expected_study_hashes.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Controlled {name} left the frozen study")
        if (
            type(self.model_artifact_protocol) is not str
            or self.model_artifact_protocol != CONTROLLED_ARTIFACT_PROTOCOL
        ):
            raise ValueError(
                "Controlled model artifact protocol changed: "
                f"{self.model_artifact_protocol!r}"
            )


@dataclass(frozen=True)
class ArtifactFileRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ControlledArtifactIdentity:
    artifact_manifest_sha256: str
    controlled_run_sha256: str
    model_sha256: str
    wrapper_config_sha256: str
    tokenizer_inventory_sha256: str
    encoder_config_inventory_sha256: str
    experiment_id: str
    outer_fold: int
    query_view: str
    sampler: str
    experiment_seed: int
    experiment_config_sha256: str
    deepspeed_config_sha256: str
    dataset_manifest_sha256: str
    dataset_output_sha256: str
    fold_manifest_sha256: str
    fold_rotation_sha256: str
    passage_index_sha256: str
    snapshot_manifest_sha256: str
    snapshot_tree_sha256: str
    model_artifact_protocol: str


@dataclass(frozen=True)
class ValidatedControlledArtifact:
    root: Path
    expectation: ControlledArtifactExpectation
    identity: ControlledArtifactIdentity
    files: tuple[ArtifactFileRecord, ...]
    model_path: Path
    tokenizer_dir: Path
    encoder_config_dir: Path
    wrapper_config_path: Path
    controlled_run_path: Path
    slot_token_id: int
    model_state_count: int


@dataclass(frozen=True)
class ControlledArtifactRuntime:
    torch_module: Any
    auto_tokenizer_class: Any
    auto_config_class: Any
    auto_model_class: Any
    load_safetensors_model: Callable[..., Any]
    retriever_class: Any


@dataclass(frozen=True)
class LoadedControlledRetriever:
    model: Any
    tokenizer: Any
    identity: ControlledArtifactIdentity
    device: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _load_json_object(path: Path, *, name: str, require_canonical: bool) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise ValueError(f"{name} must be a non-empty regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"{name} must be one JSON object")
    if require_canonical:
        expected = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if raw != expected:
            raise ValueError(f"{name} is not canonical sorted indented JSON: {path}")
    return value


def _validate_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be one non-empty exact relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{name} is unsafe or non-canonical: {value!r}")
    return value


def _regular_tree_inventory(root: Path) -> tuple[ArtifactFileRecord, ...]:
    if not isinstance(root, Path):
        raise TypeError("Artifact root must be a Path")
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Artifact root must be a real non-symlink directory: {root}")
    records: list[ArtifactFileRecord] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Controlled artifact forbids symlink: {relative}")
        if any(part.endswith(".incomplete") or part.endswith(".tmp") for part in Path(relative).parts):
            raise ValueError(f"Controlled artifact contains stale temporary entry: {relative}")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_size < 1:
            raise ValueError(f"Controlled artifact file must be non-empty and regular: {relative}")
        records.append(
            ArtifactFileRecord(
                path=relative,
                size=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )
    if not records:
        raise ValueError(f"Controlled artifact is empty: {root}")
    return tuple(records)


def _record_payload(record: ArtifactFileRecord) -> dict[str, Any]:
    return {"path": record.path, "size": record.size, "sha256": record.sha256}


def _validate_file_record(
    root: Path,
    value: object,
    *,
    name: str,
    expected_path: str | None = None,
) -> ArtifactFileRecord:
    if type(value) is not dict or set(value) != set(_FILE_RECORD_KEYS):
        raise ValueError(f"{name} must contain exactly path, size, and sha256")
    relative = _validate_relative_path(value["path"], name=f"{name}.path")
    if expected_path is not None and relative != expected_path:
        raise ValueError(f"{name}.path={relative!r}; expected {expected_path!r}")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} does not resolve to a regular artifact file: {relative}")
    size = value["size"]
    if type(size) is not int or size < 1 or size != path.stat().st_size:
        raise ValueError(f"{name} size does not match {relative}")
    digest = _require_sha256(value["sha256"], name=f"{name}.sha256")
    if digest != _sha256_file(path):
        raise ValueError(f"{name} SHA-256 does not match {relative}")
    return ArtifactFileRecord(path=relative, size=size, sha256=digest)


def _validate_directory_record(
    artifact_root: Path,
    value: object,
    *,
    name: str,
    expected_path: str,
    expected_files: Sequence[str] | None = None,
) -> tuple[ArtifactFileRecord, ...]:
    if type(value) is not dict or set(value) != set(_DIRECTORY_RECORD_KEYS):
        raise ValueError(f"{name} must contain exactly path and files")
    relative_root = _validate_relative_path(value["path"], name=f"{name}.path")
    if relative_root != expected_path:
        raise ValueError(f"{name}.path={relative_root!r}; expected {expected_path!r}")
    directory = artifact_root / relative_root
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{name} must resolve to a real directory")
    raw_files = value["files"]
    if type(raw_files) is not list or not raw_files:
        raise ValueError(f"{name}.files must be a non-empty exact list")
    records: list[ArtifactFileRecord] = []
    observed: list[str] = []
    for index, raw_record in enumerate(raw_files):
        record = _validate_file_record(
            directory,
            raw_record,
            name=f"{name}.files[{index}]",
        )
        records.append(
            ArtifactFileRecord(
                path=f"{relative_root}/{record.path}",
                size=record.size,
                sha256=record.sha256,
            )
        )
        observed.append(record.path)
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ValueError(f"{name}.files must be unique and canonically sorted")
    if expected_files is not None and observed != list(expected_files):
        raise ValueError(
            f"{name} inventory changed: actual={observed}, expected={list(expected_files)}"
        )
    actual = tuple(
        ArtifactFileRecord(
            path=f"{relative_root}/{record.path}",
            size=record.size,
            sha256=record.sha256,
        )
        for record in _regular_tree_inventory(directory)
    )
    if tuple(records) != actual:
        raise ValueError(f"{name} manifest inventory differs from disk")
    return tuple(records)


def _validate_selection(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(_SELECTION_KEYS):
        raise ValueError(f"{name} has an invalid checkpoint-selection schema")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError(f"{name}.schema_version must be exact integer 1")
    epoch = value["epoch"]
    step = value["global_step"]
    if type(epoch) is not int or epoch not in range(1, CONTROLLED_EPOCHS + 1):
        raise ValueError(f"{name}.epoch is outside the controlled training range")
    if type(step) is not int or step != epoch * 3:
        raise ValueError(f"{name}.global_step must equal epoch * 3")
    if value["checkpoint_dir"] != f"checkpoint-{step}":
        raise ValueError(f"{name}.checkpoint_dir and global_step disagree")
    if value["deepspeed_tag"] != f"global_step{step}":
        raise ValueError(f"{name}.deepspeed_tag and global_step disagree")
    for metric in ("primary_metric", "secondary_metric"):
        number = value[metric]
        if type(number) is not float or not math.isfinite(number):
            raise ValueError(f"{name}.{metric} must be one finite exact float")
    _require_sha256(value["ranking_sha256"], name=f"{name}.ranking_sha256")
    return dict(value)


def _validate_checkpoint_inventory(
    root: Path,
    value: object,
) -> tuple[tuple[ArtifactFileRecord, ...], tuple[str, ...]]:
    if type(value) is not dict or set(value) != {"schema_version", "checkpoints"}:
        raise ValueError("retained_checkpoints has an invalid schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("retained_checkpoints.schema_version must be exact integer 1")
    checkpoints = value["checkpoints"]
    if type(checkpoints) is not list or len(checkpoints) not in (1, 2):
        raise ValueError("Controlled artifact must retain exactly best and last checkpoints")
    all_records: list[ArtifactFileRecord] = []
    names: list[str] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        if type(checkpoint) is not dict or set(checkpoint) != {"path", "files"}:
            raise ValueError(f"retained checkpoint {checkpoint_index} has an invalid schema")
        checkpoint_name = _validate_relative_path(
            checkpoint["path"], name=f"retained checkpoint {checkpoint_index}.path"
        )
        if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.startswith(
            "checkpoint-"
        ):
            raise ValueError(f"Invalid retained checkpoint name: {checkpoint_name!r}")
        try:
            step = int(checkpoint_name.removeprefix("checkpoint-"))
        except ValueError as error:
            raise ValueError(f"Invalid retained checkpoint step: {checkpoint_name!r}") from error
        if step < 3 or step > 60 or step % 3:
            raise ValueError(f"Retained checkpoint is outside the controlled step range: {step}")
        tag = f"global_step{step}"
        checkpoint_root = root / checkpoint_name
        if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
            raise ValueError(f"Retained checkpoint is missing or unsafe: {checkpoint_name}")
        raw_files = checkpoint["files"]
        if type(raw_files) is not list or not raw_files:
            raise ValueError(f"Retained checkpoint {checkpoint_name} has no file inventory")
        records: list[ArtifactFileRecord] = []
        paths: list[str] = []
        for file_index, raw_record in enumerate(raw_files):
            record = _validate_file_record(
                checkpoint_root,
                raw_record,
                name=f"retained[{checkpoint_name}].files[{file_index}]",
            )
            records.append(record)
            paths.append(record.path)
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError(f"Retained checkpoint {checkpoint_name} inventory is not canonical")
        expected_paths = {
            "checkpoint_manifest.json",
            "zero_to_fp32.py",
            "scheduler.pt",
            "training_args.bin",
            "trainer_state.json",
            *(f"rng_state_{rank}.pth" for rank in range(CONTROLLED_WORLD_SIZE)),
            *(
                f"{tag}/zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
                for rank in range(CONTROLLED_WORLD_SIZE)
            ),
            *(
                f"{tag}/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                for rank in range(CONTROLLED_WORLD_SIZE)
            ),
        }
        if set(paths) != expected_paths:
            raise ValueError(
                f"Retained checkpoint {checkpoint_name} inventory schema changed"
            )
        actual = _regular_tree_inventory(checkpoint_root)
        if tuple(records) != actual:
            raise ValueError(f"Retained checkpoint {checkpoint_name} differs from its inventory")
        checkpoint_manifest_path = checkpoint_root / "checkpoint_manifest.json"
        checkpoint_manifest = _load_json_object(
            checkpoint_manifest_path,
            name=f"{checkpoint_name} manifest",
            require_canonical=True,
        )
        expected_manifest_keys = {
            "schema_version",
            "selection",
            "world_size",
            "client_state_sha256",
            "scheduler_state_sha256",
            "rng_files",
            "files",
        }
        if set(checkpoint_manifest) != expected_manifest_keys:
            raise ValueError(f"{checkpoint_name} checkpoint manifest schema changed")
        if checkpoint_manifest["schema_version"] != 1:
            raise ValueError(f"{checkpoint_name} checkpoint manifest version changed")
        selection = _validate_selection(
            checkpoint_manifest["selection"], name=f"{checkpoint_name}.selection"
        )
        if selection["checkpoint_dir"] != checkpoint_name:
            raise ValueError(f"{checkpoint_name} selection identity changed")
        if checkpoint_manifest["world_size"] != CONTROLLED_WORLD_SIZE:
            raise ValueError(f"{checkpoint_name} world size changed")
        for digest_name in ("client_state_sha256", "scheduler_state_sha256"):
            _require_sha256(
                checkpoint_manifest[digest_name],
                name=f"{checkpoint_name}.{digest_name}",
            )
        expected_rng = [f"rng_state_{rank}.pth" for rank in range(CONTROLLED_WORLD_SIZE)]
        if checkpoint_manifest["rng_files"] != expected_rng:
            raise ValueError(f"{checkpoint_name} RNG inventory changed")
        manifest_files = checkpoint_manifest["files"]
        if type(manifest_files) is not list:
            raise ValueError(f"{checkpoint_name} manifest file inventory is malformed")
        expected_without_manifest = [
            _record_payload(record)
            for record in records
            if record.path != "checkpoint_manifest.json"
        ]
        if manifest_files != expected_without_manifest:
            raise ValueError(f"{checkpoint_name} nested manifest inventory changed")
        all_records.extend(
            ArtifactFileRecord(
                path=f"{checkpoint_name}/{record.path}",
                size=record.size,
                sha256=record.sha256,
            )
            for record in records
        )
        names.append(checkpoint_name)
    if names != sorted(names, key=lambda item: int(item.removeprefix("checkpoint-"))):
        raise ValueError("Retained checkpoint records must be chronological")
    if len(names) != len(set(names)):
        raise ValueError("Retained checkpoint records contain a duplicate")
    return tuple(all_records), tuple(names)


def _validate_trace_manifest(
    root: Path,
    record: object,
    *,
    expected_passage_index_sha256: str,
    expected_queries_per_epoch: int,
) -> tuple[tuple[ArtifactFileRecord, ...], dict[str, Any]]:
    manifest_record = _validate_file_record(
        root,
        record,
        name="artifact_manifest.candidate_trace_manifest",
        expected_path="candidate_traces/manifest.json",
    )
    trace_root = root / "candidate_traces"
    manifest = _load_json_object(
        trace_root / "manifest.json",
        name="Candidate trace manifest",
        require_canonical=True,
    )
    expected_keys = {
        "schema_version",
        "merge_order",
        "epochs",
        "queries_per_epoch",
        "record_count",
        "query_ids_sha256",
        "passage_index_sha256",
        "merged",
        "shards",
    }
    if set(manifest) != expected_keys or manifest["schema_version"] != 1:
        raise ValueError("Candidate trace manifest schema changed")
    if manifest["merge_order"] != ["epoch", "query_id"]:
        raise ValueError("Candidate trace merge order changed")
    if manifest["epochs"] != CONTROLLED_EPOCHS or type(manifest["epochs"]) is not int:
        raise ValueError("Candidate trace epoch count changed")
    if (
        manifest["queries_per_epoch"] != expected_queries_per_epoch
        or type(manifest["queries_per_epoch"]) is not int
    ):
        raise ValueError("Candidate trace query count changed")
    expected_count = CONTROLLED_EPOCHS * expected_queries_per_epoch
    if manifest["record_count"] != expected_count:
        raise ValueError("Candidate trace record count changed")
    _require_sha256(manifest["query_ids_sha256"], name="trace.query_ids_sha256")
    if manifest["passage_index_sha256"] != expected_passage_index_sha256:
        raise ValueError("Candidate trace passage-index identity changed")
    merged = manifest["merged"]
    if type(merged) is not dict or set(merged) != {
        "path",
        "record_count",
        "size",
        "sha256",
    }:
        raise ValueError("Candidate trace merged record schema changed")
    if merged["record_count"] != expected_count:
        raise ValueError("Candidate trace merged coverage changed")
    merged_record = _validate_file_record(
        trace_root,
        {key: merged[key] for key in _FILE_RECORD_KEYS},
        name="candidate_traces.merged",
        expected_path="sampling_traces.jsonl",
    )
    shards = manifest["shards"]
    if type(shards) is not list or len(shards) != CONTROLLED_WORLD_SIZE:
        raise ValueError("Candidate trace shard count changed")
    shard_records: list[ArtifactFileRecord] = []
    total_lines = 0
    for rank, shard in enumerate(shards):
        if type(shard) is not dict or set(shard) != {
            "rank",
            "path",
            "record_count",
            "size",
            "sha256",
        }:
            raise ValueError(f"Candidate trace shard {rank} schema changed")
        if shard["rank"] != rank or type(shard["rank"]) is not int:
            raise ValueError("Candidate trace shard rank ordering changed")
        shard_record = _validate_file_record(
            trace_root,
            {key: shard[key] for key in _FILE_RECORD_KEYS},
            name=f"candidate_traces.shards[{rank}]",
            expected_path=f"rank-{rank:05d}.jsonl",
        )
        if type(shard["record_count"]) is not int or shard["record_count"] < 1:
            raise ValueError(f"Candidate trace shard {rank} record count is invalid")
        line_count = _line_count(trace_root / shard_record.path)
        if line_count != shard["record_count"]:
            raise ValueError(f"Candidate trace shard {rank} line count changed")
        total_lines += line_count
        shard_records.append(shard_record)
    if total_lines != expected_count:
        raise ValueError("Candidate trace shard coverage changed")
    merged_lines = _line_count(trace_root / merged_record.path)
    if merged_lines != expected_count:
        raise ValueError("Merged candidate trace line count changed")
    expected_paths = {
        "manifest.json",
        "sampling_traces.jsonl",
        *(f"rank-{rank:05d}.jsonl" for rank in range(CONTROLLED_WORLD_SIZE)),
    }
    actual = _regular_tree_inventory(trace_root)
    if {item.path for item in actual} != expected_paths:
        raise ValueError("Candidate trace directory inventory changed")
    prefixed = tuple(
        ArtifactFileRecord(
            path=f"candidate_traces/{item.path}",
            size=item.size,
            sha256=item.sha256,
        )
        for item in actual
    )
    if manifest_record not in prefixed:
        raise ValueError("Candidate trace manifest record changed after validation")
    return prefixed, manifest


def _validate_validation_manifest(
    root: Path,
    record: object,
) -> tuple[tuple[ArtifactFileRecord, ...], dict[str, Any]]:
    manifest_record = _validate_file_record(
        root,
        record,
        name="artifact_manifest.validation_manifest",
        expected_path="validation/manifest.json",
    )
    validation_root = root / "validation"
    manifest = _load_json_object(
        validation_root / "manifest.json",
        name="Validation manifest",
        require_canonical=True,
    )
    expected_keys = {
        "schema_version",
        "epochs",
        "selection_order",
        "best",
        "last",
        "retained_checkpoint_dirs",
        "records",
        "history_sha256",
        "best_sha256",
        "latest_sha256",
    }
    if set(manifest) != expected_keys or manifest["schema_version"] != 1:
        raise ValueError("Validation manifest schema changed")
    if manifest["epochs"] != CONTROLLED_EPOCHS or type(manifest["epochs"]) is not int:
        raise ValueError("Validation manifest epoch count changed")
    expected_selection_order = [
        "maximize validation case-macro set recall@20",
        "maximize validation case-macro full-ranking first-gold reciprocal rank",
        "minimize epoch number",
    ]
    if manifest["selection_order"] != expected_selection_order:
        raise ValueError("Validation checkpoint-selection order changed")
    best = _validate_selection(manifest["best"], name="validation.best")
    last = _validate_selection(manifest["last"], name="validation.last")
    if last["epoch"] != CONTROLLED_EPOCHS or last["global_step"] != 60:
        raise ValueError("Validation last checkpoint is not epoch 20 / step 60")
    retained = manifest["retained_checkpoint_dirs"]
    expected_retained = sorted(
        {best["checkpoint_dir"], last["checkpoint_dir"]},
        key=lambda item: int(item.removeprefix("checkpoint-")),
    )
    if retained != expected_retained:
        raise ValueError("Validation retained checkpoint set changed")
    records = manifest["records"]
    if type(records) is not list or len(records) != CONTROLLED_EPOCHS:
        raise ValueError("Validation history does not contain exactly 20 records")
    expected_paths = {
        "manifest.json",
        "history.json",
        "best.json",
        "latest.json",
        *(f"epoch-{epoch:03d}.json" for epoch in range(1, CONTROLLED_EPOCHS + 1)),
    }
    for digest_name, filename in (
        ("history_sha256", "history.json"),
        ("best_sha256", "best.json"),
        ("latest_sha256", "latest.json"),
    ):
        digest = _require_sha256(manifest[digest_name], name=f"validation.{digest_name}")
        path = validation_root / filename
        if not path.is_file() or path.is_symlink() or digest != _sha256_file(path):
            raise ValueError(f"Validation {filename} digest changed")
    seen_paths: list[str] = []
    for index, history_entry in enumerate(records, start=1):
        if type(history_entry) is not dict or set(history_entry) != {
            "epoch",
            "global_step",
            "path",
            "sha256",
            "is_new_best",
            "candidate",
            "best_after_epoch",
        }:
            raise ValueError(f"Validation history entry {index} schema changed")
        if history_entry["epoch"] != index or history_entry["global_step"] != index * 3:
            raise ValueError("Validation history epoch/global-step sequence changed")
        expected_path = f"epoch-{index:03d}.json"
        if history_entry["path"] != expected_path:
            raise ValueError("Validation history path sequence changed")
        if type(history_entry["is_new_best"]) is not bool:
            raise ValueError("Validation history is_new_best must be exact boolean")
        _validate_selection(history_entry["candidate"], name=f"validation.records[{index}].candidate")
        _validate_selection(
            history_entry["best_after_epoch"],
            name=f"validation.records[{index}].best_after_epoch",
        )
        digest = _require_sha256(
            history_entry["sha256"], name=f"validation.records[{index}].sha256"
        )
        epoch_path = validation_root / expected_path
        if not epoch_path.is_file() or epoch_path.is_symlink() or digest != _sha256_file(epoch_path):
            raise ValueError(f"Validation epoch artifact changed: {expected_path}")
        seen_paths.append(expected_path)
    if seen_paths != sorted(seen_paths):
        raise ValueError("Validation history paths are not canonical")
    actual = _regular_tree_inventory(validation_root)
    if {item.path for item in actual} != expected_paths:
        raise ValueError("Validation directory inventory changed")
    prefixed = tuple(
        ArtifactFileRecord(
            path=f"validation/{item.path}", size=item.size, sha256=item.sha256
        )
        for item in actual
    )
    if manifest_record not in prefixed:
        raise ValueError("Validation manifest record changed after validation")
    return prefixed, manifest


def _validate_wrapper(
    path: Path,
    *,
    expectation: ControlledArtifactExpectation,
) -> dict[str, Any]:
    wrapper = _load_json_object(
        path,
        name="Controlled wrapper config",
        require_canonical=True,
    )
    if set(wrapper) != set(_WRAPPER_KEYS):
        raise ValueError("Controlled wrapper config schema changed")
    expected_scalars = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "slot_token": CONTROLLED_SLOT_TOKEN,
        "temperature": CONTROLLED_TEMPERATURE,
        "tokenizer_size": CONTROLLED_TOKENIZER_SIZE,
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": expectation.model_artifact_protocol,
    }
    for key, expected in expected_scalars.items():
        if type(wrapper.get(key)) is not type(expected) or wrapper.get(key) != expected:
            raise ValueError(
                f"Controlled wrapper field {key!r} changed: "
                f"actual={wrapper.get(key)!r}, expected={expected!r}"
            )
    slot_token_id = wrapper["slot_token_id"]
    if type(slot_token_id) is not int or slot_token_id < 0 or slot_token_id >= CONTROLLED_TOKENIZER_SIZE:
        raise ValueError("Controlled wrapper slot_token_id is invalid")
    return wrapper


def _validate_controlled_run(
    run: object,
    *,
    expectation: ControlledArtifactExpectation,
    manifest: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    trace_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    retained_names: Sequence[str],
) -> None:
    if type(run) is not dict or set(run) != set(_CONTROLLED_RUN_KEYS):
        raise ValueError("controlled_run.json schema changed")
    expected_identity = {
        "schema_version": 1,
        "experiment_id": expectation.experiment_id,
        "outer_fold": expectation.outer_fold,
        "query_view": expectation.query_view,
        "sampler": expectation.sampler,
        "experiment_seed": expectation.experiment_seed,
    }
    for key, expected in expected_identity.items():
        if type(run.get(key)) is not type(expected) or run.get(key) != expected:
            raise ValueError(
                f"Controlled run identity {key!r} changed: "
                f"actual={run.get(key)!r}, expected={expected!r}"
            )

    if run["runtime_versions"] != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError("Controlled run runtime-version provenance changed")
    if run["training_image"] != EXPECTED_TRAINING_IMAGE:
        raise ValueError("Controlled run training-image provenance changed")
    expected_config_hashes = {
        "experiment_config": ("experiment.json", EXPECTED_EXPERIMENT_CONFIG_SHA256),
        "deepspeed_config": ("ds_zero3.json", EXPECTED_DEEPSPEED_CONFIG_SHA256),
    }
    for config_name, (expected_path, expected_config_hash) in expected_config_hashes.items():
        value = run[config_name]
        if type(value) is not dict or set(value) != {"path", "sha256"}:
            raise ValueError(f"Controlled run {config_name} schema changed")
        if value["path"] != expected_path:
            raise ValueError(f"Controlled run {config_name} path changed")
        _require_sha256(value["sha256"], name=f"controlled_run.{config_name}.sha256")
        if value["sha256"] != expected_config_hash:
            raise ValueError(f"Controlled run {config_name} left the frozen study")

    dataset = run["dataset"]
    if type(dataset) is not dict or set(dataset) != {
        "manifest_path",
        "manifest_sha256",
        "output_sha256",
    }:
        raise ValueError("Controlled run dataset schema changed")
    if dataset["manifest_sha256"] != expectation.dataset_manifest_sha256:
        raise ValueError("Controlled run dataset manifest identity changed")
    if dataset["manifest_path"] != "dataset_manifest.json":
        raise ValueError("Controlled run dataset manifest path is invalid")
    outputs = dataset["output_sha256"]
    if type(outputs) is not dict or outputs != EXPECTED_DATASET_OUTPUT_SHA256:
        raise ValueError("Controlled run dataset output digest inventory changed")
    for relative, digest in outputs.items():
        _validate_relative_path(relative, name="controlled_run.dataset.output_sha256 path")
        _require_sha256(digest, name=f"controlled_run.dataset.output_sha256[{relative}]")

    folds = run["folds"]
    if type(folds) is not dict or set(folds) != {
        "manifest_path",
        "manifest_sha256",
        "rotation",
    }:
        raise ValueError("Controlled run folds schema changed")
    if folds["manifest_sha256"] != expectation.fold_manifest_sha256:
        raise ValueError("Controlled run fold manifest identity changed")
    if folds["manifest_path"] != (
        "corporate_reorganization/modernbert/experiments/retrieval_cv/configs/folds.json"
    ):
        raise ValueError("Controlled run fold manifest logical path changed")
    rotation = folds["rotation"]
    if type(rotation) is not dict or set(rotation) != {"outer_fold", "train", "validation", "test"}:
        raise ValueError("Controlled run fold rotation schema changed")
    if rotation["outer_fold"] != expectation.outer_fold:
        raise ValueError("Controlled run rotation and expected outer fold disagree")
    role_cases: list[set[str]] = []
    for role in ("train", "validation", "test"):
        role_value = rotation[role]
        if type(role_value) is not dict or set(role_value) != {
            "case_ids",
            "fold_ids",
            "num_cases",
            "passages",
            "queries",
        }:
            raise ValueError(f"Controlled run rotation role {role!r} schema changed")
        case_ids = role_value["case_ids"]
        if (
            type(case_ids) is not list
            or not case_ids
            or any(type(case_id) is not str or not case_id for case_id in case_ids)
            or case_ids != sorted(case_ids)
            or len(case_ids) != len(set(case_ids))
            or role_value["num_cases"] != len(case_ids)
        ):
            raise ValueError(f"Controlled run rotation role {role!r} case inventory changed")
        for count_name in ("num_cases", "passages", "queries"):
            if type(role_value[count_name]) is not int or role_value[count_name] < 1:
                raise ValueError(f"Controlled run rotation {role}.{count_name} is invalid")
        if type(role_value["fold_ids"]) is not list or not role_value["fold_ids"]:
            raise ValueError(f"Controlled run rotation {role}.fold_ids is invalid")
        role_cases.append(set(case_ids))
    if any(role_cases[left] & role_cases[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Controlled run fold roles overlap")
    if sum(len(case_ids) for case_ids in role_cases) != 42:
        raise ValueError("Controlled run fold rotation does not cover exactly 42 cases")
    if rotation["train"]["queries"] != 294:
        raise ValueError("Controlled run training role must contain exactly 294 queries")
    if rotation["validation"]["queries"] != 98 or rotation["test"]["queries"] != 98:
        raise ValueError("Controlled run validation/test roles must each contain exactly 98 queries")
    if sum(rotation[role]["passages"] for role in ("train", "validation", "test")) != 5_286:
        raise ValueError("Controlled run fold roles do not cover exactly 5,286 passages")
    rotation_sha256 = hashlib.sha256(_canonical_json(rotation).encode("utf-8")).hexdigest()
    expected_rotation_sha256 = EXPECTED_FOLD_ROTATION_SHA256_BY_OUTER_FOLD[
        expectation.outer_fold
    ]
    if rotation_sha256 != expected_rotation_sha256:
        raise ValueError("Controlled run rotation left the frozen fold manifest")

    snapshot = run["snapshot"]
    if type(snapshot) is not dict or set(snapshot) != {
        "manifest_path",
        "manifest_sha256",
        "tree_sha256",
    }:
        raise ValueError("Controlled run snapshot schema changed")
    for digest_name in ("manifest_sha256", "tree_sha256"):
        _require_sha256(snapshot[digest_name], name=f"controlled_run.snapshot.{digest_name}")
    if (
        snapshot["manifest_path"] != "modernbert_snapshot.json"
        or snapshot["manifest_sha256"] != EXPECTED_SNAPSHOT_MANIFEST_SHA256
        or snapshot["tree_sha256"] != EXPECTED_SNAPSHOT_TREE_SHA256
    ):
        raise ValueError("Controlled run snapshot left the frozen ModernBERT artifact")

    passage_index = run["passage_index"]
    if type(passage_index) is not dict or set(passage_index) != {
        "schema_version",
        "size",
        "sha256",
    }:
        raise ValueError("Controlled run passage-index schema changed")
    if passage_index != {
        "schema_version": 1,
        "size": 5_286,
        "sha256": expectation.passage_index_sha256,
    }:
        raise ValueError("Controlled run passage-index identity changed")

    validation_data = run["validation_data"]
    expected_validation_keys = {
        "role",
        "query_view",
        "case_count",
        "query_count",
        "passage_count",
        "case_ids_sha256",
        "query_ids_sha256",
        "passage_ids_sha256",
        "contract_sha256",
    }
    if type(validation_data) is not dict or set(validation_data) != expected_validation_keys:
        raise ValueError("Controlled run validation-data schema changed")
    if validation_data["role"] != "validation" or validation_data["query_view"] != expectation.query_view:
        raise ValueError("Controlled run validation-data role/query view changed")
    validation_role = rotation["validation"]
    if (
        validation_data["case_count"] != validation_role["num_cases"]
        or validation_data["query_count"] != validation_role["queries"]
        or validation_data["passage_count"] != validation_role["passages"]
    ):
        raise ValueError("Controlled run validation-data counts disagree with the fold rotation")
    for digest_name in (
        "case_ids_sha256",
        "query_ids_sha256",
        "passage_ids_sha256",
        "contract_sha256",
    ):
        _require_sha256(validation_data[digest_name], name=f"validation_data.{digest_name}")
    expected_validation_identity = EXPECTED_VALIDATION_IDENTITY_BY_CELL[
        (expectation.outer_fold, expectation.query_view)
    ]
    if {
        digest_name: validation_data[digest_name]
        for digest_name in expected_validation_identity
    } != expected_validation_identity:
        raise ValueError("Controlled run validation-data identity left the frozen role")

    candidate_traces = run["candidate_traces"]
    if type(candidate_traces) is not dict or set(candidate_traces) != {
        "manifest_path",
        "manifest_sha256",
        "record_count",
        "merged_sha256",
    }:
        raise ValueError("Controlled run candidate-trace schema changed")
    if (
        candidate_traces["manifest_path"] != "candidate_traces/manifest.json"
        or candidate_traces["manifest_sha256"]
        != manifest["candidate_trace_manifest"]["sha256"]
    ):
        raise ValueError("Controlled run candidate-trace manifest identity changed")
    _require_sha256(candidate_traces["merged_sha256"], name="candidate_traces.merged_sha256")
    if (
        candidate_traces["merged_sha256"] != trace_manifest["merged"]["sha256"]
        or trace_manifest["query_ids_sha256"]
        != EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD[expectation.outer_fold]
    ):
        raise ValueError("Controlled run candidate traces left the frozen training role")
    expected_trace_count = CONTROLLED_EPOCHS * rotation["train"]["queries"]
    if candidate_traces["record_count"] != expected_trace_count:
        raise ValueError("Controlled run candidate-trace coverage changed")

    validation_history = run["validation_history"]
    if type(validation_history) is not dict or set(validation_history) != {
        "manifest_path",
        "manifest_sha256",
        "best",
        "last",
        "retained_checkpoint_dirs",
    }:
        raise ValueError("Controlled run validation-history schema changed")
    if (
        validation_history["manifest_path"] != "validation/manifest.json"
        or validation_history["manifest_sha256"] != manifest["validation_manifest"]["sha256"]
        or validation_history["best"] != validation_manifest["best"]
        or validation_history["last"] != validation_manifest["last"]
        or validation_history["retained_checkpoint_dirs"] != list(retained_names)
    ):
        raise ValueError("Controlled run validation-history identity changed")

    reload = run["best_checkpoint_reload"]
    if type(reload) is not dict or set(reload) != {"selection", "validation_result", "per_rank"}:
        raise ValueError("Controlled run best-checkpoint reload schema changed")
    if reload["selection"] != validation_manifest["best"]:
        raise ValueError("Controlled run reloaded checkpoint is not the selected best")
    if type(reload["validation_result"]) is not dict or not reload["validation_result"]:
        raise ValueError("Controlled run best-checkpoint validation result is missing")
    per_rank = reload["per_rank"]
    expected_reload_keys = {
        "rank",
        "load_path_parent",
        "client_state_sha256",
        "scheduler_state_sha256",
        "global_step",
        "rng_sha256",
        "manifest_sha256",
    }
    if type(per_rank) is not list or len(per_rank) != CONTROLLED_WORLD_SIZE:
        raise ValueError("Controlled run best-checkpoint per-rank reload coverage changed")
    for rank, rank_record in enumerate(per_rank):
        if type(rank_record) is not dict or set(rank_record) != expected_reload_keys:
            raise ValueError(f"Controlled run reload rank {rank} schema changed")
        if rank_record["rank"] != rank or rank_record["global_step"] != reload["selection"]["global_step"]:
            raise ValueError(f"Controlled run reload rank {rank} identity changed")
        if type(rank_record["load_path_parent"]) is not str or not rank_record["load_path_parent"]:
            raise ValueError(f"Controlled run reload rank {rank} path is invalid")
        for digest_name in (
            "client_state_sha256",
            "scheduler_state_sha256",
            "rng_sha256",
            "manifest_sha256",
        ):
            _require_sha256(rank_record[digest_name], name=f"reload[{rank}].{digest_name}")

    final_model = run["final_model"]
    if type(final_model) is not dict or set(final_model) != {
        "path",
        "size",
        "sha256",
        "weight_dtype",
        "gathered_tensor_count",
        "strict_round_trip_tensor_count",
    }:
        raise ValueError("Controlled run final-model schema changed")
    if {key: final_model[key] for key in _FILE_RECORD_KEYS} != manifest["model"]:
        raise ValueError("Controlled run final-model record and artifact manifest disagree")
    if final_model["weight_dtype"] != "bfloat16":
        raise ValueError("Controlled run final-model dtype changed")
    gathered = final_model["gathered_tensor_count"]
    round_trip = final_model["strict_round_trip_tensor_count"]
    if (
        type(gathered) is not int
        or gathered != CONTROLLED_MODEL_STATE_COUNT
        or round_trip != gathered
    ):
        raise ValueError("Controlled run strict model-state inventory changed")
    if run["tokenizer"] != manifest["tokenizer"]:
        raise ValueError("Controlled run tokenizer record and artifact manifest disagree")
    if run["encoder_config"] != manifest["encoder_config"]:
        raise ValueError("Controlled run encoder-config record and artifact manifest disagree")
    if run["wrapper_config"] != manifest["wrapper_config"]:
        raise ValueError("Controlled run wrapper record and artifact manifest disagree")
    if run["retained_checkpoints"] != manifest["retained_checkpoints"]:
        raise ValueError("Controlled run retained-checkpoint inventory changed")
    if wrapper["model_artifact_protocol"] != expectation.model_artifact_protocol:
        raise ValueError("Controlled wrapper and expected model artifact protocol disagree")


def validate_controlled_artifact(
    root: Path,
    *,
    expectation: ControlledArtifactExpectation,
) -> ValidatedControlledArtifact:
    """Validate one committed Step 6 artifact without importing ML dependencies."""

    if not isinstance(expectation, ControlledArtifactExpectation):
        raise TypeError("expectation must be ControlledArtifactExpectation")
    if not isinstance(root, Path):
        raise TypeError("Controlled artifact root must be a Path")
    root = root.expanduser()
    if root.is_symlink():
        raise ValueError(f"Controlled artifact root must not be a symlink: {root}")
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Controlled artifact root must be a real non-symlink directory: {root}")

    initial_inventory = _regular_tree_inventory(root)
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Controlled artifact is incomplete: artifact_manifest.json is absent")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != expectation.artifact_manifest_sha256:
        raise ValueError(
            "Controlled artifact commit-marker SHA-256 changed: "
            f"actual={actual_manifest_sha256}, "
            f"expected={expectation.artifact_manifest_sha256}"
        )
    manifest = _load_json_object(
        manifest_path,
        name="Controlled artifact commit marker",
        require_canonical=True,
    )
    if set(manifest) != set(_ARTIFACT_MANIFEST_KEYS):
        raise ValueError("Controlled artifact manifest schema changed")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != CONTROLLED_ARTIFACT_SCHEMA_VERSION
        or type(manifest["commit_marker"]) is not bool
        or manifest["commit_marker"] is not True
    ):
        raise ValueError("Controlled artifact commit marker is invalid")

    direct_records = [
        _validate_file_record(
            root,
            manifest["controlled_run"],
            name="artifact_manifest.controlled_run",
            expected_path="controlled_run.json",
        ),
        _validate_file_record(
            root,
            manifest["model"],
            name="artifact_manifest.model",
            expected_path="model.safetensors",
        ),
        _validate_file_record(
            root,
            manifest["wrapper_config"],
            name="artifact_manifest.wrapper_config",
            expected_path="wrapper_config.json",
        ),
    ]
    tokenizer_records = _validate_directory_record(
        root,
        manifest["tokenizer"],
        name="artifact_manifest.tokenizer",
        expected_path="tokenizer",
        expected_files=(
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    )
    encoder_records = _validate_directory_record(
        root,
        manifest["encoder_config"],
        name="artifact_manifest.encoder_config",
        expected_path="encoder_config",
        expected_files=("config.json",),
    )
    checkpoint_records, retained_names = _validate_checkpoint_inventory(
        root, manifest["retained_checkpoints"]
    )

    run_path = root / "controlled_run.json"
    wrapper_path = root / "wrapper_config.json"
    wrapper = _validate_wrapper(wrapper_path, expectation=expectation)
    run = _load_json_object(
        run_path,
        name="Controlled run metadata",
        require_canonical=True,
    )
    rotation = run.get("folds", {}).get("rotation") if type(run.get("folds")) is dict else None
    expected_train_queries = (
        rotation.get("train", {}).get("queries")
        if type(rotation) is dict and type(rotation.get("train")) is dict
        else None
    )
    if type(expected_train_queries) is not int or expected_train_queries < 1:
        raise ValueError("Controlled run does not declare a valid training-query count")
    trace_records, trace_manifest = _validate_trace_manifest(
        root,
        manifest["candidate_trace_manifest"],
        expected_passage_index_sha256=expectation.passage_index_sha256,
        expected_queries_per_epoch=expected_train_queries,
    )
    validation_records, validation_manifest = _validate_validation_manifest(
        root, manifest["validation_manifest"]
    )
    if list(retained_names) != validation_manifest["retained_checkpoint_dirs"]:
        raise ValueError("Retained checkpoint inventory and validation manifest disagree")
    _validate_controlled_run(
        run,
        expectation=expectation,
        manifest=manifest,
        wrapper=wrapper,
        trace_manifest=trace_manifest,
        validation_manifest=validation_manifest,
        retained_names=retained_names,
    )

    top_level_expected = {
        "artifact_manifest.json",
        "controlled_run.json",
        "model.safetensors",
        "wrapper_config.json",
        "tokenizer",
        "encoder_config",
        "candidate_traces",
        "validation",
        *retained_names,
    }
    top_level_actual = {entry.name for entry in root.iterdir()}
    if top_level_actual != top_level_expected:
        raise ValueError(
            "Controlled artifact top-level inventory changed: "
            f"actual={sorted(top_level_actual)}, expected={sorted(top_level_expected)}"
        )

    encoder_payload = _load_json_object(
        root / "encoder_config/config.json",
        name="Controlled encoder config",
        require_canonical=False,
    )
    required_encoder_values = {
        "model_type": "modernbert",
        "vocab_size": CONTROLLED_TOKENIZER_SIZE,
        "deterministic_flash_attn": True,
        "reference_compile": False,
        # Transformers 4.49 retains the base snapshot's config metadata here
        # even when from_pretrained(..., torch_dtype=bfloat16) creates an
        # all-BF16 state. The wrapper, safetensors state, and explicit factory
        # dtype below are the authoritative artifact dtype contract.
        "torch_dtype": "float32",
    }
    for key, expected in required_encoder_values.items():
        if type(encoder_payload.get(key)) is not type(expected) or encoder_payload.get(key) != expected:
            raise ValueError(
                f"Controlled encoder config field {key!r} changed: "
                f"actual={encoder_payload.get(key)!r}, expected={expected!r}"
            )

    final_inventory = _regular_tree_inventory(root)
    if final_inventory != initial_inventory:
        raise RuntimeError("Controlled artifact bytes changed during manifest validation")
    expected_records = {
        "artifact_manifest.json": ArtifactFileRecord(
            path="artifact_manifest.json",
            size=manifest_path.stat().st_size,
            sha256=actual_manifest_sha256,
        ),
        **{record.path: record for record in direct_records},
        **{record.path: record for record in tokenizer_records},
        **{record.path: record for record in encoder_records},
        **{record.path: record for record in trace_records},
        **{record.path: record for record in validation_records},
        **{record.path: record for record in checkpoint_records},
    }
    actual_records = {record.path: record for record in final_inventory}
    if actual_records != expected_records:
        missing = sorted(set(expected_records) - set(actual_records))
        unexpected = sorted(set(actual_records) - set(expected_records))
        changed = sorted(
            path
            for path in set(actual_records).intersection(expected_records)
            if actual_records[path] != expected_records[path]
        )
        raise ValueError(
            "Controlled artifact complete inventory does not match its manifests: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )

    tokenizer_inventory_sha256 = hashlib.sha256(
        _canonical_json([_record_payload(record) for record in tokenizer_records]).encode("utf-8")
    ).hexdigest()
    encoder_inventory_sha256 = hashlib.sha256(
        _canonical_json([_record_payload(record) for record in encoder_records]).encode("utf-8")
    ).hexdigest()
    identity = ControlledArtifactIdentity(
        artifact_manifest_sha256=actual_manifest_sha256,
        controlled_run_sha256=manifest["controlled_run"]["sha256"],
        model_sha256=manifest["model"]["sha256"],
        wrapper_config_sha256=manifest["wrapper_config"]["sha256"],
        tokenizer_inventory_sha256=tokenizer_inventory_sha256,
        encoder_config_inventory_sha256=encoder_inventory_sha256,
        experiment_id=expectation.experiment_id,
        outer_fold=expectation.outer_fold,
        query_view=expectation.query_view,
        sampler=expectation.sampler,
        experiment_seed=expectation.experiment_seed,
        experiment_config_sha256=run["experiment_config"]["sha256"],
        deepspeed_config_sha256=run["deepspeed_config"]["sha256"],
        dataset_manifest_sha256=expectation.dataset_manifest_sha256,
        dataset_output_sha256=hashlib.sha256(
            _canonical_json(run["dataset"]["output_sha256"]).encode("utf-8")
        ).hexdigest(),
        fold_manifest_sha256=expectation.fold_manifest_sha256,
        fold_rotation_sha256=hashlib.sha256(
            _canonical_json(run["folds"]["rotation"]).encode("utf-8")
        ).hexdigest(),
        passage_index_sha256=expectation.passage_index_sha256,
        snapshot_manifest_sha256=run["snapshot"]["manifest_sha256"],
        snapshot_tree_sha256=run["snapshot"]["tree_sha256"],
        model_artifact_protocol=expectation.model_artifact_protocol,
    )
    return ValidatedControlledArtifact(
        root=root,
        expectation=expectation,
        identity=identity,
        files=final_inventory,
        model_path=root / "model.safetensors",
        tokenizer_dir=root / "tokenizer",
        encoder_config_dir=root / "encoder_config",
        wrapper_config_path=wrapper_path,
        controlled_run_path=run_path,
        slot_token_id=wrapper["slot_token_id"],
        model_state_count=run["final_model"]["gathered_tensor_count"],
    )


def import_pinned_artifact_runtime() -> ControlledArtifactRuntime:
    """Import the exact controlled runtime only when model loading is requested."""

    from .provenance import EXPECTED_BASE_RUNTIME_VERSIONS

    required_packages = (
        "python",
        "torch",
        "transformers",
        "numpy",
        "flash-attn",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
    )
    actual = {"python": platform.python_version()}
    for package in required_packages:
        if package == "python":
            continue
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Controlled artifact runtime package is absent: {package}") from error
    expected = {package: EXPECTED_BASE_RUNTIME_VERSIONS[package] for package in required_packages}
    if actual != expected:
        raise RuntimeError(
            "Controlled artifact runtime does not match the frozen inventory: "
            f"actual={actual}, expected={expected}"
        )

    import torch
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from .models import DualEncoderRetriever

    return ControlledArtifactRuntime(
        torch_module=torch,
        auto_tokenizer_class=AutoTokenizer,
        auto_config_class=AutoConfig,
        auto_model_class=AutoModel,
        load_safetensors_model=load_model,
        retriever_class=DualEncoderRetriever,
    )


def _validate_runtime_bundle(runtime: ControlledArtifactRuntime) -> None:
    if not isinstance(runtime, ControlledArtifactRuntime):
        raise TypeError("runtime must be ControlledArtifactRuntime")
    for name in (
        "torch_module",
        "auto_tokenizer_class",
        "auto_config_class",
        "auto_model_class",
        "load_safetensors_model",
        "retriever_class",
    ):
        if getattr(runtime, name) is None:
            raise ValueError(f"Controlled artifact runtime dependency is absent: {name}")


def _validate_bf16_state(model: Any, torch_module: Any, *, context: str) -> int:
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"{context} returned an empty or invalid state dict")
    floating = 0
    for name, tensor in state.items():
        if type(name) is not str or not name:
            raise RuntimeError(f"{context} contains an invalid state key")
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"{context} state {name!r} is not a tensor")
        if tensor.is_floating_point():
            floating += 1
            if tensor.dtype != torch_module.bfloat16:
                raise TypeError(
                    f"{context} floating state {name!r} has dtype={tensor.dtype}; "
                    "expected torch.bfloat16"
                )
            if not torch_module.isfinite(tensor).all():
                raise FloatingPointError(f"{context} state {name!r} contains non-finite values")
    if floating < 1:
        raise RuntimeError(f"{context} contains no floating model state")
    return len(state)


def load_controlled_retriever(
    artifact: ValidatedControlledArtifact,
    *,
    device: str,
    runtime: ControlledArtifactRuntime,
) -> LoadedControlledRetriever:
    """Strictly construct and load one committed Step 6 BF16 retriever."""

    if not isinstance(artifact, ValidatedControlledArtifact):
        raise TypeError("artifact must be ValidatedControlledArtifact")
    _validate_runtime_bundle(runtime)
    if type(device) is not str or not device or device.strip() != device or device == "auto":
        raise ValueError("Controlled artifact loading requires one explicit exact device string")
    torch_module = runtime.torch_module
    try:
        target_device = torch_module.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid explicit controlled artifact device: {device!r}") from error
    if str(target_device) != device:
        raise ValueError(
            f"Controlled artifact device must be canonical: actual={device!r}, "
            f"canonical={str(target_device)!r}"
        )
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Controlled artifact device must be explicit CPU or CUDA")
    if target_device.type == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Explicit CUDA artifact device is unavailable")
        index = target_device.index
        if index is not None and index not in range(torch_module.cuda.device_count()):
            raise RuntimeError(f"Explicit CUDA device index is unavailable: {index}")

    before = validate_controlled_artifact(
        artifact.root,
        expectation=artifact.expectation,
    )
    if before.identity != artifact.identity or before.files != artifact.files:
        raise RuntimeError("Controlled artifact identity changed before model loading")

    tokenizer = runtime.auto_tokenizer_class.from_pretrained(
        str(artifact.tokenizer_dir),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if len(tokenizer) != CONTROLLED_TOKENIZER_SIZE:
        raise RuntimeError(
            f"Controlled tokenizer size changed: {len(tokenizer)}; "
            f"expected {CONTROLLED_TOKENIZER_SIZE}"
        )
    slot_token_id = tokenizer.convert_tokens_to_ids(CONTROLLED_SLOT_TOKEN)
    if type(slot_token_id) is not int or slot_token_id != artifact.slot_token_id:
        raise RuntimeError(
            "Controlled tokenizer slot-token ID changed: "
            f"actual={slot_token_id!r}, expected={artifact.slot_token_id!r}"
        )
    if slot_token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError("Controlled slot token resolves to the unknown token")

    config = runtime.auto_config_class.from_pretrained(
        str(artifact.encoder_config_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    required_config = {
        "model_type": "modernbert",
        "vocab_size": CONTROLLED_TOKENIZER_SIZE,
        "deterministic_flash_attn": True,
        "reference_compile": False,
    }
    for key, expected in required_config.items():
        if getattr(config, key, None) != expected:
            raise RuntimeError(
                f"Loaded controlled encoder config {key!r} changed: "
                f"actual={getattr(config, key, None)!r}, expected={expected!r}"
            )

    encoder = runtime.auto_model_class.from_config(
        config,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_module.bfloat16,
    )
    if getattr(encoder.config, "_attn_implementation", None) != "flash_attention_2":
        raise RuntimeError("Controlled encoder did not resolve flash_attention_2")
    if getattr(encoder.config, "deterministic_flash_attn", None) is not True:
        raise RuntimeError("Controlled encoder lost deterministic FlashAttention")
    if getattr(encoder.config, "reference_compile", None) is not False:
        raise RuntimeError("Controlled encoder reference_compile changed")
    module_flags = [
        module.deterministic_flash_attn
        for module in encoder.modules()
        if hasattr(module, "deterministic_flash_attn")
    ]
    if (
        len(module_flags) != CONTROLLED_ATTENTION_MODULE_COUNT
        or any(flag is not True for flag in module_flags)
    ):
        raise RuntimeError(
            "Controlled attention-module inventory changed: "
            f"count={len(module_flags)}, expected={CONTROLLED_ATTENTION_MODULE_COUNT}"
        )
    embedding_rows = len(encoder.get_input_embeddings().weight)
    if embedding_rows != CONTROLLED_TOKENIZER_SIZE:
        raise RuntimeError(
            f"Controlled encoder embedding rows changed: {embedding_rows}; "
            f"expected {CONTROLLED_TOKENIZER_SIZE}"
        )

    model = runtime.retriever_class(
        encoder=encoder,
        slot_token_id=slot_token_id,
        temperature=CONTROLLED_TEMPERATURE,
    )
    partitioned = [
        name for name, parameter in model.named_parameters() if hasattr(parameter, "ds_id")
    ]
    if partitioned:
        raise RuntimeError(
            "Controlled evaluation model must be unpartitioned; found ZeRO parameters: "
            f"{partitioned[:5]}"
        )
    expected_state_count = _validate_bf16_state(
        model, torch_module, context="Fresh controlled evaluation model"
    )
    if expected_state_count != artifact.model_state_count:
        raise RuntimeError(
            "Fresh controlled model-state inventory changed: "
            f"actual={expected_state_count}, expected={artifact.model_state_count}"
        )
    incompatibilities = runtime.load_safetensors_model(
        model,
        artifact.model_path,
        strict=True,
        device="cpu",
    )
    if (
        not isinstance(incompatibilities, tuple)
        or len(incompatibilities) != 2
        or list(incompatibilities[0])
        or list(incompatibilities[1])
    ):
        raise RuntimeError(
            "Strict tied-weight safetensors load was incomplete: "
            f"{incompatibilities!r}"
        )
    loaded_state_count = _validate_bf16_state(
        model, torch_module, context="Loaded controlled evaluation model"
    )
    if loaded_state_count != expected_state_count:
        raise RuntimeError("Strict safetensors load changed the model-state inventory")
    model.to(target_device)
    model.eval()
    _validate_bf16_state(model, torch_module, context="Device-resident controlled model")

    after = validate_controlled_artifact(
        artifact.root,
        expectation=artifact.expectation,
    )
    if after.identity != artifact.identity or after.files != artifact.files:
        raise RuntimeError("Controlled artifact identity changed during model loading")
    return LoadedControlledRetriever(
        model=model,
        tokenizer=tokenizer,
        identity=artifact.identity,
        device=device,
    )
