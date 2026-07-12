from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import ControlledArtifactRuntime
from .markup import SLOT_TOKEN, all_markup_tokens


FIXED_BASE_ARTIFACT_SCHEMA_VERSION = 1
FIXED_BASE_ARTIFACT_PROTOCOL = "fixed_seed_untrained_modernbert_dual_encoder_bf16_v1"
FIXED_BASE_SEED = 17
FIXED_BASE_QUERY_VIEW = "flat_masked"
FIXED_BASE_TOKENIZER_SIZE = 50_386
FIXED_BASE_MODEL_STATE_COUNT = 134
FIXED_BASE_TEMPERATURE = 0.07
FIXED_BASELINE_CONFIG_SHA256 = (
    "714b8c18e9e32130ebf3358a72d9c6aceceeb1e14ee0d76270306d901b81f33a"
)
FIXED_BASE_MARKUP_TOKEN_IDS = (
    50_284,
    50_368,
    50_369,
    50_370,
    50_371,
    50_372,
    50_373,
    50_374,
    50_375,
    50_376,
    50_377,
    50_378,
    50_379,
    50_380,
    50_381,
    50_382,
    50_383,
    50_384,
    50_385,
)
FIXED_BASE_RUNTIME_VERSIONS = {
    "python": "3.11.10",
    "torch": "2.5.1+cu124",
    "numpy": "1.26.4",
    "flash-attn": "2.7.3",
    "huggingface-hub": "0.29.1",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "transformers": "4.49.0",
}
MODERNBERT_MODEL_ID = "answerdotai/ModernBERT-base"
MODERNBERT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MODERNBERT_LOGICAL_IDENTITY = f"{MODERNBERT_MODEL_ID}@{MODERNBERT_REVISION}"
MODERNBERT_SNAPSHOT_MANIFEST_SHA256 = (
    "0807d16ba5b49a5e30c8b09b72acef7d8c6326823a850640027cc1363ee446b5"
)
MODERNBERT_SNAPSHOT_TREE_SHA256 = (
    "aca85feea4adb60c4b021eb1a439aff47c844495005f2acdee1baef9d611d63d"
)
E5_MODEL_ID = "intfloat/e5-base-v2"
E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
E5_SNAPSHOT_MANIFEST_SHA256 = (
    "7629cf8c8bf60569d72f653d21a4c47a8fa806d8fd907db05c65a3288b24b635"
)
E5_SNAPSHOT_TREE_SHA256 = (
    "1181a9758ea858d6679df0e04f6ac67b26dab90e91f63e76238c2eecec1c1a61"
)
E5_MODEL_STATE_COUNT = 200
E5_TOKENIZER_SIZE = 30_522


@dataclass(frozen=True)
class SnapshotIdentity:
    manifest_path: Path
    manifest_sha256: str
    model_id: str
    revision: str
    tree_sha256: str
    files: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class LoadedE5Encoder:
    model: Any
    tokenizer: Any
    snapshot_identity: SnapshotIdentity
    device: str


@dataclass(frozen=True)
class FixedBaseArtifactExpectation:
    artifact_manifest_sha256: str
    baseline_config_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.artifact_manifest_sha256,
            name="fixed_base.artifact_manifest_sha256",
        )
        _require_sha256(
            self.baseline_config_sha256,
            name="fixed_base.baseline_config_sha256",
        )


@dataclass(frozen=True)
class ValidatedFixedBaseArtifact:
    root: Path
    expectation: FixedBaseArtifactExpectation
    manifest_sha256: str
    model_sha256: str
    model_path: Path
    tokenizer_dir: Path
    encoder_config_dir: Path
    wrapper_config_path: Path
    run_path: Path
    slot_token_id: int
    state_key_sha256: str
    new_embedding_rows_sha256: str


@dataclass(frozen=True)
class LoadedFixedBaseRetriever:
    model: Any
    tokenizer: Any
    artifact: ValidatedFixedBaseArtifact
    device: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_pretty_bytes(value: object) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _load_canonical_json_with_sha256(
    path: Path,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid JSON: {path}") from error
    if type(value) is not dict or raw != _canonical_pretty_bytes(value):
        raise ValueError(f"{name} is not canonical pretty JSON: {path}")
    return value, _sha256_bytes(raw)


def _load_canonical_json(path: Path, *, name: str) -> dict[str, Any]:
    value, _ = _load_canonical_json_with_sha256(path, name=name)
    return value


def _load_fixed_base_build_contract(path: Path) -> dict[str, Any]:
    contract, _ = _load_canonical_json_with_sha256(
        path,
        name="fixed-base build contract",
    )
    expected = {
        "schema_version": 1,
        "model_artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
        "fixed_initialization_seed": FIXED_BASE_SEED,
        "query_view": FIXED_BASE_QUERY_VIEW,
        "baseline_config_sha256": FIXED_BASELINE_CONFIG_SHA256,
        "snapshot_manifest_sha256": MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
        "snapshot_tree_sha256": MODERNBERT_SNAPSHOT_TREE_SHA256,
        "weight_dtype": "bfloat16",
    }
    required_keys = {
        *expected,
        "artifact_manifest_sha256",
        "model_sha256",
        "state_key_sha256",
        "new_embedding_rows_sha256",
    }
    if set(contract) != required_keys:
        raise ValueError("Fixed-base build contract schema changed")
    for key, expected_value in expected.items():
        if contract[key] != expected_value or type(contract[key]) is not type(
            expected_value
        ):
            raise ValueError(f"Fixed-base build contract {key} changed")
    for key in (
        "artifact_manifest_sha256",
        "model_sha256",
        "state_key_sha256",
        "new_embedding_rows_sha256",
    ):
        _require_sha256(contract[key], name=f"fixed-base build contract {key}")
    return contract


def validate_snapshot(
    *,
    snapshot_dir: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_model_id: str,
    expected_revision: str,
    expected_tree_sha256: str,
) -> SnapshotIdentity:
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        name="snapshot.expected_manifest_sha256",
    )
    expected_tree_sha256 = _require_sha256(
        expected_tree_sha256,
        name="snapshot.expected_tree_sha256",
    )
    manifest_path = Path(manifest_path)
    manifest, manifest_sha256 = _load_canonical_json_with_sha256(
        manifest_path,
        name="snapshot manifest",
    )
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Snapshot manifest hash changed")
    if set(manifest) != {
        "schema_version",
        "manifest_type",
        "model_id",
        "revision",
        "tree_sha256",
        "files",
    }:
        raise ValueError("Snapshot manifest schema changed")
    expected_identity = {
        "schema_version": 1,
        "manifest_type": "huggingface_model_snapshot",
        "model_id": expected_model_id,
        "revision": expected_revision,
        "tree_sha256": expected_tree_sha256,
    }
    for key, expected in expected_identity.items():
        if manifest[key] != expected or type(manifest[key]) is not type(expected):
            raise ValueError(f"Snapshot manifest {key} changed")
    raw_files = manifest["files"]
    if type(raw_files) is not list or not raw_files:
        raise ValueError("Snapshot manifest files must be a non-empty list")
    files: list[tuple[str, int, str]] = []
    for position, record in enumerate(raw_files):
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError(f"Snapshot file record {position} has an invalid schema")
        path = record["path"]
        size = record["size"]
        sha256 = record["sha256"]
        if (
            type(path) is not str
            or path != Path(path).name
            or not path
            or type(size) is not int
            or size < 1
        ):
            raise ValueError(f"Snapshot file record {position} is invalid")
        _require_sha256(sha256, name=f"snapshot.files[{position}].sha256")
        files.append((path, size, sha256))
    if files != sorted(files) or len(files) != len({record[0] for record in files}):
        raise ValueError("Snapshot file records must be unique and sorted")
    tree_payload = [
        {"path": path, "size": size, "sha256": sha256}
        for path, size, sha256 in files
    ]
    if _sha256_bytes(_canonical_json(tree_payload).encode("utf-8")) != expected_tree_sha256:
        raise ValueError("Snapshot tree hash does not match its file records")

    snapshot_dir = Path(snapshot_dir)
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise ValueError(f"Snapshot must be a real directory: {snapshot_dir}")
    if sorted(path.name for path in snapshot_dir.iterdir()) != [record[0] for record in files]:
        raise ValueError("Snapshot directory inventory changed")
    for name, expected_size, expected_sha256 in files:
        path = snapshot_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Snapshot entry must be a regular non-symlink file: {path}")
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha256:
            raise ValueError(f"Snapshot file bytes changed: {name}")
    return SnapshotIdentity(
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha256,
        model_id=expected_model_id,
        revision=expected_revision,
        tree_sha256=expected_tree_sha256,
        files=tuple(files),
    )


def _explicit_device(device: str, torch_module: Any):
    if type(device) is not str or not device or device.strip() != device or device == "auto":
        raise ValueError("Baseline loading requires one explicit canonical device")
    resolved = torch_module.device(device)
    if str(resolved) != device or resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported or noncanonical baseline device={device!r}")
    if resolved.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(f"Requested baseline CUDA device is unavailable: {device}")
    return resolved


def _validate_floating_state(
    model: Any,
    *,
    torch_module: Any,
    dtype: Any,
    expected_count: int,
    context: str,
) -> Mapping[str, Any]:
    state = model.state_dict()
    if not isinstance(state, Mapping) or len(state) != expected_count:
        raise RuntimeError(
            f"{context} state count changed: actual={len(state)}, expected={expected_count}"
        )
    for name, tensor in state.items():
        if type(name) is not str or not name or not torch_module.is_tensor(tensor):
            raise TypeError(f"{context} contains an invalid state entry")
        if tensor.is_floating_point():
            if tensor.dtype != dtype:
                raise TypeError(
                    f"{context} state {name!r} has dtype={tensor.dtype}; expected={dtype}"
                )
            if not bool(torch_module.isfinite(tensor).all().item()):
                raise FloatingPointError(f"{context} state {name!r} is non-finite")
    return state


def load_e5_encoder(
    *,
    snapshot_dir: Path,
    manifest_path: Path,
    device: str,
    runtime: ControlledArtifactRuntime,
) -> LoadedE5Encoder:
    if not isinstance(runtime, ControlledArtifactRuntime):
        raise TypeError("runtime must be ControlledArtifactRuntime")
    identity = validate_snapshot(
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        expected_manifest_sha256=E5_SNAPSHOT_MANIFEST_SHA256,
        expected_model_id=E5_MODEL_ID,
        expected_revision=E5_REVISION,
        expected_tree_sha256=E5_SNAPSHOT_TREE_SHA256,
    )
    torch_module = runtime.torch_module
    resolved_device = _explicit_device(device, torch_module)
    tokenizer = runtime.auto_tokenizer_class.from_pretrained(
        str(snapshot_dir),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if (
        getattr(tokenizer, "is_fast", None) is not True
        or len(tokenizer) != E5_TOKENIZER_SIZE
        or tokenizer.model_max_length != 512
        or tokenizer.padding_side != "right"
        or tokenizer.truncation_side != "right"
        or tokenizer("query: ", add_special_tokens=False).input_ids != [23_032, 1_024]
    ):
        raise RuntimeError("Frozen E5 tokenizer contract changed")
    config = runtime.auto_config_class.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    required_config = {
        "model_type": "bert",
        "vocab_size": E5_TOKENIZER_SIZE,
        "max_position_embeddings": 512,
        "hidden_size": 768,
    }
    for name, expected in required_config.items():
        if getattr(config, name, None) != expected:
            raise RuntimeError(f"Frozen E5 config {name} changed")
    model = runtime.auto_model_class.from_config(
        config,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch_module.float32,
    )
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Frozen E5 encoder did not resolve eager attention")
    embeddings = getattr(model, "embeddings", None)
    if (
        embeddings is None
        or "position_ids" not in embeddings._buffers
        or "position_ids" not in embeddings._non_persistent_buffers_set
    ):
        raise RuntimeError("Frozen E5 position-ID buffer contract changed")
    embeddings._non_persistent_buffers_set.remove("position_ids")
    _validate_floating_state(
        model,
        torch_module=torch_module,
        dtype=torch_module.float32,
        expected_count=E5_MODEL_STATE_COUNT,
        context="Fresh E5",
    )
    incompatibilities = runtime.load_safetensors_model(
        model,
        Path(snapshot_dir) / "model.safetensors",
        strict=True,
        device="cpu",
    )
    if incompatibilities != (set(), set()):
        if (
            not isinstance(incompatibilities, tuple)
            or len(incompatibilities) != 2
            or list(incompatibilities[0])
            or list(incompatibilities[1])
        ):
            raise RuntimeError(f"Strict E5 safetensors load was incomplete: {incompatibilities}")
    _validate_floating_state(
        model,
        torch_module=torch_module,
        dtype=torch_module.float32,
        expected_count=E5_MODEL_STATE_COUNT,
        context="Loaded E5",
    )
    model.to(resolved_device)
    model.eval()
    after_identity = validate_snapshot(
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        expected_manifest_sha256=E5_SNAPSHOT_MANIFEST_SHA256,
        expected_model_id=E5_MODEL_ID,
        expected_revision=E5_REVISION,
        expected_tree_sha256=E5_SNAPSHOT_TREE_SHA256,
    )
    if after_identity != identity:
        raise RuntimeError("Frozen E5 snapshot identity changed during loading")
    return LoadedE5Encoder(
        model=model,
        tokenizer=tokenizer,
        snapshot_identity=after_identity,
        device=device,
    )


def _add_markup_tokens(tokenizer: Any) -> int:
    tokens = all_markup_tokens()
    if len(tokenizer) != 50_368 or len(tokens) != 19 or len(tokens) != len(set(tokens)):
        raise RuntimeError("Frozen ModernBERT tokenizer input contract changed")
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if added != 19 or len(tokenizer) != FIXED_BASE_TOKENIZER_SIZE:
        raise RuntimeError("Frozen ModernBERT tokenizer extension changed")
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    if len(token_ids) != len(set(token_ids)) or any(
        token_id == tokenizer.unk_token_id for token_id in token_ids
    ):
        raise RuntimeError("Frozen markup-token identities changed")
    slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
    if slot_token_id == tokenizer.unk_token_id:
        raise RuntimeError("Frozen slot token resolves to unknown")
    return slot_token_id


def _set_seed(seed: int, *, torch_module: Any, numpy_module: Any) -> None:
    if type(seed) is not int or seed != FIXED_BASE_SEED:
        raise ValueError(f"Fixed base seed must be exact integer {FIXED_BASE_SEED}")
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=False)


def _new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_artifact_tree(root: Path) -> None:
    entries = tuple(root.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise ValueError(f"Artifact tree contains a symlink: {root}")
    for entry in entries:
        if entry.is_file():
            with entry.open("rb") as source:
                os.fsync(source.fileno())
    directories = sorted(
        (entry for entry in entries if entry.is_dir()),
        key=lambda entry: len(entry.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _rename_directory_to_absent(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Fixed-base publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"Refusing to replace fixed-base artifact: {target}",
            str(target),
        )
    raise OSError(
        error_number,
        f"Atomic fixed-base publication failed: {source} -> {target}",
    )


def _file_record(root: Path, path: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _tree_records(root: Path, *, exclude: Sequence[str] = ()) -> list[dict[str, str | int]]:
    excluded = set(exclude)
    records = [
        _file_record(root, path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(f"Artifact tree contains a symlink: {root}")
    return records


def _tensor_sha256(tensor: Any, *, torch_module: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    if value.dtype == torch_module.bfloat16:
        value = value.view(torch_module.uint16)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _publish_fixed_base_staging(
    *,
    staging_dir: Path,
    output_dir: Path,
    manifest_payload: bytes,
    expected_manifest_sha256: str,
    expected_baseline_config_sha256: str,
    expected_model_sha256: str,
    expected_state_key_sha256: str,
    expected_new_rows_sha256: str,
    precommit_validator: Callable[[], None],
) -> None:
    if not callable(precommit_validator):
        raise TypeError("precommit_validator must be callable")
    manifest_sha256 = _sha256_bytes(manifest_payload)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "Fixed-base manifest left the frozen Step-8 contract: "
            f"actual={manifest_sha256}, "
            f"expected={expected_manifest_sha256}"
        )
    _fsync_artifact_tree(staging_dir)
    published = False
    try:
        precommit_validator()
        _new_file(staging_dir / "artifact_manifest.json", manifest_payload)
        _fsync_directory(staging_dir)
        _rename_directory_to_absent(staging_dir, output_dir)
        published = True
        _fsync_directory(output_dir.parent)
        validated = validate_fixed_base_artifact(
            output_dir,
            expectation=FixedBaseArtifactExpectation(
                artifact_manifest_sha256=expected_manifest_sha256,
                baseline_config_sha256=expected_baseline_config_sha256,
            ),
        )
        if (
            validated.model_sha256 != expected_model_sha256
            or validated.state_key_sha256 != expected_state_key_sha256
            or validated.new_embedding_rows_sha256 != expected_new_rows_sha256
        ):
            raise RuntimeError("Fixed-base post-publication identity changed")
    except BaseException:
        current_root = output_dir if published else staging_dir
        marker = current_root / "artifact_manifest.json"
        if marker.exists() or marker.is_symlink():
            marker.unlink()
            _fsync_directory(current_root)
        if published and output_dir.exists() and not staging_dir.exists():
            _rename_directory_to_absent(output_dir, staging_dir)
            _fsync_directory(output_dir.parent)
        raise


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_fixed_base_output_paths(
    *,
    output_dir: Path,
    protected_inputs: Sequence[Path],
) -> Path:
    if not output_dir.is_absolute() or output_dir.resolve(strict=False) != output_dir:
        raise ValueError("Fixed-base output must be an absolute canonical path")
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("Fixed-base output parent must be a real existing directory")
    staging_dir = output_dir.with_name(output_dir.name + ".incomplete")
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or staging_dir.exists()
        or staging_dir.is_symlink()
    ):
        raise FileExistsError(
            "Fixed-base output and sibling incomplete path must both be absent"
        )
    for input_path in protected_inputs:
        resolved_input = input_path.resolve(strict=True)
        if _paths_overlap(output_dir, resolved_input) or _paths_overlap(
            staging_dir,
            resolved_input,
        ):
            raise ValueError(
                "Fixed-base output overlaps an immutable input: "
                f"output={output_dir}, input={resolved_input}"
            )
    return staging_dir


def build_fixed_base_artifact(
    *,
    snapshot_dir: Path,
    snapshot_manifest_path: Path,
    baseline_config_path: Path,
    artifact_contract_path: Path,
    output_dir: Path,
    runtime: ControlledArtifactRuntime,
    numpy_module: Any,
) -> dict[str, Any]:
    """Build the one fold-independent fixed-seed untrained ModernBERT artifact."""

    if not isinstance(runtime, ControlledArtifactRuntime):
        raise TypeError("runtime must be ControlledArtifactRuntime")
    if os.environ.get("PYTHONHASHSEED") != str(FIXED_BASE_SEED):
        raise RuntimeError("Fixed-base builder requires PYTHONHASHSEED=17 before Python starts")
    for name, expected in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"Fixed-base builder environment {name} changed")
    snapshot_dir = Path(snapshot_dir)
    snapshot_manifest_path = Path(snapshot_manifest_path)
    baseline_config_path = Path(baseline_config_path)
    artifact_contract_path = Path(artifact_contract_path)
    baseline_config, baseline_config_sha256 = _load_canonical_json_with_sha256(
        baseline_config_path,
        name="evaluation baseline config",
    )
    if baseline_config.get("modernbert_base", {}).get("fixed_initialization_seed") != 17:
        raise ValueError("Evaluation baseline config fixed seed changed")
    if baseline_config_sha256 != FIXED_BASELINE_CONFIG_SHA256:
        raise ValueError(
            "Evaluation baseline config left the frozen Step-8 contract: "
            f"actual={baseline_config_sha256}, expected={FIXED_BASELINE_CONFIG_SHA256}"
        )
    build_contract = _load_fixed_base_build_contract(artifact_contract_path)
    snapshot_identity = validate_snapshot(
        snapshot_dir=snapshot_dir,
        manifest_path=snapshot_manifest_path,
        expected_manifest_sha256=MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
        expected_model_id=MODERNBERT_MODEL_ID,
        expected_revision=MODERNBERT_REVISION,
        expected_tree_sha256=MODERNBERT_SNAPSHOT_TREE_SHA256,
    )
    output_dir = Path(output_dir)
    staging_dir = _validate_fixed_base_output_paths(
        output_dir=output_dir,
        protected_inputs=(
            snapshot_dir,
            snapshot_manifest_path,
            baseline_config_path,
            artifact_contract_path,
            Path(__file__).resolve().parents[1],
        ),
    )
    staging_dir.mkdir()

    torch_module = runtime.torch_module
    _set_seed(FIXED_BASE_SEED, torch_module=torch_module, numpy_module=numpy_module)
    tokenizer = runtime.auto_tokenizer_class.from_pretrained(
        str(snapshot_dir),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    slot_token_id = _add_markup_tokens(tokenizer)
    config = runtime.auto_config_class.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(config, "model_type", None) != "modernbert":
        raise RuntimeError("Frozen base snapshot is not ModernBERT")
    config.deterministic_flash_attn = True
    config.reference_compile = False
    encoder = runtime.auto_model_class.from_pretrained(
        str(snapshot_dir),
        config=config,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
    )
    encoder.resize_token_embeddings(len(tokenizer))
    model = runtime.retriever_class(
        encoder=encoder,
        slot_token_id=slot_token_id,
        temperature=FIXED_BASE_TEMPERATURE,
    )
    model.to(dtype=torch_module.bfloat16, device="cpu")
    model.encoder.config._name_or_path = MODERNBERT_LOGICAL_IDENTITY
    state = _validate_floating_state(
        model,
        torch_module=torch_module,
        dtype=torch_module.bfloat16,
        expected_count=FIXED_BASE_MODEL_STATE_COUNT,
        context="Fixed-base build",
    )
    state_key_sha256 = _sha256_bytes(
        _canonical_json(sorted(state)).encode("utf-8")
    )
    embeddings = model.encoder.get_input_embeddings().weight
    if embeddings.shape[0] != FIXED_BASE_TOKENIZER_SIZE:
        raise RuntimeError("Fixed-base resized embedding row count changed")
    new_embedding_rows_sha256 = _tensor_sha256(
        embeddings[50_368:],
        torch_module=torch_module,
    )

    model_path = staging_dir / "model.safetensors"
    from safetensors.torch import save_file

    save_file(
        {name: state[name].detach().cpu().contiguous() for name in sorted(state)},
        str(model_path),
        metadata={"format": "pt"},
    )
    tokenizer_dir = staging_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    encoder_config_dir = staging_dir / "encoder_config"
    model.encoder.config.save_pretrained(str(encoder_config_dir))
    wrapper = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "slot_token": SLOT_TOKEN,
        "slot_token_id": slot_token_id,
        "temperature": FIXED_BASE_TEMPERATURE,
        "tokenizer_size": FIXED_BASE_TOKENIZER_SIZE,
        "query_view": FIXED_BASE_QUERY_VIEW,
        "query_pooling": "single_mask_slot_then_l2_normalize_v1",
        "passage_pooling": (
            "attention_masked_mean_excluding_first_token_then_l2_normalize_v1"
        ),
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
    }
    wrapper_path = staging_dir / "wrapper_config.json"
    _new_file(wrapper_path, _canonical_pretty_bytes(wrapper))
    runtime_versions = {
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "numpy": str(numpy_module.__version__),
        **{
            package: importlib.metadata.version(package)
            for package in (
                "flash-attn",
                "huggingface-hub",
                "safetensors",
                "tokenizers",
                "transformers",
            )
        },
    }
    run = {
        "schema_version": 1,
        "artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
        "fixed_initialization_seed": FIXED_BASE_SEED,
        "query_view": FIXED_BASE_QUERY_VIEW,
        "source_snapshot": {
            "model_id": snapshot_identity.model_id,
            "revision": snapshot_identity.revision,
            "manifest_sha256": snapshot_identity.manifest_sha256,
            "tree_sha256": snapshot_identity.tree_sha256,
        },
        "baseline_config_sha256": baseline_config_sha256,
        "runtime_versions": runtime_versions,
        "markup_tokens": all_markup_tokens(),
        "markup_token_ids": [
            int(tokenizer.convert_tokens_to_ids(token)) for token in all_markup_tokens()
        ],
        "add_special_tokens_return": 19,
        "base_tokenizer_size": 50_368,
        "slot_token_id": slot_token_id,
        "tokenizer_size": FIXED_BASE_TOKENIZER_SIZE,
        "weight_dtype": "bfloat16",
        "state_key_sha256": state_key_sha256,
        "new_embedding_rows_sha256": new_embedding_rows_sha256,
        "model_sha256": _sha256_file(model_path),
    }
    run_path = staging_dir / "baseline_run.json"
    _new_file(run_path, _canonical_pretty_bytes(run))
    files = _tree_records(staging_dir)
    manifest = {
        "schema_version": FIXED_BASE_ARTIFACT_SCHEMA_VERSION,
        "commit_marker": True,
        "artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
        "identity": {
            "fixed_initialization_seed": FIXED_BASE_SEED,
            "query_view": FIXED_BASE_QUERY_VIEW,
            "baseline_config_sha256": baseline_config_sha256,
            "snapshot_manifest_sha256": snapshot_identity.manifest_sha256,
            "snapshot_tree_sha256": snapshot_identity.tree_sha256,
            "model_sha256": run["model_sha256"],
            "state_key_sha256": state_key_sha256,
            "new_embedding_rows_sha256": new_embedding_rows_sha256,
        },
        "files": files,
    }
    manifest_payload = _canonical_pretty_bytes(manifest)

    expected_outputs = {
        "model_sha256": run["model_sha256"],
        "state_key_sha256": state_key_sha256,
        "new_embedding_rows_sha256": new_embedding_rows_sha256,
    }
    for key, actual in expected_outputs.items():
        if actual != build_contract[key]:
            raise RuntimeError(
                f"Fixed-base output {key} left the frozen build contract: "
                f"actual={actual}, expected={build_contract[key]}"
            )

    def validate_immutable_inputs() -> None:
        current_baseline, current_baseline_sha256 = _load_canonical_json_with_sha256(
            baseline_config_path,
            name="evaluation baseline config",
        )
        if (
            current_baseline != baseline_config
            or current_baseline_sha256 != baseline_config_sha256
        ):
            raise RuntimeError("Evaluation baseline config changed during fixed-base build")
        if _load_fixed_base_build_contract(artifact_contract_path) != build_contract:
            raise RuntimeError("Fixed-base build contract changed during build")
        current_snapshot = validate_snapshot(
            snapshot_dir=snapshot_dir,
            manifest_path=snapshot_manifest_path,
            expected_manifest_sha256=MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
            expected_model_id=MODERNBERT_MODEL_ID,
            expected_revision=MODERNBERT_REVISION,
            expected_tree_sha256=MODERNBERT_SNAPSHOT_TREE_SHA256,
        )
        if current_snapshot != snapshot_identity:
            raise RuntimeError("ModernBERT snapshot identity changed during fixed-base build")

    _publish_fixed_base_staging(
        staging_dir=staging_dir,
        output_dir=output_dir,
        manifest_payload=manifest_payload,
        expected_manifest_sha256=build_contract["artifact_manifest_sha256"],
        expected_baseline_config_sha256=build_contract["baseline_config_sha256"],
        expected_model_sha256=run["model_sha256"],
        expected_state_key_sha256=state_key_sha256,
        expected_new_rows_sha256=new_embedding_rows_sha256,
        precommit_validator=validate_immutable_inputs,
    )
    return manifest


def validate_fixed_base_artifact(
    artifact_dir: Path,
    *,
    expectation: FixedBaseArtifactExpectation,
) -> ValidatedFixedBaseArtifact:
    if not isinstance(expectation, FixedBaseArtifactExpectation):
        raise TypeError("expectation must be FixedBaseArtifactExpectation")
    artifact_dir = Path(artifact_dir)
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError(f"Fixed-base artifact must be a real directory: {artifact_dir}")
    manifest_path = artifact_dir / "artifact_manifest.json"
    if _sha256_file(manifest_path) != expectation.artifact_manifest_sha256:
        raise ValueError("Fixed-base artifact manifest hash changed")
    manifest = _load_canonical_json(manifest_path, name="fixed-base artifact manifest")
    if set(manifest) != {
        "schema_version",
        "commit_marker",
        "artifact_protocol",
        "identity",
        "files",
    }:
        raise ValueError("Fixed-base artifact manifest schema changed")
    if (
        manifest["schema_version"] != FIXED_BASE_ARTIFACT_SCHEMA_VERSION
        or type(manifest["schema_version"]) is not int
        or manifest["commit_marker"] is not True
        or type(manifest["commit_marker"]) is not bool
        or manifest["artifact_protocol"] != FIXED_BASE_ARTIFACT_PROTOCOL
    ):
        raise ValueError("Fixed-base artifact commit protocol changed")
    identity = manifest["identity"]
    identity_keys = {
        "fixed_initialization_seed",
        "query_view",
        "baseline_config_sha256",
        "snapshot_manifest_sha256",
        "snapshot_tree_sha256",
        "model_sha256",
        "state_key_sha256",
        "new_embedding_rows_sha256",
    }
    if type(identity) is not dict or set(identity) != identity_keys:
        raise ValueError("Fixed-base artifact identity schema changed")
    expected_identity = {
        "fixed_initialization_seed": FIXED_BASE_SEED,
        "query_view": FIXED_BASE_QUERY_VIEW,
        "baseline_config_sha256": expectation.baseline_config_sha256,
        "snapshot_manifest_sha256": MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
        "snapshot_tree_sha256": MODERNBERT_SNAPSHOT_TREE_SHA256,
    }
    for name, expected in expected_identity.items():
        if identity[name] != expected or type(identity[name]) is not type(expected):
            raise ValueError(f"Fixed-base artifact identity {name} changed")
    for name in ("model_sha256", "state_key_sha256", "new_embedding_rows_sha256"):
        _require_sha256(identity[name], name=f"fixed_base.identity.{name}")

    raw_files = manifest["files"]
    if type(raw_files) is not list or not raw_files:
        raise ValueError("Fixed-base artifact files must be a non-empty list")
    expected_paths: list[str] = []
    for position, record in enumerate(raw_files):
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError(f"Fixed-base file record {position} has an invalid schema")
        path = record["path"]
        if (
            type(path) is not str
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or type(record["size"]) is not int
            or record["size"] < 1
        ):
            raise ValueError(f"Fixed-base file record {position} is invalid")
        _require_sha256(record["sha256"], name=f"fixed_base.files[{position}].sha256")
        expected_paths.append(path)
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        raise ValueError("Fixed-base file records must be unique and sorted")
    actual_paths = sorted(
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file()
    )
    if actual_paths != sorted([*expected_paths, "artifact_manifest.json"]):
        raise ValueError("Fixed-base artifact complete file inventory changed")
    if any(path.is_symlink() for path in artifact_dir.rglob("*")):
        raise ValueError("Fixed-base artifact contains a symlink")
    for record in raw_files:
        path = artifact_dir / record["path"]
        if path.stat().st_size != record["size"] or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Fixed-base artifact file bytes changed: {record['path']}")

    required_paths = {
        "baseline_run.json",
        "encoder_config/config.json",
        "model.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "wrapper_config.json",
    }
    if set(expected_paths) != required_paths:
        raise ValueError("Fixed-base artifact required file inventory changed")
    run_path = artifact_dir / "baseline_run.json"
    run = _load_canonical_json(run_path, name="fixed-base run")
    if set(run) != {
        "schema_version",
        "artifact_protocol",
        "fixed_initialization_seed",
        "query_view",
        "source_snapshot",
        "baseline_config_sha256",
        "runtime_versions",
        "markup_tokens",
        "markup_token_ids",
        "add_special_tokens_return",
        "base_tokenizer_size",
        "slot_token_id",
        "tokenizer_size",
        "weight_dtype",
        "state_key_sha256",
        "new_embedding_rows_sha256",
        "model_sha256",
    }:
        raise ValueError("Fixed-base run schema changed")
    if (
        run["schema_version"] != 1
        or type(run["schema_version"]) is not int
        or run["artifact_protocol"] != FIXED_BASE_ARTIFACT_PROTOCOL
        or run["fixed_initialization_seed"] != FIXED_BASE_SEED
        or type(run["fixed_initialization_seed"]) is not int
        or run["query_view"] != FIXED_BASE_QUERY_VIEW
        or run["baseline_config_sha256"] != expectation.baseline_config_sha256
        or run["markup_tokens"] != all_markup_tokens()
        or run["markup_token_ids"] != list(FIXED_BASE_MARKUP_TOKEN_IDS)
        or run["add_special_tokens_return"] != 19
        or type(run["add_special_tokens_return"]) is not int
        or run["base_tokenizer_size"] != 50_368
        or type(run["base_tokenizer_size"]) is not int
        or type(run["slot_token_id"]) is not int
        or run["tokenizer_size"] != FIXED_BASE_TOKENIZER_SIZE
        or type(run["tokenizer_size"]) is not int
        or run["weight_dtype"] != "bfloat16"
        or run["state_key_sha256"] != identity["state_key_sha256"]
        or run["new_embedding_rows_sha256"] != identity["new_embedding_rows_sha256"]
        or run["model_sha256"] != identity["model_sha256"]
    ):
        raise ValueError("Fixed-base run identity changed")
    if run["source_snapshot"] != {
        "model_id": MODERNBERT_MODEL_ID,
        "revision": MODERNBERT_REVISION,
        "manifest_sha256": MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
        "tree_sha256": MODERNBERT_SNAPSHOT_TREE_SHA256,
    }:
        raise ValueError("Fixed-base run snapshot identity changed")
    if run["runtime_versions"] != FIXED_BASE_RUNTIME_VERSIONS:
        raise ValueError("Fixed-base runtime inventory changed")

    wrapper_path = artifact_dir / "wrapper_config.json"
    wrapper = _load_canonical_json(wrapper_path, name="fixed-base wrapper")
    expected_wrapper = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "slot_token": SLOT_TOKEN,
        "slot_token_id": run["slot_token_id"],
        "temperature": FIXED_BASE_TEMPERATURE,
        "tokenizer_size": FIXED_BASE_TOKENIZER_SIZE,
        "query_view": FIXED_BASE_QUERY_VIEW,
        "query_pooling": "single_mask_slot_then_l2_normalize_v1",
        "passage_pooling": (
            "attention_masked_mean_excluding_first_token_then_l2_normalize_v1"
        ),
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
    }
    if wrapper != expected_wrapper:
        raise ValueError("Fixed-base wrapper contract changed")
    model_path = artifact_dir / "model.safetensors"
    if _sha256_file(model_path) != identity["model_sha256"]:
        raise ValueError("Fixed-base model hash changed")
    return ValidatedFixedBaseArtifact(
        root=artifact_dir,
        expectation=expectation,
        manifest_sha256=expectation.artifact_manifest_sha256,
        model_sha256=identity["model_sha256"],
        model_path=model_path,
        tokenizer_dir=artifact_dir / "tokenizer",
        encoder_config_dir=artifact_dir / "encoder_config",
        wrapper_config_path=wrapper_path,
        run_path=run_path,
        slot_token_id=run["slot_token_id"],
        state_key_sha256=identity["state_key_sha256"],
        new_embedding_rows_sha256=identity["new_embedding_rows_sha256"],
    )


def load_fixed_base_retriever(
    artifact: ValidatedFixedBaseArtifact,
    *,
    device: str,
    runtime: ControlledArtifactRuntime,
) -> LoadedFixedBaseRetriever:
    if not isinstance(artifact, ValidatedFixedBaseArtifact):
        raise TypeError("artifact must be ValidatedFixedBaseArtifact")
    if not isinstance(runtime, ControlledArtifactRuntime):
        raise TypeError("runtime must be ControlledArtifactRuntime")
    before = validate_fixed_base_artifact(
        artifact.root,
        expectation=artifact.expectation,
    )
    if before.manifest_sha256 != artifact.manifest_sha256:
        raise RuntimeError("Fixed-base artifact changed before loading")
    torch_module = runtime.torch_module
    resolved_device = _explicit_device(device, torch_module)
    tokenizer = runtime.auto_tokenizer_class.from_pretrained(
        str(artifact.tokenizer_dir),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if len(tokenizer) != FIXED_BASE_TOKENIZER_SIZE:
        raise RuntimeError("Fixed-base tokenizer size changed")
    slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
    if slot_token_id != artifact.slot_token_id or slot_token_id == tokenizer.unk_token_id:
        raise RuntimeError("Fixed-base slot token identity changed")
    config = runtime.auto_config_class.from_pretrained(
        str(artifact.encoder_config_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    for name, expected in {
        "model_type": "modernbert",
        "vocab_size": FIXED_BASE_TOKENIZER_SIZE,
        "deterministic_flash_attn": True,
        "reference_compile": False,
    }.items():
        if getattr(config, name, None) != expected:
            raise RuntimeError(f"Fixed-base encoder config {name} changed")
    encoder = runtime.auto_model_class.from_config(
        config,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_module.bfloat16,
    )
    if getattr(encoder.config, "_attn_implementation", None) != "flash_attention_2":
        raise RuntimeError("Fixed-base encoder did not resolve flash_attention_2")
    model = runtime.retriever_class(
        encoder=encoder,
        slot_token_id=slot_token_id,
        temperature=FIXED_BASE_TEMPERATURE,
    )
    fresh_state = _validate_floating_state(
        model,
        torch_module=torch_module,
        dtype=torch_module.bfloat16,
        expected_count=FIXED_BASE_MODEL_STATE_COUNT,
        context="Fresh fixed-base",
    )
    if _sha256_bytes(_canonical_json(sorted(fresh_state)).encode("utf-8")) != artifact.state_key_sha256:
        raise RuntimeError("Fixed-base state-key inventory changed")
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
        raise RuntimeError(f"Strict fixed-base load was incomplete: {incompatibilities}")
    loaded_state = _validate_floating_state(
        model,
        torch_module=torch_module,
        dtype=torch_module.bfloat16,
        expected_count=FIXED_BASE_MODEL_STATE_COUNT,
        context="Loaded fixed-base",
    )
    embedding_hash = _tensor_sha256(
        model.encoder.get_input_embeddings().weight[50_368:],
        torch_module=torch_module,
    )
    if embedding_hash != artifact.new_embedding_rows_sha256:
        raise RuntimeError("Fixed-base resized embedding rows changed")
    if _sha256_bytes(_canonical_json(sorted(loaded_state)).encode("utf-8")) != artifact.state_key_sha256:
        raise RuntimeError("Loaded fixed-base state-key inventory changed")
    model.to(resolved_device)
    model.eval()
    after = validate_fixed_base_artifact(
        artifact.root,
        expectation=artifact.expectation,
    )
    if after.manifest_sha256 != artifact.manifest_sha256:
        raise RuntimeError("Fixed-base artifact changed during loading")
    return LoadedFixedBaseRetriever(
        model=model,
        tokenizer=tokenizer,
        artifact=artifact,
        device=device,
    )
