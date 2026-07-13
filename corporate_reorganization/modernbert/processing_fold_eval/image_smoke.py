"""Validate the immutable fold-evaluation overlay runtime without network access."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROGRAM_ROOT = Path(__file__).resolve().parents[1]
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from processing_fold_eval.build_context import (  # noqa: E402
    BASE_IMAGE_URI,
    BASE_IMAGE_CONFIG_DIGEST,
    BUILD_EXPORTER,
    DIRECTORY_MODE,
    DOCKERFILE_FRONTEND,
    FILE_MODE,
    FREEZE_PROTOCOL,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_TYPE,
    PLATFORM,
    PORTABLE_RUNTIME_IDENTITY_PROTOCOL,
    load_build_context_manifest,
)


LEGACY_CONTRACT_SHA256 = (
    "c0dba1f1a2387bce425b6c33f83e5035d3904ccb62de0e4f1422602ead0cbca8"
)
LEGACY_BUILD_IDENTITY_SHA256 = (
    "249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8"
)
LEGACY_FILES_SHA256 = (
    "96f8b4e5569404ed916cd69c4d765b3eb34cbd3f40e3eff8394e9de72f415dc4"
)
EVALUATION_ENTRYPOINT = [
    "/opt/conda/bin/python",
    "/opt/program/modernbert/processing_fold_eval/evaluate_sm.py",
]
INVENTORY_ENTRYPOINT = [
    "/opt/conda/bin/python",
    "/opt/program/modernbert/processing_fold_eval/inventory_sm.py",
]
PROCESSING_LAYOUT = {
    "archive_manifest_path": (
        "/opt/ml/processing/input/fold-archives/fold_archive_input_manifest.json"
    ),
    "archive_receipt_path": (
        "/opt/ml/processing/input/fold-inventory/archive_inventory.json"
    ),
    "baseline_config_path": (
        "/opt/ml/processing/input/control/evaluation_baselines.json"
    ),
    "bm25_scratch_dir": "/opt/ml/processing/work/bm25-evaluation",
    "dataset_dir": "/opt/ml/processing/input/dataset",
    "e5_pack_artifact_dir": "/opt/ml/processing/input/e5-pack",
    "e5_snapshot_dir": "/opt/ml/processing/input/e5-snapshot",
    "e5_snapshot_manifest_path": (
        "/opt/ml/processing/input/control/e5_snapshot.json"
    ),
    "evaluation_output_dir": "/opt/ml/processing/output/evaluation",
    "evaluation_plan_path": (
        "/opt/ml/processing/input/control/evaluation_plan.json"
    ),
    "experiment_config_path": "/opt/ml/processing/input/control/experiment.json",
    "fixed_base_artifact_dir": "/opt/ml/processing/input/fixed-base",
    "fold_manifest_path": "/opt/ml/processing/input/control/folds.json",
    "image_contract_path": (
        "/opt/program/modernbert/processing_fold_eval/image_contract.json"
    ),
    "local_bindings_path": (
        "/opt/ml/processing/input/control/local_bindings.json"
    ),
    "materialization_root": "/opt/ml/processing/work/materialized",
    "output_parent": "/opt/ml/processing/output",
    "evidence_output_dir": "/opt/ml/processing/output/evidence",
    "work_parent": "/opt/ml/processing/work",
}
EXPECTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "17",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
_CONTRACT_KEYS = {
    "base_image",
    "build_exporter",
    "build_manifest",
    "dockerfile_frontend",
    "entrypoint",
    "environment",
    "inherited_runtime",
    "inventory_entrypoint",
    "platform",
    "processing_layout",
    "program_root",
    "schema_version",
    "source_inventory",
    "workdir",
}
_OVERLAY_PACKAGE = "processing_fold_eval"
_OVERRIDDEN_RETRIEVER_FILES = {
    "retriever/artifacts.py",
    "retriever/evaluator.py",
    "retriever/provenance.py",
    "retriever/staged_data.py",
}


def _container_path_uri(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be one exact container path string")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ValueError(f"{name} must be one normalized absolute container path")
    return "container://" + value


def _validate_portable_runtime_identity(value: object) -> dict[str, Any]:
    if type(value) is not dict or not value:
        raise ValueError("Portable runtime identity must be one non-empty object")
    normalized = json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )

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

    inspect(normalized, path="runtime_identity")
    if normalized.get("runtime_identity_protocol") != PORTABLE_RUNTIME_IDENTITY_PROTOCOL:
        raise ValueError("Portable runtime identity protocol changed")
    return normalized
EXPECTED_SOURCE_INVENTORY = [
    "processing_fold_eval/__init__.py",
    "processing_fold_eval/archive_bridge.py",
    "processing_fold_eval/build_context.py",
    "processing_fold_eval/evaluate_sm.py",
    "processing_fold_eval/image_contract.json",
    "processing_fold_eval/image_smoke.py",
    "processing_fold_eval/inventory_sm.py",
    "retriever/artifacts.py",
    "retriever/evaluator.py",
    "retriever/provenance.py",
    "retriever/staged_data.py",
]


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable_regular(path: Path, *, name: str) -> bytes:
    path = Path(path)
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is absent: {path}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError(f"{name} must be one singly-linked regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        for metadata in (before, after, final):
            if (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ) != identity:
                raise RuntimeError(f"{name} changed while read: {path}")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise RuntimeError(f"{name} size changed while read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _load_contract(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = _read_stable_regular(Path(path), name="Overlay image contract")
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError("Overlay image contract must be compact canonical JSON")
    return value, _sha256_bytes(raw), raw


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if set(contract) != _CONTRACT_KEYS or contract["schema_version"] != 1:
        raise ValueError("Overlay image contract schema changed")
    if contract["base_image"] != {
        "config_digest": BASE_IMAGE_CONFIG_DIGEST,
        "digest": BASE_IMAGE_URI.rsplit("@", 1)[1],
        "uri": BASE_IMAGE_URI,
    }:
        raise ValueError("Overlay base image identity changed")
    if (
        contract["platform"] != PLATFORM
        or contract["dockerfile_frontend"] != DOCKERFILE_FRONTEND
        or contract["build_exporter"] != BUILD_EXPORTER
    ):
        raise ValueError("Overlay deterministic build contract changed")
    if contract["build_manifest"] != {
        "directory_mode": DIRECTORY_MODE,
        "file_mode": FILE_MODE,
        "manifest_type": MANIFEST_TYPE,
        "path": MANIFEST_RELATIVE_PATH,
        "protocol": FREEZE_PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError("Overlay build-manifest contract changed")
    inherited = contract["inherited_runtime"]
    if inherited != {
        "build_identity_sha256": LEGACY_BUILD_IDENTITY_SHA256,
        "build_manifest_path": "processing_eval/build_context_manifest.json",
        "files_sha256": LEGACY_FILES_SHA256,
        "image_contract_path": "processing_eval/image_contract.json",
        "image_contract_sha256": LEGACY_CONTRACT_SHA256,
    }:
        raise ValueError("Inherited evaluation runtime identity changed")
    if (
        contract["entrypoint"] != EVALUATION_ENTRYPOINT
        or contract["inventory_entrypoint"] != INVENTORY_ENTRYPOINT
        or contract["environment"] != EXPECTED_ENVIRONMENT
        or contract["processing_layout"] != PROCESSING_LAYOUT
        or contract["program_root"] != "/opt/program/modernbert"
        or contract["workdir"] != "/opt/program/modernbert"
    ):
        raise ValueError("Overlay entrypoint or Processing layout changed")


def _validate_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("Overlay source inventory contains an invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError("Overlay source inventory contains an invalid path")
    return value


def _validate_runtime_sources(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    program_root: Path,
    *,
    contract_bytes: bytes,
    manifest_bytes: bytes,
    inherited_contract: Mapping[str, Any],
    inherited_manifest: Mapping[str, Any],
    inherited_manifest_bytes: bytes,
) -> None:
    program_root = Path(program_root)
    if program_root.is_symlink() or not program_root.is_dir():
        raise RuntimeError("Overlay program root must be one real directory")
    inventory = contract["source_inventory"]
    if type(inventory) is not list or inventory != EXPECTED_SOURCE_INVENTORY:
        raise ValueError("Overlay source inventory changed")
    paths = [_validate_relative_path(value) for value in inventory]
    overlay_records = {record["path"]: record for record in manifest["files"]}
    if not set(paths).issubset(overlay_records):
        raise RuntimeError("Overlay runtime source is absent from the frozen manifest")
    inherited_inventory = inherited_contract.get("source_inventory")
    if (
        type(inherited_inventory) is not list
        or inherited_inventory != sorted(set(inherited_inventory))
        or any(_validate_relative_path(value) != value for value in inherited_inventory)
        or not _OVERRIDDEN_RETRIEVER_FILES.issubset(inherited_inventory)
    ):
        raise RuntimeError("Inherited runtime source inventory changed")
    inherited_records = {
        record["path"]: record for record in inherited_manifest["files"]
    }
    if not set(inherited_inventory).issubset(inherited_records):
        raise RuntimeError("Inherited source is absent from its frozen manifest")
    unchanged_inherited = set(inherited_inventory) - _OVERRIDDEN_RETRIEVER_FILES
    expected_files = (
        unchanged_inherited
        | set(paths)
        | {
            MANIFEST_RELATIVE_PATH,
            inherited_contract["build_manifest"]["path"],
        }
    )
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_directories = {"."}
    for current, directories, files in os.walk(
        program_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        relative_current = current_path.relative_to(program_root).as_posix()
        metadata = current_path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o755
        ):
            raise RuntimeError(f"Merged runtime directory changed: {relative_current}")
        actual_directories.add(relative_current)
        for name in tuple(directories):
            child = current_path / name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISDIR(
                child_metadata.st_mode
            ):
                raise RuntimeError("Merged runtime contains an unsafe directory")
        for name in files:
            child = current_path / name
            relative = child.relative_to(program_root).as_posix()
            child_metadata = child.lstat()
            if (
                stat.S_ISLNK(child_metadata.st_mode)
                or not stat.S_ISREG(child_metadata.st_mode)
                or child_metadata.st_nlink != 1
            ):
                raise RuntimeError(f"Merged runtime source is unsafe: {relative}")
            if stat.S_IMODE(child_metadata.st_mode) != 0o644:
                raise RuntimeError(f"Merged runtime source mode changed: {relative}")
            actual_files.add(relative)
    if (
        actual_files != expected_files or actual_directories != expected_directories
    ):
        raise RuntimeError(
            "Merged runtime inventory changed: "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"extra_files={sorted(actual_files - expected_files)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}, "
            f"extra_directories={sorted(actual_directories - expected_directories)}"
        )
    for relative in paths:
        path = program_root / relative
        record = overlay_records[relative]
        payload = _read_stable_regular(path, name=f"Overlay runtime source {relative}")
        metadata = path.stat()
        if (
            stat.S_IMODE(metadata.st_mode) != int(record["mode"], 8)
            or len(payload) != record["size"]
            or _sha256_bytes(payload) != record["sha256"]
        ):
            raise RuntimeError(f"Overlay runtime source identity changed: {relative}")
    for relative in sorted(unchanged_inherited):
        path = program_root / relative
        record = inherited_records[relative]
        payload = _read_stable_regular(path, name=f"Inherited runtime source {relative}")
        metadata = path.stat()
        if (
            stat.S_IMODE(metadata.st_mode) != int(record["mode"], 8)
            or len(payload) != record["size"]
            or _sha256_bytes(payload) != record["sha256"]
        ):
            raise RuntimeError(f"Inherited runtime source identity changed: {relative}")
    internal_contract = program_root / "processing_fold_eval/image_contract.json"
    if _read_stable_regular(internal_contract, name="Internal overlay contract") != contract_bytes:
        raise RuntimeError("Internal overlay contract differs from the validated bytes")
    if _read_stable_regular(
        program_root / MANIFEST_RELATIVE_PATH,
        name="Internal overlay build manifest",
    ) != manifest_bytes:
        raise RuntimeError("Internal overlay build manifest changed")
    inherited_manifest_path = program_root / inherited_contract["build_manifest"]["path"]
    if _read_stable_regular(
        inherited_manifest_path,
        name="Internal inherited build manifest",
    ) != inherited_manifest_bytes:
        raise RuntimeError("Internal inherited build manifest changed")


def _validate_expected_build_inputs(
    manifest: Mapping[str, Any],
    *,
    expected_build_identity_sha256: str | None,
    expected_source_parent_commit: str | None,
    expected_source_parent_epoch: str | None,
    expected_source_parent_rfc3339: str | None,
) -> None:
    supplied = (
        expected_build_identity_sha256,
        expected_source_parent_commit,
        expected_source_parent_epoch,
        expected_source_parent_rfc3339,
    )
    if all(value is None for value in supplied):
        return
    if any(value is None for value in supplied):
        raise ValueError("All expected overlay build inputs are required together")
    if (
        expected_build_identity_sha256 != manifest["build_identity_sha256"]
        or expected_source_parent_commit != manifest["source_parent_commit"]
        or expected_source_parent_epoch != str(manifest["source_parent_epoch"])
        or expected_source_parent_rfc3339 != manifest["source_parent_rfc3339"]
    ):
        raise RuntimeError("Docker build arguments differ from the overlay manifest")


def _load_inherited_documents(
    program_root: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    inherited = contract["inherited_runtime"]
    inherited_contract_path = program_root / inherited["image_contract_path"]
    inherited_contract_raw = _read_stable_regular(
        inherited_contract_path, name="Inherited image contract"
    )
    if _sha256_bytes(inherited_contract_raw) != inherited["image_contract_sha256"]:
        raise RuntimeError("Inherited image contract bytes changed")
    inherited_contract = json.loads(inherited_contract_raw)
    if inherited_contract_raw != _canonical_bytes(inherited_contract):
        raise RuntimeError("Inherited image contract is not canonical")
    from processing_eval.build_context import load_build_context_manifest as load_legacy

    inherited_manifest_path = program_root / inherited["build_manifest_path"]
    inherited_manifest_raw = _read_stable_regular(
        inherited_manifest_path, name="Inherited build manifest"
    )
    inherited_manifest = load_legacy(inherited_manifest_path)
    expected_manifest_raw = (
        json.dumps(
            inherited_manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if inherited_manifest_raw != expected_manifest_raw:
        raise RuntimeError("Inherited build manifest bytes changed")
    if (
        inherited_manifest["build_identity_sha256"]
        != inherited["build_identity_sha256"]
        or inherited_manifest["files_sha256"] != inherited["files_sha256"]
    ):
        raise RuntimeError("Inherited image build-manifest identity changed")
    return inherited_contract, inherited_manifest, inherited_manifest_raw


def _validate_inherited_runtime(
    inherited_contract: Mapping[str, Any],
    inherited_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    inherited = contract["inherited_runtime"]
    neural = inherited_contract["neural_runtime"]
    package_names = (
        "accelerate",
        "flash-attn",
        "huggingface-hub",
        "numpy",
        "packaging",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    )
    versions = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in package_names},
    }
    if versions != {name: neural[name] for name in ("python", *package_names)}:
        raise RuntimeError("Inherited neural package inventory changed")
    import torch

    if (
        str(torch.__version__) != neural["torch_runtime"]
        or str(torch.version.cuda) != neural["cuda"]
    ):
        raise RuntimeError("Inherited Torch/CUDA runtime changed")
    from retriever.bm25 import validate_bm25_runtime

    sparse_identity = validate_bm25_runtime().to_payload()
    sparse = inherited_contract["sparse_runtime"]
    if sparse_identity != {
        "protocol": sparse["protocol"],
        "java_home": inherited_contract["java"]["home"],
        "java_version": sparse_identity["java_version"],
        "pyserini": sparse["pyserini"],
        "pyjnius": sparse["pyjnius"],
        "anserini_jar_size": sparse["anserini_jar_size"],
        "anserini_jar_sha256": sparse["anserini_jar_sha256"],
    }:
        raise RuntimeError("Inherited sparse runtime changed")
    if _sha256_bytes(sparse_identity["java_version"].encode("utf-8")) != inherited_contract[
        "java"
    ]["version_output_sha256"]:
        raise RuntimeError("Inherited Java runtime changed")
    from processing_eval.image_smoke import _installed_distribution_identity

    installed_distributions = {
        name: _installed_distribution_identity(name)
        for name in ("pyjnius", "pyserini")
    }
    expected_installed_distributions = {
        name: {
            "file_count": sparse[f"{name}_installed_file_count"],
            "total_size": sparse[f"{name}_installed_total_size"],
            "tree_sha256": sparse[f"{name}_installed_tree_sha256"],
        }
        for name in ("pyjnius", "pyserini")
    }
    if installed_distributions != expected_installed_distributions:
        raise RuntimeError("Inherited sparse installed-file inventory changed")
    portable_sparse_identity = {
        **sparse_identity,
        "java_home": _container_path_uri(
            sparse_identity["java_home"], name="Validated JAVA_HOME"
        ),
    }
    return {
        "build_identity_sha256": inherited_manifest["build_identity_sha256"],
        "files_sha256": inherited_manifest["files_sha256"],
        "image_contract_sha256": inherited["image_contract_sha256"],
        "neural_runtime": {
            **versions,
            "cuda": str(torch.version.cuda),
            "torch_runtime": str(torch.__version__),
        },
        "sparse_runtime": {
            **portable_sparse_identity,
            "installed_distributions": installed_distributions,
        },
    }


def _validate_module_origins(program_root: Path) -> dict[str, str]:
    if not program_root.is_absolute():
        raise ValueError("Overlay program root must be absolute")
    expected = {
        "processing_fold_eval.archive_bridge": (
            program_root / "processing_fold_eval/archive_bridge.py"
        ),
        "retriever.artifacts": program_root / "retriever/artifacts.py",
        "retriever.evaluator": program_root / "retriever/evaluator.py",
        "retriever.provenance": program_root / "retriever/provenance.py",
        "retriever.staged_data": program_root / "retriever/staged_data.py",
    }
    origins: dict[str, str] = {}
    for module_name, path in expected.items():
        module = importlib.import_module(module_name)
        origin = Path(module.__file__) if module.__file__ is not None else None
        specification = module.__spec__
        specification_origin = None if specification is None else specification.origin
        if (
            origin != path
            or specification_origin != str(path)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
        ):
            raise RuntimeError(
                "Overlay module origin changed: "
                f"module={module_name!r}, file={origin}, spec={specification_origin!r}"
            )
        relative = path.relative_to(program_root)
        if relative.is_absolute() or relative.as_posix() in {"", "."}:
            raise RuntimeError("Overlay module origin did not produce one relative identity")
        origins[module_name] = relative.as_posix()
    return origins


def validate_image_runtime(
    contract_path: Path,
    *,
    build_manifest_path: Path | None = None,
    expected_build_identity_sha256: str | None = None,
    expected_source_parent_commit: str | None = None,
    expected_source_parent_epoch: str | None = None,
    expected_source_parent_rfc3339: str | None = None,
) -> dict[str, Any]:
    contract, contract_sha256, contract_bytes = _load_contract(Path(contract_path))
    _validate_contract(contract)
    if platform.machine() not in {"x86_64", "AMD64"} or sys.platform != "linux":
        raise RuntimeError("Overlay image is not running as linux/amd64")
    program_root = Path(contract["program_root"])
    if Path.cwd() != program_root:
        raise RuntimeError("Overlay image workdir changed")
    environment = contract["environment"]
    if type(environment) is not dict or any(
        type(name) is not str
        or type(value) is not str
        or os.environ.get(name) != value
        for name, value in environment.items()
    ):
        raise RuntimeError("Overlay deterministic/offline environment changed")
    internal_manifest_path = program_root / contract["build_manifest"]["path"]
    manifest_bytes = _read_stable_regular(
        internal_manifest_path, name="Internal overlay build manifest"
    )
    if build_manifest_path is not None:
        supplied = _read_stable_regular(
            Path(build_manifest_path), name="Supplied overlay build manifest"
        )
        internal = _read_stable_regular(
            internal_manifest_path, name="Internal overlay build manifest"
        )
        if supplied != internal:
            raise RuntimeError("Supplied overlay build manifest differs from the image")
    manifest = load_build_context_manifest(internal_manifest_path)
    expected_manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if manifest_bytes != expected_manifest_bytes:
        raise RuntimeError("Overlay build manifest bytes changed")
    if (
        manifest["base_image"] != contract["base_image"]["uri"]
        or manifest["dockerfile_frontend"] != contract["dockerfile_frontend"]
        or manifest["exporter"] != contract["build_exporter"]
        or manifest["platform"] != contract["platform"]
        or os.environ.get("SOURCE_DATE_EPOCH") != str(manifest["source_parent_epoch"])
    ):
        raise RuntimeError("Overlay manifest differs from its image contract/environment")
    _validate_expected_build_inputs(
        manifest,
        expected_build_identity_sha256=expected_build_identity_sha256,
        expected_source_parent_commit=expected_source_parent_commit,
        expected_source_parent_epoch=expected_source_parent_epoch,
        expected_source_parent_rfc3339=expected_source_parent_rfc3339,
    )
    (
        inherited_contract,
        inherited_manifest,
        inherited_manifest_bytes,
    ) = _load_inherited_documents(program_root, contract)
    _validate_runtime_sources(
        contract,
        manifest,
        program_root,
        contract_bytes=contract_bytes,
        manifest_bytes=manifest_bytes,
        inherited_contract=inherited_contract,
        inherited_manifest=inherited_manifest,
        inherited_manifest_bytes=inherited_manifest_bytes,
    )
    inherited_runtime = _validate_inherited_runtime(
        inherited_contract,
        inherited_manifest,
        contract,
    )
    module_origins = _validate_module_origins(program_root)
    forbidden_loaded = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in {"boto3", "botocore"}
    )
    if forbidden_loaded:
        raise RuntimeError(f"Overlay runtime imported an AWS SDK: {forbidden_loaded}")
    identity = {
        "runtime_identity_protocol": PORTABLE_RUNTIME_IDENTITY_PROTOCOL,
        "base_image": contract["base_image"],
        "build_context": {
            "build_identity_sha256": manifest["build_identity_sha256"],
            "files_sha256": manifest["files_sha256"],
            "source_parent_commit": manifest["source_parent_commit"],
            "source_parent_epoch": manifest["source_parent_epoch"],
            "source_parent_rfc3339": manifest["source_parent_rfc3339"],
            "toolchain": manifest["toolchain"],
        },
        "image_contract_sha256": contract_sha256,
        "inherited_runtime": inherited_runtime,
        "module_origins": module_origins,
        "platform": contract["platform"],
    }
    return _validate_portable_runtime_identity(identity)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable ARR fold-evaluation overlay image.",
        allow_abbrev=False,
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--expected-build-identity-sha256")
    parser.add_argument("--expected-source-parent-commit")
    parser.add_argument("--expected-source-parent-epoch")
    parser.add_argument("--expected-source-parent-rfc3339")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = validate_image_runtime(
        args.contract,
        build_manifest_path=args.build_manifest,
        expected_build_identity_sha256=args.expected_build_identity_sha256,
        expected_source_parent_commit=args.expected_source_parent_commit,
        expected_source_parent_epoch=args.expected_source_parent_epoch,
        expected_source_parent_rfc3339=args.expected_source_parent_rfc3339,
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
