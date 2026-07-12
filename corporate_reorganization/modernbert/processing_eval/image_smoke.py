from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Sequence


PROGRAM_ROOT = Path(__file__).resolve().parents[1]
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from processing_eval.build_context import (  # noqa: E402
    BASE_IMAGE_URI,
    BUILD_EXPORTER,
    DIRECTORY_MODE,
    DOCKERFILE_FRONTEND,
    FILE_MODE,
    FREEZE_PROTOCOL,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_TYPE,
    PLATFORM,
    load_build_context_manifest,
)


_CONTRACT_KEYS = {
    "base_image",
    "build_exporter",
    "build_manifest",
    "dockerfile_frontend",
    "entrypoint",
    "environment",
    "java",
    "neural_runtime",
    "platform",
    "program_root",
    "schema_version",
    "source_inventory",
    "sparse_runtime",
    "workdir",
}


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


def _load_contract(path: Path) -> tuple[dict[str, object], str, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Image contract must be a regular non-symlink file: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o644:
        raise ValueError("Image contract mode must be 0644")
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError("Image contract must be compact canonical JSON")
    return value, _sha256_bytes(raw), raw


def _validate_contract(contract: dict[str, object]) -> None:
    if set(contract) != _CONTRACT_KEYS:
        raise ValueError("Processing image contract schema changed")
    if contract["schema_version"] != 1 or type(contract["schema_version"]) is not int:
        raise ValueError("Processing image schema_version changed")
    if contract["platform"] != PLATFORM:
        raise ValueError("Processing image platform changed")
    if contract["base_image"] != {
        "digest": BASE_IMAGE_URI.rsplit("@", 1)[1],
        "uri": BASE_IMAGE_URI,
    }:
        raise ValueError("Processing image base identity changed")
    if contract["dockerfile_frontend"] != DOCKERFILE_FRONTEND:
        raise ValueError("Processing image Dockerfile frontend changed")
    if contract["build_exporter"] != BUILD_EXPORTER:
        raise ValueError("Processing image exporter contract changed")
    if contract["build_manifest"] != {
        "directory_mode": DIRECTORY_MODE,
        "file_mode": FILE_MODE,
        "manifest_type": MANIFEST_TYPE,
        "path": MANIFEST_RELATIVE_PATH,
        "protocol": FREEZE_PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError("Processing image build-manifest contract changed")


def _expected_directories(file_paths: Sequence[str]) -> set[str]:
    directories = {"."}
    for relative in file_paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _validate_runtime_sources(
    contract: dict[str, object],
    manifest: dict[str, object],
    program_root: Path,
    *,
    contract_bytes: bytes | None = None,
) -> None:
    program_root = Path(program_root)
    if program_root.is_symlink() or not program_root.is_dir():
        raise RuntimeError(f"Processing program root must be a real directory: {program_root}")
    source_inventory = contract["source_inventory"]
    if type(source_inventory) is not list or source_inventory != sorted(set(source_inventory)):
        raise ValueError("Processing image source inventory is not unique and sorted")
    if any(
        type(item) is not str
        or not item
        or "\\" in item
        or PurePosixPath(item).is_absolute()
        or PurePosixPath(item).as_posix() != item
        or any(part in {"", ".", ".."} for part in PurePosixPath(item).parts)
        for item in source_inventory
    ):
        raise ValueError("Processing image source inventory contains an invalid path")
    manifest_records = {record["path"]: record for record in manifest["files"]}
    missing_records = sorted(set(source_inventory) - set(manifest_records))
    if missing_records:
        raise RuntimeError(
            f"Processing runtime sources are absent from the frozen manifest: {missing_records}"
        )
    manifest_relative = contract["build_manifest"]["path"]
    expected_files = set(source_inventory) | {manifest_relative}
    expected_directories = _expected_directories(sorted(expected_files))
    actual_files: set[str] = set()
    actual_directories = {"."}
    for current, directory_names, file_names in os.walk(
        program_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        relative_current = current_path.relative_to(program_root).as_posix()
        if relative_current == ".":
            relative_current = "."
        if stat.S_IMODE(current_path.stat().st_mode) != int(DIRECTORY_MODE, 8):
            raise RuntimeError(
                f"Processing runtime directory mode changed: {relative_current}"
            )
        for name in tuple(directory_names):
            child = current_path / name
            relative = child.relative_to(program_root).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise RuntimeError(
                    f"Processing runtime contains a non-directory or symlink: {relative}"
                )
            actual_directories.add(relative)
        for name in file_names:
            child = current_path / name
            relative = child.relative_to(program_root).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(
                    f"Processing runtime contains a non-regular file or symlink: {relative}"
                )
            actual_files.add(relative)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            "Processing runtime source inventory changed: "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"extra_files={sorted(actual_files - expected_files)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}, "
            f"extra_directories={sorted(actual_directories - expected_directories)}"
        )
    for relative in source_inventory:
        path = program_root / relative
        record = manifest_records[relative]
        metadata = path.stat()
        if (
            stat.S_IMODE(metadata.st_mode) != int(record["mode"], 8)
            or metadata.st_size != record["size"]
            or _sha256_bytes(path.read_bytes()) != record["sha256"]
        ):
            raise RuntimeError(f"Processing runtime source identity changed: {relative}")
    internal_contract = program_root / "processing_eval/image_contract.json"
    expected_contract_bytes = contract_bytes if contract_bytes is not None else _canonical_bytes(contract)
    if internal_contract.read_bytes() != expected_contract_bytes:
        raise RuntimeError("Processing runtime image contract differs from the validated contract")


def _validate_expected_build_inputs(
    manifest: dict[str, object],
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
        raise ValueError(
            "Expected build identity, source parent commit, epoch, and RFC3339 time "
            "are all required"
        )
    if (
        expected_build_identity_sha256 != manifest["build_identity_sha256"]
        or expected_source_parent_commit != manifest["source_parent_commit"]
        or expected_source_parent_epoch != str(manifest["source_parent_epoch"])
        or expected_source_parent_rfc3339 != manifest["source_parent_rfc3339"]
    ):
        raise RuntimeError("Docker build arguments do not match the frozen build manifest")


def _installed_distribution_identity(distribution_name: str) -> dict[str, object]:
    distribution = importlib.metadata.distribution(distribution_name)
    if distribution.files is None:
        raise RuntimeError(f"Installed distribution has no file inventory: {distribution_name}")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in distribution.files:
        relative = str(item)
        posix = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or posix.is_absolute()
            or posix.as_posix() != relative
            or any(part in {"", ".", ".."} for part in posix.parts)
            or relative in seen
        ):
            raise RuntimeError(
                f"Installed distribution contains an unsafe/duplicate path: "
                f"{distribution_name}:{relative!r}"
            )
        seen.add(relative)
        path = Path(distribution.locate_file(item))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Installed distribution entry is absent or a symlink: "
                f"{distribution_name}:{relative}"
            )
        payload = path.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
        )
    records.sort(key=lambda record: record["path"])
    canonical_records = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "file_count": len(records),
        "total_size": sum(record["size"] for record in records),
        "tree_sha256": _sha256_bytes(canonical_records),
    }


def validate_image_runtime(
    contract_path: Path,
    *,
    build_manifest_path: Path | None = None,
    expected_build_identity_sha256: str | None = None,
    expected_source_parent_commit: str | None = None,
    expected_source_parent_epoch: str | None = None,
    expected_source_parent_rfc3339: str | None = None,
) -> dict[str, object]:
    contract, contract_sha256, contract_bytes = _load_contract(Path(contract_path))
    _validate_contract(contract)
    if platform.machine() not in {"x86_64", "AMD64"} or sys.platform != "linux":
        raise RuntimeError("Processing image is not running as linux/amd64")
    program_root = Path(contract["program_root"])
    if (
        program_root != Path("/opt/program/modernbert")
        or contract["workdir"] != str(program_root)
        or Path.cwd() != program_root
    ):
        raise RuntimeError(
            f"Processing image workdir changed: actual={Path.cwd()}, expected={program_root}"
        )
    expected_environment = contract["environment"]
    if type(expected_environment) is not dict or any(
        type(name) is not str
        or type(value) is not str
        or os.environ.get(name) != value
        for name, value in expected_environment.items()
    ):
        raise RuntimeError("Processing image deterministic/offline environment changed")

    internal_manifest_path = program_root / contract["build_manifest"]["path"]
    if build_manifest_path is not None:
        supplied_manifest = Path(build_manifest_path)
        if supplied_manifest.is_symlink() or not supplied_manifest.is_file():
            raise ValueError("Supplied build manifest is absent or a symlink")
        if supplied_manifest.read_bytes() != internal_manifest_path.read_bytes():
            raise RuntimeError("Supplied build manifest differs from the in-image manifest")
    manifest = load_build_context_manifest(internal_manifest_path)
    if (
        manifest["base_image"] != contract["base_image"]["uri"]
        or manifest["dockerfile_frontend"] != contract["dockerfile_frontend"]
        or manifest["exporter"] != contract["build_exporter"]
        or manifest["platform"] != contract["platform"]
    ):
        raise RuntimeError("Build manifest differs from the image contract")
    if os.environ.get("SOURCE_DATE_EPOCH") != str(manifest["source_parent_epoch"]):
        raise RuntimeError("Processing image SOURCE_DATE_EPOCH differs from its source parent")
    _validate_expected_build_inputs(
        manifest,
        expected_build_identity_sha256=expected_build_identity_sha256,
        expected_source_parent_commit=expected_source_parent_commit,
        expected_source_parent_epoch=expected_source_parent_epoch,
        expected_source_parent_rfc3339=expected_source_parent_rfc3339,
    )
    _validate_runtime_sources(
        contract,
        manifest,
        program_root,
        contract_bytes=contract_bytes,
    )

    neural = contract["neural_runtime"]
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
    actual_versions = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in package_names},
    }
    expected_versions = {name: neural[name] for name in ("python", *package_names)}
    if actual_versions != expected_versions:
        raise RuntimeError(
            "Processing neural dependency inventory changed: "
            f"actual={actual_versions}, expected={expected_versions}"
        )
    import torch

    if (
        str(torch.__version__) != neural["torch_runtime"]
        or str(torch.version.cuda) != neural["cuda"]
    ):
        raise RuntimeError("Processing Torch/CUDA runtime identity changed")

    from retriever.bm25 import validate_bm25_runtime

    sparse_identity = validate_bm25_runtime().to_payload()
    sparse = contract["sparse_runtime"]
    if sparse_identity != {
        "protocol": sparse["protocol"],
        "java_home": contract["java"]["home"],
        "java_version": sparse_identity["java_version"],
        "pyserini": sparse["pyserini"],
        "pyjnius": sparse["pyjnius"],
        "anserini_jar_size": sparse["anserini_jar_size"],
        "anserini_jar_sha256": sparse["anserini_jar_sha256"],
    }:
        raise RuntimeError("Processing sparse runtime identity changed")
    if _sha256_bytes(sparse_identity["java_version"].encode("utf-8")) != contract["java"][
        "version_output_sha256"
    ]:
        raise RuntimeError("Processing Java runtime is not the exact frozen Corretto build")
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
        raise RuntimeError(
            "Processing sparse installed-file inventory changed: "
            f"actual={installed_distributions}, "
            f"expected={expected_installed_distributions}"
        )
    return {
        "build_context": {
            "build_identity_sha256": manifest["build_identity_sha256"],
            "files_sha256": manifest["files_sha256"],
            "source_parent_commit": manifest["source_parent_commit"],
            "source_parent_epoch": manifest["source_parent_epoch"],
            "source_parent_rfc3339": manifest["source_parent_rfc3339"],
            "toolchain": manifest["toolchain"],
        },
        "image_contract_sha256": contract_sha256,
        "neural_runtime": {
            **actual_versions,
            "torch_runtime": str(torch.__version__),
            "cuda": str(torch.version.cuda),
        },
        "platform": contract["platform"],
        "sparse_runtime": {
            **sparse_identity,
            "installed_distributions": installed_distributions,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the immutable ARR retrieval Processing image runtime.",
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
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
