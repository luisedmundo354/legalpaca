"""Freeze and build the exact fold-evaluation overlay image context."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


BASE_IMAGE_URI = (
    "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@"
    "sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2"
)
BASE_IMAGE_CONFIG_DIGEST = (
    "sha256:76c29a7f5ca0a1a36d0f8b53fe1e49f40ab199f8ff1bc594ddbb09107c7749e8"
)
DOCKERFILE_FRONTEND = (
    "docker/dockerfile:1.7@"
    "sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e"
)
PLATFORM = "linux/amd64"
BUILD_EXPORTER = {
    "compression": "gzip",
    "compression_level": 6,
    "force_compression": False,
    "oci_mediatypes": False,
    "provenance": False,
    "push": False,
    "rewrite_timestamp": True,
    "sbom": False,
    "type": "image",
    "unpack": False,
}
MANIFEST_RELATIVE_PATH = "processing_fold_eval/build_context_manifest.json"
MANIFEST_TYPE = "arr_retrieval_fold_processing_build_context"
FREEZE_PROTOCOL = "frozen_absent_output_context_v1"
MANIFEST_SCHEMA_VERSION = 1
FILE_MODE = "0644"
DIRECTORY_MODE = "0755"
DOCKER_MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
LOCAL_IMAGE_REPOSITORY = "arr-retrieval-fold-eval"
PORTABLE_RUNTIME_IDENTITY_PROTOCOL = "arr_retrieval_fold_runtime_identity_v2"
FOLD_IMAGE_CONTRACT_SHA256 = (
    "364a57629a514c67cc3dec46605d9e2bb7af9779d140e0b0d26cd0f5161e7376"
)
INHERITED_BUILD_IDENTITY_SHA256 = (
    "249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8"
)
INHERITED_FILES_SHA256 = (
    "96f8b4e5569404ed916cd69c4d765b3eb34cbd3f40e3eff8394e9de72f415dc4"
)
INHERITED_IMAGE_CONTRACT_SHA256 = (
    "c0dba1f1a2387bce425b6c33f83e5035d3904ccb62de0e4f1422602ead0cbca8"
)
INHERITED_NEURAL_RUNTIME = {
    "accelerate": "1.4.0",
    "cuda": "12.4",
    "flash-attn": "2.7.3",
    "huggingface-hub": "0.29.1",
    "numpy": "1.26.4",
    "packaging": "24.1",
    "python": "3.11.10",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "torch": "2.5.1+cu124",
    "torch_runtime": "2.5.1+cu124",
    "transformers": "4.49.0",
}
INHERITED_JAVA_VERSION_OUTPUT_SHA256 = (
    "64dbcaf74f7772c14d5614c83acefd0aba65da9f90694b8815af908ff6bcf7f1"
)
INHERITED_JAVA_VERSION = (
    'openjdk version "21.0.11" 2026-04-21 LTS\n'
    "OpenJDK Runtime Environment Corretto-21.0.11.10.1 "
    "(build 21.0.11+10-LTS)\n"
    "OpenJDK 64-Bit Server VM Corretto-21.0.11.10.1 "
    "(build 21.0.11+10-LTS, mixed mode, sharing)"
)
INHERITED_SPARSE_RUNTIME = {
    "protocol": "pyserini_1_5_0_sparse_jni_only_v1",
    "java_home": "container:///opt/amazon-corretto-21",
    "pyserini": "1.5.0",
    "pyjnius": "1.7.0",
    "anserini_jar_size": 163855488,
    "anserini_jar_sha256": (
        "bb0761df51ef7db5be361199a40a45722cccf7f0b2271e2b25337e97dd578aea"
    ),
    "installed_distributions": {
        "pyjnius": {
            "file_count": 16,
            "total_size": 5970249,
            "tree_sha256": (
                "7f2411e7c3f6baf8eb75fc466e1f8be1720b9736bbd72d7700b21515ffab23c0"
            ),
        },
        "pyserini": {
            "file_count": 161,
            "total_size": 165653249,
            "tree_sha256": (
                "c8a6c1ae730c19a91bd091f4a282b29008f3360396b5cda2431f0f712e3e4f56"
            ),
        },
    },
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
EXPECTED_PROCESSING_LAYOUT = {
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

_CONTROL_PATHS = {
    "processing_fold_eval/Dockerfile",
    "processing_fold_eval/Dockerfile.dockerignore",
    "processing_fold_eval/build_context.py",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_TOOL_VERSION_RE = re.compile(
    r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z"
)


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


def _canonical_compact_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_dict(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _validate_image_runtime_identity(
    value: object,
    *,
    build_context_files_sha256: str,
    build_context_identity_sha256: str,
) -> dict[str, Any]:
    identity = _exact_dict(
        value,
        {
            "runtime_identity_protocol",
            "base_image",
            "build_context",
            "image_contract_sha256",
            "inherited_runtime",
            "module_origins",
            "platform",
        },
        name="image runtime identity",
    )
    normalized = json.loads(_canonical_json(identity))

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

    inspect(normalized, path="image_runtime_identity")
    if (
        normalized["runtime_identity_protocol"]
        != PORTABLE_RUNTIME_IDENTITY_PROTOCOL
        or normalized["base_image"]
        != {
            "config_digest": BASE_IMAGE_CONFIG_DIGEST,
            "digest": BASE_IMAGE_URI.rsplit("@", 1)[1],
            "uri": BASE_IMAGE_URI,
        }
        or normalized["image_contract_sha256"] != FOLD_IMAGE_CONTRACT_SHA256
        or normalized["platform"] != PLATFORM
    ):
        raise ValueError("Image runtime top-level identity changed")

    build = _exact_dict(
        normalized["build_context"],
        {
            "build_identity_sha256",
            "files_sha256",
            "source_parent_commit",
            "source_parent_epoch",
            "source_parent_rfc3339",
            "toolchain",
        },
        name="image runtime build context",
    )
    epoch = build["source_parent_epoch"]
    expected_rfc3339 = (
        datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if type(epoch) is int and epoch > 0
        else None
    )
    toolchain = _exact_dict(
        build["toolchain"],
        {"builder_driver", "buildkit_version", "buildx_version"},
        name="image runtime toolchain",
    )
    if (
        build["build_identity_sha256"] != build_context_identity_sha256
        or build["files_sha256"] != build_context_files_sha256
        or type(build["source_parent_commit"]) is not str
        or _COMMIT_RE.fullmatch(build["source_parent_commit"]) is None
        or expected_rfc3339 is None
        or build["source_parent_rfc3339"] != expected_rfc3339
        or toolchain["builder_driver"] != "docker"
        or type(toolchain["buildkit_version"]) is not str
        or _TOOL_VERSION_RE.fullmatch(toolchain["buildkit_version"]) is None
        or type(toolchain["buildx_version"]) is not str
        or _TOOL_VERSION_RE.fullmatch(toolchain["buildx_version"]) is None
    ):
        raise ValueError("Image runtime build-context identity changed")

    inherited = _exact_dict(
        normalized["inherited_runtime"],
        {
            "build_identity_sha256",
            "files_sha256",
            "image_contract_sha256",
            "neural_runtime",
            "sparse_runtime",
        },
        name="inherited runtime identity",
    )
    if (
        inherited["build_identity_sha256"] != INHERITED_BUILD_IDENTITY_SHA256
        or inherited["files_sha256"] != INHERITED_FILES_SHA256
        or inherited["image_contract_sha256"]
        != INHERITED_IMAGE_CONTRACT_SHA256
    ):
        raise ValueError("Inherited image identity changed")
    neural = _exact_dict(
        inherited["neural_runtime"],
        {
            "accelerate",
            "cuda",
            "flash-attn",
            "huggingface-hub",
            "numpy",
            "packaging",
            "python",
            "safetensors",
            "tokenizers",
            "torch",
            "torch_runtime",
            "transformers",
        },
        name="inherited neural runtime",
    )
    if neural != INHERITED_NEURAL_RUNTIME:
        raise ValueError("Inherited neural runtime version changed")
    sparse = _exact_dict(
        inherited["sparse_runtime"],
        {
            "protocol",
            "java_home",
            "java_version",
            "pyserini",
            "pyjnius",
            "anserini_jar_size",
            "anserini_jar_sha256",
            "installed_distributions",
        },
        name="inherited sparse runtime",
    )
    java_version = sparse["java_version"]
    sparse_without_java = {
        key: child for key, child in sparse.items() if key != "java_version"
    }
    if (
        type(java_version) is not str
        or java_version != INHERITED_JAVA_VERSION
        or _sha256_bytes(java_version.encode("utf-8"))
        != INHERITED_JAVA_VERSION_OUTPUT_SHA256
        or type(sparse["anserini_jar_size"]) is not int
        or sparse_without_java != INHERITED_SPARSE_RUNTIME
    ):
        raise ValueError("Inherited sparse runtime identity changed")
    distributions = _exact_dict(
        sparse["installed_distributions"],
        {"pyjnius", "pyserini"},
        name="inherited installed distributions",
    )
    for name, raw in distributions.items():
        record = _exact_dict(
            raw,
            {"file_count", "total_size", "tree_sha256"},
            name=f"installed distribution {name}",
        )
        if (
            type(record["file_count"]) is not int
            or type(record["total_size"]) is not int
        ):
            raise ValueError(f"Installed distribution {name} identity changed")
    if normalized["module_origins"] != {
        "processing_fold_eval.archive_bridge": "processing_fold_eval/archive_bridge.py",
        "retriever.artifacts": "retriever/artifacts.py",
        "retriever.evaluator": "retriever/evaluator.py",
        "retriever.provenance": "retriever/provenance.py",
        "retriever.staged_data": "retriever/staged_data.py",
    }:
        raise ValueError("Portable module-origin identity changed")
    return normalized


def _validate_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} must be one non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} is not one normalized relative POSIX path")
    return value


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{name} is absent: {path}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} contains a symlink component: {current}")
    if not stat.S_ISDIR(path.stat().st_mode):
        raise ValueError(f"{name} must be a real directory: {path}")
    return path


def _require_absent_output(path: Path) -> tuple[Path, Path]:
    path = Path(path).absolute()
    parent = _require_real_directory(path.parent, name="Build-context output parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Build-context output must be absent: {path}")
    incomplete = parent / f".{path.name}.incomplete"
    if incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError(f"Incomplete build-context path must be absent: {incomplete}")
    return path, incomplete


def _run_git(root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Git command failed ({' '.join(arguments)}): "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _git_root(modernbert_dir: Path) -> Path:
    raw = _run_git(modernbert_dir, ("rev-parse", "--show-toplevel"))
    root = _require_real_directory(
        Path(raw.decode("utf-8", errors="strict").strip()), name="Git root"
    )
    try:
        modernbert_dir.relative_to(root)
    except ValueError as error:
        raise ValueError("ModernBERT directory is outside its Git worktree") from error
    return root


def _local_toolchain_identity() -> dict[str, str]:
    buildx = subprocess.run(
        ["docker", "buildx", "version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if buildx.returncode != 0:
        raise RuntimeError(f"docker buildx version failed: {buildx.stderr.strip()}")
    match = re.fullmatch(
        r"github\.com/docker/buildx "
        r"(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)) "
        r"[0-9a-f]{40}",
        buildx.stdout.strip(),
    )
    if match is None:
        raise RuntimeError(f"Unexpected docker buildx version output: {buildx.stdout!r}")
    inspect = subprocess.run(
        ["docker", "buildx", "inspect", "--bootstrap"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if inspect.returncode != 0:
        raise RuntimeError(f"docker buildx inspect failed: {inspect.stderr.strip()}")
    drivers = re.findall(r"^Driver:\s+(\S+)$", inspect.stdout, flags=re.MULTILINE)
    versions = re.findall(
        r"^BuildKit version:\s+"
        r"(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))$",
        inspect.stdout,
        flags=re.MULTILINE,
    )
    if drivers != ["docker"] or len(versions) != 1:
        raise RuntimeError(
            "Current buildx builder must expose one exact docker driver/BuildKit: "
            f"drivers={drivers}, versions={versions}"
        )
    return {
        "builder_driver": "docker",
        "buildkit_version": versions[0],
        "buildx_version": match.group(1),
    }


def _validate_parent_inputs(
    git_root: Path,
    *,
    source_parent_commit: str,
    source_parent_epoch: int,
) -> tuple[str, dict[str, str]]:
    if (
        type(source_parent_commit) is not str
        or _COMMIT_RE.fullmatch(source_parent_commit) is None
    ):
        raise ValueError("source_parent_commit must be one lowercase 40-hex commit")
    if type(source_parent_epoch) is not int or source_parent_epoch <= 0:
        raise ValueError("source_parent_epoch must be one positive exact integer")
    resolved = _run_git(
        git_root, ("rev-parse", "--verify", f"{source_parent_commit}^{{commit}}")
    ).decode("ascii", errors="strict").strip()
    if resolved != source_parent_commit:
        raise ValueError("source_parent_commit did not resolve to itself")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(git_root),
            "merge-base",
            "--is-ancestor",
            source_parent_commit,
            "HEAD",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ancestry.returncode == 1:
        raise ValueError("source_parent_commit must be an ancestor of HEAD")
    if ancestry.returncode != 0:
        raise RuntimeError("Git ancestry validation failed")
    parent_epoch = _run_git(
        git_root, ("show", "-s", "--format=%ct", source_parent_commit)
    ).decode("ascii", errors="strict").strip()
    if not parent_epoch.isdigit() or int(parent_epoch) != source_parent_epoch:
        raise ValueError("source_parent_epoch differs from the selected commit timestamp")
    rfc3339 = datetime.fromtimestamp(source_parent_epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return rfc3339, _local_toolchain_identity()


def _load_source_contract(modernbert_dir: Path) -> tuple[dict[str, Any], list[str]]:
    path = modernbert_dir / "processing_fold_eval/image_contract.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Overlay image contract is absent or unsafe: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_compact_bytes(value):
        raise ValueError("Overlay image contract must be compact canonical JSON")
    expected_keys = {
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
    if set(value) != expected_keys or value["schema_version"] != 1:
        raise ValueError("Overlay image contract schema changed")
    if value["base_image"] != {
        "config_digest": BASE_IMAGE_CONFIG_DIGEST,
        "digest": BASE_IMAGE_URI.rsplit("@", 1)[1],
        "uri": BASE_IMAGE_URI,
    }:
        raise ValueError("Overlay base image identity changed")
    if (
        value["dockerfile_frontend"] != DOCKERFILE_FRONTEND
        or value["platform"] != PLATFORM
        or value["build_exporter"] != BUILD_EXPORTER
    ):
        raise ValueError("Overlay build contract changed")
    if value["build_manifest"] != {
        "directory_mode": DIRECTORY_MODE,
        "file_mode": FILE_MODE,
        "manifest_type": MANIFEST_TYPE,
        "path": MANIFEST_RELATIVE_PATH,
        "protocol": FREEZE_PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError("Overlay build-manifest contract changed")
    if (
        value["entrypoint"]
        != [
            "/opt/conda/bin/python",
            "/opt/program/modernbert/processing_fold_eval/evaluate_sm.py",
        ]
        or value["inventory_entrypoint"]
        != [
            "/opt/conda/bin/python",
            "/opt/program/modernbert/processing_fold_eval/inventory_sm.py",
        ]
        or value["environment"] != EXPECTED_ENVIRONMENT
        or value["processing_layout"] != EXPECTED_PROCESSING_LAYOUT
        or value["program_root"] != "/opt/program/modernbert"
        or value["workdir"] != "/opt/program/modernbert"
        or value["inherited_runtime"]
        != {
            "build_identity_sha256": (
                "249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8"
            ),
            "build_manifest_path": "processing_eval/build_context_manifest.json",
            "files_sha256": (
                "96f8b4e5569404ed916cd69c4d765b3eb34cbd3f40e3eff8394e9de72f415dc4"
            ),
            "image_contract_path": "processing_eval/image_contract.json",
            "image_contract_sha256": (
                "c0dba1f1a2387bce425b6c33f83e5035d3904ccb62de0e4f1422602ead0cbca8"
            ),
        }
    ):
        raise ValueError("Overlay runtime or Processing layout contract changed")
    inventory = value["source_inventory"]
    if type(inventory) is not list:
        raise ValueError("Overlay source inventory is absent")
    paths = [
        _validate_relative_path(item, name="source_inventory entry")
        for item in inventory
    ]
    if paths != EXPECTED_SOURCE_INVENTORY:
        raise ValueError("Overlay source inventory changed")
    required = {
        "processing_fold_eval/build_context.py",
        "processing_fold_eval/evaluate_sm.py",
        "processing_fold_eval/image_contract.json",
        "processing_fold_eval/image_smoke.py",
        "processing_fold_eval/inventory_sm.py",
        "retriever/artifacts.py",
        "retriever/evaluator.py",
        "retriever/provenance.py",
        "retriever/staged_data.py",
    }
    if not required.issubset(paths):
        raise ValueError("Overlay source inventory is incomplete")
    return value, sorted(_CONTROL_PATHS | set(paths))


def _stable_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_live_sources(
    modernbert_dir: Path, *, paths: Sequence[str]
) -> dict[str, bytes]:
    opened: list[tuple[str, Path, int, os.stat_result]] = []

    def path_metadata(relative: str, source: Path) -> os.stat_result:
        current = modernbert_dir
        metadata: os.stat_result | None = None
        for part in PurePosixPath(relative).parts:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Build-context source contains a symlink: {current}")
        assert metadata is not None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Build-context source must be singly linked: {source}")
        return metadata

    def validate_paths() -> None:
        for relative, source, _, expected in opened:
            if _stable_identity(path_metadata(relative, source)) != _stable_identity(
                expected
            ):
                raise RuntimeError(f"Build-context source path changed: {source}")

    try:
        for relative in paths:
            source = modernbert_dir / relative
            path_metadata(relative, source)
            descriptor = os.open(
                source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                os.close(descriptor)
                raise ValueError(f"Build-context source is not singly linked: {source}")
            if stat.S_IMODE(metadata.st_mode) & 0o111:
                os.close(descriptor)
                raise ValueError(f"Build-context source became executable: {relative}")
            opened.append((relative, source, descriptor, metadata))
        validate_paths()
        payloads: dict[str, bytes] = {}
        for relative, source, descriptor, expected in opened:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
            if (
                _stable_identity(os.fstat(descriptor)) != _stable_identity(expected)
                or len(payload) != expected.st_size
            ):
                raise RuntimeError(f"Build-context source changed while read: {source}")
            payloads[relative] = payload
        validate_paths()
        return payloads
    finally:
        for _, _, descriptor, _ in opened:
            os.close(descriptor)


def _file_records(payloads: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "mode": FILE_MODE,
            "path": path,
            "sha256": _sha256_bytes(payload),
            "size": len(payload),
            "type": "regular_file",
        }
        for path, payload in sorted(payloads.items())
    ]


def _identity_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "base_image",
            "dockerfile_frontend",
            "exporter",
            "files",
            "files_sha256",
            "manifest_type",
            "platform",
            "protocol",
            "schema_version",
            "source_parent_commit",
            "source_parent_epoch",
            "source_parent_rfc3339",
            "toolchain",
        )
    }


def _build_manifest(
    payloads: Mapping[str, bytes],
    *,
    source_parent_commit: str,
    source_parent_epoch: int,
    source_parent_rfc3339: str,
    toolchain: Mapping[str, str],
) -> dict[str, object]:
    files = _file_records(payloads)
    manifest: dict[str, object] = {
        "base_image": BASE_IMAGE_URI,
        "dockerfile_frontend": DOCKERFILE_FRONTEND,
        "exporter": BUILD_EXPORTER,
        "files": files,
        "files_sha256": _sha256_bytes(_canonical_json(files).encode("utf-8")),
        "manifest_type": MANIFEST_TYPE,
        "platform": PLATFORM,
        "protocol": FREEZE_PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_parent_commit": source_parent_commit,
        "source_parent_epoch": source_parent_epoch,
        "source_parent_rfc3339": source_parent_rfc3339,
        "toolchain": dict(toolchain),
    }
    identity = _sha256_bytes(
        _canonical_json(_identity_payload(manifest)).encode("utf-8")
    )
    manifest["build_identity_sha256"] = identity
    manifest["content_tag"] = f"build-sha256-{identity}"
    return manifest


def _validate_manifest_value(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("Overlay build-context manifest must be one object")
    manifest = value
    expected_keys = {
        "base_image",
        "build_identity_sha256",
        "content_tag",
        "dockerfile_frontend",
        "exporter",
        "files",
        "files_sha256",
        "manifest_type",
        "platform",
        "protocol",
        "schema_version",
        "source_parent_commit",
        "source_parent_epoch",
        "source_parent_rfc3339",
        "toolchain",
    }
    if set(manifest) != expected_keys:
        raise ValueError("Overlay build-context manifest schema changed")
    if (
        manifest["base_image"] != BASE_IMAGE_URI
        or manifest["dockerfile_frontend"] != DOCKERFILE_FRONTEND
        or manifest["exporter"] != BUILD_EXPORTER
        or manifest["manifest_type"] != MANIFEST_TYPE
        or manifest["platform"] != PLATFORM
        or manifest["protocol"] != FREEZE_PROTOCOL
        or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or type(manifest["schema_version"]) is not int
    ):
        raise ValueError("Overlay build-context fixed identity changed")
    toolchain = manifest["toolchain"]
    if type(toolchain) is not dict or set(toolchain) != {
        "builder_driver",
        "buildkit_version",
        "buildx_version",
    }:
        raise ValueError("Overlay build-context toolchain schema changed")
    if toolchain["builder_driver"] != "docker":
        raise ValueError("Overlay build-context builder driver changed")
    for name in ("buildkit_version", "buildx_version"):
        if (
            type(toolchain[name]) is not str
            or _TOOL_VERSION_RE.fullmatch(toolchain[name]) is None
        ):
            raise ValueError(f"Overlay build-context {name} is malformed")
    commit = manifest["source_parent_commit"]
    epoch = manifest["source_parent_epoch"]
    if type(commit) is not str or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("Overlay source_parent_commit is malformed")
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("Overlay source_parent_epoch is malformed")
    if manifest["source_parent_rfc3339"] != datetime.fromtimestamp(
        epoch, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ"):
        raise ValueError("Overlay source_parent_rfc3339 is not derived")
    files = manifest["files"]
    if type(files) is not list or not files:
        raise ValueError("Overlay build-context files must be non-empty")
    paths: list[str] = []
    for index, record in enumerate(files):
        if type(record) is not dict or set(record) != {
            "mode",
            "path",
            "sha256",
            "size",
            "type",
        }:
            raise ValueError(f"Overlay file record {index} schema changed")
        path = _validate_relative_path(record["path"], name=f"files[{index}].path")
        if (
            record["mode"] != FILE_MODE
            or record["type"] != "regular_file"
            or type(record["size"]) is not int
            or record["size"] < 0
            or type(record["sha256"]) is not str
            or _SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Overlay file record {index} is malformed")
        paths.append(path)
    if paths != sorted(set(paths)) or MANIFEST_RELATIVE_PATH in paths:
        raise ValueError("Overlay file records are not sorted, unique, and self-excluding")
    files_sha256 = _sha256_bytes(_canonical_json(files).encode("utf-8"))
    if manifest["files_sha256"] != files_sha256:
        raise ValueError("Overlay files_sha256 changed")
    identity = _sha256_bytes(
        _canonical_json(_identity_payload(manifest)).encode("utf-8")
    )
    if manifest["build_identity_sha256"] != identity:
        raise ValueError("Overlay build identity changed")
    if manifest["content_tag"] != f"build-sha256-{identity}":
        raise ValueError("Overlay content tag changed")
    return manifest


def load_build_context_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Overlay manifest must be singly linked and regular: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o644:
        raise ValueError("Overlay manifest mode must be 0644")
    raw = path.read_bytes()
    value = json.loads(raw)
    if raw != _canonical_pretty_bytes(value):
        raise ValueError("Overlay manifest must use canonical pretty JSON")
    return _validate_manifest_value(value)


def _expected_directories(paths: Sequence[str]) -> set[str]:
    result = {"."}
    for relative in paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def _validate_exact_tree(root: Path, manifest: Mapping[str, Any]) -> None:
    root = _require_real_directory(root, name="Frozen overlay context")
    records = {record["path"]: record for record in manifest["files"]}
    expected_files = set(records) | {MANIFEST_RELATIVE_PATH}
    expected_directories = _expected_directories(sorted(expected_files))
    actual_files: set[str] = set()
    actual_directories = {"."}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root).as_posix()
        if stat.S_IMODE(current_path.stat().st_mode) != 0o755:
            raise ValueError(f"Frozen overlay directory mode changed: {relative_current}")
        for name in tuple(directories):
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"Frozen overlay contains an unsafe directory: {relative}")
            actual_directories.add(relative)
        for name in files:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ValueError(f"Frozen overlay contains an unsafe file: {relative}")
            actual_files.add(relative)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "Frozen overlay inventory changed: "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"extra_files={sorted(actual_files - expected_files)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}, "
            f"extra_directories={sorted(actual_directories - expected_directories)}"
        )
    for relative, record in records.items():
        path = root / relative
        metadata = path.stat()
        if (
            stat.S_IMODE(metadata.st_mode) != int(record["mode"], 8)
            or metadata.st_size != record["size"]
            or _sha256_bytes(path.read_bytes()) != record["sha256"]
        ):
            raise ValueError(f"Frozen overlay file identity changed: {relative}")
    if stat.S_IMODE((root / MANIFEST_RELATIVE_PATH).stat().st_mode) != 0o644:
        raise ValueError("Frozen overlay manifest mode changed")


def validate_frozen_build_context(root: Path) -> dict[str, Any]:
    root = _require_real_directory(root, name="Frozen overlay context")
    manifest = load_build_context_manifest(root / MANIFEST_RELATIVE_PATH)
    _validate_exact_tree(root, manifest)
    _, expected_paths = _load_source_contract(root)
    if [record["path"] for record in manifest["files"]] != expected_paths:
        raise ValueError("Frozen overlay manifest left the exact source allowlist")
    return manifest


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise RuntimeError("Exclusive file writer made no progress")
            offset += written
        os.fsync(descriptor)
        os.chmod(path, 0o644, follow_symlinks=False)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Linux renameat2 is required for absent-only publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(f"Overlay output appeared: {destination}")
        raise OSError(number, os.strerror(number), str(destination))


def freeze_build_context(
    modernbert_dir: Path,
    output_dir: Path,
    *,
    source_parent_commit: str,
    source_parent_epoch: int,
) -> dict[str, Any]:
    modernbert_dir = _require_real_directory(modernbert_dir, name="ModernBERT root")
    output_dir, incomplete = _require_absent_output(output_dir)
    git_root = _git_root(modernbert_dir)
    rfc3339, toolchain = _validate_parent_inputs(
        git_root,
        source_parent_commit=source_parent_commit,
        source_parent_epoch=source_parent_epoch,
    )
    _, paths = _load_source_contract(modernbert_dir)
    payloads = _read_live_sources(modernbert_dir, paths=paths)
    if _read_live_sources(modernbert_dir, paths=paths) != payloads:
        raise RuntimeError("Overlay sources changed between complete reads")
    manifest = _build_manifest(
        payloads,
        source_parent_commit=source_parent_commit,
        source_parent_epoch=source_parent_epoch,
        source_parent_rfc3339=rfc3339,
        toolchain=toolchain,
    )
    owns_incomplete = False
    incomplete_identity: tuple[int, int] | None = None
    try:
        incomplete.mkdir(mode=0o700)
        owns_incomplete = True
        metadata = incomplete.lstat()
        incomplete_identity = (metadata.st_dev, metadata.st_ino)
        os.chmod(incomplete, 0o755)
        for relative, payload in sorted(payloads.items()):
            destination = incomplete / relative
            destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            os.chmod(destination.parent, 0o755)
            _write_exclusive(destination, payload)
        manifest_path = incomplete / MANIFEST_RELATIVE_PATH
        manifest_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        os.chmod(manifest_path.parent, 0o755)
        _write_exclusive(manifest_path, _canonical_pretty_bytes(manifest))
        for directory in sorted(
            (path for path in incomplete.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(incomplete)
        validate_frozen_build_context(incomplete)
        _rename_no_replace(incomplete, output_dir)
        owns_incomplete = False
        try:
            _fsync_directory(output_dir.parent)
            validate_frozen_build_context(output_dir)
        except BaseException:
            diagnostic = Path(
                tempfile.mkdtemp(
                    dir=output_dir.parent, prefix=f".{output_dir.name}.invalid."
                )
            )
            marker = output_dir / MANIFEST_RELATIVE_PATH
            marker.unlink()
            _fsync_directory(marker.parent)
            _fsync_directory(output_dir)
            _rename_no_replace(output_dir, diagnostic / "context")
            _fsync_directory(diagnostic)
            _fsync_directory(output_dir.parent)
            raise
    finally:
        if owns_incomplete:
            metadata = incomplete.lstat()
            if (metadata.st_dev, metadata.st_ino) != incomplete_identity:
                raise RuntimeError("Owned incomplete overlay context was replaced")
            shutil.rmtree(incomplete)
            _fsync_directory(incomplete.parent)
    return manifest


def _buildx_command(
    frozen_context: Path,
    metadata_file: Path,
    *,
    manifest: Mapping[str, Any],
    build_replica: int,
) -> tuple[list[str], str]:
    if type(build_replica) is not int or build_replica not in {1, 2}:
        raise ValueError("build_replica must be exact integer 1 or 2")
    image_name = f"{LOCAL_IMAGE_REPOSITORY}:{manifest['content_tag']}-build{build_replica}"
    exporter = manifest["exporter"]
    output = ",".join(
        (
            f"type={exporter['type']}",
            f"name={image_name}",
            f"push={str(exporter['push']).lower()}",
            f"rewrite-timestamp={str(exporter['rewrite_timestamp']).lower()}",
            f"unpack={str(exporter['unpack']).lower()}",
            f"compression={exporter['compression']}",
            f"compression-level={exporter['compression_level']}",
            f"force-compression={str(exporter['force_compression']).lower()}",
            f"oci-mediatypes={str(exporter['oci_mediatypes']).lower()}",
        )
    )
    return (
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            manifest["platform"],
            "--pull",
            "--no-cache",
            f"--provenance={str(exporter['provenance']).lower()}",
            f"--sbom={str(exporter['sbom']).lower()}",
            "--output",
            output,
            "--build-arg",
            f"SOURCE_PARENT_COMMIT={manifest['source_parent_commit']}",
            "--build-arg",
            f"SOURCE_PARENT_EPOCH={manifest['source_parent_epoch']}",
            "--build-arg",
            f"SOURCE_PARENT_RFC3339={manifest['source_parent_rfc3339']}",
            "--build-arg",
            f"BUILD_IDENTITY_SHA256={manifest['build_identity_sha256']}",
            "--metadata-file",
            str(metadata_file),
            "--file",
            str(frozen_context / "processing_fold_eval/Dockerfile"),
            str(frozen_context),
        ],
        image_name,
    )


def _validate_build_metadata(
    metadata_path: Path,
    *,
    manifest: Mapping[str, Any],
    image_name: str,
) -> dict[str, str]:
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise RuntimeError("Buildx metadata output is absent or unsafe")
    metadata = json.loads(metadata_path.read_bytes())
    if type(metadata) is not dict or set(metadata) != {
        "buildx.build.provenance",
        "buildx.build.ref",
        "containerimage.config.digest",
        "containerimage.descriptor",
        "containerimage.digest",
        "image.name",
    }:
        raise RuntimeError("Buildx metadata schema changed")
    config_digest = metadata["containerimage.config.digest"]
    image_digest = metadata["containerimage.digest"]
    for name, digest in (("config", config_digest), ("image", image_digest)):
        if (
            type(digest) is not str
            or not digest.startswith("sha256:")
            or _SHA256_RE.fullmatch(digest.removeprefix("sha256:")) is None
        ):
            raise RuntimeError(f"Buildx returned a malformed {name} digest")
    descriptor = metadata["containerimage.descriptor"]
    if type(descriptor) is not dict or set(descriptor) not in (
        {"digest", "mediaType", "platform", "size"},
        {"annotations", "digest", "mediaType", "platform", "size"},
    ):
        raise RuntimeError("Buildx image descriptor schema changed")
    if (
        descriptor["digest"] != image_digest
        or descriptor["mediaType"] != DOCKER_MANIFEST_MEDIA_TYPE
        or descriptor["platform"] != {"architecture": "amd64", "os": "linux"}
        or type(descriptor["size"]) is not int
        or descriptor["size"] <= 0
    ):
        raise RuntimeError("Buildx image descriptor identity changed")
    if "annotations" in descriptor and descriptor["annotations"] != {
        "org.opencontainers.image.created": manifest["source_parent_rfc3339"]
    }:
        raise RuntimeError("Buildx descriptor annotations changed")
    expected_name = f"docker.io/library/{image_name}"
    if metadata["image.name"] != expected_name:
        raise RuntimeError("Buildx image name changed")
    build_ref = metadata["buildx.build.ref"]
    if type(build_ref) is not str or re.fullmatch(
        r"[^/\s]+/[^/\s]+/[a-z0-9]+", build_ref
    ) is None:
        raise RuntimeError("Buildx build reference is malformed")
    provenance = metadata["buildx.build.provenance"]
    if type(provenance) is not dict:
        raise RuntimeError("Buildx provenance is absent")
    expected_parameters = {
        "args": {
            "build-arg:BUILD_IDENTITY_SHA256": manifest["build_identity_sha256"],
            "build-arg:SOURCE_DATE_EPOCH": str(manifest["source_parent_epoch"]),
            "build-arg:SOURCE_PARENT_COMMIT": manifest["source_parent_commit"],
            "build-arg:SOURCE_PARENT_EPOCH": str(manifest["source_parent_epoch"]),
            "build-arg:SOURCE_PARENT_RFC3339": manifest["source_parent_rfc3339"],
            "cmdline": DOCKERFILE_FRONTEND,
            "no-cache": "",
            "source": DOCKERFILE_FRONTEND,
        },
        "frontend": "gateway.v0",
        "locals": [{"name": "context"}, {"name": "dockerfile"}],
    }
    invocation = provenance.get("invocation")
    if (
        provenance.get("buildType") != "https://mobyproject.org/buildkit@v1"
        or provenance.get("builder") != {"id": ""}
        or type(invocation) is not dict
        or invocation.get("configSource") != {"entryPoint": "Dockerfile"}
        or invocation.get("environment") != {"platform": PLATFORM}
        or invocation.get("parameters") != expected_parameters
    ):
        raise RuntimeError("Buildx provenance invocation changed")
    materials = provenance.get("materials")
    base_digest = BASE_IMAGE_URI.rsplit("@sha256:", 1)[1]
    frontend_digest = DOCKERFILE_FRONTEND.rsplit("@sha256:", 1)[1]
    expected_materials = [
        {
            "digest": {"sha256": base_digest},
            "uri": (
                "pkg:docker/371087393859.dkr.ecr.us-east-1.amazonaws.com/"
                "arr-retrieval-eval?digest=sha256:"
                f"{base_digest}&platform=linux%2Famd64"
            ),
        },
        {
            "digest": {"sha256": frontend_digest},
            "uri": (
                "pkg:docker/docker/dockerfile@1.7?digest=sha256:"
                f"{frontend_digest}&platform=linux%2Famd64"
            ),
        },
        {
            "digest": {"sha256": frontend_digest},
            "uri": f"pkg:docker/docker/dockerfile@1.7?digest=sha256:{frontend_digest}",
        },
    ]
    if materials != expected_materials or set(provenance) != {
        "buildType",
        "builder",
        "invocation",
        "materials",
    }:
        raise RuntimeError("Buildx provenance materials changed")
    return {
        "build_ref": build_ref,
        "config_digest": config_digest,
        "image_digest": image_digest,
        "image_name": expected_name,
        "manifest_media_type": DOCKER_MANIFEST_MEDIA_TYPE,
    }


def _validate_local_image(
    image_name: str,
    *,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    build_metadata: Mapping[str, str],
) -> dict[str, object]:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker image inspect failed: {completed.stderr.strip()}")
    inspected = json.loads(completed.stdout)
    if type(inspected) is not list or len(inspected) != 1 or type(inspected[0]) is not dict:
        raise RuntimeError("docker image inspect returned an unexpected schema")
    image = inspected[0]
    if (
        image.get("Id") != build_metadata["image_digest"]
        or image.get("Architecture") != "amd64"
        or image.get("Os") != "linux"
        or image.get("RepoDigests")
        != [f"{LOCAL_IMAGE_REPOSITORY}@{build_metadata['image_digest']}"]
    ):
        raise RuntimeError("Local overlay image identity/platform changed")
    config = image.get("Config")
    if type(config) is not dict:
        raise RuntimeError("Local overlay image config is absent")
    if (
        config.get("Entrypoint") != contract["entrypoint"]
        or config.get("WorkingDir") != contract["workdir"]
    ):
        raise RuntimeError("Local overlay entrypoint/workdir changed")
    raw_environment = config.get("Env")
    if type(raw_environment) is not list or any(
        type(item) is not str or "=" not in item for item in raw_environment
    ):
        raise RuntimeError("Local overlay environment schema changed")
    environment: dict[str, str] = {}
    for item in raw_environment:
        name, value = item.split("=", 1)
        if name in environment:
            raise RuntimeError(f"Local overlay environment duplicates {name!r}")
        environment[name] = value
    expected_environment = {
        **contract["environment"],
        "SOURCE_DATE_EPOCH": str(manifest["source_parent_epoch"]),
    }
    if any(environment.get(name) != value for name, value in expected_environment.items()):
        raise RuntimeError("Local overlay deterministic environment changed")
    labels = config.get("Labels")
    expected_labels = {
        "io.arr-retrieval-fold-eval.base-config-digest": BASE_IMAGE_CONFIG_DIGEST,
        "io.arr-retrieval-fold-eval.build-identity-sha256": manifest[
            "build_identity_sha256"
        ],
        "io.arr-retrieval-fold-eval.source-parent-commit": manifest[
            "source_parent_commit"
        ],
        "io.arr-retrieval-fold-eval.source-parent-epoch": str(
            manifest["source_parent_epoch"]
        ),
        "io.arr-retrieval-fold-eval.source-parent-rfc3339": manifest[
            "source_parent_rfc3339"
        ],
        "org.opencontainers.image.base.digest": contract["base_image"]["digest"],
    }
    if type(labels) is not dict or any(
        labels.get(name) != value for name, value in expected_labels.items()
    ):
        raise RuntimeError("Local overlay provenance labels changed")
    return {
        "environment": expected_environment,
        "labels": expected_labels,
        "repo_digests": image["RepoDigests"],
    }


def _run_offline_image_smoke(
    image_name: str, *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777",
        "--entrypoint",
        "/opt/conda/bin/python",
        image_name,
        "/opt/program/modernbert/processing_fold_eval/image_smoke.py",
        "--contract",
        "/opt/program/modernbert/processing_fold_eval/image_contract.json",
        "--build-manifest",
        "/opt/program/modernbert/processing_fold_eval/build_context_manifest.json",
        "--expected-build-identity-sha256",
        manifest["build_identity_sha256"],
        "--expected-source-parent-commit",
        manifest["source_parent_commit"],
        "--expected-source-parent-epoch",
        str(manifest["source_parent_epoch"]),
        "--expected-source-parent-rfc3339",
        manifest["source_parent_rfc3339"],
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Offline overlay image smoke failed: "
            f"returncode={completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Offline overlay image smoke returned invalid JSON") from error
    if (
        type(payload) is not dict
        or completed.stdout != _canonical_json(payload) + "\n"
        or payload.get("build_context", {}).get("build_identity_sha256")
        != manifest["build_identity_sha256"]
    ):
        raise RuntimeError("Offline overlay image smoke identity changed")
    return payload


def build_frozen_image(
    frozen_context: Path,
    metadata_file: Path,
    *,
    build_replica: int,
) -> dict[str, object]:
    frozen_context = _require_real_directory(
        frozen_context, name="Frozen overlay build context"
    )
    metadata_file = Path(metadata_file).absolute()
    _require_real_directory(metadata_file.parent, name="Build metadata parent")
    if metadata_file.exists() or metadata_file.is_symlink():
        raise FileExistsError(f"Build metadata output must be absent: {metadata_file}")
    manifest = validate_frozen_build_context(frozen_context)
    contract, _ = _load_source_contract(frozen_context)
    actual_toolchain = _local_toolchain_identity()
    if actual_toolchain != manifest["toolchain"]:
        raise RuntimeError("Active Docker builder differs from the frozen toolchain")
    command, image_name = _buildx_command(
        frozen_context,
        metadata_file,
        manifest=manifest,
        build_replica=build_replica,
    )
    environment = dict(os.environ)
    environment["BUILDX_METADATA_PROVENANCE"] = "min"
    environment["SOURCE_DATE_EPOCH"] = str(manifest["source_parent_epoch"])
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"Exact overlay docker build failed: {completed.returncode}")
    if validate_frozen_build_context(frozen_context) != manifest:
        raise RuntimeError("Frozen overlay context changed while BuildKit consumed it")
    build_metadata = _validate_build_metadata(
        metadata_file, manifest=manifest, image_name=image_name
    )
    local_image = _validate_local_image(
        image_name,
        manifest=manifest,
        contract=contract,
        build_metadata=build_metadata,
    )
    smoke = _run_offline_image_smoke(image_name, manifest=manifest)
    return {
        **build_metadata,
        "build_context_files_sha256": manifest["files_sha256"],
        "build_context_identity_sha256": manifest["build_identity_sha256"],
        "build_replica": build_replica,
        "content_tag": manifest["content_tag"],
        "local_image_identity_sha256": _sha256_bytes(
            _canonical_json(local_image).encode("utf-8")
        ),
        "image_runtime_identity": smoke,
        "offline_smoke_sha256": _sha256_bytes(
            _canonical_json(smoke).encode("utf-8")
        ),
    }


def validate_reproducible_builds(
    first: Mapping[str, object], second: Mapping[str, object]
) -> dict[str, object]:
    """Require independently built replicas to have one config and manifest."""

    required = {
        "build_context_files_sha256",
        "build_context_identity_sha256",
        "build_ref",
        "build_replica",
        "config_digest",
        "content_tag",
        "image_digest",
        "image_name",
        "image_runtime_identity",
        "local_image_identity_sha256",
        "manifest_media_type",
        "offline_smoke_sha256",
    }
    for name, value in (("first", first), ("second", second)):
        if type(value) is not dict or set(value) != required:
            raise ValueError(f"{name} build receipt schema changed")
        for field in (
            "build_context_files_sha256",
            "build_context_identity_sha256",
            "local_image_identity_sha256",
            "offline_smoke_sha256",
        ):
            if (
                type(value[field]) is not str
                or _SHA256_RE.fullmatch(value[field]) is None
            ):
                raise ValueError(f"{name} build receipt {field} is malformed")
        for field in ("config_digest", "image_digest"):
            digest = value[field]
            if (
                type(digest) is not str
                or not digest.startswith("sha256:")
                or _SHA256_RE.fullmatch(digest.removeprefix("sha256:")) is None
            ):
                raise ValueError(f"{name} build receipt {field} is malformed")
        if value["manifest_media_type"] != DOCKER_MANIFEST_MEDIA_TYPE:
            raise ValueError(f"{name} build receipt media type changed")
        runtime_identity = _validate_image_runtime_identity(
            value["image_runtime_identity"],
            build_context_files_sha256=value["build_context_files_sha256"],
            build_context_identity_sha256=value[
                "build_context_identity_sha256"
            ],
        )
        if (
            _sha256_bytes(_canonical_json(runtime_identity).encode("utf-8"))
            != value["offline_smoke_sha256"]
        ):
            raise ValueError(f"{name} build receipt runtime identity changed")
        if value["content_tag"] != (
            "build-sha256-" + value["build_context_identity_sha256"]
        ):
            raise ValueError(f"{name} build receipt content tag is malformed")
        if type(value["image_name"]) is not str:
            raise ValueError(f"{name} build receipt image name is malformed")
    if first["build_replica"] != 1 or type(first["build_replica"]) is not int:
        raise ValueError("First build receipt must be exact replica 1")
    if second["build_replica"] != 2 or type(second["build_replica"]) is not int:
        raise ValueError("Second build receipt must be exact replica 2")
    build_ref_pattern = re.compile(r"[^/\s]+/[^/\s]+/[a-z0-9]+\Z")
    if any(
        type(receipt["build_ref"]) is not str
        or build_ref_pattern.fullmatch(receipt["build_ref"]) is None
        for receipt in (first, second)
    ):
        raise ValueError("Independent overlay build reference is malformed")
    if first["build_ref"] == second["build_ref"]:
        raise RuntimeError("Independent overlay builds reused one BuildKit reference")
    if first["content_tag"] != second["content_tag"]:
        raise RuntimeError("Independent overlay builds used different context tags")
    expected_names = (
        f"docker.io/library/{LOCAL_IMAGE_REPOSITORY}:"
        f"{first['content_tag']}-build1",
        f"docker.io/library/{LOCAL_IMAGE_REPOSITORY}:"
        f"{first['content_tag']}-build2",
    )
    if (first["image_name"], second["image_name"]) != expected_names:
        raise RuntimeError("Independent overlay build names/ordinals changed")
    comparable_keys = {
        "build_context_files_sha256",
        "build_context_identity_sha256",
        "config_digest",
        "content_tag",
        "image_digest",
        "image_runtime_identity",
        "local_image_identity_sha256",
        "manifest_media_type",
        "offline_smoke_sha256",
    }
    comparable = {key: first[key] for key in comparable_keys}
    other = {key: second[key] for key in comparable_keys}
    if comparable != other:
        raise RuntimeError("Independent overlay builds are not byte-identical")
    return {
        "build_context_files_sha256": str(first["build_context_files_sha256"]),
        "build_context_identity_sha256": str(first["build_context_identity_sha256"]),
        "config_digest": str(first["config_digest"]),
        "image_digest": str(first["image_digest"]),
        "image_runtime_identity": json.loads(
            _canonical_json(first["image_runtime_identity"])
        ),
        "local_image_identity_sha256": str(
            first["local_image_identity_sha256"]
        ),
        "manifest_media_type": str(first["manifest_media_type"]),
        "offline_smoke_sha256": str(first["offline_smoke_sha256"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or build the ARR fold-evaluation overlay image.",
        allow_abbrev=False,
    )
    parser.add_argument("--modernbert-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-parent-commit")
    parser.add_argument("--source-parent-epoch", type=int)
    parser.add_argument("--validate-frozen-context", type=Path)
    parser.add_argument("--build-frozen-context", type=Path)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--build-replica", type=int)
    parser.add_argument("--print-build-identity-sha256", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_frozen_context is not None:
        if any(
            value is not None
            for value in (
                args.modernbert_dir,
                args.output_dir,
                args.source_parent_commit,
                args.source_parent_epoch,
                args.validate_frozen_context,
            )
        ) or args.print_build_identity_sha256:
            raise ValueError("--build-frozen-context cannot be combined with freeze inputs")
        if args.metadata_file is None or args.build_replica is None:
            raise ValueError("--metadata-file and --build-replica are required")
        receipt = build_frozen_image(
            args.build_frozen_context,
            args.metadata_file,
            build_replica=args.build_replica,
        )
        print(_canonical_json(receipt))
        return 0
    if args.metadata_file is not None or args.build_replica is not None:
        raise ValueError("Build metadata inputs require --build-frozen-context")
    if args.validate_frozen_context is not None:
        if any(
            value is not None
            for value in (
                args.modernbert_dir,
                args.output_dir,
                args.source_parent_commit,
                args.source_parent_epoch,
            )
        ):
            raise ValueError("--validate-frozen-context cannot be combined with freeze inputs")
        manifest = validate_frozen_build_context(args.validate_frozen_context)
    else:
        required = {
            "modernbert_dir": args.modernbert_dir,
            "output_dir": args.output_dir,
            "source_parent_commit": args.source_parent_commit,
            "source_parent_epoch": args.source_parent_epoch,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"Missing required freeze inputs: {missing}")
        manifest = freeze_build_context(
            args.modernbert_dir,
            args.output_dir,
            source_parent_commit=args.source_parent_commit,
            source_parent_epoch=args.source_parent_epoch,
        )
    if args.print_build_identity_sha256:
        print(manifest["build_identity_sha256"])
    else:
        print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
