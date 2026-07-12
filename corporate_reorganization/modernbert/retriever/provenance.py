from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Mapping


EXPECTED_BASE_RUNTIME_VERSIONS = {
    "python": "3.11.10",
    "torch": "2.5.1+cu124",
    "transformers": "4.49.0",
    "accelerate": "1.4.0",
    "numpy": "1.26.4",
    "flash-attn": "2.7.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "huggingface-hub": "0.29.1",
}
EXPECTED_RUNTIME_VERSIONS = {
    **EXPECTED_BASE_RUNTIME_VERSIONS,
    "deepspeed": "0.17.1",
    "hjson": "3.1.0",
    "nvidia-ml-py": "13.590.48",
    "py-cpuinfo": "9.0.0",
}

EXPECTED_TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
EXPECTED_SNAPSHOT_TREE_SHA256 = (
    "aca85feea4adb60c4b021eb1a439aff47c844495005f2acdee1baef9d611d63d"
)

_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_version_inventory() -> dict[str, str]:
    inventory = {"python": platform.python_version()}
    for package in EXPECTED_RUNTIME_VERSIONS:
        if package == "python":
            continue
        try:
            inventory[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required runtime package is absent: {package}") from exc
    return inventory


def validate_runtime_versions(actual: Mapping[str, str] | None = None) -> dict[str, str]:
    inventory = dict(runtime_version_inventory() if actual is None else actual)
    if inventory != EXPECTED_RUNTIME_VERSIONS:
        missing = sorted(set(EXPECTED_RUNTIME_VERSIONS) - set(inventory))
        extra = sorted(set(inventory) - set(EXPECTED_RUNTIME_VERSIONS))
        mismatched = {
            name: {"expected": EXPECTED_RUNTIME_VERSIONS[name], "actual": inventory[name]}
            for name in sorted(set(inventory).intersection(EXPECTED_RUNTIME_VERSIONS))
            if inventory[name] != EXPECTED_RUNTIME_VERSIONS[name]
        }
        raise RuntimeError(
            "Training runtime does not match the frozen inventory: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    return inventory


def validate_preimport_environment(experiment_seed: int) -> None:
    if type(experiment_seed) is not int or experiment_seed < 0:
        raise ValueError("experiment_seed must be a non-negative exact int")
    expected = {
        "PYTHONHASHSEED": str(experiment_seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
    }
    mismatched = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatched:
        raise RuntimeError(
            "Required pre-import deterministic/offline environment is not exact: "
            f"{mismatched}"
        )


def load_snapshot_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Snapshot manifest must be a regular file: {path}")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    canonical = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if raw != canonical:
        raise ValueError(f"Snapshot manifest is not canonical JSON: {path}")
    if type(manifest) is not dict:
        raise TypeError("Snapshot manifest must be a JSON object")

    expected_keys = {
        "schema_version",
        "manifest_type",
        "model_id",
        "revision",
        "tree_sha256",
        "files",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"Snapshot manifest fields mismatch: missing={sorted(expected_keys - set(manifest))}, "
            f"extra={sorted(set(manifest) - expected_keys)}"
        )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Snapshot manifest schema_version must be exact integer 1")
    if manifest["manifest_type"] != "huggingface_model_snapshot":
        raise ValueError("Unexpected snapshot manifest_type")
    if manifest["model_id"] != "answerdotai/ModernBERT-base":
        raise ValueError("Unexpected snapshot model_id")
    if manifest["revision"] != "8949b909ec900327062f0ebf497f51aef5e6f0c8":
        raise ValueError("Unexpected snapshot revision")
    if (
        type(manifest["tree_sha256"]) is not str
        or _LOWER_SHA256.fullmatch(manifest["tree_sha256"]) is None
    ):
        raise ValueError("Snapshot tree_sha256 must be lowercase 64-hex")
    if type(manifest["files"]) is not list or not manifest["files"]:
        raise ValueError("Snapshot files must be a non-empty JSON list")

    expected_file_keys = {"path", "size", "sha256"}
    paths: list[str] = []
    for record in manifest["files"]:
        if type(record) is not dict or set(record) != expected_file_keys:
            raise ValueError("Every snapshot file record must contain exactly path, size, and sha256")
        relative_path = record["path"]
        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path != Path(relative_path).name
            or relative_path.strip() != relative_path
        ):
            raise ValueError(f"Snapshot file path must be one root-level filename: {relative_path!r}")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"Snapshot size must be a non-negative exact int: {relative_path}")
        if (
            type(record["sha256"]) is not str
            or _LOWER_SHA256.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Snapshot sha256 must be lowercase 64-hex: {relative_path}")
        paths.append(relative_path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Snapshot file records must be unique and sorted by path")

    expected_tree_hash = hashlib.sha256(_canonical_json_bytes(manifest["files"])).hexdigest()
    if manifest["tree_sha256"] != expected_tree_hash:
        raise ValueError(
            f"Snapshot tree hash mismatch: recorded={manifest['tree_sha256']}, "
            f"expected={expected_tree_hash}"
        )
    if manifest["tree_sha256"] != EXPECTED_SNAPSHOT_TREE_SHA256:
        raise ValueError(
            "Snapshot manifest is not the frozen ModernBERT tree: "
            f"actual={manifest['tree_sha256']}, expected={EXPECTED_SNAPSHOT_TREE_SHA256}"
        )
    return manifest


def validate_snapshot_directory(snapshot_dir: Path, manifest: Mapping[str, Any]) -> None:
    if not snapshot_dir.is_dir() or snapshot_dir.is_symlink():
        raise ValueError(f"Snapshot directory must be a real directory: {snapshot_dir}")
    expected_records = {record["path"]: record for record in manifest["files"]}
    actual_names = sorted(path.name for path in snapshot_dir.iterdir())
    expected_names = sorted(expected_records)
    if actual_names != expected_names:
        raise ValueError(
            f"Snapshot directory inventory mismatch: actual={actual_names}, expected={expected_names}"
        )

    for name in expected_names:
        path = snapshot_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Snapshot entry must be a regular non-symlink file: {path}")
        record = expected_records[name]
        actual_size = path.stat().st_size
        if actual_size != record["size"]:
            raise ValueError(
                f"Snapshot size mismatch for {name}: actual={actual_size}, expected={record['size']}"
            )
        actual_hash = _sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(
                f"Snapshot SHA-256 mismatch for {name}: actual={actual_hash}, expected={record['sha256']}"
            )
