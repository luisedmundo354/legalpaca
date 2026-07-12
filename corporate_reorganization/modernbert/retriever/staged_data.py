from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATASET_OUTPUT_PATHS = {
    "cases.jsonl",
    "corpus.jsonl",
    "pools/candidates_by_case.json",
    "pools/candidates_global.json",
    "queries/all.jsonl",
}
_FOLD_MANIFEST_KEYS = {
    "schema_version",
    "manifest_type",
    "generator",
    "dataset",
    "totals",
    "algorithm",
    "case_priority_order",
    "case_loads",
    "greedy",
    "pair_swap_refinement",
    "folds",
    "rotations",
}
_FOLD_RECORD_KEYS = {
    "capacity",
    "case_ids",
    "fold_id",
    "num_cases",
    "passage_share",
    "passages",
    "queries",
    "query_share",
}
_ROLE_KEYS = {"case_ids", "fold_ids", "num_cases", "passages", "queries"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_pretty_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    if raw != _pretty_json_bytes(value):
        raise ValueError(f"{name} does not use canonical pretty JSON bytes")
    return value


def _validate_logical_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be one exact non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} must be one normalized relative POSIX path")
    return value


def _validate_exact_dataset_inventory(
    dataset_dir: Path,
    *,
    output_sha256: Mapping[str, str],
) -> dict[str, Any]:
    expected_files = {"dataset_manifest.json", *_DATASET_OUTPUT_PATHS}
    expected_directories = {
        PurePosixPath(relative_path).parent.as_posix()
        for relative_path in _DATASET_OUTPUT_PATHS
        if PurePosixPath(relative_path).parent.as_posix() != "."
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in dataset_dir.rglob("*"):
        relative = path.relative_to(dataset_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"Staged dataset entry must not be a symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"Unexpected staged dataset entry type: {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "Staged dataset inventory changed: "
            f"files={sorted(actual_files)}, directories={sorted(actual_directories)}"
        )

    manifest = _load_pretty_json_object(
        dataset_dir / "dataset_manifest.json",
        name="Staged dataset manifest",
    )
    if manifest.get("schema_version") != 2 or type(manifest.get("schema_version")) is not int:
        raise ValueError("Staged dataset schema_version must be exact integer 2")
    counts = manifest.get("counts")
    if type(counts) is not dict or {
        key: counts.get(key) for key in ("cases", "queries", "passages")
    } != {"cases": 42, "queries": 490, "passages": 5_286}:
        raise ValueError("Staged dataset 42/490/5,286 counts changed")
    output_records = manifest.get("output_files")
    if type(output_records) is not dict or set(output_records) != _DATASET_OUTPUT_PATHS:
        raise ValueError("Staged dataset output manifest inventory changed")
    for relative_path in sorted(_DATASET_OUTPUT_PATHS):
        record = output_records[relative_path]
        expected_hash = output_sha256[relative_path]
        if (
            type(record) is not dict
            or type(record.get("bytes")) is not int
            or record["bytes"] < 1
            or type(record.get("records")) is not int
            or record["records"] < 1
            or record.get("sha256") != expected_hash
        ):
            raise ValueError(
                f"Staged dataset manifest record changed for {relative_path}"
            )
        path = dataset_dir / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Staged dataset output is missing or unsafe: {relative_path}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Staged dataset output size changed for {relative_path}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Staged dataset output hash changed for {relative_path}: "
                f"actual={actual_hash}, expected={expected_hash}"
            )
    return manifest


def _validate_fold_roles(manifest: Mapping[str, Any]) -> None:
    folds = manifest.get("folds")
    rotations = manifest.get("rotations")
    if type(folds) is not list or len(folds) != 5:
        raise ValueError("Frozen fold manifest must contain exactly five folds")
    fold_by_id: dict[int, dict[str, Any]] = {}
    all_case_ids: set[str] = set()
    expected_capacities = (9, 9, 8, 8, 8)
    for position, record in enumerate(folds):
        if type(record) is not dict or set(record) != _FOLD_RECORD_KEYS:
            raise ValueError(f"Frozen fold record {position} schema changed")
        case_ids = record["case_ids"]
        if (
            record["fold_id"] != position
            or type(record["fold_id"]) is not int
            or record["capacity"] != expected_capacities[position]
            or record["num_cases"] != record["capacity"]
            or type(case_ids) is not list
            or len(case_ids) != record["capacity"]
            or any(type(case_id) is not str or not case_id for case_id in case_ids)
            or case_ids != sorted(case_ids, key=int)
            or len(case_ids) != len(set(case_ids))
            or all_case_ids.intersection(case_ids)
            or record["queries"] != 98
            or type(record["passages"]) is not int
            or record["passages"] < 1
        ):
            raise ValueError(f"Frozen fold record {position} inventory changed")
        all_case_ids.update(case_ids)
        fold_by_id[position] = record
    if len(all_case_ids) != 42 or sum(record["passages"] for record in folds) != 5_286:
        raise ValueError("Frozen folds do not cover exactly 42 cases and 5,286 passages")

    if type(rotations) is not list or len(rotations) != 5:
        raise ValueError("Frozen fold manifest must contain exactly five rotations")
    for outer_fold, rotation in enumerate(rotations):
        if type(rotation) is not dict or set(rotation) != {
            "outer_fold",
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"Frozen rotation {outer_fold} schema changed")
        if rotation["outer_fold"] != outer_fold or type(rotation["outer_fold"]) is not int:
            raise ValueError(f"Frozen rotation {outer_fold} identity changed")
        expected_fold_ids = {
            "test": [outer_fold],
            "validation": [(outer_fold + 1) % 5],
            "train": [
                fold_id
                for fold_id in range(5)
                if fold_id not in {outer_fold, (outer_fold + 1) % 5}
            ],
        }
        role_case_sets: list[set[str]] = []
        for role in ("train", "validation", "test"):
            value = rotation[role]
            if type(value) is not dict or set(value) != _ROLE_KEYS:
                raise ValueError(f"Frozen rotation {outer_fold}.{role} schema changed")
            fold_ids = value["fold_ids"]
            expected_ids = expected_fold_ids[role]
            expected_cases = sorted(
                [
                    case_id
                    for fold_id in expected_ids
                    for case_id in fold_by_id[fold_id]["case_ids"]
                ],
                key=int,
            )
            if (
                fold_ids != expected_ids
                or value["case_ids"] != expected_cases
                or value["num_cases"] != len(expected_cases)
                or value["queries"] != sum(fold_by_id[fold_id]["queries"] for fold_id in expected_ids)
                or value["passages"]
                != sum(fold_by_id[fold_id]["passages"] for fold_id in expected_ids)
            ):
                raise ValueError(f"Frozen rotation {outer_fold}.{role} inventory changed")
            role_case_sets.append(set(expected_cases))
        if any(
            role_case_sets[left] & role_case_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ) or set().union(*role_case_sets) != all_case_ids:
            raise ValueError(f"Frozen rotation {outer_fold} roles overlap or are incomplete")


def validate_staged_dataset_and_fold(
    *,
    dataset_dir: Path,
    fold_manifest_path: Path,
    expected_dataset_manifest_sha256: str,
    expected_fold_manifest_sha256: str,
    expected_dataset_manifest_logical_path: str | None = None,
) -> dict[str, Any]:
    """Validate exact study bytes without binding them to repository locations."""

    dataset_dir = Path(dataset_dir)
    fold_manifest_path = Path(fold_manifest_path)
    expected_dataset_hash = _require_sha256(
        expected_dataset_manifest_sha256,
        name="expected_dataset_manifest_sha256",
    )
    expected_fold_hash = _require_sha256(
        expected_fold_manifest_sha256,
        name="expected_fold_manifest_sha256",
    )
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError(f"Staged dataset must be a real directory: {dataset_dir}")
    if fold_manifest_path.is_symlink() or not fold_manifest_path.is_file():
        raise ValueError(f"Fold manifest must be a regular file: {fold_manifest_path}")
    actual_fold_hash = _sha256_file(fold_manifest_path)
    if actual_fold_hash != expected_fold_hash:
        raise ValueError(
            "Frozen fold-manifest SHA-256 changed: "
            f"actual={actual_fold_hash}, expected={expected_fold_hash}"
        )
    manifest = _load_pretty_json_object(fold_manifest_path, name="Frozen fold manifest")
    if (
        set(manifest) != _FOLD_MANIFEST_KEYS
        or manifest["schema_version"] != 1
        or type(manifest["schema_version"]) is not int
        or manifest["manifest_type"] != "retrieval_case_folds"
        or manifest["totals"] != {"cases": 42, "queries": 490, "passages": 5_286}
    ):
        raise ValueError("Frozen fold manifest identity/schema changed")
    dataset_record = manifest["dataset"]
    if type(dataset_record) is not dict or set(dataset_record) != {
        "dataset_schema_version",
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "output_sha256",
    }:
        raise ValueError("Frozen fold dataset record schema changed")
    logical_path = _validate_logical_path(
        dataset_record["dataset_manifest_path"],
        name="fold.dataset.dataset_manifest_path",
    )
    if expected_dataset_manifest_logical_path is not None and logical_path != (
        expected_dataset_manifest_logical_path
    ):
        raise ValueError("Frozen fold dataset logical path changed")
    if (
        dataset_record["dataset_schema_version"] != 2
        or type(dataset_record["dataset_schema_version"]) is not int
        or dataset_record["dataset_manifest_sha256"] != expected_dataset_hash
    ):
        raise ValueError("Frozen fold dataset identity changed")
    output_sha256 = dataset_record["output_sha256"]
    if type(output_sha256) is not dict or set(output_sha256) != _DATASET_OUTPUT_PATHS:
        raise ValueError("Frozen fold dataset output inventory changed")
    for relative_path, digest in output_sha256.items():
        _validate_logical_path(relative_path, name="fold.dataset.output_sha256 path")
        _require_sha256(digest, name=f"fold.dataset.output_sha256[{relative_path}]")

    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    if _sha256_file(dataset_manifest_path) != expected_dataset_hash:
        raise ValueError("Staged dataset-manifest SHA-256 changed")
    _validate_exact_dataset_inventory(
        dataset_dir,
        output_sha256=output_sha256,
    )
    _validate_fold_roles(manifest)
    return manifest
