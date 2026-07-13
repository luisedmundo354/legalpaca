"""Fail-loud AWS control plane for S3-only retrieval-fold evaluation.

This module is intentionally narrower than a general SageMaker coordinator.
It stages the four frozen static evaluation assets once, copies twelve exact
training-output object versions into one immutable fold prefix, and launches
the explicitly defined network-isolated Phase-1 Processing gate.  Every write
is one-shot.  A partial local state tree or destination prefix is permanently
ambiguous and has no resume, retry, reconciliation, adoption, overwrite, or
deletion path.
"""

from __future__ import annotations

import base64
import binascii
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ...retriever.baseline_artifacts import (
    E5_MODEL_ID,
    E5_REVISION,
    E5_SNAPSHOT_MANIFEST_SHA256,
    E5_SNAPSHOT_TREE_SHA256,
    FIXED_BASELINE_CONFIG_SHA256,
    FixedBaseArtifactExpectation,
    validate_fixed_base_artifact,
    validate_snapshot,
)
from . import aws
from . import config as strict_config
from . import controlled_supervisor
from . import training_artifacts
from . import training_aws


MODERNBERT_DIR = Path(__file__).resolve().parents[2]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))


STATIC_STAGING_PROTOCOL = "retrieval_cv_static_evaluation_staging_v1"
STATIC_STAGING_INTENT_PROTOCOL = "retrieval_cv_static_evaluation_staging_intent_v1"
FOLD_ARCHIVE_COPY_PROTOCOL = "retrieval_cv_fold_archive_copy_v1"
FOLD_ARCHIVE_COPY_SET_PROTOCOL = "retrieval_cv_fold_archive_copy_set_v1"
FOLD_ARCHIVE_COPY_INTENT_PROTOCOL = "retrieval_cv_fold_archive_copy_intent_v1"
FOLD_ARCHIVE_MANIFEST_PUBLICATION_PROTOCOL = (
    "retrieval_cv_fold_archive_manifest_publication_v1"
)
FOLD_INVENTORY_PREFLIGHT_PROTOCOL = "retrieval_cv_fold_inventory_preflight_v1"
FOLD_INVENTORY_SUBMISSION_PROTOCOL = "retrieval_cv_fold_inventory_submission_v1"
FOLD_INVENTORY_TERMINAL_PROTOCOL = "retrieval_cv_fold_inventory_terminal_v1"
FOLD_INVENTORY_ACQUISITION_PROTOCOL = "retrieval_cv_fold_inventory_acquisition_v1"
FOLD_STORAGE_PROOF_PROTOCOL = "retrieval_cv_fold_storage_proof_v1"
ARCHIVE_INVENTORY_PROTOCOL = "retrieval_cv_fold_archive_inventory_v1"
PHASE1_STORAGE_PROTOCOL = "retrieval_cv_fold_bm25_storage_v1"
PHASE1_OUTPUT_PROTOCOL = "retrieval_cv_fold_inventory_output_v1"

FOLD_ARCHIVE_LOCAL_ROOT = Path("/opt/ml/processing/input/fold-archives")
STATIC_MANIFEST_NAME = "static_evaluation_inputs_manifest.json"
FOLD_ARCHIVE_MANIFEST_NAME = "fold_archive_input_manifest.json"
MAX_COPY_BYTES = 5_000_000_000
FOLD_OVERLAY_IMAGE_DIGEST = (
    "sha256:c2beb40771c4b2bdd6d1d1304e6836a31e834602a76dad280a977493f05bf7d0"
)
FOLD_OVERLAY_CONFIG_DIGEST = (
    "sha256:e35f68e64aec63836648823824338b11fa3764f11f3ed0a6ebb334eb944efe32"
)
FOLD_OVERLAY_BUILD_IDENTITY = (
    "4d83e9838a4d12f7ff1877cf4adb6d699c6ceb2051983f3fd6ef19d33052f1b0"
)
FOLD_OVERLAY_OFFLINE_SMOKE_SHA256 = (
    "2294d149ce733e80968ce82d1c15ed56fb851eb2ef84895db2e3d75e03cb896c"
)
FOLD_OVERLAY_PUBLICATION_PROTOCOL = (
    "immutable_ecr_fold_evaluation_image_publication_v1"
)
G5_12XLARGE_LOCAL_NVME_NOMINAL_BYTES = 3_800_000_000_000
G5_12XLARGE_LOCAL_FILESYSTEM_MIN_BYTES = 3_500_000_000_000

# These identities are intentionally repeated at the host orchestration
# boundary.  Importing the GPU evaluator merely to obtain scalar constants
# would make AWS preflight depend on Torch being installed on the workstation.
ARCHIVE_INPUT_MANIFEST_PROTOCOL = "retrieval_cv_fold_archive_input_manifest_v1"
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
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "469858f2f8e42d0b19e53ee71af690f722482120348a2fe9719b99104758e00d"
)
EXPECTED_EXPERIMENT_CONFIG_SHA256 = (
    "e51f4e8097f8888adda0382dd5c9377d7fd7417e0356b176f50ab37f7002aa96"
)
EXPECTED_BASELINE_CONFIG_SHA256 = FIXED_BASELINE_CONFIG_SHA256
EXPECTED_STATIC_INVENTORIES = {
    "e5-snapshot": {
        "inventory_sha256": "1782a649875725869b2a1631daf528c395a37f0f106f19a97f921c2a5f0e5d4a",
        "file_count": 6,
        "total_size": 438_899_505,
    },
    "e5-pack": {
        "inventory_sha256": "95bc831c3250a3b4b4c8cfa2debd4fca06528d05c127a9a38198259c8a0a4e7e",
        "file_count": 2,
        "total_size": 1_472_765,
    },
    "fixed-base": {
        "inventory_sha256": "031370aace9d48cb818228efd71bdab9b587a93fc7f6fd4285458e6c28cad2a7",
        "file_count": 8,
        "total_size": 301_690_280,
    },
    "control": {
        "inventory_sha256": "cb060bd79da17f085b595e5e0414040f7e333ef9c9a9f031de73d6da68d0c4df",
        "file_count": 4,
        "total_size": 46_055,
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BASE64_SHA256 = re.compile(r"[A-Za-z0-9+/]{43}=\Z")
_ACCOUNT = re.compile(r"[0-9]{12}\Z")
_ATTEMPT = re.compile(r"a[1-9][0-9]*\Z")
_ETAG = re.compile(r'"[0-9a-f]{32}(?:-[1-9][0-9]*)?"\Z')
_KMS_ARN = re.compile(
    r"arn:aws:kms:(?P<region>[a-z]{2}(?:-gov)?-[a-z]+-[0-9]):"
    r"(?P<account>[0-9]{12}):key/(?P<key>[A-Za-z0-9-]+)\Z"
)
_JOB_NAME = re.compile(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}\Z")

_STATIC_RECEIPT_KEYS = {
    "schema_version",
    "protocol",
    "completed_fold_evidence_sha256",
    "training_plan_sha256",
    "destination_prefix",
    "assets",
    "manifest_object",
    "receipt_sha256",
}
_STATIC_ASSET_KEYS = {
    "name",
    "s3_prefix",
    "source_identity",
    "inventory_sha256",
    "file_count",
    "total_size",
    "files",
}
_STATIC_FILE_KEYS = {
    "path",
    "size",
    "sha256",
    "bucket",
    "key",
    "version_id",
    "etag",
    "sse",
}
_ARCHIVE_COPY_RECEIPT_KEYS = {
    "schema_version",
    "protocol",
    "completed_fold_evidence_sha256",
    "training_plan_sha256",
    "destination_prefix",
    "copy_set_receipt",
    "fold_archive_input_manifest",
    "manifest_object",
    "receipt_sha256",
}
_COPY_SET_KEYS = {
    "schema_version",
    "protocol",
    "completed_fold_evidence_sha256",
    "training_plan_sha256",
    "destination_prefix",
    "systems",
    "receipt_sha256",
}
_COPY_SYSTEM_KEYS = {
    "ordinal",
    "system_id",
    "run_id",
    "job_name",
    "cell",
    "source_object",
    "destination_object",
    "terminal_receipt_sha256",
    "request_receipt_sha256",
    "copy_intent_sha256",
}
_ARCHIVE_OBJECT_KEYS = {
    "bucket",
    "key",
    "version_id",
    "size",
    "etag",
    "checksum",
    "encryption",
}
_CHECKSUM_KEYS = {"algorithm", "type", "value"}
_ENCRYPTION_KEYS = {"algorithm", "kms_key_id", "bucket_key_enabled"}
_PHASE1_PREFLIGHT_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "account_id",
    "region",
    "caller_arn",
    "job_name",
    "output_prefix",
    "image_uri",
    "image_publication_sha256",
    "completed_fold_evidence_sha256",
    "archive_copy_receipt_sha256",
    "static_staging_receipt_sha256",
    "training_staging_receipt_sha256",
    "archive_verification",
    "static_verification",
    "request",
    "request_sha256",
    "sdk_versions",
    "processing_quota",
    "receipt_sha256",
}
_PHASE1_SUBMISSION_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "job_name",
    "job_arn",
    "preflight_receipt_sha256",
    "request_sha256",
    "receipt_sha256",
}
_PHASE1_TERMINAL_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "job_name",
    "job_arn",
    "preflight_receipt_sha256",
    "submission_receipt_sha256",
    "request_sha256",
    "status",
    "failure_reason",
    "processing_start_time",
    "processing_end_time",
    "processing_time_microseconds",
    "exit_message",
    "receipt_sha256",
}
_PHASE1_ACQUISITION_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "terminal_receipt_sha256",
    "archive_copy_receipt_sha256",
    "output_prefix",
    "remote_objects",
    "files",
    "archive_inventory_receipt_sha256",
    "bm25_storage_receipt_sha256",
    "artifact_manifest_sha256",
    "receipt_sha256",
}
_STORAGE_PROOF_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "archive_copy_receipt_sha256",
    "static_staging_receipt_sha256",
    "inventory_acquisition_receipt_sha256",
    "volume_size_gb",
    "filesystem_capacity_bytes",
    "filesystem_available_bytes",
    "components",
    "required_additional_bytes",
    "remaining_bytes",
    "fits",
    "receipt_sha256",
}


def _canonical_bytes(value: object) -> bytes:
    return aws.canonical_json_bytes(value)


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(_canonical_bytes(value))


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise ValueError("Receipt payload already contains receipt_sha256")
    result = copy.deepcopy(dict(payload))
    result["receipt_sha256"] = _document_sha256(payload)
    return result


def _exact_object(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one non-empty exact string")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _validate_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    actual = _exact_sha256(value["receipt_sha256"], name=f"{name}.receipt_sha256")
    payload = {
        key: copy.deepcopy(nested)
        for key, nested in value.items()
        if key != "receipt_sha256"
    }
    if actual != _document_sha256(payload):
        raise ValueError(f"{name} self-hash changed")


def _normalized_prefix(value: str, *, name: str) -> str:
    value = _exact_string(value, name=name)
    path = PurePosixPath(value)
    if (
        value.startswith("/")
        or not value.endswith("/")
        or "//" in value
        or path.as_posix() + "/" != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith("arr-retrieval-cv/")
    ):
        raise ValueError(f"{name} must be one normalized ARR prefix ending in slash")
    return value


def _prefixes_overlap(first: str, second: str) -> bool:
    return first.startswith(second) or second.startswith(first)


def _canonical_absolute(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be one pathlib.Path")
    text = path.as_posix()
    if (
        not path.is_absolute()
        or text.startswith("//")
        or PurePosixPath(text).as_posix() != text
        or ".." in path.parts
        or path.resolve(strict=False) != path
    ):
        raise ValueError(f"{name} must be one canonical absolute path")
    return path


def _real_directory(path: Path, *, name: str) -> Path:
    path = _canonical_absolute(path, name=name)
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{name} must be one real canonical directory")
    return path


def _publish_json_absent(path: Path, value: object) -> None:
    path = _canonical_absolute(path, name="evidence output")
    parent = _real_directory(path.parent, name="evidence output parent")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    payload = strict_config.canonical_json_bytes(value)
    incomplete = path.with_name(f".{path.name}.incomplete")
    if incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError(f"Refusing stale incomplete evidence: {incomplete}")
    published = False
    try:
        descriptor = os.open(
            incomplete,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            position = 0
            while position < len(payload):
                written = os.write(descriptor, payload[position:])
                if written < 1:
                    raise RuntimeError("Evidence write made no progress")
                position += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(incomplete, path, follow_symlinks=False)
        published = True
        incomplete.unlink()
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        loaded, _ = strict_config.load_canonical_json_object(path)
        if strict_config.canonical_json_bytes(loaded) != payload:
            raise RuntimeError("Published evidence changed on canonical readback")
    except BaseException:
        if published and (path.exists() or path.is_symlink()):
            path.unlink()
        if incomplete.exists() or incomplete.is_symlink():
            incomplete.unlink()
        raise


def _rename_no_replace(parent_descriptor: int, source: str, target: str) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(target),
        1,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"Refusing existing one-shot state: {target}")
        raise OSError(number, f"One-shot state publication failed: {source} -> {target}")


def _create_state_directory(path: Path, *, protocol: str) -> Path:
    path = _canonical_absolute(path, name="one-shot state directory")
    parent = _real_directory(path.parent, name="one-shot state parent")
    incomplete = path.with_name(f".{path.name}.incomplete")
    if path.exists() or path.is_symlink() or incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError("One-shot state or sibling incomplete directory already exists")
    parent_descriptor = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        os.mkdir(incomplete.name, mode=0o700, dir_fd=parent_descriptor)
        _rename_no_replace(parent_descriptor, incomplete.name, path.name)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    state = _real_directory(path, name="one-shot state directory")
    _publish_json_absent(
        state / "state.json",
        _seal({"schema_version": 1, "protocol": protocol}),
    )
    return state


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"Static input is not singly-linked regular: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_nlink,
        ):
            raise RuntimeError(f"Static input changed while hashed: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _directory_inventory(root: Path, *, name: str) -> list[dict[str, Any]]:
    root = _real_directory(root, name=name)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{name} contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{name} contains a special or hard-linked entry: {relative}")
        if metadata.st_size < 1:
            raise ValueError(f"{name} contains an empty file: {relative}")
        records.append(
            {
                "path": relative,
                "size": metadata.st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"{name} has no files")
    return records


def _validate_static_sources(
    *,
    e5_snapshot_dir: Path,
    e5_snapshot_manifest_path: Path,
    e5_pack_dir: Path,
    fixed_base_dir: Path,
) -> list[dict[str, Any]]:
    snapshot = validate_snapshot(
        snapshot_dir=e5_snapshot_dir,
        manifest_path=e5_snapshot_manifest_path,
        expected_manifest_sha256=E5_SNAPSHOT_MANIFEST_SHA256,
        expected_model_id=E5_MODEL_ID,
        expected_revision=E5_REVISION,
        expected_tree_sha256=E5_SNAPSHOT_TREE_SHA256,
    )
    e5_records = _directory_inventory(e5_snapshot_dir, name="E5 snapshot")
    if [(row["path"], row["size"], row["sha256"]) for row in e5_records] != list(
        snapshot.files
    ):
        raise RuntimeError("E5 snapshot validator and staged inventory disagree")

    pack_records = _directory_inventory(e5_pack_dir, name="E5 focus pack")
    if [record["path"] for record in pack_records] != [
        "manifest.json",
        "packed_queries.jsonl",
    ]:
        raise ValueError("E5 focus-pack file inventory changed")
    manifest_record = pack_records[0]
    if manifest_record["sha256"] != EXPECTED_E5_PACK_MANIFEST_SHA256:
        raise ValueError("E5 focus-pack manifest hash changed")
    pack_manifest, pack_manifest_sha256 = strict_config.load_canonical_json_object(
        Path(e5_pack_dir) / "manifest.json"
    )
    if (
        pack_manifest_sha256 != EXPECTED_E5_PACK_MANIFEST_SHA256
        or pack_manifest.get("packed_query_inventory_sha256")
        != EXPECTED_E5_PACK_INVENTORY_SHA256
        or type(pack_manifest.get("packed_queries_file")) is not dict
        or pack_manifest["packed_queries_file"].get("path")
        != pack_records[1]["path"]
        or pack_manifest["packed_queries_file"].get("sha256")
        != pack_records[1]["sha256"]
        or pack_manifest["packed_queries_file"].get("size")
        != pack_records[1]["size"]
        or pack_manifest["packed_queries_file"].get("records") != 490
    ):
        raise ValueError("E5 focus-pack manifest content changed")

    fixed = validate_fixed_base_artifact(
        fixed_base_dir,
        expectation=FixedBaseArtifactExpectation(
            artifact_manifest_sha256=EXPECTED_FIXED_BASE_ARTIFACT_MANIFEST_SHA256,
            baseline_config_sha256=FIXED_BASELINE_CONFIG_SHA256,
        ),
    )
    if (
        fixed.model_sha256 != EXPECTED_FIXED_BASE_MODEL_SHA256
        or fixed.state_key_sha256 != EXPECTED_FIXED_BASE_STATE_KEYS_SHA256
        or fixed.new_embedding_rows_sha256 != EXPECTED_FIXED_BASE_NEW_ROWS_SHA256
    ):
        raise ValueError("Fixed-base scientific identity changed")
    fixed_records = _directory_inventory(fixed_base_dir, name="fixed-base artifact")

    control_root = _real_directory(
        Path(e5_snapshot_manifest_path).parent,
        name="frozen evaluation controls",
    )
    control_names = (
        "e5_snapshot.json",
        "evaluation_baselines.json",
        "experiment.json",
        "folds.json",
    )
    control_records = []
    expected_control_hashes = {
        "e5_snapshot.json": E5_SNAPSHOT_MANIFEST_SHA256,
        "evaluation_baselines.json": EXPECTED_BASELINE_CONFIG_SHA256,
        "experiment.json": EXPECTED_EXPERIMENT_CONFIG_SHA256,
        "folds.json": EXPECTED_FOLD_MANIFEST_SHA256,
    }
    for filename in control_names:
        path = control_root / filename
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Frozen evaluation control is not singly-linked: {filename}")
        digest = _sha256_file(path)
        if digest != expected_control_hashes[filename]:
            raise ValueError(f"Frozen evaluation control hash changed: {filename}")
        control_records.append(
            {"path": filename, "size": metadata.st_size, "sha256": digest}
        )

    assets = [
        {
            "name": "e5-snapshot",
            "root": _real_directory(e5_snapshot_dir, name="E5 snapshot"),
            "source_identity": {
                "manifest_sha256": snapshot.manifest_sha256,
                "model_id": snapshot.model_id,
                "revision": snapshot.revision,
                "tree_sha256": snapshot.tree_sha256,
            },
            "files": e5_records,
        },
        {
            "name": "e5-pack",
            "root": _real_directory(e5_pack_dir, name="E5 focus pack"),
            "source_identity": {
                "manifest_sha256": EXPECTED_E5_PACK_MANIFEST_SHA256,
                "packed_query_inventory_sha256": EXPECTED_E5_PACK_INVENTORY_SHA256,
            },
            "files": pack_records,
        },
        {
            "name": "fixed-base",
            "root": _real_directory(fixed_base_dir, name="fixed-base artifact"),
            "source_identity": {
                "artifact_manifest_sha256": fixed.manifest_sha256,
                "model_sha256": fixed.model_sha256,
                "new_embedding_rows_sha256": fixed.new_embedding_rows_sha256,
                "state_key_sha256": fixed.state_key_sha256,
            },
            "files": fixed_records,
        },
        {
            "name": "control",
            "root": control_root,
            "source_identity": {
                "baseline_config_sha256": EXPECTED_BASELINE_CONFIG_SHA256,
                "e5_snapshot_manifest_sha256": E5_SNAPSHOT_MANIFEST_SHA256,
                "experiment_config_sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
                "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
            },
            "files": control_records,
        },
    ]
    for asset in assets:
        expected = EXPECTED_STATIC_INVENTORIES[asset["name"]]
        if (
            _document_sha256(asset["files"]) != expected["inventory_sha256"]
            or len(asset["files"]) != expected["file_count"]
            or sum(record["size"] for record in asset["files"])
            != expected["total_size"]
        ):
            raise ValueError(f"Frozen static inventory changed: {asset['name']}")
    return assets


def _put_json_object_once(
    s3: object,
    *,
    payload: Mapping[str, Any],
    bucket: str,
    key: str,
    expected_bucket_owner: str,
) -> dict[str, Any]:
    raw = _canonical_bytes(payload)
    digest = hashlib.sha256(raw).digest()
    checksum = base64.b64encode(digest).decode("ascii")
    md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    response = s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentLength=len(raw),
        ContentType="application/json",
        ChecksumAlgorithm="SHA256",
        ChecksumSHA256=checksum,
        ExpectedBucketOwner=expected_bucket_owner,
        IfNoneMatch="*",
        Metadata={"sha256": digest.hex()},
        ServerSideEncryption="AES256",
    )
    version_id = response.get("VersionId")
    etag = f'"{md5}"'
    if (
        type(version_id) is not str
        or not version_id
        or response.get("ChecksumSHA256") != checksum
        or response.get("ETag") != etag
        or response.get("ServerSideEncryption") != "AES256"
    ):
        raise RuntimeError("Static manifest publication returned incomplete identity")
    head = s3.head_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=expected_bucket_owner,
    )
    if (
        head.get("ContentLength") != len(raw)
        or head.get("ContentType") != "application/json"
        or head.get("ChecksumSHA256") != checksum
        or head.get("ETag") != etag
        or head.get("VersionId") != version_id
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("Metadata") != {"sha256": digest.hex()}
    ):
        raise RuntimeError("Static manifest metadata changed on versioned readback")
    body = s3.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=expected_bucket_owner,
    )["Body"]
    observed = body.read(len(raw) + 1)
    if type(observed) is not bytes or observed != raw or body.read(1) != b"":
        raise RuntimeError("Static manifest bytes changed on versioned readback")
    return {
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "size": len(raw),
        "sha256": digest.hex(),
        "etag": etag,
        "sse": "AES256",
    }


def _static_file_record(
    *,
    source: Mapping[str, Any],
    staged: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        staged.get("schema_version") != 1
        or staged.get("size") != source["size"]
        or staged.get("sha256") != source["sha256"]
        or staged.get("sse") != "AES256"
    ):
        raise RuntimeError("Static staged-object receipt differs from its source")
    return {
        "path": source["path"],
        "size": source["size"],
        "sha256": source["sha256"],
        "bucket": staged["bucket"],
        "key": staged["key"],
        "version_id": staged["version_id"],
        "etag": staged["etag"],
        "sse": staged["sse"],
    }


def validate_static_evaluation_staging_receipt(
    value: object,
    *,
    completed_fold_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    receipt = _exact_object(
        value,
        _STATIC_RECEIPT_KEYS,
        name="static evaluation staging receipt",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != STATIC_STAGING_PROTOCOL
        or receipt["completed_fold_evidence_sha256"] != _document_sha256(completed)
        or receipt["training_plan_sha256"] != completed["training_plan_sha256"]
    ):
        raise ValueError("Static staging receipt evidence binding changed")
    prefix = _normalized_prefix(
        receipt["destination_prefix"], name="static destination prefix"
    )
    plan = completed["training_plan"]
    bucket = plan["infrastructure"]["artifact_bucket"]
    assets = receipt["assets"]
    expected_names = ["e5-snapshot", "e5-pack", "fixed-base", "control"]
    if type(assets) is not list or len(assets) != 4:
        raise ValueError("Static staging receipt requires exactly four assets")
    normalized_assets: list[dict[str, Any]] = []
    all_coordinates: set[tuple[str, str, str]] = set()
    for index, (raw, expected_name) in enumerate(zip(assets, expected_names)):
        asset = _exact_object(
            raw,
            _STATIC_ASSET_KEYS,
            name=f"static assets[{index}]",
        )
        s3_prefix = f"{prefix}{expected_name}/"
        files = asset["files"]
        if (
            asset["name"] != expected_name
            or asset["s3_prefix"] != s3_prefix
            or type(files) is not list
            or not files
            or type(asset["file_count"]) is not int
            or asset["file_count"] != len(files)
            or asset["inventory_sha256"]
            != EXPECTED_STATIC_INVENTORIES[expected_name]["inventory_sha256"]
            or asset["file_count"]
            != EXPECTED_STATIC_INVENTORIES[expected_name]["file_count"]
            or asset["total_size"]
            != EXPECTED_STATIC_INVENTORIES[expected_name]["total_size"]
        ):
            raise ValueError("Static asset identity or file count changed")
        source_records: list[dict[str, Any]] = []
        normalized_files: list[dict[str, Any]] = []
        total_size = 0
        previous_path = ""
        for position, raw_file in enumerate(files):
            record = _exact_object(
                raw_file,
                _STATIC_FILE_KEYS,
                name=f"static assets[{index}].files[{position}]",
            )
            path = _exact_string(record["path"], name="static relative path")
            pure = PurePosixPath(path)
            if (
                pure.is_absolute()
                or pure.as_posix() != path
                or any(part in {"", ".", ".."} for part in pure.parts)
                or path <= previous_path
                or record["bucket"] != bucket
                or record["key"] != s3_prefix + path
                or type(record["size"]) is not int
                or record["size"] < 1
                or record["sse"] != "AES256"
                or _ETAG.fullmatch(_exact_string(record["etag"], name="static ETag"))
                is None
            ):
                raise ValueError("Static staged-file identity changed")
            _exact_sha256(record["sha256"], name="static file SHA-256")
            _exact_string(record["version_id"], name="static file VersionId")
            coordinate = (record["bucket"], record["key"], record["version_id"])
            if coordinate in all_coordinates:
                raise ValueError("Static staging aliases an object version")
            all_coordinates.add(coordinate)
            source_records.append(
                {
                    "path": path,
                    "size": record["size"],
                    "sha256": record["sha256"],
                }
            )
            total_size += record["size"]
            previous_path = path
            normalized_files.append(copy.deepcopy(record))
        if (
            asset["inventory_sha256"] != _document_sha256(source_records)
            or asset["total_size"] != total_size
        ):
            raise ValueError("Static asset inventory summary changed")
        if expected_name == "e5-snapshot" and asset["source_identity"] != {
            "manifest_sha256": E5_SNAPSHOT_MANIFEST_SHA256,
            "model_id": E5_MODEL_ID,
            "revision": E5_REVISION,
            "tree_sha256": E5_SNAPSHOT_TREE_SHA256,
        }:
            raise ValueError("E5 snapshot staged identity changed")
        if expected_name == "e5-pack" and asset["source_identity"] != {
            "manifest_sha256": EXPECTED_E5_PACK_MANIFEST_SHA256,
            "packed_query_inventory_sha256": EXPECTED_E5_PACK_INVENTORY_SHA256,
        }:
            raise ValueError("E5 focus-pack staged identity changed")
        if expected_name == "fixed-base" and asset["source_identity"] != {
            "artifact_manifest_sha256": EXPECTED_FIXED_BASE_ARTIFACT_MANIFEST_SHA256,
            "model_sha256": EXPECTED_FIXED_BASE_MODEL_SHA256,
            "new_embedding_rows_sha256": EXPECTED_FIXED_BASE_NEW_ROWS_SHA256,
            "state_key_sha256": EXPECTED_FIXED_BASE_STATE_KEYS_SHA256,
        }:
            raise ValueError("Fixed-base staged identity changed")
        if expected_name == "control" and asset["source_identity"] != {
            "baseline_config_sha256": EXPECTED_BASELINE_CONFIG_SHA256,
            "e5_snapshot_manifest_sha256": E5_SNAPSHOT_MANIFEST_SHA256,
            "experiment_config_sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
            "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        }:
            raise ValueError("Frozen evaluation-control staged identity changed")
        normalized = copy.deepcopy(asset)
        normalized["files"] = normalized_files
        normalized_assets.append(normalized)
    manifest = _exact_object(
        receipt["manifest_object"],
        _STATIC_FILE_KEYS - {"path"},
        name="static staging manifest object",
    )
    manifest_key = f"{prefix}{STATIC_MANIFEST_NAME}"
    if (
        manifest["bucket"] != bucket
        or manifest["key"] != manifest_key
        or manifest["sse"] != "AES256"
        or type(manifest["size"]) is not int
        or manifest["size"] < 1
        or _ETAG.fullmatch(_exact_string(manifest["etag"], name="manifest ETag"))
        is None
    ):
        raise ValueError("Static staging manifest-object identity changed")
    _exact_sha256(manifest["sha256"], name="static manifest SHA-256")
    _exact_string(manifest["version_id"], name="static manifest VersionId")
    _validate_self_hash(receipt, name="static evaluation staging receipt")
    normalized = copy.deepcopy(receipt)
    normalized["assets"] = normalized_assets
    normalized["manifest_object"] = copy.deepcopy(manifest)
    return normalized


def stage_static_evaluation_inputs_once(
    clients: aws.AwsClients,
    *,
    completed_fold_evidence: Mapping[str, Any],
    e5_snapshot_dir: Path,
    e5_snapshot_manifest_path: Path,
    e5_pack_dir: Path,
    fixed_base_dir: Path,
    destination_prefix: str,
    state_dir: Path,
) -> dict[str, Any]:
    """Publish the exact E5, focus-pack, and fixed-base trees one time."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    aws.validate_aws_sdk_versions()
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    plan = completed["training_plan"]
    infrastructure = plan["infrastructure"]
    bucket = infrastructure["artifact_bucket"]
    account = infrastructure["account_id"]
    region = infrastructure["region"]
    prefix = _normalized_prefix(destination_prefix, name="static destination prefix")
    sources = _validate_static_sources(
        e5_snapshot_dir=e5_snapshot_dir,
        e5_snapshot_manifest_path=e5_snapshot_manifest_path,
        e5_pack_dir=e5_pack_dir,
        fixed_base_dir=fixed_base_dir,
    )
    aws.validate_artifact_bucket(clients.s3, bucket=bucket, region=region)
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=bucket,
        prefix=prefix,
        expected_bucket_owner=account,
    )
    state = _create_state_directory(
        state_dir,
        protocol=STATIC_STAGING_INTENT_PROTOCOL,
    )
    intent = _seal(
        {
            "schema_version": 1,
            "protocol": STATIC_STAGING_INTENT_PROTOCOL,
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "training_plan_sha256": completed["training_plan_sha256"],
            "destination_prefix": prefix,
            "assets": [
                {
                    "name": source["name"],
                    "source_identity": source["source_identity"],
                    "inventory_sha256": _document_sha256(source["files"]),
                    "file_count": len(source["files"]),
                    "total_size": sum(record["size"] for record in source["files"]),
                }
                for source in sources
            ],
        }
    )
    _publish_json_absent(state / "intent.json", intent)
    staged_assets: list[dict[str, Any]] = []
    ordinal = 0
    for source in sources:
        asset_prefix = f"{prefix}{source['name']}/"
        staged_files: list[dict[str, Any]] = []
        for record in source["files"]:
            object_receipt = aws.stage_file_once(
                clients.s3,
                source_path=source["root"] / record["path"],
                bucket=bucket,
                key=asset_prefix + record["path"],
                expected_bucket_owner=account,
            )
            staged_record = _static_file_record(source=record, staged=object_receipt)
            _publish_json_absent(
                state / f"object-{ordinal:03d}.json",
                _seal(
                    {
                        "schema_version": 1,
                        "protocol": STATIC_STAGING_PROTOCOL,
                        "ordinal": ordinal,
                        "asset": source["name"],
                        "object": staged_record,
                    }
                ),
            )
            staged_files.append(staged_record)
            ordinal += 1
        staged_assets.append(
            {
                "name": source["name"],
                "s3_prefix": asset_prefix,
                "source_identity": copy.deepcopy(source["source_identity"]),
                "inventory_sha256": _document_sha256(source["files"]),
                "file_count": len(source["files"]),
                "total_size": sum(record["size"] for record in source["files"]),
                "files": staged_files,
            }
        )
    manifest_payload = {
        "schema_version": 1,
        "protocol": STATIC_STAGING_PROTOCOL,
        "completed_fold_evidence_sha256": _document_sha256(completed),
        "training_plan_sha256": completed["training_plan_sha256"],
        "destination_prefix": prefix,
        "assets": copy.deepcopy(staged_assets),
    }
    _publish_json_absent(state / "manifest-intent.json", _seal(manifest_payload))
    manifest_object = _put_json_object_once(
        clients.s3,
        payload=manifest_payload,
        bucket=bucket,
        key=f"{prefix}{STATIC_MANIFEST_NAME}",
        expected_bucket_owner=account,
    )
    _require_exact_copy_history(
        _list_prefix_history(
            clients.s3,
            bucket=bucket,
            prefix=prefix,
            expected_bucket_owner=account,
        ),
        expected_objects=[
            *(record for asset in staged_assets for record in asset["files"]),
            manifest_object,
        ],
    )
    receipt = _seal(
        {
            **manifest_payload,
            "manifest_object": manifest_object,
        }
    )
    validated = validate_static_evaluation_staging_receipt(
        receipt,
        completed_fold_evidence=completed,
    )
    _publish_json_absent(state / "receipt.json", validated)
    return validated


def _archive_system_id(cell: Mapping[str, Any]) -> str:
    return f"{cell['query_view']}_{cell['sampler']}_seed{cell['experiment_seed']}"


def _archive_source_record(remote: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bucket": remote["bucket"],
        "key": remote["key"],
        "version_id": remote["version_id"],
        "size": remote["size"],
        "etag": remote["etag"],
        "checksum": copy.deepcopy(remote["checksum"]),
        "encryption": copy.deepcopy(remote["encryption"]),
    }


def _list_prefix_history(
    s3: object,
    *,
    bucket: str,
    prefix: str,
    expected_bucket_owner: str,
) -> dict[str, list[dict[str, Any]]]:
    versions: list[dict[str, Any]] = []
    delete_markers: list[dict[str, Any]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    seen: set[tuple[str, str]] = set()
    while True:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": expected_bucket_owner,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = s3.list_object_versions(**request)
        if (
            type(response) is not dict
            or response.get("Name") != bucket
            or response.get("Prefix") != prefix
            or response.get("MaxKeys") != 1000
            or type(response.get("IsTruncated")) is not bool
        ):
            raise RuntimeError("Version-history response identity changed")
        raw_versions = response.get("Versions", [])
        raw_deletes = response.get("DeleteMarkers", [])
        if type(raw_versions) is not list or type(raw_deletes) is not list:
            raise RuntimeError("Version-history collections are malformed")
        versions.extend(copy.deepcopy(raw_versions))
        delete_markers.extend(copy.deepcopy(raw_deletes))
        if not response["IsTruncated"]:
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        marker = (next_key, next_version)
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or marker in seen
            or marker == (key_marker, version_marker)
        ):
            raise RuntimeError("Version-history pagination did not advance")
        seen.add(marker)
        key_marker, version_marker = marker
    for collection, name in ((versions, "version"), (delete_markers, "delete marker")):
        for record in collection:
            if type(record) is not dict:
                raise RuntimeError(f"Version history contains a non-object {name}")
            key = record.get("Key")
            version_id = record.get("VersionId")
            if (
                type(key) is not str
                or not key.startswith(prefix)
                or type(version_id) is not str
                or not version_id
            ):
                raise RuntimeError(f"Version history contains an invalid {name}")
    return {
        "versions": sorted(
            versions, key=lambda item: (item["Key"], item["VersionId"])
        ),
        "delete_markers": sorted(
            delete_markers, key=lambda item: (item["Key"], item["VersionId"])
        ),
    }


def _history_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    checksum_algorithms = record.get("ChecksumAlgorithm")
    if type(checksum_algorithms) is not list:
        raise RuntimeError("Destination version omitted ChecksumAlgorithm")
    return {
        "key": record.get("Key"),
        "version_id": record.get("VersionId"),
        "is_latest": record.get("IsLatest"),
        "size": record.get("Size"),
        "etag": record.get("ETag"),
        "storage_class": record.get("StorageClass"),
        "checksum_algorithm": copy.deepcopy(checksum_algorithms),
        "checksum_type": record.get("ChecksumType"),
    }


def _require_exact_copy_history(
    history: Mapping[str, Any],
    *,
    expected_objects: Sequence[Mapping[str, Any]],
) -> None:
    if history["delete_markers"]:
        raise RuntimeError("One-shot fold prefix contains a delete marker")
    versions = history["versions"]
    if len(versions) != len(expected_objects):
        raise RuntimeError("One-shot fold prefix version count changed")
    expected_by_key = {record["key"]: record for record in expected_objects}
    if len(expected_by_key) != len(expected_objects):
        raise ValueError("Expected fold destination keys are not unique")
    observed_keys = {record.get("Key") for record in versions}
    if observed_keys != set(expected_by_key):
        raise RuntimeError("One-shot fold prefix key inventory changed")
    for raw in versions:
        identity = _history_identity(raw)
        expected = expected_by_key[identity["key"]]
        if identity != {
            "key": expected["key"],
            "version_id": expected["version_id"],
            "is_latest": True,
            "size": expected["size"],
            "etag": expected["etag"],
            "storage_class": "STANDARD",
            "checksum_algorithm": ["SHA256"],
            "checksum_type": "FULL_OBJECT",
        }:
            raise RuntimeError("One-shot fold destination version identity changed")


def _normalize_datetime(value: object, *, name: str) -> str:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be one timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _copy_archive_object_once(
    s3: object,
    *,
    source: Mapping[str, Any],
    destination_bucket: str,
    destination_key: str,
    expected_bucket_owner: str,
) -> dict[str, Any]:
    if source["size"] >= MAX_COPY_BYTES:
        raise ValueError("Source archive exceeds the one-call CopyObject boundary")
    encryption = source["encryption"]
    kms_key = encryption["kms_key_id"]
    kms_match = _KMS_ARN.fullmatch(kms_key)
    if (
        encryption != {
            "algorithm": "aws:kms",
            "kms_key_id": kms_key,
            "bucket_key_enabled": True,
        }
        or kms_match is None
        or kms_match.group("account") != expected_bucket_owner
    ):
        raise ValueError("Source archive KMS identity changed")
    response = s3.copy_object(
        Bucket=destination_bucket,
        Key=destination_key,
        CopySource={
            "Bucket": source["bucket"],
            "Key": source["key"],
            "VersionId": source["version_id"],
        },
        ExpectedBucketOwner=expected_bucket_owner,
        ExpectedSourceBucketOwner=expected_bucket_owner,
        ChecksumAlgorithm="SHA256",
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=kms_key,
        BucketKeyEnabled=True,
        MetadataDirective="COPY",
        TaggingDirective="COPY",
    )
    version_id = response.get("VersionId")
    result = response.get("CopyObjectResult")
    if type(result) is not dict:
        raise RuntimeError("CopyObject omitted CopyObjectResult")
    checksum = result.get("ChecksumSHA256")
    checksum_type = result.get("ChecksumType")
    etag = result.get("ETag")
    last_modified = _normalize_datetime(
        result.get("LastModified"), name="CopyObject LastModified"
    )
    if (
        type(version_id) is not str
        or not version_id
        or type(checksum) is not str
        or _BASE64_SHA256.fullmatch(checksum) is None
        or checksum_type != "FULL_OBJECT"
        or type(etag) is not str
        or _ETAG.fullmatch(etag) is None
        or response.get("ServerSideEncryption") != "aws:kms"
        or response.get("SSEKMSKeyId") != kms_key
        or response.get("BucketKeyEnabled") is not True
    ):
        raise RuntimeError("CopyObject returned an incomplete destination identity")
    try:
        raw_checksum = base64.b64decode(checksum, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("CopyObject SHA-256 is not canonical Base64") from error
    if len(raw_checksum) != 32 or base64.b64encode(raw_checksum).decode("ascii") != checksum:
        raise RuntimeError("CopyObject SHA-256 is not one 32-byte digest")
    head = s3.head_object(
        Bucket=destination_bucket,
        Key=destination_key,
        VersionId=version_id,
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=expected_bucket_owner,
    )
    if (
        head.get("ContentLength") != source["size"]
        or head.get("ContentType") != "application/gzip"
        or head.get("ETag") != etag
        or head.get("VersionId") != version_id
        or head.get("ChecksumSHA256") != checksum
        or head.get("ChecksumType") != "FULL_OBJECT"
        or head.get("ServerSideEncryption") != "aws:kms"
        or head.get("SSEKMSKeyId") != kms_key
        or head.get("BucketKeyEnabled") is not True
        or head.get("Metadata") != {}
        or _normalize_datetime(head.get("LastModified"), name="HeadObject LastModified")
        != last_modified
    ):
        raise RuntimeError("Copied archive metadata changed on versioned readback")
    return {
        "bucket": destination_bucket,
        "key": destination_key,
        "version_id": version_id,
        "size": source["size"],
        "etag": etag,
        "checksum": {
            "algorithm": "SHA256",
            "type": "FULL_OBJECT",
            "value": checksum,
        },
        "encryption": {
            "algorithm": "aws:kms",
            "kms_key_id": kms_key,
            "bucket_key_enabled": True,
        },
    }


def _validate_archive_object(value: object, *, name: str) -> dict[str, Any]:
    record = _exact_object(value, _ARCHIVE_OBJECT_KEYS, name=name)
    _exact_string(record["bucket"], name=f"{name}.bucket")
    key = _exact_string(record["key"], name=f"{name}.key")
    path = PurePosixPath(key)
    if key.startswith("/") or path.as_posix() != key or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name}.key is not canonical")
    _exact_string(record["version_id"], name=f"{name}.version_id")
    if type(record["size"]) is not int or not 0 < record["size"] < MAX_COPY_BYTES:
        raise ValueError(f"{name}.size left the CopyObject boundary")
    if _ETAG.fullmatch(_exact_string(record["etag"], name=f"{name}.etag")) is None:
        raise ValueError(f"{name}.etag is malformed")
    checksum = _exact_object(
        record["checksum"], _CHECKSUM_KEYS, name=f"{name}.checksum"
    )
    for field in _CHECKSUM_KEYS:
        _exact_string(checksum[field], name=f"{name}.checksum.{field}")
    encryption = _exact_object(
        record["encryption"], _ENCRYPTION_KEYS, name=f"{name}.encryption"
    )
    if (
        encryption["algorithm"] != "aws:kms"
        or type(encryption["kms_key_id"]) is not str
        or _KMS_ARN.fullmatch(encryption["kms_key_id"]) is None
        or encryption["bucket_key_enabled"] is not True
    ):
        raise ValueError(f"{name}.encryption changed")
    return copy.deepcopy(record)


def _validate_destination_sha256(record: Mapping[str, Any], *, name: str) -> None:
    checksum = record["checksum"]
    if checksum["algorithm"] != "SHA256" or checksum["type"] != "FULL_OBJECT":
        raise ValueError(f"{name} lacks full-object SHA-256")
    encoded = checksum["value"]
    if type(encoded) is not str or _BASE64_SHA256.fullmatch(encoded) is None:
        raise ValueError(f"{name} SHA-256 encoding changed")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{name} SHA-256 is not Base64") from error
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError(f"{name} SHA-256 is not canonical")


def _expected_fold_manifest(
    *,
    completed: Mapping[str, Any],
    copy_set: Mapping[str, Any],
) -> dict[str, Any]:
    systems: list[dict[str, Any]] = []
    for record in copy_set["systems"]:
        systems.append(
            {
                "ordinal": record["ordinal"],
                "system_id": record["system_id"],
                "run_id": record["run_id"],
                "job_name": record["job_name"],
                "cell": copy.deepcopy(record["cell"]),
                "archive_path": (
                    FOLD_ARCHIVE_LOCAL_ROOT
                    / f"{record['ordinal']:02d}-{record['system_id']}.model.tar.gz"
                ).as_posix(),
                "source_object": copy.deepcopy(record["source_object"]),
                "destination_object": copy.deepcopy(record["destination_object"]),
                "terminal_receipt_sha256": record["terminal_receipt_sha256"],
                "request_receipt_sha256": record["request_receipt_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "protocol": ARCHIVE_INPUT_MANIFEST_PROTOCOL,
        "experiment_id": "arr_retrieval_cv_v1",
        "outer_fold": completed["outer_fold"],
        "attempt_id": completed["attempt_id"],
        "archive_root": FOLD_ARCHIVE_LOCAL_ROOT.as_posix(),
        "training_plan_sha256": completed["training_plan_sha256"],
        "training_staging_receipt_sha256": completed[
            "training_staging_receipt_sha256"
        ],
        "source_bundle": copy.deepcopy(completed["source_bundle"]),
        "copy_set_receipt_sha256": _document_sha256(copy_set),
        "systems": systems,
    }


def _validate_copy_set(
    value: object,
    *,
    completed: Mapping[str, Any],
) -> dict[str, Any]:
    copy_set = _exact_object(value, _COPY_SET_KEYS, name="fold copy-set receipt")
    if (
        type(copy_set["schema_version"]) is not int
        or copy_set["schema_version"] != 1
        or copy_set["protocol"] != FOLD_ARCHIVE_COPY_SET_PROTOCOL
        or copy_set["completed_fold_evidence_sha256"] != _document_sha256(completed)
        or copy_set["training_plan_sha256"] != completed["training_plan_sha256"]
    ):
        raise ValueError("Fold copy-set evidence binding changed")
    prefix = _normalized_prefix(
        copy_set["destination_prefix"], name="fold archive destination prefix"
    )
    systems = copy_set["systems"]
    if type(systems) is not list or len(systems) != 12:
        raise ValueError("Fold copy-set must contain twelve systems")
    normalized_systems: list[dict[str, Any]] = []
    source_coordinates: set[tuple[str, str, str]] = set()
    destination_coordinates: set[tuple[str, str, str]] = set()
    for ordinal, (raw, expected) in enumerate(zip(systems, completed["systems"])):
        system = _exact_object(
            raw, _COPY_SYSTEM_KEYS, name=f"fold copy-set systems[{ordinal}]"
        )
        cell = expected["cell"]
        system_id = _archive_system_id(cell)
        source = _validate_archive_object(
            system["source_object"], name=f"copy systems[{ordinal}].source"
        )
        destination = _validate_archive_object(
            system["destination_object"],
            name=f"copy systems[{ordinal}].destination",
        )
        expected_key = f"{prefix}{ordinal:02d}-{system_id}.model.tar.gz"
        expected_source_bucket, expected_source_key, _, _ = (
            training_artifacts._expected_remote_coordinates(
                plan=completed["training_plan"],
                preflight=expected["preflight_receipt"],
                terminal=expected["terminal_receipt"],
            )
        )
        if (
            type(system["ordinal"]) is not int
            or system["ordinal"] != ordinal
            or system["system_id"] != system_id
            or system["run_id"] != expected["run_id"]
            or system["job_name"] != expected["job_name"]
            or system["cell"] != cell
            or system["terminal_receipt_sha256"]
            != expected["terminal_receipt_sha256"]
            or system["request_receipt_sha256"]
            != expected["request_receipt_sha256"]
            or source["bucket"] != expected_source_bucket
            or source["key"] != expected_source_key
            or destination["bucket"]
            != completed["training_plan"]["infrastructure"]["artifact_bucket"]
            or destination["key"] != expected_key
            or destination["size"] != source["size"]
        ):
            raise ValueError("Fold copied-system identity changed")
        expected_intent = _seal(
            {
                "schema_version": 1,
                "protocol": FOLD_ARCHIVE_COPY_INTENT_PROTOCOL,
                "ordinal": ordinal,
                "system_id": system_id,
                "run_id": expected["run_id"],
                "job_name": expected["job_name"],
                "source_object": source,
                "destination_bucket": destination["bucket"],
                "destination_key": destination["key"],
                "terminal_receipt_sha256": expected["terminal_receipt_sha256"],
                "request_receipt_sha256": expected["request_receipt_sha256"],
            }
        )
        if system["copy_intent_sha256"] != _document_sha256(expected_intent):
            raise ValueError("Fold copied-system intent binding changed")
        _validate_destination_sha256(destination, name="copied destination")
        source_coordinate = (source["bucket"], source["key"], source["version_id"])
        destination_coordinate = (
            destination["bucket"],
            destination["key"],
            destination["version_id"],
        )
        if (
            source_coordinate in source_coordinates
            or destination_coordinate in destination_coordinates
        ):
            raise ValueError("Fold copy-set aliases an object version")
        source_coordinates.add(source_coordinate)
        destination_coordinates.add(destination_coordinate)
        normalized = copy.deepcopy(system)
        normalized["source_object"] = source
        normalized["destination_object"] = destination
        normalized_systems.append(normalized)
    _validate_self_hash(copy_set, name="fold copy-set receipt")
    normalized = copy.deepcopy(copy_set)
    normalized["systems"] = normalized_systems
    return normalized


def validate_fold_archive_copy_receipt(
    value: object,
    *,
    completed_fold_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    receipt = _exact_object(
        value, _ARCHIVE_COPY_RECEIPT_KEYS, name="fold archive copy receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != FOLD_ARCHIVE_COPY_PROTOCOL
        or receipt["completed_fold_evidence_sha256"] != _document_sha256(completed)
        or receipt["training_plan_sha256"] != completed["training_plan_sha256"]
    ):
        raise ValueError("Fold archive copy receipt evidence binding changed")
    prefix = _normalized_prefix(
        receipt["destination_prefix"], name="fold archive destination prefix"
    )
    copy_set = _validate_copy_set(receipt["copy_set_receipt"], completed=completed)
    if copy_set["destination_prefix"] != prefix:
        raise ValueError("Fold archive receipt prefixes disagree")
    expected_manifest = _expected_fold_manifest(completed=completed, copy_set=copy_set)
    if receipt["fold_archive_input_manifest"] != expected_manifest:
        raise ValueError("Fold archive input manifest differs from its copy set")
    manifest = _exact_object(
        receipt["manifest_object"],
        _STATIC_FILE_KEYS - {"path"},
        name="fold archive manifest object",
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    if (
        manifest["bucket"] != infrastructure["artifact_bucket"]
        or manifest["key"] != f"{prefix}{FOLD_ARCHIVE_MANIFEST_NAME}"
        or manifest["size"] != len(_canonical_bytes(expected_manifest))
        or manifest["sha256"] != hashlib.sha256(
            _canonical_bytes(expected_manifest)
        ).hexdigest()
        or manifest["sse"] != "AES256"
    ):
        raise ValueError("Fold archive manifest-object identity changed")
    _exact_string(manifest["version_id"], name="fold manifest VersionId")
    if _ETAG.fullmatch(_exact_string(manifest["etag"], name="fold manifest ETag")) is None:
        raise ValueError("Fold manifest ETag changed")
    _validate_self_hash(receipt, name="fold archive copy receipt")
    normalized = copy.deepcopy(receipt)
    normalized["copy_set_receipt"] = copy_set
    normalized["fold_archive_input_manifest"] = expected_manifest
    normalized["manifest_object"] = copy.deepcopy(manifest)
    return normalized


def copy_completed_fold_archives_once(
    clients: aws.AwsClients,
    *,
    completed_fold_evidence: Mapping[str, Any],
    destination_prefix: str,
    state_dir: Path,
    local_archive_root: Path = FOLD_ARCHIVE_LOCAL_ROOT,
) -> dict[str, Any]:
    """Copy twelve exact training object versions and publish the manifest last."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    aws.validate_aws_sdk_versions()
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    if local_archive_root != FOLD_ARCHIVE_LOCAL_ROOT:
        raise ValueError("Fold archive local root left the image contract")
    plan = completed["training_plan"]
    infrastructure = plan["infrastructure"]
    bucket = infrastructure["artifact_bucket"]
    account = infrastructure["account_id"]
    region = infrastructure["region"]
    prefix = _normalized_prefix(
        destination_prefix, name="fold archive destination prefix"
    )
    aws.validate_artifact_bucket(clients.s3, bucket=bucket, region=region)
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=bucket,
        prefix=prefix,
        expected_bucket_owner=account,
    )
    history = _list_prefix_history(
        clients.s3,
        bucket=bucket,
        prefix=prefix,
        expected_bucket_owner=account,
    )
    _require_exact_copy_history(history, expected_objects=[])

    inspected: list[dict[str, Any]] = []
    source_coordinates: set[tuple[str, str, str]] = set()
    kms_keys: set[str] = set()
    for system in completed["systems"]:
        remote = training_artifacts._inspect_remote_output(
            clients.s3,
            plan=plan,
            preflight=system["preflight_receipt"],
            terminal=system["terminal_receipt"],
        )
        source = _archive_source_record(remote)
        coordinate = (source["bucket"], source["key"], source["version_id"])
        if coordinate in source_coordinates:
            raise ValueError("Completed fold aliases a source object version")
        source_coordinates.add(coordinate)
        kms_keys.add(source["encryption"]["kms_key_id"])
        inspected.append(source)
    if len(inspected) != 12 or len(kms_keys) != 1:
        raise ValueError("Completed-fold source archives do not share one KMS identity")

    state = _create_state_directory(
        state_dir,
        protocol=FOLD_ARCHIVE_COPY_INTENT_PROTOCOL,
    )
    root_intent = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_ARCHIVE_COPY_INTENT_PROTOCOL,
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "training_plan_sha256": completed["training_plan_sha256"],
            "destination_prefix": prefix,
            "source_objects": inspected,
        }
    )
    _publish_json_absent(state / "intent.json", root_intent)

    copied_systems: list[dict[str, Any]] = []
    expected_destinations: list[dict[str, Any]] = []
    for ordinal, (system, source) in enumerate(zip(completed["systems"], inspected)):
        fresh = _archive_source_record(
            training_artifacts._inspect_remote_output(
                clients.s3,
                plan=plan,
                preflight=system["preflight_receipt"],
                terminal=system["terminal_receipt"],
            )
        )
        if fresh != source:
            raise RuntimeError("Training source object changed before CopyObject")
        _require_exact_copy_history(
            _list_prefix_history(
                clients.s3,
                bucket=bucket,
                prefix=prefix,
                expected_bucket_owner=account,
            ),
            expected_objects=expected_destinations,
        )
        system_id = _archive_system_id(system["cell"])
        destination_key = f"{prefix}{ordinal:02d}-{system_id}.model.tar.gz"
        copy_intent = _seal(
            {
                "schema_version": 1,
                "protocol": FOLD_ARCHIVE_COPY_INTENT_PROTOCOL,
                "ordinal": ordinal,
                "system_id": system_id,
                "run_id": system["run_id"],
                "job_name": system["job_name"],
                "source_object": source,
                "destination_bucket": bucket,
                "destination_key": destination_key,
                "terminal_receipt_sha256": system["terminal_receipt_sha256"],
                "request_receipt_sha256": system["request_receipt_sha256"],
            }
        )
        _publish_json_absent(state / f"copy-{ordinal:02d}-intent.json", copy_intent)
        destination = _copy_archive_object_once(
            clients.s3,
            source=source,
            destination_bucket=bucket,
            destination_key=destination_key,
            expected_bucket_owner=account,
        )
        copy_record = {
            "ordinal": ordinal,
            "system_id": system_id,
            "run_id": system["run_id"],
            "job_name": system["job_name"],
            "cell": copy.deepcopy(system["cell"]),
            "source_object": copy.deepcopy(source),
            "destination_object": destination,
            "terminal_receipt_sha256": system["terminal_receipt_sha256"],
            "request_receipt_sha256": system["request_receipt_sha256"],
            "copy_intent_sha256": _document_sha256(copy_intent),
        }
        _publish_json_absent(
            state / f"copy-{ordinal:02d}-receipt.json",
            _seal(
                {
                    "schema_version": 1,
                    "protocol": FOLD_ARCHIVE_COPY_PROTOCOL,
                    "copy": copy_record,
                }
            ),
        )
        copied_systems.append(copy_record)
        expected_destinations.append(destination)
        _require_exact_copy_history(
            _list_prefix_history(
                clients.s3,
                bucket=bucket,
                prefix=prefix,
                expected_bucket_owner=account,
            ),
            expected_objects=expected_destinations,
        )

    copy_set = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_ARCHIVE_COPY_SET_PROTOCOL,
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "training_plan_sha256": completed["training_plan_sha256"],
            "destination_prefix": prefix,
            "systems": copied_systems,
        }
    )
    copy_set = _validate_copy_set(copy_set, completed=completed)
    _publish_json_absent(state / "copy-set-receipt.json", copy_set)
    manifest_payload = _expected_fold_manifest(completed=completed, copy_set=copy_set)
    manifest_intent = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_ARCHIVE_MANIFEST_PUBLICATION_PROTOCOL,
            "destination_prefix": prefix,
            "copy_set_receipt_sha256": _document_sha256(copy_set),
            "fold_archive_input_manifest": manifest_payload,
        }
    )
    _publish_json_absent(state / "manifest-intent.json", manifest_intent)
    _require_exact_copy_history(
        _list_prefix_history(
            clients.s3,
            bucket=bucket,
            prefix=prefix,
            expected_bucket_owner=account,
        ),
        expected_objects=expected_destinations,
    )
    manifest_object = _put_json_object_once(
        clients.s3,
        payload=manifest_payload,
        bucket=bucket,
        key=f"{prefix}{FOLD_ARCHIVE_MANIFEST_NAME}",
        expected_bucket_owner=account,
    )
    _require_exact_copy_history(
        _list_prefix_history(
            clients.s3,
            bucket=bucket,
            prefix=prefix,
            expected_bucket_owner=account,
        ),
        expected_objects=[*expected_destinations, manifest_object],
    )
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_ARCHIVE_COPY_PROTOCOL,
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "training_plan_sha256": completed["training_plan_sha256"],
            "destination_prefix": prefix,
            "copy_set_receipt": copy_set,
            "fold_archive_input_manifest": manifest_payload,
            "manifest_object": manifest_object,
        }
    )
    validated = validate_fold_archive_copy_receipt(
        receipt,
        completed_fold_evidence=completed,
    )
    _publish_json_absent(state / "receipt.json", validated)
    return validated


def _read_exact_object(
    s3: object,
    *,
    record: Mapping[str, Any],
    expected_bucket_owner: str,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    if record["size"] > maximum_bytes:
        raise ValueError("Compact control object exceeds the readback ceiling")
    response = s3.get_object(
        Bucket=record["bucket"],
        Key=record["key"],
        VersionId=record["version_id"],
        ExpectedBucketOwner=expected_bucket_owner,
    )
    body = response.get("Body")
    if body is None:
        raise RuntimeError("Versioned GetObject omitted its body")
    payload = body.read(record["size"] + 1)
    if type(payload) is not bytes or len(payload) != record["size"] or body.read(1) != b"":
        raise RuntimeError("Compact object readback size changed")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError("Compact object readback SHA-256 changed")
    return payload


def _verify_aes256_object(
    s3: object,
    *,
    record: Mapping[str, Any],
    expected_bucket_owner: str,
) -> None:
    checksum = base64.b64encode(bytes.fromhex(record["sha256"])).decode("ascii")
    head = s3.head_object(
        Bucket=record["bucket"],
        Key=record["key"],
        VersionId=record["version_id"],
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=expected_bucket_owner,
    )
    if (
        head.get("ContentLength") != record["size"]
        or head.get("ETag") != record["etag"]
        or head.get("VersionId") != record["version_id"]
        or head.get("ChecksumSHA256") != checksum
        or head.get("ChecksumType") != "FULL_OBJECT"
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("Metadata") != {"sha256": record["sha256"]}
    ):
        raise RuntimeError("Static object metadata changed after staging")


def verify_remote_static_evaluation_inputs(
    clients: aws.AwsClients,
    *,
    receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate every exact staged static version without model retention."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    staged = validate_static_evaluation_staging_receipt(
        copy.deepcopy(receipt), completed_fold_evidence=completed_fold_evidence
    )
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    account = completed["training_plan"]["infrastructure"]["account_id"]
    expected: list[dict[str, Any]] = []
    for asset in staged["assets"]:
        for record in asset["files"]:
            _verify_aes256_object(
                clients.s3,
                record=record,
                expected_bucket_owner=account,
            )
            expected.append(record)
    manifest = staged["manifest_object"]
    _verify_aes256_object(
        clients.s3,
        record=manifest,
        expected_bucket_owner=account,
    )
    manifest_payload = {
        key: copy.deepcopy(staged[key])
        for key in (
            "schema_version",
            "protocol",
            "completed_fold_evidence_sha256",
            "training_plan_sha256",
            "destination_prefix",
            "assets",
        )
    }
    if _read_exact_object(
        clients.s3,
        record=manifest,
        expected_bucket_owner=account,
    ) != _canonical_bytes(manifest_payload):
        raise RuntimeError("Remote static manifest content changed")
    _require_exact_copy_history(
        _list_prefix_history(
            clients.s3,
            bucket=manifest["bucket"],
            prefix=staged["destination_prefix"],
            expected_bucket_owner=account,
        ),
        expected_objects=[*expected, manifest],
    )
    return _seal(
        {
            "schema_version": 1,
            "protocol": STATIC_STAGING_PROTOCOL,
            "staging_receipt_sha256": _document_sha256(staged),
            "verified_object_versions": len(expected) + 1,
            "verified": True,
        }
    )


def verify_remote_fold_archives(
    clients: aws.AwsClients,
    *,
    receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate source and copied versions plus the manifest-last prefix."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    copied = validate_fold_archive_copy_receipt(
        copy.deepcopy(receipt), completed_fold_evidence=completed_fold_evidence
    )
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    plan = completed["training_plan"]
    account = plan["infrastructure"]["account_id"]
    expected: list[dict[str, Any]] = []
    for system, completed_system in zip(
        copied["copy_set_receipt"]["systems"], completed["systems"]
    ):
        fresh_source = _archive_source_record(
            training_artifacts._inspect_remote_output(
                clients.s3,
                plan=plan,
                preflight=completed_system["preflight_receipt"],
                terminal=completed_system["terminal_receipt"],
            )
        )
        if fresh_source != system["source_object"]:
            raise RuntimeError("Training source archive changed after fold copy")
        destination = system["destination_object"]
        head = clients.s3.head_object(
            Bucket=destination["bucket"],
            Key=destination["key"],
            VersionId=destination["version_id"],
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=account,
        )
        encryption = destination["encryption"]
        if (
            head.get("ContentLength") != destination["size"]
            or head.get("ContentType") != "application/gzip"
            or head.get("ETag") != destination["etag"]
            or head.get("VersionId") != destination["version_id"]
            or head.get("ChecksumSHA256") != destination["checksum"]["value"]
            or head.get("ChecksumType") != "FULL_OBJECT"
            or head.get("ServerSideEncryption") != "aws:kms"
            or head.get("SSEKMSKeyId") != encryption["kms_key_id"]
            or head.get("BucketKeyEnabled") is not True
            or head.get("Metadata") != {}
        ):
            raise RuntimeError("Copied archive changed after publication")
        expected.append(destination)
    manifest = copied["manifest_object"]
    _verify_aes256_object(
        clients.s3,
        record=manifest,
        expected_bucket_owner=account,
    )
    if _read_exact_object(
        clients.s3,
        record=manifest,
        expected_bucket_owner=account,
    ) != _canonical_bytes(copied["fold_archive_input_manifest"]):
        raise RuntimeError("Remote fold archive manifest content changed")
    _require_exact_copy_history(
        _list_prefix_history(
            clients.s3,
            bucket=manifest["bucket"],
            prefix=copied["destination_prefix"],
            expected_bucket_owner=account,
        ),
        expected_objects=[*expected, manifest],
    )
    return _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_ARCHIVE_COPY_PROTOCOL,
            "copy_receipt_sha256": _document_sha256(copied),
            "verified_source_versions": 12,
            "verified_destination_versions": 13,
            "verified": True,
        }
    )


def _validate_overlay_publication(
    value: object,
    *,
    account_id: str,
    region: str,
) -> dict[str, Any]:
    publication = _exact_object(
        value,
        {
            "content_tag",
            "identity",
            "manifest_digest",
            "media_type",
            "protocol",
            "raw_manifest_sha256",
            "remote_digest_uri",
            "remote_tag_uri",
        },
        name="fold overlay publication",
    )
    identity = _exact_object(
        publication["identity"],
        {
            "build_context_files_sha256",
            "build_context_identity_sha256",
            "config_digest",
            "image_digest",
            "local_image_identity_sha256",
            "manifest_media_type",
            "offline_smoke_sha256",
        },
        name="fold overlay publication identity",
    )
    repository = f"{account_id}.dkr.ecr.{region}.amazonaws.com/arr-retrieval-eval"
    expected_uri = f"{repository}@{FOLD_OVERLAY_IMAGE_DIGEST}"
    if (
        publication["protocol"] != FOLD_OVERLAY_PUBLICATION_PROTOCOL
        or publication["manifest_digest"] != FOLD_OVERLAY_IMAGE_DIGEST
        or publication["raw_manifest_sha256"]
        != FOLD_OVERLAY_IMAGE_DIGEST.removeprefix("sha256:")
        or publication["remote_digest_uri"] != expected_uri
        or identity["build_context_identity_sha256"]
        != FOLD_OVERLAY_BUILD_IDENTITY
        or identity["config_digest"] != FOLD_OVERLAY_CONFIG_DIGEST
        or identity["image_digest"] != FOLD_OVERLAY_IMAGE_DIGEST
        or identity["offline_smoke_sha256"] != FOLD_OVERLAY_OFFLINE_SMOKE_SHA256
        or identity["manifest_media_type"] != aws.ECR_MEDIA_TYPE
        or publication["media_type"] != aws.ECR_MEDIA_TYPE
    ):
        raise ValueError("Fold overlay publication identity changed")
    for field in (
        "build_context_files_sha256",
        "build_context_identity_sha256",
        "local_image_identity_sha256",
        "offline_smoke_sha256",
    ):
        _exact_sha256(identity[field], name=f"fold overlay identity.{field}")
    return copy.deepcopy(publication)


def _phase1_input(*, name: str, s3_uri: str, local_path: str) -> dict[str, Any]:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": s3_uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }


def _static_asset(receipt: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [asset for asset in receipt["assets"] if asset["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Static staging receipt lacks exactly one {name} asset")
    return matches[0]


def _render_fold_inventory_request(
    *,
    completed: Mapping[str, Any],
    archive_copy: Mapping[str, Any],
    static_staging: Mapping[str, Any],
    publication: Mapping[str, Any],
    job_name: str,
    output_prefix: str,
) -> dict[str, Any]:
    plan = completed["training_plan"]
    infrastructure = plan["infrastructure"]
    bucket = infrastructure["artifact_bucket"]
    expected_job_name = (
        f"arr-ret-cv1-f{completed['outer_fold']}-inventory-{completed['attempt_id']}"
    )
    if (
        type(job_name) is not str
        or _JOB_NAME.fullmatch(job_name) is None
        or job_name != expected_job_name
    ):
        raise ValueError(f"Phase-1 job name must equal {expected_job_name}")
    output_prefix = _normalized_prefix(output_prefix, name="Phase-1 output prefix")
    dataset_uris = {
        run["input_channels"]["data"]["s3_uri"]
        for run in plan["controlled_runs"]
    }
    if len(dataset_uris) != 1:
        raise ValueError("Controlled plan does not share one corrected dataset input")
    dataset_uri = next(iter(dataset_uris)).rstrip("/") + "/"
    dataset_bucket, dataset_prefix = dataset_uri.removeprefix("s3://").split(
        "/", 1
    )
    dataset_prefix = _normalized_prefix(
        dataset_prefix,
        name="corrected dataset prefix",
    )
    for input_name, input_bucket, input_prefix in (
        (
            "fold archives",
            bucket,
            _normalized_prefix(
                archive_copy["destination_prefix"],
                name="fold archive prefix",
            ),
        ),
        (
            "static assets",
            bucket,
            _normalized_prefix(
                static_staging["destination_prefix"],
                name="static staging prefix",
            ),
        ),
        ("corrected dataset", dataset_bucket, dataset_prefix),
    ):
        if input_bucket == bucket and _prefixes_overlap(output_prefix, input_prefix):
            raise ValueError(
                f"Phase-1 output prefix overlaps the {input_name} input prefix"
            )
    archive_uri = f"s3://{bucket}/{archive_copy['destination_prefix']}"
    control_uri = f"s3://{bucket}/{_static_asset(static_staging, 'control')['s3_prefix']}"
    kms_keys = {
        system["destination_object"]["encryption"]["kms_key_id"]
        for system in archive_copy["copy_set_receipt"]["systems"]
    }
    if len(kms_keys) != 1:
        raise ValueError("Fold archives do not share one output KMS identity")
    kms_key = next(iter(kms_keys))
    tags = [
        {"Key": "Experiment", "Value": "arr_retrieval_cv_v1"},
        {"Key": "ManagedBy", "Value": "arr-retrieval-cv"},
        {"Key": "Purpose", "Value": "fold-inventory"},
    ]
    return {
        "AppSpecification": {
            "ContainerArguments": [
                "/opt/program/modernbert/processing_fold_eval/inventory_sm.py",
                "--archive-manifest",
                "/opt/ml/processing/input/fold-archives/fold_archive_input_manifest.json",
                "--dataset-dir",
                "/opt/ml/processing/input/dataset",
                "--fold-manifest",
                "/opt/ml/processing/input/control/folds.json",
                "--scratch-parent",
                "/opt/ml/processing/work",
                "--output-dir",
                "/opt/ml/processing/output/evidence",
            ],
            "ContainerEntrypoint": ["/opt/conda/bin/python"],
            "ImageUri": publication["remote_digest_uri"],
        },
        "Environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "17",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "NetworkConfig": {
            "EnableInterContainerTrafficEncryption": False,
            "EnableNetworkIsolation": True,
        },
        "ProcessingInputs": [
            _phase1_input(
                name="fold-archives",
                s3_uri=archive_uri,
                local_path="/opt/ml/processing/input/fold-archives",
            ),
            _phase1_input(
                name="dataset",
                s3_uri=dataset_uri,
                local_path="/opt/ml/processing/input/dataset",
            ),
            _phase1_input(
                name="control",
                s3_uri=control_uri,
                local_path="/opt/ml/processing/input/control",
            ),
        ],
        "ProcessingJobName": job_name,
        "ProcessingOutputConfig": {
            "KmsKeyId": kms_key,
            "Outputs": [
                {
                    "OutputName": "evidence",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{output_prefix}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": infrastructure["processing_instance_count"],
                "InstanceType": infrastructure["processing_instance_type"],
                "VolumeSizeInGB": infrastructure["processing_volume_size_gb"],
            }
        },
        "RoleArn": infrastructure["role_arn"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": infrastructure[
                "processing_max_runtime_seconds"
            ]
        },
        "Tags": tags,
    }


def validate_fold_inventory_preflight_receipt(
    value: object,
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    archive_copy = validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt), completed_fold_evidence=completed
    )
    static_staging = validate_static_evaluation_staging_receipt(
        copy.deepcopy(static_staging_receipt), completed_fold_evidence=completed
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    publication = _validate_overlay_publication(
        copy.deepcopy(overlay_publication_receipt),
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    receipt = _exact_object(
        value, _PHASE1_PREFLIGHT_KEYS, name="fold inventory preflight receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != FOLD_INVENTORY_PREFLIGHT_PROTOCOL
        or receipt["outer_fold"] != completed["outer_fold"]
        or receipt["account_id"] != infrastructure["account_id"]
        or receipt["region"] != infrastructure["region"]
        or receipt["image_uri"] != publication["remote_digest_uri"]
        or receipt["image_publication_sha256"] != _document_sha256(publication)
        or receipt["completed_fold_evidence_sha256"] != _document_sha256(completed)
        or receipt["archive_copy_receipt_sha256"] != _document_sha256(archive_copy)
        or receipt["static_staging_receipt_sha256"]
        != _document_sha256(static_staging)
        or receipt["training_staging_receipt_sha256"]
        != completed["training_staging_receipt_sha256"]
        or receipt["sdk_versions"] != aws.EXPECTED_AWS_SDK_VERSIONS
        or type(receipt["processing_quota"]) is not int
        or receipt["processing_quota"] < 1
    ):
        raise ValueError("Fold inventory preflight evidence binding changed")
    caller = _exact_string(receipt["caller_arn"], name="Phase-1 caller ARN")
    if not caller.startswith(
        (
            f"arn:aws:iam::{receipt['account_id']}:",
            f"arn:aws:sts::{receipt['account_id']}:",
        )
    ):
        raise ValueError("Phase-1 caller ARN differs from its account")
    output_prefix = _normalized_prefix(
        receipt["output_prefix"], name="Phase-1 output prefix"
    )
    request = _render_fold_inventory_request(
        completed=completed,
        archive_copy=archive_copy,
        static_staging=static_staging,
        publication=publication,
        job_name=receipt["job_name"],
        output_prefix=output_prefix,
    )
    if (
        receipt["request"] != request
        or receipt["request_sha256"] != _document_sha256(request)
    ):
        raise ValueError("Fold inventory request differs from exact re-rendering")
    static_version_count = (
        sum(len(asset["files"]) for asset in static_staging["assets"]) + 1
    )
    expected_archive_verification = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_ARCHIVE_COPY_PROTOCOL,
            "copy_receipt_sha256": _document_sha256(archive_copy),
            "verified_source_versions": 12,
            "verified_destination_versions": 13,
            "verified": True,
        }
    )
    expected_static_verification = _seal(
        {
            "schema_version": 1,
            "protocol": STATIC_STAGING_PROTOCOL,
            "staging_receipt_sha256": _document_sha256(static_staging),
            "verified_object_versions": static_version_count,
            "verified": True,
        }
    )
    if receipt["archive_verification"] != expected_archive_verification:
        raise ValueError("Phase-1 archive verification receipt changed")
    if receipt["static_verification"] != expected_static_verification:
        raise ValueError("Phase-1 static verification receipt changed")
    _validate_self_hash(receipt, name="fold inventory preflight receipt")
    return copy.deepcopy(receipt)


def preflight_fold_inventory(
    clients: aws.AwsClients,
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
    job_name: str,
    output_prefix: str,
) -> dict[str, Any]:
    """Perform all read-only gates and freeze one exact Phase-1 request."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    sdk_versions = aws.validate_aws_sdk_versions()
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    archive_copy = validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt), completed_fold_evidence=completed
    )
    static_staging = validate_static_evaluation_staging_receipt(
        copy.deepcopy(static_staging_receipt), completed_fold_evidence=completed
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    publication = _validate_overlay_publication(
        copy.deepcopy(overlay_publication_receipt),
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    output_prefix = _normalized_prefix(output_prefix, name="Phase-1 output prefix")
    request = _render_fold_inventory_request(
        completed=completed,
        archive_copy=archive_copy,
        static_staging=static_staging,
        publication=publication,
        job_name=job_name,
        output_prefix=output_prefix,
    )
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != infrastructure["account_id"]:
        raise ValueError("Active AWS account differs from the completed-fold contract")
    aws._assert_role_trust(clients.iam, infrastructure["role_arn"])
    training_aws.verify_remote_training_staging(
        clients.s3,
        training_plan=completed["training_plan"],
        staging_receipt=completed["training_staging_receipt"],
        deep_read=False,
    )
    archive_verification = verify_remote_fold_archives(
        clients,
        receipt=archive_copy,
        completed_fold_evidence=completed,
    )
    static_verification = verify_remote_static_evaluation_inputs(
        clients,
        receipt=static_staging,
        completed_fold_evidence=completed,
    )
    image_response = clients.ecr.batch_get_image(
        registryId=infrastructure["account_id"],
        repositoryName="arr-retrieval-eval",
        imageIds=[{"imageDigest": FOLD_OVERLAY_IMAGE_DIGEST}],
        acceptedMediaTypes=[aws.ECR_MEDIA_TYPE],
    )
    if image_response.get("failures") or len(image_response.get("images", [])) != 1:
        raise ValueError("Fold overlay image is not readable by exact digest")
    raw_manifest = image_response["images"][0].get("imageManifest")
    if (
        type(raw_manifest) is not str
        or "sha256:" + hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()
        != FOLD_OVERLAY_IMAGE_DIGEST
    ):
        raise ValueError("Fold overlay ECR manifest differs from its digest")
    quota_response = clients.service_quotas.get_service_quota(
        ServiceCode="sagemaker", QuotaCode="L-B013C051"
    )
    if type(quota_response) is not dict or type(quota_response.get("Quota")) is not dict:
        raise RuntimeError("Service Quotas returned a malformed Processing response")
    quota = quota_response["Quota"]
    quota_value = quota.get("Value")
    if type(quota_value) is int:
        exact_quota = quota_value
    elif (
        type(quota_value) is float
        and math.isfinite(quota_value)
        and quota_value.is_integer()
    ):
        exact_quota = int(quota_value)
    else:
        raise ValueError("Processing quota value is not one exact integer")
    if exact_quota < 1:
        raise RuntimeError("Processing ml.g5.12xlarge quota is below one")
    offerings = clients.ec2.describe_instance_type_offerings(
        LocationType="region",
        Filters=[{"Name": "instance-type", "Values": ["g5.12xlarge"]}],
    ).get("InstanceTypeOfferings", [])
    if not any(record.get("InstanceType") == "g5.12xlarge" for record in offerings):
        raise RuntimeError("g5.12xlarge is not offered in the configured region")
    existing = clients.sagemaker.list_processing_jobs(
        NameContains=job_name,
        MaxResults=100,
        SortBy="Name",
        SortOrder="Ascending",
    ).get("ProcessingJobSummaries", [])
    if any(record.get("ProcessingJobName") == job_name for record in existing):
        raise FileExistsError(f"Processing job name already exists: {job_name}")
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=infrastructure["artifact_bucket"],
        prefix=output_prefix,
        expected_bucket_owner=infrastructure["account_id"],
    )
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_INVENTORY_PREFLIGHT_PROTOCOL,
            "outer_fold": completed["outer_fold"],
            "account_id": infrastructure["account_id"],
            "region": infrastructure["region"],
            "caller_arn": caller.get("Arn"),
            "job_name": job_name,
            "output_prefix": output_prefix,
            "image_uri": publication["remote_digest_uri"],
            "image_publication_sha256": _document_sha256(publication),
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "archive_copy_receipt_sha256": _document_sha256(archive_copy),
            "static_staging_receipt_sha256": _document_sha256(static_staging),
            "training_staging_receipt_sha256": completed[
                "training_staging_receipt_sha256"
            ],
            "archive_verification": archive_verification,
            "static_verification": static_verification,
            "request": request,
            "request_sha256": _document_sha256(request),
            "sdk_versions": sdk_versions,
            "processing_quota": exact_quota,
        }
    )
    return validate_fold_inventory_preflight_receipt(
        receipt,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive_copy,
        static_staging_receipt=static_staging,
        overlay_publication_receipt=publication,
    )


def validate_fold_inventory_submission_receipt(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_fold_inventory_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    receipt = _exact_object(
        value, _PHASE1_SUBMISSION_KEYS, name="fold inventory submission receipt"
    )
    expected_arn = (
        f"arn:aws:sagemaker:{preflight['region']}:{preflight['account_id']}:"
        f"processing-job/{preflight['job_name']}"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != FOLD_INVENTORY_SUBMISSION_PROTOCOL
        or receipt["outer_fold"] != preflight["outer_fold"]
        or receipt["job_name"] != preflight["job_name"]
        or receipt["job_arn"] != expected_arn
        or receipt["preflight_receipt_sha256"] != _document_sha256(preflight)
        or receipt["request_sha256"] != preflight["request_sha256"]
    ):
        raise ValueError("Fold inventory submission binding changed")
    _validate_self_hash(receipt, name="fold inventory submission receipt")
    return copy.deepcopy(receipt)


def submit_fold_inventory_once(
    clients: aws.AwsClients,
    *,
    preflight_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
    state_dir: Path,
) -> dict[str, Any]:
    """Persist a create intent, then invoke CreateProcessingJob exactly once."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    preflight = validate_fold_inventory_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    fresh_preflight = preflight_fold_inventory(
        clients,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
        job_name=preflight["job_name"],
        output_prefix=preflight["output_prefix"],
    )
    if fresh_preflight != preflight:
        raise RuntimeError(
            "Phase-1 preflight is no longer exactly reproducible at submission"
        )
    caller = clients.sts.get_caller_identity()
    if (
        caller.get("Account") != preflight["account_id"]
        or caller.get("Arn") != preflight["caller_arn"]
    ):
        raise ValueError("Active AWS caller differs from the Phase-1 preflight")
    existing = clients.sagemaker.list_processing_jobs(
        NameContains=preflight["job_name"],
        MaxResults=100,
        SortBy="Name",
        SortOrder="Ascending",
    ).get("ProcessingJobSummaries", [])
    if any(
        record.get("ProcessingJobName") == preflight["job_name"]
        for record in existing
    ):
        raise FileExistsError("Phase-1 Processing job already exists")
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=preflight["request"]["ProcessingOutputConfig"]["Outputs"][0][
            "S3Output"
        ]["S3Uri"].split("/", 3)[2],
        prefix=preflight["output_prefix"],
        expected_bucket_owner=preflight["account_id"],
    )
    state = _create_state_directory(
        state_dir,
        protocol=FOLD_INVENTORY_SUBMISSION_PROTOCOL,
    )
    intent = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_INVENTORY_SUBMISSION_PROTOCOL,
            "outer_fold": preflight["outer_fold"],
            "job_name": preflight["job_name"],
            "preflight_receipt_sha256": _document_sha256(preflight),
            "request": preflight["request"],
            "request_sha256": preflight["request_sha256"],
        }
    )
    _publish_json_absent(state / "create-intent.json", intent)
    response = clients.sagemaker.create_processing_job(**preflight["request"])
    expected_arn = (
        f"arn:aws:sagemaker:{preflight['region']}:{preflight['account_id']}:"
        f"processing-job/{preflight['job_name']}"
    )
    if response.get("ProcessingJobArn") != expected_arn:
        raise RuntimeError("CreateProcessingJob returned an unexpected ARN")
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_INVENTORY_SUBMISSION_PROTOCOL,
            "outer_fold": preflight["outer_fold"],
            "job_name": preflight["job_name"],
            "job_arn": expected_arn,
            "preflight_receipt_sha256": _document_sha256(preflight),
            "request_sha256": preflight["request_sha256"],
        }
    )
    validated = validate_fold_inventory_submission_receipt(
        receipt,
        preflight_receipt=preflight,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    _publish_json_absent(state / "submission.json", validated)
    return validated


def describe_fold_inventory(
    sagemaker: object,
    *,
    job_name: str,
) -> dict[str, Any]:
    if type(job_name) is not str or _JOB_NAME.fullmatch(job_name) is None:
        raise ValueError("Phase-1 job name is invalid")
    response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
    return {
        "schema_version": 1,
        "protocol": FOLD_INVENTORY_TERMINAL_PROTOCOL,
        "job_name": response.get("ProcessingJobName"),
        "job_arn": response.get("ProcessingJobArn"),
        "status": response.get("ProcessingJobStatus"),
        "failure_reason": response.get("FailureReason"),
        "exit_message": response.get("ExitMessage"),
        "processing_start_time": (
            None
            if response.get("ProcessingStartTime") is None
            else _normalize_datetime(
                response["ProcessingStartTime"], name="ProcessingStartTime"
            )
        ),
        "processing_end_time": (
            None
            if response.get("ProcessingEndTime") is None
            else _normalize_datetime(
                response["ProcessingEndTime"], name="ProcessingEndTime"
            )
        ),
    }


def _validate_processing_io_readback(
    response: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> None:
    """Accept only SageMaker's explicit false AppManaged readback defaults."""

    actual_inputs = response.get("ProcessingInputs")
    expected_inputs = request["ProcessingInputs"]
    if type(actual_inputs) is not list or len(actual_inputs) != len(expected_inputs):
        raise RuntimeError("DescribeProcessingJob ProcessingInputs differs from request")
    normalized_inputs: list[dict[str, Any]] = []
    for record in actual_inputs:
        if (
            type(record) is not dict
            or set(record) != {"AppManaged", "InputName", "S3Input"}
            or record["AppManaged"] is not False
        ):
            raise RuntimeError(
                "DescribeProcessingJob ProcessingInputs has an unexpected service default"
            )
        normalized_inputs.append(
            {key: copy.deepcopy(value) for key, value in record.items() if key != "AppManaged"}
        )
    if normalized_inputs != expected_inputs:
        raise RuntimeError("DescribeProcessingJob ProcessingInputs differs from request")

    actual_output = response.get("ProcessingOutputConfig")
    expected_output = request["ProcessingOutputConfig"]
    if (
        type(actual_output) is not dict
        or set(actual_output) != set(expected_output)
        or type(actual_output.get("Outputs")) is not list
        or len(actual_output["Outputs"]) != len(expected_output["Outputs"])
    ):
        raise RuntimeError("DescribeProcessingJob ProcessingOutputConfig differs from request")
    normalized_outputs: list[dict[str, Any]] = []
    for record in actual_output["Outputs"]:
        if (
            type(record) is not dict
            or set(record) != {"AppManaged", "OutputName", "S3Output"}
            or record["AppManaged"] is not False
        ):
            raise RuntimeError(
                "DescribeProcessingJob ProcessingOutputConfig has an unexpected service default"
            )
        normalized_outputs.append(
            {key: copy.deepcopy(value) for key, value in record.items() if key != "AppManaged"}
        )
    normalized_output = {
        **{key: copy.deepcopy(value) for key, value in actual_output.items() if key != "Outputs"},
        "Outputs": normalized_outputs,
    }
    if normalized_output != expected_output:
        raise RuntimeError("DescribeProcessingJob ProcessingOutputConfig differs from request")


def validate_fold_inventory_terminal_receipt(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_fold_inventory_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    submission = validate_fold_inventory_submission_receipt(
        copy.deepcopy(submission_receipt),
        preflight_receipt=preflight,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    receipt = _exact_object(
        value, _PHASE1_TERMINAL_KEYS, name="fold inventory terminal receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != FOLD_INVENTORY_TERMINAL_PROTOCOL
        or receipt["outer_fold"] != preflight["outer_fold"]
        or receipt["job_name"] != preflight["job_name"]
        or receipt["job_arn"] != submission["job_arn"]
        or receipt["preflight_receipt_sha256"] != _document_sha256(preflight)
        or receipt["submission_receipt_sha256"] != _document_sha256(submission)
        or receipt["request_sha256"] != preflight["request_sha256"]
        or receipt["status"] != "Completed"
        or receipt["failure_reason"] is not None
        or type(receipt["processing_time_microseconds"]) is not int
        or receipt["processing_time_microseconds"] < 0
    ):
        raise ValueError("Fold inventory terminal evidence changed")
    _exact_string(receipt["processing_start_time"], name="Phase-1 start time")
    _exact_string(receipt["processing_end_time"], name="Phase-1 end time")
    if receipt["exit_message"] is not None and type(receipt["exit_message"]) is not str:
        raise ValueError("Phase-1 exit message must be null or a string")
    _validate_self_hash(receipt, name="fold inventory terminal receipt")
    return copy.deepcopy(receipt)


def verify_completed_fold_inventory(
    clients: aws.AwsClients,
    *,
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a terminal receipt only for one exact clean Phase-1 completion."""

    preflight = validate_fold_inventory_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    submission = validate_fold_inventory_submission_receipt(
        copy.deepcopy(submission_receipt),
        preflight_receipt=preflight,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != preflight["account_id"]:
        raise ValueError("Active AWS account differs from the Phase-1 receipt")
    response = clients.sagemaker.describe_processing_job(
        ProcessingJobName=preflight["job_name"]
    )
    if response.get("ProcessingJobStatus") != "Completed" or response.get(
        "FailureReason"
    ):
        raise RuntimeError(
            "Fold inventory is not cleanly complete: "
            f"status={response.get('ProcessingJobStatus')}, "
            f"reason={response.get('FailureReason')!r}"
        )
    if (
        response.get("ProcessingJobName") != preflight["job_name"]
        or response.get("ProcessingJobArn") != submission["job_arn"]
    ):
        raise RuntimeError("DescribeProcessingJob identity differs from submission")
    request = preflight["request"]
    for field in (
        "AppSpecification",
        "Environment",
        "NetworkConfig",
        "ProcessingResources",
        "RoleArn",
        "StoppingCondition",
    ):
        if response.get(field) != request[field]:
            raise RuntimeError(f"DescribeProcessingJob {field} differs from request")
    _validate_processing_io_readback(response, request=request)
    start = response.get("ProcessingStartTime")
    end = response.get("ProcessingEndTime")
    if (
        type(start) is not datetime
        or type(end) is not datetime
        or start.tzinfo is None
        or end.tzinfo is None
        or end < start
    ):
        raise RuntimeError("Completed Phase-1 timing evidence is invalid")
    elapsed = end - start
    elapsed_microseconds = (
        elapsed.days * 86_400_000_000
        + elapsed.seconds * 1_000_000
        + elapsed.microseconds
    )
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_INVENTORY_TERMINAL_PROTOCOL,
            "outer_fold": preflight["outer_fold"],
            "job_name": preflight["job_name"],
            "job_arn": submission["job_arn"],
            "preflight_receipt_sha256": _document_sha256(preflight),
            "submission_receipt_sha256": _document_sha256(submission),
            "request_sha256": preflight["request_sha256"],
            "status": "Completed",
            "failure_reason": None,
            "processing_start_time": _normalize_datetime(
                start, name="ProcessingStartTime"
            ),
            "processing_end_time": _normalize_datetime(
                end, name="ProcessingEndTime"
            ),
            "processing_time_microseconds": elapsed_microseconds,
            "exit_message": response.get("ExitMessage"),
        }
    )
    return validate_fold_inventory_terminal_receipt(
        receipt,
        preflight_receipt=preflight,
        submission_receipt=submission,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )


def _load_compact_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not dict or payload != _canonical_bytes(value):
        raise ValueError(f"{name} is not one compact canonical JSON object")
    return value


def _validate_phase1_documents(
    *,
    archive_inventory: object,
    bm25_storage: object,
    artifact_manifest: object,
    archive_copy: Mapping[str, Any],
    completed: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(archive_inventory) is not dict:
        raise ValueError("Phase-1 archive inventory must be an object")
    expected_archive_keys = {
        "schema_version",
        "protocol",
        "input_manifest_sha256",
        "experiment_id",
        "outer_fold",
        "systems",
        "aggregate",
        "receipt_sha256",
    }
    inventory = _exact_object(
        archive_inventory, expected_archive_keys, name="Phase-1 archive inventory"
    )
    input_manifest = archive_copy["fold_archive_input_manifest"]
    if (
        inventory["schema_version"] != 1
        or type(inventory["schema_version"]) is not int
        or inventory["protocol"] != ARCHIVE_INVENTORY_PROTOCOL
        or inventory["input_manifest_sha256"] != _document_sha256(input_manifest)
        or inventory["experiment_id"] != "arr_retrieval_cv_v1"
        or inventory["outer_fold"] != completed["outer_fold"]
        or type(inventory["systems"]) is not list
        or len(inventory["systems"]) != 12
        or type(inventory["aggregate"]) is not dict
    ):
        raise ValueError("Phase-1 archive inventory binding changed")
    for index, (observed, expected) in enumerate(
        zip(inventory["systems"], input_manifest["systems"])
    ):
        if type(observed) is not dict or observed.get("archive_evidence") is None:
            raise ValueError("Phase-1 archive system lacks archive evidence")
        without_evidence = {
            key: copy.deepcopy(value)
            for key, value in observed.items()
            if key != "archive_evidence"
        }
        evidence = observed["archive_evidence"]
        if without_evidence != expected or type(evidence) is not dict:
            raise ValueError(f"Phase-1 archive system {index} was spliced")
        archive = evidence.get("archive")
        expected_checksum = expected["destination_object"]["checksum"]["value"]
        expected_sha = base64.b64decode(expected_checksum, validate=True).hex()
        if (
            type(archive) is not dict
            or archive.get("size") != expected["destination_object"]["size"]
            or archive.get("sha256") != expected_sha
        ):
            raise ValueError(f"Phase-1 archive system {index} bytes changed")
    _validate_self_hash(inventory, name="Phase-1 archive inventory")

    storage_keys = {
        "schema_version",
        "protocol",
        "experiment_id",
        "outer_fold",
        "role",
        "regime",
        "archive_input_manifest_sha256",
        "archive_inventory_receipt_sha256",
        "dataset_manifest_sha256",
        "fold_manifest_sha256",
        "passage_index_sha256",
        "case_ids",
        "case_ids_sha256",
        "query_count",
        "query_ids_sha256",
        "passage_count",
        "passage_ids_sha256",
        "candidate_pools_sha256",
        "evaluation_contract_sha256",
        "bm25_index_arguments",
        "bm25_runtime",
        "bm25_replicas",
        "bm25_allocation_tree",
        "filesystem_before",
        "filesystem_after",
        "image_runtime",
        "receipt_sha256",
    }
    storage = _exact_object(
        bm25_storage, storage_keys, name="Phase-1 BM25 storage receipt"
    )
    if (
        storage["schema_version"] != 1
        or type(storage["schema_version"]) is not int
        or storage["protocol"] != PHASE1_STORAGE_PROTOCOL
        or storage["experiment_id"] != "arr_retrieval_cv_v1"
        or storage["outer_fold"] != completed["outer_fold"]
        or storage["role"] != "test"
        or storage["regime"] != "fold_global"
        or storage["archive_input_manifest_sha256"]
        != inventory["input_manifest_sha256"]
        or storage["archive_inventory_receipt_sha256"]
        != inventory["receipt_sha256"]
        or storage["dataset_manifest_sha256"]
        != completed["training_plan"]["study"]["dataset_manifest_sha256"]
        or storage["fold_manifest_sha256"] != EXPECTED_FOLD_MANIFEST_SHA256
        or type(storage["bm25_replicas"]) is not list
        or len(storage["bm25_replicas"]) != 2
        or type(storage["bm25_allocation_tree"]) is not dict
        or type(storage["filesystem_before"]) is not dict
        or type(storage["filesystem_after"]) is not dict
    ):
        raise ValueError("Phase-1 BM25 storage binding changed")
    replicas = storage["bm25_replicas"]
    tree_hash = storage["bm25_allocation_tree"].get("allocation_tree_sha256")
    if (
        replicas[0].get("ordinal") != 1
        or replicas[1].get("ordinal") != 2
        or replicas[0].get("allocation_tree_sha256") != tree_hash
        or replicas[1].get("allocation_tree_sha256") != tree_hash
        or type(storage["bm25_allocation_tree"].get("allocated_bytes")) is not int
        or storage["bm25_allocation_tree"]["allocated_bytes"] < 1
    ):
        raise ValueError("Phase-1 BM25 replica allocation changed")
    _validate_self_hash(storage, name="Phase-1 BM25 storage receipt")

    artifact_keys = {
        "schema_version",
        "protocol",
        "experiment_id",
        "outer_fold",
        "archive_input_manifest_sha256",
        "archive_inventory_receipt_sha256",
        "bm25_storage_receipt_sha256",
        "files",
        "artifact_manifest_sha256",
    }
    artifact = _exact_object(
        artifact_manifest, artifact_keys, name="Phase-1 artifact manifest"
    )
    artifact_payload = {
        key: copy.deepcopy(value)
        for key, value in artifact.items()
        if key != "artifact_manifest_sha256"
    }
    files = artifact["files"]
    if (
        artifact["schema_version"] != 1
        or type(artifact["schema_version"]) is not int
        or artifact["protocol"] != PHASE1_OUTPUT_PROTOCOL
        or artifact["experiment_id"] != "arr_retrieval_cv_v1"
        or artifact["outer_fold"] != completed["outer_fold"]
        or artifact["archive_input_manifest_sha256"]
        != inventory["input_manifest_sha256"]
        or artifact["archive_inventory_receipt_sha256"]
        != inventory["receipt_sha256"]
        or artifact["bm25_storage_receipt_sha256"] != storage["receipt_sha256"]
        or artifact["artifact_manifest_sha256"] != _document_sha256(artifact_payload)
        or type(files) is not list
        or [record.get("path") for record in files]
        != ["archive_inventory.json", "bm25_storage.json"]
    ):
        raise ValueError("Phase-1 artifact manifest binding changed")
    return copy.deepcopy(inventory), copy.deepcopy(storage), copy.deepcopy(artifact)


def _write_bytes_at(directory: Path, name: str, payload: bytes) -> None:
    if Path(name).name != name or not name:
        raise ValueError("Acquisition output name is not one basename")
    descriptor = os.open(
        directory / name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position:])
            if written < 1:
                raise RuntimeError("Acquisition write made no progress")
            position += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_fold_inventory_acquisition_receipt(
    value: object,
    *,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = validate_fold_inventory_terminal_receipt(
        copy.deepcopy(terminal_receipt),
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    preflight = validate_fold_inventory_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    archive_copy = validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt),
        completed_fold_evidence=completed_fold_evidence,
    )
    receipt = _exact_object(
        value, _PHASE1_ACQUISITION_KEYS, name="fold inventory acquisition receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != FOLD_INVENTORY_ACQUISITION_PROTOCOL
        or receipt["outer_fold"] != preflight["outer_fold"]
        or receipt["terminal_receipt_sha256"] != _document_sha256(terminal)
        or receipt["archive_copy_receipt_sha256"] != _document_sha256(archive_copy)
        or receipt["output_prefix"] != preflight["output_prefix"]
    ):
        raise ValueError("Fold inventory acquisition evidence binding changed")
    files = receipt["files"]
    expected_names = [
        "archive_inventory.json",
        "artifact_manifest.json",
        "bm25_storage.json",
    ]
    if (
        type(files) is not list
        or len(files) != 3
        or [record.get("path") for record in files] != expected_names
    ):
        raise ValueError("Fold inventory acquisition file inventory changed")
    file_by_name: dict[str, dict[str, Any]] = {}
    for record in files:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Fold inventory acquired-file schema changed")
        if type(record["size"]) is not int or not 0 < record["size"] <= 64 * 1024 * 1024:
            raise ValueError("Fold inventory acquired-file size changed")
        _exact_sha256(record["sha256"], name="acquired file SHA-256")
        file_by_name[record["path"]] = record
    remote = receipt["remote_objects"]
    if type(remote) is not list or len(remote) != 3:
        raise ValueError("Fold inventory remote-object inventory changed")
    observed_names: set[str] = set()
    observed_order: list[str] = []
    for record in remote:
        if type(record) is not dict or set(record) != {
            "bucket",
            "key",
            "version_id",
            "size",
            "etag",
            "sha256",
            "encryption",
        }:
            raise ValueError("Fold inventory remote-object schema changed")
        key = _exact_string(record["key"], name="Phase-1 output key")
        prefix = preflight["output_prefix"] + "evidence/"
        if not key.startswith(prefix):
            raise ValueError("Fold inventory remote object left its output prefix")
        name = key.removeprefix(prefix)
        if name not in file_by_name or name in observed_names:
            raise ValueError("Fold inventory remote object coverage changed")
        expected = file_by_name[name]
        if (
            record["bucket"] != preflight["request"]["ProcessingOutputConfig"][
                "Outputs"
            ][0]["S3Output"]["S3Uri"].split("/", 3)[2]
            or record["size"] != expected["size"]
            or record["sha256"] != expected["sha256"]
            or _ETAG.fullmatch(_exact_string(record["etag"], name="output ETag"))
            is None
            or record["encryption"]
            != {
                "algorithm": "aws:kms",
                "kms_key_id": preflight["request"]["ProcessingOutputConfig"][
                    "KmsKeyId"
                ],
                "bucket_key_enabled": True,
            }
        ):
            raise ValueError("Fold inventory remote-object identity changed")
        _exact_string(record["version_id"], name="output VersionId")
        observed_names.add(name)
        observed_order.append(name)
    if observed_order != expected_names:
        raise ValueError("Fold inventory remote-object order changed")
    for field in (
        "archive_inventory_receipt_sha256",
        "bm25_storage_receipt_sha256",
        "artifact_manifest_sha256",
    ):
        _exact_sha256(receipt[field], name=f"acquisition.{field}")
    _validate_self_hash(receipt, name="fold inventory acquisition receipt")
    return copy.deepcopy(receipt)


def acquire_fold_inventory_once(
    clients: aws.AwsClients,
    *,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Acquire exactly the three compact Phase-1 evidence files once."""

    terminal = validate_fold_inventory_terminal_receipt(
        copy.deepcopy(terminal_receipt),
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    preflight = validate_fold_inventory_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    archive_copy = validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt), completed_fold_evidence=completed
    )
    account = preflight["account_id"]
    bucket = completed["training_plan"]["infrastructure"]["artifact_bucket"]
    output_prefix = preflight["output_prefix"]
    history = _list_prefix_history(
        clients.s3,
        bucket=bucket,
        prefix=output_prefix,
        expected_bucket_owner=account,
    )
    if history["delete_markers"] or len(history["versions"]) != 3:
        raise RuntimeError("Phase-1 output prefix must have exactly three versions")
    expected_paths = {
        "evidence/archive_inventory.json": "archive_inventory.json",
        "evidence/bm25_storage.json": "bm25_storage.json",
        "evidence/artifact_manifest.json": "artifact_manifest.json",
    }
    by_relative: dict[str, dict[str, Any]] = {}
    for version in history["versions"]:
        key = version.get("Key")
        if type(key) is not str or not key.startswith(output_prefix):
            raise RuntimeError("Phase-1 output key escaped its prefix")
        relative = key.removeprefix(output_prefix)
        if relative not in expected_paths or relative in by_relative:
            raise RuntimeError("Phase-1 output key inventory changed")
        if (
            version.get("IsLatest") is not True
            or type(version.get("VersionId")) is not str
            or type(version.get("Size")) is not int
            or not 0 < version["Size"] <= 64 * 1024 * 1024
            or _ETAG.fullmatch(version.get("ETag", "")) is None
        ):
            raise RuntimeError("Phase-1 output version identity changed")
        by_relative[relative] = version
    if set(by_relative) != set(expected_paths):
        raise RuntimeError("Phase-1 output file coverage changed")
    kms_key = preflight["request"]["ProcessingOutputConfig"]["KmsKeyId"]
    payloads: dict[str, bytes] = {}
    remote_objects: list[dict[str, Any]] = []
    for relative in sorted(expected_paths):
        version = by_relative[relative]
        key = version["Key"]
        version_id = version["VersionId"]
        head = clients.s3.head_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=account,
        )
        if (
            head.get("ContentLength") != version["Size"]
            or head.get("ETag") != version["ETag"]
            or head.get("VersionId") != version_id
            or head.get("ServerSideEncryption") != "aws:kms"
            or head.get("SSEKMSKeyId") != kms_key
            or head.get("BucketKeyEnabled") is not True
        ):
            raise RuntimeError("Phase-1 output object metadata changed")
        response = clients.s3.get_object(
            Bucket=bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=account,
        )
        body = response.get("Body")
        if body is None:
            raise RuntimeError("Phase-1 output GetObject omitted body")
        payload = body.read(version["Size"] + 1)
        if type(payload) is not bytes or len(payload) != version["Size"] or body.read(1) != b"":
            raise RuntimeError("Phase-1 compact output size changed")
        name = expected_paths[relative]
        payloads[name] = payload
        remote_objects.append(
            {
                "bucket": bucket,
                "key": key,
                "version_id": version_id,
                "size": len(payload),
                "etag": version["ETag"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "encryption": {
                    "algorithm": "aws:kms",
                    "kms_key_id": kms_key,
                    "bucket_key_enabled": True,
                },
            }
        )
    documents = {
        name: _load_compact_json_bytes(payload, name=f"Phase-1 {name}")
        for name, payload in payloads.items()
    }
    inventory, storage, artifact = _validate_phase1_documents(
        archive_inventory=documents["archive_inventory.json"],
        bm25_storage=documents["bm25_storage.json"],
        artifact_manifest=documents["artifact_manifest.json"],
        archive_copy=archive_copy,
        completed=completed,
    )
    for record in artifact["files"]:
        payload = payloads[record["path"]]
        if (
            record.get("size") != len(payload)
            or record.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("Phase-1 artifact manifest file hash changed")

    final_history = _list_prefix_history(
        clients.s3,
        bucket=bucket,
        prefix=output_prefix,
        expected_bucket_owner=account,
    )
    if final_history != history:
        raise RuntimeError("Phase-1 output prefix changed during acquisition")

    output_dir = _canonical_absolute(output_dir, name="Phase-1 acquisition output")
    parent = _real_directory(output_dir.parent, name="Phase-1 acquisition parent")
    incomplete = output_dir.with_name(f".{output_dir.name}.incomplete")
    if output_dir.exists() or output_dir.is_symlink() or incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError("Phase-1 acquisition output must be initially absent")
    os.mkdir(incomplete, mode=0o700)
    files = [
        {
            "path": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(payloads.items())
    ]
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_INVENTORY_ACQUISITION_PROTOCOL,
            "outer_fold": completed["outer_fold"],
            "terminal_receipt_sha256": _document_sha256(terminal),
            "archive_copy_receipt_sha256": _document_sha256(archive_copy),
            "output_prefix": output_prefix,
            "remote_objects": remote_objects,
            "files": files,
            "archive_inventory_receipt_sha256": inventory["receipt_sha256"],
            "bm25_storage_receipt_sha256": storage["receipt_sha256"],
            "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        }
    )
    receipt = validate_fold_inventory_acquisition_receipt(
        receipt,
        terminal_receipt=terminal,
        preflight_receipt=preflight,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive_copy,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    try:
        for name, payload in payloads.items():
            _write_bytes_at(incomplete, name, payload)
        _write_bytes_at(
            incomplete,
            "acquisition_receipt.json",
            strict_config.canonical_json_bytes(receipt),
        )
        descriptor = os.open(incomplete, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(
            parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        try:
            _rename_no_replace(parent_descriptor, incomplete.name, output_dir.name)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        # Preserve the incomplete tree as permanent evidence of an ambiguous
        # acquisition; there is deliberately no cleanup or resume path.
        raise
    return receipt


def _round_allocation(size: int, fragment_size: int) -> int:
    if type(size) is not int or size < 0:
        raise ValueError("Allocation size must be one non-negative exact integer")
    if type(fragment_size) is not int or fragment_size < 512:
        raise ValueError("Filesystem fragment size is invalid")
    return ((size + fragment_size - 1) // fragment_size) * fragment_size


def _load_phase1_acquisition_files(
    acquisition_dir: Path,
    acquisition: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    root = _real_directory(acquisition_dir, name="Phase-1 acquisition directory")
    expected_names = {
        "acquisition_receipt.json",
        "archive_inventory.json",
        "artifact_manifest.json",
        "bm25_storage.json",
    }
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("Phase-1 acquisition directory inventory changed")
    for path in root.iterdir():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Phase-1 acquisition contains an unsafe file")
    saved, _ = strict_config.load_canonical_json_object(
        root / "acquisition_receipt.json"
    )
    if _canonical_bytes(saved) != _canonical_bytes(acquisition):
        raise ValueError("Saved Phase-1 acquisition receipt changed")
    expected_files = {record["path"]: record for record in acquisition["files"]}
    documents: dict[str, dict[str, Any]] = {}
    for name, record in expected_files.items():
        path = root / name
        payload = path.read_bytes()
        if (
            len(payload) != record["size"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Phase-1 acquired file changed: {name}")
        documents[name] = _load_compact_json_bytes(payload, name=f"Phase-1 {name}")
    return documents


def validate_fold_storage_proof(value: object) -> dict[str, Any]:
    proof = _exact_object(value, _STORAGE_PROOF_KEYS, name="fold storage proof")
    if (
        type(proof["schema_version"]) is not int
        or proof["schema_version"] != 1
        or proof["protocol"] != FOLD_STORAGE_PROOF_PROTOCOL
        or type(proof["outer_fold"]) is not int
        or proof["outer_fold"] not in range(5)
        or proof["volume_size_gb"] != 100
        or type(proof["filesystem_capacity_bytes"]) is not int
        or proof["filesystem_capacity_bytes"]
        < G5_12XLARGE_LOCAL_FILESYSTEM_MIN_BYTES
        or proof["filesystem_capacity_bytes"]
        > G5_12XLARGE_LOCAL_NVME_NOMINAL_BYTES
        or type(proof["filesystem_available_bytes"]) is not int
        or proof["filesystem_available_bytes"] < 1
        or proof["filesystem_available_bytes"]
        > proof["filesystem_capacity_bytes"]
        or proof["fits"] is not True
    ):
        raise ValueError("Fold storage proof identity changed")
    for field in (
        "archive_copy_receipt_sha256",
        "static_staging_receipt_sha256",
        "inventory_acquisition_receipt_sha256",
    ):
        _exact_sha256(proof[field], name=f"storage proof.{field}")
    components = proof["components"]
    expected_components = {
        "static_phase2_inputs_allocated_upper_bound",
        "phase1_evidence_input_allocated_upper_bound",
        "phase2_control_bundle_allocated_upper_bound",
        "controlled_artifact_extraction_allocated_upper_bound",
        "bm25_index_allocated_bytes",
        "phase2_output_reserve_bytes",
        "safety_reserve_bytes",
    }
    if type(components) is not dict or set(components) != expected_components:
        raise ValueError("Fold storage proof component schema changed")
    for name, amount in components.items():
        if type(amount) is not int or amount < 0:
            raise ValueError(f"Fold storage component {name} is invalid")
    required = sum(components.values())
    if (
        proof["required_additional_bytes"] != required
        or proof["remaining_bytes"]
        != proof["filesystem_available_bytes"] - required
        or proof["remaining_bytes"] <= 0
    ):
        raise ValueError("Fold storage proof arithmetic changed")
    _validate_self_hash(proof, name="fold storage proof")
    return copy.deepcopy(proof)


def build_fold_storage_proof(
    *,
    acquisition_receipt: Mapping[str, Any],
    acquisition_dir: Path,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    overlay_publication_receipt: Mapping[str, Any],
    phase2_control_file_sizes: Sequence[int],
    phase2_output_reserve_bytes: int,
    safety_reserve_bytes: int,
) -> dict[str, Any]:
    """Prove fit on the measured fixed NVMe store of the frozen g5.12xlarge."""

    acquisition = validate_fold_inventory_acquisition_receipt(
        copy.deepcopy(acquisition_receipt),
        terminal_receipt=terminal_receipt,
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        overlay_publication_receipt=overlay_publication_receipt,
    )
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    archive_copy = validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt), completed_fold_evidence=completed
    )
    static_staging = validate_static_evaluation_staging_receipt(
        copy.deepcopy(static_staging_receipt), completed_fold_evidence=completed
    )
    documents = _load_phase1_acquisition_files(acquisition_dir, acquisition)
    inventory, storage, _ = _validate_phase1_documents(
        archive_inventory=documents["archive_inventory.json"],
        bm25_storage=documents["bm25_storage.json"],
        artifact_manifest=documents["artifact_manifest.json"],
        archive_copy=archive_copy,
        completed=completed,
    )
    if (
        acquisition["archive_inventory_receipt_sha256"]
        != inventory["receipt_sha256"]
        or acquisition["bm25_storage_receipt_sha256"]
        != storage["receipt_sha256"]
    ):
        raise ValueError("Storage proof acquisition documents changed")
    filesystem = storage["filesystem_before"]
    required_filesystem_keys = {
        "block_size",
        "fragment_size",
        "blocks",
        "blocks_free",
        "blocks_available",
        "capacity_bytes",
        "free_bytes",
        "available_bytes",
    }
    if type(filesystem) is not dict or set(filesystem) != required_filesystem_keys:
        raise ValueError("Phase-1 filesystem measurement schema changed")
    for name, amount in filesystem.items():
        if type(amount) is not int or amount < 0:
            raise ValueError(f"Phase-1 filesystem measurement {name} changed")
    fragment = filesystem["fragment_size"]
    if (
        filesystem["capacity_bytes"] != filesystem["blocks"] * fragment
        or filesystem["free_bytes"] != filesystem["blocks_free"] * fragment
        or filesystem["available_bytes"]
        != filesystem["blocks_available"] * fragment
        or not (
            0
            <= filesystem["blocks_available"]
            <= filesystem["blocks_free"]
            <= filesystem["blocks"]
        )
        or not (
            0
            <= filesystem["available_bytes"]
            <= filesystem["free_bytes"]
            <= filesystem["capacity_bytes"]
        )
        or not G5_12XLARGE_LOCAL_FILESYSTEM_MIN_BYTES
        <= filesystem["capacity_bytes"]
        <= G5_12XLARGE_LOCAL_NVME_NOMINAL_BYTES
    ):
        raise ValueError("Phase-1 filesystem capacity/free arithmetic changed")
    if (
        type(phase2_control_file_sizes) not in {list, tuple}
        or not phase2_control_file_sizes
        or any(type(size) is not int or size < 1 for size in phase2_control_file_sizes)
    ):
        raise ValueError("Phase-2 control-bundle sizes must be explicit positive integers")
    if (
        type(phase2_output_reserve_bytes) is not int
        or phase2_output_reserve_bytes < 1
        or type(safety_reserve_bytes) is not int
        or safety_reserve_bytes < 1
    ):
        raise ValueError("Phase-2 output and safety reserves must be explicit positive integers")

    static_sizes = [
        record["size"]
        for asset in static_staging["assets"]
        if asset["name"] != "control"
        for record in asset["files"]
    ]
    phase1_sizes = [record["size"] for record in acquisition["files"]]
    extracted = 0
    for system in inventory["systems"]:
        members = system["archive_evidence"].get("members")
        if type(members) is not list or not members:
            raise ValueError("Archive inventory lacks extraction members")
        for member in members:
            if type(member) is not dict or member.get("kind") not in {
                "file",
                "directory",
            }:
                raise ValueError("Archive extraction member changed")
            size = member.get("size")
            if type(size) is not int or size < 0:
                raise ValueError("Archive extraction member size changed")
            extracted += _round_allocation(
                fragment if member["kind"] == "directory" else size,
                fragment,
            )
    bm25_bytes = storage["bm25_allocation_tree"].get("allocated_bytes")
    if type(bm25_bytes) is not int or bm25_bytes < 1:
        raise ValueError("Measured BM25 allocation changed")
    components = {
        "static_phase2_inputs_allocated_upper_bound": sum(
            _round_allocation(size, fragment) for size in static_sizes
        ),
        "phase1_evidence_input_allocated_upper_bound": sum(
            _round_allocation(size, fragment) for size in phase1_sizes
        ),
        "phase2_control_bundle_allocated_upper_bound": sum(
            _round_allocation(size, fragment)
            for size in phase2_control_file_sizes
        ),
        "controlled_artifact_extraction_allocated_upper_bound": extracted,
        "bm25_index_allocated_bytes": bm25_bytes,
        "phase2_output_reserve_bytes": phase2_output_reserve_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
    }
    required = sum(components.values())
    available = filesystem["available_bytes"]
    remaining = available - required
    if remaining <= 0:
        raise RuntimeError(
            "Measured Phase-2 high-water bound does not fit the g5.12xlarge filesystem: "
            f"required_additional={required}, available={available}"
        )
    proof = _seal(
        {
            "schema_version": 1,
            "protocol": FOLD_STORAGE_PROOF_PROTOCOL,
            "outer_fold": completed["outer_fold"],
            "archive_copy_receipt_sha256": _document_sha256(archive_copy),
            "static_staging_receipt_sha256": _document_sha256(static_staging),
            "inventory_acquisition_receipt_sha256": _document_sha256(acquisition),
            "volume_size_gb": completed["training_plan"]["infrastructure"][
                "processing_volume_size_gb"
            ],
            "filesystem_capacity_bytes": filesystem["capacity_bytes"],
            "filesystem_available_bytes": available,
            "components": components,
            "required_additional_bytes": required,
            "remaining_bytes": remaining,
            "fits": True,
        }
    )
    return validate_fold_storage_proof(proof)
