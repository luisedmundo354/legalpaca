"""Strict acquisition of completed retrieval-CV SageMaker model artifacts.

The acquisition boundary is deliberately narrower than a general S3 downloader.
It accepts only a completed determinism-smoke launch chain, proves that the run
prefix has exactly one historical object version and no delete markers, reads
that exact version, and publishes one locally committed evidence bundle.  It
has no retry, resume, reconciliation, archive-format fallback, or overwrite
path.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import errno
import gzip
import hashlib
import math
import os
import re
import shutil
import stat
import tarfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

from ...retriever.determinism_artifacts import (
    DeterminismSmokeArtifactExpectation,
    ValidatedDeterminismSmokeArtifact,
    validate_determinism_smoke_artifact,
)

from . import aws
from . import config as strict_config
from . import manifest
from . import training_aws
from . import training_launch


DETERMINISM_SMOKE_ACQUISITION_PROTOCOL = (
    "retrieval_cv_determinism_smoke_acquisition_v1"
)
ACQUISITION_RECEIPT_NAME = "acquisition_receipt.json"
ARCHIVE_NAME = "model.tar.gz"
ARTIFACT_DIRECTORY_NAME = "artifact"

MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_DECOMPRESSED_BYTES = 32 * 1024**3
MAX_TREE_BYTES = 32 * 1024**3
MAX_PHYSICAL_MEMBERS = 4_096
MAX_LOGICAL_MEMBERS = 4_096
MAX_PAX_PAYLOAD_BYTES = 4_096
MAX_PATH_BYTES = 512
_COPY_CHUNK_BYTES = 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OWNER_ID = re.compile(r"[0-9a-f]{64}\Z")
_MULTIPART_ETAG = re.compile(r'"(?P<digest>[0-9a-f]{32})-(?P<parts>[1-9][0-9]*)"\Z')
_COMPOSITE_CRC32 = re.compile(
    r"(?P<value>[A-Za-z0-9+/]{6}==)-(?P<parts>[1-9][0-9]*)\Z"
)
_KMS_KEY_ARN = re.compile(
    r"arn:aws:kms:(?P<region>[a-z]{2}(?:-gov)?-[a-z]+-[0-9]):"
    r"(?P<account>[0-9]{12}):key/(?P<key>[A-Za-z0-9-]+)\Z"
)
_PAX_PHYSICAL_NAME = re.compile(r"\./PaxHeaders\.X/[A-Za-z0-9._-]+\Z")
_PAX_TIME = re.compile(r"[1-9][0-9]*\.[0-9]{7}\Z")
_PAX_KEYS = frozenset(
    {"atime", "ctime", "mtime", "LIBARCHIVE.creationtime"}
)

_TOP_LEVEL_RECEIPT_KEYS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "run_id",
    "job_name",
    "job_arn",
    "model_artifact_s3_uri",
    "launch_evidence",
    "evidence_chain",
    "remote_object",
    "local_bundle",
    "artifact",
    "receipt_sha256",
}
_LAUNCH_EVIDENCE_KEYS = {
    "preflight_receipt",
    "submission_receipt",
    "terminal_receipt",
}
_EVIDENCE_CHAIN_KEYS = {
    "plan_sha256",
    "staging_receipt_sha256",
    "preflight_receipt_sha256",
    "submission_receipt_sha256",
    "status_receipt_sha256",
    "terminal_receipt_sha256",
    "request_receipt_sha256",
    "request_sha256",
}
_REMOTE_OBJECT_KEYS = {
    "bucket",
    "key",
    "s3_uri",
    "version_id",
    "size",
    "sha256",
    "etag",
    "last_modified",
    "storage_class",
    "owner_id",
    "multipart_part_count",
    "checksum",
    "encryption",
    "content_type",
    "metadata",
}
_CHECKSUM_KEYS = {"algorithm", "type", "value"}
_ENCRYPTION_KEYS = {"algorithm", "kms_key_id", "bucket_key_enabled"}
_LOCAL_BUNDLE_KEYS = {
    "bundle_root",
    "artifact_root",
    "archive_path",
    "receipt_path",
    "archive",
    "gzip",
    "tar",
}
_ARCHIVE_KEYS = {"size", "sha256"}
_GZIP_KEYS = {"member_count", "uncompressed_size"}
_TAR_KEYS = {
    "physical_member_count",
    "logical_member_count",
    "file_count",
    "directory_count",
    "file_bytes",
    "pax_header_count",
    "member_inventory_sha256",
}
_ARTIFACT_KEYS = {
    "artifact_manifest_sha256",
    "file_count",
    "total_size",
    "inventory_sha256",
    "files",
    "identity",
}
_FILE_RECORD_KEYS = {"path", "size", "sha256"}
_ArchiveFileIdentity = tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True)
class ValidatedDeterminismSmokeAcquisition:
    """Fully cross-bound local acquisition and launch evidence."""

    receipt: dict[str, Any]
    receipt_path: Path
    bundle_root: Path
    artifact_root: Path
    archive_sha256: str
    archive_size: int
    inventory_sha256: str
    file_count: int
    total_size: int
    remote_object: dict[str, Any]
    request_receipt: dict[str, Any]
    preflight_receipt: dict[str, Any]
    submission_receipt: dict[str, Any]
    terminal_receipt: dict[str, Any]
    validated_artifact: ValidatedDeterminismSmokeArtifact


@dataclass(frozen=True)
class _ArchiveSnapshot:
    path: Path
    descriptor: int
    identity: _ArchiveFileIdentity

    def assert_stable(self) -> None:
        try:
            descriptor_metadata = os.fstat(self.descriptor)
        except OSError as error:
            raise RuntimeError("Archive snapshot descriptor became invalid") from error
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_nlink != 1
            or _regular_file_identity(descriptor_metadata) != self.identity
        ):
            raise RuntimeError("Archive snapshot descriptor identity changed")
        _assert_regular_path_identity(self.path, self.identity)

    @contextmanager
    def reader(self) -> Iterator[BinaryIO]:
        self.assert_stable()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        source = os.fdopen(self.descriptor, "rb", buffering=0, closefd=False)
        try:
            yield source
        finally:
            source.close()
            self.assert_stable()


def _regular_file_identity(metadata: os.stat_result) -> _ArchiveFileIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_nlink,
    )


def _assert_regular_path_identity(
    path: Path,
    expected: _ArchiveFileIdentity,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError("Archive snapshot path identity disappeared") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or _regular_file_identity(metadata) != expected
    ):
        raise RuntimeError("Archive snapshot path identity changed")


@contextmanager
def _open_archive_snapshot(path: Path) -> Iterator[_ArchiveSnapshot]:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(
                "Archive snapshot must be one singly-linked exact regular file"
            )
        snapshot = _ArchiveSnapshot(
            path=path,
            descriptor=descriptor,
            identity=_regular_file_identity(metadata),
        )
        snapshot.assert_stable()
        try:
            yield snapshot
        finally:
            snapshot.assert_stable()
    finally:
        os.close(descriptor)


def _require_plain_json(value: object, *, name: str) -> None:
    def visit(current: object, path: str) -> None:
        if type(current) is dict:
            for key, nested in current.items():
                if type(key) is not str:
                    raise TypeError(f"{path} contains a non-string key")
                visit(nested, f"{path}.{key}")
            return
        if type(current) is list:
            for index, nested in enumerate(current):
                visit(nested, f"{path}[{index}]")
            return
        if current is None or type(current) in {str, bool, int}:
            return
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError(f"{path} contains a non-finite float")
            return
        raise TypeError(f"{path} contains a non-JSON type")

    visit(value, name)


def _exact_object(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one non-empty exact string")
    return value


def _exact_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be one positive exact integer")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _seal_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    _require_plain_json(payload, name="acquisition receipt payload")
    if "receipt_sha256" in payload:
        raise ValueError("Receipt payload already contains its self-hash")
    receipt = copy.deepcopy(payload)
    receipt["receipt_sha256"] = _document_sha256(payload)
    return receipt


def _validate_self_hash(receipt: Mapping[str, Any]) -> None:
    actual = _exact_sha256(
        receipt["receipt_sha256"], name="acquisition.receipt_sha256"
    )
    payload = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    if actual != _document_sha256(payload):
        raise ValueError("Acquisition receipt self-hash changed")


def _normalize_datetime(value: object, *, name: str) -> str:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise TypeError(f"{name} must be one timezone-aware datetime")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: object, *, name: str) -> str:
    text = _exact_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is not a canonical UTC timestamp") from error
    canonical = (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    if text != canonical:
        raise ValueError(f"{name} is not a canonical UTC timestamp")
    return text


def _validated_plan_staging(
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(training_plan) is not dict:
        raise TypeError("training_plan must be one exact object")
    if type(staging_receipt) is not dict:
        raise TypeError("staging_receipt must be one exact object")
    plan = manifest.validate_dry_manifest(copy.deepcopy(training_plan))
    execution = plan["execution"]
    if (
        execution["status"] != "ready"
        or execution["submittable"] is not True
        or execution["blockers"] != []
    ):
        raise ValueError("Training plan is not exactly ready and submittable")
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(staging_receipt), training_plan=plan
    )
    return plan, staged


def _find_smoke_run(plan: Mapping[str, Any], run_id: object) -> dict[str, Any]:
    selected = _exact_string(run_id, name="run_id")
    matches = [
        run
        for run in (*plan["controlled_runs"], *plan["auxiliary_runs"])
        if run["run_id"] == selected
    ]
    if len(matches) != 1 or matches[0]["kind"] != manifest.SMOKE_KIND:
        raise ValueError("Acquisition requires one planned determinism-smoke run")
    return matches[0]


def _validate_launch_chain(
    *,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = training_launch.validate_training_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        training_plan=plan,
        staging_receipt=staged,
    )
    submission = training_launch.validate_training_submission_receipt(
        copy.deepcopy(submission_receipt),
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight,
    )
    terminal = training_launch.validate_training_terminal_receipt(
        copy.deepcopy(terminal_receipt),
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight,
        submission_receipt=submission,
    )
    if (
        terminal["terminal_status"] != "Completed"
        or terminal["succeeded"] is not True
        or terminal["failure_reason"] is not None
    ):
        raise ValueError("Artifact acquisition requires exact successful terminal evidence")
    run = _find_smoke_run(plan, terminal["run_id"])
    request_receipt = preflight["request_receipt"]
    request = request_receipt["request"]
    if (
        submission["run_id"] != run["run_id"]
        or terminal["run_id"] != run["run_id"]
        or request_receipt["run_id"] != run["run_id"]
        or terminal["job_name"] != run["job_name"]
        or terminal["job_arn"] != submission["job_arn"]
    ):
        raise ValueError("Artifact launch coordinates changed across the receipt chain")
    expected_uri = (
        f"{request['OutputDataConfig']['S3OutputPath']}/"
        f"{request['TrainingJobName']}/output/{ARCHIVE_NAME}"
    )
    if terminal["model_artifact_s3_uri"] != expected_uri:
        raise ValueError("Terminal model-artifact URI differs from the exact request")
    return preflight, submission, terminal, request_receipt


def _evidence_chain(
    *,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
    preflight: Mapping[str, Any],
    submission: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, str]:
    request_receipt = preflight["request_receipt"]
    return {
        "plan_sha256": _document_sha256(plan),
        "staging_receipt_sha256": _document_sha256(staged),
        "preflight_receipt_sha256": _document_sha256(preflight),
        "submission_receipt_sha256": _document_sha256(submission),
        "status_receipt_sha256": _document_sha256(terminal["status_receipt"]),
        "terminal_receipt_sha256": _document_sha256(terminal),
        "request_receipt_sha256": _document_sha256(request_receipt),
        "request_sha256": request_receipt["request_sha256"],
    }


def _expected_remote_coordinates(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    uri = terminal["model_artifact_s3_uri"]
    bucket, key = strict_config._s3_uri_coordinates(uri)
    expected_bucket = plan["infrastructure"]["artifact_bucket"]
    prefix = preflight["output_version_prefix"]
    if (
        bucket != expected_bucket
        or not prefix.endswith("/")
        or key != f"{prefix}{terminal['job_name']}/output/{ARCHIVE_NAME}"
    ):
        raise ValueError("Model artifact is outside its exact planned output prefix")
    return bucket, key, prefix, uri


def _list_exact_output_version(
    s3: object,
    *,
    bucket: str,
    key: str,
    prefix: str,
    account_id: str,
) -> dict[str, Any]:
    versions: list[dict[str, Any]] = []
    delete_markers: list[dict[str, Any]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    seen_markers: set[tuple[str, str]] = set()
    while True:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "ExpectedBucketOwner": account_id,
            "MaxKeys": 1000,
            "Prefix": prefix,
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
            raise RuntimeError("S3 output version listing identity changed")
        raw_versions = response.get("Versions", [])
        raw_markers = response.get("DeleteMarkers", [])
        if type(raw_versions) is not list or type(raw_markers) is not list:
            raise RuntimeError("S3 output history collections are malformed")
        versions.extend(raw_versions)
        delete_markers.extend(raw_markers)
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
            or marker in seen_markers
            or marker == (key_marker, version_marker)
        ):
            raise RuntimeError("S3 output version pagination did not advance")
        seen_markers.add(marker)
        key_marker, version_marker = marker
    if delete_markers or len(versions) != 1:
        raise RuntimeError(
            "Training output must have exactly one version and no delete markers"
        )
    version = _exact_object(
        versions[0],
        {
            "ChecksumAlgorithm",
            "ChecksumType",
            "ETag",
            "IsLatest",
            "Key",
            "LastModified",
            "Owner",
            "Size",
            "StorageClass",
            "VersionId",
        },
        name="S3 output version",
    )
    owner = _exact_object(version["Owner"], {"ID"}, name="S3 output owner")
    owner_id = _exact_string(owner["ID"], name="S3 output owner ID")
    etag = _exact_string(version["ETag"], name="S3 output ETag")
    etag_match = _MULTIPART_ETAG.fullmatch(etag)
    if etag_match is None or int(etag_match.group("parts")) < 2:
        raise ValueError("S3 output must have one exact multipart ETag")
    size = _exact_positive_int(version["Size"], name="S3 output size")
    version_id = _exact_string(version["VersionId"], name="S3 output VersionId")
    if (
        version["Key"] != key
        or version["IsLatest"] is not True
        or version["StorageClass"] != "STANDARD"
        or version["ChecksumAlgorithm"] != ["CRC32"]
        or version["ChecksumType"] != "COMPOSITE"
        or _OWNER_ID.fullmatch(owner_id) is None
        or size > MAX_ARCHIVE_BYTES
    ):
        raise ValueError("S3 output version left the exact acquisition contract")
    return {
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "size": size,
        "etag": etag,
        "last_modified": _normalize_datetime(
            version["LastModified"], name="S3 output LastModified"
        ),
        "storage_class": "STANDARD",
        "owner_id": owner_id,
        "multipart_part_count": int(etag_match.group("parts")),
    }


def _without_response_metadata(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError(f"{name} returned a non-object")
    result = {key: nested for key, nested in value.items() if key != "ResponseMetadata"}
    return result


def _head_exact_output_version(
    s3: object,
    *,
    version: Mapping[str, Any],
    account_id: str,
    region: str,
) -> dict[str, Any]:
    response = s3.head_object(
        Bucket=version["bucket"],
        Key=version["key"],
        VersionId=version["version_id"],
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=account_id,
    )
    head = _exact_object(
        _without_response_metadata(response, name="HeadObject"),
        {
            "AcceptRanges",
            "BucketKeyEnabled",
            "ChecksumCRC32",
            "ChecksumType",
            "ContentLength",
            "ContentType",
            "ETag",
            "LastModified",
            "Metadata",
            "SSEKMSKeyId",
            "ServerSideEncryption",
            "VersionId",
        },
        name="version-addressed HeadObject",
    )
    checksum = _exact_string(head["ChecksumCRC32"], name="HeadObject CRC32")
    checksum_match = _COMPOSITE_CRC32.fullmatch(checksum)
    kms_key = _exact_string(head["SSEKMSKeyId"], name="HeadObject KMS key")
    kms_match = _KMS_KEY_ARN.fullmatch(kms_key)
    if (
        checksum_match is None
        or int(checksum_match.group("parts")) != version["multipart_part_count"]
        or kms_match is None
        or kms_match.group("region") != region
        or kms_match.group("account") != account_id
        or head["AcceptRanges"] != "bytes"
        or head["BucketKeyEnabled"] is not True
        or head["ChecksumType"] != "COMPOSITE"
        or head["ContentLength"] != version["size"]
        or head["ContentType"] != "application/gzip"
        or head["ETag"] != version["etag"]
        or _normalize_datetime(head["LastModified"], name="HeadObject LastModified")
        != version["last_modified"]
        or head["Metadata"] != {}
        or head["ServerSideEncryption"] != "aws:kms"
        or head["VersionId"] != version["version_id"]
    ):
        raise ValueError("Version-addressed HeadObject metadata changed")
    return {
        **copy.deepcopy(version),
        "checksum": {
            "algorithm": "CRC32",
            "type": "COMPOSITE",
            "value": checksum,
        },
        "encryption": {
            "algorithm": "aws:kms",
            "kms_key_id": kms_key,
            "bucket_key_enabled": True,
        },
        "content_type": "application/gzip",
        "metadata": {},
    }


def _inspect_remote_output(
    s3: object,
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    bucket, key, prefix, uri = _expected_remote_coordinates(
        plan=plan, preflight=preflight, terminal=terminal
    )
    infrastructure = plan["infrastructure"]
    listed = _list_exact_output_version(
        s3,
        bucket=bucket,
        key=key,
        prefix=prefix,
        account_id=infrastructure["account_id"],
    )
    headed = _head_exact_output_version(
        s3,
        version=listed,
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    headed["s3_uri"] = uri
    return headed


def _get_response_metadata(
    response: object,
    *,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], object]:
    value = _without_response_metadata(response, name="GetObject")
    if "Body" not in value:
        raise RuntimeError("Version-addressed GetObject omitted its body")
    body = value.pop("Body")
    metadata = _exact_object(
        value,
        {
            "AcceptRanges",
            "BucketKeyEnabled",
            "ChecksumCRC32",
            "ChecksumType",
            "ContentLength",
            "ContentType",
            "ETag",
            "LastModified",
            "Metadata",
            "SSEKMSKeyId",
            "ServerSideEncryption",
            "VersionId",
        },
        name="version-addressed GetObject",
    )
    if (
        metadata["AcceptRanges"] != "bytes"
        or metadata["BucketKeyEnabled"] is not True
        or metadata["ChecksumCRC32"] != expected["checksum"]["value"]
        or metadata["ChecksumType"] != expected["checksum"]["type"]
        or metadata["ContentLength"] != expected["size"]
        or metadata["ContentType"] != expected["content_type"]
        or metadata["ETag"] != expected["etag"]
        or _normalize_datetime(metadata["LastModified"], name="GetObject LastModified")
        != expected["last_modified"]
        or metadata["Metadata"] != expected["metadata"]
        or metadata["SSEKMSKeyId"] != expected["encryption"]["kms_key_id"]
        or metadata["ServerSideEncryption"]
        != expected["encryption"]["algorithm"]
        or metadata["VersionId"] != expected["version_id"]
    ):
        raise ValueError("Version-addressed GetObject metadata changed")
    return metadata, body


def _download_exact_version(
    s3: object,
    *,
    remote: Mapping[str, Any],
    account_id: str,
    destination: Path,
) -> tuple[int, str]:
    response = s3.get_object(
        Bucket=remote["bucket"],
        Key=remote["key"],
        VersionId=remote["version_id"],
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=account_id,
    )
    _, body = _get_response_metadata(response, expected=remote)
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as target:
            while True:
                chunk = body.read(_COPY_CHUNK_BYTES)
                if type(chunk) is not bytes:
                    raise RuntimeError("Version-addressed GetObject returned non-bytes")
                if not chunk:
                    break
                size += len(chunk)
                if size > remote["size"]:
                    raise RuntimeError("Version-addressed GetObject exceeded listed size")
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if size != remote["size"]:
        raise RuntimeError("Version-addressed GetObject ended before listed size")
    return size, digest.hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected one regular non-symlink file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _archive_size_sha256(snapshot: _ArchiveSnapshot) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with snapshot.reader() as source:
        while True:
            chunk = source.read(_COPY_CHUNK_BYTES)
            if type(chunk) is not bytes:
                raise RuntimeError("Archive snapshot returned non-bytes")
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size != snapshot.identity[2]:
        raise RuntimeError("Archive snapshot size changed while hashing")
    return size, digest.hexdigest()


def _audit_single_gzip_snapshot(snapshot: _ArchiveSnapshot) -> dict[str, int]:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    output_size = 0
    reached_end = False
    with snapshot.reader() as source:
        while True:
            chunk = source.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            pending = chunk
            while pending:
                output = decompressor.decompress(pending, _COPY_CHUNK_BYTES)
                output_size += len(output)
                if output_size > MAX_DECOMPRESSED_BYTES:
                    raise ValueError("Gzip output exceeds the fixed smoke-artifact cap")
                pending = decompressor.unconsumed_tail
                if decompressor.eof:
                    if decompressor.unused_data or pending or source.read(1):
                        raise ValueError("Archive contains trailing bytes or another gzip member")
                    reached_end = True
                    break
            if reached_end:
                break
    if not reached_end or not decompressor.eof:
        raise ValueError("Archive is not one complete gzip member")
    if decompressor.flush():
        raise ValueError("Gzip decompressor retained unexpected output")
    return {"member_count": 1, "uncompressed_size": output_size}


def _audit_single_gzip(path: Path) -> dict[str, int]:
    with _open_archive_snapshot(path) as snapshot:
        return _audit_single_gzip_snapshot(snapshot)


def _read_exact(source: BinaryIO, count: int, *, name: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = source.read(remaining)
        if type(chunk) is not bytes:
            raise RuntimeError(f"{name} returned non-bytes")
        if not chunk:
            raise ValueError(f"{name} is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _tar_info(header: bytes, *, name: str) -> tarfile.TarInfo:
    try:
        return tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
    except (tarfile.HeaderError, UnicodeError) as error:
        raise ValueError(f"{name} header is invalid") from error


def _parse_pax_payload(payload: bytes) -> dict[str, str]:
    records: dict[str, str] = {}
    position = 0
    while position < len(payload):
        space = payload.find(b" ", position)
        if space < 0:
            raise ValueError("PAX record lacks an exact length prefix")
        raw_length = payload[position:space]
        if not raw_length or not raw_length.isdigit() or raw_length.startswith(b"0"):
            raise ValueError("PAX record length is non-canonical")
        length = int(raw_length)
        end = position + length
        if length < 5 or end > len(payload) or payload[end - 1 : end] != b"\n":
            raise ValueError("PAX record framing is invalid")
        record = payload[space + 1 : end - 1]
        raw_key, separator, raw_value = record.partition(b"=")
        if not raw_key or separator != b"=":
            raise ValueError("PAX record lacks one key/value separator")
        try:
            key = raw_key.decode("utf-8", errors="strict")
            value = raw_value.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("PAX record is not strict UTF-8") from error
        if key in records:
            raise ValueError("PAX header contains a duplicate key")
        records[key] = value
        position = end
    if set(records) != _PAX_KEYS or any(
        _PAX_TIME.fullmatch(value) is None for value in records.values()
    ):
        raise ValueError("PAX header left the exact time-only contract")
    return records


def _canonical_tar_path(member: tarfile.TarInfo, *, is_directory: bool) -> str:
    raw = member.name
    if type(raw) is not str or not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("TAR member path is empty or unsafe")
    normalized = raw[:-1] if is_directory and raw.endswith("/") else raw
    if (
        not normalized
        or len(normalized.encode("utf-8")) > MAX_PATH_BYTES
        or normalized.startswith("/")
        or normalized.startswith("./")
        or "//" in normalized
    ):
        raise ValueError("TAR member path is not canonical")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or path.as_posix() != normalized
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("TAR member path escapes or aliases the artifact root")
    return normalized


def _scan_tar_snapshot(
    snapshot: _ArchiveSnapshot,
    *,
    gzip_evidence: Mapping[str, Any],
    extraction_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if extraction_root is not None:
        if extraction_root.exists() or extraction_root.is_symlink():
            raise FileExistsError("Artifact extraction root must be absent")
        extraction_root.mkdir(mode=0o700)
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    declared_directories: set[str] = set()
    physical_count = 0
    logical_count = 0
    file_count = 0
    directory_count = 0
    file_bytes = 0
    stream_position = 0
    with snapshot.reader() as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
            while True:
                header = _read_exact(source, 512, name="TAR header")
                stream_position += 512
                if header == b"\0" * 512:
                    second = _read_exact(source, 512, name="TAR terminator")
                    stream_position += 512
                    if second != b"\0" * 512:
                        raise ValueError("TAR archive lacks two zero terminator blocks")
                    while True:
                        tail = source.read(_COPY_CHUNK_BYTES)
                        if type(tail) is not bytes:
                            raise RuntimeError("TAR zero padding returned non-bytes")
                        if not tail:
                            break
                        stream_position += len(tail)
                        if any(tail):
                            raise ValueError("TAR archive has nonzero trailing data")
                    break
                physical_count += 1
                if physical_count > MAX_PHYSICAL_MEMBERS:
                    raise ValueError("TAR physical-member cap exceeded")
                pax = _tar_info(header, name="PAX")
                if (
                    pax.type != tarfile.XHDTYPE
                    or _PAX_PHYSICAL_NAME.fullmatch(pax.name) is None
                    or type(pax.size) is not int
                    or pax.size < 1
                    or pax.size > MAX_PAX_PAYLOAD_BYTES
                    or pax.linkname
                ):
                    raise ValueError("Every logical member requires one exact local PAX header")
                pax_padded_size = math.ceil(pax.size / 512) * 512
                pax_block = _read_exact(source, pax_padded_size, name="PAX payload")
                stream_position += pax_padded_size
                if any(pax_block[pax.size :]):
                    raise ValueError("PAX payload padding is nonzero")
                _parse_pax_payload(pax_block[: pax.size])

                logical_header = _read_exact(source, 512, name="logical TAR header")
                stream_position += 512
                physical_count += 1
                if physical_count > MAX_PHYSICAL_MEMBERS:
                    raise ValueError("TAR physical-member cap exceeded")
                member = _tar_info(logical_header, name="logical TAR member")
                is_file = member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                is_directory = member.type == tarfile.DIRTYPE
                if (
                    not (is_file or is_directory)
                    or member.linkname
                    or getattr(member, "sparse", None) is not None
                ):
                    raise ValueError("TAR contains a link, special, sparse, or unknown member")
                path = _canonical_tar_path(member, is_directory=is_directory)
                if path in seen_paths:
                    raise ValueError("TAR contains a duplicate logical path")
                parent = PurePosixPath(path).parent
                if parent != PurePosixPath(".") and parent.as_posix() not in declared_directories:
                    raise ValueError("TAR member appears before its explicit parent directory")
                seen_paths.add(path)
                logical_count += 1
                if logical_count > MAX_LOGICAL_MEMBERS:
                    raise ValueError("TAR logical-member cap exceeded")
                destination = (
                    None if extraction_root is None else extraction_root / Path(path)
                )
                if is_directory:
                    if member.size != 0:
                        raise ValueError("TAR directory has nonzero data size")
                    declared_directories.add(path)
                    directory_count += 1
                    if destination is not None:
                        destination.mkdir(mode=0o700)
                    record = {
                        "kind": "directory",
                        "path": path,
                        "size": 0,
                        "sha256": None,
                    }
                else:
                    size = _exact_positive_int(member.size, name=f"TAR file {path} size")
                    file_count += 1
                    file_bytes += size
                    if file_bytes > MAX_TREE_BYTES:
                        raise ValueError("Extracted tree exceeds the fixed smoke-artifact cap")
                    digest = hashlib.sha256()
                    remaining = size
                    target = None
                    try:
                        if destination is not None:
                            target = destination.open("xb")
                        while remaining:
                            chunk = _read_exact(
                                source,
                                min(_COPY_CHUNK_BYTES, remaining),
                                name=f"TAR file {path}",
                            )
                            stream_position += len(chunk)
                            digest.update(chunk)
                            if target is not None:
                                target.write(chunk)
                            remaining -= len(chunk)
                        if target is not None:
                            target.flush()
                            os.fsync(target.fileno())
                    finally:
                        if target is not None:
                            target.close()
                    padding_size = (-size) % 512
                    if padding_size:
                        padding = _read_exact(
                            source, padding_size, name=f"TAR file {path} padding"
                        )
                        stream_position += padding_size
                        if any(padding):
                            raise ValueError("TAR file padding is nonzero")
                    record = {
                        "kind": "file",
                        "path": path,
                        "size": size,
                        "sha256": digest.hexdigest(),
                    }
                records.append(record)
    if stream_position != gzip_evidence["uncompressed_size"]:
        raise ValueError("Gzip and TAR uncompressed sizes disagree")
    if not records or file_count < 1:
        raise ValueError("TAR archive contains no artifact files")
    summary = {
        "physical_member_count": physical_count,
        "logical_member_count": logical_count,
        "file_count": file_count,
        "directory_count": directory_count,
        "file_bytes": file_bytes,
        "pax_header_count": logical_count,
        "member_inventory_sha256": _document_sha256(records),
    }
    return summary, records


def _scan_tar(
    archive_path: Path,
    *,
    gzip_evidence: Mapping[str, Any],
    extraction_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with _open_archive_snapshot(archive_path) as snapshot:
        return _scan_tar_snapshot(
            snapshot,
            gzip_evidence=gzip_evidence,
            extraction_root=extraction_root,
        )


def _read_archive_snapshot(
    archive_path: Path,
    *,
    extraction_root: Path | None,
    expected_size_sha256: tuple[int, str],
) -> tuple[
    int,
    str,
    dict[str, int],
    dict[str, Any],
    list[dict[str, Any]],
    _ArchiveFileIdentity,
]:
    with _open_archive_snapshot(archive_path) as snapshot:
        size, sha256 = _archive_size_sha256(snapshot)
        if (size, sha256) != expected_size_sha256:
            raise ValueError("Archive snapshot differs from its exact expected bytes")
        gzip_evidence = _audit_single_gzip_snapshot(snapshot)
        tar_evidence, records = _scan_tar_snapshot(
            snapshot,
            gzip_evidence=gzip_evidence,
            extraction_root=extraction_root,
        )
        return (
            size,
            sha256,
            gzip_evidence,
            tar_evidence,
            records,
            snapshot.identity,
        )


def _tar_file_inventory(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    files = [
        {
            "path": record["path"],
            "size": record["size"],
            "sha256": record["sha256"],
        }
        for record in records
        if record["kind"] == "file"
    ]
    return sorted(files, key=lambda record: record["path"])


def _artifact_expectation(
    *,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
    artifact_manifest_sha256: str,
) -> DeterminismSmokeArtifactExpectation:
    source = plan["sources"]
    return DeterminismSmokeArtifactExpectation(
        artifact_manifest_sha256=artifact_manifest_sha256,
        training_plan_sha256=_document_sha256(plan),
        training_staging_receipt_sha256=_document_sha256(staged),
        source_bundle_name=source["source_bundle_path"],
        source_bundle_size=source["source_bundle_size"],
        source_bundle_sha256=source["source_bundle_sha256"],
        source_bundle_inventory_sha256=source["source_inventory_sha256"],
        source_bundle_commit_epoch=source["commit_epoch"],
    )


def _artifact_payload(
    *,
    artifact_root: Path,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], ValidatedDeterminismSmokeArtifact]:
    manifest_path = artifact_root / "artifact_manifest.json"
    _, manifest_sha256 = _sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Extracted artifact commit-marker SHA-256 changed")
    expectation = _artifact_expectation(
        plan=plan,
        staged=staged,
        artifact_manifest_sha256=manifest_sha256,
    )
    validated = validate_determinism_smoke_artifact(
        artifact_root, expectation=expectation
    )
    files = [
        {"path": item.path, "size": item.size, "sha256": item.sha256}
        for item in validated.files
    ]
    if files != sorted(files, key=lambda item: item["path"]):
        raise RuntimeError("Artifact validator returned a non-canonical inventory")
    total_size = sum(item["size"] for item in files)
    return (
        {
            "artifact_manifest_sha256": manifest_sha256,
            "file_count": len(files),
            "total_size": total_size,
            "inventory_sha256": _document_sha256(files),
            "files": files,
            "identity": dataclasses.asdict(validated.identity),
        },
        validated,
    )


def _validate_absolute_path_text(value: object, *, name: str) -> Path:
    text = _exact_string(value, name=name)
    path = Path(text)
    if (
        not path.is_absolute()
        or path.as_posix() != text
        or ".." in path.parts
        or text.startswith("//")
    ):
        raise ValueError(f"{name} must be one normalized absolute path")
    return path


def _validate_output_paths(output_bundle: Path) -> tuple[Path, Path]:
    if not isinstance(output_bundle, Path):
        raise TypeError("output_bundle must be one pathlib.Path")
    if (
        not output_bundle.is_absolute()
        or output_bundle.resolve(strict=False) != output_bundle
        or output_bundle.as_posix().startswith("//")
    ):
        raise ValueError("Output bundle must be one canonical absolute path")
    parent = output_bundle.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("Output bundle parent must be one real canonical directory")
    incomplete = output_bundle.with_name(output_bundle.name + ".incomplete")
    if (
        output_bundle.exists()
        or output_bundle.is_symlink()
        or incomplete.exists()
        or incomplete.is_symlink()
    ):
        raise FileExistsError("Output bundle and sibling incomplete path must be absent")
    return output_bundle, incomplete


def _require_disk_space(parent: Path, *, archive_size: int) -> None:
    required = archive_size + MAX_DECOMPRESSED_BYTES
    if shutil.disk_usage(parent).free < required:
        raise OSError(
            "Artifact acquisition requires free space for the archive and fixed tree cap"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        if path.is_symlink():
            raise ValueError("Acquisition bundle contains a symlink before publication")
        _fsync_directory(path)
    _fsync_directory(root)


def _rename_no_replace(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic acquisition publication requires Linux renameat2")
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
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"Refusing to overwrite acquisition output: {target}")
        raise OSError(
            error_number,
            f"Atomic acquisition publication failed: {source} -> {target}",
        )


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("Owned acquisition path is no longer a real directory")
    return metadata.st_dev, metadata.st_ino


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = strict_config.canonical_json_bytes(receipt)
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _validate_remote_object_record(
    value: object,
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    record = _exact_object(value, _REMOTE_OBJECT_KEYS, name="remote_object")
    bucket, key, _, uri = _expected_remote_coordinates(
        plan=plan, preflight=preflight, terminal=terminal
    )
    size = _exact_positive_int(record["size"], name="remote_object.size")
    sha256 = _exact_sha256(record["sha256"], name="remote_object.sha256")
    etag = _exact_string(record["etag"], name="remote_object.etag")
    etag_match = _MULTIPART_ETAG.fullmatch(etag)
    parts = _exact_positive_int(
        record["multipart_part_count"], name="remote_object.multipart_part_count"
    )
    checksum = _exact_object(
        record["checksum"], _CHECKSUM_KEYS, name="remote_object.checksum"
    )
    checksum_value = _exact_string(
        checksum["value"], name="remote_object.checksum.value"
    )
    checksum_match = _COMPOSITE_CRC32.fullmatch(checksum_value)
    encryption = _exact_object(
        record["encryption"], _ENCRYPTION_KEYS, name="remote_object.encryption"
    )
    kms_key = _exact_string(
        encryption["kms_key_id"], name="remote_object.encryption.kms_key_id"
    )
    kms_match = _KMS_KEY_ARN.fullmatch(kms_key)
    infrastructure = plan["infrastructure"]
    if (
        record["bucket"] != bucket
        or record["key"] != key
        or record["s3_uri"] != uri
        or type(record["version_id"]) is not str
        or not record["version_id"]
        or size > MAX_ARCHIVE_BYTES
        or etag_match is None
        or parts < 2
        or int(etag_match.group("parts")) != parts
        or record["storage_class"] != "STANDARD"
        or type(record["owner_id"]) is not str
        or _OWNER_ID.fullmatch(record["owner_id"]) is None
        or checksum != {
            "algorithm": "CRC32",
            "type": "COMPOSITE",
            "value": checksum_value,
        }
        or checksum_match is None
        or int(checksum_match.group("parts")) != parts
        or encryption["algorithm"] != "aws:kms"
        or encryption["bucket_key_enabled"] is not True
        or kms_match is None
        or kms_match.group("region") != infrastructure["region"]
        or kms_match.group("account") != infrastructure["account_id"]
        or record["content_type"] != "application/gzip"
        or record["metadata"] != {}
    ):
        raise ValueError("Remote artifact record left the exact live wire contract")
    _validate_timestamp(record["last_modified"], name="remote_object.last_modified")
    _exact_string(record["version_id"], name="remote_object.version_id")
    del sha256
    return copy.deepcopy(record)


def _validate_file_records(value: object) -> tuple[list[dict[str, Any]], int]:
    if type(value) is not list or not value:
        raise ValueError("artifact.files must be one non-empty exact list")
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    total = 0
    for index, raw in enumerate(value):
        record = _exact_object(
            raw, _FILE_RECORD_KEYS, name=f"artifact.files[{index}]"
        )
        path = _exact_string(record["path"], name=f"artifact.files[{index}].path")
        if PurePosixPath(path).as_posix() != path or ".." in PurePosixPath(path).parts:
            raise ValueError("Artifact file inventory contains an unsafe path")
        size = _exact_positive_int(
            record["size"], name=f"artifact.files[{index}].size"
        )
        digest = _exact_sha256(
            record["sha256"], name=f"artifact.files[{index}].sha256"
        )
        records.append({"path": path, "size": size, "sha256": digest})
        paths.append(path)
        total += size
    if paths != sorted(set(paths)):
        raise ValueError("Artifact file inventory coverage/order changed")
    return records, total


def validate_determinism_smoke_acquisition_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    receipt_path: Path | None = None,
) -> ValidatedDeterminismSmokeAcquisition:
    """Validate the embedded launch chain and complete local acquisition bundle."""

    plan, staged = _validated_plan_staging(training_plan, staging_receipt)
    receipt = _exact_object(value, _TOP_LEVEL_RECEIPT_KEYS, name="acquisition receipt")
    _require_plain_json(receipt, name="acquisition receipt")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != DETERMINISM_SMOKE_ACQUISITION_PROTOCOL
    ):
        raise ValueError("Acquisition receipt protocol identity changed")
    launch = _exact_object(
        receipt["launch_evidence"],
        _LAUNCH_EVIDENCE_KEYS,
        name="acquisition.launch_evidence",
    )
    preflight, submission, terminal, request_receipt = _validate_launch_chain(
        plan=plan,
        staged=staged,
        preflight_receipt=launch["preflight_receipt"],
        submission_receipt=launch["submission_receipt"],
        terminal_receipt=launch["terminal_receipt"],
    )
    run = _find_smoke_run(plan, receipt["run_id"])
    if (
        receipt["attempt_id"] != plan["attempt"]["attempt_id"]
        or receipt["run_id"] != terminal["run_id"]
        or receipt["job_name"] != terminal["job_name"]
        or receipt["job_arn"] != terminal["job_arn"]
        or receipt["model_artifact_s3_uri"] != terminal["model_artifact_s3_uri"]
        or receipt["job_name"] != run["job_name"]
    ):
        raise ValueError("Acquisition launch identity changed")
    chain = _exact_object(
        receipt["evidence_chain"],
        _EVIDENCE_CHAIN_KEYS,
        name="acquisition.evidence_chain",
    )
    expected_chain = _evidence_chain(
        plan=plan,
        staged=staged,
        preflight=preflight,
        submission=submission,
        terminal=terminal,
    )
    if chain != expected_chain:
        raise ValueError("Acquisition evidence-chain hashes changed")
    for field in _EVIDENCE_CHAIN_KEYS:
        _exact_sha256(chain[field], name=f"acquisition.evidence_chain.{field}")
    remote = _validate_remote_object_record(
        receipt["remote_object"],
        plan=plan,
        preflight=preflight,
        terminal=terminal,
    )

    local = _exact_object(
        receipt["local_bundle"], _LOCAL_BUNDLE_KEYS, name="acquisition.local_bundle"
    )
    bundle_root = _validate_absolute_path_text(
        local["bundle_root"], name="local_bundle.bundle_root"
    )
    artifact_root = _validate_absolute_path_text(
        local["artifact_root"], name="local_bundle.artifact_root"
    )
    archive_path = _validate_absolute_path_text(
        local["archive_path"], name="local_bundle.archive_path"
    )
    embedded_receipt_path = _validate_absolute_path_text(
        local["receipt_path"], name="local_bundle.receipt_path"
    )
    if (
        artifact_root != bundle_root / ARTIFACT_DIRECTORY_NAME
        or archive_path != bundle_root / ARCHIVE_NAME
        or embedded_receipt_path != bundle_root / ACQUISITION_RECEIPT_NAME
    ):
        raise ValueError("Acquisition local paths left the exact bundle layout")
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError("Acquisition bundle root is not one real directory")
    if bundle_root.resolve(strict=True) != bundle_root:
        raise ValueError("Acquisition bundle root is not strict-resolved")
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("Acquisition artifact root is not one real directory")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("Acquisition archive is not one regular file")
    if embedded_receipt_path.is_symlink() or not embedded_receipt_path.is_file():
        raise ValueError("Acquisition receipt path is not one regular file")
    if {entry.name for entry in bundle_root.iterdir()} != {
        ARCHIVE_NAME,
        ARTIFACT_DIRECTORY_NAME,
        ACQUISITION_RECEIPT_NAME,
    }:
        raise ValueError("Acquisition bundle top-level inventory changed")
    bundle_identity = _path_identity(bundle_root)
    artifact_root_identity = _path_identity(artifact_root)
    selected_receipt_path = embedded_receipt_path
    if receipt_path is not None:
        if not isinstance(receipt_path, Path):
            raise TypeError("receipt_path must be one pathlib.Path")
        if (
            not receipt_path.is_absolute()
            or receipt_path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.resolve(strict=True) != embedded_receipt_path
            or receipt_path != embedded_receipt_path
        ):
            raise ValueError("Provided receipt_path is a detached or aliased copy")
        selected_receipt_path = receipt_path
    loaded, _ = strict_config.load_canonical_json_object(selected_receipt_path)
    if aws.canonical_json_bytes(loaded) != aws.canonical_json_bytes(receipt):
        raise ValueError("Acquisition receipt differs from its canonical bundle file")

    archive_record = _exact_object(
        local["archive"], _ARCHIVE_KEYS, name="local_bundle.archive"
    )
    archive_size = _exact_positive_int(
        archive_record["size"], name="local_bundle.archive.size"
    )
    archive_sha256 = _exact_sha256(
        archive_record["sha256"], name="local_bundle.archive.sha256"
    )
    gzip_record = _exact_object(local["gzip"], _GZIP_KEYS, name="local_bundle.gzip")
    if (
        type(gzip_record["member_count"]) is not int
        or gzip_record["member_count"] != 1
        or type(gzip_record["uncompressed_size"]) is not int
        or gzip_record["uncompressed_size"] < 1
    ):
        raise ValueError("Local gzip evidence changed")
    tar_record = _exact_object(local["tar"], _TAR_KEYS, name="local_bundle.tar")
    (
        actual_archive_size,
        actual_archive_sha256,
        actual_gzip,
        actual_tar,
        actual_tar_records,
        archive_file_identity,
    ) = _read_archive_snapshot(
        archive_path,
        extraction_root=None,
        expected_size_sha256=(archive_size, archive_sha256),
    )
    if (
        archive_size != actual_archive_size
        or archive_sha256 != actual_archive_sha256
        or archive_size != remote["size"]
        or archive_sha256 != remote["sha256"]
    ):
        raise ValueError("Local archive bytes differ from remote acquisition evidence")
    if aws.canonical_json_bytes(actual_gzip) != aws.canonical_json_bytes(
        gzip_record
    ):
        raise ValueError("Local gzip bytes differ from the receipt")
    if aws.canonical_json_bytes(actual_tar) != aws.canonical_json_bytes(tar_record):
        raise ValueError("Local TAR bytes differ from the receipt")

    artifact_record = _exact_object(
        receipt["artifact"], _ARTIFACT_KEYS, name="acquisition.artifact"
    )
    declared_files, declared_total = _validate_file_records(artifact_record["files"])
    if _tar_file_inventory(actual_tar_records) != declared_files:
        raise ValueError("TAR file inventory differs from the extracted artifact")
    if (
        type(artifact_record["file_count"]) is not int
        or artifact_record["file_count"] != len(declared_files)
        or type(artifact_record["total_size"]) is not int
        or artifact_record["total_size"] != declared_total
        or artifact_record["inventory_sha256"] != _document_sha256(declared_files)
    ):
        raise ValueError("Acquisition artifact inventory summary changed")
    _exact_sha256(
        artifact_record["inventory_sha256"], name="artifact.inventory_sha256"
    )
    manifest_sha256 = _exact_sha256(
        artifact_record["artifact_manifest_sha256"],
        name="artifact.artifact_manifest_sha256",
    )
    actual_artifact, validated_artifact = _artifact_payload(
        artifact_root=artifact_root,
        plan=plan,
        staged=staged,
        expected_manifest_sha256=manifest_sha256,
    )
    if aws.canonical_json_bytes(actual_artifact) != aws.canonical_json_bytes(
        artifact_record
    ):
        raise ValueError("Extracted determinism-smoke artifact changed")
    _validate_self_hash(receipt)
    _assert_regular_path_identity(archive_path, archive_file_identity)
    if (
        _path_identity(bundle_root) != bundle_identity
        or _path_identity(artifact_root) != artifact_root_identity
    ):
        raise RuntimeError("Acquisition bundle or artifact directory was replaced")
    return ValidatedDeterminismSmokeAcquisition(
        receipt=copy.deepcopy(receipt),
        receipt_path=selected_receipt_path,
        bundle_root=bundle_root,
        artifact_root=artifact_root,
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        inventory_sha256=artifact_record["inventory_sha256"],
        file_count=artifact_record["file_count"],
        total_size=artifact_record["total_size"],
        remote_object=copy.deepcopy(remote),
        request_receipt=copy.deepcopy(request_receipt),
        preflight_receipt=copy.deepcopy(preflight),
        submission_receipt=copy.deepcopy(submission),
        terminal_receipt=copy.deepcopy(terminal),
        validated_artifact=validated_artifact,
    )


def load_and_validate_determinism_smoke_acquisition_receipt(
    receipt_path: Path,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> ValidatedDeterminismSmokeAcquisition:
    """Load the canonical in-bundle receipt and reject detached copies."""

    if not isinstance(receipt_path, Path):
        raise TypeError("receipt_path must be one pathlib.Path")
    if not receipt_path.is_absolute():
        raise ValueError("receipt_path must be absolute")
    value, _ = strict_config.load_canonical_json_object(receipt_path)
    return validate_determinism_smoke_acquisition_receipt(
        value,
        training_plan=training_plan,
        staging_receipt=staging_receipt,
        receipt_path=receipt_path,
    )


def _same_remote_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    expected_without_sha = {
        key: value for key, value in expected.items() if key != "sha256"
    }
    if actual != expected_without_sha:
        raise RuntimeError("Remote model-artifact identity/history changed during acquisition")


def acquire_completed_determinism_smoke_artifact(
    s3: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any],
    output_bundle: Path,
) -> dict[str, Any]:
    """Acquire, validate, and atomically publish one completed smoke artifact."""

    plan, staged = _validated_plan_staging(training_plan, staging_receipt)
    preflight, submission, terminal, _ = _validate_launch_chain(
        plan=plan,
        staged=staged,
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        terminal_receipt=terminal_receipt,
    )
    output_bundle, incomplete = _validate_output_paths(output_bundle)
    remote_before = _inspect_remote_output(
        s3, plan=plan, preflight=preflight, terminal=terminal
    )
    _require_disk_space(output_bundle.parent, archive_size=remote_before["size"])
    incomplete.mkdir(mode=0o700)
    owned_identity = _path_identity(incomplete)
    published = False
    receipt: dict[str, Any] | None = None
    try:
        archive_path = incomplete / ARCHIVE_NAME
        download_size, download_sha256 = _download_exact_version(
            s3,
            remote=remote_before,
            account_id=plan["infrastructure"]["account_id"],
            destination=archive_path,
        )
        artifact_root = incomplete / ARTIFACT_DIRECTORY_NAME
        (
            archive_size,
            archive_sha256,
            gzip_evidence,
            tar_evidence,
            tar_records,
            archive_file_identity,
        ) = _read_archive_snapshot(
            archive_path,
            extraction_root=artifact_root,
            expected_size_sha256=(download_size, download_sha256),
        )
        remote = {**copy.deepcopy(remote_before), "sha256": archive_sha256}
        artifact_evidence, _ = _artifact_payload(
            artifact_root=artifact_root,
            plan=plan,
            staged=staged,
        )
        _assert_regular_path_identity(archive_path, archive_file_identity)
        if (
            tar_evidence["file_count"] != artifact_evidence["file_count"]
            or tar_evidence["file_bytes"] != artifact_evidence["total_size"]
            or _tar_file_inventory(tar_records) != artifact_evidence["files"]
        ):
            raise ValueError("TAR extraction and artifact inventories disagree")
        _same_remote_identity(
            _inspect_remote_output(
                s3, plan=plan, preflight=preflight, terminal=terminal
            ),
            remote,
        )
        final_artifact_root = output_bundle / ARTIFACT_DIRECTORY_NAME
        final_archive_path = output_bundle / ARCHIVE_NAME
        final_receipt_path = output_bundle / ACQUISITION_RECEIPT_NAME
        receipt = _seal_receipt(
            {
                "schema_version": 1,
                "protocol": DETERMINISM_SMOKE_ACQUISITION_PROTOCOL,
                "attempt_id": plan["attempt"]["attempt_id"],
                "run_id": terminal["run_id"],
                "job_name": terminal["job_name"],
                "job_arn": terminal["job_arn"],
                "model_artifact_s3_uri": terminal["model_artifact_s3_uri"],
                "launch_evidence": {
                    "preflight_receipt": preflight,
                    "submission_receipt": submission,
                    "terminal_receipt": terminal,
                },
                "evidence_chain": _evidence_chain(
                    plan=plan,
                    staged=staged,
                    preflight=preflight,
                    submission=submission,
                    terminal=terminal,
                ),
                "remote_object": remote,
                "local_bundle": {
                    "bundle_root": output_bundle.as_posix(),
                    "artifact_root": final_artifact_root.as_posix(),
                    "archive_path": final_archive_path.as_posix(),
                    "receipt_path": final_receipt_path.as_posix(),
                    "archive": {"size": archive_size, "sha256": archive_sha256},
                    "gzip": gzip_evidence,
                    "tar": tar_evidence,
                },
                "artifact": artifact_evidence,
            }
        )
        _write_receipt(incomplete / ACQUISITION_RECEIPT_NAME, receipt)
        _fsync_tree(incomplete)
        _same_remote_identity(
            _inspect_remote_output(
                s3, plan=plan, preflight=preflight, terminal=terminal
            ),
            remote,
        )
        if _path_identity(incomplete) != owned_identity:
            raise RuntimeError("Owned incomplete acquisition directory was replaced")
        _rename_no_replace(incomplete, output_bundle)
        published = True
        if _path_identity(output_bundle) != owned_identity:
            raise RuntimeError("Published acquisition directory was replaced")
        _fsync_directory(output_bundle.parent)
        validated = validate_determinism_smoke_acquisition_receipt(
            receipt,
            training_plan=plan,
            staging_receipt=staged,
            receipt_path=output_bundle / ACQUISITION_RECEIPT_NAME,
        )
        if _path_identity(output_bundle) != owned_identity:
            raise RuntimeError("Published acquisition directory was replaced")
        _same_remote_identity(
            _inspect_remote_output(
                s3, plan=plan, preflight=preflight, terminal=terminal
            ),
            validated.remote_object,
        )
        if _path_identity(output_bundle) != owned_identity:
            raise RuntimeError("Published acquisition directory was replaced")
        return copy.deepcopy(validated.receipt)
    except BaseException:
        if published:
            if _path_identity(output_bundle) != owned_identity:
                raise RuntimeError(
                    "Published acquisition directory was replaced; refusing rollback"
                )
            marker = output_bundle / ACQUISITION_RECEIPT_NAME
            if marker.exists() or marker.is_symlink():
                marker.unlink()
                _fsync_directory(output_bundle)
            _rename_no_replace(output_bundle, incomplete)
            _fsync_directory(output_bundle.parent)
        elif incomplete.exists() or incomplete.is_symlink():
            if _path_identity(incomplete) != owned_identity:
                raise RuntimeError(
                    "Owned incomplete acquisition directory was replaced; refusing cleanup"
                )
            shutil.rmtree(incomplete)
            _fsync_directory(incomplete.parent)
        raise


__all__: Sequence[str] = (
    "ACQUISITION_RECEIPT_NAME",
    "ARCHIVE_NAME",
    "ARTIFACT_DIRECTORY_NAME",
    "DETERMINISM_SMOKE_ACQUISITION_PROTOCOL",
    "ValidatedDeterminismSmokeAcquisition",
    "acquire_completed_determinism_smoke_artifact",
    "load_and_validate_determinism_smoke_acquisition_receipt",
    "validate_determinism_smoke_acquisition_receipt",
)
