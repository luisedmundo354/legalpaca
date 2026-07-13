"""Strict, network-free bridge from sealed training archives to fold evaluation.

The bridge has two deliberately separate operations.  Inventory scans twelve
SageMaker ``model.tar.gz`` files without extracting them and seals the exact
bytes and logical member inventory.  Materialization accepts only that sealed
evidence, scans every archive again, extracts into a new sibling-incomplete
tree, and invokes the existing controlled-artifact validator with expectations
provided by the caller.  It does not construct expectations from an artifact,
contact AWS, retry, resume, overwrite, or accept another archive format.
"""

from __future__ import annotations

import base64
import binascii
import copy
import ctypes
import dataclasses
import errno
import gzip
import hashlib
import json
import math
import os
import re
import stat
import tarfile
import zlib
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

from retriever.artifacts import (
    ControlledArtifactExpectation,
    ValidatedControlledArtifact,
    validate_controlled_artifact,
)


ARCHIVE_INPUT_MANIFEST_PROTOCOL = "retrieval_cv_fold_archive_input_manifest_v1"
ARCHIVE_INVENTORY_RECEIPT_PROTOCOL = "retrieval_cv_fold_archive_inventory_v1"
FOLD_MATERIALIZATION_PROTOCOL = "retrieval_cv_fold_archive_materialization_v1"

MAX_ARCHIVE_BYTES = 5_000_000_000
MAX_STREAM_BYTES = 100 * 1024**3
MAX_TREE_BYTES = MAX_STREAM_BYTES
MAX_PHYSICAL_MEMBERS = 8_192
MAX_LOGICAL_MEMBERS = 8_192
MAX_PAX_PAYLOAD_BYTES = 4_096
MAX_CAPTURE_BYTES = 1024 * 1024
MAX_CANONICAL_JSON_BYTES = 64 * 1024 * 1024
MAX_PATH_BYTES = 99
MAX_USTAR_FILE_BYTES = 0o77777777777
_COPY_CHUNK_BYTES = 1024 * 1024

_EXPERIMENT_ID = "arr_retrieval_cv_v1"
_QUERY_VIEWS = ("flat_masked", "structured")
_SAMPLERS = ("local_unique", "global_uniform")
_SEEDS = (17, 29, 43)
_PAX_KEYS = ("atime", "ctime", "mtime", "LIBARCHIVE.creationtime")
_PAX_PHYSICAL_NAME = re.compile(r"\./PaxHeaders\.X/[A-Za-z0-9._-]+\Z")
_PAX_TIME = re.compile(r"[1-9][0-9]{9}\.[0-9]{7}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_ID = re.compile(r"a[1-9][0-9]*\Z")
_ETAG = re.compile(r'"[0-9a-f]{32}(?:-[1-9][0-9]*)?"\Z')
_ArchiveIdentity = tuple[int, int, int, int, int, int, int]

_MANIFEST_KEYS = {
    "schema_version",
    "protocol",
    "experiment_id",
    "outer_fold",
    "attempt_id",
    "archive_root",
    "training_plan_sha256",
    "training_staging_receipt_sha256",
    "source_bundle",
    "copy_set_receipt_sha256",
    "systems",
}
_SOURCE_BUNDLE_KEYS = {
    "name",
    "size",
    "sha256",
    "inventory_sha256",
    "commit_epoch",
}
_SYSTEM_KEYS = {
    "ordinal",
    "system_id",
    "run_id",
    "job_name",
    "cell",
    "archive_path",
    "source_object",
    "destination_object",
    "terminal_receipt_sha256",
    "request_receipt_sha256",
}
_CELL_KEYS = {"outer_fold", "query_view", "sampler", "experiment_seed"}
_OBJECT_KEYS = {
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

_RECEIPT_KEYS = {
    "schema_version",
    "protocol",
    "input_manifest_sha256",
    "experiment_id",
    "outer_fold",
    "systems",
    "aggregate",
    "receipt_sha256",
}
_RECEIPT_SYSTEM_KEYS = {*_SYSTEM_KEYS, "archive_evidence"}
_EVIDENCE_KEYS = {"archive", "gzip", "tar", "members", "artifact"}
_ARCHIVE_KEYS = {"size", "allocated_bytes", "sha256"}
_GZIP_KEYS = {
    "compression_method",
    "flags",
    "mtime",
    "extra_flags",
    "operating_system",
    "member_count",
    "crc32",
    "isize_mod_2_32",
    "uncompressed_size",
}
_TAR_KEYS = {
    "physical_member_count",
    "logical_member_count",
    "pax_header_count",
    "file_count",
    "directory_count",
    "file_bytes",
    "pax_payload_bytes",
    "max_pax_payload_bytes",
    "max_file_bytes",
    "max_path_bytes",
    "terminator_block_count",
    "trailing_zero_bytes",
    "stream_inventory_sha256",
    "member_inventory_sha256",
}
_MEMBER_KEYS = {"kind", "path", "size", "sha256"}
_ARTIFACT_KEYS = {
    "artifact_manifest_size",
    "artifact_manifest_sha256",
    "artifact_manifest_capture_sha256",
    "file_count",
    "file_bytes",
    "file_inventory_sha256",
}
_AGGREGATE_KEYS = {
    "archive_count",
    "archive_bytes",
    "archive_allocated_bytes",
    "uncompressed_bytes",
    "artifact_file_bytes",
    "file_count",
    "directory_count",
    "max_archive_bytes",
    "max_uncompressed_bytes",
    "max_artifact_file_bytes",
    "archive_set_sha256",
    "artifact_inventory_set_sha256",
}


@dataclass(frozen=True)
class FoldArchiveMaterialization:
    """Published artifact roots and their independently validated identities."""

    root: Path
    receipt: dict[str, Any]
    artifacts: tuple[ValidatedControlledArtifact, ...]


@dataclass(frozen=True)
class _ArchiveSnapshot:
    path: Path
    descriptor: int
    identity: _ArchiveIdentity

    def assert_stable(self) -> None:
        try:
            metadata = os.fstat(self.descriptor)
        except OSError as error:
            raise RuntimeError("Archive descriptor became invalid") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or _file_identity(metadata) != self.identity
        ):
            raise RuntimeError("Archive descriptor identity changed")
        _assert_path_identity(self.path, self.identity)

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


def _file_identity(metadata: os.stat_result) -> _ArchiveIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_nlink,
    )


def _assert_path_identity(path: Path, expected: _ArchiveIdentity) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError("Archive path identity disappeared") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or _file_identity(metadata) != expected
    ):
        raise RuntimeError("Archive path identity changed")


@contextmanager
def _open_archive_snapshot(path: Path) -> Iterator[_ArchiveSnapshot]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Archive must be one singly-linked regular file")
        if metadata.st_size < 1 or metadata.st_size >= MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"Archive size must be in [1, {MAX_ARCHIVE_BYTES}): {metadata.st_size}"
            )
        snapshot = _ArchiveSnapshot(path, descriptor, _file_identity(metadata))
        snapshot.assert_stable()
        try:
            yield snapshot
        finally:
            snapshot.assert_stable()
    finally:
        os.close(descriptor)


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


def _document_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def _read_descriptor_exact(descriptor: int, size: int, *, name: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(_COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise RuntimeError(f"{name} changed size while being read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise RuntimeError(f"{name} grew while being read")
    return b"".join(chunks)


def _load_canonical_json(path: Path, *, name: str) -> dict[str, Any]:
    path = _strict_absolute_path(str(Path(path)), name=f"{name} path")
    _require_real_directory(path.parent, name=f"{name} parent")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{name} must be one singly-linked regular file")
        if metadata.st_size < 1 or metadata.st_size > MAX_CANONICAL_JSON_BYTES:
            raise ValueError(
                f"{name} size must be in [1, {MAX_CANONICAL_JSON_BYTES}]"
            )
        identity = _file_identity(metadata)
        _assert_path_identity(path, identity)
        raw = _read_descriptor_exact(descriptor, metadata.st_size, name=name)
        if _file_identity(os.fstat(descriptor)) != identity:
            raise RuntimeError(f"{name} descriptor identity changed while being read")
        _require_real_directory(path.parent, name=f"{name} parent")
        _assert_path_identity(path, identity)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid strict JSON") from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError(f"{name} must be one canonical JSON object")
    return value


def _exact_object(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be one non-empty exact string")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _strict_absolute_path(value: object, *, name: str) -> Path:
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


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = Path(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"{name} is absent: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{name} contains a symlink component: {current}")
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{name} must be one canonical real directory")
    return path


def _validate_source_bundle(value: object) -> dict[str, Any]:
    record = _exact_object(value, _SOURCE_BUNDLE_KEYS, name="source_bundle")
    digest = _exact_sha256(record["sha256"], name="source_bundle.sha256")
    if record["name"] != f"source-{digest}.tar.gz":
        raise ValueError("source_bundle.name does not bind its SHA-256")
    _exact_int(record["size"], name="source_bundle.size", minimum=1)
    _exact_sha256(record["inventory_sha256"], name="source_bundle.inventory_sha256")
    _exact_int(record["commit_epoch"], name="source_bundle.commit_epoch", minimum=1)
    return copy.deepcopy(record)


def _validate_object_record(value: object, *, name: str) -> dict[str, Any]:
    record = _exact_object(value, _OBJECT_KEYS, name=name)
    _exact_string(record["bucket"], name=f"{name}.bucket")
    key = _exact_string(record["key"], name=f"{name}.key")
    path = PurePosixPath(key)
    if key.startswith("/") or path.as_posix() != key or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name}.key is not one canonical object key")
    _exact_string(record["version_id"], name=f"{name}.version_id")
    size = _exact_int(record["size"], name=f"{name}.size", minimum=1)
    if size >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"{name}.size exceeds the atomic-copy boundary")
    etag = _exact_string(record["etag"], name=f"{name}.etag")
    if _ETAG.fullmatch(etag) is None:
        raise ValueError(f"{name}.etag is malformed")
    checksum = _exact_object(record["checksum"], _CHECKSUM_KEYS, name=f"{name}.checksum")
    for field in _CHECKSUM_KEYS:
        _exact_string(checksum[field], name=f"{name}.checksum.{field}")
    encryption = _exact_object(
        record["encryption"], _ENCRYPTION_KEYS, name=f"{name}.encryption"
    )
    _exact_string(encryption["algorithm"], name=f"{name}.encryption.algorithm")
    _exact_string(encryption["kms_key_id"], name=f"{name}.encryption.kms_key_id")
    if type(encryption["bucket_key_enabled"]) is not bool:
        raise ValueError(f"{name}.encryption.bucket_key_enabled must be boolean")
    return copy.deepcopy(record)


def _destination_object_sha256(record: Mapping[str, Any], *, name: str) -> str:
    checksum = _exact_object(
        record["checksum"], _CHECKSUM_KEYS, name=f"{name}.checksum"
    )
    if checksum["algorithm"] != "SHA256" or checksum["type"] != "FULL_OBJECT":
        raise ValueError(f"{name} must carry one full-object SHA-256 checksum")
    encoded = _exact_string(checksum["value"], name=f"{name}.checksum.value")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{name}.checksum.value is not canonical Base64") from error
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError(f"{name}.checksum.value is not one canonical SHA-256")
    return raw.hex()


def _expected_cells(outer_fold: int) -> list[dict[str, Any]]:
    cells = [
        {
            "outer_fold": outer_fold,
            "query_view": query_view,
            "sampler": sampler,
            "experiment_seed": seed,
        }
        for query_view in _QUERY_VIEWS
        for sampler in _SAMPLERS
        for seed in _SEEDS
    ]
    return sorted(cells, key=lambda item: _system_id(item))


def _system_id(cell: Mapping[str, Any]) -> str:
    return f"{cell['query_view']}_{cell['sampler']}_seed{cell['experiment_seed']}"


def _run_id(cell: Mapping[str, Any]) -> str:
    query = {"flat_masked": "flat", "structured": "struct"}[cell["query_view"]]
    sampler = {"local_unique": "local", "global_uniform": "global"}[cell["sampler"]]
    return (
        f"controlled-f{cell['outer_fold']}-{query}-{sampler}-"
        f"s{cell['experiment_seed']}"
    )


def _job_name(cell: Mapping[str, Any], attempt_id: str) -> str:
    return "arr-ret-cv1-" + _run_id(cell).removeprefix("controlled-") + f"-{attempt_id}"


def validate_fold_archive_input_manifest(value: object) -> dict[str, Any]:
    """Validate the exact twelve-cell, one-fold local archive binding."""

    manifest = _exact_object(value, _MANIFEST_KEYS, name="fold archive input manifest")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["protocol"] != ARCHIVE_INPUT_MANIFEST_PROTOCOL
        or manifest["experiment_id"] != _EXPERIMENT_ID
    ):
        raise ValueError("Fold archive input-manifest protocol identity changed")
    outer_fold = _exact_int(manifest["outer_fold"], name="outer_fold")
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer from 0 through 4")
    attempt_id = _exact_string(manifest["attempt_id"], name="attempt_id")
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("attempt_id is not canonical")
    archive_root = _strict_absolute_path(manifest["archive_root"], name="archive_root")
    _require_real_directory(archive_root, name="archive_root")
    _exact_sha256(manifest["training_plan_sha256"], name="training_plan_sha256")
    _exact_sha256(
        manifest["training_staging_receipt_sha256"],
        name="training_staging_receipt_sha256",
    )
    _validate_source_bundle(manifest["source_bundle"])
    _exact_sha256(manifest["copy_set_receipt_sha256"], name="copy_set_receipt_sha256")
    systems = manifest["systems"]
    expected_cells = _expected_cells(outer_fold)
    if type(systems) is not list or len(systems) != len(expected_cells):
        raise ValueError("Fold archive input manifest must contain exactly 12 systems")
    source_coordinates: set[tuple[str, str, str]] = set()
    destination_coordinates: set[tuple[str, str, str]] = set()
    normalized_systems: list[dict[str, Any]] = []
    for ordinal, (raw, expected_cell) in enumerate(zip(systems, expected_cells)):
        record = _exact_object(raw, _SYSTEM_KEYS, name=f"systems[{ordinal}]")
        cell = _exact_object(record["cell"], _CELL_KEYS, name=f"systems[{ordinal}].cell")
        if cell != expected_cell:
            raise ValueError(f"systems[{ordinal}] is not the canonical fold cell")
        system_id = _system_id(cell)
        run_id = _run_id(cell)
        if (
            type(record["ordinal"]) is not int
            or record["ordinal"] != ordinal
            or record["system_id"] != system_id
            or record["run_id"] != run_id
            or record["job_name"] != _job_name(cell, attempt_id)
        ):
            raise ValueError(f"systems[{ordinal}] identity is not canonical")
        expected_name = f"{ordinal:02d}-{system_id}.model.tar.gz"
        archive_path = _strict_absolute_path(
            record["archive_path"], name=f"systems[{ordinal}].archive_path"
        )
        if archive_path != archive_root / expected_name:
            raise ValueError(f"systems[{ordinal}] archive path left the fixed root layout")
        source = _validate_object_record(
            record["source_object"], name=f"systems[{ordinal}].source_object"
        )
        destination = _validate_object_record(
            record["destination_object"],
            name=f"systems[{ordinal}].destination_object",
        )
        _destination_object_sha256(
            destination, name=f"systems[{ordinal}].destination_object"
        )
        if (
            source["size"] != destination["size"]
            or not source["key"].endswith("/output/model.tar.gz")
            or not destination["key"].endswith("/" + expected_name)
        ):
            raise ValueError(f"systems[{ordinal}] copy coordinates or sizes changed")
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
            raise ValueError("Fold archive input manifest aliases an object version")
        source_coordinates.add(source_coordinate)
        destination_coordinates.add(destination_coordinate)
        _exact_sha256(
            record["terminal_receipt_sha256"],
            name=f"systems[{ordinal}].terminal_receipt_sha256",
        )
        _exact_sha256(
            record["request_receipt_sha256"],
            name=f"systems[{ordinal}].request_receipt_sha256",
        )
        normalized_systems.append(copy.deepcopy(record))
    normalized = copy.deepcopy(manifest)
    normalized["systems"] = normalized_systems
    return normalized


def load_fold_archive_input_manifest(path: Path) -> dict[str, Any]:
    return validate_fold_archive_input_manifest(
        _load_canonical_json(Path(path), name="Fold archive input manifest")
    )


def _hash_archive(snapshot: _ArchiveSnapshot) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with snapshot.reader() as source:
        while True:
            chunk = source.read(_COPY_CHUNK_BYTES)
            if type(chunk) is not bytes:
                raise RuntimeError("Archive reader returned non-bytes")
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size != snapshot.identity[2]:
        raise RuntimeError("Archive size changed while hashing")
    return size, digest.hexdigest()


def _gzip_header(snapshot: _ArchiveSnapshot) -> dict[str, int]:
    raw = os.pread(snapshot.descriptor, 10, 0)
    if len(raw) != 10:
        raise ValueError("Archive lacks one complete gzip header")
    if (
        raw[:3] != b"\x1f\x8b\x08"
        or raw[3] != 0
        or raw[4:8] != b"\0" * 4
        or raw[8] != 0
        or raw[9] != 3
    ):
        raise ValueError("Gzip header left the exact metadata-free Linux envelope")
    return {
        "compression_method": raw[2],
        "flags": raw[3],
        "mtime": int.from_bytes(raw[4:8], "little"),
        "extra_flags": raw[8],
        "operating_system": raw[9],
    }


def _audit_single_gzip(snapshot: _ArchiveSnapshot) -> dict[str, int | str]:
    header = _gzip_header(snapshot)
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    output_size = 0
    crc32 = 0
    reached_end = False
    try:
        with snapshot.reader() as source:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                pending = chunk
                while pending:
                    previous = pending
                    output = decompressor.decompress(pending, _COPY_CHUNK_BYTES)
                    output_size += len(output)
                    if output_size > MAX_STREAM_BYTES:
                        raise ValueError("Gzip output exceeds the fixed 100-GiB ceiling")
                    crc32 = zlib.crc32(output, crc32)
                    pending = decompressor.unconsumed_tail
                    if not output and pending == previous and not decompressor.eof:
                        raise RuntimeError("Gzip decompressor made no progress")
                    if decompressor.eof:
                        if decompressor.unused_data or pending or source.read(1):
                            raise ValueError(
                                "Archive contains trailing bytes or another gzip member"
                            )
                        reached_end = True
                        break
                if reached_end:
                    break
    except zlib.error as error:
        raise ValueError("Archive gzip stream failed CRC/format validation") from error
    if not reached_end or not decompressor.eof:
        raise ValueError("Archive is not one complete gzip member")
    try:
        flushed = decompressor.flush()
    except zlib.error as error:
        raise ValueError("Archive gzip trailer is invalid") from error
    if flushed:
        output_size += len(flushed)
        crc32 = zlib.crc32(flushed, crc32)
        if output_size > MAX_STREAM_BYTES:
            raise ValueError("Gzip output exceeds the fixed 100-GiB ceiling")
    trailer = os.pread(snapshot.descriptor, 8, snapshot.identity[2] - 8)
    if len(trailer) != 8:
        raise ValueError("Archive lacks one complete gzip trailer")
    expected_crc = int.from_bytes(trailer[:4], "little")
    expected_isize = int.from_bytes(trailer[4:], "little")
    if expected_crc != crc32 or expected_isize != output_size % (2**32):
        raise ValueError("Gzip trailer CRC/ISIZE differs from streamed output")
    return {
        **header,
        "member_count": 1,
        "crc32": f"{crc32:08x}",
        "isize_mod_2_32": expected_isize,
        "uncompressed_size": output_size,
    }


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


def _decode_nul_field(raw: bytes, *, name: str) -> str:
    prefix, marker, suffix = raw.partition(b"\0")
    if not marker or any(suffix):
        raise ValueError(f"{name} is not canonically NUL padded")
    try:
        return prefix.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{name} is not strict UTF-8") from error


def _validate_ustar_header(
    header: bytes,
    *,
    name: str,
    expected_mode: bytes,
    expected_type: bytes,
    expected_uname: str,
    expected_gname: str,
) -> str:
    if (
        len(header) != 512
        or header[100:108] != expected_mode
        or header[108:116] != b"0000000 "
        or header[116:124] != b"0000000 "
        or re.fullmatch(rb"[0-7]{11} ", header[124:136]) is None
        or re.fullmatch(rb"[0-7]{11} ", header[136:148]) is None
        or re.fullmatch(rb"[0-7]{6}\x00 ", header[148:156]) is None
        or header[156:157] != expected_type
        or any(header[157:257])
        or header[257:263] != b"ustar\0"
        or header[263:265] != b"00"
        or header[329:337] != b"0000000 "
        or header[337:345] != b"0000000 "
        or any(header[345:500])
        or any(header[500:512])
    ):
        raise ValueError(f"{name} left the exact ustar header envelope")
    raw_name = _decode_nul_field(header[:100], name=f"{name}.name")
    uname = _decode_nul_field(header[265:297], name=f"{name}.uname")
    gname = _decode_nul_field(header[297:329], name=f"{name}.gname")
    if uname != expected_uname or gname != expected_gname:
        raise ValueError(f"{name} owner names changed")
    return raw_name


def _parse_pax_payload(payload: bytes) -> dict[str, str]:
    records: dict[str, str] = {}
    key_order: list[str] = []
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
        key_order.append(key)
        position = end
    if tuple(key_order) != _PAX_KEYS or any(
        _PAX_TIME.fullmatch(value) is None for value in records.values()
    ):
        raise ValueError("PAX header left the exact ordered time-only contract")
    return records


def _canonical_member_path(raw: str, *, is_directory: bool) -> str:
    if type(raw) is not str or not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("TAR member path is empty or unsafe")
    if is_directory:
        if not raw.endswith("/"):
            raise ValueError("TAR directory path lacks its canonical trailing slash")
        normalized = raw[:-1]
    else:
        if raw.endswith("/"):
            raise ValueError("TAR file path has a directory suffix")
        normalized = raw
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


def _exact_archive_member_names(path: str, *, kind: str) -> tuple[str, str]:
    if kind not in {"file", "directory"}:
        raise ValueError("Archive member kind is not representable")
    raw_name = path + "/" if kind == "directory" else path
    raw_bytes = raw_name.encode("utf-8")
    pax_name = "./PaxHeaders.X/" + raw_name.replace("/", "_")
    pax_bytes = pax_name.encode("utf-8")
    if (
        len(raw_bytes) >= 100
        or len(pax_bytes) >= 100
        or _PAX_PHYSICAL_NAME.fullmatch(pax_name) is None
    ):
        raise ValueError("Archive member names do not fit the exact USTAR/PAX fields")
    return raw_name, pax_name


@contextmanager
def _opened_directory(root_fd: int, parts: Sequence[str]) -> Iterator[int]:
    current = os.dup(root_fd)
    try:
        for part in parts:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = next_descriptor
        yield current
    finally:
        os.close(current)


class _ExtractionRoot:
    def __init__(self, root: Path) -> None:
        self.root = root
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"Extraction root must be absent: {root}")
        parent = root.parent
        _require_real_directory(parent, name="Extraction-root parent")
        os.mkdir(root, mode=0o700)
        self._identity = _directory_identity(root)
        descriptor = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        metadata = os.fstat(descriptor)
        descriptor_identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
        if not stat.S_ISDIR(metadata.st_mode) or descriptor_identity != self._identity:
            os.close(descriptor)
            raise RuntimeError("Extraction root changed between creation and open")
        self._descriptor = descriptor

    def close(self) -> None:
        try:
            os.fsync(self._descriptor)
            metadata = os.fstat(self._descriptor)
            descriptor_identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or descriptor_identity != self._identity
                or _directory_identity(self.root) != self._identity
            ):
                raise RuntimeError("Extraction root identity changed")
        finally:
            os.close(self._descriptor)

    def mkdir(self, relative: str) -> None:
        path = PurePosixPath(relative)
        with _opened_directory(self._descriptor, path.parts[:-1]) as parent_fd:
            os.mkdir(path.parts[-1], mode=0o700, dir_fd=parent_fd)
            child = os.open(
                path.parts[-1],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            try:
                os.fsync(child)
            finally:
                os.close(child)

    @contextmanager
    def file(self, relative: str) -> Iterator[BinaryIO]:
        path = PurePosixPath(relative)
        with _opened_directory(self._descriptor, path.parts[:-1]) as parent_fd:
            descriptor = os.open(
                path.parts[-1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
                dir_fd=parent_fd,
            )
            target = os.fdopen(descriptor, "wb", buffering=0)
            try:
                yield target
                target.flush()
                os.fsync(target.fileno())
            finally:
                target.close()


def _directory_identity(path: Path) -> tuple[int, int, int]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Expected one real directory: {path}")
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


@dataclass
class _DirectorySnapshot:
    path: Path
    descriptor: int
    identity: tuple[int, int, int]
    name: str

    def assert_stable(self) -> None:
        metadata = os.fstat(self.descriptor)
        actual = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
        )
        if not stat.S_ISDIR(metadata.st_mode) or actual != self.identity:
            raise RuntimeError(f"{self.name} descriptor identity changed")
        _require_real_directory(self.path, name=self.name)
        if _directory_identity(self.path) != self.identity:
            raise RuntimeError(f"{self.name} path identity changed")

    def rebind(self, path: Path, *, name: str) -> None:
        self.path = Path(path)
        self.name = name
        self.assert_stable()


@contextmanager
def _open_directory_snapshot(path: Path, *, name: str) -> Iterator[_DirectorySnapshot]:
    path = _require_real_directory(Path(path), name=name)
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        snapshot = _DirectorySnapshot(
            path=path,
            descriptor=descriptor,
            identity=_directory_identity(path),
            name=name,
        )
        snapshot.assert_stable()
        try:
            yield snapshot
        finally:
            snapshot.assert_stable()
    finally:
        os.close(descriptor)


def _scan_tar(
    snapshot: _ArchiveSnapshot,
    *,
    gzip_evidence: Mapping[str, Any],
    extraction_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    extractor = None if extraction_root is None else _ExtractionRoot(extraction_root)
    stream_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_pax_paths: set[str] = set()
    declared_directories: set[str] = set()
    physical_count = logical_count = file_count = directory_count = 0
    file_bytes = pax_bytes = max_pax = max_file = max_path = 0
    stream_position = 0
    manifest_capture: bytes | None = None
    try:
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
                        if source.read(1):
                            raise ValueError("TAR archive has trailing zero or nonzero bytes")
                        break
                    physical_count += 1
                    if physical_count > MAX_PHYSICAL_MEMBERS:
                        raise ValueError("TAR physical-member cap exceeded")
                    pax = _tar_info(header, name="PAX")
                    pax_raw_name = _validate_ustar_header(
                        header,
                        name="PAX",
                        expected_mode=b"0100644 ",
                        expected_type=tarfile.XHDTYPE,
                        expected_uname="",
                        expected_gname="",
                    )
                    if (
                        pax.type != tarfile.XHDTYPE
                        or _PAX_PHYSICAL_NAME.fullmatch(pax_raw_name) is None
                        or pax_raw_name in seen_pax_paths
                        or type(pax.size) is not int
                        or pax.size != 130
                        or pax.uid != 0
                        or pax.gid != 0
                        or pax.linkname
                        or pax.devmajor != 0
                        or pax.devminor != 0
                    ):
                        raise ValueError("Every logical member requires one exact local PAX header")
                    seen_pax_paths.add(pax_raw_name)
                    pax_padded_size = math.ceil(pax.size / 512) * 512
                    pax_block = _read_exact(source, pax_padded_size, name="PAX payload")
                    stream_position += pax_padded_size
                    if any(pax_block[pax.size :]):
                        raise ValueError("PAX payload padding is nonzero")
                    pax_records = _parse_pax_payload(pax_block[: pax.size])
                    if pax_records["LIBARCHIVE.creationtime"] != pax_records["mtime"]:
                        raise ValueError("PAX creationtime and mtime differ")
                    pax_bytes += pax.size
                    max_pax = max(max_pax, pax.size)

                    logical_header = _read_exact(source, 512, name="logical TAR header")
                    stream_position += 512
                    physical_count += 1
                    if physical_count > MAX_PHYSICAL_MEMBERS:
                        raise ValueError("TAR physical-member cap exceeded")
                    member = _tar_info(logical_header, name="logical TAR member")
                    is_file = member.type == tarfile.REGTYPE
                    is_directory = member.type == tarfile.DIRTYPE
                    if not (is_file or is_directory):
                        raise ValueError("TAR contains a link, special, sparse, or unknown member")
                    raw_name = _validate_ustar_header(
                        logical_header,
                        name="logical TAR member",
                        expected_mode=b"0100644 " if is_file else b"0040755 ",
                        expected_type=tarfile.REGTYPE if is_file else tarfile.DIRTYPE,
                        expected_uname="root",
                        expected_gname="root",
                    )
                    if (
                        member.uid != 0
                        or member.gid != 0
                        or member.linkname
                        or member.devmajor != 0
                        or member.devminor != 0
                        or getattr(member, "sparse", None) is not None
                        or int(pax_records["mtime"].split(".", 1)[0]) != member.mtime
                        or pax.mtime != member.mtime
                    ):
                        raise ValueError("Logical TAR member metadata changed")
                    path = _canonical_member_path(raw_name, is_directory=is_directory)
                    kind = "directory" if is_directory else "file"
                    expected_raw_name, expected_pax_name = _exact_archive_member_names(
                        path, kind=kind
                    )
                    if (
                        member.name != path
                        or raw_name != expected_raw_name
                        or pax_raw_name != expected_pax_name
                    ):
                        raise ValueError("TAR parser path differs from the raw canonical path")
                    if path in seen_paths:
                        raise ValueError("TAR contains a duplicate logical path")
                    parent = PurePosixPath(path).parent
                    if parent != PurePosixPath(".") and parent.as_posix() not in declared_directories:
                        raise ValueError("TAR member appears before its explicit parent directory")
                    seen_paths.add(path)
                    logical_count += 1
                    if logical_count > MAX_LOGICAL_MEMBERS:
                        raise ValueError("TAR logical-member cap exceeded")
                    max_path = max(max_path, len(path.encode("utf-8")))
                    if is_directory:
                        if member.size != 0:
                            raise ValueError("TAR directory has nonzero data size")
                        declared_directories.add(path)
                        directory_count += 1
                        if extractor is not None:
                            extractor.mkdir(path)
                        record = {"kind": "directory", "path": path, "size": 0, "sha256": None}
                    else:
                        size = _exact_int(member.size, name=f"TAR file {path} size", minimum=1)
                        if size > MAX_USTAR_FILE_BYTES:
                            raise ValueError("TAR file size exceeds the exact USTAR field")
                        file_count += 1
                        file_bytes += size
                        max_file = max(max_file, size)
                        if file_bytes > MAX_TREE_BYTES:
                            raise ValueError("Artifact tree exceeds the fixed 100-GiB ceiling")
                        digest = hashlib.sha256()
                        captured = bytearray() if path == "artifact_manifest.json" else None
                        remaining = size
                        target_context = (
                            nullcontext(None) if extractor is None else extractor.file(path)
                        )
                        with target_context as target:
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
                                if captured is not None:
                                    if len(captured) + len(chunk) > MAX_CAPTURE_BYTES:
                                        raise ValueError("artifact_manifest.json exceeds capture cap")
                                    captured.extend(chunk)
                                remaining -= len(chunk)
                        if captured is not None:
                            manifest_capture = bytes(captured)
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
                    stream_records.append(record)
    finally:
        if extractor is not None:
            extractor.close()
    if stream_position != gzip_evidence["uncompressed_size"]:
        raise ValueError("Gzip and TAR uncompressed sizes disagree")
    if not stream_records or file_count < 1 or manifest_capture is None:
        raise ValueError("TAR archive lacks a captured artifact_manifest.json")
    sorted_records = sorted(stream_records, key=lambda item: item["path"])
    summary = {
        "physical_member_count": physical_count,
        "logical_member_count": logical_count,
        "pax_header_count": logical_count,
        "file_count": file_count,
        "directory_count": directory_count,
        "file_bytes": file_bytes,
        "pax_payload_bytes": pax_bytes,
        "max_pax_payload_bytes": max_pax,
        "max_file_bytes": max_file,
        "max_path_bytes": max_path,
        "terminator_block_count": 2,
        "trailing_zero_bytes": 0,
        "stream_inventory_sha256": _document_sha256(stream_records),
        "member_inventory_sha256": _document_sha256(sorted_records),
    }
    return summary, stream_records, manifest_capture


def _validate_allocated_bytes(value: object, *, size: int, name: str) -> int:
    allocated = _exact_int(value, name=name)
    if allocated % 512 != 0 or allocated < size:
        raise ValueError(f"{name} must be whole blocks covering the archive")
    return allocated


def scan_controlled_archive(path: Path, *, expected_size: int) -> dict[str, Any]:
    """Scan one exact archive without extracting it."""

    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Archive path must be absolute")
    expected_size = _exact_int(expected_size, name="expected_size", minimum=1)
    if expected_size >= MAX_ARCHIVE_BYTES:
        raise ValueError("expected_size exceeds the atomic-copy boundary")
    with _open_archive_snapshot(path) as snapshot:
        size, digest = _hash_archive(snapshot)
        if size != expected_size:
            raise ValueError("Mounted archive size differs from the input manifest")
        gzip_evidence = _audit_single_gzip(snapshot)
        tar_evidence, members, manifest_capture = _scan_tar(
            snapshot, gzip_evidence=gzip_evidence, extraction_root=None
        )
        snapshot.assert_stable()
        allocated_bytes = _validate_allocated_bytes(
            os.fstat(snapshot.descriptor).st_blocks * 512,
            size=size,
            name="archive.allocated_bytes",
        )
        archive = {
            "size": size,
            "allocated_bytes": allocated_bytes,
            "sha256": digest,
        }
    files = sorted(
        (record for record in members if record["kind"] == "file"),
        key=lambda item: item["path"],
    )
    manifest_record = next(
        (record for record in files if record["path"] == "artifact_manifest.json"),
        None,
    )
    if manifest_record is None:
        raise ValueError("Archive has no artifact_manifest.json file record")
    artifact = {
        "artifact_manifest_size": manifest_record["size"],
        "artifact_manifest_sha256": manifest_record["sha256"],
        "artifact_manifest_capture_sha256": hashlib.sha256(manifest_capture).hexdigest(),
        "file_count": len(files),
        "file_bytes": sum(record["size"] for record in files),
        "file_inventory_sha256": _document_sha256(files),
    }
    if artifact["artifact_manifest_capture_sha256"] != artifact["artifact_manifest_sha256"]:
        raise RuntimeError("Captured artifact manifest differs from its streamed digest")
    return {
        "archive": archive,
        "gzip": gzip_evidence,
        "tar": tar_evidence,
        "members": members,
        "artifact": artifact,
    }


def _aggregate_evidence(systems: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidences = [record["archive_evidence"] for record in systems]
    return {
        "archive_count": len(evidences),
        "archive_bytes": sum(item["archive"]["size"] for item in evidences),
        "archive_allocated_bytes": sum(
            item["archive"]["allocated_bytes"] for item in evidences
        ),
        "uncompressed_bytes": sum(item["gzip"]["uncompressed_size"] for item in evidences),
        "artifact_file_bytes": sum(item["artifact"]["file_bytes"] for item in evidences),
        "file_count": sum(item["tar"]["file_count"] for item in evidences),
        "directory_count": sum(item["tar"]["directory_count"] for item in evidences),
        "max_archive_bytes": max(item["archive"]["size"] for item in evidences),
        "max_uncompressed_bytes": max(
            item["gzip"]["uncompressed_size"] for item in evidences
        ),
        "max_artifact_file_bytes": max(
            item["artifact"]["file_bytes"] for item in evidences
        ),
        "archive_set_sha256": _document_sha256(
            [
                {
                    "system_id": record["system_id"],
                    "size": record["archive_evidence"]["archive"]["size"],
                    "sha256": record["archive_evidence"]["archive"]["sha256"],
                }
                for record in systems
            ]
        ),
        "artifact_inventory_set_sha256": _document_sha256(
            [
                {
                    "system_id": record["system_id"],
                    "artifact_manifest_sha256": record["archive_evidence"]["artifact"][
                        "artifact_manifest_sha256"
                    ],
                    "file_inventory_sha256": record["archive_evidence"]["artifact"][
                        "file_inventory_sha256"
                    ],
                }
                for record in systems
            ]
        ),
    }


def _seal_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(payload)
    sealed["receipt_sha256"] = _document_sha256(sealed)
    return sealed


def build_fold_archive_inventory_receipt(input_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Scan all twelve mounted archives and return one canonical sealed receipt."""

    manifest = validate_fold_archive_input_manifest(copy.deepcopy(input_manifest))
    systems: list[dict[str, Any]] = []
    for record in manifest["systems"]:
        system = copy.deepcopy(record)
        evidence = scan_controlled_archive(
            Path(record["archive_path"]),
            expected_size=record["destination_object"]["size"],
        )
        expected_archive_sha256 = _destination_object_sha256(
            record["destination_object"],
            name=f"{record['system_id']}.destination_object",
        )
        if evidence["archive"]["sha256"] != expected_archive_sha256:
            raise ValueError(
                f"{record['system_id']} archive differs from its copied-object SHA-256"
            )
        system["archive_evidence"] = evidence
        systems.append(system)
    return _seal_receipt(
        {
            "schema_version": 1,
            "protocol": ARCHIVE_INVENTORY_RECEIPT_PROTOCOL,
            "input_manifest_sha256": _document_sha256(manifest),
            "experiment_id": manifest["experiment_id"],
            "outer_fold": manifest["outer_fold"],
            "systems": systems,
            "aggregate": _aggregate_evidence(systems),
        }
    )


def _validate_evidence(value: object, *, name: str) -> dict[str, Any]:
    evidence = _exact_object(value, _EVIDENCE_KEYS, name=name)
    archive = _exact_object(evidence["archive"], _ARCHIVE_KEYS, name=f"{name}.archive")
    size = _exact_int(archive["size"], name=f"{name}.archive.size", minimum=1)
    if size >= MAX_ARCHIVE_BYTES:
        raise ValueError(f"{name}.archive.size exceeds the atomic-copy boundary")
    _validate_allocated_bytes(
        archive["allocated_bytes"],
        size=size,
        name=f"{name}.archive.allocated_bytes",
    )
    _exact_sha256(archive["sha256"], name=f"{name}.archive.sha256")
    gzip_record = _exact_object(evidence["gzip"], _GZIP_KEYS, name=f"{name}.gzip")
    expected_gzip_fields = {
        "compression_method": 8,
        "flags": 0,
        "mtime": 0,
        "extra_flags": 0,
        "operating_system": 3,
        "member_count": 1,
    }
    for field, expected in expected_gzip_fields.items():
        actual = _exact_int(gzip_record[field], name=f"{name}.gzip.{field}")
        if actual != expected:
            raise ValueError(f"{name}.gzip.{field} changed")
    if (
        type(gzip_record["crc32"]) is not str
        or re.fullmatch(r"[0-9a-f]{8}", gzip_record["crc32"]) is None
    ):
        raise ValueError(f"{name}.gzip identity changed")
    uncompressed = _exact_int(
        gzip_record["uncompressed_size"], name=f"{name}.gzip.uncompressed_size", minimum=1
    )
    if uncompressed > MAX_STREAM_BYTES:
        raise ValueError(f"{name}.gzip exceeds the streaming ceiling")
    isize = _exact_int(
        gzip_record["isize_mod_2_32"], name=f"{name}.gzip.isize_mod_2_32"
    )
    if isize >= 2**32 or isize != uncompressed % (2**32):
        raise ValueError(f"{name}.gzip ISIZE changed")
    tar_record = _exact_object(evidence["tar"], _TAR_KEYS, name=f"{name}.tar")
    for field in _TAR_KEYS - {"stream_inventory_sha256", "member_inventory_sha256"}:
        _exact_int(tar_record[field], name=f"{name}.tar.{field}")
    if (
        tar_record["physical_member_count"] != 2 * tar_record["logical_member_count"]
        or tar_record["pax_header_count"] != tar_record["logical_member_count"]
        or tar_record["logical_member_count"]
        != tar_record["file_count"] + tar_record["directory_count"]
        or tar_record["terminator_block_count"] != 2
        or tar_record["trailing_zero_bytes"] != 0
        or tar_record["physical_member_count"] > MAX_PHYSICAL_MEMBERS
        or tar_record["logical_member_count"] > MAX_LOGICAL_MEMBERS
        or tar_record["file_bytes"] > MAX_TREE_BYTES
        or tar_record["max_pax_payload_bytes"] > MAX_PAX_PAYLOAD_BYTES
        or tar_record["max_path_bytes"] > MAX_PATH_BYTES
    ):
        raise ValueError(f"{name}.tar summary is inconsistent")
    for field in ("stream_inventory_sha256", "member_inventory_sha256"):
        _exact_sha256(tar_record[field], name=f"{name}.tar.{field}")
    members = evidence["members"]
    if type(members) is not list or not members:
        raise ValueError(f"{name}.members must be a non-empty list")
    paths: list[str] = []
    pax_names: set[str] = set()
    declared_directories: set[str] = set()
    files: list[dict[str, Any]] = []
    file_bytes = directory_count = 0
    for index, raw in enumerate(members):
        member = _exact_object(raw, _MEMBER_KEYS, name=f"{name}.members[{index}]")
        kind = member["kind"]
        if kind not in {"file", "directory"}:
            raise ValueError(f"{name}.members[{index}].kind changed")
        path = _exact_string(member["path"], name=f"{name}.members[{index}].path")
        if _canonical_member_path(path + "/" if kind == "directory" else path, is_directory=kind == "directory") != path:
            raise ValueError(f"{name}.members[{index}].path is unsafe")
        _, pax_name = _exact_archive_member_names(path, kind=kind)
        if pax_name in pax_names:
            raise ValueError(f"{name}.members contains a derived PAX-name collision")
        pax_names.add(pax_name)
        if path in paths:
            raise ValueError(f"{name}.members contains a duplicate path")
        parent = PurePosixPath(path).parent
        if parent != PurePosixPath(".") and parent.as_posix() not in declared_directories:
            raise ValueError(f"{name}.members violates explicit parent-before-child order")
        size_value = _exact_int(
            member["size"], name=f"{name}.members[{index}].size", minimum=0
        )
        if kind == "directory":
            if size_value != 0 or member["sha256"] is not None:
                raise ValueError(f"{name}.members[{index}] directory record changed")
            directory_count += 1
            declared_directories.add(path)
        else:
            if size_value < 1 or size_value > MAX_USTAR_FILE_BYTES:
                raise ValueError(
                    f"{name}.members[{index}] file size is not exact USTAR"
                )
            _exact_sha256(member["sha256"], name=f"{name}.members[{index}].sha256")
            files.append(copy.deepcopy(member))
            file_bytes += size_value
        paths.append(path)
    sorted_members = sorted(members, key=lambda item: item["path"])
    files.sort(key=lambda item: item["path"])
    reconstructed_stream_bytes = (
        len(members) * 1536
        + sum(((record["size"] + 511) // 512) * 512 for record in files)
        + 1024
    )
    if (
        len(members) != tar_record["logical_member_count"]
        or len(files) != tar_record["file_count"]
        or directory_count != tar_record["directory_count"]
        or file_bytes != tar_record["file_bytes"]
        or tar_record["pax_payload_bytes"] != len(members) * 130
        or tar_record["max_pax_payload_bytes"] != 130
        or tar_record["max_file_bytes"] != max(record["size"] for record in files)
        or tar_record["max_path_bytes"]
        != max(len(record["path"].encode("utf-8")) for record in members)
        or _document_sha256(members) != tar_record["stream_inventory_sha256"]
        or _document_sha256(sorted_members) != tar_record["member_inventory_sha256"]
        or uncompressed != reconstructed_stream_bytes
    ):
        raise ValueError(f"{name}.member inventory and TAR summary disagree")
    artifact = _exact_object(evidence["artifact"], _ARTIFACT_KEYS, name=f"{name}.artifact")
    for field in ("artifact_manifest_size", "file_count", "file_bytes"):
        _exact_int(artifact[field], name=f"{name}.artifact.{field}", minimum=1)
    if artifact["artifact_manifest_size"] > MAX_CAPTURE_BYTES:
        raise ValueError(f"{name}.artifact manifest exceeds the capture ceiling")
    for field in (
        "artifact_manifest_sha256",
        "artifact_manifest_capture_sha256",
        "file_inventory_sha256",
    ):
        _exact_sha256(artifact[field], name=f"{name}.artifact.{field}")
    manifest_file = next(
        (record for record in files if record["path"] == "artifact_manifest.json"), None
    )
    if (
        manifest_file is None
        or artifact["artifact_manifest_size"] != manifest_file["size"]
        or artifact["artifact_manifest_sha256"] != manifest_file["sha256"]
        or artifact["artifact_manifest_capture_sha256"] != manifest_file["sha256"]
        or artifact["file_count"] != len(files)
        or artifact["file_bytes"] != file_bytes
        or artifact["file_inventory_sha256"] != _document_sha256(files)
    ):
        raise ValueError(f"{name}.artifact summary changed")
    return copy.deepcopy(evidence)


def validate_fold_archive_inventory_receipt(
    value: object, *, input_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = validate_fold_archive_input_manifest(copy.deepcopy(input_manifest))
    receipt = _exact_object(value, _RECEIPT_KEYS, name="fold archive inventory receipt")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != ARCHIVE_INVENTORY_RECEIPT_PROTOCOL
        or receipt["input_manifest_sha256"] != _document_sha256(manifest)
        or receipt["experiment_id"] != manifest["experiment_id"]
        or receipt["outer_fold"] != manifest["outer_fold"]
    ):
        raise ValueError("Fold archive inventory receipt identity changed")
    receipt_outer_fold = _exact_int(
        receipt["outer_fold"], name="receipt.outer_fold"
    )
    if receipt_outer_fold != manifest["outer_fold"]:
        raise ValueError("Fold archive inventory receipt outer fold changed")
    systems = receipt["systems"]
    if type(systems) is not list or len(systems) != 12:
        raise ValueError("Fold archive inventory receipt must contain 12 systems")
    normalized_systems: list[dict[str, Any]] = []
    for index, (raw, expected) in enumerate(zip(systems, manifest["systems"])):
        record = _exact_object(raw, _RECEIPT_SYSTEM_KEYS, name=f"receipt.systems[{index}]")
        without_evidence = {key: record[key] for key in _SYSTEM_KEYS}
        if without_evidence != expected:
            raise ValueError(f"receipt.systems[{index}] was spliced from another manifest")
        evidence = _validate_evidence(
            record["archive_evidence"], name=f"receipt.systems[{index}].archive_evidence"
        )
        expected_archive_sha256 = _destination_object_sha256(
            expected["destination_object"],
            name=f"receipt.systems[{index}].destination_object",
        )
        if (
            evidence["archive"]["size"] != expected["destination_object"]["size"]
            or evidence["archive"]["sha256"] != expected_archive_sha256
        ):
            raise ValueError(
                f"receipt.systems[{index}] archive left its copied-object binding"
            )
        normalized = copy.deepcopy(record)
        normalized["archive_evidence"] = evidence
        normalized_systems.append(normalized)
    aggregate = _exact_object(receipt["aggregate"], _AGGREGATE_KEYS, name="receipt.aggregate")
    for field in _AGGREGATE_KEYS - {
        "archive_set_sha256",
        "artifact_inventory_set_sha256",
    }:
        _exact_int(aggregate[field], name=f"receipt.aggregate.{field}")
    _exact_sha256(
        aggregate["archive_set_sha256"], name="receipt.aggregate.archive_set_sha256"
    )
    _exact_sha256(
        aggregate["artifact_inventory_set_sha256"],
        name="receipt.aggregate.artifact_inventory_set_sha256",
    )
    expected_aggregate = _aggregate_evidence(normalized_systems)
    if aggregate != expected_aggregate:
        raise ValueError("Fold archive inventory aggregate changed")
    claimed_hash = _exact_sha256(receipt["receipt_sha256"], name="receipt.receipt_sha256")
    unsealed = copy.deepcopy(receipt)
    del unsealed["receipt_sha256"]
    if claimed_hash != _document_sha256(unsealed):
        raise ValueError("Fold archive inventory receipt self-hash changed")
    normalized_receipt = copy.deepcopy(receipt)
    normalized_receipt["systems"] = normalized_systems
    return normalized_receipt


def load_fold_archive_inventory_receipt(
    path: Path, *, input_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return validate_fold_archive_inventory_receipt(
        _load_canonical_json(Path(path), name="Fold archive inventory receipt"),
        input_manifest=input_manifest,
    )


def _open_absent_regular(parent_descriptor: int, name: str) -> int:
    return os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=parent_descriptor,
    )


def _write_descriptor_exact(descriptor: int, payload: bytes, *, name: str) -> None:
    view = memoryview(payload)
    position = 0
    while position < len(view):
        written = os.write(descriptor, view[position:])
        if type(written) is not int or written <= 0 or written > len(view) - position:
            raise RuntimeError(f"{name} writer made no valid progress")
        position += written


def _write_absent_canonical(path: Path, value: Mapping[str, Any]) -> None:
    path = _strict_absolute_path(str(Path(path)), name="Receipt output path")
    if not path.name:
        raise ValueError("Receipt output path must name one file")
    payload = _canonical_bytes(value)
    with _open_directory_snapshot(path.parent, name="Receipt output parent") as parent:
        descriptor = _open_absent_regular(parent.descriptor, path.name)
        try:
            _write_descriptor_exact(descriptor, payload, name="Receipt output")
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != len(payload)
            ):
                raise RuntimeError("Receipt output stopped being singly-linked and regular")
            identity = _file_identity(metadata)
            os.lseek(descriptor, 0, os.SEEK_SET)
            readback = _read_descriptor_exact(
                descriptor, len(payload), name="Receipt output readback"
            )
            if readback != payload or _file_identity(os.fstat(descriptor)) != identity:
                raise RuntimeError("Receipt output readback or descriptor identity changed")
            os.fsync(parent.descriptor)
            parent.assert_stable()
            child = os.stat(path.name, dir_fd=parent.descriptor, follow_symlinks=False)
            if _file_identity(child) != identity:
                raise RuntimeError("Receipt output directory entry identity changed")
            _assert_path_identity(path, identity)
        finally:
            os.close(descriptor)


def write_fold_archive_inventory_receipt(
    input_manifest: Mapping[str, Any], *, output_path: Path
) -> dict[str, Any]:
    receipt = build_fold_archive_inventory_receipt(input_manifest)
    _write_absent_canonical(Path(output_path), receipt)
    return receipt


def _rename_no_replace(parent_descriptor: int, source_name: str, target_name: str) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic publication requires Linux renameat2")
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
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"Refusing to overwrite materialization output: {target_name}"
            )
        raise OSError(
            error_number,
            f"Atomic publication failed: {source_name} -> {target_name}",
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_output_root(output_root: Path) -> tuple[Path, Path]:
    output_root = Path(output_root)
    if (
        not output_root.is_absolute()
        or output_root.resolve(strict=False) != output_root
        or output_root.as_posix().startswith("//")
    ):
        raise ValueError("Materialization root must be one canonical absolute path")
    parent = output_root.parent
    _require_real_directory(parent, name="Materialization parent")
    incomplete = output_root.with_name(f".{output_root.name}.incomplete")
    if (
        output_root.exists()
        or output_root.is_symlink()
        or incomplete.exists()
        or incomplete.is_symlink()
    ):
        raise FileExistsError("Materialization output and sibling incomplete path must be absent")
    return output_root, incomplete


def _expectations_by_system(
    expectations: Mapping[str, ControlledArtifactExpectation],
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, ControlledArtifactExpectation]:
    if type(expectations) is not dict:
        raise TypeError("expectations must be one exact dict")
    system_ids = [record["system_id"] for record in manifest["systems"]]
    if list(expectations) != system_ids:
        raise ValueError("Externally supplied expectations have wrong order or coverage")
    source = manifest["source_bundle"]
    normalized: dict[str, ControlledArtifactExpectation] = {}
    for system, receipt_system in zip(manifest["systems"], receipt["systems"]):
        system_id = system["system_id"]
        expectation = expectations[system_id]
        if not isinstance(expectation, ControlledArtifactExpectation):
            raise TypeError(f"Expectation for {system_id!r} has the wrong type")
        cell = system["cell"]
        if (
            expectation.artifact_manifest_sha256
            != receipt_system["archive_evidence"]["artifact"]["artifact_manifest_sha256"]
            or expectation.training_plan_sha256 != manifest["training_plan_sha256"]
            or expectation.training_staging_receipt_sha256
            != manifest["training_staging_receipt_sha256"]
            or expectation.source_bundle_name != source["name"]
            or expectation.source_bundle_size != source["size"]
            or expectation.source_bundle_sha256 != source["sha256"]
            or expectation.source_bundle_inventory_sha256 != source["inventory_sha256"]
            or expectation.source_bundle_commit_epoch != source["commit_epoch"]
            or expectation.experiment_id != manifest["experiment_id"]
            or expectation.outer_fold != cell["outer_fold"]
            or expectation.query_view != cell["query_view"]
            or expectation.sampler != cell["sampler"]
            or expectation.experiment_seed != cell["experiment_seed"]
        ):
            raise ValueError(f"Expectation for {system_id!r} left external launch evidence")
        normalized[system_id] = expectation
    return normalized


def _extract_and_compare(
    archive_path: Path,
    *,
    expected_size: int,
    expected_evidence: Mapping[str, Any],
    extraction_root: Path,
) -> None:
    with _open_archive_snapshot(archive_path) as snapshot:
        size, digest = _hash_archive(snapshot)
        if size != expected_size:
            raise ValueError("Archive size changed before materialization")
        gzip_evidence = _audit_single_gzip(snapshot)
        tar_evidence, members, manifest_capture = _scan_tar(
            snapshot,
            gzip_evidence=gzip_evidence,
            extraction_root=extraction_root,
        )
        evidence = {
            "archive": {
                "size": size,
                "allocated_bytes": os.fstat(snapshot.descriptor).st_blocks * 512,
                "sha256": digest,
            },
            "gzip": gzip_evidence,
            "tar": tar_evidence,
            "members": members,
            "artifact": {},
        }
        files = sorted(
            (record for record in members if record["kind"] == "file"),
            key=lambda item: item["path"],
        )
        manifest_record = next(
            record for record in files if record["path"] == "artifact_manifest.json"
        )
        evidence["artifact"] = {
            "artifact_manifest_size": manifest_record["size"],
            "artifact_manifest_sha256": manifest_record["sha256"],
            "artifact_manifest_capture_sha256": hashlib.sha256(manifest_capture).hexdigest(),
            "file_count": len(files),
            "file_bytes": sum(record["size"] for record in files),
            "file_inventory_sha256": _document_sha256(files),
        }
        snapshot.assert_stable()
    if evidence != expected_evidence:
        raise ValueError("Materialization re-scan differs from the sealed archive evidence")


def _validated_files_payload(artifact: ValidatedControlledArtifact) -> list[dict[str, Any]]:
    return [
        {"kind": "file", "path": item.path, "size": item.size, "sha256": item.sha256}
        for item in artifact.files
    ]


def materialize_fold_archives(
    input_manifest: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    *,
    output_root: Path,
    expectations: Mapping[str, ControlledArtifactExpectation],
) -> FoldArchiveMaterialization:
    """Re-scan, extract, validate, and atomically publish twelve artifacts."""

    manifest = validate_fold_archive_input_manifest(copy.deepcopy(input_manifest))
    receipt = validate_fold_archive_inventory_receipt(
        copy.deepcopy(inventory_receipt), input_manifest=manifest
    )
    bound_expectations = _expectations_by_system(
        expectations, manifest=manifest, receipt=receipt
    )
    output_root, incomplete = _validate_output_root(Path(output_root))
    validated_after: list[ValidatedControlledArtifact] = []
    systems_payload: list[dict[str, Any]] = []
    with _open_directory_snapshot(
        output_root.parent, name="Materialization parent"
    ) as publication_parent:
        os.mkdir(incomplete.name, mode=0o700, dir_fd=publication_parent.descriptor)
        publication_parent.assert_stable()
        with _open_directory_snapshot(
            incomplete, name="Incomplete materialization root"
        ) as staging_root:
            for system, receipt_system in zip(
                manifest["systems"], receipt["systems"]
            ):
                publication_parent.assert_stable()
                staging_root.assert_stable()
                system_id = system["system_id"]
                artifact_root = incomplete / system_id
                _extract_and_compare(
                    Path(system["archive_path"]),
                    expected_size=system["destination_object"]["size"],
                    expected_evidence=receipt_system["archive_evidence"],
                    extraction_root=artifact_root,
                )
                staging_root.assert_stable()
                artifact = validate_controlled_artifact(
                    artifact_root, expectation=bound_expectations[system_id]
                )
                staging_root.assert_stable()
                expected_files = sorted(
                    (
                        record
                        for record in receipt_system["archive_evidence"]["members"]
                        if record["kind"] == "file"
                    ),
                    key=lambda item: item["path"],
                )
                if _validated_files_payload(artifact) != expected_files:
                    raise ValueError(
                        f"Validated artifact {system_id!r} differs from archive inventory"
                    )
            os.fsync(staging_root.descriptor)
            publication_parent.assert_stable()
            staging_root.assert_stable()
            _rename_no_replace(
                publication_parent.descriptor,
                incomplete.name,
                output_root.name,
            )
            staging_root.rebind(output_root, name="Published materialization root")
            os.fsync(publication_parent.descriptor)
            publication_parent.assert_stable()
            for system, receipt_system in zip(
                manifest["systems"], receipt["systems"]
            ):
                publication_parent.assert_stable()
                staging_root.assert_stable()
                system_id = system["system_id"]
                artifact = validate_controlled_artifact(
                    output_root / system_id,
                    expectation=bound_expectations[system_id],
                )
                staging_root.assert_stable()
                expected_files = sorted(
                    (
                        record
                        for record in receipt_system["archive_evidence"]["members"]
                        if record["kind"] == "file"
                    ),
                    key=lambda item: item["path"],
                )
                if _validated_files_payload(artifact) != expected_files:
                    raise ValueError(
                        f"Published artifact {system_id!r} differs from archive inventory"
                    )
                validated_after.append(artifact)
                systems_payload.append(
                    {
                        "system_id": system_id,
                        "artifact_root": str(output_root / system_id),
                        "identity": dataclasses.asdict(artifact.identity),
                    }
                )
    materialization = {
        "schema_version": 1,
        "protocol": FOLD_MATERIALIZATION_PROTOCOL,
        "input_manifest_sha256": _document_sha256(manifest),
        "inventory_receipt_sha256": receipt["receipt_sha256"],
        "experiment_id": manifest["experiment_id"],
        "outer_fold": manifest["outer_fold"],
        "artifact_root": str(output_root),
        "systems": systems_payload,
    }
    return FoldArchiveMaterialization(
        root=output_root,
        receipt={**materialization, "materialization_sha256": _document_sha256(materialization)},
        artifacts=tuple(validated_after),
    )


__all__ = [
    "ARCHIVE_INPUT_MANIFEST_PROTOCOL",
    "ARCHIVE_INVENTORY_RECEIPT_PROTOCOL",
    "FOLD_MATERIALIZATION_PROTOCOL",
    "FoldArchiveMaterialization",
    "MAX_ARCHIVE_BYTES",
    "MAX_STREAM_BYTES",
    "build_fold_archive_inventory_receipt",
    "load_fold_archive_input_manifest",
    "load_fold_archive_inventory_receipt",
    "materialize_fold_archives",
    "scan_controlled_archive",
    "validate_fold_archive_input_manifest",
    "validate_fold_archive_inventory_receipt",
    "write_fold_archive_inventory_receipt",
]
