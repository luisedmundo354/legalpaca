"""Fail-closed bootstrap for network-isolated retrieval training.

The SageMaker training toolkit executes this baked program from a local
``submit_directory``.  Every MPI rank independently verifies and extracts the
same immutable source-channel archive into its own previously absent directory,
then replaces itself with the verified ``train_sm.py`` process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import sys
import tarfile
import zlib
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Callable, Mapping, Sequence


BOOTSTRAP_PROTOCOL = "arr_retrieval_training_source_bootstrap_v1"
BOOTSTRAP_PATH = Path("/opt/training_bootstrap/bootstrap.py")
ACTIVE_BOOTSTRAP_PATH = Path(__file__)
RUNTIME_CONTRACT_PATH = Path("/opt/training_image/runtime_contract.py")
IMAGE_CONTRACT_PATH = Path("/opt/training_image/image_contract.json")
RUNTIME_INVENTORY_PATH = Path("/opt/training_image/runtime_inventory.json")
EXTRACTION_PARENT = Path("/opt/ml/code")
SOURCE_CHANNEL_PATH = Path("/opt/ml/input/data/source")
ENTRYPOINT_NAME = "train_sm.py"
WORLD_SIZE = 4

SOURCE_IDENTITY_ENV = {
    "name": "ARR_SOURCE_BUNDLE_NAME",
    "size": "ARR_SOURCE_BUNDLE_SIZE",
    "sha256": "ARR_SOURCE_BUNDLE_SHA256",
    "inventory_sha256": "ARR_SOURCE_INVENTORY_SHA256",
    "commit_epoch": "ARR_SOURCE_COMMIT_EPOCH",
}
VERIFIED_ENV = {
    "name": "ARR_VERIFIED_SOURCE_BUNDLE_NAME",
    "size": "ARR_VERIFIED_SOURCE_BUNDLE_SIZE",
    "sha256": "ARR_VERIFIED_SOURCE_BUNDLE_SHA256",
    "inventory_sha256": "ARR_VERIFIED_SOURCE_INVENTORY_SHA256",
    "commit_epoch": "ARR_VERIFIED_SOURCE_COMMIT_EPOCH",
    "contract_sha256": "ARR_VERIFIED_TRAINING_CONTRACT_SHA256",
    "runtime_inventory_sha256": (
        "ARR_VERIFIED_TRAINING_RUNTIME_INVENTORY_SHA256"
    ),
    "protocol": "ARR_VERIFIED_TRAINING_BOOTSTRAP_PROTOCOL",
}
EXPECTED_RUNTIME_INVENTORY_ENV = "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_DECIMAL_RE = re.compile(r"[1-9][0-9]*\Z")
_SOURCE_NAME_RE = re.compile(r"source-([0-9a-f]{64})\.tar\.gz\Z")


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


def _read_regular_file(path: Path, *, expected_mode: int | None = None) -> bytes:
    path = Path(path)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise ValueError(
            f"File mode changed for {path}: "
            f"actual={stat.S_IMODE(metadata.st_mode):04o}, expected={expected_mode:04o}"
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or (
            observed.st_dev,
            observed.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError(f"File identity changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_runtime_contract(path: Path) -> ModuleType:
    _read_regular_file(path, expected_mode=0o644)
    specification = importlib.util.spec_from_file_location(
        "arr_training_runtime_contract", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load the baked runtime contract: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_canonical_json(path: Path) -> tuple[dict[str, object], bytes]:
    raw = _read_regular_file(path, expected_mode=0o644)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Baked JSON is invalid: {path}") from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError(f"Baked JSON must be one compact canonical object: {path}")
    return value, raw


def _verify_baked_runtime(
    *,
    environ: Mapping[str, str],
    runtime_contract_path: Path,
    image_contract_path: Path,
    runtime_inventory_path: Path,
    bootstrap_path: Path,
    active_bootstrap_path: Path,
) -> tuple[str, str]:
    runtime_contract = _load_runtime_contract(runtime_contract_path)
    contract, contract_sha256 = runtime_contract.load_contract(image_contract_path)
    expected_bootstrap = contract["bootstrap"]
    bootstrap_raw = _read_regular_file(bootstrap_path, expected_mode=0o555)
    actual_bootstrap = {
        "entrypoint": ENTRYPOINT_NAME,
        "path": str(BOOTSTRAP_PATH),
        "protocol": BOOTSTRAP_PROTOCOL,
        "sha256": _sha256_bytes(bootstrap_raw),
    }
    if expected_bootstrap != actual_bootstrap:
        raise RuntimeError("Baked training bootstrap identity changed")
    active_bootstrap_raw = _read_regular_file(active_bootstrap_path)
    if _sha256_bytes(active_bootstrap_raw) != expected_bootstrap["sha256"]:
        raise RuntimeError("Active training bootstrap identity changed")

    inventory, inventory_raw = _load_canonical_json(runtime_inventory_path)
    runtime_contract.validate_inventory(
        inventory,
        contract_sha256=contract_sha256,
    )
    if inventory["bootstrap"] != expected_bootstrap:
        raise RuntimeError("Runtime inventory bootstrap identity changed")
    inventory_sha256 = _sha256_bytes(inventory_raw)
    expected_inventory_sha256 = environ.get(EXPECTED_RUNTIME_INVENTORY_ENV)
    if (
        type(expected_inventory_sha256) is not str
        or _SHA256_RE.fullmatch(expected_inventory_sha256) is None
        or inventory_sha256 != expected_inventory_sha256
    ):
        raise RuntimeError(
            "Baked runtime inventory digest differs from the requested image identity"
        )

    baked_environment = inventory["environment"]
    if baked_environment.get("PYTHONHASHSEED") != "17":
        raise RuntimeError("Baked runtime inventory must use image-default seed 17")
    live_environment = {
        name: environ.get(name) for name in sorted(baked_environment)
    }
    live_inventory = dict(inventory)
    live_inventory["environment"] = live_environment
    runtime_contract.validate_inventory(
        live_inventory,
        contract_sha256=contract_sha256,
    )
    normalized_baked = dict(baked_environment)
    normalized_live = dict(live_environment)
    normalized_baked["PYTHONHASHSEED"] = "<allowed-controlled-seed>"
    normalized_live["PYTHONHASHSEED"] = "<allowed-controlled-seed>"
    if normalized_live != normalized_baked:
        raise RuntimeError(
            "Live training environment differs from the baked runtime inventory"
        )
    return contract_sha256, inventory_sha256


def _required_environment(environ: Mapping[str, str]) -> dict[str, str | int]:
    values: dict[str, str | int] = {}
    for logical_name, environment_name in SOURCE_IDENTITY_ENV.items():
        value = environ.get(environment_name)
        if type(value) is not str:
            raise RuntimeError(f"Missing strict source identity: {environment_name}")
        values[logical_name] = value
    name = values["name"]
    digest = values["sha256"]
    if (
        type(name) is not str
        or (match := _SOURCE_NAME_RE.fullmatch(name)) is None
        or type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
        or match.group(1) != digest
    ):
        raise ValueError("Source archive name and SHA-256 identity differ")
    inventory_sha256 = values["inventory_sha256"]
    if (
        type(inventory_sha256) is not str
        or _SHA256_RE.fullmatch(inventory_sha256) is None
    ):
        raise ValueError("Source inventory SHA-256 is not canonical")
    for logical_name in ("size", "commit_epoch"):
        value = values[logical_name]
        if type(value) is not str or _POSITIVE_DECIMAL_RE.fullmatch(value) is None:
            raise ValueError(f"Source {logical_name} must be a positive canonical decimal")
        values[logical_name] = int(value)
    if values["commit_epoch"] > 0xFFFFFFFF:
        raise ValueError("Source commit epoch does not fit the normalized gzip header")
    return values


def _distributed_rank(environ: Mapping[str, str]) -> int:
    coordinates: dict[str, int] = {}
    for source, destination, logical_name in (
        ("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK", "local_rank"),
        ("OMPI_COMM_WORLD_RANK", "RANK", "rank"),
        ("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE", "world_size"),
    ):
        source_value = environ.get(source)
        destination_value = environ.get(destination)
        if source_value is None:
            raise RuntimeError(f"Missing required MPI coordinate: {source}")
        if destination_value is not None and destination_value != source_value:
            raise RuntimeError(
                f"Conflicting MPI coordinate: {source}={source_value!r}, "
                f"{destination}={destination_value!r}"
            )
        if not source_value.isascii() or not source_value.isdecimal():
            raise ValueError(f"MPI coordinate is not a canonical decimal: {source}")
        if str(int(source_value)) != source_value:
            raise ValueError(f"MPI coordinate is not canonical: {source}")
        coordinates[logical_name] = int(source_value)
    if (
        coordinates["world_size"] != WORLD_SIZE
        or coordinates["rank"] not in range(WORLD_SIZE)
        or coordinates["local_rank"] != coordinates["rank"]
    ):
        raise RuntimeError(
            "Training bootstrap requires exactly four local MPI ranks: "
            f"{coordinates}"
        )
    return coordinates["rank"]


def _source_archive(channel: Path, identity: Mapping[str, str | int]) -> bytes:
    channel = Path(channel)
    metadata = channel.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Source channel must be one real directory: {channel}")
    entries = list(channel.iterdir())
    expected_name = identity["name"]
    if len(entries) != 1 or entries[0].name != expected_name:
        raise ValueError(
            "Source channel must contain exactly the named source archive: "
            f"actual={sorted(path.name for path in entries)}, expected={expected_name!r}"
        )
    raw = _read_regular_file(entries[0], expected_mode=0o644)
    if len(raw) != identity["size"] or _sha256_bytes(raw) != identity["sha256"]:
        raise ValueError("Source archive bytes differ from the strict source identity")
    return raw


def _relative_path(name: str) -> str:
    if type(name) is not str or not name or "\\" in name or name.startswith("/"):
        raise ValueError(f"Unsafe source member path: {name!r}")
    if name.endswith("/"):
        name = name[:-1]
    path = PurePosixPath(name)
    if (
        not name
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Source member path is not normalized: {name!r}")
    return name


def _read_normalized_archive(
    raw: bytes,
    *,
    expected_epoch: int,
    expected_inventory_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    if len(raw) < 18 or raw[:3] != b"\x1f\x8b\x08" or raw[3] != 0:
        raise ValueError("Source bundle is not one normalized gzip stream")
    if int.from_bytes(raw[4:8], "little") != expected_epoch:
        raise ValueError("Source bundle gzip mtime differs from the source commit epoch")
    if raw[8] != 0 or raw[9] != 255:
        raise ValueError("Source bundle gzip header is not normalized")
    try:
        decompressor = zlib.decompressobj(wbits=31)
        tar_payload = decompressor.decompress(raw) + decompressor.flush()
    except zlib.error as error:
        raise ValueError("Source bundle gzip stream is invalid") from error
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("Source bundle must contain exactly one gzip member")

    inventory: list[dict[str, object]] = []
    contents: dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        archive_context = tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:")
    except tarfile.TarError as error:
        raise ValueError("Source bundle tar stream is invalid") from error
    with archive_context as archive:
        try:
            members = list(archive)
        except (tarfile.TarError, OSError) as error:
            raise ValueError("Source bundle tar members are invalid") from error
        for member in members:
            relative = _relative_path(member.name)
            if relative in seen:
                raise ValueError(f"Source bundle contains duplicate member: {relative}")
            seen.add(relative)
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != expected_epoch
            ):
                raise ValueError(f"Source member metadata is not normalized: {relative}")
            if member.isdir():
                if member.mode != 0o755 or member.size != 0:
                    raise ValueError(f"Source directory is not normalized: {relative}")
                inventory.append(
                    {"mode": "0755", "path": relative, "type": "directory"}
                )
            elif member.isfile():
                if member.mode != 0o644:
                    raise ValueError(f"Source file mode is not normalized: {relative}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Source file cannot be read: {relative}")
                payload = stream.read()
                if len(payload) != member.size:
                    raise ValueError(f"Source file is truncated: {relative}")
                contents[relative] = payload
                inventory.append(
                    {
                        "mode": "0644",
                        "path": relative,
                        "sha256": _sha256_bytes(payload),
                        "size": len(payload),
                        "type": "file",
                    }
                )
            else:
                raise ValueError(f"Source bundle forbids links and special members: {relative}")
    paths = [record["path"] for record in inventory]
    if paths != sorted(paths):
        raise ValueError("Source bundle members must be sorted")
    for record in inventory:
        relative = PurePosixPath(str(record["path"]))
        parents = [parent.as_posix() for parent in relative.parents if parent.as_posix() != "."]
        for parent in parents:
            matching = [row for row in inventory if row["path"] == parent]
            if len(matching) != 1 or matching[0]["type"] != "directory":
                raise ValueError(f"Source bundle omits normalized parent directory: {parent}")
    if ENTRYPOINT_NAME not in contents:
        raise ValueError(f"Source bundle omits the fixed entry point: {ENTRYPOINT_NAME}")
    inventory_sha256 = _sha256_bytes(_canonical_bytes(inventory))
    if inventory_sha256 != expected_inventory_sha256:
        raise ValueError("Source bundle inventory differs from the strict source identity")
    return inventory, contents


def _require_real_directory(path: Path) -> None:
    metadata = Path(path).lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Extraction parent must be one real directory: {path}")


def _extract_absent(
    parent: Path,
    *,
    rank: int,
    inventory: Sequence[Mapping[str, object]],
    contents: Mapping[str, bytes],
) -> Path:
    parent = Path(parent)
    _require_real_directory(parent)
    destination = parent / f"arr-source-rank-{rank}"
    if os.path.lexists(destination):
        raise FileExistsError(f"Per-rank source destination must be absent: {destination}")
    os.mkdir(destination, 0o700)
    for record in inventory:
        relative = str(record["path"])
        target = destination.joinpath(*PurePosixPath(relative).parts)
        if record["type"] == "directory":
            os.mkdir(target, 0o700)
            os.chmod(target, 0o755)
        else:
            descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o644,
            )
            try:
                payload = contents[relative]
                os.fchmod(descriptor, 0o644)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
    os.chmod(destination, 0o755)
    return destination


def run(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    runtime_contract_path: Path = RUNTIME_CONTRACT_PATH,
    image_contract_path: Path = IMAGE_CONTRACT_PATH,
    runtime_inventory_path: Path = RUNTIME_INVENTORY_PATH,
    bootstrap_path: Path = BOOTSTRAP_PATH,
    active_bootstrap_path: Path = ACTIVE_BOOTSTRAP_PATH,
    source_channel_path: Path = SOURCE_CHANNEL_PATH,
    extraction_parent: Path = EXTRACTION_PARENT,
    execvpe: Callable[[str, Sequence[str], Mapping[str, str]], object] = os.execvpe,
) -> object:
    contract_sha256, inventory_sha256 = _verify_baked_runtime(
        environ=environ,
        runtime_contract_path=runtime_contract_path,
        image_contract_path=image_contract_path,
        runtime_inventory_path=runtime_inventory_path,
        bootstrap_path=bootstrap_path,
        active_bootstrap_path=active_bootstrap_path,
    )
    if environ.get("SM_CHANNEL_SOURCE") != str(source_channel_path):
        raise RuntimeError(
            "SM_CHANNEL_SOURCE differs from the exact source channel mount: "
            f"actual={environ.get('SM_CHANNEL_SOURCE')!r}, "
            f"expected={str(source_channel_path)!r}"
        )
    identity = _required_environment(environ)
    rank = _distributed_rank(environ)
    raw = _source_archive(source_channel_path, identity)
    inventory, contents = _read_normalized_archive(
        raw,
        expected_epoch=int(identity["commit_epoch"]),
        expected_inventory_sha256=str(identity["inventory_sha256"]),
    )
    source_root = _extract_absent(
        extraction_parent,
        rank=rank,
        inventory=inventory,
        contents=contents,
    )
    entrypoint = source_root / ENTRYPOINT_NAME
    child_environment = dict(environ)
    child_environment.update(
        {
            VERIFIED_ENV["name"]: str(identity["name"]),
            VERIFIED_ENV["size"]: str(identity["size"]),
            VERIFIED_ENV["sha256"]: str(identity["sha256"]),
            VERIFIED_ENV["inventory_sha256"]: str(identity["inventory_sha256"]),
            VERIFIED_ENV["commit_epoch"]: str(identity["commit_epoch"]),
            VERIFIED_ENV["contract_sha256"]: contract_sha256,
            VERIFIED_ENV["runtime_inventory_sha256"]: inventory_sha256,
            VERIFIED_ENV["protocol"]: BOOTSTRAP_PROTOCOL,
        }
    )
    command = [sys.executable, str(entrypoint), *list(argv[1:])]
    return execvpe(sys.executable, command, child_environment)


def main(argv: Sequence[str] | None = None) -> int:
    run(sys.argv if argv is None else argv, environ=os.environ)
    raise RuntimeError("Verified training entry point returned after exec")


if __name__ == "__main__":
    raise SystemExit(main())
