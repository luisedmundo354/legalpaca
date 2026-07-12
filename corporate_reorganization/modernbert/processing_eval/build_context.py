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
from typing import Sequence


BASE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
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
MANIFEST_RELATIVE_PATH = "processing_eval/build_context_manifest.json"
MANIFEST_TYPE = "arr_retrieval_processing_build_context"
FREEZE_PROTOCOL = "frozen_absent_output_context_v1"
MANIFEST_SCHEMA_VERSION = 1
FILE_MODE = "0644"
DIRECTORY_MODE = "0755"
CORRETTO_ARCHIVE_SHA256 = "5b4dc8817df13f88f9bfc434e5d018adb535889ff2fe0ccf758bcebcc216f394"
CORRETTO_ARCHIVE_URL = (
    "https://corretto.aws/downloads/resources/21.0.11.10.1/"
    "amazon-corretto-21.0.11.10.1-linux-x64.tar.gz"
)
DOCKER_MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"

_CONTROL_PATHS = {
    "processing_eval/Dockerfile",
    "processing_eval/Dockerfile.dockerignore",
    "processing_eval/build_context.py",
    "processing_eval/build_requirements.lock",
    "processing_eval/requirements.lock",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_TOOL_VERSION_RE = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


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


def _canonical_contract_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} is not a normalized relative POSIX path: {value!r}")
    return value


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise ValueError(f"{name} is absent: {path}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} must not contain a symlink component: {current}")
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


def _run_git(git_root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(git_root), *arguments],
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
    try:
        value = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("Git root is not valid UTF-8") from exc
    root = _require_real_directory(Path(value), name="Git root")
    try:
        modernbert_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("ModernBERT directory is outside its Git worktree") from exc
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
    if drivers != ["docker"]:
        raise RuntimeError(
            f"Current buildx builder must use the exact docker driver: actual={drivers}"
        )
    buildkit_versions = re.findall(
        r"^BuildKit version:\s+"
        r"(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))$",
        inspect.stdout,
        flags=re.MULTILINE,
    )
    if len(buildkit_versions) != 1:
        raise RuntimeError(
            "Current buildx builder must expose exactly one BuildKit version: "
            f"actual={buildkit_versions}"
        )
    return {
        "builder_driver": "docker",
        "buildkit_version": buildkit_versions[0],
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
        raise ValueError("source_parent_commit must be an exact lowercase 40-hex commit ID")
    if type(source_parent_epoch) is not int or source_parent_epoch <= 0:
        raise ValueError("source_parent_epoch must be a positive integer")
    resolved_commit = _run_git(
        git_root,
        ("rev-parse", "--verify", f"{source_parent_commit}^{{commit}}"),
    ).decode("ascii", errors="strict").strip()
    if resolved_commit != source_parent_commit:
        raise ValueError(
            "source_parent_commit did not resolve to itself: "
            f"actual={source_parent_commit!r}, resolved={resolved_commit!r}"
        )
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
        raise ValueError("source_parent_commit must be an ancestor of the current HEAD")
    if ancestry.returncode != 0:
        raise RuntimeError(
            "Git ancestry validation failed: "
            f"{ancestry.stderr.decode('utf-8', errors='replace').strip()}"
        )
    parent_epoch_text = _run_git(
        git_root, ("show", "-s", "--format=%ct", source_parent_commit)
    ).decode("ascii", errors="strict").strip()
    if not parent_epoch_text.isdigit() or int(parent_epoch_text) != source_parent_epoch:
        raise ValueError(
            "source_parent_epoch must equal the selected ancestor commit timestamp: "
            f"actual={source_parent_epoch}, parent={parent_epoch_text!r}"
        )
    source_parent_rfc3339 = datetime.fromtimestamp(
        source_parent_epoch, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return source_parent_rfc3339, _local_toolchain_identity()


def _load_source_contract(modernbert_dir: Path) -> tuple[dict[str, object], list[str]]:
    path = modernbert_dir / "processing_eval/image_contract.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Image contract must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != _canonical_contract_bytes(value):
        raise ValueError("Image contract must be compact canonical JSON")
    if value.get("base_image", {}).get("uri") != BASE_IMAGE_URI:
        raise ValueError("Image contract base image changed")
    if value.get("dockerfile_frontend") != DOCKERFILE_FRONTEND:
        raise ValueError("Image contract Dockerfile frontend changed")
    if value.get("platform") != PLATFORM or value.get("build_exporter") != BUILD_EXPORTER:
        raise ValueError("Image contract platform/exporter changed")
    expected_manifest = {
        "directory_mode": DIRECTORY_MODE,
        "file_mode": FILE_MODE,
        "manifest_type": MANIFEST_TYPE,
        "path": MANIFEST_RELATIVE_PATH,
        "protocol": FREEZE_PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
    if value.get("build_manifest") != expected_manifest:
        raise ValueError("Image contract build-manifest protocol changed")
    source_inventory = value.get("source_inventory")
    if type(source_inventory) is not list:
        raise ValueError("Image contract source inventory is absent")
    paths = [_validate_relative_path(item, name="source_inventory entry") for item in source_inventory]
    if paths != sorted(set(paths)):
        raise ValueError("Image contract source inventory must be sorted and unique")
    if "processing_eval/build_context.py" not in paths:
        raise ValueError("Image contract must include build_context.py in the runtime source inventory")
    return value, sorted(_CONTROL_PATHS | set(paths))


def _read_live_sources(
    modernbert_dir: Path,
    *,
    paths: Sequence[str],
) -> dict[str, bytes]:
    opened: list[tuple[str, Path, int, os.stat_result]] = []

    def path_metadata(relative: str, source: Path) -> os.stat_result:
        current = modernbert_dir
        for part in PurePosixPath(relative).parts:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Build-context source contains a symlink: {current}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Build-context entry must be a regular non-symlink file: {source}"
            )
        return metadata

    def stable_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def validate_open_paths() -> None:
        for relative, source, _, opened_metadata in opened:
            if stable_identity(path_metadata(relative, source)) != stable_identity(
                opened_metadata
            ):
                raise RuntimeError(
                    f"Build-context source path changed during the inventory snapshot: {source}"
                )

    try:
        for relative in paths:
            source = modernbert_dir / relative
            path_metadata(relative, source)
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            opened_metadata = os.fstat(descriptor)
            if not stat.S_ISREG(opened_metadata.st_mode):
                os.close(descriptor)
                raise ValueError(f"Build-context source is not a regular file: {source}")
            mode = stat.S_IMODE(opened_metadata.st_mode)
            if mode & 0o111:
                os.close(descriptor)
                raise ValueError(
                    "Build-context source became executable in the worktree: "
                    f"path={relative!r}, mode={mode:04o}"
                )
            opened.append((relative, source, descriptor, opened_metadata))
        validate_open_paths()
        payloads: dict[str, bytes] = {}
        for relative, source, descriptor, opened_metadata in opened:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            live = b"".join(chunks)
            if (
                stable_identity(after) != stable_identity(opened_metadata)
                or len(live) != opened_metadata.st_size
            ):
                raise RuntimeError(f"Build-context source changed while it was read: {source}")
            payloads[relative] = live
        validate_open_paths()
        return payloads
    finally:
        for _, _, descriptor, _ in opened:
            os.close(descriptor)


def _file_records(payloads: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "mode": FILE_MODE,
            "path": relative,
            "sha256": _sha256_bytes(payload),
            "size": len(payload),
            "type": "regular_file",
        }
        for relative, payload in sorted(payloads.items())
    ]


def _identity_payload(manifest: dict[str, object]) -> dict[str, object]:
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
    payloads: dict[str, bytes],
    *,
    source_parent_commit: str,
    source_parent_epoch: int,
    source_parent_rfc3339: str,
    toolchain: dict[str, str],
) -> dict[str, object]:
    records = _file_records(payloads)
    manifest: dict[str, object] = {
        "base_image": BASE_IMAGE_URI,
        "dockerfile_frontend": DOCKERFILE_FRONTEND,
        "exporter": BUILD_EXPORTER,
        "files": records,
        "files_sha256": _sha256_bytes(_canonical_json(records).encode("utf-8")),
        "manifest_type": MANIFEST_TYPE,
        "platform": PLATFORM,
        "protocol": FREEZE_PROTOCOL,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_parent_commit": source_parent_commit,
        "source_parent_epoch": source_parent_epoch,
        "source_parent_rfc3339": source_parent_rfc3339,
        "toolchain": toolchain,
    }
    identity = _sha256_bytes(_canonical_json(_identity_payload(manifest)).encode("utf-8"))
    manifest["build_identity_sha256"] = identity
    manifest["content_tag"] = f"build-sha256-{identity}"
    return manifest


def _validate_manifest_value(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("Build-context manifest must be a JSON object")
    manifest = value
    if set(manifest) != {
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
    }:
        raise ValueError("Build-context manifest schema changed")
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
        raise ValueError("Build-context fixed identity fields changed")
    toolchain = manifest["toolchain"]
    if type(toolchain) is not dict or set(toolchain) != {
        "builder_driver",
        "buildkit_version",
        "buildx_version",
    }:
        raise ValueError("Build-context toolchain schema changed")
    if toolchain["builder_driver"] != "docker":
        raise ValueError("Build-context builder driver changed")
    for name in ("buildkit_version", "buildx_version"):
        version = toolchain[name]
        if type(version) is not str or _TOOL_VERSION_RE.fullmatch(version) is None:
            raise ValueError(f"Build-context {name} is malformed")
    source_parent_commit = manifest["source_parent_commit"]
    if (
        type(source_parent_commit) is not str
        or _COMMIT_RE.fullmatch(source_parent_commit) is None
    ):
        raise ValueError("Build-context source_parent_commit is malformed")
    epoch = manifest["source_parent_epoch"]
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("Build-context source_parent_epoch is malformed")
    expected_rfc3339 = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if manifest["source_parent_rfc3339"] != expected_rfc3339:
        raise ValueError("Build-context source_parent_rfc3339 is not derived from its epoch")
    files = manifest["files"]
    if type(files) is not list or not files:
        raise ValueError("Build-context files must be a non-empty list")
    paths: list[str] = []
    for index, record in enumerate(files):
        if type(record) is not dict or set(record) != {
            "mode",
            "path",
            "sha256",
            "size",
            "type",
        }:
            raise ValueError(f"Build-context file record {index} schema changed")
        path = _validate_relative_path(record["path"], name=f"files[{index}].path")
        if (
            record["type"] != "regular_file"
            or record["mode"] != FILE_MODE
            or type(record["size"]) is not int
            or record["size"] < 0
            or type(record["sha256"]) is not str
            or _SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise ValueError(f"Build-context file record {index} is malformed")
        paths.append(path)
    if paths != sorted(set(paths)) or MANIFEST_RELATIVE_PATH in paths:
        raise ValueError("Build-context file records are not sorted, unique, and self-excluding")
    expected_files_sha256 = _sha256_bytes(_canonical_json(files).encode("utf-8"))
    if manifest["files_sha256"] != expected_files_sha256:
        raise ValueError("Build-context files_sha256 changed")
    expected_identity = _sha256_bytes(
        _canonical_json(_identity_payload(manifest)).encode("utf-8")
    )
    if manifest["build_identity_sha256"] != expected_identity:
        raise ValueError("Build-context identity hash changed")
    if manifest["content_tag"] != f"build-sha256-{expected_identity}":
        raise ValueError("Build-context content tag changed")
    return manifest


def load_build_context_manifest(path: Path) -> dict[str, object]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Build-context manifest must be a regular non-symlink file: {path}")
    if stat.S_IMODE(path.stat().st_mode) != 0o644:
        raise ValueError("Build-context manifest mode must be 0644")
    raw = path.read_bytes()
    value = json.loads(raw)
    if raw != _canonical_pretty_bytes(value):
        raise ValueError("Build-context manifest must be canonical pretty JSON")
    return _validate_manifest_value(value)


def _expected_directories(file_paths: Sequence[str]) -> set[str]:
    directories = {"."}
    for relative in file_paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _validate_exact_tree(root: Path, manifest: dict[str, object]) -> None:
    root = _require_real_directory(root, name="Frozen build-context root")
    records = {record["path"]: record for record in manifest["files"]}
    expected_files = set(records) | {MANIFEST_RELATIVE_PATH}
    expected_directories = _expected_directories(sorted(expected_files))
    actual_files: set[str] = set()
    actual_directories = {"."}
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root).as_posix()
        if relative_current == ".":
            relative_current = "."
        if stat.S_IMODE(current_path.stat().st_mode) != 0o755:
            raise ValueError(f"Frozen build-context directory mode changed: {relative_current}")
        for name in tuple(directory_names):
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError(f"Frozen build-context contains a non-directory: {relative}")
            actual_directories.add(relative)
        for name in file_names:
            child = current_path / name
            relative = child.relative_to(root).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(f"Frozen build-context contains a non-regular file: {relative}")
            actual_files.add(relative)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "Frozen build-context inventory changed: "
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
            raise ValueError(f"Frozen build-context file identity changed: {relative}")
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o644:
        raise ValueError("Frozen build-context manifest mode changed")


def validate_frozen_build_context(root: Path) -> dict[str, object]:
    root = _require_real_directory(root, name="Frozen build-context root")
    manifest = load_build_context_manifest(root / MANIFEST_RELATIVE_PATH)
    _validate_exact_tree(root, manifest)
    return manifest


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
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
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Linux renameat2 is required for atomic absent-output publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"Build-context output appeared during publication: {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _validate_build_metadata(
    metadata_path: Path,
    *,
    manifest: dict[str, object],
    image_name: str,
) -> dict[str, str]:
    metadata_path = Path(metadata_path)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise RuntimeError(f"Buildx metadata output is absent or a symlink: {metadata_path}")
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
    if (
        type(config_digest) is not str
        or not config_digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(config_digest.removeprefix("sha256:")) is None
        or type(image_digest) is not str
        or not image_digest.startswith("sha256:")
        or _SHA256_RE.fullmatch(image_digest.removeprefix("sha256:")) is None
    ):
        raise RuntimeError("Buildx returned malformed image digests")
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
        raise RuntimeError("Buildx image descriptor annotations changed")
    expected_metadata_name = f"docker.io/library/{image_name}"
    if metadata["image.name"] != expected_metadata_name:
        raise RuntimeError(
            f"Buildx image name changed: actual={metadata['image.name']!r}, "
            f"expected={expected_metadata_name!r}"
        )
    build_ref = metadata["buildx.build.ref"]
    if type(build_ref) is not str or re.fullmatch(r"[^/\s]+/[^/\s]+/[a-z0-9]+", build_ref) is None:
        raise RuntimeError("Buildx build reference is malformed")
    frontend_digest = DOCKERFILE_FRONTEND.rsplit("@sha256:", 1)[1]
    base_digest = BASE_IMAGE_URI.rsplit("@sha256:", 1)[1]
    expected_provenance = {
        "buildType": "https://mobyproject.org/buildkit@v1",
        "builder": {"id": ""},
        "invocation": {
            "configSource": {"entryPoint": "Dockerfile"},
            "environment": {"platform": PLATFORM},
            "parameters": {
                "args": {
                    "build-arg:BUILD_IDENTITY_SHA256": manifest[
                        "build_identity_sha256"
                    ],
                    "build-arg:SOURCE_DATE_EPOCH": str(
                        manifest["source_parent_epoch"]
                    ),
                    "build-arg:SOURCE_PARENT_COMMIT": manifest[
                        "source_parent_commit"
                    ],
                    "build-arg:SOURCE_PARENT_EPOCH": str(manifest["source_parent_epoch"]),
                    "build-arg:SOURCE_PARENT_RFC3339": manifest[
                        "source_parent_rfc3339"
                    ],
                    "cmdline": DOCKERFILE_FRONTEND,
                    "no-cache": "",
                    "source": DOCKERFILE_FRONTEND,
                },
                "frontend": "gateway.v0",
                "locals": [{"name": "context"}, {"name": "dockerfile"}],
            },
        },
        "materials": [
            {
                "digest": {"sha256": base_digest},
                "uri": (
                    "pkg:docker/763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                    "huggingface-pytorch-training?digest=sha256:"
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
                "uri": (
                    "pkg:docker/docker/dockerfile@1.7?digest=sha256:"
                    f"{frontend_digest}"
                ),
            },
            {
                "digest": {"sha256": CORRETTO_ARCHIVE_SHA256},
                "uri": CORRETTO_ARCHIVE_URL,
            },
        ],
    }
    if metadata["buildx.build.provenance"] != expected_provenance:
        raise RuntimeError("Buildx provenance does not match the frozen build command")
    return {
        "build_ref": build_ref,
        "config_digest": config_digest,
        "image_digest": image_digest,
        "image_name": expected_metadata_name,
        "manifest_media_type": DOCKER_MANIFEST_MEDIA_TYPE,
    }


def _buildx_command(
    frozen_context: Path,
    metadata_file: Path,
    *,
    manifest: dict[str, object],
    build_replica: int,
) -> tuple[list[str], str]:
    if type(build_replica) is not int or build_replica not in {1, 2}:
        raise ValueError("build_replica must be exact integer 1 or 2")
    image_name = (
        f"arr-retrieval-eval:{manifest['content_tag']}-build{build_replica}"
    )
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
    command = [
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
        str(frozen_context / "processing_eval/Dockerfile"),
        str(frozen_context),
    ]
    return command, image_name


def _validate_local_image(
    image_name: str,
    *,
    manifest: dict[str, object],
    contract: dict[str, object],
    build_metadata: dict[str, str],
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
    ):
        raise RuntimeError("Locally stored image identity/platform changed")
    repo_digests = image.get("RepoDigests")
    expected_repo_digests = [
        f"arr-retrieval-eval@{build_metadata['image_digest']}"
    ]
    if repo_digests != expected_repo_digests:
        raise RuntimeError(
            "Locally stored image RepoDigests changed: "
            f"actual={repo_digests}, expected={expected_repo_digests}"
        )
    config = image.get("Config")
    if type(config) is not dict:
        raise RuntimeError("Locally stored image config is absent")
    if (
        config.get("Entrypoint") != contract["entrypoint"]
        or config.get("WorkingDir") != contract["workdir"]
    ):
        raise RuntimeError("Locally stored image entrypoint/workdir changed")
    raw_environment = config.get("Env")
    if type(raw_environment) is not list or any(
        type(item) is not str or "=" not in item for item in raw_environment
    ):
        raise RuntimeError("Locally stored image environment schema changed")
    environment: dict[str, str] = {}
    for item in raw_environment:
        name, value = item.split("=", 1)
        if name in environment:
            raise RuntimeError(f"Locally stored image environment duplicates {name!r}")
        environment[name] = value
    expected_environment = {
        **contract["environment"],
        "SOURCE_DATE_EPOCH": str(manifest["source_parent_epoch"]),
    }
    if any(environment.get(name) != value for name, value in expected_environment.items()):
        raise RuntimeError("Locally stored image deterministic environment changed")
    labels = config.get("Labels")
    expected_labels = {
        "io.arr-retrieval.build-identity-sha256": manifest[
            "build_identity_sha256"
        ],
        "io.arr-retrieval.source-parent-commit": manifest["source_parent_commit"],
        "io.arr-retrieval.source-parent-epoch": str(manifest["source_parent_epoch"]),
        "io.arr-retrieval.source-parent-rfc3339": manifest["source_parent_rfc3339"],
        "org.opencontainers.image.base.digest": contract["base_image"]["digest"],
    }
    if type(labels) is not dict or any(
        labels.get(name) != value for name, value in expected_labels.items()
    ):
        raise RuntimeError("Locally stored image provenance labels changed")
    return {
        "environment": expected_environment,
        "labels": expected_labels,
        "repo_digests": repo_digests,
    }


def build_frozen_image(
    frozen_context: Path,
    metadata_file: Path,
    *,
    build_replica: int,
) -> dict[str, str]:
    frozen_context = _require_real_directory(
        frozen_context, name="Frozen image build context"
    )
    metadata_file = Path(metadata_file).absolute()
    _require_real_directory(metadata_file.parent, name="Build metadata parent")
    if metadata_file.exists() or metadata_file.is_symlink():
        raise FileExistsError(f"Build metadata output must be absent: {metadata_file}")
    manifest = validate_frozen_build_context(frozen_context)
    contract, _ = _load_source_contract(frozen_context)
    actual_toolchain = _local_toolchain_identity()
    if actual_toolchain != manifest["toolchain"]:
        raise RuntimeError(
            "Active Docker builder differs from the frozen toolchain: "
            f"actual={actual_toolchain}, expected={manifest['toolchain']}"
        )
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
        raise RuntimeError(f"Exact frozen docker buildx build failed: {completed.returncode}")
    if validate_frozen_build_context(frozen_context) != manifest:
        raise RuntimeError("Frozen build context changed while BuildKit consumed it")
    build_metadata = _validate_build_metadata(
        metadata_file,
        manifest=manifest,
        image_name=image_name,
    )
    local_image = _validate_local_image(
        image_name,
        manifest=manifest,
        contract=contract,
        build_metadata=build_metadata,
    )
    return {
        **build_metadata,
        "local_image_identity_sha256": _sha256_bytes(
            _canonical_json(local_image).encode("utf-8")
        ),
    }


def freeze_build_context(
    modernbert_dir: Path,
    output_dir: Path,
    *,
    source_parent_commit: str,
    source_parent_epoch: int,
) -> dict[str, object]:
    modernbert_dir = _require_real_directory(modernbert_dir, name="ModernBERT build root")
    output_dir, incomplete = _require_absent_output(output_dir)
    git_root = _git_root(modernbert_dir)
    source_parent_rfc3339, toolchain = _validate_parent_inputs(
        git_root,
        source_parent_commit=source_parent_commit,
        source_parent_epoch=source_parent_epoch,
    )
    _, paths = _load_source_contract(modernbert_dir)
    payloads = _read_live_sources(
        modernbert_dir,
        paths=paths,
    )
    if _read_live_sources(modernbert_dir, paths=paths) != payloads:
        raise RuntimeError("Build-context source inventory changed between complete reads")
    manifest = _build_manifest(
        payloads,
        source_parent_commit=source_parent_commit,
        source_parent_epoch=source_parent_epoch,
        source_parent_rfc3339=source_parent_rfc3339,
        toolchain=toolchain,
    )
    owns_incomplete = False
    incomplete_identity: tuple[int, int] | None = None
    try:
        incomplete.mkdir(mode=0o700)
        owns_incomplete = True
        incomplete_metadata = incomplete.lstat()
        incomplete_identity = (incomplete_metadata.st_dev, incomplete_metadata.st_ino)
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
            diagnostic_root = Path(
                tempfile.mkdtemp(
                    dir=output_dir.parent,
                    prefix=f".{output_dir.name}.invalid.",
                )
            )
            _fsync_directory(output_dir.parent)
            marker = output_dir / MANIFEST_RELATIVE_PATH
            marker.unlink()
            _fsync_directory(marker.parent)
            _fsync_directory(output_dir)
            _rename_no_replace(output_dir, diagnostic_root / "context")
            _fsync_directory(diagnostic_root)
            _fsync_directory(output_dir.parent)
            raise
    finally:
        if owns_incomplete:
            current = incomplete.lstat()
            current_identity = (current.st_dev, current.st_ino)
            if current_identity != incomplete_identity:
                raise RuntimeError(
                    "Owned incomplete build-context path was replaced; refusing cleanup"
                )
            shutil.rmtree(incomplete)
            _fsync_directory(incomplete.parent)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the exact Processing image bytes and bind their parent provenance to "
            "a selected ancestor of HEAD."
        ),
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
            raise ValueError("--build-frozen-context cannot be combined with freeze/validate inputs")
        if args.metadata_file is None or args.build_replica is None:
            raise ValueError("--metadata-file and --build-replica are required for image build")
        payload = build_frozen_image(
            args.build_frozen_context,
            args.metadata_file,
            build_replica=args.build_replica,
        )
        print(_canonical_json(payload))
        return 0
    if args.metadata_file is not None or args.build_replica is not None:
        raise ValueError("Build metadata/replica inputs require --build-frozen-context")
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
