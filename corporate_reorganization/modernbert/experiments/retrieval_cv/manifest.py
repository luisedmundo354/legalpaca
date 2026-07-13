"""Deterministic source bundles and immutable retrieval-CV training plans."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import platform
import re
import stat
import subprocess
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import config as strict_config


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_TYPE = "retrieval_cv_training_plan"
CONTROLLED_KIND = "controlled_full"
LEGACY_KIND = "corrected_legacy_diagnostic"
SMOKE_KIND = "determinism_smoke"
EXECUTION_BLOCKERS: tuple[str, ...] = ()
EXPECTED_BUNDLER_RUNTIME = {
    "python": "3.11.13",
    "zlib_compile": "1.2.13",
    "zlib_runtime": "1.2.13",
}
EXPECTED_SOURCE_INCLUDES = (
    "corrected_legacy_train.py",
    "ds_zero3.json",
    "experiments/retrieval_cv/configs/corrected_legacy.json",
    "experiments/retrieval_cv/configs/corrected_legacy_membership",
    "experiments/retrieval_cv/configs/experiment.json",
    "experiments/retrieval_cv/configs/folds.json",
    "experiments/retrieval_cv/configs/modernbert_snapshot.json",
    "experiments/retrieval_cv/corrected_legacy_config.py",
    "legacy_diagnostic_trainer.py",
    "legacy_eval",
    "legacy_train_sm.py",
    "retriever",
    "tests",
    "train_sm.py",
    "trainer.py",
)

_ATTEMPT_RE = re.compile(r"a([1-9][0-9]*)\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_JOB_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_VIEW_ALIAS = {"flat_masked": "flat", "structured": "struct"}
_SAMPLER_ALIAS = {"local_unique": "local", "global_uniform": "global"}


@dataclass(frozen=True)
class SourceBundle:
    path: Path
    size: int
    sha256: str
    inventory: tuple[dict[str, Any], ...]
    inventory_sha256: str
    commit_epoch: int
    bundler_runtime: dict[str, str]


def source_bundler_runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    }


def _load_exact_json_object(path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    raw = strict_config._read_regular_file_once(Path(path))
    digest = strict_config.sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"JSON hash mismatch for {path}: actual={digest}, expected={expected_sha256}"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=strict_config._reject_duplicate_keys,
            parse_constant=strict_config._reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON is not UTF-8: {path}") from exc
    if type(value) is not dict:
        raise TypeError(f"JSON must contain one object: {path}")
    strict_config._validate_json_value(value, name=str(path))
    return value, digest


def validate_scientific_source_claims(
    source_root: Path,
    scientific_config: Mapping[str, Any],
) -> dict[str, str]:
    """Reconcile study identities with the exact tracked files they describe."""

    root = _require_real_directory(Path(source_root), name="source_root")
    scientific = strict_config.validate_scientific_config(
        copy.deepcopy(dict(scientific_config))
    )
    study = scientific["study"]
    if tuple(scientific["sources"]["include_paths"]) != EXPECTED_SOURCE_INCLUDES:
        raise ValueError("Frozen scientific source include paths changed")
    claimed_files = {
        "experiment_config_sha256": Path(
            "experiments/retrieval_cv/configs/experiment.json"
        ),
        "fold_manifest_sha256": Path(
            "experiments/retrieval_cv/configs/folds.json"
        ),
        "deepspeed_config_sha256": Path("ds_zero3.json"),
    }
    identities: dict[str, str] = {}
    for claim, relative in claimed_files.items():
        path = root / relative
        size, actual = _hash_regular_file(path)
        if size < 1 or actual != study[claim]:
            raise ValueError(
                f"Scientific {claim} differs from {relative.as_posix()}: "
                f"actual={actual}, expected={study[claim]}"
            )
        identities[claim] = actual

    experiment, _ = _load_exact_json_object(
        root / claimed_files["experiment_config_sha256"],
        expected_sha256=study["experiment_config_sha256"],
    )
    if experiment.get("dataset", {}).get("manifest_sha256") != study[
        "dataset_manifest_sha256"
    ]:
        raise ValueError("Experiment dataset identity differs from the study")
    if experiment.get("folds", {}).get("manifest_sha256") != study[
        "fold_manifest_sha256"
    ]:
        raise ValueError("Experiment fold identity differs from the study")
    if experiment.get("models", {}).get("modernbert_base", {}).get(
        "snapshot_tree_sha256"
    ) != study["model_snapshot_tree_sha256"]:
        raise ValueError("Experiment model snapshot identity differs from the study")
    if experiment.get("aws_training", {}).get("training_image") != study[
        "training_base_image_uri"
    ]:
        raise ValueError("Experiment base training image differs from the study")

    snapshot_path = root / "experiments/retrieval_cv/configs/modernbert_snapshot.json"
    snapshot, snapshot_file_sha256 = _load_exact_json_object(snapshot_path)
    if snapshot.get("tree_sha256") != study["model_snapshot_tree_sha256"]:
        raise ValueError("Snapshot manifest tree identity differs from the study")
    identities["model_snapshot_manifest_sha256"] = snapshot_file_sha256
    identities["model_snapshot_tree_sha256"] = study["model_snapshot_tree_sha256"]
    identities["dataset_manifest_sha256"] = study["dataset_manifest_sha256"]
    return identities


def validate_clean_source_checkout(
    source_root: Path,
    *,
    expected_git_commit: str,
    expected_git_tree: str,
    expected_commit_epoch: int,
) -> Path:
    """Bind a source directory to one clean Git/LFS checkout exactly."""

    root = _require_real_directory(Path(source_root), name="source_root")
    strict_config._require_git_object("expected_git_commit", expected_git_commit)
    strict_config._require_git_object("expected_git_tree", expected_git_tree)
    strict_config._require_int("expected_commit_epoch", expected_commit_epoch)

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: "
                f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
            )
        return completed.stdout

    repository_root = Path(
        git("rev-parse", "--show-toplevel").decode("utf-8", errors="strict").strip()
    ).resolve(strict=True)
    if repository_root.is_symlink() or not repository_root.is_dir():
        raise ValueError("Git source root is not a real directory")
    commit = git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    tree = git("rev-parse", "HEAD^{tree}").decode("ascii", errors="strict").strip()
    epoch_text = git("show", "-s", "--format=%ct", "HEAD").decode(
        "ascii", errors="strict"
    ).strip()
    if (
        commit != expected_git_commit
        or tree != expected_git_tree
        or not epoch_text.isdecimal()
        or int(epoch_text) != expected_commit_epoch
    ):
        raise ValueError(
            "Source checkout identity differs from the scientific configuration: "
            f"commit={commit}, tree={tree}, epoch={epoch_text}"
        )
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("Source checkout must be completely clean")
    attributes = git("lfs", "fsck")
    if b"Git LFS fsck OK" not in attributes:
        raise RuntimeError("git lfs fsck did not report an exact success")
    try:
        root.resolve(strict=True).relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("Source root is outside its resolved Git worktree") from exc
    return repository_root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = Path(path)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be a real directory: {path}")
    return path


def _hash_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Source entry is not a regular file: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        if total != metadata.st_size:
            raise RuntimeError(f"Source file changed while hashing: {path}")
        return total, digest.hexdigest()
    finally:
        os.close(descriptor)


def _path_record(root: Path, relative: str) -> dict[str, Any]:
    strict_config._require_posix_relative_path("source entry path", relative)
    path = root.joinpath(*PurePosixPath(relative).parts)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"Source bundle forbids symlink: {relative}")
    if stat.S_ISDIR(metadata.st_mode):
        return {"path": relative, "type": "directory", "mode": "0755"}
    if stat.S_ISREG(metadata.st_mode):
        size, digest = _hash_regular_file(path)
        return {
            "path": relative,
            "type": "file",
            "mode": "0644",
            "size": size,
            "sha256": digest,
        }
    raise ValueError(f"Source bundle forbids special filesystem entry: {relative}")


def _check_source_component(root: Path, parts: Sequence[str]) -> None:
    current = root
    for part in parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            relative = current.relative_to(root).as_posix()
            raise ValueError(f"Source bundle forbids symlink: {relative}")


def build_source_inventory(
    source_root: Path,
    include_paths: Sequence[str],
) -> list[dict[str, Any]]:
    """Inventory explicit source paths without following any filesystem link."""

    root = _require_real_directory(Path(source_root), name="source_root")
    if type(include_paths) not in {list, tuple} or not include_paths:
        raise TypeError("include_paths must be one non-empty list or tuple")
    normalized = [
        strict_config._require_posix_relative_path(f"include_paths[{index}]", value)
        for index, value in enumerate(include_paths)
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("include_paths contains a duplicate path")
    sorted_paths = sorted(normalized)
    for index, left in enumerate(sorted_paths):
        left_path = PurePosixPath(left)
        for right in sorted_paths[index + 1 :]:
            if left_path in PurePosixPath(right).parents:
                raise ValueError(
                    f"include_paths contains overlapping entries: {left!r}, {right!r}"
                )

    selected: set[str] = set()

    def add_ancestors(relative: str) -> None:
        path = PurePosixPath(relative)
        for parent in reversed(path.parents):
            text = parent.as_posix()
            if text != ".":
                selected.add(text)

    def visit(relative: str) -> None:
        _check_source_component(root, PurePosixPath(relative).parts)
        record = _path_record(root, relative)
        add_ancestors(relative)
        selected.add(relative)
        if record["type"] == "directory":
            directory = root.joinpath(*PurePosixPath(relative).parts)
            with os.scandir(directory) as entries:
                names = sorted(entry.name for entry in entries)
            for name in names:
                child = f"{relative}/{name}"
                visit(child)

    for relative in sorted_paths:
        visit(relative)

    inventory = [_path_record(root, relative) for relative in sorted(selected)]
    if [record["path"] for record in inventory] != sorted(
        record["path"] for record in inventory
    ):
        raise AssertionError("Source inventory is not sorted")
    return inventory


def build_commit_exact_source_inventory(
    source_root: Path,
    include_paths: Sequence[str],
    *,
    expected_git_commit: str,
) -> list[dict[str, Any]]:
    """Inventory exactly the commit-tracked files selected by ``include_paths``.

    The ordinary filesystem inventory is still constructed first so ignored
    files, links, special files, and unexpected directories are observed and
    rejected rather than silently omitted from the source claim.
    """

    root = _require_real_directory(Path(source_root), name="source_root")
    strict_config._require_git_object("expected_git_commit", expected_git_commit)
    filesystem_inventory = build_source_inventory(root, include_paths)
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git rev-parse --show-toplevel failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    repository_root = Path(
        completed.stdout.decode("utf-8", errors="strict").strip()
    ).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    try:
        source_prefix = resolved_root.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError("Source root is outside its Git worktree") from exc
    if source_prefix == ".":
        source_prefix = ""

    tracked_files: set[str] = set()
    for index, raw_relative in enumerate(include_paths):
        relative = strict_config._require_posix_relative_path(
            f"include_paths[{index}]", raw_relative
        )
        repository_path = f"{source_prefix}/{relative}" if source_prefix else relative
        listed = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                expected_git_commit,
                "--",
                repository_path,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if listed.returncode != 0:
            raise RuntimeError(
                f"git ls-tree failed for {relative}: "
                + listed.stderr.decode("utf-8", errors="replace").strip()
            )
        names = listed.stdout.split(b"\0")
        if names[-1] != b"":
            raise RuntimeError("git ls-tree NUL output is truncated")
        selected_count = 0
        for raw_name in names[:-1]:
            name = raw_name.decode("utf-8", errors="strict")
            if source_prefix:
                expected_prefix = source_prefix + "/"
                if not name.startswith(expected_prefix):
                    raise RuntimeError("git ls-tree returned a path outside source_root")
                name = name.removeprefix(expected_prefix)
            strict_config._require_posix_relative_path("tracked source path", name)
            path = PurePosixPath(name)
            include = PurePosixPath(relative)
            if path != include and include not in path.parents:
                raise RuntimeError("git ls-tree returned a path outside its include path")
            tracked_files.add(name)
            selected_count += 1
        if selected_count == 0:
            raise ValueError(f"Source include path selects no tracked file: {relative}")

    expected_paths = set(tracked_files)
    for relative in tracked_files:
        for parent in PurePosixPath(relative).parents:
            text = parent.as_posix()
            if text != ".":
                expected_paths.add(text)
    actual_paths = {record["path"] for record in filesystem_inventory}
    if actual_paths != expected_paths:
        raise ValueError(
            "Source filesystem differs from the exact commit-tracked selection: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    inventory = [_path_record(root, relative) for relative in sorted(expected_paths)]
    if inventory != filesystem_inventory:
        raise RuntimeError("Commit-exact source inventory changed during validation")
    return inventory


def _source_inventory_bytes(
    inventory: Sequence[Mapping[str, Any]],
) -> bytes:
    """Encode the source identity exactly as the published bootstrap does."""

    return (
        json.dumps(
            list(inventory),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def source_inventory_sha256(inventory: Sequence[Mapping[str, Any]]) -> str:
    return strict_config.sha256_bytes(_source_inventory_bytes(inventory))


def _read_source_bytes(
    source_root: Path,
    record: Mapping[str, Any],
) -> bytes:
    relative = record["path"]
    path = source_root.joinpath(*PurePosixPath(relative).parts)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Source entry stopped being a regular file: {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != record["size"] or actual_sha256 != record["sha256"]:
        raise RuntimeError(f"Source file changed after inventory: {relative}")
    return payload


def _write_source_archive(
    path: Path,
    *,
    source_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    commit_epoch: int,
) -> None:
    with path.open("xb") as raw_output:
        with gzip.GzipFile(
            fileobj=raw_output,
            mode="wb",
            filename="",
            compresslevel=6,
            mtime=commit_epoch,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w|",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for record in inventory:
                    relative = record["path"]
                    information = tarfile.TarInfo(
                        name=relative + ("/" if record["type"] == "directory" else "")
                    )
                    information.uid = 0
                    information.gid = 0
                    information.uname = ""
                    information.gname = ""
                    information.mtime = commit_epoch
                    if record["type"] == "directory":
                        information.type = tarfile.DIRTYPE
                        information.mode = 0o755
                        information.size = 0
                        archive.addfile(information)
                    else:
                        payload = _read_source_bytes(source_root, record)
                        information.type = tarfile.REGTYPE
                        information.mode = 0o644
                        information.size = len(payload)
                        archive.addfile(information, io.BytesIO(payload))
        raw_output.flush()
        os.fsync(raw_output.fileno())


def _validate_inventory_shape(inventory: object) -> list[dict[str, Any]]:
    if type(inventory) is not list or not inventory:
        raise TypeError("Source inventory must be a non-empty exact list")
    paths: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw_record in enumerate(inventory):
        if type(raw_record) is not dict:
            raise TypeError(f"Source inventory record {index} must be an exact object")
        entry_type = raw_record.get("type")
        if entry_type == "directory":
            record = strict_config._require_object(
                f"source inventory[{index}]",
                raw_record,
                keys={"path", "type", "mode"},
            )
            if record["mode"] != "0755":
                raise ValueError(f"Source directory {record.get('path')!r} mode is not 0755")
        elif entry_type == "file":
            record = strict_config._require_object(
                f"source inventory[{index}]",
                raw_record,
                keys={"path", "type", "mode", "size", "sha256"},
            )
            if record["mode"] != "0644":
                raise ValueError(f"Source file {record.get('path')!r} mode is not 0644")
            strict_config._require_int(
                f"source inventory[{index}].size", record["size"]
            )
            strict_config._require_sha256(
                f"source inventory[{index}].sha256", record["sha256"]
            )
        else:
            raise ValueError(f"Source inventory record {index} has invalid type {entry_type!r}")
        relative = strict_config._require_posix_relative_path(
            f"source inventory[{index}].path", record["path"]
        )
        paths.append(relative)
        normalized.append(record)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Source inventory paths must be sorted and unique")
    return normalized


def read_source_bundle(
    path: Path,
    *,
    expected_inventory: Sequence[Mapping[str, Any]],
    expected_commit_epoch: int,
    expected_sha256: str | None = None,
) -> SourceBundle:
    """Deep-read one archive and verify its bytes, gzip header, and every member."""

    commit_epoch = strict_config._require_int(
        "expected_commit_epoch", expected_commit_epoch
    )
    if commit_epoch > 0xFFFFFFFF:
        raise ValueError("expected_commit_epoch does not fit the gzip header")
    inventory = _validate_inventory_shape(list(expected_inventory))
    raw = strict_config._read_regular_file_once(Path(path))
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        strict_config._require_sha256("expected_sha256", expected_sha256)
        if digest != expected_sha256:
            raise ValueError(
                f"Source bundle hash mismatch: actual={digest}, expected={expected_sha256}"
            )
    if len(raw) < 18 or raw[:3] != b"\x1f\x8b\x08":
        raise ValueError("Source bundle is not one gzip stream")
    if raw[3] != 0:
        raise ValueError("Source bundle gzip header must have no filename or optional fields")
    header_mtime = int.from_bytes(raw[4:8], "little")
    if header_mtime != commit_epoch:
        raise ValueError(
            f"Source bundle gzip mtime mismatch: actual={header_mtime}, expected={commit_epoch}"
        )
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

    observed: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:") as archive:
        for member in archive:
            relative = member.name.rstrip("/")
            strict_config._require_posix_relative_path("tar member path", relative)
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != commit_epoch
            ):
                raise ValueError(f"Source tar metadata is not normalized: {relative}")
            if member.isdir():
                if member.mode != 0o755 or member.size != 0:
                    raise ValueError(f"Source tar directory is not normalized: {relative}")
                observed.append({"path": relative, "type": "directory", "mode": "0755"})
            elif member.isfile():
                if member.mode != 0o644:
                    raise ValueError(f"Source tar file mode is not 0644: {relative}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Source tar file cannot be read: {relative}")
                payload = stream.read()
                if len(payload) != member.size:
                    raise ValueError(f"Source tar file is truncated: {relative}")
                observed.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": "0644",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            else:
                raise ValueError(f"Source tar forbids links and special members: {relative}")
    _validate_inventory_shape(observed)
    if observed != inventory:
        raise ValueError("Source bundle inventory differs from the expected source inventory")
    inventory_digest = source_inventory_sha256(inventory)
    return SourceBundle(
        path=Path(path),
        size=len(raw),
        sha256=digest,
        inventory=tuple(copy.deepcopy(inventory)),
        inventory_sha256=inventory_digest,
        commit_epoch=commit_epoch,
        bundler_runtime=source_bundler_runtime(),
    )


def build_source_bundle(
    *,
    source_root: Path,
    include_paths: Sequence[str],
    output_path: Path,
    commit_epoch: int,
    expected_git_commit: str | None = None,
    expected_bundler_runtime: Mapping[str, str] | None = None,
) -> SourceBundle:
    """Build and atomically publish a deterministic, previously absent tar.gz."""

    root = _require_real_directory(Path(source_root), name="source_root")
    output = Path(output_path)
    parent = _require_real_directory(output.parent, name="output parent")
    epoch = strict_config._require_int("commit_epoch", commit_epoch)
    if epoch > 0xFFFFFFFF:
        raise ValueError("commit_epoch does not fit the gzip header")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite source bundle: {output}")
    temporary = parent / f".{output.name}.incomplete"
    if os.path.lexists(temporary):
        raise FileExistsError(f"Refusing stale source-bundle temporary file: {temporary}")
    actual_bundler_runtime = source_bundler_runtime()
    if (
        expected_bundler_runtime is not None
        and dict(expected_bundler_runtime) != actual_bundler_runtime
    ):
        raise RuntimeError(
            "Source bundler runtime changed: "
            f"actual={actual_bundler_runtime}, expected={dict(expected_bundler_runtime)}"
        )

    if expected_git_commit is None:
        inventory = build_source_inventory(root, include_paths)
    else:
        inventory = build_commit_exact_source_inventory(
            root,
            include_paths,
            expected_git_commit=expected_git_commit,
        )
    published = False
    try:
        _write_source_archive(
            temporary,
            source_root=root,
            inventory=inventory,
            commit_epoch=epoch,
        )
        temporary_bundle = read_source_bundle(
            temporary,
            expected_inventory=inventory,
            expected_commit_epoch=epoch,
        )
        os.link(temporary, output)
        published = True
        temporary.unlink()
        _fsync_directory(parent)
        final_bundle = read_source_bundle(
            output,
            expected_inventory=inventory,
            expected_commit_epoch=epoch,
            expected_sha256=temporary_bundle.sha256,
        )
        return final_bundle
    except BaseException:
        if published and os.path.lexists(output):
            output.unlink()
        if os.path.lexists(temporary):
            temporary.unlink()
        _fsync_directory(parent)
        raise


def _attempt_number(attempt_id: object) -> int:
    if type(attempt_id) is not str:
        raise TypeError("attempt_id must be an exact string")
    match = _ATTEMPT_RE.fullmatch(attempt_id)
    if match is None:
        raise ValueError("attempt_id must have canonical form a1, a2, ...")
    return int(match.group(1))


def _validate_attempt(value: object) -> dict[str, Any]:
    attempt = strict_config._require_object(
        "manifest.attempt",
        value,
        keys={"attempt_id", "parent_manifest_sha256"},
    )
    number = _attempt_number(attempt["attempt_id"])
    parent = attempt["parent_manifest_sha256"]
    if number == 1:
        if parent is not None:
            raise ValueError("First attempt must have null parent_manifest_sha256")
    else:
        strict_config._require_sha256("manifest.attempt.parent_manifest_sha256", parent)
    return attempt


def _validate_study(value: object) -> dict[str, Any]:
    dummy = {
        "schema_version": 1,
        "study": value,
        "sources": {
            "git_commit": "0" * 40,
            "git_tree": "0" * 40,
            "commit_epoch": 0,
            "include_paths": ["placeholder"],
        },
        "run_templates": {
            "controlled": _dummy_template("controlled", {}, value),
            "legacy": _dummy_template(
                "legacy",
                {
                    "base_seed": 17,
                    "epochs": 20,
                    "run_kind": LEGACY_KIND,
                    "total_optimizer_updates": 80,
                },
                value,
            ),
            "determinism_smoke": _dummy_template(
                "determinism_smoke",
                {
                    "run_kind": "determinism_smoke",
                    "epochs": 2,
                    "total_optimizer_updates": 6,
                },
                value,
            ),
        },
    }
    strict_config.validate_scientific_config(dummy)
    return value


def _dummy_template(
    kind: str,
    hyperparameters: Mapping[str, Any],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    entries = {
        "controlled": (
            "train_sm.py",
            "controlled_retriever",
            "controlled_retrieval_artifact_v1",
        ),
        "legacy": (
            "train_sm.py",
            "corrected_legacy_diagnostic_retriever",
            "corrected_legacy_diagnostic_artifact_v1",
        ),
        "determinism_smoke": (
            "train_sm.py",
            "determinism_smoke_retriever",
            "determinism_smoke_artifact_v1",
        ),
    }
    entry_point, artifact_type, validator_version = entries[kind]
    return {
        "entry_point": entry_point,
        "hyperparameters": dict(hyperparameters),
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "input_channels": {
            "base_model": {
                "s3_uri": (
                    "s3://placeholder-bucket/base-model-"
                    + study["model_snapshot_tree_sha256"]
                ),
                "identity_sha256": study["model_snapshot_tree_sha256"],
            },
            "data": {
                "s3_uri": (
                    "s3://placeholder-bucket/data-"
                    + study["dataset_manifest_sha256"]
                ),
                "identity_sha256": study["dataset_manifest_sha256"],
            },
        },
        "expected_artifact_identity": {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "validator_version": validator_version,
        },
    }


def _validate_sources(value: object) -> dict[str, Any]:
    sources = strict_config._require_object(
        "manifest.sources",
        value,
        keys={
            "git_commit",
            "git_tree",
            "commit_epoch",
            "source_bundle_path",
            "source_bundle_size",
            "source_bundle_sha256",
            "source_inventory_sha256",
            "bundler_runtime",
        },
    )
    strict_config._require_git_object("manifest.sources.git_commit", sources["git_commit"])
    strict_config._require_git_object("manifest.sources.git_tree", sources["git_tree"])
    strict_config._require_int("manifest.sources.commit_epoch", sources["commit_epoch"])
    bundle_path = strict_config._require_posix_relative_path(
        "manifest.sources.source_bundle_path", sources["source_bundle_path"]
    )
    if "/" in bundle_path:
        raise ValueError("manifest.sources.source_bundle_path must be a logical basename")
    strict_config._require_int(
        "manifest.sources.source_bundle_size", sources["source_bundle_size"], minimum=1
    )
    strict_config._require_sha256(
        "manifest.sources.source_bundle_sha256", sources["source_bundle_sha256"]
    )
    strict_config._require_sha256(
        "manifest.sources.source_inventory_sha256", sources["source_inventory_sha256"]
    )
    bundler_runtime = strict_config._require_object(
        "manifest.sources.bundler_runtime",
        sources["bundler_runtime"],
        keys={"python", "zlib_compile", "zlib_runtime"},
    )
    if bundler_runtime != EXPECTED_BUNDLER_RUNTIME:
        raise ValueError("Manifest source bundler runtime changed")
    return sources


def _validate_infrastructure(value: object) -> dict[str, Any]:
    infrastructure = strict_config._require_object(
        "manifest.infrastructure",
        value,
        keys={
            "account_id",
            "region",
            "role_arn",
            "artifact_bucket",
            "artifact_root_prefix",
            "ecr_repository",
            "training_instance_type",
            "training_instance_count",
            "training_volume_size_gb",
            "training_max_runtime_seconds",
            "processing_instance_type",
            "processing_instance_count",
            "processing_volume_size_gb",
            "processing_max_runtime_seconds",
            "max_concurrent_training_jobs",
            "tags",
        },
    )
    strict_config.validate_aws_local_config(
        {"schema_version": 1, **copy.deepcopy(infrastructure)}
    )
    return infrastructure


def _copy_template(template: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(template))


def _output_prefix(infrastructure: Mapping[str, Any], attempt: str, run_id: str) -> str:
    return (
        f"s3://{infrastructure['artifact_bucket']}/"
        f"{infrastructure['artifact_root_prefix']}/{attempt}/{run_id}"
    )


def _base_run(
    *,
    run_id: str,
    kind: str,
    cell: Mapping[str, Any],
    job_name: str,
    template: Mapping[str, Any],
    source_bundle_sha256: str,
    output_prefix: str,
    python_hash_seed: int,
) -> dict[str, Any]:
    environment = copy.deepcopy(template["environment"])
    environment["PYTHONHASHSEED"] = str(python_hash_seed)
    return {
        "run_id": run_id,
        "kind": kind,
        "cell": copy.deepcopy(dict(cell)),
        "job_name": job_name,
        "entry_point": template["entry_point"],
        "source_bundle_sha256": source_bundle_sha256,
        "hyperparameters": copy.deepcopy(template["hyperparameters"]),
        "environment": environment,
        "input_channels": copy.deepcopy(template["input_channels"]),
        "output_prefix": output_prefix,
        "expected_artifact_identity": copy.deepcopy(
            template["expected_artifact_identity"]
        ),
    }


def build_dry_manifest(
    *,
    scientific_config: Mapping[str, Any],
    aws_local_config: Mapping[str, Any],
    source_bundle: SourceBundle,
    attempt_id: str = "a1",
    parent_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Expand the exact ready 60+2+2 training plan in canonical order."""

    scientific = strict_config.validate_scientific_config(copy.deepcopy(dict(scientific_config)))
    aws_config = strict_config.validate_aws_local_config(copy.deepcopy(dict(aws_local_config)))
    attempt_number = _attempt_number(attempt_id)
    if attempt_number == 1 and parent_manifest_sha256 is not None:
        raise ValueError("First attempt must not name a parent manifest")
    if attempt_number > 1:
        strict_config._require_sha256("parent_manifest_sha256", parent_manifest_sha256)
    if source_bundle.commit_epoch != scientific["sources"]["commit_epoch"]:
        raise ValueError("Source bundle commit epoch differs from scientific configuration")
    verified_bundle = read_source_bundle(
        source_bundle.path,
        expected_inventory=source_bundle.inventory,
        expected_commit_epoch=source_bundle.commit_epoch,
        expected_sha256=source_bundle.sha256,
    )
    if (
        verified_bundle.size != source_bundle.size
        or verified_bundle.inventory_sha256 != source_bundle.inventory_sha256
    ):
        raise ValueError("Source bundle record differs from deep archive readback")
    configured_paths = [
        PurePosixPath(path) for path in scientific["sources"]["include_paths"]
    ]
    inventory_paths = {
        PurePosixPath(record["path"]) for record in source_bundle.inventory
    }
    if any(path not in inventory_paths for path in configured_paths):
        raise ValueError("Source bundle omits a configured include path")
    for path in inventory_paths:
        if not any(
            path == configured
            or path in configured.parents
            or configured in path.parents
            for configured in configured_paths
        ):
            raise ValueError(f"Source bundle contains an unconfigured path: {path}")

    infrastructure = {
        key: copy.deepcopy(aws_config[key])
        for key in aws_config
        if key != "schema_version"
    }
    for image_name in ("training_image_uri", "evaluation_image_uri"):
        image_account, image_region, image_repository, _ = strict_config._ecr_image_coordinates(
            scientific["study"][image_name]
        )
        if image_account != infrastructure["account_id"]:
            raise ValueError(f"{image_name} is not in the configured AWS account")
        if image_region != infrastructure["region"]:
            raise ValueError(f"{image_name} is not in the configured AWS region")
        if image_repository != infrastructure["ecr_repository"]:
            raise ValueError(f"{image_name} is not in the configured ECR repository")
    for template_name, template in scientific["run_templates"].items():
        for channel_name, channel in template["input_channels"].items():
            bucket, _ = strict_config._s3_uri_coordinates(channel["s3_uri"])
            if bucket != infrastructure["artifact_bucket"]:
                raise ValueError(
                    f"{template_name}.{channel_name} is not in the configured artifact bucket"
                )
    source_digest = source_bundle.sha256
    controlled_template = scientific["run_templates"]["controlled"]
    controlled_runs: list[dict[str, Any]] = []
    for fold in strict_config.CONTROLLED_FOLDS:
        for view in strict_config.CONTROLLED_QUERY_VIEWS:
            for sampler in strict_config.CONTROLLED_SAMPLERS:
                for seed in strict_config.CONTROLLED_SEEDS:
                    flat_view = _VIEW_ALIAS[view]
                    flat_sampler = _SAMPLER_ALIAS[sampler]
                    run_id = f"controlled-f{fold}-{flat_view}-{flat_sampler}-s{seed}"
                    job_name = (
                        f"arr-ret-cv1-f{fold}-{flat_view}-{flat_sampler}-s{seed}-{attempt_id}"
                    )
                    cell = {
                        "outer_fold": fold,
                        "query_view": view,
                        "sampler": sampler,
                        "experiment_seed": seed,
                    }
                    run = _base_run(
                        run_id=run_id,
                        kind=CONTROLLED_KIND,
                        cell=cell,
                        job_name=job_name,
                        template=controlled_template,
                        source_bundle_sha256=source_digest,
                        output_prefix=_output_prefix(
                            infrastructure, attempt_id, run_id
                        ),
                        python_hash_seed=seed,
                    )
                    run["hyperparameters"].update(cell)
                    controlled_runs.append(run)

    auxiliary_runs: list[dict[str, Any]] = []
    legacy_template = scientific["run_templates"]["legacy"]
    legacy_seed = legacy_template["hyperparameters"]["base_seed"]
    for view in strict_config.CONTROLLED_QUERY_VIEWS:
        view_alias = "flat" if view == "flat_masked" else "structured"
        run_id = f"corrected-legacy-{view_alias}"
        run = _base_run(
            run_id=run_id,
            kind=LEGACY_KIND,
            cell={"query_view": view},
            job_name=f"arr-ret-cv1-corrected-legacy-{view_alias}-{attempt_id}",
            template=legacy_template,
            source_bundle_sha256=source_digest,
            output_prefix=_output_prefix(infrastructure, attempt_id, run_id),
            python_hash_seed=legacy_seed,
        )
        run["hyperparameters"]["query_view"] = view
        run["launch_metadata"] = {"replica_id": None}
        auxiliary_runs.append(run)

    smoke_template = scientific["run_templates"]["determinism_smoke"]
    smoke_cell = {
        "outer_fold": 0,
        "query_view": "structured",
        "sampler": "global_uniform",
        "experiment_seed": 17,
    }
    for replica_id in ("a", "b"):
        run_id = f"determinism-smoke-{replica_id}"
        run = _base_run(
            run_id=run_id,
            kind=SMOKE_KIND,
            cell=smoke_cell,
            job_name=f"arr-ret-cv1-smoke-{replica_id}-{attempt_id}",
            template=smoke_template,
            source_bundle_sha256=source_digest,
            output_prefix=_output_prefix(infrastructure, attempt_id, run_id),
            python_hash_seed=17,
        )
        run["hyperparameters"].update(smoke_cell)
        run["launch_metadata"] = {"replica_id": replica_id}
        auxiliary_runs.append(run)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "attempt": {
            "attempt_id": attempt_id,
            "parent_manifest_sha256": parent_manifest_sha256,
        },
        "study": copy.deepcopy(scientific["study"]),
        "execution": {
            "blockers": list(EXECUTION_BLOCKERS),
            "status": "ready",
            "submittable": True,
        },
        "sources": {
            "git_commit": scientific["sources"]["git_commit"],
            "git_tree": scientific["sources"]["git_tree"],
            "commit_epoch": scientific["sources"]["commit_epoch"],
            "source_bundle_path": f"source-{source_bundle.sha256}.tar.gz",
            "source_bundle_size": source_bundle.size,
            "source_bundle_sha256": source_bundle.sha256,
            "source_inventory_sha256": source_bundle.inventory_sha256,
            "bundler_runtime": copy.deepcopy(source_bundle.bundler_runtime),
        },
        "infrastructure": infrastructure,
        "controlled_runs": controlled_runs,
        "auxiliary_runs": auxiliary_runs,
    }
    return validate_dry_manifest(manifest)


def _validate_identifier(name: str, value: object) -> str:
    text = strict_config._require_string(name, value)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise ValueError(f"{name} is not a canonical lowercase identifier")
    return text


def _validate_job_name(name: str, value: object) -> str:
    text = strict_config._require_string(name, value)
    if len(text) > 63 or _JOB_NAME_RE.fullmatch(text) is None:
        raise ValueError(f"{name} is not a valid deterministic SageMaker job name")
    return text


def _validate_run(
    raw_run: object,
    *,
    index_name: str,
    auxiliary: bool,
) -> dict[str, Any]:
    keys = {
        "run_id",
        "kind",
        "cell",
        "job_name",
        "entry_point",
        "source_bundle_sha256",
        "hyperparameters",
        "environment",
        "input_channels",
        "output_prefix",
        "expected_artifact_identity",
    }
    if auxiliary:
        keys.add("launch_metadata")
    run = strict_config._require_object(index_name, raw_run, keys=keys)
    _validate_identifier(f"{index_name}.run_id", run["run_id"])
    strict_config._require_string(f"{index_name}.kind", run["kind"])
    _validate_job_name(f"{index_name}.job_name", run["job_name"])
    strict_config._require_posix_relative_path(
        f"{index_name}.entry_point", run["entry_point"]
    )
    strict_config._require_sha256(
        f"{index_name}.source_bundle_sha256", run["source_bundle_sha256"]
    )
    strict_config._validate_hyperparameters(
        f"{index_name}.hyperparameters", run["hyperparameters"]
    )
    environment = run["environment"]
    if type(environment) is not dict or not environment:
        raise TypeError(f"{index_name}.environment must be a non-empty object")
    for key, value in environment.items():
        if type(key) is not str or not key or type(value) is not str or not value:
            raise TypeError(f"{index_name}.environment must map strings to strings")
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(f"{index_name} has wrong CUBLAS_WORKSPACE_CONFIG")
    seed_text = environment.get("PYTHONHASHSEED")
    if type(seed_text) is not str or not seed_text.isdecimal():
        raise ValueError(f"{index_name} must bind a decimal PYTHONHASHSEED")
    strict_config.validate_input_channels(
        f"{index_name}.input_channels", run["input_channels"]
    )
    strict_config._require_s3_uri(f"{index_name}.output_prefix", run["output_prefix"])
    strict_config._validate_artifact_identity(
        f"{index_name}.expected_artifact_identity",
        run["expected_artifact_identity"],
    )
    if type(run["cell"]) is not dict:
        raise TypeError(f"{index_name}.cell must be an exact object")
    if auxiliary:
        launch = strict_config._require_object(
            f"{index_name}.launch_metadata",
            run["launch_metadata"],
            keys={"replica_id"},
        )
        if launch["replica_id"] is not None and launch["replica_id"] not in {"a", "b"}:
            raise ValueError(f"{index_name}.launch_metadata.replica_id is invalid")
    return run


def _expected_controlled_cells() -> list[dict[str, Any]]:
    return [
        {
            "outer_fold": fold,
            "query_view": view,
            "sampler": sampler,
            "experiment_seed": seed,
        }
        for fold in strict_config.CONTROLLED_FOLDS
        for view in strict_config.CONTROLLED_QUERY_VIEWS
        for sampler in strict_config.CONTROLLED_SAMPLERS
        for seed in strict_config.CONTROLLED_SEEDS
    ]


def _expected_controlled_identity(cell: Mapping[str, Any], attempt_id: str) -> tuple[str, str]:
    view = _VIEW_ALIAS[cell["query_view"]]
    sampler = _SAMPLER_ALIAS[cell["sampler"]]
    run_id = f"controlled-f{cell['outer_fold']}-{view}-{sampler}-s{cell['experiment_seed']}"
    job_name = (
        f"arr-ret-cv1-f{cell['outer_fold']}-{view}-{sampler}-"
        f"s{cell['experiment_seed']}-{attempt_id}"
    )
    return run_id, job_name


def _require_exact_cell(name: str, value: object, expected: Mapping[str, Any]) -> None:
    cell = strict_config._require_object(name, value, keys=set(expected))
    for key, expected_value in expected.items():
        if type(cell[key]) is not type(expected_value) or cell[key] != expected_value:
            raise ValueError(
                f"{name}.{key} mismatch: actual={cell[key]!r}, expected={expected_value!r}"
            )


def _container_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(run[key])
        for key in (
            "kind",
            "cell",
            "entry_point",
            "source_bundle_sha256",
            "hyperparameters",
            "environment",
            "input_channels",
            "expected_artifact_identity",
        )
    }


def _validate_plan_run_contract(
    run: Mapping[str, Any],
    *,
    name: str,
    study: Mapping[str, Any],
    infrastructure: Mapping[str, Any],
    expected_kind: str,
) -> None:
    expected_by_kind = {
        CONTROLLED_KIND: (
            "train_sm.py",
            {
                "artifact_type": "controlled_retriever",
                "schema_version": 1,
                "validator_version": "controlled_retrieval_artifact_v1",
            },
        ),
        LEGACY_KIND: (
            "train_sm.py",
            {
                "artifact_type": "corrected_legacy_diagnostic_retriever",
                "schema_version": 1,
                "validator_version": "corrected_legacy_diagnostic_artifact_v1",
            },
        ),
        SMOKE_KIND: (
            "train_sm.py",
            {
                "artifact_type": "determinism_smoke_retriever",
                "schema_version": 1,
                "validator_version": "determinism_smoke_artifact_v1",
            },
        ),
    }
    expected_entry_point, expected_artifact = expected_by_kind[expected_kind]
    if run["entry_point"] != expected_entry_point:
        raise ValueError(f"{name} entry point changed")
    if run["expected_artifact_identity"] != expected_artifact:
        raise ValueError(f"{name} artifact protocol changed")
    expected_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": run["environment"]["PYTHONHASHSEED"],
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if run["environment"] != expected_environment:
        raise ValueError(f"{name} environment contract changed")
    channels = run["input_channels"]
    if set(channels) != {"base_model", "data"}:
        raise ValueError(f"{name} must bind exactly data and base_model channels")
    expected_identities = {
        "base_model": study["model_snapshot_tree_sha256"],
        "data": study["dataset_manifest_sha256"],
    }
    for channel_name, expected_identity in expected_identities.items():
        channel = channels[channel_name]
        bucket, key = strict_config._s3_uri_coordinates(channel["s3_uri"])
        if (
            channel["identity_sha256"] != expected_identity
            or bucket != infrastructure["artifact_bucket"]
            or not key.endswith(expected_identity)
        ):
            raise ValueError(f"{name}.{channel_name} differs from the study contract")


def validate_dry_manifest(value: object) -> dict[str, Any]:
    """Validate exact schema, matrix order, identities, and smoke equivalence."""

    manifest = strict_config._require_object(
        "training plan",
        value,
        keys={
            "schema_version",
            "manifest_type",
            "attempt",
            "study",
            "execution",
            "sources",
            "infrastructure",
            "controlled_runs",
            "auxiliary_runs",
        },
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Launch manifest schema_version must be exact integer 1")
    if manifest["manifest_type"] != MANIFEST_TYPE:
        raise ValueError(f"Launch manifest_type must equal {MANIFEST_TYPE!r}")
    execution = strict_config._require_object(
        "training plan.execution",
        manifest["execution"],
        keys={"blockers", "status", "submittable"},
    )
    if execution != {
        "blockers": list(EXECUTION_BLOCKERS),
        "status": "ready",
        "submittable": True,
    }:
        raise ValueError("Training plan must be explicitly ready and unblocked")
    attempt = _validate_attempt(manifest["attempt"])
    _validate_study(manifest["study"])
    sources = _validate_sources(manifest["sources"])
    infrastructure = _validate_infrastructure(manifest["infrastructure"])
    for image_name in ("training_image_uri", "evaluation_image_uri"):
        account, region, repository, _ = strict_config._ecr_image_coordinates(
            manifest["study"][image_name]
        )
        if (
            account != infrastructure["account_id"]
            or region != infrastructure["region"]
            or repository != infrastructure["ecr_repository"]
        ):
            raise ValueError(f"Training plan {image_name} differs from infrastructure")
    attempt_id = attempt["attempt_id"]

    controlled = manifest["controlled_runs"]
    if type(controlled) is not list or len(controlled) != 60:
        raise ValueError("Launch manifest must contain exactly 60 controlled runs")
    expected_cells = _expected_controlled_cells()
    for index, (raw_run, expected_cell) in enumerate(zip(controlled, expected_cells)):
        name = f"controlled_runs[{index}]"
        run = _validate_run(raw_run, index_name=name, auxiliary=False)
        if run["kind"] != CONTROLLED_KIND:
            raise ValueError(f"{name}.kind must equal {CONTROLLED_KIND!r}")
        _validate_plan_run_contract(
            run,
            name=name,
            study=manifest["study"],
            infrastructure=infrastructure,
            expected_kind=CONTROLLED_KIND,
        )
        _require_exact_cell(f"{name}.cell", run["cell"], expected_cell)
        run_id, job_name = _expected_controlled_identity(expected_cell, attempt_id)
        if run["run_id"] != run_id or run["job_name"] != job_name:
            raise ValueError(f"{name} has a non-canonical run ID or job name")
        for key, expected_value in expected_cell.items():
            actual = run["hyperparameters"].get(key)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{name}.hyperparameters.{key} does not match its cell")
        if set(run["hyperparameters"]) != set(expected_cell):
            raise ValueError(f"{name}.hyperparameters contains an unplanned key")
        if run["environment"]["PYTHONHASHSEED"] != str(expected_cell["experiment_seed"]):
            raise ValueError(f"{name} PYTHONHASHSEED does not match experiment_seed")
        expected_output = _output_prefix(infrastructure, attempt_id, run_id)
        if run["output_prefix"] != expected_output:
            raise ValueError(f"{name}.output_prefix is not canonical")
        if run["source_bundle_sha256"] != sources["source_bundle_sha256"]:
            raise ValueError(f"{name} uses the wrong source bundle")

    auxiliary = manifest["auxiliary_runs"]
    if type(auxiliary) is not list or len(auxiliary) != 4:
        raise ValueError("Launch manifest must contain exactly four auxiliary runs")
    expected_auxiliary = (
        (
            LEGACY_KIND,
            "corrected-legacy-flat",
            "arr-ret-cv1-corrected-legacy-flat",
            {"query_view": "flat_masked"},
            None,
        ),
        (
            LEGACY_KIND,
            "corrected-legacy-structured",
            "arr-ret-cv1-corrected-legacy-structured",
            {"query_view": "structured"},
            None,
        ),
        (
            SMOKE_KIND,
            "determinism-smoke-a",
            "arr-ret-cv1-smoke-a",
            {
                "outer_fold": 0,
                "query_view": "structured",
                "sampler": "global_uniform",
                "experiment_seed": 17,
            },
            "a",
        ),
        (
            SMOKE_KIND,
            "determinism-smoke-b",
            "arr-ret-cv1-smoke-b",
            {
                "outer_fold": 0,
                "query_view": "structured",
                "sampler": "global_uniform",
                "experiment_seed": 17,
            },
            "b",
        ),
    )
    validated_auxiliary: list[dict[str, Any]] = []
    for index, (raw_run, expected) in enumerate(zip(auxiliary, expected_auxiliary)):
        kind, run_id, job_prefix, cell, replica_id = expected
        name = f"auxiliary_runs[{index}]"
        run = _validate_run(raw_run, index_name=name, auxiliary=True)
        validated_auxiliary.append(run)
        if run["kind"] != kind or run["run_id"] != run_id:
            raise ValueError(f"{name} has wrong kind or run_id")
        _validate_plan_run_contract(
            run,
            name=name,
            study=manifest["study"],
            infrastructure=infrastructure,
            expected_kind=kind,
        )
        if run["job_name"] != f"{job_prefix}-{attempt_id}":
            raise ValueError(f"{name} has a non-canonical job name")
        _require_exact_cell(f"{name}.cell", run["cell"], cell)
        if run["launch_metadata"]["replica_id"] != replica_id:
            raise ValueError(f"{name} has wrong launch replica metadata")
        if run["output_prefix"] != _output_prefix(infrastructure, attempt_id, run_id):
            raise ValueError(f"{name}.output_prefix is not canonical")
        if run["source_bundle_sha256"] != sources["source_bundle_sha256"]:
            raise ValueError(f"{name} uses the wrong source bundle")
        if kind == SMOKE_KIND:
            for key, expected_value in cell.items():
                actual = run["hyperparameters"].get(key)
                if type(actual) is not type(expected_value) or actual != expected_value:
                    raise ValueError(f"{name}.hyperparameters.{key} differs from smoke cell")
            if run["environment"]["PYTHONHASHSEED"] != "17":
                raise ValueError(f"{name} has wrong smoke PYTHONHASHSEED")
            if run["hyperparameters"].get("run_kind") != SMOKE_KIND:
                raise ValueError(f"{name} must bind determinism_smoke run_kind")
            if (
                type(run["hyperparameters"].get("epochs")) is not int
                or run["hyperparameters"]["epochs"] != 2
            ):
                raise ValueError(f"{name} must bind exactly two smoke epochs")
            if (
                type(run["hyperparameters"].get("total_optimizer_updates")) is not int
                or run["hyperparameters"]["total_optimizer_updates"] != 6
            ):
                raise ValueError(f"{name} must bind exactly six smoke optimizer updates")
            if set(run["hyperparameters"]) != {
                *cell,
                "epochs",
                "run_kind",
                "total_optimizer_updates",
            }:
                raise ValueError(f"{name} smoke hyperparameters contain an unplanned key")
        else:
            if run["hyperparameters"].get("query_view") != cell["query_view"]:
                raise ValueError(
                    f"{name} corrected legacy diagnostic query_view differs from its cell"
                )
            base_seed = run["hyperparameters"].get("base_seed")
            if type(base_seed) is not int or base_seed < 0:
                raise ValueError(f"{name} must bind a non-negative exact base_seed")
            if run["environment"]["PYTHONHASHSEED"] != str(base_seed):
                raise ValueError(
                    f"{name} PYTHONHASHSEED differs from corrected legacy base_seed"
                )
            if run["hyperparameters"] != {
                "base_seed": 17,
                "epochs": 20,
                "query_view": cell["query_view"],
                "run_kind": LEGACY_KIND,
                "total_optimizer_updates": 80,
            }:
                raise ValueError(
                    f"{name} corrected legacy diagnostic hyperparameters changed"
                )

    if _container_identity(validated_auxiliary[2]) != _container_identity(
        validated_auxiliary[3]
    ):
        raise ValueError(
            "Determinism smoke replicas must have identical scientific/container inputs"
        )

    all_runs = [*controlled, *auxiliary]
    for field in ("run_id", "job_name", "output_prefix"):
        values = [run[field] for run in all_runs]
        if len(values) != len(set(values)):
            raise ValueError(f"Launch manifest contains duplicate {field} values")
    for fold in strict_config.CONTROLLED_FOLDS:
        fold_runs = [run for run in controlled if run["cell"]["outer_fold"] == fold]
        if len(fold_runs) != 12:
            raise ValueError(f"Fold {fold} must contain exactly 12 controlled runs")
    return manifest


def read_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    value, digest = strict_config.load_canonical_json_object(
        path, expected_sha256=expected_sha256
    )
    return validate_dry_manifest(value), digest


def publish_manifest_absent(
    path: Path,
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Atomically publish canonical bytes to an absent path and deep-read them."""

    manifest = validate_dry_manifest(copy.deepcopy(dict(value)))
    payload = strict_config.canonical_json_bytes(manifest)
    digest = strict_config.sha256_bytes(payload)
    output = Path(path)
    parent = _require_real_directory(output.parent, name="manifest output parent")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite training plan: {output}")
    temporary = parent / f".{output.name}.incomplete"
    if os.path.lexists(temporary):
        raise FileExistsError(f"Refusing stale manifest temporary file: {temporary}")
    published = False
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        published = True
        temporary.unlink()
        _fsync_directory(parent)
        readback, readback_digest = read_manifest(output, expected_sha256=digest)
        if readback != manifest or readback_digest != digest:
            raise RuntimeError("Launch-manifest readback differs from published bytes")
        return readback, digest
    except BaseException:
        if published and os.path.lexists(output):
            output.unlink()
        if os.path.lexists(temporary):
            temporary.unlink()
        _fsync_directory(parent)
        raise
