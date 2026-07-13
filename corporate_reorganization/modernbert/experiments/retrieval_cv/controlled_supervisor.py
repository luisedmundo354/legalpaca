"""Fail-loud, restart-safe supervision for the 60 controlled training jobs.

The supervisor is deliberately separate from the one-shot launch boundary.  It
validates one already sealed v3 determinism gate, persists an immutable local
schedule, and then composes the existing preflight, submission, status, and
terminal functions.  A create intent is published immediately before the
underlying ``CreateTrainingJob`` call.  An intent without a sealed submission
receipt is ambiguous forever: restart fails instead of retrying, reconciling,
or adopting the remote job.

No corrected-legacy or determinism-smoke job can enter the schedule.  The
controlled queue is fold-major, then seed, query view, and sampler, which makes
each consecutive group of four one matched ``(fold, seed)`` quartet.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import errno
import os
import re
import stat
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import aws
from . import config as strict_config
from . import determinism_gate
from . import manifest
from . import training_aws
from . import training_launch


CONTROLLED_SUPERVISOR_PROTOCOL = "retrieval_cv_controlled_supervisor_v1"
CONTROLLED_CREATE_INTENT_PROTOCOL = (
    "retrieval_cv_controlled_create_intent_v1"
)
CONTROLLED_SUPERVISOR_COMPLETION_PROTOCOL = (
    "retrieval_cv_controlled_supervisor_completion_v1"
)
CONTROLLED_SUPERVISOR_SNAPSHOT_PROTOCOL = (
    "retrieval_cv_controlled_supervisor_snapshot_v1"
)
COMPLETED_FOLD_EVIDENCE_PROTOCOL = "retrieval_cv_completed_fold_evidence_v1"
CONTROLLED_MAX_ACTIVE = 4

_STATE_FILE = "supervisor.json"
_RUNS_DIRECTORY = "runs"
_OBSERVATIONS_DIRECTORY = "observations"
_COMPLETION_FILE = "completion.json"
_CREATE_INTENT_FILE = "create-intent.json"
_SUBMISSION_FILE = "submission.json"
_TERMINAL_FILE = "terminal.json"
_STATUS_FILE = re.compile(r"status-([0-9]{6})\.json\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_SUPERVISOR_KEYS = {
    "schema_version",
    "protocol",
    "max_active",
    "training_plan",
    "training_plan_sha256",
    "staging_receipt",
    "staging_receipt_sha256",
    "determinism_gate_receipt",
    "determinism_gate_receipt_sha256",
    "schedule",
    "receipt_sha256",
}
_SCHEDULE_KEYS = {
    "queue_index",
    "quartet_index",
    "quartet_member_index",
    "outer_fold",
    "experiment_seed",
    "query_view",
    "sampler",
    "run_id",
    "job_name",
}
_INTENT_KEYS = {
    "schema_version",
    "protocol",
    "state_manifest_sha256",
    "queue_index",
    "run_id",
    "job_name",
    "plan_sha256",
    "staging_receipt_sha256",
    "determinism_gate_receipt_sha256",
    "preflight_receipt",
    "preflight_receipt_sha256",
    "request_receipt_sha256",
    "request_sha256",
    "receipt_sha256",
}
_COMPLETION_KEYS = {
    "schema_version",
    "protocol",
    "state_manifest_sha256",
    "completed_runs",
    "terminal_receipt_sha256_by_run",
    "succeeded",
    "receipt_sha256",
}
_COMPLETED_FOLD_EVIDENCE_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "attempt_id",
    "state_manifest_sha256",
    "training_plan",
    "training_plan_sha256",
    "training_staging_receipt",
    "training_staging_receipt_sha256",
    "source_bundle",
    "systems",
    "receipt_sha256",
}
_COMPLETED_FOLD_SYSTEM_KEYS = {
    "ordinal",
    "queue_index",
    "cell",
    "run_id",
    "job_name",
    "preflight_receipt",
    "preflight_receipt_sha256",
    "submission_receipt",
    "submission_receipt_sha256",
    "terminal_receipt",
    "terminal_receipt_sha256",
    "request_receipt_sha256",
}


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
        if current is None or type(current) in {str, bool, int, float}:
            strict_config.canonical_json_bytes(current)
            return
        raise TypeError(f"{path} contains a non-JSON type: {type(current).__name__}")

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


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be one non-negative exact integer")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    _require_plain_json(payload, name="receipt payload")
    if "receipt_sha256" in payload:
        raise ValueError("Receipt payload already contains receipt_sha256")
    sealed = copy.deepcopy(payload)
    sealed["receipt_sha256"] = _document_sha256(payload)
    return sealed


def _validate_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    actual = _exact_sha256(value["receipt_sha256"], name=f"{name}.receipt_sha256")
    payload = {
        key: copy.deepcopy(nested)
        for key, nested in value.items()
        if key != "receipt_sha256"
    }
    if actual != _document_sha256(payload):
        raise ValueError(f"{name} self-hash changed")


def _canonical_absolute_path(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be one pathlib.Path")
    text = path.as_posix()
    if (
        not path.is_absolute()
        or text == "/"
        or text.startswith("//")
        or text != PurePosixPath(text).as_posix()
        or ".." in path.parts
        or path.resolve(strict=False) != path
    ):
        raise ValueError(f"{name} must be one canonical absolute path")
    return path


def _real_directory(path: Path, *, name: str) -> Path:
    path = _canonical_absolute_path(path, name=name)
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{name} must be one real canonical directory")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _assert_directory_path_identity(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    name: str,
) -> None:
    path = _canonical_absolute_path(path, name=name)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{name} disappeared") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _stat_identity(metadata) != expected_identity
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"{name} path identity changed")


def _open_stable_directory(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    name: str,
) -> int:
    path = _canonical_absolute_path(path, name=name)
    flags = os.O_RDONLY | os.O_DIRECTORY
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Stable evidence publication requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _stat_identity(metadata) != expected_identity
        ):
            raise RuntimeError(f"{name} descriptor identity changed")
        _assert_directory_path_identity(
            path,
            expected_identity,
            name=name,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Canonical evidence write made no progress")
        offset += written


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _publish_json_absent(
    path: Path,
    value: object,
    *,
    parent_identity: tuple[int, int],
) -> None:
    path = _canonical_absolute_path(path, name="canonical JSON output")
    parent = path.parent
    target_name = path.name
    temporary_name = f".{target_name}.incomplete"
    payload = strict_config.canonical_json_bytes(value)
    directory_fd = _open_stable_directory(
        parent,
        parent_identity,
        name="canonical JSON output parent",
    )
    published = False
    try:
        if _entry_exists_at(directory_fd, target_name):
            raise FileExistsError(
                f"Refusing to overwrite supervisor evidence: {path}"
            )
        if _entry_exists_at(directory_fd, temporary_name):
            raise FileExistsError(
                f"Refusing stale supervisor evidence: {parent / temporary_name}"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        temporary_fd = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            _write_all(temporary_fd, payload)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        os.link(
            temporary_name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        read_fd = os.open(
            target_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            metadata = os.fstat(read_fd)
            if not stat.S_ISREG(metadata.st_mode) or _read_all(read_fd) != payload:
                raise RuntimeError(
                    "Published supervisor evidence changed on readback"
                )
        finally:
            os.close(read_fd)
        _assert_directory_path_identity(
            parent,
            parent_identity,
            name="canonical JSON output parent",
        )
        if _stat_identity(os.fstat(directory_fd)) != parent_identity:
            raise RuntimeError("Published supervisor evidence changed on readback")
    except BaseException:
        if published and _entry_exists_at(directory_fd, target_name):
            os.unlink(target_name, dir_fd=directory_fd)
        if _entry_exists_at(directory_fd, temporary_name):
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        raise
    finally:
        os.close(directory_fd)


def _rename_no_replace(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic supervisor publication requires Linux renameat2")
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
            raise FileExistsError(f"Refusing to overwrite supervisor state: {target}")
        raise OSError(
            error_number,
            f"Atomic supervisor publication failed: {source} -> {target}",
        )


def _directory_identity(path: Path, *, name: str) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{name} is no longer one real directory")
    return _stat_identity(metadata)


def _controlled_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    controlled = plan["controlled_runs"]
    if type(controlled) is not list or len(controlled) != 60:
        raise ValueError("Controlled supervisor requires exactly 60 controlled runs")
    by_cell: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for run in controlled:
        if type(run) is not dict or run.get("kind") != manifest.CONTROLLED_KIND:
            raise ValueError("Controlled supervisor refuses every non-controlled run")
        cell = run.get("cell")
        if type(cell) is not dict:
            raise TypeError("Controlled run cell must be one exact object")
        key = (
            cell.get("outer_fold"),
            cell.get("experiment_seed"),
            cell.get("query_view"),
            cell.get("sampler"),
        )
        if key in by_cell:
            raise ValueError("Controlled supervisor schedule contains a duplicate cell")
        by_cell[key] = run

    schedule: list[dict[str, Any]] = []
    queue_index = 0
    for fold_index, fold in enumerate(strict_config.CONTROLLED_FOLDS):
        for seed_index, seed in enumerate(strict_config.CONTROLLED_SEEDS):
            quartet_index = fold_index * len(strict_config.CONTROLLED_SEEDS) + seed_index
            for view_index, view in enumerate(strict_config.CONTROLLED_QUERY_VIEWS):
                for sampler_index, sampler in enumerate(
                    strict_config.CONTROLLED_SAMPLERS
                ):
                    run = by_cell.get((fold, seed, view, sampler))
                    if run is None:
                        raise ValueError("Controlled supervisor schedule is incomplete")
                    schedule.append(
                        {
                            "queue_index": queue_index,
                            "quartet_index": quartet_index,
                            "quartet_member_index": (
                                view_index * len(strict_config.CONTROLLED_SAMPLERS)
                                + sampler_index
                            ),
                            "outer_fold": fold,
                            "experiment_seed": seed,
                            "query_view": view,
                            "sampler": sampler,
                            "run_id": run["run_id"],
                            "job_name": run["job_name"],
                        }
                    )
                    queue_index += 1
    if len(schedule) != 60 or len(by_cell) != 60:
        raise ValueError("Controlled supervisor schedule coverage changed")
    return schedule


def _validate_gate_binding(
    value: object,
    *,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
    deep: bool,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("Determinism gate receipt must be one exact object")
    gate = copy.deepcopy(value)
    if deep:
        gate = determinism_gate.validate_determinism_gate_receipt(
            gate,
            training_plan=plan,
            staging_receipt=staged,
        )
    _require_plain_json(gate, name="determinism gate receipt")
    if (
        gate.get("schema_version") != 3
        or type(gate.get("schema_version")) is not int
        or gate.get("protocol") != determinism_gate.DETERMINISM_GATE_PROTOCOL
        or gate.get("exact_match") is not True
        or gate.get("plan_sha256") != _document_sha256(plan)
        or gate.get("staging_receipt_sha256") != _document_sha256(staged)
    ):
        raise ValueError("Controlled launch requires the matching sealed v3 gate")
    if "receipt_sha256" not in gate:
        raise ValueError("Determinism gate receipt lacks its self-hash")
    _validate_self_hash(gate, name="determinism gate receipt")
    return gate


def _build_supervisor_manifest(
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    determinism_gate_receipt: Mapping[str, Any],
    deep_gate: bool,
) -> dict[str, Any]:
    if type(training_plan) is not dict or type(staging_receipt) is not dict:
        raise TypeError("Supervisor plan and staging receipt must be exact objects")
    plan = manifest.validate_dry_manifest(copy.deepcopy(training_plan))
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(staging_receipt), training_plan=plan
    )
    gate = _validate_gate_binding(
        determinism_gate_receipt,
        plan=plan,
        staged=staged,
        deep=deep_gate,
    )
    payload = {
        "schema_version": 1,
        "protocol": CONTROLLED_SUPERVISOR_PROTOCOL,
        "max_active": CONTROLLED_MAX_ACTIVE,
        "training_plan": plan,
        "training_plan_sha256": _document_sha256(plan),
        "staging_receipt": staged,
        "staging_receipt_sha256": _document_sha256(staged),
        "determinism_gate_receipt": gate,
        "determinism_gate_receipt_sha256": _document_sha256(gate),
        "schedule": _controlled_schedule(plan),
    }
    return _seal(payload)


def _run_directory_name(entry: Mapping[str, Any]) -> str:
    return f"{entry['queue_index']:02d}-{entry['run_id']}"


def initialize_controlled_supervisor_state(
    *,
    state_dir: Path,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    determinism_gate_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create one absent-only state tree after the complete v3 gate."""

    state_dir = _canonical_absolute_path(state_dir, name="supervisor state directory")
    parent = _real_directory(state_dir.parent, name="supervisor state parent")
    incomplete = state_dir.with_name(f".{state_dir.name}.incomplete")
    if (
        state_dir.exists()
        or state_dir.is_symlink()
        or incomplete.exists()
        or incomplete.is_symlink()
    ):
        raise FileExistsError("Supervisor state and sibling incomplete path must be absent")
    supervisor = _build_supervisor_manifest(
        training_plan=training_plan,
        staging_receipt=staging_receipt,
        determinism_gate_receipt=determinism_gate_receipt,
        deep_gate=True,
    )
    incomplete.mkdir(mode=0o700)
    runs = incomplete / _RUNS_DIRECTORY
    runs.mkdir(mode=0o700)
    for entry in supervisor["schedule"]:
        run_directory = runs / _run_directory_name(entry)
        run_directory.mkdir(mode=0o700)
        observations = run_directory / _OBSERVATIONS_DIRECTORY
        observations.mkdir(mode=0o700)
        _fsync_directory(observations)
        _fsync_directory(run_directory)
    incomplete_identity = _directory_identity(
        incomplete,
        name="incomplete supervisor state",
    )
    _publish_json_absent(
        incomplete / _STATE_FILE,
        supervisor,
        parent_identity=incomplete_identity,
    )
    _fsync_directory(runs)
    _fsync_directory(incomplete)
    expected_identity = _directory_identity(
        incomplete,
        name="incomplete supervisor state",
    )
    _rename_no_replace(incomplete, state_dir)
    if _directory_identity(state_dir, name="published supervisor state") != expected_identity:
        raise RuntimeError("Published supervisor state directory identity changed")
    _fsync_directory(parent)
    loaded, _ = strict_config.load_canonical_json_object(state_dir / _STATE_FILE)
    if aws.canonical_json_bytes(loaded) != aws.canonical_json_bytes(supervisor):
        raise RuntimeError("Published supervisor manifest changed on readback")
    return copy.deepcopy(supervisor)


def _validate_supervisor_manifest(value: object, *, deep_gate: bool) -> dict[str, Any]:
    supervisor = _exact_object(value, _SUPERVISOR_KEYS, name="supervisor manifest")
    _require_plain_json(supervisor, name="supervisor manifest")
    if (
        supervisor["schema_version"] != 1
        or type(supervisor["schema_version"]) is not int
        or supervisor["protocol"] != CONTROLLED_SUPERVISOR_PROTOCOL
        or supervisor["max_active"] != CONTROLLED_MAX_ACTIVE
        or type(supervisor["max_active"]) is not int
    ):
        raise ValueError("Supervisor manifest identity changed")
    plan = manifest.validate_dry_manifest(copy.deepcopy(supervisor["training_plan"]))
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(supervisor["staging_receipt"]), training_plan=plan
    )
    gate = _validate_gate_binding(
        supervisor["determinism_gate_receipt"],
        plan=plan,
        staged=staged,
        deep=deep_gate,
    )
    expected_schedule = _controlled_schedule(plan)
    if (
        supervisor["training_plan_sha256"] != _document_sha256(plan)
        or supervisor["staging_receipt_sha256"] != _document_sha256(staged)
        or supervisor["determinism_gate_receipt_sha256"] != _document_sha256(gate)
        or supervisor["schedule"] != expected_schedule
    ):
        raise ValueError("Supervisor manifest evidence binding changed")
    for field in (
        "training_plan_sha256",
        "staging_receipt_sha256",
        "determinism_gate_receipt_sha256",
    ):
        _exact_sha256(supervisor[field], name=f"supervisor.{field}")
    for index, raw in enumerate(supervisor["schedule"]):
        entry = _exact_object(raw, _SCHEDULE_KEYS, name=f"schedule[{index}]")
        if entry != expected_schedule[index]:
            raise ValueError("Supervisor schedule changed")
    _validate_self_hash(supervisor, name="supervisor manifest")
    return copy.deepcopy(supervisor)


def _load_supervisor_manifest(
    state_dir: Path, *, deep_gate: bool
) -> tuple[dict[str, Any], str]:
    state_dir = _real_directory(state_dir, name="supervisor state directory")
    allowed = {_STATE_FILE, _RUNS_DIRECTORY, _COMPLETION_FILE}
    observed = {path.name for path in state_dir.iterdir()}
    if not {_STATE_FILE, _RUNS_DIRECTORY}.issubset(observed) or not observed.issubset(
        allowed
    ):
        raise ValueError("Supervisor state root inventory changed")
    value, file_sha256 = strict_config.load_canonical_json_object(
        state_dir / _STATE_FILE
    )
    supervisor = _validate_supervisor_manifest(value, deep_gate=deep_gate)
    runs = _real_directory(state_dir / _RUNS_DIRECTORY, name="supervisor runs directory")
    expected_names = {_run_directory_name(entry) for entry in supervisor["schedule"]}
    actual_names = {path.name for path in runs.iterdir()}
    if actual_names != expected_names:
        raise ValueError("Supervisor run-directory inventory changed")
    for entry in supervisor["schedule"]:
        run_directory = _real_directory(
            runs / _run_directory_name(entry),
            name=f"state for {entry['run_id']}",
        )
        _real_directory(
            run_directory / _OBSERVATIONS_DIRECTORY,
            name=f"status observations for {entry['run_id']}",
        )
    return supervisor, file_sha256


def _capture_layout_identities(
    state_dir: Path,
    supervisor: Mapping[str, Any],
) -> dict[str, Any]:
    state_identity = _directory_identity(
        state_dir,
        name="supervisor state directory",
    )
    runs = state_dir / _RUNS_DIRECTORY
    runs_identity = _directory_identity(
        runs,
        name="supervisor runs directory",
    )
    per_run: dict[str, dict[str, tuple[int, int]]] = {}
    for entry in supervisor["schedule"]:
        paths = _run_paths(state_dir, entry)
        per_run[entry["run_id"]] = {
            "root": _directory_identity(
                paths["root"],
                name=f"state for {entry['run_id']}",
            ),
            "observations": _directory_identity(
                paths["observations"],
                name=f"observations for {entry['run_id']}",
            ),
        }
    _assert_directory_path_identity(
        state_dir,
        state_identity,
        name="supervisor state directory",
    )
    _assert_directory_path_identity(
        runs,
        runs_identity,
        name="supervisor runs directory",
    )
    return {
        "state": state_identity,
        "runs": runs_identity,
        "per_run": per_run,
    }


def _intent_receipt(
    *,
    supervisor: Mapping[str, Any],
    entry: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    request_receipt = preflight["request_receipt"]
    payload = {
        "schema_version": 1,
        "protocol": CONTROLLED_CREATE_INTENT_PROTOCOL,
        "state_manifest_sha256": _document_sha256(supervisor),
        "queue_index": entry["queue_index"],
        "run_id": entry["run_id"],
        "job_name": entry["job_name"],
        "plan_sha256": supervisor["training_plan_sha256"],
        "staging_receipt_sha256": supervisor["staging_receipt_sha256"],
        "determinism_gate_receipt_sha256": supervisor[
            "determinism_gate_receipt_sha256"
        ],
        "preflight_receipt": copy.deepcopy(preflight),
        "preflight_receipt_sha256": _document_sha256(preflight),
        "request_receipt_sha256": _document_sha256(request_receipt),
        "request_sha256": request_receipt["request_sha256"],
    }
    return _seal(payload)


def _validate_intent(
    value: object,
    *,
    supervisor: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    intent = _exact_object(value, _INTENT_KEYS, name="controlled create intent")
    _require_plain_json(intent, name="controlled create intent")
    preflight = training_launch.validate_training_preflight_receipt(
        intent["preflight_receipt"],
        training_plan=supervisor["training_plan"],
        staging_receipt=supervisor["staging_receipt"],
    )
    if (
        preflight["run_id"] != entry["run_id"]
        or preflight["job_name"] != entry["job_name"]
    ):
        raise ValueError("Create intent preflight selected a different run")
    expected = _intent_receipt(
        supervisor=supervisor,
        entry=entry,
        preflight=preflight,
    )
    if aws.canonical_json_bytes(intent) != aws.canonical_json_bytes(expected):
        raise ValueError("Controlled create intent evidence changed")
    _validate_self_hash(intent, name="controlled create intent")
    return copy.deepcopy(intent)


def _run_paths(state_dir: Path, entry: Mapping[str, Any]) -> dict[str, Path]:
    root = state_dir / _RUNS_DIRECTORY / _run_directory_name(entry)
    return {
        "root": root,
        "observations": root / _OBSERVATIONS_DIRECTORY,
        "intent": root / _CREATE_INTENT_FILE,
        "submission": root / _SUBMISSION_FILE,
        "terminal": root / _TERMINAL_FILE,
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    value, _ = strict_config.load_canonical_json_object(path)
    return value


def _load_run_state(
    *,
    state_dir: Path,
    supervisor: Mapping[str, Any],
    entry: Mapping[str, Any],
    root_identity: tuple[int, int],
    observations_identity: tuple[int, int],
) -> dict[str, Any]:
    paths = _run_paths(state_dir, entry)
    root = _real_directory(paths["root"], name=f"state for {entry['run_id']}")
    _assert_directory_path_identity(
        root,
        root_identity,
        name=f"state for {entry['run_id']}",
    )
    observations = _real_directory(
        paths["observations"], name=f"observations for {entry['run_id']}"
    )
    _assert_directory_path_identity(
        observations,
        observations_identity,
        name=f"observations for {entry['run_id']}",
    )
    allowed = {
        _OBSERVATIONS_DIRECTORY,
        _CREATE_INTENT_FILE,
        _SUBMISSION_FILE,
        _TERMINAL_FILE,
    }
    if not {path.name for path in root.iterdir()}.issubset(allowed):
        raise ValueError(f"Run-state inventory changed: {entry['run_id']}")

    plan = supervisor["training_plan"]
    staged = supervisor["staging_receipt"]
    intent = _load_optional_json(paths["intent"])
    if intent is not None:
        intent = _validate_intent(
            intent,
            supervisor=supervisor,
            entry=entry,
        )
    preflight = None if intent is None else intent["preflight_receipt"]

    submission = _load_optional_json(paths["submission"])
    if submission is not None:
        if intent is None:
            raise ValueError("Submission exists without its create intent")
        submission = training_launch.validate_training_submission_receipt(
            submission,
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
        )
        if submission["run_id"] != entry["run_id"]:
            raise ValueError("Persisted submission selected a different run")

    observation_files = sorted(observations.iterdir(), key=lambda path: path.name)
    statuses: list[dict[str, Any]] = []
    if observation_files and submission is None:
        raise ValueError("Status observations exist without a submission")
    for index, path in enumerate(observation_files, start=1):
        match = _STATUS_FILE.fullmatch(path.name)
        if match is None or int(match.group(1)) != index:
            raise ValueError("Status observation sequence changed")
        status, _ = strict_config.load_canonical_json_object(path)
        status = training_launch.validate_training_status_receipt(
            status,
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
        if status["run_id"] != entry["run_id"]:
            raise ValueError("Persisted status selected a different run")
        statuses.append(status)

    terminal = _load_optional_json(paths["terminal"])
    if terminal is not None:
        if submission is None:
            raise ValueError("Terminal receipt exists without a submission")
        terminal = training_launch.validate_training_terminal_receipt(
            terminal,
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
        if terminal["run_id"] != entry["run_id"]:
            raise ValueError("Persisted terminal selected a different run")

    if intent is None:
        phase = "queued"
    elif submission is None:
        phase = "ambiguous"
    elif terminal is None:
        phase = "active"
    elif (
        terminal["terminal_status"] == "Completed"
        and terminal["succeeded"] is True
    ):
        phase = "completed"
    else:
        phase = "failed"
    _assert_directory_path_identity(
        root,
        root_identity,
        name=f"state for {entry['run_id']}",
    )
    _assert_directory_path_identity(
        observations,
        observations_identity,
        name=f"observations for {entry['run_id']}",
    )
    return {
        "entry": copy.deepcopy(entry),
        "paths": paths,
        "phase": phase,
        "preflight": preflight,
        "intent": intent,
        "submission": submission,
        "statuses": statuses,
        "terminal": terminal,
        "root_identity": root_identity,
        "observations_identity": observations_identity,
    }


def _completed_fold_order(outer_fold: int) -> list[dict[str, Any]]:
    if type(outer_fold) is not int or outer_fold not in strict_config.CONTROLLED_FOLDS:
        raise ValueError("outer_fold must be one frozen controlled fold")
    cells = [
        {
            "outer_fold": outer_fold,
            "query_view": query_view,
            "sampler": sampler,
            "experiment_seed": seed,
        }
        for query_view in strict_config.CONTROLLED_QUERY_VIEWS
        for sampler in strict_config.CONTROLLED_SAMPLERS
        for seed in strict_config.CONTROLLED_SEEDS
    ]
    return sorted(
        cells,
        key=lambda cell: (
            f"{cell['query_view']}_{cell['sampler']}_"
            f"seed{cell['experiment_seed']}"
        ),
    )


def _completed_fold_source_bundle(plan: Mapping[str, Any]) -> dict[str, Any]:
    source = plan["sources"]
    digest = source["source_bundle_sha256"]
    return {
        "name": f"source-{digest}.tar.gz",
        "size": source["source_bundle_size"],
        "sha256": digest,
        "inventory_sha256": source["source_inventory_sha256"],
        "commit_epoch": source["commit_epoch"],
    }


def _completed_fold_payload(
    *,
    supervisor: Mapping[str, Any],
    outer_fold: int,
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = supervisor["training_plan"]
    staged = supervisor["staging_receipt"]
    expected_cells = _completed_fold_order(outer_fold)
    by_cell = {
        (
            state["entry"]["outer_fold"],
            state["entry"]["query_view"],
            state["entry"]["sampler"],
            state["entry"]["experiment_seed"],
        ): state
        for state in states
    }
    systems: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(expected_cells):
        key = (
            cell["outer_fold"],
            cell["query_view"],
            cell["sampler"],
            cell["experiment_seed"],
        )
        state = by_cell.get(key)
        if state is None or state["phase"] != "completed":
            run_id = None if state is None else state["entry"]["run_id"]
            raise RuntimeError(
                "Completed-fold evidence requires all twelve successful terminals: "
                f"missing_or_incomplete={run_id or key!r}"
            )
        entry = state["entry"]
        preflight = state["preflight"]
        submission = state["submission"]
        terminal = state["terminal"]
        systems.append(
            {
                "ordinal": ordinal,
                "queue_index": entry["queue_index"],
                "cell": copy.deepcopy(cell),
                "run_id": entry["run_id"],
                "job_name": entry["job_name"],
                "preflight_receipt": copy.deepcopy(preflight),
                "preflight_receipt_sha256": _document_sha256(preflight),
                "submission_receipt": copy.deepcopy(submission),
                "submission_receipt_sha256": _document_sha256(submission),
                "terminal_receipt": copy.deepcopy(terminal),
                "terminal_receipt_sha256": _document_sha256(terminal),
                "request_receipt_sha256": _document_sha256(
                    preflight["request_receipt"]
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol": COMPLETED_FOLD_EVIDENCE_PROTOCOL,
        "outer_fold": outer_fold,
        "attempt_id": plan["attempt"]["attempt_id"],
        "state_manifest_sha256": _document_sha256(supervisor),
        "training_plan": copy.deepcopy(plan),
        "training_plan_sha256": _document_sha256(plan),
        "training_staging_receipt": copy.deepcopy(staged),
        "training_staging_receipt_sha256": _document_sha256(staged),
        "source_bundle": _completed_fold_source_bundle(plan),
        "systems": systems,
    }


def validate_completed_fold_evidence(value: object) -> dict[str, Any]:
    """Contextually validate one sealed twelve-system completed-fold view."""

    evidence = _exact_object(
        value,
        _COMPLETED_FOLD_EVIDENCE_KEYS,
        name="completed-fold evidence",
    )
    _require_plain_json(evidence, name="completed-fold evidence")
    if (
        type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != 1
        or evidence["protocol"] != COMPLETED_FOLD_EVIDENCE_PROTOCOL
    ):
        raise ValueError("Completed-fold evidence protocol identity changed")
    outer_fold = evidence["outer_fold"]
    expected_cells = _completed_fold_order(outer_fold)
    plan = manifest.validate_dry_manifest(copy.deepcopy(evidence["training_plan"]))
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(evidence["training_staging_receipt"]),
        training_plan=plan,
    )
    if (
        evidence["attempt_id"] != plan["attempt"]["attempt_id"]
        or evidence["training_plan_sha256"] != _document_sha256(plan)
        or evidence["training_staging_receipt_sha256"] != _document_sha256(staged)
        or evidence["source_bundle"] != _completed_fold_source_bundle(plan)
    ):
        raise ValueError("Completed-fold plan/staging/source binding changed")
    _exact_sha256(
        evidence["state_manifest_sha256"],
        name="completed-fold.state_manifest_sha256",
    )
    controlled_by_cell = {
        (
            run["cell"]["outer_fold"],
            run["cell"]["query_view"],
            run["cell"]["sampler"],
            run["cell"]["experiment_seed"],
        ): run
        for run in plan["controlled_runs"]
    }
    schedule_by_cell = {
        (
            entry["outer_fold"],
            entry["query_view"],
            entry["sampler"],
            entry["experiment_seed"],
        ): entry
        for entry in _controlled_schedule(plan)
    }
    systems = evidence["systems"]
    if type(systems) is not list or len(systems) != 12:
        raise ValueError("Completed-fold evidence requires exactly twelve systems")
    normalized_systems: list[dict[str, Any]] = []
    queue_indexes: set[int] = set()
    for ordinal, (raw, expected_cell) in enumerate(zip(systems, expected_cells)):
        system = _exact_object(
            raw,
            _COMPLETED_FOLD_SYSTEM_KEYS,
            name=f"completed-fold.systems[{ordinal}]",
        )
        cell = system["cell"]
        if type(cell) is not dict or cell != expected_cell:
            raise ValueError("Completed-fold system cell order changed")
        key = (
            cell["outer_fold"],
            cell["query_view"],
            cell["sampler"],
            cell["experiment_seed"],
        )
        run = controlled_by_cell.get(key)
        schedule_entry = schedule_by_cell.get(key)
        if run is None or schedule_entry is None:
            raise ValueError("Completed-fold cell is absent from the training plan")
        queue_index = system["queue_index"]
        if (
            type(system["ordinal"]) is not int
            or system["ordinal"] != ordinal
            or type(queue_index) is not int
            or queue_index != schedule_entry["queue_index"]
            or queue_index in queue_indexes
            or system["run_id"] != run["run_id"]
            or system["job_name"] != run["job_name"]
        ):
            raise ValueError("Completed-fold system launch identity changed")
        queue_indexes.add(queue_index)
        preflight = training_launch.validate_training_preflight_receipt(
            copy.deepcopy(system["preflight_receipt"]),
            training_plan=plan,
            staging_receipt=staged,
        )
        submission = training_launch.validate_training_submission_receipt(
            copy.deepcopy(system["submission_receipt"]),
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
        )
        terminal = training_launch.validate_training_terminal_receipt(
            copy.deepcopy(system["terminal_receipt"]),
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
        if (
            preflight["run_id"] != run["run_id"]
            or submission["run_id"] != run["run_id"]
            or terminal["run_id"] != run["run_id"]
            or terminal["terminal_status"] != "Completed"
            or terminal["succeeded"] is not True
            or system["preflight_receipt_sha256"] != _document_sha256(preflight)
            or system["submission_receipt_sha256"] != _document_sha256(submission)
            or system["terminal_receipt_sha256"] != _document_sha256(terminal)
            or system["request_receipt_sha256"]
            != _document_sha256(preflight["request_receipt"])
        ):
            raise ValueError("Completed-fold receipt chain changed")
        normalized_system = copy.deepcopy(system)
        normalized_system["preflight_receipt"] = preflight
        normalized_system["submission_receipt"] = submission
        normalized_system["terminal_receipt"] = terminal
        normalized_systems.append(normalized_system)
    _validate_self_hash(evidence, name="completed-fold evidence")
    normalized = copy.deepcopy(evidence)
    normalized["training_plan"] = plan
    normalized["training_staging_receipt"] = staged
    normalized["systems"] = normalized_systems
    return normalized


def load_completed_fold_evidence(
    *,
    state_dir: Path,
    outer_fold: int,
) -> dict[str, Any]:
    """Load one fold only after all twelve local terminal chains are sealed."""

    _completed_fold_order(outer_fold)
    state_dir = _real_directory(state_dir, name="supervisor state directory")
    supervisor, manifest_file_sha256 = _load_supervisor_manifest(
        state_dir,
        deep_gate=True,
    )
    identities = _capture_layout_identities(state_dir, supervisor)
    selected = [
        entry for entry in supervisor["schedule"] if entry["outer_fold"] == outer_fold
    ]
    if len(selected) != 12:
        raise ValueError("Supervisor schedule does not contain twelve fold systems")
    states = [
        _load_run_state(
            state_dir=state_dir,
            supervisor=supervisor,
            entry=entry,
            root_identity=identities["per_run"][entry["run_id"]]["root"],
            observations_identity=identities["per_run"][entry["run_id"]][
                "observations"
            ],
        )
        for entry in selected
    ]
    payload = _completed_fold_payload(
        supervisor=supervisor,
        outer_fold=outer_fold,
        states=states,
    )
    reloaded, reloaded_sha256 = _load_supervisor_manifest(state_dir, deep_gate=False)
    if (
        reloaded_sha256 != manifest_file_sha256
        or aws.canonical_json_bytes(reloaded)
        != aws.canonical_json_bytes(supervisor)
    ):
        raise RuntimeError("Supervisor manifest changed during completed-fold loading")
    return validate_completed_fold_evidence(_seal(payload))


def _completion_receipt(
    supervisor: Mapping[str, Any], states: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(states) != 60 or any(state["phase"] != "completed" for state in states):
        raise ValueError("Completion requires all 60 successful controlled jobs")
    terminal_hashes = {
        state["entry"]["run_id"]: _document_sha256(state["terminal"])
        for state in states
    }
    payload = {
        "schema_version": 1,
        "protocol": CONTROLLED_SUPERVISOR_COMPLETION_PROTOCOL,
        "state_manifest_sha256": _document_sha256(supervisor),
        "completed_runs": 60,
        "terminal_receipt_sha256_by_run": terminal_hashes,
        "succeeded": True,
    }
    return _seal(payload)


def _validate_completion(
    value: object,
    *,
    supervisor: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completion = _exact_object(value, _COMPLETION_KEYS, name="supervisor completion")
    expected = _completion_receipt(supervisor, states)
    if aws.canonical_json_bytes(completion) != aws.canonical_json_bytes(expected):
        raise ValueError("Supervisor completion evidence changed")
    _validate_self_hash(completion, name="supervisor completion")
    return copy.deepcopy(completion)


def _assert_restartable(states: Sequence[Mapping[str, Any]]) -> None:
    active = [state for state in states if state["phase"] == "active"]
    if len(active) > CONTROLLED_MAX_ACTIVE:
        raise RuntimeError("Persisted state exceeds four active controlled jobs")
    ambiguous = [state["entry"]["run_id"] for state in states if state["phase"] == "ambiguous"]
    if ambiguous:
        raise RuntimeError(
            "Create intent lacks a sealed submission; refusing retry or reconciliation: "
            + ", ".join(ambiguous)
        )
    failed = [state["entry"]["run_id"] for state in states if state["phase"] == "failed"]
    if failed:
        raise RuntimeError(
            "Controlled training has terminal failure evidence: " + ", ".join(failed)
        )
    queued_seen = False
    for state in states:
        if state["phase"] == "queued":
            queued_seen = True
        elif queued_seen:
            raise ValueError("Started controlled runs are not one schedule prefix")


def _snapshot(
    supervisor: Mapping[str, Any], states: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    phases = ("queued", "active", "completed")
    counts = {
        phase: sum(state["phase"] == phase for state in states) for phase in phases
    }
    active = [state["entry"]["run_id"] for state in states if state["phase"] == "active"]
    completed = [
        state["entry"]["run_id"] for state in states if state["phase"] == "completed"
    ]
    next_run = next(
        (
            state["entry"]["run_id"]
            for state in states
            if state["phase"] == "queued"
        ),
        None,
    )
    payload = {
        "schema_version": 1,
        "protocol": CONTROLLED_SUPERVISOR_SNAPSHOT_PROTOCOL,
        "state_manifest_sha256": _document_sha256(supervisor),
        "max_active": CONTROLLED_MAX_ACTIVE,
        "counts": counts,
        "active_run_ids": active,
        "completed_run_ids": completed,
        "next_run_id": next_run,
        "complete": counts["completed"] == 60,
    }
    return _seal(payload)


class _CreateIntentSageMakerProxy:
    def __init__(
        self,
        delegate: object,
        *,
        expected_request: Mapping[str, Any],
        intent_path: Path,
        intent_receipt: Mapping[str, Any],
        intent_parent_identity: tuple[int, int],
    ) -> None:
        self._delegate = delegate
        self._expected_request = copy.deepcopy(expected_request)
        self._intent_path = intent_path
        self._intent_receipt = copy.deepcopy(intent_receipt)
        self._intent_parent_identity = intent_parent_identity
        self.called = False

    def create_training_job(self, **request: Any) -> Any:
        if self.called:
            raise RuntimeError("CreateTrainingJob was invoked more than once")
        if aws.canonical_json_bytes(request) != aws.canonical_json_bytes(
            self._expected_request
        ):
            raise ValueError("CreateTrainingJob request differs from the sealed preflight")
        _publish_json_absent(
            self._intent_path,
            self._intent_receipt,
            parent_identity=self._intent_parent_identity,
        )
        _assert_directory_path_identity(
            self._intent_path.parent,
            self._intent_parent_identity,
            name="create-intent parent",
        )
        self.called = True
        return self._delegate.create_training_job(**request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class ControlledTrainingSupervisor:
    """One validated supervisor process over one immutable local state tree."""

    def __init__(self, clients: aws.AwsClients, *, state_dir: Path) -> None:
        if not isinstance(clients, aws.AwsClients):
            raise TypeError("clients must be one AwsClients bundle")
        self._clients = clients
        self._state_dir = _real_directory(state_dir, name="supervisor state directory")
        supervisor, manifest_file_sha256 = _load_supervisor_manifest(
            self._state_dir, deep_gate=True
        )
        self._supervisor = supervisor
        self._manifest_file_sha256 = manifest_file_sha256
        self._layout_identities = _capture_layout_identities(
            self._state_dir,
            self._supervisor,
        )
        states = self._load_states()
        _assert_restartable(states)
        self._validate_optional_completion(states)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def _reload_supervisor(self) -> dict[str, Any]:
        _assert_directory_path_identity(
            self._state_dir,
            self._layout_identities["state"],
            name="supervisor state directory",
        )
        _assert_directory_path_identity(
            self._state_dir / _RUNS_DIRECTORY,
            self._layout_identities["runs"],
            name="supervisor runs directory",
        )
        supervisor, manifest_file_sha256 = _load_supervisor_manifest(
            self._state_dir, deep_gate=False
        )
        if (
            manifest_file_sha256 != self._manifest_file_sha256
            or aws.canonical_json_bytes(supervisor)
            != aws.canonical_json_bytes(self._supervisor)
        ):
            raise RuntimeError("Supervisor manifest changed during this process")
        _assert_directory_path_identity(
            self._state_dir,
            self._layout_identities["state"],
            name="supervisor state directory",
        )
        return supervisor

    def _load_states(self) -> list[dict[str, Any]]:
        supervisor = self._reload_supervisor()
        return [
            _load_run_state(
                state_dir=self._state_dir,
                supervisor=supervisor,
                entry=entry,
                root_identity=self._layout_identities["per_run"][
                    entry["run_id"]
                ]["root"],
                observations_identity=self._layout_identities["per_run"][
                    entry["run_id"]
                ]["observations"],
            )
            for entry in supervisor["schedule"]
        ]

    def _validate_optional_completion(
        self, states: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        path = self._state_dir / _COMPLETION_FILE
        completion = _load_optional_json(path)
        if completion is None:
            return None
        return _validate_completion(
            completion,
            supervisor=self._supervisor,
            states=states,
        )

    def _publish_status(
        self, state: Mapping[str, Any], status: Mapping[str, Any]
    ) -> None:
        path = state["paths"]["observations"] / (
            f"status-{len(state['statuses']) + 1:06d}.json"
        )
        _publish_json_absent(
            path,
            status,
            parent_identity=state["observations_identity"],
        )

    def _observe_active(self, states: Sequence[Mapping[str, Any]]) -> None:
        plan = self._supervisor["training_plan"]
        staged = self._supervisor["staging_receipt"]
        for state in states:
            if state["phase"] != "active":
                continue
            status = training_launch.describe_training_job_status(
                self._clients,
                training_plan=plan,
                staging_receipt=staged,
                preflight_receipt=state["preflight"],
                submission_receipt=state["submission"],
            )
            status = training_launch.validate_training_status_receipt(
                status,
                training_plan=plan,
                staging_receipt=staged,
                preflight_receipt=state["preflight"],
                submission_receipt=state["submission"],
            )
            self._publish_status(state, status)
            remote_status = status["snapshot"]["training_job_status"]
            if remote_status in {"InProgress", "Stopping"}:
                continue
            terminal = training_launch.verify_terminal_training_job(
                self._clients,
                training_plan=plan,
                staging_receipt=staged,
                preflight_receipt=state["preflight"],
                submission_receipt=state["submission"],
            )
            terminal = training_launch.validate_training_terminal_receipt(
                terminal,
                training_plan=plan,
                staging_receipt=staged,
                preflight_receipt=state["preflight"],
                submission_receipt=state["submission"],
            )
            _publish_json_absent(
                state["paths"]["terminal"],
                terminal,
                parent_identity=state["root_identity"],
            )
            if terminal["succeeded"] is not True:
                raise RuntimeError(
                    "Controlled training ended unsuccessfully: "
                    f"{state['entry']['run_id']}={terminal['terminal_status']}"
                )

    def _preflight(self, state: Mapping[str, Any]) -> dict[str, Any]:
        plan = self._supervisor["training_plan"]
        staged = self._supervisor["staging_receipt"]
        preflight = training_launch.preflight_training_job(
            self._clients,
            training_plan=plan,
            staging_receipt=staged,
            run_id=state["entry"]["run_id"],
        )
        preflight = training_launch.validate_training_preflight_receipt(
            preflight,
            training_plan=plan,
            staging_receipt=staged,
        )
        return preflight

    def _submit(self, state: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
        plan = self._supervisor["training_plan"]
        staged = self._supervisor["staging_receipt"]
        intent = _intent_receipt(
            supervisor=self._supervisor,
            entry=state["entry"],
            preflight=preflight,
        )
        proxy = _CreateIntentSageMakerProxy(
            self._clients.sagemaker,
            expected_request=preflight["request_receipt"]["request"],
            intent_path=state["paths"]["intent"],
            intent_receipt=intent,
            intent_parent_identity=state["root_identity"],
        )
        proxied_clients = dataclasses.replace(self._clients, sagemaker=proxy)
        submission = training_launch.submit_training_job_once(
            proxied_clients,
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
        )
        if not proxy.called:
            raise RuntimeError("Submission returned without one CreateTrainingJob call")
        submission = training_launch.validate_training_submission_receipt(
            submission,
            training_plan=plan,
            staging_receipt=staged,
            preflight_receipt=preflight,
        )
        persisted_intent, _ = strict_config.load_canonical_json_object(
            state["paths"]["intent"]
        )
        _validate_intent(
            persisted_intent,
            supervisor=self._supervisor,
            entry=state["entry"],
        )
        _publish_json_absent(
            state["paths"]["submission"],
            submission,
            parent_identity=state["root_identity"],
        )

    def _fill_available_slots(self) -> None:
        while True:
            states = self._load_states()
            _assert_restartable(states)
            active_count = sum(state["phase"] == "active" for state in states)
            if active_count >= CONTROLLED_MAX_ACTIVE:
                return
            candidate = next(
                (
                    state
                    for state in states
                    if state["phase"] == "queued"
                ),
                None,
            )
            if candidate is None:
                return
            preflight = self._preflight(candidate)
            self._submit(candidate, preflight)

    def advance_once(self) -> dict[str, Any]:
        """Observe every active job once and immediately fill every free slot."""

        self._reload_supervisor()
        states = self._load_states()
        _assert_restartable(states)
        completion = self._validate_optional_completion(states)
        if completion is not None:
            return _snapshot(self._supervisor, states)
        self._observe_active(states)
        states = self._load_states()
        _assert_restartable(states)
        if all(state["phase"] == "completed" for state in states):
            completion = _completion_receipt(self._supervisor, states)
            _publish_json_absent(
                self._state_dir / _COMPLETION_FILE,
                completion,
                parent_identity=self._layout_identities["state"],
            )
            return _snapshot(self._supervisor, states)
        self._fill_available_slots()
        states = self._load_states()
        _assert_restartable(states)
        return _snapshot(self._supervisor, states)

    def run_until_complete(self, *, poll_interval_seconds: int) -> dict[str, Any]:
        """Run until all 60 jobs complete or the first strict failure occurs."""

        if type(poll_interval_seconds) is not int or poll_interval_seconds < 1:
            raise ValueError("poll_interval_seconds must be one positive exact integer")
        while True:
            snapshot = self.advance_once()
            if snapshot["complete"] is True:
                return snapshot
            time.sleep(poll_interval_seconds)


__all__: Sequence[str] = (
    "CONTROLLED_CREATE_INTENT_PROTOCOL",
    "CONTROLLED_MAX_ACTIVE",
    "CONTROLLED_SUPERVISOR_COMPLETION_PROTOCOL",
    "CONTROLLED_SUPERVISOR_PROTOCOL",
    "CONTROLLED_SUPERVISOR_SNAPSHOT_PROTOCOL",
    "ControlledTrainingSupervisor",
    "initialize_controlled_supervisor_state",
)
