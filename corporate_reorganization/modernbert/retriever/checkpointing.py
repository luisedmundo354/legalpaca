from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist


CHECKPOINT_PROTOCOL_SCHEMA_VERSION = 1
SELECTION_METADATA_SCHEMA_VERSION = 1
VALIDATION_HISTORY_SCHEMA_VERSION = 1
VALIDATION_PRIMARY_METRIC = "eval_validation_case_macro_set_recall_at_20"
VALIDATION_SECONDARY_METRIC = (
    "eval_validation_case_macro_first_gold_reciprocal_rank_full_ranking"
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())


def publish_new_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"Refusing stale artifact temporary file: {temporary_path}")
    published = False
    try:
        _write_fsynced_text(temporary_path, content)
        os.link(temporary_path, path)
        published = True
        temporary_path.unlink()
        fsync_directory(path.parent)
    except BaseException:
        if published and (path.exists() or path.is_symlink()):
            path.unlink()
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()
        fsync_directory(path.parent)
        raise


def replace_text_atomically(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.replace.tmp")
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlink artifact: {path}")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"Refusing stale replacement temporary file: {temporary_path}")
    try:
        _write_fsynced_text(temporary_path, content)
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()
            fsync_directory(path.parent)
        raise


def _failure_payload(context: str, error: BaseException, *, rank: int = 0) -> dict[str, Any]:
    return {
        "ok": False,
        "context": context,
        "rank": rank,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _raise_collective_failure(payload: Mapping[str, Any]) -> None:
    raise RuntimeError(
        f"{payload.get('context')} failed on rank {payload.get('rank')}: "
        f"{payload.get('error_type')}: {payload.get('message')}"
    )


def rank_zero_call(context: str, operation: Callable[[], Any]) -> Any:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(f"{context} requires an initialized process group")
    rank = dist.get_rank()
    status: list[object] = [None]
    if rank == 0:
        try:
            status[0] = {"ok": True, "value": operation()}
        except BaseException as error:
            status[0] = _failure_payload(context, error)
    dist.broadcast_object_list(status, src=0)
    payload = status[0]
    if type(payload) is not dict or type(payload.get("ok")) is not bool:
        raise RuntimeError(f"{context} returned a malformed collective status")
    if payload["ok"] is not True:
        _raise_collective_failure(payload)
    return payload.get("value")


def _gather_local_status(context: str, operation: Callable[[], Any]) -> list[dict[str, Any]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        value = operation()
        local_status: dict[str, Any] = {"ok": True, "rank": rank, "value": value}
    except BaseException as error:
        local_status = _failure_payload(context, error, rank=rank)
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_status)
    failures = [
        payload
        for payload in gathered
        if type(payload) is not dict or payload.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"{context} failed collectively: {failures}")
    return [dict(payload) for payload in gathered]


def _jsonable_state(value: object, *, path: str = "state") -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite value at {path}")
        return value
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite tensor at {path}")
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite NumPy array at {path}")
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable_state(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"Non-string mapping key at {path}")
        return {
            key: _jsonable_state(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    raise TypeError(f"Unsupported checkpoint state type at {path}: {type(value).__name__}")


def canonical_state_sha256(value: object) -> str:
    normalized = _jsonable_state(value)
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class CheckpointSelection:
    schema_version: int
    epoch: int
    global_step: int
    checkpoint_dir: str
    deepspeed_tag: str
    primary_metric: float
    secondary_metric: float
    ranking_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "checkpoint_dir": self.checkpoint_dir,
            "deepspeed_tag": self.deepspeed_tag,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "ranking_sha256": self.ranking_sha256,
        }


def validate_selection(selection: CheckpointSelection) -> None:
    if not isinstance(selection, CheckpointSelection):
        raise TypeError("selection must be CheckpointSelection")
    if (
        type(selection.schema_version) is not int
        or selection.schema_version != SELECTION_METADATA_SCHEMA_VERSION
    ):
        raise ValueError("Checkpoint selection schema version changed")
    if type(selection.epoch) is not int or selection.epoch < 1:
        raise ValueError("Checkpoint selection epoch must be a positive exact int")
    if type(selection.global_step) is not int or selection.global_step < 1:
        raise ValueError("Checkpoint selection global_step must be a positive exact int")
    if (
        type(selection.checkpoint_dir) is not str
        or selection.checkpoint_dir != f"checkpoint-{selection.global_step}"
    ):
        raise ValueError("Checkpoint selection directory and global step disagree")
    if (
        type(selection.deepspeed_tag) is not str
        or selection.deepspeed_tag != f"global_step{selection.global_step}"
    ):
        raise ValueError("Checkpoint selection DeepSpeed tag and global step disagree")
    for name, value in (
        ("primary_metric", selection.primary_metric),
        ("secondary_metric", selection.secondary_metric),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"Checkpoint selection {name} must be a finite exact float")
    if not _is_sha256(selection.ranking_sha256):
        raise ValueError("Checkpoint selection ranking_sha256 is invalid")


def choose_better_checkpoint(
    current: CheckpointSelection | None,
    candidate: CheckpointSelection,
) -> tuple[CheckpointSelection, bool]:
    validate_selection(candidate)
    if current is None:
        return candidate, True
    validate_selection(current)
    if candidate.epoch <= current.epoch:
        raise ValueError("Checkpoint candidates must be considered in increasing epoch order")
    if candidate.primary_metric > current.primary_metric:
        return candidate, True
    if (
        candidate.primary_metric == current.primary_metric
        and candidate.secondary_metric > current.secondary_metric
    ):
        return candidate, True
    return current, False


def _torch_save_new(value: object, path: Path) -> None:
    with path.open("xb") as target:
        torch.save(value, target)
        target.flush()
        os.fsync(target.fileno())


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.random.get_rng_state_all()
    return state


def _tree_inventory(root: Path, *, include_hashes: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Checkpoint inventory forbids symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Unexpected checkpoint entry type: {relative}")
        record: dict[str, Any] = {"path": relative, "size": path.stat().st_size}
        if record["size"] < 1:
            raise ValueError(f"Checkpoint file is empty: {relative}")
        if include_hashes:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def _validate_deepspeed_layout(
    checkpoint_root: Path,
    *,
    tag: str,
    world_size: int,
) -> None:
    if not isinstance(checkpoint_root, Path):
        raise TypeError("checkpoint_root must be a Path")
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise RuntimeError(f"Checkpoint root must be a real directory: {checkpoint_root}")
    if type(tag) is not str or not tag or tag != Path(tag).name:
        raise ValueError(f"Invalid DeepSpeed checkpoint tag: {tag!r}")
    if type(world_size) is not int or world_size < 2:
        raise ValueError("DeepSpeed checkpoint world_size must be an exact integer >= 2")
    root_entries = sorted(checkpoint_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() for path in root_entries):
        raise RuntimeError("DeepSpeed checkpoint root forbids symlink entries")
    tag_dir = checkpoint_root / tag
    if not tag_dir.is_dir() or tag_dir.is_symlink():
        raise RuntimeError(f"Missing DeepSpeed tag directory: {tag_dir}")
    root_directories = sorted(
        path.name for path in root_entries if path.is_dir()
    )
    if root_directories != [tag]:
        raise RuntimeError(
            f"DeepSpeed checkpoint root directories changed: {root_directories}; expected {[tag]}"
        )
    expected_tag_files = {
        *{
            f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
            for rank in range(world_size)
        },
        *{
            f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
            for rank in range(world_size)
        },
    }
    actual_tag_entries = sorted(tag_dir.iterdir(), key=lambda path: path.name)
    actual_tag_files = {path.name for path in actual_tag_entries}
    if actual_tag_files != expected_tag_files:
        raise RuntimeError(
            "DeepSpeed tag inventory changed: "
            f"actual={sorted(actual_tag_files)}, expected={sorted(expected_tag_files)}"
        )
    if any(
        path.is_symlink() or not path.is_file() or path.stat().st_size < 1
        for path in actual_tag_entries
    ):
        raise RuntimeError("DeepSpeed shards must be non-empty regular files")
    recovery_script = checkpoint_root / "zero_to_fp32.py"
    if (
        not recovery_script.is_file()
        or recovery_script.is_symlink()
        or recovery_script.stat().st_size < 1
    ):
        raise RuntimeError("DeepSpeed checkpoint is missing zero_to_fp32.py")
    latest_path = checkpoint_root / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        raise RuntimeError("Controlled save_latest=False checkpoint unexpectedly contains latest")


def _selection_from_payload(value: object) -> CheckpointSelection:
    expected_keys = {
        "schema_version",
        "epoch",
        "global_step",
        "checkpoint_dir",
        "deepspeed_tag",
        "primary_metric",
        "secondary_metric",
        "ranking_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise RuntimeError("Checkpoint manifest selection schema changed")
    selection = CheckpointSelection(**value)
    validate_selection(selection)
    return selection


def _expected_checkpoint_inventory_paths(
    selection: CheckpointSelection,
    *,
    world_size: int,
) -> set[str]:
    validate_selection(selection)
    if type(world_size) is not int or world_size < 2:
        raise ValueError("Checkpoint inventory world_size must be an exact integer >= 2")
    tag = selection.deepspeed_tag
    return {
        "zero_to_fp32.py",
        "scheduler.pt",
        "training_args.bin",
        "trainer_state.json",
        *{f"rng_state_{rank}.pth" for rank in range(world_size)},
        *{
            f"{tag}/zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
            for rank in range(world_size)
        },
        *{
            f"{tag}/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
            for rank in range(world_size)
        },
    }


def _validate_client_state(client_state: object) -> dict[str, Any]:
    if type(client_state) is not dict or set(client_state) != {"controlled_state"}:
        raise ValueError(
            "DeepSpeed client_state must contain exactly the reserved controlled_state namespace"
        )
    controlled_state = client_state["controlled_state"]
    if type(controlled_state) is not dict or not controlled_state:
        raise ValueError("controlled_state must be a non-empty exact object")
    _jsonable_state(controlled_state, path="client_state.controlled_state")
    return {"controlled_state": dict(controlled_state)}


def _extract_loaded_client_state(client_state: object) -> dict[str, Any]:
    if type(client_state) is not dict or "controlled_state" not in client_state:
        raise ValueError("Loaded DeepSpeed client state is missing controlled_state")
    controlled_state = client_state["controlled_state"]
    if type(controlled_state) is not dict or not controlled_state:
        raise ValueError("Loaded controlled_state must be a non-empty exact object")
    _jsonable_state(controlled_state, path="loaded_client_state.controlled_state")
    return {"controlled_state": dict(controlled_state)}


def save_controlled_checkpoint(
    *,
    output_dir: Path,
    engine,
    scheduler,
    trainer_state,
    training_args,
    selection: CheckpointSelection,
    client_state: Mapping[str, Any],
    expected_world_size: int,
) -> dict[str, Any]:
    """Collectively save and atomically publish one complete ZeRO-3 checkpoint."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Controlled checkpoint saving requires an initialized process group")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_preflight: dict[str, Any] = {}

    def run_local_preflight() -> dict[str, Any]:
        if type(expected_world_size) is not int or expected_world_size < 2:
            raise ValueError("expected_world_size must be an exact integer >= 2")
        if world_size != expected_world_size:
            raise RuntimeError(
                f"Checkpoint world_size={world_size}; expected {expected_world_size}"
            )
        if not isinstance(output_dir, Path):
            raise TypeError("Checkpoint output_dir must be a Path")
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(f"Checkpoint output_dir must be a real directory: {output_dir}")
        validate_selection(selection)
        engine_global_steps = getattr(engine, "global_steps", None)
        if type(engine_global_steps) is not int or engine_global_steps != selection.global_step:
            raise RuntimeError(
                f"DeepSpeed global_steps={engine_global_steps} does not match "
                f"Trainer global_step={selection.global_step}"
            )
        if not callable(getattr(engine, "zero_optimization_stage", None)):
            raise TypeError("DeepSpeed engine does not expose zero_optimization_stage")
        if int(engine.zero_optimization_stage()) != 3:
            raise RuntimeError("Controlled checkpoint requires DeepSpeed ZeRO stage 3")
        if not callable(getattr(engine, "bfloat16_enabled", None)):
            raise TypeError("DeepSpeed engine does not expose bfloat16_enabled")
        if engine.bfloat16_enabled() is not True:
            raise RuntimeError("Controlled checkpoint requires DeepSpeed BF16")
        if getattr(engine, "optimizer", None) is None:
            raise RuntimeError("Controlled checkpoint requires a DeepSpeed optimizer")
        dp_world_size = getattr(engine, "dp_world_size", None)
        if callable(dp_world_size):
            dp_world_size = dp_world_size()
        if type(dp_world_size) is not int or dp_world_size != expected_world_size:
            raise RuntimeError(
                f"DeepSpeed data-parallel world size={dp_world_size}; "
                f"expected {expected_world_size}"
            )
        if getattr(engine, "lr_scheduler", None) is not None:
            raise RuntimeError(
                "Controlled checkpoint requires the Transformers scheduler to remain external"
            )
        if type(getattr(scheduler, "last_epoch", None)) is not int:
            raise TypeError("External scheduler must expose an exact integer last_epoch")
        if scheduler.last_epoch != selection.global_step:
            raise RuntimeError(
                f"External scheduler last_epoch={scheduler.last_epoch} does not match "
                f"global_step={selection.global_step}"
            )
        if type(getattr(trainer_state, "global_step", None)) is not int:
            raise TypeError("Trainer state must expose an exact integer global_step")
        if trainer_state.global_step != selection.global_step:
            raise RuntimeError("Trainer state global step and checkpoint selection disagree")
        trainer_epoch = getattr(trainer_state, "epoch", None)
        if type(trainer_epoch) not in (int, float) or not math.isfinite(float(trainer_epoch)):
            raise TypeError("Trainer state epoch must be a finite number")
        if float(trainer_epoch) != float(selection.epoch):
            raise RuntimeError("Trainer state epoch and checkpoint selection disagree")
        exact_client_state = _validate_client_state(client_state)
        normalized_client_state = _jsonable_state(
            exact_client_state,
            path="client_state",
        )
        scheduler_state = scheduler.state_dict()
        values = {
            "output_dir": str(output_dir.resolve()),
            "selection_sha256": hashlib.sha256(
                canonical_json(selection.to_payload()).encode("utf-8")
            ).hexdigest(),
            "client_state_sha256": hashlib.sha256(
                canonical_json(normalized_client_state).encode("utf-8")
            ).hexdigest(),
            "scheduler_state_sha256": canonical_state_sha256(scheduler_state),
        }
        local_preflight.update(
            {
                "client_state": exact_client_state,
                "scheduler_state": scheduler_state,
                **values,
            }
        )
        return values

    preflight_statuses = _gather_local_status(
        "Controlled checkpoint preflight",
        run_local_preflight,
    )
    preflight_values = [status["value"] for status in preflight_statuses]
    if preflight_values != [preflight_values[0]] * world_size:
        raise RuntimeError(f"Checkpoint preflight differs across ranks: {preflight_values}")
    exact_client_state = local_preflight["client_state"]
    scheduler_state = local_preflight["scheduler_state"]
    client_state_sha256 = local_preflight["client_state_sha256"]
    scheduler_state_digest = local_preflight["scheduler_state_sha256"]

    final_path = output_dir / selection.checkpoint_dir
    incomplete_path = output_dir / f".{selection.checkpoint_dir}.incomplete"

    def create_incomplete() -> dict[str, Any]:
        if final_path.exists() or final_path.is_symlink():
            raise FileExistsError(f"Checkpoint target already exists: {final_path}")
        if incomplete_path.exists() or incomplete_path.is_symlink():
            raise FileExistsError(f"Stale incomplete checkpoint exists: {incomplete_path}")
        incomplete_path.mkdir()
        fsync_directory(output_dir)
        return {"incomplete_dir": incomplete_path.name}

    rank_zero_call("Controlled checkpoint directory creation", create_incomplete)

    save_statuses = _gather_local_status(
        "DeepSpeed checkpoint shard save",
        lambda: engine.save_checkpoint(
            str(incomplete_path),
            tag=selection.deepspeed_tag,
            client_state=exact_client_state,
            save_latest=False,
            exclude_frozen_parameters=False,
        ),
    )
    if any(status.get("value") is not True for status in save_statuses):
        raise RuntimeError(f"DeepSpeed save_checkpoint did not return exact True: {save_statuses}")

    rng_path = incomplete_path / f"rng_state_{rank}.pth"
    _gather_local_status(
        "Rank-local checkpoint RNG save",
        lambda: (_torch_save_new(capture_rng_state(), rng_path), True)[1],
    )

    def publish_rank_zero_checkpoint() -> dict[str, Any]:
        _torch_save_new(scheduler_state, incomplete_path / "scheduler.pt")
        _torch_save_new(training_args, incomplete_path / "training_args.bin")
        trainer_state_path = incomplete_path / "trainer_state.json"
        if trainer_state_path.exists() or trainer_state_path.is_symlink():
            raise FileExistsError(f"Trainer state already exists: {trainer_state_path}")
        trainer_state.save_to_json(str(trainer_state_path))
        parsed_trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        trainer_state_content = json.dumps(
            parsed_trainer_state,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        replace_text_atomically(trainer_state_path, trainer_state_content)
        with trainer_state_path.open("rb") as source:
            os.fsync(source.fileno())

        _validate_deepspeed_layout(
            incomplete_path,
            tag=selection.deepspeed_tag,
            world_size=world_size,
        )
        expected_rng_names = [f"rng_state_{process_rank}.pth" for process_rank in range(world_size)]
        if any(
            not (incomplete_path / name).is_file()
            or (incomplete_path / name).is_symlink()
            or (incomplete_path / name).stat().st_size < 1
            for name in expected_rng_names
        ):
            raise RuntimeError("Rank-local checkpoint RNG inventory is incomplete")
        expected_root_files = {
            "zero_to_fp32.py",
            "scheduler.pt",
            "training_args.bin",
            "trainer_state.json",
            *expected_rng_names,
        }
        actual_root_files = {
            path.name
            for path in incomplete_path.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if actual_root_files != expected_root_files:
            raise RuntimeError(
                "Checkpoint root file inventory changed before manifest publication: "
                f"actual={sorted(actual_root_files)}, expected={sorted(expected_root_files)}"
            )
        inventory = _tree_inventory(incomplete_path, include_hashes=True)
        manifest = {
            "schema_version": CHECKPOINT_PROTOCOL_SCHEMA_VERSION,
            "selection": selection.to_payload(),
            "world_size": world_size,
            "client_state_sha256": client_state_sha256,
            "scheduler_state_sha256": scheduler_state_digest,
            "rng_files": expected_rng_names,
            "files": inventory,
        }
        manifest_path = incomplete_path / "checkpoint_manifest.json"
        publish_new_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        manifest_sha256 = sha256_file(manifest_path)
        metadata = {
            "checkpoint_dir": final_path.name,
            "deepspeed_tag": selection.deepspeed_tag,
            "manifest_sha256": manifest_sha256,
            "scheduler_state_sha256": scheduler_state_digest,
            "client_state_sha256": client_state_sha256,
        }
        fsync_directory(incomplete_path / selection.deepspeed_tag)
        fsync_directory(incomplete_path)
        renamed = False
        try:
            if final_path.exists() or final_path.is_symlink():
                raise FileExistsError(
                    f"Checkpoint target appeared before atomic publication: {final_path}"
                )
            os.rename(incomplete_path, final_path)
            renamed = True
            fsync_directory(output_dir)
        except BaseException:
            if renamed and final_path.exists() and not incomplete_path.exists():
                os.rename(final_path, incomplete_path)
                fsync_directory(output_dir)
            raise
        return metadata

    published = rank_zero_call(
        "Controlled checkpoint atomic publication",
        publish_rank_zero_checkpoint,
    )
    if type(published) is not dict or published.get("checkpoint_dir") != final_path.name:
        raise RuntimeError("Controlled checkpoint publication returned malformed metadata")
    try:
        _gather_local_status(
            "Published checkpoint visibility check",
            lambda: (
                True
                if final_path.is_dir() and not final_path.is_symlink()
                else (_ for _ in ()).throw(
                    RuntimeError(
                        f"Published checkpoint is not a real directory: {final_path}"
                    )
                )
            ),
        )
    except BaseException as visibility_error:

        def rollback_invisible_checkpoint() -> dict[str, Any]:
            if incomplete_path.exists() or incomplete_path.is_symlink():
                raise RuntimeError(
                    f"Cannot roll back published checkpoint over {incomplete_path}"
                )
            if final_path.is_dir() and not final_path.is_symlink():
                os.rename(final_path, incomplete_path)
                fsync_directory(output_dir)
            elif final_path.exists() or final_path.is_symlink():
                raise RuntimeError(
                    f"Published checkpoint changed type before rollback: {final_path}"
                )
            return {"rolled_back": not final_path.exists()}

        rollback = rank_zero_call(
            "Published checkpoint visibility rollback",
            rollback_invisible_checkpoint,
        )
        if rollback != {"rolled_back": True}:
            raise RuntimeError("Checkpoint visibility rollback returned malformed metadata")
        raise RuntimeError("Published checkpoint failed collective visibility") from visibility_error
    dist.barrier()
    return dict(published)


def _load_checkpoint_manifest(checkpoint_root: Path) -> dict[str, Any]:
    if not isinstance(checkpoint_root, Path):
        raise TypeError("checkpoint_root must be a Path")
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise ValueError(f"Checkpoint root must be a real directory: {checkpoint_root}")
    manifest_path = checkpoint_root / "checkpoint_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Checkpoint manifest must be a regular file: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "selection",
        "world_size",
        "client_state_sha256",
        "scheduler_state_sha256",
        "rng_files",
        "files",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise RuntimeError("Checkpoint manifest schema changed")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != CHECKPOINT_PROTOCOL_SCHEMA_VERSION
    ):
        raise RuntimeError("Checkpoint manifest version changed")
    selection = _selection_from_payload(manifest["selection"])
    if selection.checkpoint_dir != checkpoint_root.name:
        raise RuntimeError("Checkpoint manifest selection and directory disagree")
    world_size = manifest["world_size"]
    if type(world_size) is not int or world_size < 2:
        raise RuntimeError("Checkpoint manifest world_size is invalid")
    _validate_deepspeed_layout(
        checkpoint_root,
        tag=selection.deepspeed_tag,
        world_size=world_size,
    )
    if not _is_sha256(manifest["client_state_sha256"]):
        raise RuntimeError("Checkpoint manifest client-state digest is invalid")
    if not _is_sha256(manifest["scheduler_state_sha256"]):
        raise RuntimeError("Checkpoint manifest scheduler-state digest is invalid")
    expected_rng_files = [f"rng_state_{rank}.pth" for rank in range(world_size)]
    if manifest["rng_files"] != expected_rng_files:
        raise RuntimeError("Checkpoint manifest RNG inventory changed")
    files = manifest["files"]
    if type(files) is not list or not files:
        raise RuntimeError("Checkpoint manifest file inventory is empty")
    expected_file_keys = {"path", "size", "sha256"}
    seen_paths: set[str] = set()
    observed_paths: list[str] = []
    for record in files:
        if type(record) is not dict or set(record) != expected_file_keys:
            raise RuntimeError("Checkpoint manifest file record is malformed")
        relative = record["path"]
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen_paths
        ):
            raise RuntimeError(f"Checkpoint manifest has invalid file path: {relative!r}")
        seen_paths.add(relative)
        observed_paths.append(relative)
        path = checkpoint_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or type(record["size"]) is not int
            or record["size"] != path.stat().st_size
            or not _is_sha256(record["sha256"])
            or record["sha256"] != sha256_file(path)
        ):
            raise RuntimeError(f"Checkpoint file does not match manifest: {relative}")
    if observed_paths != sorted(observed_paths):
        raise RuntimeError("Checkpoint manifest file inventory is not canonically ordered")
    expected_paths = _expected_checkpoint_inventory_paths(
        selection,
        world_size=world_size,
    )
    if seen_paths != expected_paths:
        raise RuntimeError(
            "Checkpoint manifest inventory schema changed: "
            f"recorded={sorted(seen_paths)}, expected={sorted(expected_paths)}"
        )
    actual_inventory = [
        record
        for record in _tree_inventory(checkpoint_root, include_hashes=True)
        if record["path"] != "checkpoint_manifest.json"
    ]
    if actual_inventory != files:
        raise RuntimeError("Checkpoint manifest inventory differs from disk")
    return manifest


def restore_rank_rng_state(checkpoint_root: Path) -> str:
    rank = dist.get_rank()
    rng_path = checkpoint_root / f"rng_state_{rank}.pth"
    if not rng_path.is_file() or rng_path.is_symlink():
        raise ValueError(f"Missing rank RNG state: {rng_path}")
    state = torch.load(rng_path, map_location="cpu", weights_only=False)
    expected_keys = {"python", "numpy", "cpu"}
    if torch.cuda.is_available():
        expected_keys.add("cuda")
    if type(state) is not dict or set(state) != expected_keys:
        raise RuntimeError(
            f"Rank RNG state fields changed: actual={sorted(state) if type(state) is dict else state}, "
            f"expected={sorted(expected_keys)}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        cuda_states = state["cuda"]
        if type(cuda_states) is not list or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint CUDA RNG state count changed")
        torch.cuda.random.set_rng_state_all(cuda_states)
    return sha256_file(rng_path)


def load_controlled_checkpoint(
    *,
    checkpoint_root: Path,
    engine,
    scheduler,
    selection: CheckpointSelection,
    expected_world_size: int,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Strictly load one explicit-tag checkpoint into a pristine engine on all ranks."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Controlled checkpoint loading requires an initialized process group")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_preflight: dict[str, Any] = {}

    def preflight() -> dict[str, Any]:
        if type(expected_world_size) is not int or expected_world_size < 2:
            raise ValueError("expected_world_size must be an exact integer >= 2")
        if world_size != expected_world_size:
            raise RuntimeError(
                f"Checkpoint load world_size={world_size}; expected {expected_world_size}"
            )
        validate_selection(selection)
        if checkpoint_root.name != selection.checkpoint_dir:
            raise ValueError("Checkpoint root and selection directory disagree")
        if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
            raise ValueError(f"Checkpoint root must be a real directory: {checkpoint_root}")
        _validate_deepspeed_layout(
            checkpoint_root,
            tag=selection.deepspeed_tag,
            world_size=world_size,
        )
        manifest = _load_checkpoint_manifest(checkpoint_root)
        if manifest["selection"] != selection.to_payload():
            raise RuntimeError("Checkpoint manifest selection does not match requested selection")
        if manifest["world_size"] != world_size:
            raise RuntimeError("Checkpoint manifest world size changed")
        if not callable(getattr(engine, "zero_optimization_stage", None)):
            raise TypeError("Fresh engine does not expose zero_optimization_stage")
        if int(engine.zero_optimization_stage()) != 3:
            raise RuntimeError("Fresh checkpoint engine must use ZeRO stage 3")
        if not callable(getattr(engine, "bfloat16_enabled", None)):
            raise TypeError("Fresh engine does not expose bfloat16_enabled")
        if engine.bfloat16_enabled() is not True:
            raise RuntimeError("Fresh checkpoint engine must use BF16")
        dp_world_size = getattr(engine, "dp_world_size", None)
        if callable(dp_world_size):
            dp_world_size = dp_world_size()
        if type(dp_world_size) is not int or dp_world_size != expected_world_size:
            raise RuntimeError("Fresh engine data-parallel world size changed")
        if getattr(engine, "optimizer", None) is None:
            raise RuntimeError("Fresh checkpoint engine has no optimizer")
        if getattr(engine, "lr_scheduler", None) is not None:
            raise RuntimeError("Fresh checkpoint engine unexpectedly owns a scheduler")
        fresh_global_steps = getattr(engine, "global_steps", None)
        if type(fresh_global_steps) is not int or fresh_global_steps != 0:
            raise RuntimeError(
                f"Checkpoint loading requires a pristine step-zero engine; got {fresh_global_steps}"
            )
        scheduler_state = torch.load(
            checkpoint_root / "scheduler.pt",
            map_location="cpu",
            weights_only=True,
        )
        scheduler_sha256 = canonical_state_sha256(scheduler_state)
        if scheduler_sha256 != manifest["scheduler_state_sha256"]:
            raise RuntimeError("External scheduler state digest changed")
        local_preflight.update(
            {"manifest": manifest, "scheduler_state": scheduler_state}
        )
        return {
            "checkpoint_root": str(checkpoint_root.resolve()),
            "manifest_sha256": sha256_file(checkpoint_root / "checkpoint_manifest.json"),
            "client_state_sha256": manifest["client_state_sha256"],
            "scheduler_state_sha256": scheduler_sha256,
        }

    preflight_statuses = _gather_local_status("Controlled checkpoint load preflight", preflight)
    preflight_values = [status["value"] for status in preflight_statuses]
    if preflight_values != [preflight_values[0]] * world_size:
        raise RuntimeError(f"Checkpoint load preflight differs across ranks: {preflight_values}")
    manifest = local_preflight["manifest"]
    scheduler_state = local_preflight["scheduler_state"]

    load_result: dict[str, Any] = {}

    def load_engine_and_scheduler() -> dict[str, Any]:
        load_path, loaded_client_state = engine.load_checkpoint(
            str(checkpoint_root),
            tag=selection.deepspeed_tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_only=False,
        )
        if load_path is None:
            raise RuntimeError("DeepSpeed returned no loaded checkpoint path")
        resolved_load_path = Path(load_path).resolve()
        expected_load_path = (
            checkpoint_root
            / selection.deepspeed_tag
            / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        ).resolve()
        if resolved_load_path != expected_load_path:
            raise RuntimeError(
                f"DeepSpeed loaded path {resolved_load_path}; expected {expected_load_path}"
            )
        exact_client_state = _extract_loaded_client_state(loaded_client_state)
        loaded_client_sha256 = hashlib.sha256(
            canonical_json(_jsonable_state(exact_client_state)).encode("utf-8")
        ).hexdigest()
        if loaded_client_sha256 != manifest["client_state_sha256"]:
            raise RuntimeError("Loaded DeepSpeed client state digest changed")
        loaded_global_steps = getattr(engine, "global_steps", None)
        if type(loaded_global_steps) is not int or loaded_global_steps != selection.global_step:
            raise RuntimeError("Loaded DeepSpeed global step does not match selection")
        if getattr(engine, "optimizer", None) is None:
            raise RuntimeError("Loaded DeepSpeed engine has no optimizer")
        scheduler.load_state_dict(scheduler_state)
        loaded_scheduler_sha256 = canonical_state_sha256(scheduler.state_dict())
        if loaded_scheduler_sha256 != manifest["scheduler_state_sha256"]:
            raise RuntimeError("Loaded external scheduler state digest changed")
        if getattr(scheduler, "last_epoch", None) != selection.global_step:
            raise RuntimeError("Loaded external scheduler step does not match selection")
        rng_sha256 = restore_rank_rng_state(checkpoint_root) if restore_rng else None
        values = {
            "load_path_parent": str(resolved_load_path.parent),
            "client_state_sha256": loaded_client_sha256,
            "scheduler_state_sha256": loaded_scheduler_sha256,
            "global_step": engine.global_steps,
            "rng_sha256": rng_sha256,
        }
        load_result.update(values)
        return values

    load_statuses = _gather_local_status(
        "Pristine DeepSpeed engine checkpoint load",
        load_engine_and_scheduler,
    )
    comparable = [
        {
            key: value
            for key, value in status["value"].items()
            if key not in {"load_path_parent", "rng_sha256"}
        }
        for status in load_statuses
    ]
    if comparable != [comparable[0]] * world_size:
        raise RuntimeError(f"Loaded checkpoint state differs across ranks: {comparable}")
    if not load_result:
        raise RuntimeError(f"Rank {rank} did not retain checkpoint load metadata")
    dist.barrier()
    return {
        **load_result,
        "manifest_sha256": preflight_values[0]["manifest_sha256"],
    }


class ValidationMetadataStore:
    def __init__(self, output_dir: Path, *, expected_epochs: int) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Validation metadata requires an initialized process group")

        def initialization_preflight() -> dict[str, Any]:
            if type(expected_epochs) is not int or expected_epochs < 1:
                raise ValueError("expected_epochs must be a positive exact int")
            if not isinstance(output_dir, Path):
                raise TypeError("Validation output root must be a Path")
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise ValueError(f"Validation output root must be a real directory: {output_dir}")
            return {
                "output_dir": str(output_dir.resolve()),
                "expected_epochs": expected_epochs,
            }

        initialization_statuses = _gather_local_status(
            "Validation metadata initialization preflight",
            initialization_preflight,
        )
        initialization_values = [status["value"] for status in initialization_statuses]
        if initialization_values != [initialization_values[0]] * dist.get_world_size():
            raise RuntimeError(
                "Validation metadata initialization differs across ranks: "
                f"{initialization_values}"
            )
        self.output_dir = output_dir
        self.expected_epochs = expected_epochs
        self.validation_dir = output_dir / "validation"
        self._records: list[dict[str, Any]] = []
        self._best: CheckpointSelection | None = None

        def create_validation_dir() -> dict[str, Any]:
            self.validation_dir.mkdir(exist_ok=False)
            fsync_directory(output_dir)
            return {"path": self.validation_dir.name}

        rank_zero_call("Validation metadata directory creation", create_validation_dir)
        if not self.validation_dir.is_dir() or self.validation_dir.is_symlink():
            raise RuntimeError("Validation metadata directory was not created safely")

    @property
    def best(self) -> CheckpointSelection | None:
        return self._best

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def register_checkpoint(
        self,
        *,
        candidate: CheckpointSelection,
        validation_result: Mapping[str, Any],
        checkpoint_metadata: Mapping[str, Any],
    ) -> tuple[CheckpointSelection, bool, dict[str, Any]]:
        prepared: dict[str, Any] = {}

        def registration_preflight() -> dict[str, Any]:
            validate_selection(candidate)
            expected_epoch = len(self._records) + 1
            if candidate.epoch != expected_epoch:
                raise RuntimeError(
                    f"Validation epoch={candidate.epoch}; expected sequential epoch={expected_epoch}"
                )
            best, is_new_best = choose_better_checkpoint(self._best, candidate)
            result_payload = dict(validation_result)
            metrics = result_payload.get("metrics")
            if type(metrics) is not dict:
                raise RuntimeError("Validation result is missing its metric mapping")
            if metrics.get(VALIDATION_PRIMARY_METRIC) != candidate.primary_metric:
                raise RuntimeError("Validation primary metric and checkpoint candidate disagree")
            if metrics.get(VALIDATION_SECONDARY_METRIC) != candidate.secondary_metric:
                raise RuntimeError("Validation secondary metric and checkpoint candidate disagree")
            if result_payload.get("ranking_sha256") != candidate.ranking_sha256:
                raise RuntimeError(
                    "Validation result and checkpoint candidate ranking digests disagree"
                )
            checkpoint_payload = dict(checkpoint_metadata)
            expected_checkpoint_keys = {
                "checkpoint_dir",
                "deepspeed_tag",
                "manifest_sha256",
                "scheduler_state_sha256",
                "client_state_sha256",
            }
            if set(checkpoint_payload) != expected_checkpoint_keys:
                raise RuntimeError("Checkpoint publication metadata schema changed")
            if checkpoint_payload["checkpoint_dir"] != candidate.checkpoint_dir:
                raise RuntimeError("Checkpoint publication directory and candidate disagree")
            if checkpoint_payload["deepspeed_tag"] != candidate.deepspeed_tag:
                raise RuntimeError("Checkpoint publication tag and candidate disagree")
            for name in (
                "manifest_sha256",
                "scheduler_state_sha256",
                "client_state_sha256",
            ):
                if not _is_sha256(checkpoint_payload[name]):
                    raise RuntimeError(f"Checkpoint publication {name} is invalid")
            identity = {
                "record_count": len(self._records),
                "current_best": None if self._best is None else self._best.to_payload(),
                "candidate": candidate.to_payload(),
                "best": best.to_payload(),
                "is_new_best": is_new_best,
                "validation_result": result_payload,
                "checkpoint_metadata": checkpoint_payload,
            }
            prepared.update(
                {
                    "best": best,
                    "is_new_best": is_new_best,
                    "result_payload": result_payload,
                    "checkpoint_payload": checkpoint_payload,
                }
            )
            return {
                "identity_sha256": hashlib.sha256(
                    canonical_json(identity).encode("utf-8")
                ).hexdigest()
            }

        registration_statuses = _gather_local_status(
            "Validation checkpoint registration preflight",
            registration_preflight,
        )
        registration_values = [status["value"] for status in registration_statuses]
        if registration_values != [registration_values[0]] * dist.get_world_size():
            raise RuntimeError(
                "Validation checkpoint registration differs across ranks: "
                f"{registration_values}"
            )
        best = prepared["best"]
        is_new_best = prepared["is_new_best"]
        result_payload = prepared["result_payload"]
        checkpoint_payload = prepared["checkpoint_payload"]
        record = {
            "schema_version": VALIDATION_HISTORY_SCHEMA_VERSION,
            "epoch": candidate.epoch,
            "global_step": candidate.global_step,
            "checkpoint": checkpoint_payload,
            "candidate": candidate.to_payload(),
            "is_new_best": is_new_best,
            "best_after_epoch": best.to_payload(),
            "validation_result": result_payload,
        }
        record_content = json.dumps(
            record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        record_sha256 = hashlib.sha256(record_content.encode("utf-8")).hexdigest()
        history_entry = {
            "epoch": candidate.epoch,
            "global_step": candidate.global_step,
            "path": f"epoch-{candidate.epoch:03d}.json",
            "sha256": record_sha256,
            "is_new_best": is_new_best,
            "candidate": candidate.to_payload(),
            "best_after_epoch": best.to_payload(),
        }

        def publish_metadata() -> dict[str, Any]:
            checkpoint_root = self.output_dir / candidate.checkpoint_dir
            checkpoint_manifest = _load_checkpoint_manifest(checkpoint_root)
            if (
                sha256_file(checkpoint_root / "checkpoint_manifest.json")
                != checkpoint_payload["manifest_sha256"]
            ):
                raise RuntimeError("Checkpoint publication manifest digest changed")
            if (
                checkpoint_manifest["scheduler_state_sha256"]
                != checkpoint_payload["scheduler_state_sha256"]
                or checkpoint_manifest["client_state_sha256"]
                != checkpoint_payload["client_state_sha256"]
            ):
                raise RuntimeError(
                    "Checkpoint publication state digests disagree with manifest"
                )
            epoch_path = self.validation_dir / history_entry["path"]
            publish_new_text(epoch_path, record_content)
            if sha256_file(epoch_path) != record_sha256:
                raise RuntimeError("Published validation epoch digest changed")
            complete_history = [*self._records, history_entry]
            history_payload = {
                "schema_version": VALIDATION_HISTORY_SCHEMA_VERSION,
                "records": complete_history,
            }
            replace_text_atomically(
                self.validation_dir / "history.json",
                json.dumps(
                    history_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
            )
            replace_text_atomically(
                self.validation_dir / "latest.json",
                json.dumps(
                    history_entry,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
            )
            if is_new_best:
                replace_text_atomically(
                    self.validation_dir / "best.json",
                    json.dumps(
                        history_entry,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                )
            return history_entry

        published_entry = rank_zero_call("Validation epoch metadata publication", publish_metadata)
        if published_entry != history_entry:
            raise RuntimeError("Validation metadata broadcast changed the epoch record")
        self._records.append(history_entry)
        self._best = best
        return best, is_new_best, dict(history_entry)

    def finalize(self, *, retained_checkpoint_dirs: Sequence[str]) -> dict[str, Any]:
        def finalization_preflight() -> dict[str, Any]:
            if len(self._records) != self.expected_epochs or self._best is None:
                raise RuntimeError(
                    f"Validation history has {len(self._records)} epochs; "
                    f"expected {self.expected_epochs}"
                )
            retained = _validate_checkpoint_dir_names(retained_checkpoint_dirs)
            expected_retained = _validate_checkpoint_dir_names(
                [self._best.checkpoint_dir, self._records[-1]["candidate"]["checkpoint_dir"]]
            )
            if retained != expected_retained:
                raise RuntimeError(
                    f"Retained checkpoints={retained}; expected best/last={expected_retained}"
                )
            identity = {
                "records": self._records,
                "best": self._best.to_payload(),
                "retained": list(retained),
            }
            return {
                "identity_sha256": hashlib.sha256(
                    canonical_json(identity).encode("utf-8")
                ).hexdigest()
            }

        finalization_statuses = _gather_local_status(
            "Validation metadata finalization preflight",
            finalization_preflight,
        )
        finalization_values = [status["value"] for status in finalization_statuses]
        if finalization_values != [finalization_values[0]] * dist.get_world_size():
            raise RuntimeError(
                "Validation metadata finalization differs across ranks: "
                f"{finalization_values}"
            )
        if len(self._records) != self.expected_epochs or self._best is None:
            raise RuntimeError(
                f"Validation history has {len(self._records)} epochs; "
                f"expected {self.expected_epochs}"
            )
        retained = _validate_checkpoint_dir_names(retained_checkpoint_dirs)
        expected_retained = _validate_checkpoint_dir_names(
            [self._best.checkpoint_dir, self._records[-1]["candidate"]["checkpoint_dir"]]
        )
        if retained != expected_retained:
            raise RuntimeError(
                f"Retained checkpoints={retained}; expected best/last={expected_retained}"
            )

        def publish_manifest() -> dict[str, Any]:
            expected_epoch_files = [
                f"epoch-{epoch:03d}.json" for epoch in range(1, self.expected_epochs + 1)
            ]
            actual_epoch_files = sorted(path.name for path in self.validation_dir.glob("epoch-*.json"))
            if actual_epoch_files != expected_epoch_files:
                raise RuntimeError("Validation epoch artifact inventory is incomplete")
            for record in self._records:
                path = self.validation_dir / record["path"]
                if sha256_file(path) != record["sha256"]:
                    raise RuntimeError(f"Validation epoch artifact hash changed: {path.name}")
            manifest = {
                "schema_version": VALIDATION_HISTORY_SCHEMA_VERSION,
                "epochs": self.expected_epochs,
                "selection_order": [
                    "maximize validation case-macro set recall@20",
                    "maximize validation case-macro full-ranking first-gold reciprocal rank",
                    "minimize epoch number",
                ],
                "best": self._best.to_payload(),
                "last": self._records[-1]["candidate"],
                "retained_checkpoint_dirs": list(retained),
                "records": list(self._records),
                "history_sha256": sha256_file(self.validation_dir / "history.json"),
                "best_sha256": sha256_file(self.validation_dir / "best.json"),
                "latest_sha256": sha256_file(self.validation_dir / "latest.json"),
            }
            publish_new_text(
                self.validation_dir / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            return manifest

        manifest = rank_zero_call("Validation manifest publication", publish_manifest)
        if type(manifest) is not dict or manifest.get("epochs") != self.expected_epochs:
            raise RuntimeError("Validation manifest broadcast is malformed")
        return dict(manifest)


def _validate_checkpoint_dir_names(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("retained checkpoint directories must be a non-empty list or tuple")
    names: list[str] = []
    for value in values:
        if type(value) is not str or not value.startswith("checkpoint-"):
            raise ValueError(f"Invalid checkpoint directory name: {value!r}")
        suffix = value.removeprefix("checkpoint-")
        if not suffix.isdigit() or int(suffix) < 1 or value != f"checkpoint-{int(suffix)}":
            raise ValueError(f"Invalid checkpoint directory name: {value!r}")
        names.append(value)
    return tuple(sorted(set(names), key=lambda name: int(name.removeprefix("checkpoint-"))))


def retain_best_and_last_checkpoints(
    output_dir: Path,
    *,
    best_checkpoint_dir: str,
    last_checkpoint_dir: str,
) -> tuple[str, ...]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Checkpoint retention requires an initialized process group")
    world_size = dist.get_world_size()

    def retention_preflight() -> dict[str, Any]:
        if not isinstance(output_dir, Path):
            raise TypeError("Checkpoint retention output_dir must be a Path")
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(
                f"Checkpoint retention output_dir must be a real directory: {output_dir}"
            )
        keep = _validate_checkpoint_dir_names(
            [best_checkpoint_dir, last_checkpoint_dir]
        )
        return {
            "output_dir": str(output_dir.resolve()),
            "keep": list(keep),
        }

    preflight_statuses = _gather_local_status(
        "Checkpoint retention preflight",
        retention_preflight,
    )
    preflight_values = [status["value"] for status in preflight_statuses]
    if preflight_values != [preflight_values[0]] * world_size:
        raise RuntimeError(
            f"Checkpoint retention preflight differs across ranks: {preflight_values}"
        )
    keep = tuple(preflight_values[0]["keep"])
    dist.barrier()

    def delete_unretained() -> list[str]:
        incomplete = sorted(path.name for path in output_dir.glob(".checkpoint-*.incomplete"))
        if incomplete:
            raise RuntimeError(f"Incomplete checkpoint directories remain: {incomplete}")
        checkpoint_paths = list(output_dir.glob("checkpoint-*"))
        if any(path.is_symlink() or not path.is_dir() for path in checkpoint_paths):
            raise RuntimeError("Every checkpoint entry must be a real non-symlink directory")
        _validate_checkpoint_dir_names([path.name for path in checkpoint_paths])
        checkpoint_paths.sort(key=lambda path: int(path.name.removeprefix("checkpoint-")))
        actual_names = {path.name for path in checkpoint_paths}
        if not set(keep).issubset(actual_names):
            raise RuntimeError(
                f"Required retained checkpoints are missing: actual={sorted(actual_names)}, "
                f"required={list(keep)}"
            )
        for required_name in keep:
            required_path = output_dir / required_name
            manifest = _load_checkpoint_manifest(required_path)
            selection = manifest.get("selection")
            if (
                type(selection) is not dict
                or selection.get("checkpoint_dir") != required_name
            ):
                raise RuntimeError(
                    f"Retained checkpoint manifest identity changed: {required_name}"
                )
            _validate_deepspeed_layout(
                required_path,
                tag=selection.get("deepspeed_tag"),
                world_size=manifest.get("world_size"),
            )
        for path in checkpoint_paths:
            if path.name not in keep:
                shutil.rmtree(path)
                if path.exists() or path.is_symlink():
                    raise RuntimeError(f"Checkpoint deletion did not remove {path}")
        remaining = sorted(
            path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
        )
        if set(remaining) != set(keep):
            raise RuntimeError(
                f"Checkpoint retention mismatch: remaining={remaining}, expected={list(keep)}"
            )
        fsync_directory(output_dir)
        return remaining

    remaining = rank_zero_call("Best/last checkpoint retention", delete_unretained)
    if set(remaining) != set(keep):
        raise RuntimeError("Checkpoint retention broadcast changed the retained set")
    dist.barrier()
    return keep


def retained_checkpoint_inventory(
    output_dir: Path,
    retained_checkpoint_dirs: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(output_dir, Path):
        raise TypeError("Retained checkpoint output_dir must be a Path")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError(
            f"Retained checkpoint output_dir must be a real directory: {output_dir}"
        )
    names = _validate_checkpoint_dir_names(retained_checkpoint_dirs)
    checkpoints: list[dict[str, Any]] = []
    for name in names:
        path = output_dir / name
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"Retained checkpoint is missing or unsafe: {path}")
        _load_checkpoint_manifest(path)
        files = _tree_inventory(path, include_hashes=True)
        checkpoints.append({"path": name, "files": files})
    return {
        "schema_version": CHECKPOINT_PROTOCOL_SCHEMA_VERSION,
        "checkpoints": checkpoints,
    }
