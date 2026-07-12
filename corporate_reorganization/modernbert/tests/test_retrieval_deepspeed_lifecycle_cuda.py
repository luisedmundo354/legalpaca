from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.provenance import (  # noqa: E402
    EXPECTED_RUNTIME_VERSIONS,
    validate_runtime_versions,
)


WORLD_SIZE = 2
WORKER_TIMEOUT_SECONDS = 300
PINNED_RUNTIME = EXPECTED_RUNTIME_VERSIONS


def _runtime_skip_reason() -> str | None:
    try:
        validate_runtime_versions()
    except RuntimeError as error:
        return str(error)

    import deepspeed
    import torch

    if deepspeed.__version__ != PINNED_RUNTIME["deepspeed"]:
        raise RuntimeError(f"DeepSpeed module version changed: {deepspeed.__version__}")

    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.device_count() < WORLD_SIZE:
        return f"only {torch.cuda.device_count()} CUDA device(s) are visible"
    if not torch.distributed.is_available():
        return "torch.distributed is unavailable"
    if not torch.distributed.is_nccl_available():
        return "the PyTorch build has no NCCL backend"
    for device_index in range(WORLD_SIZE):
        with torch.cuda.device(device_index):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                return f"CUDA device {device_index} lacks native BF16 support"
    return None


def _deep_copy_to_cpu(value: Any) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _deep_copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy_to_cpu(item) for item in value)
    return value


def _assert_nested_exact(actual: Any, expected: Any, *, path: str) -> None:
    import numpy as np
    import torch

    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise AssertionError(f"{path}: expected tensor, got {type(actual).__name__}")
        if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
            raise AssertionError(
                f"{path}: tensor metadata changed from "
                f"{expected.dtype}/{tuple(expected.shape)} to "
                f"{actual.dtype}/{tuple(actual.shape)}"
            )
        if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise AssertionError(f"{path}: tensor values changed")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            raise AssertionError(f"{path}: expected ndarray, got {type(actual).__name__}")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"{path}: NumPy metadata changed")
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{path}: NumPy values changed")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            raise AssertionError(
                f"{path}: mapping keys/order changed; "
                f"actual={list(actual) if isinstance(actual, dict) else type(actual).__name__}, "
                f"expected={list(expected)}"
            )
        for key in expected:
            _assert_nested_exact(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence type or length changed")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_exact(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise AssertionError(
            f"{path}: value changed from {expected!r} ({type(expected).__name__}) "
            f"to {actual!r} ({type(actual).__name__})"
        )


def _deepspeed_config() -> dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": WORLD_SIZE,
        "gradient_clipping": 1.0,
        "steps_per_print": 2**31 - 1,
        "wall_clock_breakdown": False,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "stage3_param_persistence_threshold": 0,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_tiny_model():
    import torch

    class TinyStochasticRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(8, 16, bias=True)
            self.dropout = torch.nn.Dropout(p=0.25)
            self.output_projection = torch.nn.Linear(16, 4, bias=True)

        def forward(self, inputs):
            hidden = self.input_projection(inputs)
            hidden = hidden * hidden
            return self.output_projection(self.dropout(hidden))

    return TinyStochasticRegressor()


def _prepare_engine(accelerator, *, model_seed: int):
    import torch

    _seed_everything(model_seed)
    model = _make_tiny_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.1 * step,
    )
    if scheduler.last_epoch != 0:
        raise AssertionError(
            f"Pinned PyTorch scheduler starts at last_epoch={scheduler.last_epoch}; expected 0"
        )

    engine, prepared_optimizer = accelerator.prepare(model, optimizer)
    if engine.module is not model:
        raise AssertionError("Accelerate did not preserve the pristine model as Engine.module")
    if int(engine.zero_optimization_stage()) != 3:
        raise AssertionError("The prepared engine is not ZeRO stage 3")
    if engine.bfloat16_enabled() is not True:
        raise AssertionError("The prepared engine is not BF16")
    dp_world_size = engine.dp_world_size
    if callable(dp_world_size):
        dp_world_size = dp_world_size()
    if dp_world_size != WORLD_SIZE:
        raise AssertionError(
            f"The prepared data-parallel world size is {dp_world_size}; expected {WORLD_SIZE}"
        )
    if engine.lr_scheduler is not None:
        raise AssertionError("DeepSpeed unexpectedly owns the external scheduler")
    if engine.global_steps != 0:
        raise AssertionError("A freshly prepared engine did not start at global step zero")
    if prepared_optimizer.optimizer is not engine.optimizer:
        raise AssertionError("Accelerate's optimizer wrapper does not reference Engine.optimizer")
    if engine.optimizer.optimizer is not optimizer:
        raise AssertionError(
            "The external scheduler's optimizer is not the DeepSpeed base optimizer"
        )
    if scheduler.optimizer is not optimizer:
        raise AssertionError("The external scheduler lost its exact base optimizer")
    return engine, model, prepared_optimizer, optimizer, scheduler


def _release_engine(
    accelerator,
    *,
    engine,
    model,
    prepared_optimizer,
    base_optimizer,
    scheduler,
) -> list[None]:
    # The production Trainer fixes world size at four. This two-rank gate exercises
    # the same Accelerator.free_memory -> DeepSpeedEngine.destroy path directly.
    accelerator.wait_for_everyone()
    released = accelerator.free_memory(
        engine,
        model,
        prepared_optimizer,
        base_optimizer,
        scheduler,
    )
    if released != [None, None, None, None, None]:
        raise AssertionError(f"Accelerate release returned live objects: {released}")
    if (
        accelerator.deepspeed_engine_wrapped is not None
        or accelerator._models
        or accelerator._optimizers
        or accelerator._schedulers
        or accelerator._dataloaders
    ):
        raise AssertionError("Accelerate retained Engine/model/optimizer/scheduler objects")
    return released


def _finish_release(accelerator) -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


def _fixed_batch(*, rank: int, step_index: int, device):
    import torch

    start = 1 + rank * 11 + step_index * 7
    inputs = torch.arange(start, start + 8, dtype=torch.float32).reshape(1, 8)
    inputs = (inputs / 17.0).to(device=device, dtype=torch.bfloat16)
    targets = torch.tensor(
        [[
            0.15 + rank * 0.05 + step_index * 0.01,
            -0.20 + rank * 0.03 - step_index * 0.02,
            0.30 - rank * 0.04 + step_index * 0.03,
            -0.10 - rank * 0.02 - step_index * 0.01,
        ]],
        dtype=torch.float32,
        device=device,
    )
    return inputs, targets


def _train_one_step(engine, scheduler, *, rank: int, step_index: int) -> float:
    import torch

    expected_before = step_index
    if engine.global_steps != expected_before or scheduler.last_epoch != expected_before:
        raise AssertionError(
            f"Step {step_index} started from engine/scheduler "
            f"{engine.global_steps}/{scheduler.last_epoch}; expected {expected_before}"
        )
    engine.train()
    engine.zero_grad()
    inputs, targets = _fixed_batch(
        rank=rank,
        step_index=step_index,
        device=engine.device,
    )
    predictions = engine(inputs)
    loss = torch.square(predictions.float() - targets).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Step {step_index} produced non-finite loss")
    loss_value = float(loss.detach().cpu().item())
    engine.backward(loss)
    engine.step()
    scheduler.step()
    expected_after = step_index + 1
    if engine.global_steps != expected_after or scheduler.last_epoch != expected_after:
        raise AssertionError(
            f"Step {step_index} ended at engine/scheduler "
            f"{engine.global_steps}/{scheduler.last_epoch}; expected {expected_after}"
        )
    if engine.lr_scheduler is not None:
        raise AssertionError("DeepSpeed took ownership of the external scheduler during training")
    return loss_value


def _rng_probe(*, device) -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "cpu": torch.rand(5, dtype=torch.float32),
        "cuda": torch.rand(5, dtype=torch.float32, device=device).cpu(),
    }


def _perturb_every_rng(*, device) -> None:
    import numpy as np
    import torch

    for _ in range(17):
        random.random()
        np.random.random()
    torch.rand(37, dtype=torch.float32)
    torch.rand(41, dtype=torch.float32, device=device)


class _TinyTrainerState:
    def __init__(self, *, epoch: int, global_step: int) -> None:
        self.epoch = float(epoch)
        self.global_step = global_step

    def save_to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(
                {"epoch": self.epoch, "global_step": self.global_step},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _zero_optimizer_state(engine) -> dict[str, Any]:
    state = engine.optimizer.state_dict()
    expected_keys = {
        "zero_stage",
        "loss_scaler",
        "dynamic_loss_scale",
        "overflow",
        "partition_count",
        "optimizer_state_dict",
        "fp32_flat_groups",
    }
    if not isinstance(state, dict) or set(state) != expected_keys:
        raise AssertionError(
            "Pinned ZeRO-3 optimizer state schema changed: "
            f"actual={sorted(state) if isinstance(state, dict) else type(state).__name__}, "
            f"expected={sorted(expected_keys)}"
        )
    loss_scaler = state["loss_scaler"]
    if not hasattr(loss_scaler, "__dict__"):
        raise AssertionError("ZeRO-3 loss scaler has no serializable attribute state")
    normalized = dict(state)
    normalized["loss_scaler"] = {
        "class": f"{type(loss_scaler).__module__}.{type(loss_scaler).__qualname__}",
        "attributes": dict(vars(loss_scaler)),
    }
    return _deep_copy_to_cpu(normalized)


def _assert_full_state_dict(state_dict: Any, *, context: str) -> None:
    import torch

    expected_keys = list(_make_tiny_model().state_dict())
    if not isinstance(state_dict, dict) or list(state_dict) != expected_keys:
        actual_keys = (
            list(state_dict)
            if isinstance(state_dict, dict)
            else type(state_dict).__name__
        )
        raise AssertionError(
            f"{context}: consolidated state keys changed; "
            f"actual={actual_keys}, expected={expected_keys}"
        )
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise AssertionError(f"{context}.{name}: expected tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
            raise AssertionError(
                f"{context}.{name}: expected CPU BF16, got {tensor.device}/{tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{context}.{name}: non-finite consolidated weight")


def _fp32_zero_consolidation_diagnostic(
    *,
    checkpoint_root: Path,
    deepspeed_tag: str,
    expected_bf16_state: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from deepspeed.utils.zero_to_fp32 import (
        get_fp32_state_dict_from_zero_checkpoint,
    )

    fp32_state = get_fp32_state_dict_from_zero_checkpoint(
        str(checkpoint_root),
        tag=deepspeed_tag,
        exclude_frozen_parameters=False,
        lazy_mode=False,
    )
    expected_keys = list(_make_tiny_model().state_dict())
    if not isinstance(fp32_state, dict) or list(fp32_state) != expected_keys:
        actual_keys = (
            list(fp32_state)
            if isinstance(fp32_state, dict)
            else type(fp32_state).__name__
        )
        raise AssertionError(
            "Explicit-tag FP32 ZeRO consolidation changed model keys: "
            f"actual={actual_keys}, expected={expected_keys}"
        )
    digest = hashlib.sha256()
    for name, tensor in fp32_state.items():
        if not torch.is_tensor(tensor):
            raise AssertionError(f"fp32_zero.{name}: expected tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise AssertionError(
                f"fp32_zero.{name}: expected CPU FP32, got {tensor.device}/{tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"fp32_zero.{name}: non-finite tensor")
        contiguous = tensor.detach().contiguous()
        descriptor = json.dumps(
            {"name": name, "shape": list(contiguous.shape), "dtype": str(contiguous.dtype)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, byteorder="big"))
        digest.update(descriptor)
        digest.update(contiguous.numpy().tobytes(order="C"))

    fp32_model = _make_tiny_model().to(device="cpu", dtype=torch.float32)
    incompatible = fp32_model.load_state_dict(fp32_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Strict FP32 consolidation load changed keys: {incompatible}")
    projected_bf16 = {
        name: tensor.detach().to(device="cpu", dtype=torch.bfloat16)
        for name, tensor in fp32_model.state_dict().items()
    }
    _assert_nested_exact(
        projected_bf16,
        expected_bf16_state,
        path="fp32_zero_bf16_projection",
    )
    return {
        "checkpoint_dir": checkpoint_root.name,
        "deepspeed_tag": deepspeed_tag,
        "exclude_frozen_parameters": False,
        "lazy_mode": False,
        "tensor_count": len(fp32_state),
        "state_sha256": digest.hexdigest(),
    }


def _strict_safetensors_round_trip(state_dict: dict[str, Any], artifact_dir: Path) -> str:
    import torch
    from safetensors.torch import load_model, save_model

    export_model = _make_tiny_model().to(device="cpu", dtype=torch.bfloat16)
    incompatible = export_model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Strict export-model load changed keys: {incompatible}")
    model_path = artifact_dir / "roundtrip-model.safetensors"
    save_model(
        export_model,
        str(model_path),
        metadata={"format": "pt", "test_contract": "zero3-bf16-lifecycle-v1"},
    )
    if not model_path.is_file() or model_path.stat().st_size < 1:
        raise AssertionError("Safetensors export was not written")

    reloaded_model = _make_tiny_model().to(device="cpu", dtype=torch.bfloat16)
    missing, unexpected = load_model(
        reloaded_model,
        model_path,
        strict=True,
        device="cpu",
    )
    if missing or unexpected:
        raise AssertionError(
            f"Strict safetensors reload returned missing={missing}, unexpected={unexpected}"
        )
    _assert_nested_exact(
        reloaded_model.state_dict(),
        state_dict,
        path="safetensors_round_trip",
    )
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if len(digest) != 64:
        raise AssertionError("Safetensors SHA-256 is malformed")
    return digest


def _gpu_worker(artifact_dir: Path) -> None:
    import deepspeed
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DistributedType, InitProcessGroupKwargs

    import retriever.checkpointing as checkpointing

    validate_runtime_versions()
    if deepspeed.__version__ != PINNED_RUNTIME["deepspeed"]:
        raise RuntimeError(f"DeepSpeed module version changed: {deepspeed.__version__}")
    if torch.cuda.device_count() < WORLD_SIZE or not dist.is_nccl_available():
        raise RuntimeError("GPU worker was launched without two CUDA/NCCL devices")

    rank_from_environment = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_from_environment = int(os.environ["WORLD_SIZE"])
    if world_from_environment != WORLD_SIZE or local_rank not in range(WORLD_SIZE):
        raise RuntimeError(
            f"torchrun topology changed: world={world_from_environment}, local_rank={local_rank}"
        )
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"Worker device cuda:{local_rank} has no native BF16 support")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    plugin = DeepSpeedPlugin(
        hf_ds_config=_deepspeed_config(),
        zero3_init_flag=False,
        zero3_save_16bit_model=True,
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        deepspeed_plugin=plugin,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=120))],
    )
    if accelerator.distributed_type is not DistributedType.DEEPSPEED:
        raise RuntimeError(
            f"Accelerate selected {accelerator.distributed_type}; expected DeepSpeed"
        )
    if accelerator.num_processes != WORLD_SIZE:
        raise RuntimeError(
            f"Accelerate world size={accelerator.num_processes}; expected {WORLD_SIZE}"
        )
    if accelerator.process_index != rank_from_environment:
        raise RuntimeError("Accelerate rank differs from torchrun RANK")
    if accelerator.local_process_index != local_rank:
        raise RuntimeError("Accelerate local rank differs from torchrun LOCAL_RANK")
    if dist.get_backend() != "nccl":
        raise RuntimeError(f"Process group backend={dist.get_backend()}; expected nccl")

    rank = accelerator.process_index
    artifact_dir = artifact_dir.resolve()
    if rank == 0:
        artifact_dir.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise RuntimeError("Shared lifecycle artifact directory was not created safely")

    model_seed = 73013
    training_seed = 88101 + rank

    # Reference: two uninterrupted optimizer updates.
    (
        reference_engine,
        reference_model,
        reference_prepared_optimizer,
        reference_base_optimizer,
        reference_scheduler,
    ) = _prepare_engine(accelerator, model_seed=model_seed)
    _seed_everything(training_seed)
    reference_first_loss = _train_one_step(
        reference_engine,
        reference_scheduler,
        rank=rank,
        step_index=0,
    )
    reference_rng_after_step_one = _deep_copy_to_cpu(
        checkpointing.capture_rng_state()
    )
    reference_probe = _rng_probe(device=reference_engine.device)
    reference_second_loss = _train_one_step(
        reference_engine,
        reference_scheduler,
        rank=rank,
        step_index=1,
    )
    reference_rng_after_step_two = _deep_copy_to_cpu(
        checkpointing.capture_rng_state()
    )
    reference_optimizer_state = _zero_optimizer_state(reference_engine)
    reference_scheduler_state = _deep_copy_to_cpu(reference_scheduler.state_dict())
    reference_weights = accelerator.get_state_dict(reference_engine)
    if rank == 0:
        _assert_full_state_dict(reference_weights, context="uninterrupted")
        reference_weights = _deep_copy_to_cpu(reference_weights)
    elif reference_weights is not None:
        raise AssertionError("Nonzero rank received a consolidated reference state dict")
    (
        reference_engine,
        reference_model,
        reference_prepared_optimizer,
        reference_base_optimizer,
        reference_scheduler,
    ) = _release_engine(
        accelerator,
        engine=reference_engine,
        model=reference_model,
        prepared_optimizer=reference_prepared_optimizer,
        base_optimizer=reference_base_optimizer,
        scheduler=reference_scheduler,
    )
    _finish_release(accelerator)

    # Engine A: repeat the first update and publish an exact controlled checkpoint.
    (
        engine_a,
        model_a,
        prepared_optimizer_a,
        base_optimizer_a,
        scheduler_a,
    ) = _prepare_engine(accelerator, model_seed=model_seed)
    _seed_everything(training_seed)
    resumed_first_loss = _train_one_step(
        engine_a,
        scheduler_a,
        rank=rank,
        step_index=0,
    )
    if resumed_first_loss != reference_first_loss:
        raise AssertionError(
            f"Rank {rank} first loss changed: {resumed_first_loss} != {reference_first_loss}"
        )

    selection = checkpointing.CheckpointSelection(
        schema_version=checkpointing.SELECTION_METADATA_SCHEMA_VERSION,
        epoch=1,
        global_step=1,
        checkpoint_dir="checkpoint-1",
        deepspeed_tag="global_step1",
        primary_metric=0.0,
        secondary_metric=0.0,
        ranking_sha256=hashlib.sha256(b"zero3-bf16-lifecycle-ranking").hexdigest(),
    )
    client_state = {
        "controlled_state": {
            "schema_version": 1,
            "epoch": 1,
            "global_step": 1,
            "test_contract": "two-process-zero3-bf16-lifecycle-v1",
        }
    }
    checkpoint_metadata = checkpointing.save_controlled_checkpoint(
        output_dir=artifact_dir,
        engine=engine_a,
        scheduler=scheduler_a,
        trainer_state=_TinyTrainerState(epoch=1, global_step=1),
        training_args={"test_contract": "two-process-zero3-bf16-lifecycle-v1"},
        selection=selection,
        client_state=client_state,
        expected_world_size=WORLD_SIZE,
    )
    if checkpoint_metadata["checkpoint_dir"] != selection.checkpoint_dir:
        raise AssertionError("Controlled save returned the wrong checkpoint directory")
    engine_a_step_one_weights = accelerator.get_state_dict(engine_a)
    if rank == 0:
        _assert_full_state_dict(engine_a_step_one_weights, context="engine_a_step_one")
    elif engine_a_step_one_weights is not None:
        raise AssertionError("Nonzero rank received Engine A's consolidated state dict")
    fp32_diagnostic = checkpointing.rank_zero_call(
        "Explicit-tag FP32 ZeRO consolidation diagnostic",
        lambda: _fp32_zero_consolidation_diagnostic(
            checkpoint_root=artifact_dir / selection.checkpoint_dir,
            deepspeed_tag=selection.deepspeed_tag,
            expected_bf16_state=engine_a_step_one_weights,
        ),
    )
    if (
        not isinstance(fp32_diagnostic, dict)
        or fp32_diagnostic.get("deepspeed_tag") != selection.deepspeed_tag
        or fp32_diagnostic.get("lazy_mode") is not False
        or fp32_diagnostic.get("exclude_frozen_parameters") is not False
    ):
        raise AssertionError(
            f"FP32 ZeRO consolidation metadata is malformed: {fp32_diagnostic}"
        )
    _perturb_every_rng(device=engine_a.device)
    (
        engine_a,
        model_a,
        prepared_optimizer_a,
        base_optimizer_a,
        scheduler_a,
    ) = _release_engine(
        accelerator,
        engine=engine_a,
        model=model_a,
        prepared_optimizer=prepared_optimizer_a,
        base_optimizer=base_optimizer_a,
        scheduler=scheduler_a,
    )
    _finish_release(accelerator)

    # Engine B: construct from scratch, full-load the explicit tag, and resume.
    (
        engine_b,
        model_b,
        prepared_optimizer_b,
        base_optimizer_b,
        scheduler_b,
    ) = _prepare_engine(accelerator, model_seed=model_seed + 999)
    load_metadata = checkpointing.load_controlled_checkpoint(
        checkpoint_root=artifact_dir / selection.checkpoint_dir,
        engine=engine_b,
        scheduler=scheduler_b,
        selection=selection,
        expected_world_size=WORLD_SIZE,
        restore_rng=True,
    )
    if load_metadata["global_step"] != 1 or engine_b.global_steps != 1:
        raise AssertionError("Strict explicit-tag load did not restore global step one")
    if scheduler_b.last_epoch != 1:
        raise AssertionError("Strict explicit-tag load did not restore the external scheduler")
    for digest_name in (
        "manifest_sha256",
        "scheduler_state_sha256",
        "client_state_sha256",
    ):
        if load_metadata[digest_name] != checkpoint_metadata[digest_name]:
            raise AssertionError(
                f"Strict explicit-tag load changed {digest_name}: "
                f"{load_metadata[digest_name]} != {checkpoint_metadata[digest_name]}"
            )
    if not load_metadata.get("rng_sha256"):
        raise AssertionError("Strict explicit-tag load did not report rank-local RNG restoration")

    resumed_rng_after_load = _deep_copy_to_cpu(checkpointing.capture_rng_state())
    _assert_nested_exact(
        resumed_rng_after_load,
        reference_rng_after_step_one,
        path=f"rank_{rank}.rng_after_load",
    )
    resumed_probe = _rng_probe(device=engine_b.device)
    _assert_nested_exact(resumed_probe, reference_probe, path=f"rank_{rank}.rng_probe")
    resumed_second_loss = _train_one_step(
        engine_b,
        scheduler_b,
        rank=rank,
        step_index=1,
    )
    resumed_rng_after_step_two = _deep_copy_to_cpu(checkpointing.capture_rng_state())
    _assert_nested_exact(
        resumed_rng_after_step_two,
        reference_rng_after_step_two,
        path=f"rank_{rank}.rng_after_step_two",
    )
    if resumed_second_loss != reference_second_loss:
        raise AssertionError(
            f"Rank {rank} second loss changed: {resumed_second_loss} != {reference_second_loss}"
        )

    resumed_optimizer_state = _zero_optimizer_state(engine_b)
    resumed_scheduler_state = _deep_copy_to_cpu(scheduler_b.state_dict())
    _assert_nested_exact(
        resumed_optimizer_state,
        reference_optimizer_state,
        path=f"rank_{rank}.optimizer",
    )
    _assert_nested_exact(
        resumed_scheduler_state,
        reference_scheduler_state,
        path=f"rank_{rank}.scheduler",
    )
    resumed_weights = accelerator.get_state_dict(engine_b)
    safetensors_sha256 = None
    if rank == 0:
        _assert_full_state_dict(resumed_weights, context="resumed")
        _assert_nested_exact(
            resumed_weights,
            reference_weights,
            path="consolidated_weights",
        )
        safetensors_sha256 = _strict_safetensors_round_trip(
            resumed_weights,
            artifact_dir,
        )
    elif resumed_weights is not None:
        raise AssertionError("Nonzero rank received a consolidated resumed state dict")
    accelerator.wait_for_everyone()

    (
        engine_b,
        model_b,
        prepared_optimizer_b,
        base_optimizer_b,
        scheduler_b,
    ) = _release_engine(
        accelerator,
        engine=engine_b,
        model=model_b,
        prepared_optimizer=prepared_optimizer_b,
        base_optimizer=base_optimizer_b,
        scheduler=scheduler_b,
    )
    _finish_release(accelerator)
    if rank == 0:
        success = {
            "schema_version": 1,
            "world_size": WORLD_SIZE,
            "backend": "nccl",
            "zero_stage": 3,
            "dtype": "torch.bfloat16",
            "reference_steps": 2,
            "resumed_steps": 2,
            "checkpoint_dir": selection.checkpoint_dir,
            "deepspeed_tag": selection.deepspeed_tag,
            "fp32_consolidation_sha256": fp32_diagnostic["state_sha256"],
            "safetensors_sha256": safetensors_sha256,
        }
        (artifact_dir / "success.json").write_text(
            json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


class TwoProcessDeepSpeedLifecycleCudaTest(unittest.TestCase):
    def test_exact_zero3_bf16_checkpoint_destroy_reload_and_export(self) -> None:
        reason = _runtime_skip_reason()
        if reason is not None:
            self.skipTest(reason)

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = Path(temporary_directory) / "lifecycle-artifacts"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={WORLD_SIZE}",
                "--max-restarts=0",
                str(Path(__file__).resolve()),
                "--gpu-worker",
                str(artifact_dir),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "NCCL_ALGO": "Ring",
                    "NCCL_PROTO": "Simple",
                    "OMP_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "73013",
                    "PYTHONUNBUFFERED": "1",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                }
            )
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                output, _ = process.communicate(timeout=WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output, _ = process.communicate()
                self.fail(
                    f"Two-process DeepSpeed lifecycle exceeded "
                    f"{WORKER_TIMEOUT_SECONDS}s. Output:\n{output[-30000:]}"
                )
            self.assertEqual(
                process.returncode,
                0,
                msg=(
                    "Two-process DeepSpeed lifecycle worker failed. "
                    f"Output:\n{output[-30000:]}"
                ),
            )
            success_path = artifact_dir / "success.json"
            self.assertTrue(success_path.is_file(), "GPU workers produced no success marker")
            success = json.loads(success_path.read_text(encoding="utf-8"))
            self.assertEqual(
                success,
                {
                    "schema_version": 1,
                    "world_size": WORLD_SIZE,
                    "backend": "nccl",
                    "zero_stage": 3,
                    "dtype": "torch.bfloat16",
                    "reference_steps": 2,
                    "resumed_steps": 2,
                    "checkpoint_dir": "checkpoint-1",
                    "deepspeed_tag": "global_step1",
                    "fp32_consolidation_sha256": success[
                        "fp32_consolidation_sha256"
                    ],
                    "safetensors_sha256": success["safetensors_sha256"],
                },
            )
            self.assertRegex(
                success["fp32_consolidation_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(success["safetensors_sha256"], r"^[0-9a-f]{64}$")


def _parse_worker_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-worker", action="store_true", required=True)
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    if "--gpu-worker" in sys.argv:
        worker_arguments = _parse_worker_arguments()
        _gpu_worker(worker_arguments.artifact_dir)
    else:
        unittest.main()
