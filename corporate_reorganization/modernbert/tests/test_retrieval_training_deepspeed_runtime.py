from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
DEEPSPEED_CONFIG = MODERNBERT_DIR / "ds_zero3.json"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.provenance import validate_runtime_versions  # noqa: E402


class DerivedDeepSpeedRuntimeTest(unittest.TestCase):
    def test_complete_installed_inventory_is_exact(self) -> None:
        inventory = validate_runtime_versions()
        self.assertEqual(inventory["deepspeed"], "0.17.1")
        self.assertEqual(inventory["hjson"], "3.1.0")
        self.assertEqual(inventory["nvidia-ml-py"], "13.590.48")
        self.assertEqual(inventory["py-cpuinfo"], "9.0.0")

    def test_deepspeed_reads_the_frozen_batch_clip_and_zero_contract(self) -> None:
        class FourRankMpu:
            @staticmethod
            def get_data_parallel_world_size() -> int:
                return 4

        with mock.patch("deepspeed.runtime.config.dist.get_rank", return_value=0):
            config = DeepSpeedConfig(str(DEEPSPEED_CONFIG), mpu=FourRankMpu())
        self.assertEqual(config.train_micro_batch_size_per_gpu, 4)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.train_batch_size, 128)
        self.assertEqual(config.gradient_clipping, 1.0)
        self.assertTrue(config.bfloat16_config.enabled)
        self.assertEqual(int(config.zero_config.stage), 3)
        self.assertTrue(config.zero_config.reduce_scatter)
        self.assertTrue(config.zero_config.overlap_comm)

    def test_hugging_face_reconciliation_has_no_mismatch(self) -> None:
        args = SimpleNamespace(
            world_size=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            max_grad_norm=1.0,
            learning_rate=1e-5,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.01,
            fp16=False,
            fp16_full_eval=False,
            fp16_backend="auto",
            fp16_opt_level="O1",
            bf16=True,
            bf16_full_eval=False,
            save_on_each_node=False,
            get_warmup_steps=lambda total: 6,
        )
        config = HfTrainerDeepSpeedConfig(str(DEEPSPEED_CONFIG))
        config.trainer_config_process(args)
        config.trainer_config_finalize(
            args,
            SimpleNamespace(config=SimpleNamespace(hidden_size=768)),
            num_training_steps=60,
        )
        self.assertEqual(config.mismatches, [])
        self.assertEqual(config.dtype(), torch.bfloat16)

    def test_explicit_boundary_api_is_the_pinned_engine_contract(self) -> None:
        signature = inspect.signature(DeepSpeedEngine.set_gradient_accumulation_boundary)
        self.assertEqual(list(signature.parameters), ["self", "is_boundary"])


if __name__ == "__main__":
    unittest.main()
