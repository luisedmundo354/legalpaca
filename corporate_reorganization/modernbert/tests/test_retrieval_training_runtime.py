from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import numpy
import transformers
from accelerate.data_loader import BatchSamplerShard, prepare_data_loader
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.models.modernbert import modeling_modernbert
from transformers.training_args import OptimizerNames


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.batching import (  # noqa: E402
    DUMMY_QUERY_INDEX,
    GlobalQueryBatchSampler,
    SentinelQueryDataset,
)
from retriever.collator import RetrievalBatchCollator  # noqa: E402
from retriever.losses import multi_positive_nce_loss, multi_positive_nce_loss_sum  # noqa: E402
from retriever.provenance import EXPECTED_BASE_RUNTIME_VERSIONS  # noqa: E402
from trainer import ControlledRetrievalTrainer  # noqa: E402
import train_sm as controlled_train  # noqa: E402


class PinnedRuntimeContractTest(unittest.TestCase):
    def test_aws_base_image_versions_are_exact(self) -> None:
        self.assertEqual(platform.python_version(), EXPECTED_BASE_RUNTIME_VERSIONS["python"])
        for package, expected in EXPECTED_BASE_RUNTIME_VERSIONS.items():
            if package == "python":
                continue
            self.assertEqual(importlib.metadata.version(package), expected, package)

    def test_explicit_determinism_replays_rngs_and_installs_strict_flags(self) -> None:
        def seed_and_draw():
            controlled_train._configure_determinism(
                experiment_seed=17,
                torch_module=torch,
                numpy_module=numpy,
                transformers_module=transformers,
            )
            return (
                random.random(),
                float(numpy.random.random()),
                float(torch.rand(1).item()),
            )

        self.assertEqual(seed_and_draw(), seed_and_draw())
        controlled_train._validate_determinism_state(torch)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_trainer_initialization_preserves_frozen_cublas_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                clear=False,
            ):
                args = TrainingArguments(
                    output_dir=tmp_dir,
                    full_determinism=False,
                    use_cpu=True,
                    report_to=[],
                )
                Trainer(model=torch.nn.Linear(2, 1), args=args)
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_modernbert_passes_its_deterministic_flag_to_flash_attention(self) -> None:
        captured = {}

        def fake_flash(qkv, **kwargs):
            captured.update(kwargs)
            return qkv[:, 0]

        class IdentityRotary:
            @staticmethod
            def __call__(qkv, **kwargs):
                del kwargs
                return qkv

        module = SimpleNamespace(
            attention_dropout=0.1,
            training=True,
            deterministic_flash_attn=True,
        )
        qkv = torch.zeros((2, 3, 1, 2), dtype=torch.float32)
        with mock.patch.object(
            modeling_modernbert,
            "flash_attn_varlen_qkvpacked_func",
            side_effect=fake_flash,
            create=True,
        ):
            output = modeling_modernbert.flash_attention_forward(
                module,
                qkv=qkv,
                rotary_emb=IdentityRotary(),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
                max_seqlen=2,
                local_attention=(-1, -1),
                bs=1,
                dim=4,
            )[0]
        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertIs(captured["deterministic"], True)
        self.assertEqual(captured["dropout_p"], 0.1)

    def test_accelerate_shards_the_global_plan_once_without_repetition(self) -> None:
        sampler = GlobalQueryBatchSampler(
            [f"q{index:03d}" for index in range(294)],
            experiment_seed=17,
            world_size=4,
            per_device_batch_size=4,
        )
        shards = [
            list(
                BatchSamplerShard(
                    sampler,
                    num_processes=4,
                    process_index=rank,
                    split_batches=False,
                    even_batches=False,
                )
            )
            for rank in range(4)
        ]
        self.assertEqual([len(shard) for shard in shards], [19] * 4)
        self.assertEqual(
            [sum(index != DUMMY_QUERY_INDEX for index in shard[-1]) for shard in shards],
            [2, 2, 1, 1],
        )
        real = [
            index
            for shard in shards
            for batch in shard
            for index in batch
            if index != DUMMY_QUERY_INDEX
        ]
        self.assertEqual(len(real), 294)
        self.assertEqual(len(set(real)), 294)
        self.assertEqual(sorted(real), list(range(294)))

    def test_prepared_dataloader_epoch_updates_dataset_and_custom_sampler(self) -> None:
        class EpochDataset:
            def __init__(self) -> None:
                self.epoch = 0

            @staticmethod
            def __len__() -> int:
                return 294

            @staticmethod
            def __getitem__(index: int):
                return {"index": index}

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

        source = EpochDataset()
        sampler = GlobalQueryBatchSampler(
            [f"q{index:03d}" for index in range(294)],
            experiment_seed=17,
            world_size=4,
            per_device_batch_size=4,
        )
        epoch_zero = sampler.batches()
        wrapped = SentinelQueryDataset(source, epoch_target=sampler)
        dataloader = DataLoader(
            wrapped,
            batch_sampler=sampler,
            collate_fn=lambda rows: rows,
            num_workers=0,
        )
        prepared = prepare_data_loader(
            dataloader,
            device=None,
            num_processes=4,
            process_index=0,
            split_batches=False,
            put_on_device=False,
            dispatch_batches=False,
            even_batches=False,
            use_seedable_sampler=False,
        )
        self.assertEqual(len(prepared), 19)
        prepared.set_epoch(3)
        self.assertEqual(source.epoch, 3)
        self.assertEqual(sampler.epoch, 3)
        self.assertNotEqual(epoch_zero, sampler.batches())


class CollatorAndLossTest(unittest.TestCase):
    class FakeTokenizer:
        unk_token_id = 0

        def __init__(self) -> None:
            self.truncation_side = "right"
            self.calls: list[tuple[str, list[str]]] = []

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            del token
            return 9

        def __call__(self, texts, **kwargs):
            del kwargs
            values = list(texts)
            self.calls.append((self.truncation_side, values))
            if self.truncation_side == "left":
                input_ids = torch.tensor([[1, 9, 2] for _ in values], dtype=torch.long)
            else:
                input_ids = torch.tensor([[1, 2, 0] for _ in values], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": input_ids.ne(0).to(torch.long),
            }

    def test_mixed_dummy_batch_tokenizes_only_real_scientific_content(self) -> None:
        tokenizer = self.FakeTokenizer()
        collator = RetrievalBatchCollator(
            tokenizer,
            {"p1": "one", "p2": "two", "p3": "three"},
            max_len_query=16,
            max_len_passage=8,
        )
        result = collator(
            [
                {
                    "is_dummy": False,
                    "query_text": "query one",
                    "positive_passage_ids": ["p1"],
                    "candidate_passage_ids": ["p1", "p2"],
                },
                {"is_dummy": True},
                {
                    "is_dummy": False,
                    "query_text": "query two",
                    "positive_passage_ids": ["p3"],
                    "candidate_passage_ids": ["p3"],
                },
                {"is_dummy": True},
            ]
        )
        self.assertEqual(result["valid_query_count"].dtype, torch.long)
        self.assertEqual(result["valid_query_count"].item(), 2)
        self.assertEqual(tuple(result["query_input_ids"].shape), (2, 3))
        self.assertEqual(tuple(result["passage_input_ids"].shape), (3, 3))
        self.assertEqual(
            tokenizer.calls,
            [
                ("left", ["query one", "query two"]),
                ("right", ["one", "two", "three"]),
            ],
        )

        prior_calls = list(tokenizer.calls)
        with self.assertRaisesRegex(ValueError, "all-dummy"):
            collator([{"is_dummy": True}, {"is_dummy": True}])
        self.assertEqual(tokenizer.calls, prior_calls)

    def test_summed_multi_positive_loss_matches_per_query_definition(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.0, -1.0], [0.0, 3.0, 2.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        mask = torch.tensor(
            [[True, False, False], [False, True, True]],
            dtype=torch.bool,
        )
        mean_loss, mean_per_query = multi_positive_nce_loss(logits, mask)
        sum_loss, sum_per_query = multi_positive_nce_loss_sum(logits, mask)
        torch.testing.assert_close(sum_loss, mean_loss * logits.shape[0])
        torch.testing.assert_close(sum_per_query, mean_per_query)
        sum_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        with self.assertRaisesRegex(ValueError, "at least one positive"):
            multi_positive_nce_loss_sum(logits.detach(), torch.zeros_like(mask))


class AccumulationContractTest(unittest.TestCase):
    @staticmethod
    def _normalized_gradient(counts_by_rank: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients: list[float] = []
        next_coefficient = 1.0
        local_sums: list[torch.Tensor] = []
        parameter = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        total_valid = sum(sum(rank_counts) for rank_counts in counts_by_rank)
        for rank_counts in counts_by_rank:
            for count in rank_counts:
                local_coefficients = [next_coefficient + offset for offset in range(count)]
                next_coefficient += count
                coefficients.extend(local_coefficients)
                local_sums.append(parameter * sum(local_coefficients))
        deepspeed_averaged_loss = sum(
            local_sum * (4.0 / total_valid) for local_sum in local_sums
        ) / 4.0
        deepspeed_averaged_loss.backward()

        reference = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        (reference * (sum(coefficients) / total_valid)).backward()
        return parameter.grad, reference.grad

    def test_full_and_incomplete_windows_equal_direct_valid_query_mean(self) -> None:
        full = [[4] * 8 for _ in range(4)]
        final = [[4, 4, 2], [4, 4, 2], [4, 4, 1], [4, 4, 1]]
        for layout in (full, final):
            actual, expected = self._normalized_gradient(layout)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_transformers_parent_does_not_apply_an_extra_gas_division(self) -> None:
        class RecordingAccelerator:
            distributed_type = DistributedType.DEEPSPEED

            def __init__(self) -> None:
                self.loss = None
                self.kwargs = None

            def backward(self, loss, **kwargs) -> None:
                self.loss = loss.detach().clone()
                self.kwargs = kwargs
                loss.backward()

        class FakeTrainer:
            def __init__(self) -> None:
                self.optimizer = SimpleNamespace()
                self.args = SimpleNamespace(
                    optim=OptimizerNames.ADAMW_TORCH,
                    torch_empty_cache_steps=None,
                    n_gpu=1,
                    gradient_accumulation_steps=8,
                    device=torch.device("cpu"),
                )
                self.state = SimpleNamespace(global_step=0)
                self.use_apex = False
                self.model_accepts_loss_kwargs = True
                self.compute_loss_func = None
                self.accelerator = RecordingAccelerator()

            @staticmethod
            def _prepare_inputs(inputs):
                return inputs

            @staticmethod
            def compute_loss_context_manager():
                return nullcontext()

            @staticmethod
            def compute_loss(model, inputs, num_items_in_batch=None):
                del inputs, num_items_in_batch
                return model.weight * 2.0

        class ScalarModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

        fake = FakeTrainer()
        model = ScalarModel()
        returned = Trainer.training_step(fake, model, {}, num_items_in_batch=38)
        self.assertEqual(returned.item(), 2.0)
        self.assertEqual(model.weight.grad.item(), 2.0)
        self.assertEqual(fake.accelerator.loss.item(), 2.0)
        self.assertEqual(fake.accelerator.kwargs, {"scale_wrt_gas": False})

    def test_ceiling_schedule_and_exact_window_prefetch(self) -> None:
        fake_schedule = SimpleNamespace(
            EXPECTED_EPOCHS=20,
            EXPECTED_UPDATES_PER_EPOCH=3,
            EXPECTED_QUERIES=294,
            EXPECTED_PREPARED_BATCHES=19,
            EXPECTED_TOTAL_UPDATES=60,
            num_examples=lambda dataloader: 294,
        )

        class NineteenBatches:
            @staticmethod
            def __len__() -> int:
                return 19

        values = ControlledRetrievalTrainer.set_initial_training_values(
            fake_schedule,
            SimpleNamespace(max_steps=-1, num_train_epochs=20.0),
            NineteenBatches(),
            128,
        )
        self.assertEqual(values, (20, 3, 294, 5880, True, 19, 60))

        reduction_results = iter([128, 128, 38])

        def reduce_counts(*, local_valid_count: int, local_microbatches: int) -> int:
            expected_local = {8: 32, 3: 10}[local_microbatches]
            self.assertEqual(local_valid_count, expected_local)
            return next(reduction_results)

        fake_windows = SimpleNamespace(
            EXPECTED_GRADIENT_ACCUMULATION=8,
            EXPECTED_WINDOW_VALID_QUERIES=(128, 128, 38),
            EXPECTED_WINDOW_MICROBATCHES=(8, 8, 3),
            _global_batch_sampler=SimpleNamespace(epoch=0),
            _window_epoch=None,
            _window_index=0,
            _exact_scalar_count=ControlledRetrievalTrainer._exact_scalar_count,
            _reduce_window_counts=reduce_counts,
        )
        batches = [
            {"valid_query_count": torch.tensor(4, dtype=torch.long)} for _ in range(18)
        ] + [{"valid_query_count": torch.tensor(2, dtype=torch.long)}]
        iterator = iter(batches)
        observed = []
        for _ in range(3):
            window, count = ControlledRetrievalTrainer.get_batch_samples(
                fake_windows,
                iterator,
                num_batches=999,
            )
            observed.append((len(window), count))
            self.assertEqual(
                [batch["is_window_end"] for batch in window],
                [False] * (len(window) - 1) + [True],
            )
            self.assertTrue(
                all(batch["global_window_valid_count"] == count for batch in window)
            )
        self.assertEqual(observed, [(8, 128), (8, 128), (3, 38)])

    def test_training_step_sets_every_explicit_deepspeed_boundary(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.boundaries: list[bool] = []

            def set_gradient_accumulation_boundary(self, marker: bool) -> None:
                self.boundaries.append(marker)

        engine = FakeEngine()
        fake = SimpleNamespace(
            accelerator=SimpleNamespace(sync_gradients=False),
            args=SimpleNamespace(device=torch.device("cpu")),
        )
        with mock.patch.object(Trainer, "training_step", return_value=torch.tensor(1.0)):
            ControlledRetrievalTrainer.training_step(
                fake,
                engine,
                {"is_window_end": False},
                num_items_in_batch=128,
            )
            fake.accelerator.sync_gradients = True
            ControlledRetrievalTrainer.training_step(
                fake,
                engine,
                {"is_window_end": True},
                num_items_in_batch=38,
            )
        self.assertEqual(engine.boundaries, [False, True])


if __name__ == "__main__":
    unittest.main()
