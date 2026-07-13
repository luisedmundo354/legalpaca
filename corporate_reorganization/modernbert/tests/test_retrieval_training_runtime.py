from __future__ import annotations

import copy
import importlib.metadata
import inspect
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
from retriever.collator import (  # noqa: E402
    ControlledRetrievalBatchCollator,
    RetrievalBatchCollator,
)
from retriever.losses import (  # noqa: E402
    build_index_positive_mask,
    masked_multi_positive_nce_loss_sum,
    multi_positive_nce_loss,
    multi_positive_nce_loss_sum,
)
from retriever.models import DualEncoderRetriever  # noqa: E402
from retriever.provenance import EXPECTED_BASE_RUNTIME_VERSIONS  # noqa: E402
from retriever.sampling import (  # noqa: E402
    SELECTION_ALGORITHM,
    TRACE_SCHEMA_VERSION,
    sampling_trace_checksum,
)
from trainer import (  # noqa: E402
    ControlledRetrievalTrainer,
    DETERMINISM_SMOKE_TRAINING_SCHEDULE,
    FULL_CONTROLLED_TRAINING_SCHEDULE,
)
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

    @staticmethod
    def _controlled_example(
        *,
        query_id: str,
        positive_indices: list[int],
        selected_positive_count: int,
    ) -> dict:
        positive_ids = [f"p{index:03d}" for index in positive_indices]
        selected_ids = positive_ids[:selected_positive_count]
        negative_indices = [
            index for index in range(100, 200) if index not in positive_indices
        ][:60]
        negative_ids = [f"p{index:03d}" for index in negative_indices]
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "selection_algorithm": SELECTION_ALGORITHM,
            "sampler": "global_uniform",
            "experiment_seed": 17,
            "epoch": 0,
            "query_id": query_id,
            "doc_id": "case",
            "positive_passage_ids": positive_ids,
            "selected_positive_passage_ids": selected_ids,
            "negative_passage_ids_by_stratum": {"global": negative_ids},
            "eligible_pool_sizes_by_stratum": {"global": 100},
            "candidate_passage_ids": [*selected_ids, *negative_ids],
        }
        trace = {**payload, "trace_sha256": sampling_trace_checksum(payload)}
        return {
            "is_dummy": False,
            "query_id": query_id,
            "doc_id": "case",
            "query_text": f"query {query_id}",
            "positive_passage_indices": positive_indices,
            "candidate_passage_indices": [
                *positive_indices[:selected_positive_count],
                *negative_indices,
            ],
            "sampling_trace": trace,
            "sampling_trace_sha256": trace["trace_sha256"],
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

    def test_controlled_collator_transports_indices_and_traces_without_passage_tokenization(self) -> None:
        tokenizer = self.FakeTokenizer()
        collator = ControlledRetrievalBatchCollator(
            tokenizer,
            corpus_size=200,
            max_len_query=16,
        )
        first = self._controlled_example(
            query_id="q1",
            positive_indices=[0, 1, 2, 3, 4],
            selected_positive_count=4,
        )
        second = self._controlled_example(
            query_id="q2",
            positive_indices=[5],
            selected_positive_count=1,
        )
        result = collator([first, {"is_dummy": True}, second, {"is_dummy": True}])

        self.assertEqual(tokenizer.calls, [("left", ["query q1", "query q2"])])
        self.assertEqual(tuple(result["query_input_ids"].shape), (2, 3))
        self.assertEqual(tuple(result["candidate_passage_indices"].shape), (2, 64))
        self.assertEqual(tuple(result["positive_passage_indices"].shape), (2, 5))
        self.assertEqual(result["candidate_passage_indices"][1, -3:].tolist(), [-1, -1, -1])
        self.assertEqual(result["positive_passage_indices"][1, 1:].tolist(), [-1, -1, -1, -1])
        self.assertEqual([trace["query_id"] for trace in result["sampling_traces"]], ["q1", "q2"])
        self.assertNotIn("passage_input_ids", result)
        self.assertNotIn("passage_id_hashes", result)

        malformed = []
        duplicate = copy.deepcopy(first)
        duplicate["candidate_passage_indices"][1] = duplicate["candidate_passage_indices"][0]
        malformed.append(duplicate)
        boolean_index = copy.deepcopy(first)
        boolean_index["candidate_passage_indices"][0] = True
        malformed.append(boolean_index)
        out_of_range = copy.deepcopy(first)
        out_of_range["candidate_passage_indices"][0] = 200
        malformed.append(out_of_range)
        no_positive = copy.deepcopy(first)
        no_positive["candidate_passage_indices"] = list(range(100, 164))
        malformed.append(no_positive)
        for example in malformed:
            with self.subTest(first_candidate=example["candidate_passage_indices"][0]):
                with self.assertRaises((TypeError, ValueError)):
                    collator([example])

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

    def test_controlled_loss_masks_invalid_padding_from_numerator_and_denominator(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 100.0]], dtype=torch.float64, requires_grad=True)
        passage_indices = torch.tensor([4, 7, -1], dtype=torch.long)
        positives = torch.tensor([[4, -1]], dtype=torch.long)
        valid = torch.tensor([True, True, False], dtype=torch.bool)
        positive_mask = build_index_positive_mask(passage_indices, positives, valid)
        loss, _ = masked_multi_positive_nce_loss_sum(logits, positive_mask, valid)
        expected = torch.logsumexp(logits[0, :2], dim=0) - logits[0, 0]
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0.0)

    def test_controlled_forward_uses_the_top_level_engine_call_once(self) -> None:
        class ForbiddenDirectModule:
            @staticmethod
            def encode_queries(*args, **kwargs):
                raise AssertionError("Trainer bypassed the top-level engine call")

            @staticmethod
            def encode_passages(*args, **kwargs):
                raise AssertionError("Trainer bypassed the top-level engine call")

        class RecordingEngine:
            module = ForbiddenDirectModule()

            def __init__(self) -> None:
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "query_embeddings": torch.ones((2, 3)),
                    "passage_embeddings": torch.ones((4, 3)),
                }

        engine = RecordingEngine()
        query_embeddings, passage_embeddings = ControlledRetrievalTrainer._forward_controlled_batch(
            engine,
            query_input_ids=torch.ones((2, 5), dtype=torch.long),
            query_attention_mask=torch.ones((2, 5), dtype=torch.long),
            passage_input_ids=torch.ones((4, 6), dtype=torch.long),
            passage_attention_mask=torch.ones((4, 6), dtype=torch.long),
        )
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(tuple(query_embeddings.shape), (2, 3))
        self.assertEqual(tuple(passage_embeddings.shape), (4, 3))

    def test_dual_encoder_forward_requires_and_encodes_both_branches(self) -> None:
        self.assertEqual(
            inspect.signature(DualEncoderRetriever.forward).parameters["unused"].kind,
            inspect.Parameter.VAR_KEYWORD,
        )

        class FakeEncoder(torch.nn.Module):
            @staticmethod
            def forward(input_ids, attention_mask, return_dict):
                del attention_mask
                if return_dict is not True:
                    raise AssertionError("Retriever changed the encoder return contract")
                hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
                return SimpleNamespace(last_hidden_state=hidden)

        retriever = DualEncoderRetriever(
            FakeEncoder(),
            slot_token_id=9,
            temperature=0.07,
        )
        output = retriever(
            query_input_ids=torch.tensor([[1, 9, 2]], dtype=torch.long),
            query_attention_mask=torch.ones((1, 3), dtype=torch.long),
            passage_input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            passage_attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
        self.assertEqual(set(output), {"query_embeddings", "passage_embeddings"})
        self.assertEqual(tuple(output["query_embeddings"].shape), (1, 3))
        self.assertEqual(tuple(output["passage_embeddings"].shape), (1, 3))
        with self.assertRaisesRegex(ValueError, "missing"):
            retriever(query_input_ids=torch.ones((1, 1), dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "Unexpected"):
            retriever(
                query_input_ids=torch.tensor([[9]], dtype=torch.long),
                query_attention_mask=torch.ones((1, 1), dtype=torch.long),
                passage_input_ids=torch.ones((1, 1), dtype=torch.long),
                passage_attention_mask=torch.ones((1, 1), dtype=torch.long),
                unexpected=True,
            )

    def test_owned_passage_tokenizer_state_restores_after_failure(self) -> None:
        class FailingTokenizer:
            truncation_side = "left"

            def __call__(self, texts, **kwargs):
                del texts, kwargs
                if self.truncation_side != "right":
                    raise AssertionError("Passage tokenization did not switch to right truncation")
                raise RuntimeError("injected tokenizer failure")

        tokenizer = FailingTokenizer()
        with self.assertRaisesRegex(RuntimeError, "injected tokenizer failure"):
            ControlledRetrievalTrainer._tokenize_owned_passages(
                tokenizer,
                ["passage"],
                max_len_passage=500,
            )
        self.assertEqual(tokenizer.truncation_side, "left")


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
            EXPECTED_QUERIES=294,
            EXPECTED_PREPARED_BATCHES=19,
            training_schedule=FULL_CONTROLLED_TRAINING_SCHEDULE,
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

        fake_schedule.training_schedule = DETERMINISM_SMOKE_TRAINING_SCHEDULE
        smoke_values = ControlledRetrievalTrainer.set_initial_training_values(
            fake_schedule,
            SimpleNamespace(max_steps=-1, num_train_epochs=2.0),
            NineteenBatches(),
            128,
        )
        self.assertEqual(smoke_values, (2, 3, 294, 588, True, 19, 6))

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
            _candidate_trace_store=SimpleNamespace(record_batch=mock.Mock()),
        )
        batch_tensors = {
            "sampling_traces": [{"query_id": "q"}],
            "candidate_passage_indices": torch.tensor([[1]], dtype=torch.long),
            "positive_passage_indices": torch.tensor([[1]], dtype=torch.long),
        }
        with mock.patch.object(
            Trainer,
            "training_step",
            return_value=torch.tensor(1.0),
        ) as parent_training_step:
            ControlledRetrievalTrainer.training_step(
                fake,
                engine,
                {"is_window_end": False, **batch_tensors},
                num_items_in_batch=128,
            )
            fake.accelerator.sync_gradients = True
            ControlledRetrievalTrainer.training_step(
                fake,
                engine,
                {"is_window_end": True, **batch_tensors},
                num_items_in_batch=38,
            )
        self.assertEqual(engine.boundaries, [False, True])
        self.assertEqual(fake._candidate_trace_store.record_batch.call_count, 2)
        self.assertTrue(
            all(
                "sampling_traces" not in call.args[2]
                for call in parent_training_step.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
