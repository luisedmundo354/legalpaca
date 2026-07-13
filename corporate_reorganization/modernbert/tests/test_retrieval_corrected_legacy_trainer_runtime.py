from __future__ import annotations

import inspect
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from accelerate import Accelerator
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers import Trainer
from transformers.trainer_callback import TrainerControl


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
CORRECTED_CONFIG = (
    MODERNBERT_DIR / "experiments/retrieval_cv/configs/corrected_legacy.json"
)
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import corrected_legacy_train  # noqa: E402
import legacy_diagnostic_trainer as trainer_module  # noqa: E402
import train_sm  # noqa: E402
from retriever.batching import DUMMY_QUERY_INDEX  # noqa: E402
from retriever.legacy_diagnostic_batching import (  # noqa: E402
    BATCH_ORDER_ALGORITHM,
    CorrectedLegacyQueryBatchSampler,
)
from retriever.legacy_diagnostic_sampling import SELECTION_ALGORITHM  # noqa: E402


class _LengthOnlyDataloader:
    def __len__(self) -> int:
        return 27


class CorrectedLegacyPinnedApiTest(unittest.TestCase):
    def test_exact_transformers_accelerate_and_deepspeed_override_signatures(self) -> None:
        self.assertEqual(
            list(inspect.signature(Trainer.set_initial_training_values).parameters),
            ["self", "args", "dataloader", "total_train_batch_size"],
        )
        self.assertEqual(
            list(inspect.signature(Trainer.get_batch_samples).parameters),
            ["self", "epoch_iterator", "num_batches"],
        )
        self.assertEqual(
            list(inspect.signature(Trainer.training_step).parameters),
            ["self", "model", "inputs", "num_items_in_batch"],
        )
        self.assertEqual(
            list(inspect.signature(Trainer.evaluate).parameters),
            ["self", "eval_dataset", "ignore_keys", "metric_key_prefix"],
        )
        self.assertEqual(
            list(inspect.signature(Trainer.get_eval_dataloader).parameters),
            ["self", "eval_dataset"],
        )
        self.assertEqual(
            list(inspect.signature(Accelerator.prepare).parameters),
            ["self", "args", "device_placement"],
        )
        self.assertEqual(
            list(inspect.signature(Accelerator.get_state_dict).parameters),
            ["self", "model", "unwrap"],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    DeepSpeedEngine.set_gradient_accumulation_boundary
                ).parameters
            ),
            ["self", "is_boundary"],
        )

        self.assertEqual(
            list(
                inspect.signature(
                    trainer_module.CorrectedLegacyDiagnosticTrainer.set_initial_training_values
                ).parameters
            ),
            ["self", "args", "dataloader", "total_train_batch_size"],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    trainer_module.CorrectedLegacyDiagnosticTrainer.get_batch_samples
                ).parameters
            ),
            ["self", "epoch_iterator", "num_batches"],
        )
        self.assertEqual(
            list(
                inspect.signature(
                    trainer_module.CorrectedLegacyDiagnosticTrainer.training_step
                ).parameters
            ),
            ["self", "model", "inputs", "num_items_in_batch"],
        )

    def test_accelerate_14_epoch_propagation_reaches_wrapped_dataset(self) -> None:
        source = inspect.getsource(DataLoaderShard.set_epoch)
        self.assertIn('hasattr(self.batch_sampler, "set_epoch")', source)
        self.assertIn('hasattr(self.dataset, "set_epoch")', source)
        self.assertFalse(hasattr(BatchSamplerShard, "set_epoch"))
        wrapped = BatchSamplerShard(
            CorrectedLegacyQueryBatchSampler(
                [f"query-{index:03d}" for index in range(418)],
                experiment_seed=17,
                world_size=4,
                per_device_batch_size=4,
            ),
            num_processes=4,
            process_index=0,
            split_batches=False,
            even_batches=False,
        )
        self.assertTrue(hasattr(wrapped, "batch_sampler"))


class CorrectedLegacyEntrypointTest(unittest.TestCase):
    @staticmethod
    def argv(*, query_view: str = "structured") -> list[str]:
        return [
            "--data-dir",
            "/data",
            "--base-model-dir",
            "/model",
            "--output-dir",
            "/output",
            "--query-view",
            query_view,
            "--base-seed",
            "17",
            "--run-kind",
            "corrected_legacy_diagnostic",
            "--epochs",
            "20",
            "--total-optimizer-updates",
            "80",
        ]

    def test_parser_accepts_only_the_two_sealed_job_coordinates(self) -> None:
        for query_view in ("flat_masked", "structured"):
            args = train_sm.parse_args(self.argv(query_view=query_view))
            self.assertEqual(args.run_kind, "corrected_legacy_diagnostic")
            self.assertEqual(args.query_view, query_view)
            self.assertEqual(args.base_seed, 17)
            self.assertEqual((args.epochs, args.total_optimizer_updates), (20, 80))
            self.assertIsNone(args.outer_fold)
            self.assertIsNone(args.sampler)
            self.assertIsNone(args.experiment_seed)

        invalid = (
            self.argv() + ["--outer-fold", "0"],
            self.argv() + ["--sampler", "local_unique"],
            self.argv() + ["--experiment-seed", "17"],
            self.argv()[:-1] + ["79"],
            self.argv()[: self.argv().index("--base-seed")]
            + self.argv()[self.argv().index("--base-seed") + 2 :],
        )
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                train_sm.parse_args(argv)

    def test_main_validates_then_dispatches_without_entering_controlled_path(self) -> None:
        events: list[object] = []
        provenance = {"source_bundle": "sealed"}

        def validate_files(**kwargs):
            events.append(("files", kwargs))

        def validate_preimport(seed):
            events.append(("preimport", seed))

        def validate_image():
            events.append("image")
            return provenance

        def validate_versions():
            events.append("versions")

        def run(args, *, training_launch_provenance):
            events.append(
                (
                    "run",
                    args.run_kind,
                    args.query_view,
                    training_launch_provenance,
                )
            )
            return 73

        with (
            mock.patch.object(
                train_sm,
                "_validate_frozen_control_file_hashes",
                side_effect=validate_files,
            ),
            mock.patch.object(
                train_sm,
                "validate_preimport_environment",
                side_effect=validate_preimport,
            ),
            mock.patch.object(
                train_sm,
                "validate_training_image_environment",
                side_effect=validate_image,
            ),
            mock.patch.object(
                train_sm,
                "validate_runtime_versions",
                side_effect=validate_versions,
            ),
            mock.patch.object(
                corrected_legacy_train,
                "run_corrected_legacy_diagnostic",
                side_effect=run,
            ),
        ):
            self.assertEqual(train_sm.main(self.argv(query_view="flat_masked")), 73)

        self.assertEqual(
            events,
            [
                (
                    "files",
                    {
                        "experiment_config_path": train_sm.DEFAULT_EXPERIMENT_CONFIG,
                        "deepspeed_config_path": train_sm.DEFAULT_DEEPSPEED_CONFIG,
                    },
                ),
                ("preimport", 17),
                "image",
                "versions",
                (
                    "run",
                    "corrected_legacy_diagnostic",
                    "flat_masked",
                    provenance,
                ),
            ],
        )

    def test_frozen_config_labels_match_the_executed_digest_algorithms(self) -> None:
        config = json.loads(CORRECTED_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            BATCH_ORDER_ALGORITHM,
            "sha256_corrected_legacy_query_order_v1",
        )
        self.assertEqual(
            SELECTION_ALGORITHM,
            "sha256_corrected_legacy_occurrences_v1",
        )
        self.assertEqual(config["batching"]["batch_order_algorithm"], BATCH_ORDER_ALGORITHM)
        self.assertEqual(
            config["candidate_sampling"]["selection_algorithm"],
            SELECTION_ALGORITHM,
        )

    def test_exact_staged_dataset_preflight_precedes_config_and_model_construction(self) -> None:
        source = inspect.getsource(
            corrected_legacy_train.run_corrected_legacy_diagnostic
        )
        positions = [
            source.index("validate_staged_dataset("),
            source.index("loaded_config = load_corrected_legacy_config("),
            source.index("corrected_data = load_corrected_legacy_data("),
            source.index("tokenizer = AutoTokenizer.from_pretrained("),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256",
            source,
        )


class CorrectedLegacyScheduleAndTraceTest(unittest.TestCase):
    def test_exact_20_epoch_80_update_trainer_schedule(self) -> None:
        fake = SimpleNamespace(num_examples=lambda dataloader: 418)
        args = SimpleNamespace(max_steps=-1, num_train_epochs=20.0)
        self.assertEqual(
            trainer_module.CorrectedLegacyDiagnosticTrainer.set_initial_training_values(
                fake,
                args,
                _LengthOnlyDataloader(),
                128,
            ),
            (20, 4, 418, 8_360, True, 27, 80),
        )

        for changed_args, total_batch_size in (
            (SimpleNamespace(max_steps=80, num_train_epochs=20.0), 128),
            (SimpleNamespace(max_steps=-1, num_train_epochs=19.0), 128),
            (args, 64),
        ):
            with self.subTest(
                args=changed_args,
                total_batch_size=total_batch_size,
            ), self.assertRaises(RuntimeError):
                trainer_module.CorrectedLegacyDiagnosticTrainer.set_initial_training_values(
                    fake,
                    changed_args,
                    _LengthOnlyDataloader(),
                    total_batch_size,
                )

    def test_optimizer_windows_are_exact_and_reset_only_at_next_epoch(self) -> None:
        reductions = mock.Mock(side_effect=[128, 128, 128, 34, 128])
        fake = SimpleNamespace(
            _global_batch_sampler=SimpleNamespace(epoch=0),
            _window_epoch=None,
            _window_index=0,
            _reduce_window_count=reductions,
            _valid_query_count=lambda batch: int(batch["valid_query_count"].item()),
        )

        for window_index, (microbatches, global_valid) in enumerate(
            zip((8, 8, 8, 3), (128, 128, 128, 34))
        ):
            batches = [
                {"valid_query_count": torch.tensor(4, dtype=torch.long)}
                for _ in range(microbatches)
            ]
            returned, returned_count = (
                trainer_module.CorrectedLegacyDiagnosticTrainer.get_batch_samples(
                    fake,
                    iter(batches),
                    2 if window_index == 3 else 8,
                )
            )
            self.assertEqual(returned, batches)
            self.assertEqual(returned_count, global_valid)
            self.assertEqual(
                [batch["is_window_end"] for batch in returned],
                [False] * (microbatches - 1) + [True],
            )
            self.assertTrue(
                all(
                    batch["global_window_valid_count"] == global_valid
                    for batch in returned
                )
            )

        with self.assertRaisesRegex(RuntimeError, "empty"):
            trainer_module.CorrectedLegacyDiagnosticTrainer.get_batch_samples(
                fake,
                iter(()),
                8,
            )

        fake._global_batch_sampler.epoch = 1
        first_next_epoch = [
            {"valid_query_count": torch.tensor(4, dtype=torch.long)}
            for _ in range(8)
        ]
        trainer_module.CorrectedLegacyDiagnosticTrainer.get_batch_samples(
            fake,
            iter(first_next_epoch),
            8,
        )
        self.assertEqual(fake._window_epoch, 1)
        self.assertEqual(fake._window_index, 1)
        self.assertEqual(
            [call.args for call in reductions.call_args_list],
            [(32, 8), (32, 8), (32, 8), (12, 3), (32, 8)],
        )

    def test_exact_trace_chronology_covers_8360_rows_across_four_shards(self) -> None:
        record_counts = []
        query_counts = []
        query_ids = [f"query-{index:03d}" for index in range(418)]
        for rank in range(4):
            fake = SimpleNamespace(
                _trace_epoch=None,
                _trace_microbatch_index=0,
                _trace_lines=[],
                _trace_query_ids_by_epoch={},
                accelerator=SimpleNamespace(process_index=rank),
            )
            sampler = CorrectedLegacyQueryBatchSampler(
                query_ids,
                experiment_seed=17,
                world_size=4,
                per_device_batch_size=4,
            )
            for epoch in range(20):
                sampler.set_epoch(epoch)
                rank_batches = sampler.batches()[rank::4]
                self.assertEqual(len(rank_batches), 27)
                for batch in rank_batches:
                    traces = [
                        {"epoch": epoch, "query_id": query_ids[index]}
                        for index in batch
                        if index != DUMMY_QUERY_INDEX
                    ]
                    trainer_module.CorrectedLegacyDiagnosticTrainer._record_traces(
                        fake,
                        traces,
                    )
            shard = trainer_module.CorrectedLegacyDiagnosticTrainer.local_trace_shard(
                fake
            )
            record_counts.append(shard["record_count"])
            query_counts.append(shard["query_counts_by_epoch"])

        self.assertEqual(record_counts, [2_120, 2_080, 2_080, 2_080])
        self.assertEqual(sum(record_counts), 8_360)
        self.assertEqual(query_counts[0], [106] * 20)
        self.assertEqual(query_counts[1:], [[104] * 20] * 3)

    def test_trace_and_cross_rank_window_divergence_fail_loudly(self) -> None:
        trace_fake = SimpleNamespace(
            _trace_epoch=None,
            _trace_microbatch_index=0,
            _trace_lines=[],
            _trace_query_ids_by_epoch={},
            accelerator=SimpleNamespace(process_index=0),
        )
        trainer_module.CorrectedLegacyDiagnosticTrainer._record_traces(
            trace_fake,
            [{"epoch": 0, "query_id": "q0"}],
        )
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            trainer_module.CorrectedLegacyDiagnosticTrainer._record_traces(
                trace_fake,
                [{"epoch": 0, "query_id": "q0"}],
            )
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            trainer_module.CorrectedLegacyDiagnosticTrainer._record_traces(
                trace_fake,
                [{"epoch": 2, "query_id": "q2"}],
            )

        reduction_fake = SimpleNamespace(args=SimpleNamespace(device=torch.device("cpu")))

        def divergent_reduce(tensor, *, op):
            if op == trainer_module.dist.ReduceOp.MIN:
                tensor.fill_(3)
            elif op == trainer_module.dist.ReduceOp.MAX:
                tensor.fill_(4)
            else:
                self.fail(f"Unexpected reduction op: {op}")

        with mock.patch.object(
            trainer_module.dist,
            "all_reduce",
            side_effect=divergent_reduce,
        ), self.assertRaisesRegex(RuntimeError, "different window lengths"):
            trainer_module.CorrectedLegacyDiagnosticTrainer._reduce_window_count(
                reduction_fake,
                local_valid=10,
                local_microbatches=3,
            )


class CorrectedLegacyEvaluationAndFinalStateTest(unittest.TestCase):
    def test_epoch_validation_requires_and_uses_the_active_engine(self) -> None:
        engine = object()
        result = SimpleNamespace(
            metrics={"case_macro_set_recall_at_20": 0.75},
            to_payload=lambda: {"schema_version": 1, "rankings": []},
        )
        callback = mock.Mock(
            return_value=TrainerControl(should_evaluate=False)
        )
        logged: list[dict[str, float]] = []
        fake = SimpleNamespace(
            eval_dataset=object(),
            model_wrapped=engine,
            model=object(),
            deepspeed=engine,
            state=SimpleNamespace(epoch=1.0, global_step=4),
            _completed_epoch=lambda: 1,
            _evaluated_epochs=set(),
            _validation_records=[],
            validation_data=object(),
            passage_index_table=object(),
            processing_class=object(),
            log=lambda metrics: logged.append(metrics),
            callback_handler=SimpleNamespace(on_evaluate=callback),
            args=object(),
            control=TrainerControl(should_evaluate=True),
        )

        with mock.patch.object(
            trainer_module,
            "evaluate_corrected_legacy_validation_evidence_distributed",
            return_value=result,
        ) as evaluator:
            metrics = trainer_module.CorrectedLegacyDiagnosticTrainer.evaluate(fake)
        evaluator.assert_called_once_with(
            engine,
            fake.processing_class,
            validation_data=fake.validation_data,
            passage_index_table=fake.passage_index_table,
        )
        self.assertEqual(metrics, {"eval_validation_case_macro_set_recall_at_20": 0.75})
        self.assertEqual(logged, [metrics])
        self.assertEqual(fake._evaluated_epochs, {1})
        self.assertEqual(
            (fake._validation_records[0]["epoch"], fake._validation_records[0]["global_step"]),
            (1, 4),
        )
        callback.assert_called_once()

        inactive = SimpleNamespace(**vars(fake))
        inactive.model_wrapped = inactive.model
        inactive._evaluated_epochs = set()
        with self.assertRaisesRegex(RuntimeError, "active DeepSpeed engine"):
            trainer_module.CorrectedLegacyDiagnosticTrainer.evaluate(inactive)

    def test_no_checkpoint_or_best_reload_and_final_engine_order_are_sealed(self) -> None:
        source = inspect.getsource(
            corrected_legacy_train.run_corrected_legacy_diagnostic
        )
        training_arguments_region = source[
            source.index("training_args = TrainingArguments(") : source.index(
                "trainer = CorrectedLegacyDiagnosticTrainer("
            )
        ]
        self.assertIn('save_strategy="no"', training_arguments_region)
        self.assertIn("load_best_model_at_end=False", training_arguments_region)
        self.assertIn("metric_for_best_model=None", training_arguments_region)
        self.assertIn("save_total_limit=None", training_arguments_region)
        self.assertNotIn("save_only_model=True", training_arguments_region)
        self.assertNotIn("load_and_verify_best_checkpoint", source)
        self.assertNotIn("release_current_deepspeed_engine", source)
        self.assertNotIn("_load_best_model", source)

        ordered_fragments = (
            "trainer.train()",
            "active_engine = trainer.deepspeed",
            "if active_engine is None or trainer.model_wrapped is not active_engine:",
            "validation_history = trainer.validation_history()",
            "final_results = evaluate_corrected_legacy_test_distributed(\n        active_engine,",
            "trace_manifest = _publish_trace_artifacts(",
            "gathered_state = trainer.accelerator.get_state_dict(active_engine)",
            "unset_hf_deepspeed_config()",
            'published = rank_zero_call("Corrected legacy artifact publication", publish_artifacts)',
        )
        positions = [source.index(fragment) for fragment in ordered_fragments]
        self.assertEqual(positions, sorted(positions))

    def test_completed_epoch_and_validation_history_chronology_fail_loudly(self) -> None:
        fake = SimpleNamespace(state=SimpleNamespace(epoch=20.0, global_step=80))
        self.assertEqual(
            trainer_module.CorrectedLegacyDiagnosticTrainer._completed_epoch(fake),
            20,
        )
        for epoch, step in ((19.5, 78), (20.0, 79), (21.0, 84)):
            fake.state.epoch = epoch
            fake.state.global_step = step
            with self.subTest(epoch=epoch, step=step), self.assertRaises(RuntimeError):
                trainer_module.CorrectedLegacyDiagnosticTrainer._completed_epoch(fake)

        records = [
            {"epoch": epoch, "global_step": epoch * 4}
            for epoch in range(1, 21)
        ]
        history_fake = SimpleNamespace(_validation_records=records)
        self.assertEqual(
            trainer_module.CorrectedLegacyDiagnosticTrainer.validation_history(history_fake),
            tuple(records),
        )
        history_fake._validation_records[-1] = {"epoch": 20, "global_step": 79}
        with self.assertRaisesRegex(RuntimeError, "steps changed"):
            trainer_module.CorrectedLegacyDiagnosticTrainer.validation_history(
                history_fake
            )


if __name__ == "__main__":
    unittest.main()
