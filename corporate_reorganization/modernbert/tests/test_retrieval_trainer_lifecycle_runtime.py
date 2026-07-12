from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Trainer, TrainerState, TrainingArguments
from transformers.trainer_callback import DefaultFlowCallback, TrainerControl


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import train_sm as controlled_train  # noqa: E402
import trainer as trainer_module  # noqa: E402
from retriever.checkpointing import CheckpointSelection  # noqa: E402
from retriever.evaluation import (  # noqa: E402
    VALIDATION_PRIMARY_METRIC,
    VALIDATION_SECONDARY_METRIC,
)
from trainer import ControlledRetrievalTrainer  # noqa: E402


def _selection(epoch: int) -> CheckpointSelection:
    global_step = epoch * 3
    return CheckpointSelection(
        schema_version=1,
        epoch=epoch,
        global_step=global_step,
        checkpoint_dir=f"checkpoint-{global_step}",
        deepspeed_tag=f"global_step{global_step}",
        primary_metric=0.5,
        secondary_metric=0.25,
        ranking_sha256=hashlib.sha256(f"ranking-{epoch}".encode()).hexdigest(),
    )


class TrainerHookContractTest(unittest.TestCase):
    def test_final_step_save_is_deferred_until_epoch_end_validation(self) -> None:
        completed_epoch = mock.Mock(return_value=20)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                report_to=[],
            )
            state = TrainerState(global_step=60, max_steps=60, epoch=20.0)
            flow = DefaultFlowCallback()
            control = flow.on_step_end(args, state, TrainerControl())
            self.assertTrue(control.should_training_stop)
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_evaluate)
            fake = SimpleNamespace(
                EXPECTED_EPOCHS=20,
                EXPECTED_TOTAL_UPDATES=60,
                state=state,
                control=control,
                _completed_epoch_number=completed_epoch,
            )
            with mock.patch.object(
                Trainer,
                "_maybe_log_save_evaluate",
                autospec=True,
            ) as parent:
                ControlledRetrievalTrainer._maybe_log_save_evaluate(
                    fake,
                    torch.tensor(1.0),
                    None,
                    object(),
                    None,
                    19,
                    None,
                    0.0,
                )
                self.assertFalse(fake.control.should_save)
                completed_epoch.assert_not_called()
                self.assertEqual(parent.call_count, 1)

                fake.control = flow.on_epoch_end(args, state, fake.control)
                self.assertTrue(fake.control.should_save)
                self.assertTrue(fake.control.should_evaluate)
                ControlledRetrievalTrainer._maybe_log_save_evaluate(
                    fake,
                    torch.tensor(1.0),
                    None,
                    object(),
                    None,
                    19,
                    None,
                    0.0,
                )
                completed_epoch.assert_called_once_with()
                self.assertEqual(parent.call_count, 2)

                fake.state.global_step = 57
                fake.state.epoch = 19.0
                fake.control.should_training_stop = False
                fake.control.should_evaluate = False
                with self.assertRaisesRegex(RuntimeError, "flags diverged"):
                    ControlledRetrievalTrainer._maybe_log_save_evaluate(
                        fake,
                        torch.tensor(1.0),
                        None,
                        object(),
                        None,
                        18,
                        None,
                        0.0,
                    )

    def test_evaluate_keeps_canonical_metrics_outside_mutating_log(self) -> None:
        canonical_metrics = MappingProxyType(
            {
                VALIDATION_PRIMARY_METRIC: 0.5,
                VALIDATION_SECONDARY_METRIC: 0.25,
            }
        )
        result = SimpleNamespace(metrics=canonical_metrics)
        logged: list[dict[str, float]] = []

        def mutating_log(metrics):
            metrics["epoch"] = 1.0
            logged.append(metrics)

        def on_evaluate(args, state, control, metrics):
            del args, state, metrics
            control.should_evaluate = False
            return control

        engine = object()
        fake = SimpleNamespace(
            eval_dataset=object(),
            model_wrapped=engine,
            model=object(),
            deepspeed=engine,
            _completed_epoch_number=lambda: 1,
            _evaluated_epochs=set(),
            _pending_validation_result=None,
            _pending_selection=None,
            _run_controlled_validation=lambda model: result,
            log=mutating_log,
            callback_handler=SimpleNamespace(on_evaluate=on_evaluate),
            args=object(),
            state=SimpleNamespace(),
            control=TrainerControl(should_evaluate=True),
            is_world_process_zero=lambda: False,
        )
        returned = ControlledRetrievalTrainer.evaluate(fake)
        self.assertEqual(returned, dict(canonical_metrics))
        self.assertNotIn("epoch", returned)
        self.assertNotIn("epoch", result.metrics)
        self.assertEqual(logged[0]["epoch"], 1.0)
        self.assertIs(fake._pending_validation_result, result)
        self.assertFalse(fake.control.should_evaluate)

    def test_trainer_control_state_is_replaced_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            selection = _selection(1)
            control = TrainerControl(should_save=True)
            state = TrainerState(
                global_step=selection.global_step,
                epoch=float(selection.epoch),
                best_metric=selection.primary_metric,
                best_model_checkpoint=str(Path(tmp_dir) / selection.checkpoint_dir),
            )
            fake = SimpleNamespace(
                EXPECTED_EPOCHS=20,
                _pending_selection=selection,
                _pending_best=selection,
                store_flos=mock.Mock(),
                callback_handler=SimpleNamespace(callbacks=[]),
                control=control,
                state=state,
                args=SimpleNamespace(output_dir=tmp_dir),
            )
            ControlledRetrievalTrainer._prepare_trainer_state_for_checkpoint(fake)
            first = state.stateful_callbacks
            self.assertEqual(set(first), {"TrainerControl"})
            self.assertIsInstance(first["TrainerControl"], dict)

            ControlledRetrievalTrainer._prepare_trainer_state_for_checkpoint(fake)
            self.assertEqual(state.stateful_callbacks, first)
            self.assertNotIsInstance(state.stateful_callbacks["TrainerControl"], list)
            state_path = Path(tmp_dir) / "trainer_state.json"
            state.save_to_json(str(state_path))
            loaded = TrainerState.load_from_json(str(state_path))
            self.assertEqual(loaded.global_step, selection.global_step)
            self.assertEqual(loaded.epoch, float(selection.epoch))
            self.assertEqual(loaded.best_metric, selection.primary_metric)
            self.assertEqual(loaded.best_model_checkpoint, state.best_model_checkpoint)
            self.assertEqual(loaded.stateful_callbacks, state.stateful_callbacks)
            common_rank_zero = trainer_module._common_trainer_state_sha256(
                loaded,
                rank=0,
            )
            loaded.is_local_process_zero = False
            loaded.is_world_process_zero = False
            common_rank_one = trainer_module._common_trainer_state_sha256(
                loaded,
                rank=1,
            )
            self.assertEqual(common_rank_zero, common_rank_one)
            with self.assertRaisesRegex(RuntimeError, "rank flags changed"):
                trainer_module._common_trainer_state_sha256(loaded, rank=0)

            final_selection = _selection(20)
            fake._pending_selection = final_selection
            fake._pending_best = final_selection
            fake.state.global_step = final_selection.global_step
            fake.state.epoch = float(final_selection.epoch)
            fake.state.best_metric = final_selection.primary_metric
            fake.state.best_model_checkpoint = str(
                Path(tmp_dir) / final_selection.checkpoint_dir
            )
            fake.control.should_training_stop = True
            ControlledRetrievalTrainer._prepare_trainer_state_for_checkpoint(fake)
            self.assertTrue(
                fake.state.stateful_callbacks["TrainerControl"]["args"][
                    "should_training_stop"
                ]
            )


class _ReleaseAccelerator:
    def __init__(self, *, rank: int, fail: bool) -> None:
        self.rank = rank
        self.fail = fail
        self.deepspeed_engine_wrapped = object()
        self._models = [object()]
        self._optimizers = [object()]
        self._schedulers = [object()]
        self._dataloaders = [object()]
        self.free_arguments = None

    @staticmethod
    def wait_for_everyone() -> None:
        dist.barrier()

    def free_memory(self, *objects) -> None:
        self.free_arguments = objects
        if self.fail:
            raise RuntimeError(f"injected release failure rank={self.rank}")
        self.deepspeed_engine_wrapped = None
        self._models = []
        self._optimizers = []
        self._schedulers = []
        self._dataloaders = []


def _release_worker(rank: int, init_file: str, inject_failure: bool, queue) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{init_file}",
    )
    engine = object()
    model = object()
    optimizer = object()
    scheduler = object()
    accelerator = _ReleaseAccelerator(rank=rank, fail=inject_failure and rank == 1)
    callback_handler = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=object(),
        eval_dataloader=object(),
    )
    fake = SimpleNamespace(
        deepspeed=engine,
        model_wrapped=engine,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        callback_handler=callback_handler,
        accelerator=accelerator,
    )
    try:
        ControlledRetrievalTrainer.release_current_deepspeed_engine(fake)
        queue.put(
            (
                rank,
                "ok",
                fake.deepspeed is None
                and fake.model is None
                and callback_handler.train_dataloader is None
                and accelerator.free_arguments == (engine, model, optimizer, scheduler),
            )
        )
    except BaseException as error:
        queue.put((rank, "error", str(error)))
    finally:
        dist.destroy_process_group()


class CollectiveReleaseTest(unittest.TestCase):
    def _run_release(self, *, inject_failure: bool):
        context = mp.get_context("spawn")
        queue = context.Queue()
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_file = str(Path(tmp_dir) / "gloo-init")
            processes = [
                context.Process(
                    target=_release_worker,
                    args=(rank, init_file, inject_failure, queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertFalse(process.is_alive(), "Collective release worker hung")
                self.assertEqual(process.exitcode, 0)
            return sorted(queue.get(timeout=5) for _ in processes)

    def test_release_success_and_rank_local_failure_are_collective(self) -> None:
        success = self._run_release(inject_failure=False)
        self.assertEqual(success, [(0, "ok", True), (1, "ok", True)])
        failure = self._run_release(inject_failure=True)
        self.assertEqual([record[1] for record in failure], ["error", "error"])
        for _, _, message in failure:
            self.assertIn("failed collectively", message)
            self.assertIn("injected release failure rank=1", message)


class FreshEngineAndArtifactHelperTest(unittest.TestCase):
    def test_fresh_engine_rejects_partitioned_model_and_keeps_scheduler_external(self) -> None:
        fresh_model = torch.nn.Linear(2, 2)
        optimizer = object()
        scheduler = SimpleNamespace(step=lambda: None)
        prepared_optimizer = object()

        class Engine:
            module = fresh_model
            global_steps = 0
            lr_scheduler = None
            dp_world_size = 4

            @staticmethod
            def zero_optimization_stage() -> int:
                return 3

            @staticmethod
            def bfloat16_enabled() -> bool:
                return True

        engine = Engine()
        accelerator = SimpleNamespace(prepare=lambda model, opt: (engine, prepared_optimizer))
        fake = SimpleNamespace(
            EXPECTED_TOTAL_UPDATES=60,
            EXPECTED_WORLD_SIZE=4,
            _checkpoint_manifest={"complete": True},
            _validation_metadata_store=SimpleNamespace(best=object()),
            _engine_generation=1,
            deepspeed=None,
            model=None,
            model_wrapped=None,
            optimizer=None,
            lr_scheduler=None,
            callback_handler=SimpleNamespace(model=None, optimizer=None, lr_scheduler=None),
            accelerator=accelerator,
        )
        with mock.patch.object(
            trainer_module,
            "_coordinated_local_operation",
            side_effect=lambda context, operation: operation(),
        ), mock.patch(
            "transformers.integrations.deepspeed.deepspeed_init",
            return_value=(optimizer, scheduler),
        ) as deepspeed_init:
            ControlledRetrievalTrainer.prepare_fresh_deepspeed_engine(fake, fresh_model)
        deepspeed_init.assert_called_once_with(fake, num_training_steps=60)
        self.assertIs(fake.deepspeed, engine)
        self.assertIs(fake.lr_scheduler, scheduler)
        self.assertIsNone(engine.lr_scheduler)
        self.assertEqual(fake._engine_generation, 2)

        partitioned_model = torch.nn.Linear(2, 2)
        partitioned_model.weight.ds_id = 1
        fake.model = fake.model_wrapped = fake.deepspeed = None
        fake._engine_generation = 1
        with mock.patch.object(
            trainer_module,
            "_coordinated_local_operation",
            side_effect=lambda context, operation: operation(),
        ), mock.patch(
            "transformers.integrations.deepspeed.deepspeed_init",
            return_value=(optimizer, scheduler),
        ):
            with self.assertRaisesRegex(RuntimeError, "already ZeRO-partitioned"):
                ControlledRetrievalTrainer.prepare_fresh_deepspeed_engine(
                    fake,
                    partitioned_model,
                )

    def test_bf16_state_and_atomic_artifact_helpers_fail_loudly(self) -> None:
        state = {
            "weight": torch.ones((2, 2), dtype=torch.bfloat16),
            "counter": torch.tensor(1, dtype=torch.long),
        }
        self.assertIs(
            controlled_train._validate_gathered_bf16_state_dict(state, torch),
            state,
        )
        with self.assertRaisesRegex(RuntimeError, "expected BF16"):
            controlled_train._validate_gathered_bf16_state_dict(
                {"weight": torch.ones((2, 2), dtype=torch.float32)},
                torch,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            binary = controlled_train._publish_new_binary(
                root / "model.safetensors",
                lambda path: path.write_bytes(b"exact-binary"),
            )
            self.assertEqual(binary["sha256"], hashlib.sha256(b"exact-binary").hexdigest())
            with self.assertRaises(FileExistsError):
                controlled_train._publish_new_binary(
                    root / "model.safetensors",
                    lambda path: path.write_bytes(b"changed"),
                )

            directory = controlled_train._publish_pretrained_directory(
                root / "tokenizer",
                lambda path: (path / "tokenizer.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                ),
            )
            self.assertEqual(directory["files"][0]["path"], "tokenizer.json")
            self.assertFalse((root / ".tokenizer.incomplete").exists())


if __name__ == "__main__":
    unittest.main()
