from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import trainer as trainer_module  # noqa: E402
from retriever.determinism import (  # noqa: E402
    SMOKE_GLOBAL_WINDOW_VALID_QUERIES,
    SMOKE_MODEL_STATE_PROTOCOL,
    SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH,
    SMOKE_WINDOW_MICROBATCHES,
    build_smoke_microbatch_loss_record,
    canonical_model_state_identity,
)
from trainer import (  # noqa: E402
    ControlledRetrievalTrainer,
    DETERMINISM_SMOKE_TRAINING_SCHEDULE,
    DeterminismSmokeRetrievalTrainer,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_identity(label: str) -> dict[str, object]:
    return {
        "protocol": SMOKE_MODEL_STATE_PROTOCOL,
        "tensor_count": 134,
        "sha256": _digest(label),
    }


def _bare_smoke_trainer() -> DeterminismSmokeRetrievalTrainer:
    value = object.__new__(DeterminismSmokeRetrievalTrainer)
    value.training_schedule = DETERMINISM_SMOKE_TRAINING_SCHEDULE
    value._smoke_initial_model_state = None
    value._smoke_loss_records = []
    value._smoke_pending_loss = None
    value._smoke_capture_active = False
    value.state = SimpleNamespace(global_step=0)
    return value


def _window_for_index(index: int) -> tuple[int, int, bool]:
    offset = 0
    for window, count in enumerate(SMOKE_WINDOW_MICROBATCHES):
        if index < offset + count:
            within = index - offset
            return window, within, within == count - 1
        offset += count
    raise AssertionError(index)


def _local_count(rank: int, local_index: int) -> int:
    if local_index < 18:
        return 4
    return 2 if rank < 2 else 1


def _synthetic_gathered_records() -> list[list[dict[str, object]]]:
    records: list[list[dict[str, object]]] = [[] for _ in range(4)]
    query_position = [0, 0]
    for epoch in range(2):
        for local_index in range(19):
            window, within, is_end = _window_for_index(local_index)
            for rank in range(4):
                count = _local_count(rank, local_index)
                query_ids = []
                for _ in range(count):
                    query_ids.append(f"e{epoch}-q{query_position[epoch]:03d}")
                    query_position[epoch] += 1
                losses = torch.arange(1, count + 1, dtype=torch.float32)
                records[rank].append(
                    build_smoke_microbatch_loss_record(
                        epoch=epoch,
                        rank=rank,
                        local_microbatch_index=local_index,
                        optimizer_window_index=window,
                        window_microbatch_index=within,
                        global_step_before=epoch * 3 + window,
                        is_window_end=is_end,
                        query_ids=query_ids,
                        candidate_trace_sha256=[
                            _digest(f"candidate:{query_id}") for query_id in query_ids
                        ],
                        local_valid_query_count=count,
                        global_window_valid_query_count=(
                            SMOKE_GLOBAL_WINDOW_VALID_QUERIES[window]
                        ),
                        local_loss_sum=losses.sum(),
                        scaled_loss=torch.tensor(0.25, dtype=torch.float32),
                        per_query_losses=losses,
                        torch_module=torch,
                    )
                )
    if query_position != [294, 294]:
        raise AssertionError(query_position)
    return records


class EngineStateCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = object()
        self.trainer = _bare_smoke_trainer()
        self.trainer.deepspeed = self.engine
        self.trainer.model_wrapped = self.engine

    @staticmethod
    def _bf16_state() -> dict[str, torch.Tensor]:
        return {
            f"tensor_{index:03d}": torch.tensor(float(index), dtype=torch.bfloat16)
            for index in range(134)
        }

    def test_rank_zero_requires_and_hashes_exact_cpu_bf16_state(self) -> None:
        state = self._bf16_state()
        expected = canonical_model_state_identity(state, torch)
        self.trainer.accelerator = SimpleNamespace(
            get_state_dict=mock.Mock(return_value=state)
        )
        with (
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=4),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=0),
            mock.patch.object(
                trainer_module.dist, "broadcast_object_list", return_value=None
            ) as broadcast,
            mock.patch.object(
                trainer_module,
                "_coordinated_local_operation",
                side_effect=lambda _context, operation: operation(),
            ),
        ):
            actual = self.trainer._capture_engine_model_state(
                self.engine, context="test capture"
            )
        self.assertEqual(actual, expected)
        self.trainer.accelerator.get_state_dict.assert_called_once_with(self.engine)
        broadcast.assert_called_once()

        fp32 = dict(state)
        fp32["tensor_000"] = fp32["tensor_000"].float()
        self.trainer.accelerator.get_state_dict = mock.Mock(return_value=fp32)
        with (
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=4),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=0),
            mock.patch.object(
                trainer_module,
                "_coordinated_local_operation",
                side_effect=lambda _context, operation: operation(),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "expected BF16"):
                self.trainer._capture_engine_model_state(
                    self.engine, context="test capture"
                )

    def test_nonzero_rank_rejects_a_gathered_state_and_accepts_rank_zero_broadcast(self) -> None:
        expected = _model_identity("rank-zero")

        def broadcast(values: list[object], *, src: int) -> None:
            self.assertEqual(src, 0)
            values[0] = expected

        with (
            mock.patch.object(trainer_module.dist, "get_world_size", return_value=4),
            mock.patch.object(trainer_module.dist, "get_rank", return_value=1),
            mock.patch.object(
                trainer_module,
                "_coordinated_local_operation",
                side_effect=lambda _context, operation: operation(),
            ),
            mock.patch.object(
                trainer_module.dist, "broadcast_object_list", side_effect=broadcast
            ),
        ):
            self.trainer.accelerator = SimpleNamespace(
                get_state_dict=mock.Mock(return_value=self._bf16_state())
            )
            with self.assertRaisesRegex(RuntimeError, "unexpectedly returned"):
                self.trainer._capture_engine_model_state(
                    self.engine, context="nonzero capture"
                )

            self.trainer.accelerator = SimpleNamespace(
                get_state_dict=mock.Mock(return_value=None)
            )
            self.assertEqual(
                self.trainer._capture_engine_model_state(
                    self.engine, context="nonzero capture"
                ),
                expected,
            )


class SmokeLossLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = _bare_smoke_trainer()
        self.engine = object()

    @staticmethod
    def _parent_training_step(self, model, inputs, num_items_in_batch=None):
        del model, num_items_in_batch
        if self._smoke_capture_active is not True:
            raise AssertionError("smoke capture was not active during parent training_step")
        traces = inputs["sampling_traces"]
        count = len(traces)
        losses = torch.arange(1, count + 1, dtype=torch.float32)
        self._smoke_pending_loss = {
            "local_loss_sum": losses.sum(),
            "scaled_loss": torch.tensor(0.5, dtype=torch.float32),
            "per_query_losses": losses,
            "local_valid_query_count": count,
            "global_window_valid_query_count": inputs["global_window_valid_count"],
        }
        return torch.tensor(0.5, dtype=torch.float32)

    def test_initial_capture_once_and_all_38_records_have_exact_coordinates(self) -> None:
        capture = mock.Mock(return_value=_model_identity("initial"))
        self.trainer._capture_engine_model_state = capture
        query_counter = 0
        with (
            mock.patch.object(trainer_module.dist, "get_rank", return_value=0),
            mock.patch.object(
                ControlledRetrievalTrainer,
                "training_step",
                new=self._parent_training_step,
            ),
        ):
            for epoch in range(2):
                for local_index in range(19):
                    window, within, is_end = _window_for_index(local_index)
                    count = _local_count(0, local_index)
                    traces = []
                    for _ in range(count):
                        query_id = f"epoch-{epoch}-query-{query_counter:03d}"
                        query_counter += 1
                        traces.append(
                            {
                                "epoch": epoch,
                                "query_id": query_id,
                                "trace_sha256": _digest(f"trace:{query_id}"),
                            }
                        )
                    inputs = {
                        "sampling_traces": traces,
                        "is_window_end": is_end,
                        "global_window_valid_count": (
                            SMOKE_GLOBAL_WINDOW_VALID_QUERIES[window]
                        ),
                    }
                    returned = DeterminismSmokeRetrievalTrainer.training_step(
                        self.trainer,
                        self.engine,
                        inputs,
                        num_items_in_batch=SMOKE_GLOBAL_WINDOW_VALID_QUERIES[window],
                    )
                    self.assertEqual(returned.item(), 0.5)
                    if is_end:
                        self.trainer.state.global_step += 1

        capture.assert_called_once_with(
            self.engine,
            context="Smoke initial Engine-A state capture",
        )
        self.assertEqual(self.trainer.state.global_step, 6)
        self.assertEqual(len(self.trainer._smoke_loss_records), 38)
        self.assertEqual(
            sum(record["local_valid_query_count"] for record in self.trainer._smoke_loss_records),
            2 * SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH[0],
        )
        for position, record in enumerate(self.trainer._smoke_loss_records):
            epoch, local_index = divmod(position, 19)
            window, within, is_end = _window_for_index(local_index)
            self.assertEqual(record["epoch"], epoch)
            self.assertEqual(record["rank"], 0)
            self.assertEqual(record["local_microbatch_index"], local_index)
            self.assertEqual(record["optimizer_window_index"], window)
            self.assertEqual(record["window_microbatch_index"], within)
            self.assertEqual(record["global_step_before"], epoch * 3 + window)
            self.assertIs(record["is_window_end"], is_end)

    def test_compute_loss_captures_once_and_rejects_a_pending_prior_loss(self) -> None:
        self.trainer._smoke_capture_active = True
        losses = torch.tensor([1.0, 2.0], dtype=torch.float32)
        outputs = {
            "loss": torch.tensor(0.25, dtype=torch.float32),
            "local_loss_sum": losses.sum(),
            "per_query_loss": losses,
            "local_valid_query_count": 2,
            "global_window_valid_count": 38,
            "global_unique_passage_count": 10,
        }

        def parent_compute(*_args, **kwargs):
            self.assertIs(kwargs["return_outputs"], True)
            return outputs["loss"], outputs

        with mock.patch.object(
            ControlledRetrievalTrainer,
            "compute_loss",
            side_effect=parent_compute,
        ):
            actual = DeterminismSmokeRetrievalTrainer.compute_loss(
                self.trainer,
                self.engine,
                {},
                num_items_in_batch=38,
            )
        self.assertTrue(torch.equal(actual, outputs["loss"]))
        self.assertEqual(set(self.trainer._smoke_pending_loss), {
            "local_loss_sum",
            "scaled_loss",
            "per_query_losses",
            "local_valid_query_count",
            "global_window_valid_query_count",
        })
        with self.assertRaisesRegex(RuntimeError, "prior microbatch"):
            DeterminismSmokeRetrievalTrainer.compute_loss(
                self.trainer,
                self.engine,
                {},
                num_items_in_batch=38,
            )

    def test_training_step_rejects_missing_loss_components_and_nonzero_initial_capture(self) -> None:
        self.trainer._smoke_initial_model_state = _model_identity("initial")
        traces = [
            {
                "epoch": 0,
                "query_id": f"q{index}",
                "trace_sha256": _digest(f"trace:{index}"),
            }
            for index in range(4)
        ]

        def parent_without_capture(_self, _model, _inputs, num_items_in_batch=None):
            del num_items_in_batch
            return torch.tensor(1.0, dtype=torch.float32)

        with (
            mock.patch.object(trainer_module.dist, "get_rank", return_value=0),
            mock.patch.object(
                ControlledRetrievalTrainer,
                "training_step",
                new=parent_without_capture,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not capture"):
                DeterminismSmokeRetrievalTrainer.training_step(
                    self.trainer,
                    self.engine,
                    {
                        "sampling_traces": traces,
                        "is_window_end": False,
                        "global_window_valid_count": 128,
                    },
                    num_items_in_batch=128,
                )

        other = _bare_smoke_trainer()
        other.state.global_step = 1
        other._capture_engine_model_state = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "captured first"):
            DeterminismSmokeRetrievalTrainer.training_step(
                other,
                self.engine,
                {"sampling_traces": traces},
                num_items_in_batch=128,
            )
        other._capture_engine_model_state.assert_not_called()

    def test_last_capture_requires_step_six(self) -> None:
        self.trainer.deepspeed = self.engine
        self.trainer._capture_engine_model_state = mock.Mock(
            return_value=_model_identity("last")
        )
        self.trainer.state.global_step = 5
        with self.assertRaisesRegex(RuntimeError, "six updates"):
            self.trainer.capture_smoke_last_model_state()
        self.trainer._capture_engine_model_state.assert_not_called()

        self.trainer.state.global_step = 6
        self.assertEqual(
            self.trainer.capture_smoke_last_model_state(),
            _model_identity("last"),
        )
        self.trainer._capture_engine_model_state.assert_called_once_with(
            self.engine,
            context="Smoke last Engine-A state capture",
        )

    def test_four_rank_finalization_accepts_complete_and_rejects_malformed_gathers(self) -> None:
        gathered = _synthetic_gathered_records()
        self.trainer._smoke_loss_records = list(gathered[0])

        def complete_gather(output, _local):
            output[:] = gathered

        with mock.patch.object(
            trainer_module.dist,
            "all_gather_object",
            side_effect=complete_gather,
        ):
            package = self.trainer.finalize_smoke_loss_traces()
        self.assertEqual(package["per_rank_records"], gathered)
        self.assertEqual(package["identity"]["record_count"], 152)
        self.assertEqual(package["identity"]["query_link_count"], 588)

        malformed = [list(records) for records in gathered]
        malformed[2] = malformed[2][:-1]

        def malformed_gather(output, _local):
            output[:] = malformed

        with mock.patch.object(
            trainer_module.dist,
            "all_gather_object",
            side_effect=malformed_gather,
        ):
            with self.assertRaisesRegex(ValueError, "38"):
                self.trainer.finalize_smoke_loss_traces()

        self.trainer._smoke_pending_loss = {"unconsumed": True}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.trainer.finalize_smoke_loss_traces()


if __name__ == "__main__":
    unittest.main()
