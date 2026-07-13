from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.determinism import (  # noqa: E402
    SMOKE_CELL,
    SMOKE_EPOCHS,
    SMOKE_GLOBAL_WINDOW_VALID_QUERIES,
    SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH,
    SMOKE_MODEL_STATE_PROTOCOL,
    SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH,
    SMOKE_SCHEDULE,
    SMOKE_TOTAL_MICROBATCH_RECORDS,
    SMOKE_TOTAL_OPTIMIZER_UPDATES,
    SMOKE_TOTAL_QUERY_LINKS,
    SMOKE_UPDATES_PER_EPOCH,
    SMOKE_WINDOW_MICROBATCHES,
    SMOKE_WORLD_SIZE,
    build_smoke_loss_trace_identity,
    build_smoke_microbatch_loss_record,
    build_smoke_scientific_evidence,
    canonical_model_state_identity,
    compare_smoke_scientific_evidence,
    decode_float32_scalar_bits,
    encode_float32_scalar_bits,
    validate_smoke_loss_trace_identity,
    validate_smoke_microbatch_loss_record,
    validate_smoke_scientific_evidence,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _model_identity(label: str) -> dict[str, object]:
    return {
        "protocol": SMOKE_MODEL_STATE_PROTOCOL,
        "tensor_count": 134,
        "sha256": _digest(label),
    }


def _synthetic_records() -> list[list[dict[str, object]]]:
    records: list[list[dict[str, object]]] = [[] for _ in range(SMOKE_WORLD_SIZE)]
    query_counter = [0 for _ in range(SMOKE_EPOCHS)]
    offsets = (0, 8, 16)
    for epoch in range(SMOKE_EPOCHS):
        for local_index in range(SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH):
            if local_index < offsets[1]:
                window = 0
            elif local_index < offsets[2]:
                window = 1
            else:
                window = 2
            within = local_index - offsets[window]
            for rank in range(SMOKE_WORLD_SIZE):
                local_count = 4 if local_index < 18 else (2 if rank < 2 else 1)
                query_ids = []
                for _ in range(local_count):
                    query_ids.append(f"epoch-{epoch}-query-{query_counter[epoch]:03d}")
                    query_counter[epoch] += 1
                trace_hashes = [_digest(f"trace:{query_id}") for query_id in query_ids]
                losses = torch.tensor(
                    [1.0 + epoch + rank / 10.0 + position / 100.0 for position in range(local_count)],
                    dtype=torch.float32,
                )
                records[rank].append(
                    build_smoke_microbatch_loss_record(
                        epoch=epoch,
                        rank=rank,
                        local_microbatch_index=local_index,
                        optimizer_window_index=window,
                        window_microbatch_index=within,
                        global_step_before=epoch * SMOKE_UPDATES_PER_EPOCH + window,
                        is_window_end=within == SMOKE_WINDOW_MICROBATCHES[window] - 1,
                        query_ids=query_ids,
                        candidate_trace_sha256=trace_hashes,
                        local_valid_query_count=local_count,
                        global_window_valid_query_count=(
                            SMOKE_GLOBAL_WINDOW_VALID_QUERIES[window]
                        ),
                        local_loss_sum=losses.sum(dtype=torch.float32),
                        scaled_loss=torch.tensor(
                            float(losses.sum().item())
                            * SMOKE_WORLD_SIZE
                            / SMOKE_GLOBAL_WINDOW_VALID_QUERIES[window],
                            dtype=torch.float32,
                        ),
                        per_query_losses=losses,
                        torch_module=torch,
                    )
                )
    if query_counter != [294, 294]:
        raise AssertionError(f"Synthetic smoke query coverage changed: {query_counter}")
    return records


class CanonicalModelStateIdentityTest(unittest.TestCase):
    def test_key_order_is_irrelevant_and_rng_is_untouched(self) -> None:
        first = OrderedDict(
            [
                ("weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
                ("counter", torch.tensor(7, dtype=torch.int64)),
            ]
        )
        second = OrderedDict(reversed(list(first.items())))
        rng_before = torch.random.get_rng_state().clone()
        first_identity = canonical_model_state_identity(first, torch, expected_tensor_count=2)
        rng_after = torch.random.get_rng_state()
        second_identity = canonical_model_state_identity(second, torch, expected_tensor_count=2)
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(first_identity["protocol"], SMOKE_MODEL_STATE_PROTOCOL)
        self.assertEqual(first_identity["tensor_count"], 2)

    def test_logically_equal_noncontiguous_tensor_hashes_identically(self) -> None:
        noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        self.assertFalse(noncontiguous.is_contiguous())
        contiguous = noncontiguous.contiguous()
        self.assertEqual(
            canonical_model_state_identity(
                {"weight": noncontiguous}, torch, expected_tensor_count=1
            ),
            canonical_model_state_identity(
                {"weight": contiguous}, torch, expected_tensor_count=1
            ),
        )

    def test_name_dtype_shape_and_one_value_change_identity(self) -> None:
        base = {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
        base_identity = canonical_model_state_identity(base, torch, expected_tensor_count=1)
        mutations = {
            "name": {"other": base["weight"].clone()},
            "dtype": {"weight": base["weight"].to(torch.float64)},
            "shape": {"weight": base["weight"].reshape(2, 1)},
            "value": {"weight": torch.tensor([[1.0, 2.0000002]], dtype=torch.float32)},
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    base_identity,
                    canonical_model_state_identity(
                        mutation, torch, expected_tensor_count=1
                    ),
                )

    def test_invalid_state_fails_loudly(self) -> None:
        with self.assertRaisesRegex(TypeError, "mapping"):
            canonical_model_state_identity([], torch, expected_tensor_count=1)
        with self.assertRaisesRegex(ValueError, "expected exactly 2"):
            canonical_model_state_identity(
                {"weight": torch.ones(1)}, torch, expected_tensor_count=2
            )
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            canonical_model_state_identity(
                {"weight": torch.tensor([math.nan])}, torch, expected_tensor_count=1
            )
        with self.assertRaisesRegex(ValueError, "strided"):
            canonical_model_state_identity(
                {"weight": torch.ones(2, 2).to_sparse()},
                torch,
                expected_tensor_count=1,
            )
        with self.assertRaisesRegex(ValueError, "meta"):
            canonical_model_state_identity(
                {"weight": torch.empty(1, device="meta")},
                torch,
                expected_tensor_count=1,
            )
        quantized = torch.quantize_per_tensor(
            torch.tensor([1.0]), scale=0.1, zero_point=0, dtype=torch.qint8
        )
        with self.assertRaisesRegex(ValueError, "quantized"):
            canonical_model_state_identity(
                {"weight": quantized}, torch, expected_tensor_count=1
            )


class Float32BitsTest(unittest.TestCase):
    def test_exact_encode_decode(self) -> None:
        self.assertEqual(encode_float32_scalar_bits(torch.tensor(1.0), torch), "3f800000")
        self.assertEqual(decode_float32_scalar_bits("3f800000"), 1.0)
        negative_zero = decode_float32_scalar_bits("80000000")
        self.assertEqual(negative_zero, 0.0)
        self.assertEqual(math.copysign(1.0, negative_zero), -1.0)

    def test_wrong_tensor_and_bits_fail(self) -> None:
        invalid_tensors = (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(float("nan"), dtype=torch.float32),
            True,
        )
        for value in invalid_tensors:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError, FloatingPointError)):
                    encode_float32_scalar_bits(value, torch)
        for value in ("3F800000", "3f80000", "zzzzzzzz", True, "7f800000"):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, FloatingPointError)):
                    decode_float32_scalar_bits(value)


class SmokeLossTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _synthetic_records()

    def test_complete_152_record_trace_has_exact_identity(self) -> None:
        identity = build_smoke_loss_trace_identity(self.records)
        self.assertEqual(identity["record_count"], SMOKE_TOTAL_MICROBATCH_RECORDS)
        self.assertEqual(identity["query_link_count"], SMOKE_TOTAL_QUERY_LINKS)
        self.assertEqual(
            [record["query_link_count"] for record in identity["rank_traces"]],
            [
                SMOKE_EPOCHS * count
                for count in SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH
            ],
        )
        self.assertEqual(validate_smoke_loss_trace_identity(identity), identity)

    def test_record_builder_binds_all_window_coordinates(self) -> None:
        record = self.records[0][0]
        self.assertEqual(validate_smoke_microbatch_loss_record(record), record)
        mutations = {
            "window": {"optimizer_window_index": 1},
            "within": {"window_microbatch_index": 1},
            "step": {"global_step_before": 1},
            "end": {"is_window_end": True},
            "global_count": {"global_window_valid_query_count": 38},
            "local_count": {"local_valid_query_count": 3},
            "nonfinite": {"scaled_loss_float32_bits": "7f800000"},
        }
        for name, changes in mutations.items():
            changed = copy.deepcopy(record)
            changed.update(changes)
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError, FloatingPointError)):
                    validate_smoke_microbatch_loss_record(changed)

    def test_missing_duplicate_and_noncanonical_order_fail(self) -> None:
        missing = copy.deepcopy(self.records)
        missing[0].pop()
        with self.assertRaisesRegex(ValueError, "38"):
            build_smoke_loss_trace_identity(missing)

        reordered = copy.deepcopy(self.records)
        reordered[0][0], reordered[0][1] = reordered[0][1], reordered[0][0]
        with self.assertRaisesRegex(ValueError, "canonical"):
            build_smoke_loss_trace_identity(reordered)

        duplicate = copy.deepcopy(self.records)
        duplicate[1][0]["query_ids"][0] = duplicate[0][0]["query_ids"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_smoke_loss_trace_identity(duplicate)

        duplicate_trace = copy.deepcopy(self.records)
        duplicate_trace[1][0]["candidate_trace_sha256"][0] = (
            duplicate_trace[0][0]["candidate_trace_sha256"][0]
        )
        with self.assertRaisesRegex(ValueError, "candidate-trace"):
            build_smoke_loss_trace_identity(duplicate_trace)

    def test_builder_rejects_nonfinite_or_wrong_per_query_loss(self) -> None:
        template = self.records[0][0]
        common = {
            key: template[key]
            for key in (
                "epoch",
                "rank",
                "local_microbatch_index",
                "optimizer_window_index",
                "window_microbatch_index",
                "global_step_before",
                "is_window_end",
                "query_ids",
                "candidate_trace_sha256",
                "local_valid_query_count",
                "global_window_valid_query_count",
            )
        }
        for losses in (
            torch.tensor([1.0, 1.0, 1.0, float("nan")], dtype=torch.float32),
            torch.ones(4, dtype=torch.float64),
            torch.ones(2, 2, dtype=torch.float32),
        ):
            with self.subTest(dtype=losses.dtype, shape=tuple(losses.shape)):
                with self.assertRaises((TypeError, ValueError, FloatingPointError)):
                    build_smoke_microbatch_loss_record(
                        **common,
                        local_loss_sum=torch.tensor(4.0),
                        scaled_loss=torch.tensor(0.125),
                        per_query_losses=losses,
                        torch_module=torch,
                    )


class SmokeScientificEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = _synthetic_records()
        cls.loss_identity = build_smoke_loss_trace_identity(cls.records)

    def make_evidence(self, **overrides: object) -> dict[str, object]:
        selected = _model_identity("selected")
        values: dict[str, object] = {
            "initial_model_state": _model_identity("initial"),
            "last_model_state": _model_identity("last"),
            "selected_model_state": selected,
            "roundtrip_model_state": copy.deepcopy(selected),
            "candidate_traces": {
                "manifest_sha256": _digest("candidate manifest"),
                "merged_sha256": _digest("candidate merged"),
                "record_count": 588,
                "rank_shards": [
                    {
                        "rank": rank,
                        "record_count": 2 * SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH[rank],
                        "sha256": _digest(f"candidate rank {rank}"),
                    }
                    for rank in range(4)
                ],
            },
            "loss_traces": copy.deepcopy(self.loss_identity),
            "validation_selection": {"epochs": 2, "sha256": _digest("validation")},
            "reload": {
                "validation_sha256": _digest("reload validation"),
                "scheduler_state_sha256": _digest("reload scheduler"),
                "client_state_sha256": _digest("reload client"),
                "per_rank_rng_sha256": [_digest(f"rng {rank}") for rank in range(4)],
            },
            "final_artifacts": {
                "model_sha256": _digest("model file"),
                "tokenizer_inventory_sha256": _digest("tokenizer"),
                "encoder_config_sha256": _digest("encoder config"),
                "wrapper_config_sha256": _digest("wrapper"),
            },
            "launch_ledger": {"sha256": _digest("launch ledger")},
        }
        values.update(overrides)
        return build_smoke_scientific_evidence(**values)

    def test_exact_evidence_roundtrip_and_comparison_receipt(self) -> None:
        first = self.make_evidence()
        serialized = json.dumps(first, sort_keys=True)
        second = json.loads(serialized)
        self.assertEqual(validate_smoke_scientific_evidence(second), first)
        receipt = compare_smoke_scientific_evidence(first, second)
        self.assertTrue(receipt["exact_match"])
        self.assertEqual(receipt["replicas"], 2)
        self.assertEqual(receipt["scientific_identity_sha256"], first["sha256"])

    def test_every_scientific_component_mismatch_fails(self) -> None:
        first = self.make_evidence()
        alternatives: dict[str, dict[str, object]] = {
            "initial": self.make_evidence(initial_model_state=_model_identity("initial changed")),
            "last": self.make_evidence(last_model_state=_model_identity("last changed")),
            "selected": self.make_evidence(
                selected_model_state=_model_identity("selected changed"),
                roundtrip_model_state=_model_identity("selected changed"),
            ),
        }
        candidate = copy.deepcopy(first["candidate_traces"])
        candidate["rank_shards"][2]["sha256"] = _digest("candidate rank 2 changed")
        alternatives["candidate"] = self.make_evidence(candidate_traces=candidate)

        changed_records = copy.deepcopy(self.records)
        changed_records[3][37]["scaled_loss_float32_bits"] = encode_float32_scalar_bits(
            torch.tensor(9.5, dtype=torch.float32), torch
        )
        alternatives["loss"] = self.make_evidence(
            loss_traces=build_smoke_loss_trace_identity(changed_records)
        )
        alternatives["validation"] = self.make_evidence(
            validation_selection={"epochs": 2, "sha256": _digest("validation changed")}
        )
        changed_reload = copy.deepcopy(first["reload"])
        changed_reload["per_rank_rng_sha256"][1] = _digest("rng changed")
        alternatives["reload"] = self.make_evidence(reload=changed_reload)
        changed_final = copy.deepcopy(first["final_artifacts"])
        changed_final["wrapper_config_sha256"] = _digest("wrapper changed")
        alternatives["final"] = self.make_evidence(final_artifacts=changed_final)
        alternatives["ledger"] = self.make_evidence(
            launch_ledger={"sha256": _digest("ledger changed")}
        )

        for name, second in alternatives.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    compare_smoke_scientific_evidence(first, second)

    def test_selected_and_roundtrip_must_be_identical(self) -> None:
        with self.assertRaisesRegex(ValueError, "round-trip"):
            self.make_evidence(roundtrip_model_state=_model_identity("different roundtrip"))

    def test_cell_schedule_and_path_metadata_are_closed(self) -> None:
        evidence = self.make_evidence()
        attacks = []
        changed_cell = copy.deepcopy(evidence)
        changed_cell["cell"]["outer_fold"] = 1
        attacks.append(changed_cell)
        changed_schedule = copy.deepcopy(evidence)
        changed_schedule["schedule"]["epochs"] = 20
        attacks.append(changed_schedule)
        type_confused_schedule = copy.deepcopy(evidence)
        type_confused_schedule["schedule"]["window_microbatches"][0] = 8.0
        attacks.append(type_confused_schedule)
        absolute_path = copy.deepcopy(evidence)
        absolute_path["output_path"] = "/opt/ml/model"
        attacks.append(absolute_path)
        replica = copy.deepcopy(evidence)
        replica["replica_id"] = "a"
        attacks.append(replica)
        for attack in attacks:
            with self.subTest(keys=sorted(attack)):
                with self.assertRaises((TypeError, ValueError)):
                    validate_smoke_scientific_evidence(attack)

    def test_fixed_constants_are_exact(self) -> None:
        self.assertEqual(dict(SMOKE_CELL), {
            "outer_fold": 0,
            "query_view": "structured",
            "sampler": "global_uniform",
            "experiment_seed": 17,
        })
        self.assertEqual(SMOKE_EPOCHS, 2)
        self.assertEqual(SMOKE_UPDATES_PER_EPOCH, 3)
        self.assertEqual(SMOKE_TOTAL_OPTIMIZER_UPDATES, 6)
        self.assertEqual(SMOKE_WORLD_SIZE, 4)
        self.assertEqual(SMOKE_WINDOW_MICROBATCHES, (8, 8, 3))
        self.assertEqual(tuple(SMOKE_SCHEDULE["window_microbatches"]), (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
