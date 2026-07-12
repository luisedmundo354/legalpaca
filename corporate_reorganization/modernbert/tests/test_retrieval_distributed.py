from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.data import CorpusPassage, PassageIndexTable, load_corpus  # noqa: E402
from retriever.distributed import (  # noqa: E402
    build_global_candidate_plan,
    gather_owned_embeddings,
)
from retriever.losses import (  # noqa: E402
    build_index_positive_mask,
    masked_multi_positive_nce_loss_sum,
)
from retriever.sampling import (  # noqa: E402
    SELECTION_ALGORITHM,
    TRACE_SCHEMA_VERSION,
    sampling_trace_checksum,
)
import retriever.traces as trace_module  # noqa: E402
from retriever.traces import CandidateTraceStore  # noqa: E402


def _passage_features(indices: torch.Tensor) -> torch.Tensor:
    values = indices.to(dtype=torch.float64)
    return torch.stack((values + 1.0, (values + 1.0) ** 2 / 10.0, torch.ones_like(values)), dim=1)


def _query_features(query_numbers: list[int]) -> torch.Tensor:
    values = torch.tensor(query_numbers, dtype=torch.float64)
    return torch.stack((values + 0.5, (values + 2.0) / 3.0, torch.ones_like(values)), dim=1)


def _new_encoder() -> torch.nn.Linear:
    encoder = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.copy_(
            torch.tensor(
                [[0.20, -0.10, 0.05], [-0.15, 0.25, 0.30]],
                dtype=torch.float64,
            )
        )
    return encoder


def _padded_rows(rows: list[list[int]]) -> torch.Tensor:
    width = max(len(row) for row in rows)
    result = torch.full((len(rows), width), -1, dtype=torch.long)
    for row_index, row in enumerate(rows):
        result[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def _distributed_loss(
    encoder: torch.nn.Linear,
    *,
    local_candidates: list[list[int]],
    query_numbers: list[int],
    positive_indices: list[list[int]],
    global_window_count: int,
) -> tuple[torch.Tensor, object, torch.Tensor]:
    plan = build_global_candidate_plan(
        _padded_rows(local_candidates),
        corpus_size=100,
    )
    owned_embeddings = encoder(_passage_features(plan.local_owned_indices))
    owned_embeddings.retain_grad()
    gathered_embeddings = gather_owned_embeddings(owned_embeddings, plan)
    query_embeddings = encoder(_query_features(query_numbers))
    logits = query_embeddings @ gathered_embeddings.T
    positives = _padded_rows(positive_indices)
    positive_mask = build_index_positive_mask(
        plan.gathered_passage_indices,
        positives,
        plan.valid_passage_mask,
    )
    local_sum, _ = masked_multi_positive_nce_loss_sum(
        logits,
        positive_mask,
        plan.valid_passage_mask,
    )
    return local_sum * (dist.get_world_size() / global_window_count), plan, owned_embeddings


def _reference_loss_sum(
    encoder: torch.nn.Linear,
    *,
    global_candidates: list[int],
    query_numbers: list[int],
    positive_indices: list[list[int]],
) -> torch.Tensor:
    candidate_tensor = torch.tensor(sorted(set(global_candidates)), dtype=torch.long)
    passage_embeddings = encoder(_passage_features(candidate_tensor))
    query_embeddings = encoder(_query_features(query_numbers))
    logits = query_embeddings @ passage_embeddings.T
    positives = _padded_rows(positive_indices)
    valid = torch.ones(candidate_tensor.numel(), dtype=torch.bool)
    positive_mask = build_index_positive_mask(candidate_tensor, positives, valid)
    loss_sum, _ = masked_multi_positive_nce_loss_sum(logits, positive_mask, valid)
    return loss_sum


def _make_trace(table: PassageIndexTable, *, epoch: int, query_id: str, positive_index: int) -> dict:
    positive_id = table.id_for_index(positive_index)
    negative_ids = [
        passage_id
        for index, passage_id in enumerate(table.passage_ids)
        if index != positive_index
    ][:60]
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sampler": "global_uniform",
        "experiment_seed": 17,
        "epoch": epoch,
        "query_id": query_id,
        "doc_id": "case",
        "positive_passage_ids": [positive_id],
        "selected_positive_passage_ids": [positive_id],
        "negative_passage_ids_by_stratum": {"global": negative_ids},
        "eligible_pool_sizes_by_stratum": {"global": len(table) - 1},
        "candidate_passage_ids": [positive_id, *negative_ids],
    }
    return {**payload, "trace_sha256": sampling_trace_checksum(payload)}


def _distributed_worker(
    rank: int,
    init_file: str,
    trace_output: str,
    failure_output: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        # Variable local shapes, cross-rank duplicates, one invalid owner pad,
        # and positives whose owner is the opposite rank.
        local_candidates = (
            [[0, 1, 2], [2, 3]]
            if rank == 0
            else [[1, 2, 3, 4], [4]]
        )
        local_queries = [0] if rank == 0 else [1]
        # Passage 4 is an all-gold target for rank 0's query but enters the
        # shared table only through rank 1's candidates.
        local_positives = [[1, 4]] if rank == 0 else [[0]]

        probe_plan = build_global_candidate_plan(
            _padded_rows(local_candidates),
            corpus_size=100,
        )
        reordered_plan = build_global_candidate_plan(
            _padded_rows([list(reversed(row)) for row in reversed(local_candidates)]),
            corpus_size=100,
        )
        torch.testing.assert_close(
            reordered_plan.gathered_passage_indices,
            probe_plan.gathered_passage_indices,
        )
        probe_owned = torch.ones(
            (probe_plan.local_owned_indices.numel(), 2),
            dtype=torch.float64,
            requires_grad=True,
        )
        probe_gathered = gather_owned_embeddings(probe_owned, probe_plan)
        remote_column = int(
            probe_plan.gathered_passage_indices.eq(1).nonzero(as_tuple=False).item()
        )
        probe_objective = (
            probe_gathered[remote_column].sum()
            if rank == 0
            else probe_gathered.sum() * 0.0
        )
        probe_objective.backward()
        if rank == 1:
            remote_owner_row = probe_plan.local_owned_indices.tolist().index(1)
            torch.testing.assert_close(
                probe_owned.grad[remote_owner_row],
                torch.ones(2, dtype=torch.float64),
                rtol=0.0,
                atol=0.0,
            )

        encoder = _new_encoder()
        scaled_loss, plan, owned_embeddings = _distributed_loss(
            encoder,
            local_candidates=local_candidates,
            query_numbers=local_queries,
            positive_indices=local_positives,
            global_window_count=2,
        )
        torch.testing.assert_close(
            plan.global_unique_indices,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        )
        if plan.valid_passage_mask.sum().item() != 5:
            raise AssertionError("A duplicate candidate received more than one denominator column")
        if plan.gathered_passage_indices.tolist().count(2) != 1:
            raise AssertionError("Cross-rank duplicate passage index 2 was not globally deduplicated")
        if plan.valid_passage_mask.tolist().count(False) != 1:
            raise AssertionError("Expected exactly one balanced-owner padding column")

        scaled_loss.backward()
        if owned_embeddings.grad is None or not torch.isfinite(owned_embeddings.grad).all():
            raise AssertionError("Owned passage embeddings did not receive finite gradients")

        distributed_loss_mean = scaled_loss.detach().clone()
        dist.all_reduce(distributed_loss_mean, op=dist.ReduceOp.SUM)
        distributed_loss_mean /= 2.0
        distributed_grad = encoder.weight.grad.detach().clone()
        dist.all_reduce(distributed_grad, op=dist.ReduceOp.SUM)
        distributed_grad /= 2.0

        reference = _new_encoder()
        reference_loss = _reference_loss_sum(
            reference,
            global_candidates=[0, 1, 2, 3, 4],
            query_numbers=[0, 1],
            positive_indices=[[1, 4], [0]],
        ) / 2.0
        reference_loss.backward()
        torch.testing.assert_close(distributed_loss_mean, reference_loss.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(distributed_grad, reference.weight.grad, rtol=1e-12, atol=1e-12)
        distributed_updated = encoder.weight.detach() - 0.05 * distributed_grad
        reference_updated = reference.weight.detach() - 0.05 * reference.weight.grad
        torch.testing.assert_close(distributed_updated, reference_updated, rtol=1e-12, atol=1e-12)

        # Three-microbatch incomplete accumulation window with a 2/1 final
        # rank layout. Parameter averaging occurs once, as DeepSpeed does at the
        # explicit Step-4 window boundary.
        layouts = [
            (
                [[0, 1, 2], [2, 3]] if rank == 0 else [[1, 2, 3], [3, 4]],
                [10, 11] if rank == 0 else [12, 13],
                [[0], [3]] if rank == 0 else [[1], [4]],
            ),
            (
                [[1, 2, 4], [4, 5]] if rank == 0 else [[1, 3, 5], [2, 5]],
                [14, 15] if rank == 0 else [16, 17],
                [[2], [5]] if rank == 0 else [[3], [1]],
            ),
            (
                [[0, 2, 3], [1, 3]] if rank == 0 else [[0, 1, 3]],
                [18, 19] if rank == 0 else [20],
                [[2], [1]] if rank == 0 else [[0]],
            ),
        ]
        accumulated = _new_encoder()
        for candidates, query_numbers, positives in layouts:
            micro_loss, _, _ = _distributed_loss(
                accumulated,
                local_candidates=candidates,
                query_numbers=query_numbers,
                positive_indices=positives,
                global_window_count=11,
            )
            micro_loss.backward()
        accumulated_grad = accumulated.weight.grad.detach().clone()
        dist.all_reduce(accumulated_grad, op=dist.ReduceOp.SUM)
        accumulated_grad /= 2.0

        reference_accumulated = _new_encoder()
        reference_microbatches = [
            ([0, 1, 2, 3, 4], [10, 11, 12, 13], [[0], [3], [1], [4]]),
            ([1, 2, 3, 4, 5], [14, 15, 16, 17], [[2], [5], [3], [1]]),
            ([0, 1, 2, 3], [18, 19, 20], [[2], [1], [0]]),
        ]
        reference_window_loss = sum(
            _reference_loss_sum(
                reference_accumulated,
                global_candidates=candidates,
                query_numbers=query_numbers,
                positive_indices=positives,
            )
            for candidates, query_numbers, positives in reference_microbatches
        ) / 11.0
        reference_window_loss.backward()
        torch.testing.assert_close(
            accumulated_grad,
            reference_accumulated.weight.grad,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            accumulated.weight.detach() - 0.05 * accumulated_grad,
            reference_accumulated.weight.detach() - 0.05 * reference_accumulated.weight.grad,
            rtol=1e-12,
            atol=1e-12,
        )

        # Synthetic |U| < world_size exercises a zero-owner rank without
        # dropping the autograd collective from that rank's graph.
        zero_owner_plan = build_global_candidate_plan(
            torch.tensor([[0]], dtype=torch.long),
            corpus_size=2,
        )
        zero_owner_embeddings = torch.empty(
            (zero_owner_plan.local_owned_indices.numel(), 2),
            dtype=torch.float64,
            requires_grad=True,
        )
        if zero_owner_embeddings.numel():
            with torch.no_grad():
                zero_owner_embeddings.fill_(1.0)
        zero_owner_gather = gather_owned_embeddings(zero_owner_embeddings, zero_owner_plan)
        zero_owner_objective = zero_owner_gather.sum() * (1.0 if rank == 0 else 0.0)
        zero_owner_objective.backward()
        if zero_owner_embeddings.grad is None:
            raise AssertionError("Zero-owner rank was detached from autograd gather")

        invalid_candidate_tables = (
            torch.tensor([[-1, 0]], dtype=torch.long),
            torch.tensor([[0, 0]], dtype=torch.long),
            torch.tensor([[100]], dtype=torch.long),
            torch.tensor([[True]], dtype=torch.bool),
        )
        for invalid_candidates in invalid_candidate_tables:
            try:
                build_global_candidate_plan(invalid_candidates, corpus_size=100)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(
                    f"Invalid candidate table was accepted: {invalid_candidates.tolist()}"
                )

        corpus = {
            f"p{index:03d}": CorpusPassage(
                passage_id=f"p{index:03d}",
                doc_id="case",
                label="Analysis",
                text=f"passage {index}",
            )
            for index in range(65)
        }
        table = PassageIndexTable(corpus)
        store = CandidateTraceStore(
            Path(trace_output),
            passage_index_table=table,
            rank=rank,
            world_size=2,
        )
        mismatched_trace = _make_trace(
            table,
            epoch=0,
            query_id=f"bad-{rank}",
            positive_index=0,
        )
        mismatched_candidates = table.indices_for_ids(
            mismatched_trace["candidate_passage_ids"]
        )
        mismatched_candidates[0], mismatched_candidates[1] = (
            mismatched_candidates[1],
            mismatched_candidates[0],
        )
        try:
            store.record_batch(
                [mismatched_trace],
                candidate_passage_indices=_padded_rows([mismatched_candidates]),
                positive_passage_indices=_padded_rows([[0]]),
            )
        except ValueError as exc:
            if "Trace/index candidate mismatch" not in str(exc):
                raise
        else:
            raise AssertionError("Trace/index disagreement was accepted")
        assignments = {
            0: ((0, "q0", 0), (0, "q2", 2), (1, "q1", 1)),
            1: ((0, "q1", 1), (1, "q0", 0), (1, "q2", 2)),
        }
        for assignment_index, (epoch, query_id, positive_index) in enumerate(assignments[rank]):
            trace = _make_trace(
                table,
                epoch=epoch,
                query_id=query_id,
                positive_index=positive_index,
            )
            candidate_indices = table.indices_for_ids(trace["candidate_passage_ids"])
            store.record_batch(
                [trace],
                candidate_passage_indices=_padded_rows([candidate_indices]),
                positive_passage_indices=_padded_rows([[positive_index]]),
            )
            if rank == 0 and assignment_index == 0:
                try:
                    store.record_batch(
                        [trace],
                        candidate_passage_indices=_padded_rows([candidate_indices]),
                        positive_passage_indices=_padded_rows([[positive_index]]),
                    )
                except ValueError as exc:
                    if "Duplicate rank-local" not in str(exc):
                        raise
                else:
                    raise AssertionError("Duplicate rank-local trace was accepted")
        manifest = store.finalize(
            expected_epochs=2,
            expected_query_ids=["q0", "q1", "q2"],
        )
        if manifest["record_count"] != 6 or manifest["merged"]["record_count"] != 6:
            raise AssertionError("Trace merger did not preserve exact epoch/query coverage")

        failure_store = CandidateTraceStore(
            Path(failure_output),
            passage_index_table=table,
            rank=rank,
            world_size=2,
        )
        failure_trace = _make_trace(
            table,
            epoch=0,
            query_id=f"q{rank}",
            positive_index=rank,
        )
        failure_candidates = table.indices_for_ids(
            failure_trace["candidate_passage_ids"]
        )
        failure_store.record_batch(
            [failure_trace],
            candidate_passage_indices=_padded_rows([failure_candidates]),
            positive_passage_indices=_padded_rows([[rank]]),
        )

        original_publish = trace_module._publish_new_file

        def fail_manifest_publish(path: Path, content: str) -> None:
            if path.name == "manifest.json":
                raise OSError("injected rank-zero manifest publication failure")
            original_publish(path, content)

        patch_context = (
            mock.patch.object(
                trace_module,
                "_publish_new_file",
                side_effect=fail_manifest_publish,
            )
            if rank == 0
            else nullcontext()
        )
        with patch_context:
            try:
                failure_store.finalize(
                    expected_epochs=1,
                    expected_query_ids=["q0", "q1"],
                )
            except RuntimeError as error:
                if "injected rank-zero manifest publication failure" not in str(error):
                    raise
            else:
                raise AssertionError("Injected rank-zero merge failure was not propagated")
        failure_trace_dir = Path(failure_output) / "candidate_traces"
        if (failure_trace_dir / "manifest.json").exists():
            raise AssertionError("Failed merge published a final manifest")
        if (failure_trace_dir / "sampling_traces.jsonl").exists():
            raise AssertionError("Failed merge left a partially published merged artifact")
        try:
            CandidateTraceStore(
                Path(failure_output),
                passage_index_table=table,
                rank=rank,
                world_size=2,
            )
        except RuntimeError as error:
            if "Candidate trace directory creation" not in str(error):
                raise
        else:
            raise AssertionError("Existing trace directory was not rejected on every rank")
    finally:
        dist.destroy_process_group()


def _nccl_worker(rank: int, init_file: str, trace_output: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
    )
    try:
        candidates_by_rank = (
            [0, 1, 2],
            [1, 2, 3, 4],
            [0, 4, 5, 6, 7, 8],
            [2, 6],
        )
        candidate_tensor = torch.tensor(
            [candidates_by_rank[rank]],
            dtype=torch.long,
            device=torch.device("cuda", rank),
        )
        plan = build_global_candidate_plan(candidate_tensor, corpus_size=100)
        owned_embeddings = torch.ones(
            (plan.local_owned_indices.numel(), 3),
            dtype=torch.bfloat16,
            device=candidate_tensor.device,
            requires_grad=True,
        )
        gathered_embeddings = gather_owned_embeddings(owned_embeddings, plan)
        if plan.global_unique_indices.tolist() != list(range(9)):
            raise AssertionError("NCCL smoke global candidate table changed")
        if int(plan.valid_passage_mask.sum().item()) != 9:
            raise AssertionError("NCCL smoke lost or duplicated a real passage")
        if plan.valid_passage_mask.tolist().count(False) != 3:
            raise AssertionError("NCCL smoke did not exercise padded owner rows")

        remote_column = int(
            plan.gathered_passage_indices.eq(1).nonzero(as_tuple=False).item()
        )
        objective = (
            gathered_embeddings[remote_column].sum()
            if rank == 0
            else gathered_embeddings.sum() * 0.0
        )
        objective.backward()
        if rank == 1:
            owner_row = plan.local_owned_indices.tolist().index(1)
            torch.testing.assert_close(
                owned_embeddings.grad[owner_row],
                torch.ones(3, dtype=torch.bfloat16, device=candidate_tensor.device),
                rtol=0.0,
                atol=0.0,
            )

        corpus = {
            f"p{index:03d}": CorpusPassage(
                passage_id=f"p{index:03d}",
                doc_id="case",
                label="Analysis",
                text=f"passage {index}",
            )
            for index in range(65)
        }
        table = PassageIndexTable(corpus)
        trace_store = CandidateTraceStore(
            Path(trace_output),
            passage_index_table=table,
            rank=rank,
            world_size=4,
        )
        trace = _make_trace(
            table,
            epoch=0,
            query_id=f"q{rank}",
            positive_index=rank,
        )
        trace_store.record_batch(
            [trace],
            candidate_passage_indices=_padded_rows(
                [table.indices_for_ids(trace["candidate_passage_ids"])]
            ),
            positive_passage_indices=_padded_rows([[rank]]),
        )
        manifest = trace_store.finalize(
            expected_epochs=1,
            expected_query_ids=["q0", "q1", "q2", "q3"],
        )
        if manifest["record_count"] != 4:
            raise AssertionError("NCCL trace finalization lost rank shards")
    finally:
        dist.destroy_process_group()


class PassageIndexTableTest(unittest.TestCase):
    def test_table_is_contiguous_order_invariant_and_content_addressed(self) -> None:
        records = {
            "z": CorpusPassage("z", "2", "Analysis", "last"),
            "a": CorpusPassage("a", "1", "Rule", "first"),
            "m": CorpusPassage("m", "1", "Analysis", "middle"),
        }
        table = PassageIndexTable(records)
        replay = PassageIndexTable(dict(reversed(list(records.items()))))
        self.assertEqual(table.passage_ids, ("a", "m", "z"))
        self.assertEqual([table.index_for_id(value) for value in table.passage_ids], [0, 1, 2])
        self.assertEqual(table.sha256, replay.sha256)
        self.assertEqual(table.id_for_index(1), "m")
        self.assertEqual(table.text_for_index(1), "middle")

        mismatched = {"a": CorpusPassage("different", "1", "Rule", "text")}
        with self.assertRaisesRegex(ValueError, "does not match"):
            PassageIndexTable(mismatched)
        with self.assertRaises(TypeError):
            table.index_for_id(1)  # type: ignore[arg-type]
        with self.assertRaises(IndexError):
            table.id_for_index(3)
        with self.assertRaises(FrozenInstanceError):
            table._sha256 = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            table._index_by_passage_id["new"] = 3  # type: ignore[index]

    def test_duplicate_source_passage_ids_are_rejected_before_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = [
                {"passage_id": "p", "doc_id": "case", "label": "Rule", "text": "one"},
                {"passage_id": "p", "doc_id": "case", "label": "Rule", "text": "two"},
            ]
            (root / "corpus.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate corpus passage_id"):
                load_corpus(root)

    def test_atomic_trace_publication_refuses_overwrite_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "artifact.json"
            trace_module._publish_new_file(target, "complete\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "complete\n")
            self.assertFalse((target.parent / ".artifact.json.tmp").exists())
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                trace_module._publish_new_file(target, "replacement\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "complete\n")


class DistributedContrastiveTest(unittest.TestCase):
    def test_two_rank_training_and_trace_artifacts_match_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_file = root / "process-group"
            trace_output = root / "output"
            failure_output = root / "failure-output"
            trace_output.mkdir()
            failure_output.mkdir()
            mp.spawn(
                _distributed_worker,
                args=(str(init_file), str(trace_output), str(failure_output)),
                nprocs=2,
                join=True,
            )
            manifest = json.loads(
                (trace_output / "candidate_traces/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["record_count"], 6)
            merged_lines = (
                trace_output / "candidate_traces/sampling_traces.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            merged_keys = [
                (record["epoch"], record["query_id"])
                for record in map(json.loads, merged_lines)
            ]
            self.assertEqual(
                merged_keys,
                [(0, "q0"), (0, "q1"), (0, "q2"), (1, "q0"), (1, "q1"), (1, "q2")],
            )
            self.assertFalse((failure_output / "candidate_traces/manifest.json").exists())
            self.assertFalse(
                (failure_output / "candidate_traces/sampling_traces.jsonl").exists()
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 4,
        "requires four visible CUDA devices and the pinned NCCL runtime",
    )
    def test_four_gpu_nccl_padded_autograd_gather(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_output = Path(temp_dir) / "trace-output"
            trace_output.mkdir()
            mp.spawn(
                _nccl_worker,
                args=(
                    str(Path(temp_dir) / "nccl-process-group"),
                    str(trace_output),
                ),
                nprocs=4,
                join=True,
            )
            manifest = json.loads(
                (trace_output / "candidate_traces/manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["record_count"], 4)


if __name__ == "__main__":
    unittest.main()
