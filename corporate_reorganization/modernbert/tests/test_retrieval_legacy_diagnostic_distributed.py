from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from corporate_reorganization.modernbert.retriever.distributed import (
    build_global_candidate_plan,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_distributed import (
    gather_global_candidate_multiplicities,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_losses import (
    multiplicity_aware_multi_positive_nce_loss_sum,
)


def _worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
    )
    try:
        rows = (
            ([0, 1], [60, 4]),
            ([1, 2], [3, 61]),
            ([2, 3], [2, 62]),
            ([0, 3], [1, 63]),
        )
        indices, multiplicities = rows[rank]
        index_tensor = torch.tensor([indices], dtype=torch.long)
        multiplicity_tensor = torch.tensor([multiplicities], dtype=torch.long)
        plan = build_global_candidate_plan(index_tensor, corpus_size=10)
        gathered = gather_global_candidate_multiplicities(
            index_tensor,
            multiplicity_tensor,
            plan,
            corpus_size=10,
        )
        expected = {0: 61, 1: 7, 2: 63, 3: 125}
        actual = {
            int(index): int(count)
            for index, count, valid in zip(
                plan.gathered_passage_indices.tolist(),
                gathered.tolist(),
                plan.valid_passage_mask.tolist(),
            )
            if valid
        }
        if actual != expected or int(gathered.sum().item()) != 256:
            raise AssertionError(f"Global corrected legacy multiplicities changed: {actual}")

        unique_logits = torch.tensor(
            [[0.2, -0.1, 0.4, 0.3]],
            dtype=torch.float64,
            requires_grad=True,
        )
        ordered_indices = plan.gathered_passage_indices[plan.valid_passage_mask].tolist()
        aligned_multiplicities = gathered[plan.valid_passage_mask]
        positive_mask = torch.tensor(
            [[index in {0, 2} for index in ordered_indices]],
            dtype=torch.bool,
        )
        collapsed, _ = multiplicity_aware_multi_positive_nce_loss_sum(
            unique_logits,
            positive_mask,
            aligned_multiplicities,
        )
        collapsed.backward()
        collapsed_gradient = unique_logits.grad.detach().clone()

        explicit_logits = unique_logits.detach().clone().requires_grad_(True)
        repeated = torch.repeat_interleave(explicit_logits, aligned_multiplicities, dim=1)
        repeated_positive = torch.repeat_interleave(
            positive_mask,
            aligned_multiplicities,
            dim=1,
        )
        numerator = torch.logsumexp(
            repeated.masked_fill(~repeated_positive, float("-inf")),
            dim=1,
        )
        explicit = -(numerator - torch.logsumexp(repeated, dim=1)).sum()
        explicit.backward()
        torch.testing.assert_close(collapsed, explicit, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            collapsed_gradient,
            explicit_logits.grad,
            rtol=1e-12,
            atol=1e-12,
        )
    finally:
        dist.destroy_process_group()


class CorrectedLegacyDistributedMultiplicityTest(unittest.TestCase):
    def test_four_rank_global_multiplicity_matches_explicit_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            init_file = Path(temporary) / "init"
            mp.spawn(_worker, args=(str(init_file),), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
