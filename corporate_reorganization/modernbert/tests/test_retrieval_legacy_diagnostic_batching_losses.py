from __future__ import annotations

import unittest

import torch

from corporate_reorganization.modernbert.retriever.batching import DUMMY_QUERY_INDEX
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_batching import (
    CorrectedLegacyQueryBatchSampler,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_losses import (
    collapse_candidate_occurrence_indices,
    multiplicity_aware_multi_positive_nce_loss_sum,
)
from corporate_reorganization.modernbert.retriever.losses import multi_positive_nce_loss_sum


class CorrectedLegacyBatchingTest(unittest.TestCase):
    def make_sampler(self) -> CorrectedLegacyQueryBatchSampler:
        return CorrectedLegacyQueryBatchSampler(
            [f"query-{index:03d}" for index in range(418)],
            experiment_seed=17,
            world_size=4,
            per_device_batch_size=4,
        )

    def test_exact_27_batch_14_sentinel_and_optimizer_window_contract(self) -> None:
        sampler = self.make_sampler()
        batches = sampler.batches()
        self.assertEqual(len(sampler), 108)
        self.assertEqual(sampler.prepared_batches_per_rank, 27)
        self.assertEqual(sampler.num_sentinel_rows, 14)
        self.assertEqual(sum(row.count(DUMMY_QUERY_INDEX) for row in batches), 14)
        self.assertEqual(sampler.global_real_query_counts, (16,) * 25 + (9, 9))
        self.assertEqual(sampler.optimizer_window_real_query_counts, (128, 128, 128, 34))
        self.assertEqual(
            [
                sum(index != DUMMY_QUERY_INDEX for index in batches[-8 + rank])
                for rank in range(4)
            ],
            [3, 2, 2, 2],
        )
        self.assertEqual(
            [
                sum(index != DUMMY_QUERY_INDEX for index in batches[-4 + rank])
                for rank in range(4)
            ],
            [3, 2, 2, 2],
        )

    def test_every_query_occurs_once_and_each_rank_gets_27_prepared_batches(self) -> None:
        sampler = self.make_sampler()
        batches = sampler.batches()
        real = [index for batch in batches for index in batch if index != DUMMY_QUERY_INDEX]
        self.assertEqual(sorted(real), list(range(418)))
        for rank in range(4):
            rank_batches = batches[rank::4]
            self.assertEqual(len(rank_batches), 27)
            self.assertTrue(all(any(index != DUMMY_QUERY_INDEX for index in row) for row in rank_batches))

    def test_order_is_reproducible_and_epoch_sensitive(self) -> None:
        sampler = self.make_sampler()
        epoch_zero = sampler.batches()
        self.assertEqual(epoch_zero, self.make_sampler().batches())
        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, sampler.batches())

    def test_wrong_research_shape_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 418"):
            CorrectedLegacyQueryBatchSampler(
                [f"q{i}" for i in range(417)],
                experiment_seed=17,
                world_size=4,
                per_device_batch_size=4,
            )
        with self.assertRaisesRegex(ValueError, "world_size=4"):
            CorrectedLegacyQueryBatchSampler(
                [f"q{i}" for i in range(418)],
                experiment_seed=17,
                world_size=2,
                per_device_batch_size=4,
            )


class CorrectedLegacyMultiplicityLossTest(unittest.TestCase):
    def test_collapse_is_sorted_and_preserves_every_occurrence(self) -> None:
        collapsed = collapse_candidate_occurrence_indices(
            [[7, 2, 7, 5], [5, 5, 2]],
            corpus_size=10,
        )
        self.assertEqual(collapsed.unique_passage_indices, (2, 5, 7))
        self.assertEqual(collapsed.multiplicities, (2, 3, 2))
        self.assertEqual(collapsed.total_occurrences, 7)

    def test_log_m_loss_and_gradient_match_explicit_duplicate_columns(self) -> None:
        base_logits = torch.tensor(
            [[0.2, -0.3, 1.1, 0.7], [-0.8, 0.4, 0.1, 1.5]],
            dtype=torch.float64,
        )
        positive_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True]],
            dtype=torch.bool,
        )
        multiplicities = torch.tensor([3, 1, 2, 4], dtype=torch.long)

        unique_logits = base_logits.clone().requires_grad_(True)
        weighted_loss, weighted_per_query = multiplicity_aware_multi_positive_nce_loss_sum(
            unique_logits,
            positive_mask,
            multiplicities,
        )
        weighted_loss.backward()
        weighted_gradient = unique_logits.grad.detach().clone()

        expanded_indices = torch.repeat_interleave(
            torch.arange(multiplicities.numel()), multiplicities
        )
        explicit_logits = base_logits.clone().requires_grad_(True)
        expanded_logits = explicit_logits.index_select(1, expanded_indices)
        expanded_positive_mask = positive_mask.index_select(1, expanded_indices)
        explicit_loss, explicit_per_query = multi_positive_nce_loss_sum(
            expanded_logits,
            expanded_positive_mask,
        )
        explicit_loss.backward()

        torch.testing.assert_close(weighted_loss, explicit_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            weighted_per_query, explicit_per_query, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            weighted_gradient, explicit_logits.grad, rtol=1e-12, atol=1e-12
        )

    def test_invalid_multiplicity_fails_loudly(self) -> None:
        logits = torch.zeros((1, 2), dtype=torch.float32)
        positives = torch.tensor([[True, False]])
        with self.assertRaisesRegex(ValueError, "multiplicity >= 1"):
            multiplicity_aware_multi_positive_nce_loss_sum(
                logits,
                positives,
                torch.tensor([1, 0], dtype=torch.long),
            )
        with self.assertRaisesRegex(TypeError, "torch.long"):
            multiplicity_aware_multi_positive_nce_loss_sum(
                logits,
                positives,
                torch.tensor([1.0, 1.0]),
            )


if __name__ == "__main__":
    unittest.main()
