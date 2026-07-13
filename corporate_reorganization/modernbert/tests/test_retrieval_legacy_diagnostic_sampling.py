from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from corporate_reorganization.modernbert.retriever.legacy_diagnostic_data import (
    TRAIN_CASE_IDS,
    load_corrected_legacy_data,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_sampling import (
    CANDIDATE_OCCURRENCES_PER_QUERY,
    CorrectedLegacyDiagnosticDataset,
    ROLE_OTHER_BACKGROUND,
    ROLE_POSITIVE,
    ROLE_SAME_CASE,
    validate_legacy_diagnostic_trace,
)
from corporate_reorganization.modernbert.retriever.query_views import (
    QUERY_VIEW_FLAT_MASKED,
    QUERY_VIEW_STRUCTURED,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CORRECTED_V2_DIR = (
    REPO_ROOT / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)


class CorrectedLegacySamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_corrected_legacy_data(CORRECTED_V2_DIR)

    def test_every_training_query_has_exact_occurrence_and_provenance_contract(self) -> None:
        dataset = CorrectedLegacyDiagnosticDataset(self.data, experiment_seed=17)
        replacement_query_count = 0
        train_cases = set(TRAIN_CASE_IDS)
        for index in range(len(dataset)):
            record = dataset[index]
            trace = record["sampling_trace"]
            validate_legacy_diagnostic_trace(trace)
            self.assertEqual(
                len(record["candidate_passage_occurrence_indices"]),
                CANDIDATE_OCCURRENCES_PER_QUERY,
            )
            self.assertEqual(sum(record["candidate_multiplicities"]), 64)
            self.assertEqual(
                len(record["unique_candidate_passage_indices"]),
                len(record["candidate_multiplicities"]),
            )
            case_gold = self.data.gold_passage_ids_by_case[record["doc_id"]]
            for occurrence in trace["occurrences"]:
                passage_id = occurrence["passage_id"]
                passage = self.data.corpus_by_passage_id[passage_id]
                if occurrence["role"] == ROLE_SAME_CASE:
                    self.assertEqual(passage.doc_id, record["doc_id"])
                    self.assertNotIn(passage_id, case_gold)
                elif occurrence["role"] == ROLE_OTHER_BACKGROUND:
                    self.assertIn(passage.doc_id, train_cases)
                    self.assertNotEqual(passage.doc_id, record["doc_id"])
                    self.assertEqual(passage.label, "Background Facts")
            if trace["replacement_count_by_role"][ROLE_SAME_CASE]:
                replacement_query_count += 1
        self.assertEqual(replacement_query_count, 103)

    def test_first_trace_is_frozen_and_changes_with_epoch_or_seed(self) -> None:
        dataset = CorrectedLegacyDiagnosticDataset(self.data, experiment_seed=17)
        first = dataset[0]
        self.assertEqual(
            first["sampling_trace_sha256"],
            "badbb785b78c4b7f832ad3727365125439dd80cbd8745db5c7385af351c00d63",
        )
        self.assertEqual(
            first["sampling_trace"]["selected_positive_passage_ids"],
            [
                "36::SENT_00075",
                "36::SENT_00074",
                "36::SENT_00076",
                "36::SENT_00072",
            ],
        )
        dataset.set_epoch(1)
        self.assertNotEqual(dataset[0]["sampling_trace_sha256"], first["sampling_trace_sha256"])
        other_seed = CorrectedLegacyDiagnosticDataset(self.data, experiment_seed=29)
        self.assertNotEqual(
            other_seed[0]["sampling_trace_sha256"], first["sampling_trace_sha256"]
        )

    def test_sampling_is_view_independent_and_pool_order_independent(self) -> None:
        structured = CorrectedLegacyDiagnosticDataset(
            self.data,
            experiment_seed=17,
            query_view=QUERY_VIEW_STRUCTURED,
        )
        flat = CorrectedLegacyDiagnosticDataset(
            self.data,
            experiment_seed=17,
            query_view=QUERY_VIEW_FLAT_MASKED,
        )
        self.assertNotEqual(structured[0]["query_text"], flat[0]["query_text"])
        self.assertEqual(structured[0]["sampling_trace"], flat[0]["sampling_trace"])

        reversed_pools = MappingProxyType(
            {
                case_id: tuple(reversed(passage_ids))
                for case_id, passage_ids in self.data.candidate_passage_ids_by_case.items()
            }
        )
        reversed_data = replace(
            self.data,
            candidate_passage_ids_by_case=reversed_pools,
            training_background_passage_ids=tuple(
                reversed(self.data.training_background_passage_ids)
            ),
        )
        reversed_dataset = CorrectedLegacyDiagnosticDataset(reversed_data, experiment_seed=17)
        self.assertEqual(
            structured[0]["sampling_trace"], reversed_dataset[0]["sampling_trace"]
        )

    def test_whole_draw_switches_to_replacement_only_when_pool_is_too_small(self) -> None:
        dataset = CorrectedLegacyDiagnosticDataset(self.data, experiment_seed=17)
        trace = next(
            dataset[index]["sampling_trace"]
            for index in range(len(dataset))
            if dataset[index]["sampling_trace"]["replacement_count_by_role"][ROLE_SAME_CASE]
        )
        same = [row for row in trace["occurrences"] if row["role"] == ROLE_SAME_CASE]
        pool_size = trace["eligible_pool_sizes_by_role"][ROLE_SAME_CASE]
        self.assertLess(pool_size, len(same))
        self.assertTrue(all(row["with_replacement"] for row in same))
        self.assertEqual(
            len(same),
            trace["replacement_count_by_role"][ROLE_SAME_CASE],
        )

        no_replacement_trace = dataset[0]["sampling_trace"]
        no_replacement_same = [
            row
            for row in no_replacement_trace["occurrences"]
            if row["role"] == ROLE_SAME_CASE
        ]
        self.assertGreaterEqual(
            no_replacement_trace["eligible_pool_sizes_by_role"][ROLE_SAME_CASE],
            len(no_replacement_same),
        )
        self.assertTrue(all(not row["with_replacement"] for row in no_replacement_same))
        self.assertEqual(
            len({row["passage_id"] for row in no_replacement_same}),
            len(no_replacement_same),
        )

    def test_trace_mutation_fails_checksum_validation(self) -> None:
        dataset = CorrectedLegacyDiagnosticDataset(self.data, experiment_seed=17)
        trace = copy.deepcopy(dataset[0]["sampling_trace"])
        trace["occurrences"][0]["selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_legacy_diagnostic_trace(trace)

    def test_positive_and_negative_role_counts_always_sum_to_64(self) -> None:
        dataset = CorrectedLegacyDiagnosticDataset(self.data, experiment_seed=17)
        for index in (0, 7, 100, 417):
            trace = dataset[index]["sampling_trace"]
            quotas = trace["quota_by_role"]
            self.assertLessEqual(quotas[ROLE_POSITIVE], 4)
            self.assertEqual(quotas[ROLE_OTHER_BACKGROUND], 4)
            self.assertEqual(
                quotas[ROLE_POSITIVE] + quotas[ROLE_SAME_CASE] + quotas[ROLE_OTHER_BACKGROUND],
                64,
            )


if __name__ == "__main__":
    unittest.main()
