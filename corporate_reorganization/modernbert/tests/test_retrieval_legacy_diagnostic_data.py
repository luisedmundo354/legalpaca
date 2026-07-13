from __future__ import annotations

import unittest
from pathlib import Path

from corporate_reorganization.modernbert.retriever.legacy_diagnostic_data import (
    CASE_IDS_BY_SPLIT,
    EXPECTED_PASSAGE_COUNTS,
    EXPECTED_QUERY_COUNTS,
    MEMBERSHIP_SHA256,
    TEST_CASE_IDS,
    TRAIN_CASE_IDS,
    VALIDATION_CASE_IDS,
    load_corrected_legacy_data,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CORRECTED_V2_DIR = (
    REPO_ROOT / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)


class CorrectedLegacyDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_corrected_legacy_data(CORRECTED_V2_DIR)

    def test_frozen_march_membership_and_digest_are_exact(self) -> None:
        self.assertEqual(len(TRAIN_CASE_IDS), 34)
        self.assertEqual(TRAIN_CASE_IDS[0], "36")
        self.assertEqual(TRAIN_CASE_IDS[-1], "97")
        self.assertEqual(VALIDATION_CASE_IDS, ("45", "47", "60", "62"))
        self.assertEqual(TEST_CASE_IDS, ("37", "46", "65", "96"))
        self.assertEqual(
            MEMBERSHIP_SHA256,
            "b0b34cb174a5bf615322ad36e05702d5f65fb8f7c89024a7f3b01eec5b59df3b",
        )
        all_cases = [case_id for case_ids in CASE_IDS_BY_SPLIT.values() for case_id in case_ids]
        self.assertEqual(len(all_cases), 42)
        self.assertEqual(len(set(all_cases)), 42)

    def test_corrected_v2_partitions_have_exact_query_and_passage_counts(self) -> None:
        self.assertEqual(
            {split: len(rows) for split, rows in self.data.queries_by_split.items()},
            dict(EXPECTED_QUERY_COUNTS),
        )
        self.assertEqual(
            {split: len(rows) for split, rows in self.data.passage_ids_by_split.items()},
            dict(EXPECTED_PASSAGE_COUNTS),
        )
        self.assertEqual(sum(map(len, self.data.queries_by_split.values())), 490)
        self.assertEqual(sum(map(len, self.data.passage_ids_by_split.values())), 5286)
        self.assertEqual(self.data.membership_sha256, MEMBERSHIP_SHA256)
        self.assertEqual(len(self.data.training_background_passage_ids), 1191)

    def test_membership_is_deeply_immutable_at_public_boundaries(self) -> None:
        with self.assertRaises(TypeError):
            self.data.queries_by_split["train"] = ()  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.data.candidate_passage_ids_by_case["36"] = ()  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.data.gold_passage_ids_by_case["36"] = frozenset()  # type: ignore[index]
        query = self.data.queries_by_split["train"][0]
        with self.assertRaises((AttributeError, TypeError)):
            query.positive_passage_ids.append("changed")  # type: ignore[attr-defined]

    def test_every_row_is_assigned_only_by_the_frozen_case_membership(self) -> None:
        for split, case_ids in CASE_IDS_BY_SPLIT.items():
            case_set = set(case_ids)
            self.assertEqual(
                {query.doc_id for query in self.data.queries_by_split[split]}, case_set
            )
            self.assertEqual(
                {
                    self.data.corpus_by_passage_id[passage_id].doc_id
                    for passage_id in self.data.passage_ids_by_split[split]
                },
                case_set,
            )

    def test_missing_corrected_v2_directory_fails_loudly(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_corrected_legacy_data(CORRECTED_V2_DIR / "missing")


if __name__ == "__main__":
    unittest.main()
