from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import torch

from corporate_reorganization.modernbert.retriever import corrected_legacy_evaluation
from corporate_reorganization.modernbert.retriever.data import (
    PassageIndexTable,
    load_corpus,
    load_queries,
)
from corporate_reorganization.modernbert.retriever.evaluation import (
    canonical_result_from_payload,
    compute_canonical_retrieval_result_from_scores,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_data import (
    TEST_CASE_IDS,
    VALIDATION_CASE_IDS,
)


DATASET = (
    Path(__file__).resolve().parents[2]
    / "data/final_annotations_gold/processed_retrieval_v2"
)


class CorrectedLegacyEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(DATASET)
        cls.queries = load_queries(DATASET, "all")
        cls.passage_index = PassageIndexTable(cls.corpus)

    def test_exact_validation_and_test_contracts_for_both_views(self) -> None:
        validation_hashes = []
        test_hashes = []
        for view in ("flat_masked", "structured"):
            validation = (
                corrected_legacy_evaluation.build_corrected_legacy_validation_evidence_data(
                    all_queries=self.queries,
                    corpus_by_passage_id=self.corpus,
                    passage_index_table=self.passage_index,
                    validation_case_ids=VALIDATION_CASE_IDS,
                    query_view=view,
                )
            )
            self.assertEqual(len(validation.case_ids), 4)
            self.assertEqual(len(validation.queries), 32)
            self.assertEqual(len(validation.passage_indices), 398)
            self.assertEqual(validation.evaluation_data.regime_name, "fold_global")
            validation_hashes.append(validation.contract_sha256)

            test = corrected_legacy_evaluation.build_corrected_legacy_test_data(
                all_queries=self.queries,
                corpus_by_passage_id=self.corpus,
                passage_index_table=self.passage_index,
                test_case_ids=TEST_CASE_IDS,
                query_view=view,
            )
            self.assertEqual(len(test.case_ids), 4)
            self.assertEqual(len(test.queries), 40)
            self.assertEqual(len(test.passage_indices), 581)
            self.assertEqual(
                tuple(test.evaluation_data_by_regime),
                corrected_legacy_evaluation.CORRECTED_LEGACY_TEST_REGIMES,
            )
            self.assertEqual(
                {data.query_count for data in test.evaluation_data_by_regime.values()},
                {40},
            )
            self.assertEqual(
                {data.passage_count for data in test.evaluation_data_by_regime.values()},
                {581},
            )
            test_hashes.append(test.contract_sha256)
        self.assertEqual(len(set(validation_hashes)), 2)
        self.assertEqual(len(set(test_hashes)), 2)

    def test_validation_payload_is_complete_and_independently_replayable(self) -> None:
        validation = (
            corrected_legacy_evaluation.build_corrected_legacy_validation_evidence_data(
                all_queries=self.queries,
                corpus_by_passage_id=self.corpus,
                passage_index_table=self.passage_index,
                validation_case_ids=VALIDATION_CASE_IDS,
                query_view="structured",
            )
        )
        scores = torch.zeros((32, 398), dtype=torch.float32)
        result = compute_canonical_retrieval_result_from_scores(
            scores=scores,
            evaluation_data=validation.evaluation_data,
        )
        self.assertEqual(len(result.rankings), 32)
        self.assertEqual(len(result.source_rankings), 32)
        replay = canonical_result_from_payload(
            result.to_payload(),
            validation.evaluation_data,
        )
        self.assertEqual(replay.to_payload(), result.to_payload())

    def test_mutated_role_contracts_fail_loudly(self) -> None:
        validation = (
            corrected_legacy_evaluation.build_corrected_legacy_validation_evidence_data(
                all_queries=self.queries,
                corpus_by_passage_id=self.corpus,
                passage_index_table=self.passage_index,
                validation_case_ids=VALIDATION_CASE_IDS,
                query_view="flat_masked",
            )
        )
        with self.assertRaisesRegex(ValueError, "digest changed"):
            corrected_legacy_evaluation._validate_validation_evidence_data(
                replace(validation, contract_sha256="0" * 64),
                self.passage_index,
            )

        test = corrected_legacy_evaluation.build_corrected_legacy_test_data(
            all_queries=self.queries,
            corpus_by_passage_id=self.corpus,
            passage_index_table=self.passage_index,
            test_case_ids=TEST_CASE_IDS,
            query_view="flat_masked",
        )
        with self.assertRaisesRegex(ValueError, "contract digest changed"):
            corrected_legacy_evaluation._validate_test_data(
                replace(test, contract_sha256="f" * 64),
                self.passage_index,
            )


if __name__ == "__main__":
    unittest.main()
