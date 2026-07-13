from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from corporate_reorganization.modernbert.retriever.batching import SentinelQueryDataset
from corporate_reorganization.modernbert.retriever.data import PassageIndexTable
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_collator import (
    CorrectedLegacyDiagnosticCollator,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_data import (
    load_corrected_legacy_data,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_sampling import (
    CorrectedLegacyDiagnosticDataset,
)
from corporate_reorganization.modernbert.retriever.markup import SLOT_TOKEN


DATASET = (
    Path(__file__).resolve().parents[2]
    / "data/final_annotations_gold/processed_retrieval_v2"
)


class _Tokenizer:
    unk_token_id = -999
    truncation_side = "right"

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return 7 if token == SLOT_TOKEN else -999

    def __call__(self, texts, **kwargs):
        rows = []
        for text in texts:
            if text.count(SLOT_TOKEN) != 1:
                rows.append([1, 2])
            else:
                rows.append([1, 7])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 2), dtype=torch.long),
        }


class _RecordingTokenizer(_Tokenizer):
    def __init__(self) -> None:
        self.truncation_side = "right"
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(
            {
                "texts": tuple(texts),
                "truncation_side": self.truncation_side,
                **kwargs,
            }
        )
        return super().__call__(texts, **kwargs)


class CorrectedLegacyCollatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = load_corrected_legacy_data(DATASET)
        cls.index = PassageIndexTable(data.corpus_by_passage_id)
        cls.dataset = CorrectedLegacyDiagnosticDataset(
            data,
            experiment_seed=17,
            query_view="structured",
        )
        cls.collator = CorrectedLegacyDiagnosticCollator(
            _Tokenizer(),
            passage_index_table=cls.index,
            max_len_query=4_096,
        )

    def test_real_and_sentinel_rows_preserve_unique_multiplicity_contract(self) -> None:
        wrapped = SentinelQueryDataset(self.dataset)
        batch = self.collator([wrapped[0], wrapped[-1], wrapped[1], wrapped[-1]])
        self.assertEqual(batch["valid_query_count"].item(), 2)
        self.assertEqual(batch["candidate_passage_indices"].shape[0], 2)
        self.assertEqual(batch["candidate_multiplicities"].shape, batch["candidate_passage_indices"].shape)
        self.assertTrue(batch["candidate_multiplicities"].sum(dim=1).eq(64).all())
        self.assertEqual(len(batch["sampling_traces"]), 2)

    def test_occurrence_multiplicity_trace_and_slot_mutations_fail(self) -> None:
        wrapped = SentinelQueryDataset(self.dataset)
        original = wrapped[0]
        bad_multiplicity = copy.deepcopy(original)
        bad_multiplicity["candidate_multiplicities"][0] += 1
        with self.assertRaisesRegex(ValueError, "multiplicities"):
            self.collator([bad_multiplicity])

        bad_occurrence = copy.deepcopy(original)
        bad_occurrence["candidate_passage_occurrence_indices"][0] = (
            bad_occurrence["candidate_passage_occurrence_indices"][1]
        )
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            self.collator([bad_occurrence])

        bad_trace = copy.deepcopy(original)
        bad_trace["sampling_trace"]["query_id"] += "-changed"
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.collator([bad_trace])

        bad_slot = copy.deepcopy(original)
        bad_slot["query_text"] = bad_slot["query_text"].replace(SLOT_TOKEN, "missing")
        with self.assertRaisesRegex(ValueError, "retain one"):
            self.collator([bad_slot])

    def test_both_query_views_use_and_restore_focus_preserving_left_truncation(self) -> None:
        data = load_corrected_legacy_data(DATASET)
        for query_view in ("flat_masked", "structured"):
            with self.subTest(query_view=query_view):
                tokenizer = _RecordingTokenizer()
                dataset = CorrectedLegacyDiagnosticDataset(
                    data,
                    experiment_seed=17,
                    query_view=query_view,
                )
                collator = CorrectedLegacyDiagnosticCollator(
                    tokenizer,
                    passage_index_table=self.index,
                    max_len_query=4_096,
                )
                collator([SentinelQueryDataset(dataset)[0]])
                self.assertEqual(tokenizer.truncation_side, "right")
                self.assertEqual(len(tokenizer.calls), 1)
                call = tokenizer.calls[0]
                self.assertEqual(call["truncation_side"], "left")
                self.assertEqual(call["max_length"], 4_096)
                self.assertTrue(call["truncation"])
                self.assertEqual(call["texts"][0].count(SLOT_TOKEN), 1)


if __name__ == "__main__":
    unittest.main()
