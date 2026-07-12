from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from legacy_eval import trainer_eval  # noqa: E402
from retriever.data import CorpusPassage, QueryExample  # noqa: E402


class _LegacyTokenizer:
    def __init__(self) -> None:
        self.truncation_side = "unset"
        self.observed_sides: list[str] = []
        self.id_by_text = {
            "passage one": 1,
            "passage two": 2,
            "distractor": 3,
            "query one [MASK]": 11,
            "query two [MASK]": 12,
        }

    def __call__(
        self,
        texts,
        *,
        truncation,
        max_length,
        padding,
        return_tensors,
    ):
        if (
            truncation is not True
            or type(max_length) is not int
            or max_length < 1
            or padding is not True
            or return_tensors != "pt"
        ):
            raise AssertionError("Historical tokenizer call changed")
        self.observed_sides.append(self.truncation_side)
        input_ids = torch.tensor(
            [[self.id_by_text[text]] for text in texts],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class _LegacyRetriever(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_passages(self, input_ids, attention_mask):
        del attention_mask
        vectors = {
            1: (1.0, 0.0),
            2: (0.0, 1.0),
            3: (-1.0, -1.0),
        }
        return torch.tensor(
            [vectors[int(value)] for value in input_ids[:, 0].tolist()],
            dtype=torch.float32,
            device=input_ids.device,
        )

    def encode_queries(self, input_ids, attention_mask):
        del attention_mask
        vectors = {11: (1.0, 0.0), 12: (0.0, 1.0)}
        return torch.tensor(
            [vectors[int(value)] for value in input_ids[:, 0].tolist()],
            dtype=torch.float32,
            device=input_ids.device,
        )


def _fixture():
    corpus = {
        "c1::p1": CorpusPassage("c1::p1", "c1", "Analysis", "passage one"),
        "c1::p2": CorpusPassage("c1::p2", "c1", "Analysis", "passage two"),
        "c1::p3": CorpusPassage("c1::p3", "c1", "Analysis", "distractor"),
    }
    queries = [
        QueryExample(
            query_id="c1::q1",
            doc_id="c1",
            motion_root_id="root",
            mask_parent_id="p1",
            query_text="query one [MASK]",
            positive_passage_ids=["c1::p1"],
            positive_labels=["Analysis"],
            visible_passage_ids=[],
        ),
        QueryExample(
            query_id="c1::q2",
            doc_id="c1",
            motion_root_id="root",
            mask_parent_id="p2",
            query_text="query two [MASK]",
            positive_passage_ids=["c1::p2"],
            positive_labels=["Analysis"],
            visible_passage_ids=[],
        ),
    ]
    return corpus, queries


class LegacyTrainerEvaluationTest(unittest.TestCase):
    def test_trainer_import_uses_the_explicit_legacy_adapter(self) -> None:
        import trainer

        self.assertIs(trainer.evaluate_retrieval, trainer_eval.evaluate_retrieval)

    def test_historical_trainer_metrics_and_candidate_semantics_are_preserved(self) -> None:
        corpus, queries = _fixture()
        retriever = _LegacyRetriever()
        tokenizer = _LegacyTokenizer()
        with (
            mock.patch.object(trainer_eval, "load_corpus", return_value=corpus),
            mock.patch.object(
                trainer_eval,
                "load_candidates_by_case",
                return_value={"c1": list(corpus)},
            ),
            mock.patch.object(trainer_eval, "load_queries", return_value=queries),
            mock.patch.object(
                trainer_eval,
                "load_split_doc_ids",
                return_value=["c1"],
            ),
        ):
            result = trainer_eval.evaluate_retrieval(
                retriever,
                tokenizer,
                processed_dir=Path("unused-by-patched-loaders"),
                split="validation",
                max_len_query=4_096,
                max_len_passage=600,
                query_batch_size=1,
                passage_batch_size=2,
                ks=(1, 5, 10, 20, 50),
            )

        self.assertEqual(result.query_count, 2)
        self.assertEqual(result.corpus_count, 3)
        self.assertEqual(result.metrics["eval_num_queries"], 2.0)
        self.assertEqual(result.metrics["eval_recall_at_1"], 1.0)
        self.assertEqual(result.metrics["eval_recall_at_50"], 1.0)
        self.assertEqual(result.metrics["eval_mrr"], 1.0)
        self.assertEqual(result.metrics["eval_avg_candidates"], 2.0)
        self.assertTrue(math.isfinite(result.metrics["eval_retrieval_loss"]))
        self.assertEqual(tokenizer.observed_sides, ["right", "right", "left", "left"])
        self.assertFalse(retriever.training)


if __name__ == "__main__":
    unittest.main()
