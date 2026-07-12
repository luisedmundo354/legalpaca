from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.rankers import (  # noqa: E402
    complete_bm25_scores_from_hits,
    score_embedding_matrices,
    score_loaded_dual_encoder,
    score_loaded_mean_pool_encoder,
    validate_complete_score_matrix,
)


class _FakeTokenizer:
    def __init__(self, *, slot_token_id: int = 9) -> None:
        self.truncation_side = "right"
        self.slot_token_id = slot_token_id
        self.sides: list[str] = []

    def __call__(
        self,
        texts,
        *,
        truncation,
        max_length,
        padding,
        return_tensors,
    ):
        if truncation is not True or padding is not True or return_tensors != "pt":
            raise AssertionError("unexpected fake-tokenizer arguments")
        if max_length < 3:
            raise AssertionError("fake fixture requires at least three tokens")
        self.sides.append(self.truncation_side)
        rows: list[list[int]] = []
        for position, text in enumerate(texts):
            if "query" in text:
                rows.append([1, self.slot_token_id, 10 + position])
            else:
                rows.append([1, 20 + position, 2])
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class _FakeDualEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def encode_queries(self, input_ids, attention_mask):
        del attention_mask
        values = input_ids[:, -1].to(dtype=torch.float32)
        return torch.nn.functional.normalize(
            torch.stack((values, torch.ones_like(values)), dim=1),
            p=2,
            dim=-1,
        )

    def encode_passages(self, input_ids, attention_mask):
        del attention_mask
        values = input_ids[:, 1].to(dtype=torch.float32)
        return torch.nn.functional.normalize(
            torch.stack((torch.ones_like(values), values), dim=1),
            p=2,
            dim=-1,
        )


class _FakeMeanPoolEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, *, input_ids, attention_mask, return_dict):
        if return_dict is not True:
            raise AssertionError("controlled ranker must request a mapping-like output")
        hidden = torch.stack(
            (
                input_ids.to(dtype=torch.float32),
                attention_mask.to(dtype=torch.float32),
            ),
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


class CompleteScoreTest(unittest.TestCase):
    def test_embedding_scores_are_owned_cpu_float32_and_complete(self) -> None:
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        passage_embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        scores = score_embedding_matrices(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            query_ids=("q1", "q2"),
            passage_ids=("p1", "p2", "p3"),
            torch_module=torch,
        )
        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(scores.device.type, "cpu")
        self.assertTrue(torch.equal(scores, torch.tensor([[1, 0, 1], [0, 1, 1.0]])))

        query_embeddings[0, 0] = 0
        self.assertEqual(float(scores[0, 0]), 1.0)

    def test_complete_matrix_rejects_shape_order_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_complete_score_matrix(
                torch.ones((1, 1)),
                query_ids=("q1",),
                passage_ids=("p1", "p2"),
                torch_module=torch,
            )
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_complete_score_matrix(
                torch.ones((1, 2)),
                query_ids=("q1",),
                passage_ids=("p2", "p1"),
                torch_module=torch,
            )
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            validate_complete_score_matrix(
                torch.tensor([[float("nan")]]),
                query_ids=("q1",),
                passage_ids=("p1",),
                torch_module=torch,
            )


class Bm25ScoreTest(unittest.TestCase):
    def test_missing_hits_receive_zero_before_canonical_ranking(self) -> None:
        scores = complete_bm25_scores_from_hits(
            query_ids=("q1", "q2"),
            passage_ids=("p1", "p2", "p3"),
            hits_by_query={
                "q1": [{"passage_id": "p2", "score": 3.5}],
                "q2": [
                    {"passage_id": "p3", "score": 1.25},
                    {"passage_id": "p1", "score": 0.0},
                ],
            },
            torch_module=torch,
        )
        self.assertTrue(
            torch.equal(
                scores,
                torch.tensor([[0.0, 3.5, 0.0], [0.0, 0.0, 1.25]], dtype=torch.float32),
            )
        )

    def test_duplicate_foreign_and_nonfinite_hits_fail_loudly(self) -> None:
        base = {
            "query_ids": ("q1",),
            "passage_ids": ("p1", "p2"),
            "torch_module": torch,
        }
        for hits, pattern, exception in (
            (
                [
                    {"passage_id": "p1", "score": 1.0},
                    {"passage_id": "p1", "score": 0.0},
                ],
                "duplicate",
                ValueError,
            ),
            ([{"passage_id": "foreign", "score": 1.0}], "foreign", ValueError),
            (
                [{"passage_id": "p1", "score": float("inf")}],
                "non-finite",
                FloatingPointError,
            ),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(exception, pattern):
                complete_bm25_scores_from_hits(
                    **base,
                    hits_by_query={"q1": hits},
                )


class LoadedEncoderScoreTest(unittest.TestCase):
    def test_dual_encoder_requires_slot_and_restores_modes_and_tokenizer_side(self) -> None:
        model = _FakeDualEncoder()
        model.train()
        tokenizer = _FakeTokenizer(slot_token_id=9)
        scores = score_loaded_dual_encoder(
            model=model,
            tokenizer=tokenizer,
            query_ids=("q1", "q2"),
            query_texts=("query one", "query two"),
            passage_ids=("p1", "p2", "p3"),
            passage_texts=("passage one", "passage two", "passage three"),
            slot_token_id=9,
            query_batch_size=1,
            passage_batch_size=2,
            max_len_query=4096,
            max_len_passage=500,
            device="cpu",
            torch_module=torch,
        )
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertEqual(scores.dtype, torch.float32)
        self.assertTrue(model.training)
        self.assertEqual(tokenizer.truncation_side, "right")
        self.assertEqual(tokenizer.sides, ["left", "left", "right", "right"])

        with self.assertRaisesRegex(RuntimeError, "slot token"):
            score_loaded_dual_encoder(
                model=model,
                tokenizer=_FakeTokenizer(slot_token_id=8),
                query_ids=("q1",),
                query_texts=("query one",),
                passage_ids=("p1",),
                passage_texts=("passage one",),
                slot_token_id=9,
                query_batch_size=1,
                passage_batch_size=1,
                max_len_query=4096,
                max_len_passage=500,
                device="cpu",
                torch_module=torch,
            )

    def test_mean_pool_encoder_requires_explicit_query_truncation_side(self) -> None:
        model = _FakeMeanPoolEncoder()
        tokenizer = _FakeTokenizer()
        scores = score_loaded_mean_pool_encoder(
            model=model,
            tokenizer=tokenizer,
            query_ids=("q1",),
            query_texts=("query one",),
            passage_ids=("p1", "p2"),
            passage_texts=("passage one", "passage two"),
            query_prefix="query: ",
            passage_prefix="passage: ",
            query_truncation_side="right",
            query_batch_size=1,
            passage_batch_size=2,
            max_len_query=512,
            max_len_passage=512,
            device="cpu",
            torch_module=torch,
        )
        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertEqual(tokenizer.sides, ["right", "right"])
        with self.assertRaisesRegex(ValueError, "explicitly"):
            score_loaded_mean_pool_encoder(
                model=model,
                tokenizer=tokenizer,
                query_ids=("q1",),
                query_texts=("query one",),
                passage_ids=("p1",),
                passage_texts=("passage one",),
                query_prefix="query: ",
                passage_prefix="passage: ",
                query_truncation_side="auto",
                query_batch_size=1,
                passage_batch_size=1,
                max_len_query=512,
                max_len_passage=512,
                device="cpu",
                torch_module=torch,
            )


if __name__ == "__main__":
    unittest.main()
