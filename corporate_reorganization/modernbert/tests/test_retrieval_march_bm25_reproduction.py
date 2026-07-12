from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.bm25 import (  # noqa: E402
    build_and_score_bm25,
    build_bm25_index,
    validate_bm25_runtime,
)


PROCESSED_DIR = REPO_ROOT / "corporate_reorganization/data/final_annotations_gold/processed"
ARCHIVE_RANKINGS = (
    REPO_ROOT
    / "corporate_reorganization/test_results/retrieval_ablation_local/runs/rankings.jsonl"
)
TEST_CASES = {"37", "46", "65", "96"}


class Bm25PublicPreflightTest(unittest.TestCase):
    def test_runtime_failure_precedes_inputs_scratch_and_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            with (
                mock.patch(
                    "retriever.bm25.validate_bm25_runtime",
                    side_effect=RuntimeError("wrong sparse runtime"),
                ) as runtime_validator,
                mock.patch("retriever.bm25.build_bm25_index") as index_builder,
                self.assertRaisesRegex(RuntimeError, "wrong sparse runtime"),
            ):
                build_and_score_bm25(
                    query_ids=(),
                    query_texts=(),
                    passage_ids=(),
                    passage_texts=(),
                    scratch_dir=scratch,
                    torch_module=torch,
                )
            runtime_validator.assert_called_once_with()
            index_builder.assert_not_called()
            self.assertFalse(scratch.exists())

    def test_index_requires_atomic_absent_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            with self.assertRaisesRegex(FileExistsError, "absent"):
                build_bm25_index(
                    passage_ids=("p1",),
                    passage_texts=("text",),
                    scratch_dir=scratch,
                )


@unittest.skipUnless(
    os.environ.get("ARR_RUN_BM25_MARCH_INTEGRATION") == "1",
    "ARR_RUN_BM25_MARCH_INTEGRATION=1 is required",
)
class MarchBm25ReproductionTest(unittest.TestCase):
    def test_all_archived_bm25_rankings_and_scores_match_exactly(self) -> None:
        runtime = validate_bm25_runtime()
        self.assertEqual(runtime.pyserini, "1.5.0")
        corpus = [
            json.loads(line)
            for line in (PROCESSED_DIR / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        passages = sorted(
            (row for row in corpus if str(row["doc_id"]) in TEST_CASES),
            key=lambda row: row["passage_id"],
        )
        queries = sorted(
            (
                json.loads(line)
                for line in (PROCESSED_DIR / "queries/test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            key=lambda row: row["query_id"],
        )
        self.assertEqual(len(passages), 581)
        self.assertEqual(len(queries), 40)
        with tempfile.TemporaryDirectory() as temporary:
            scores = build_and_score_bm25(
                query_ids=[row["query_id"] for row in queries],
                query_texts=[row["flat_query_text_plain"] for row in queries],
                passage_ids=[row["passage_id"] for row in passages],
                passage_texts=[row["text"] for row in passages],
                scratch_dir=Path(temporary) / "bm25",
                torch_module=torch,
            )
        query_position = {row["query_id"]: position for position, row in enumerate(queries)}
        passage_position = {
            row["passage_id"]: position for position, row in enumerate(passages)
        }
        archive_rows = [
            json.loads(line)
            for line in ARCHIVE_RANKINGS.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["system"] == "bm25_flat"
        ]
        self.assertEqual(len(archive_rows), 120)
        candidate_scores_checked = 0
        for row in archive_rows:
            query_index = query_position[row["query_id"]]
            expected_candidates = row["ranked_candidates"]
            candidate_ids = [record["passage_id"] for record in expected_candidates]
            actual = sorted(
                (
                    {
                        "rank": 0,
                        "passage_id": passage_id,
                        "score": float(scores[query_index, passage_position[passage_id]].item()),
                    }
                    for passage_id in candidate_ids
                ),
                key=lambda record: (-record["score"], record["passage_id"]),
            )
            for rank, record in enumerate(actual, start=1):
                record["rank"] = rank
            self.assertEqual(actual, expected_candidates, msg=f"{row['regime']}::{row['query_id']}")
            candidate_scores_checked += len(actual)
        self.assertEqual(candidate_scores_checked, 35_071)


if __name__ == "__main__":
    unittest.main()
