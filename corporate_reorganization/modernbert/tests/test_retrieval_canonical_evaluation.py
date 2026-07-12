from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.data import CorpusPassage, QueryExample, load_corpus, load_queries
from retriever.evaluation import (
    build_canonical_evaluation_data,
    canonical_result_from_payload,
    compute_canonical_retrieval_result_from_scores,
    rank_complete_scores,
)
from retriever.regimes import (
    CANONICAL_CANDIDATE_REGIMES,
    REGIME_FOLD_GLOBAL,
    REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
    REGIME_GLOBAL_SPLIT,
    REGIME_SAME_CASE_FULL,
    REGIME_SAME_CASE_LEGACY,
    normalize_candidate_regime,
    normalize_legacy_candidate_regime,
)


DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
FOLD_CONFIG = (
    REPO_ROOT
    / "corporate_reorganization/modernbert/experiments/retrieval_cv/configs/folds.json"
)


def _query(
    query_id: str,
    doc_id: str,
    gold: list[str],
    visible: list[str],
) -> QueryExample:
    return QueryExample(
        query_id=query_id,
        doc_id=doc_id,
        motion_root_id="root",
        mask_parent_id="target",
        query_text="structured [MASK]",
        positive_passage_ids=list(gold),
        positive_labels=["Analysis"],
        visible_passage_ids=list(visible),
        flat_query_text_masked="flat [MASK]",
    )


def _fixture():
    corpus = {
        passage_id: CorpusPassage(
            passage_id=passage_id,
            doc_id=passage_id.split("::", 1)[0],
            label="Analysis",
            text=f"text {passage_id}",
        )
        for passage_id in (
            "c1::p1",
            "c1::p2",
            "c1::p3",
            "c1::p4",
            "c1::p5",
            "c2::p1",
            "c2::p2",
            "c2::p3",
            "c2::p4",
        )
    }
    queries = [
        _query("c1::q1", "c1", ["c1::p1", "c1::p2"], ["c1::p1", "c1::p3"]),
        _query("c1::q2", "c1", ["c1::p2", "c1::p4"], ["c1::p5"]),
        _query("c2::q1", "c2", ["c2::p1"], ["c2::p2"]),
    ]
    return corpus, queries


class CanonicalRegimeContractTest(unittest.TestCase):
    def test_exact_four_regimes_and_legacy_alias_boundary(self) -> None:
        self.assertEqual(
            CANONICAL_CANDIDATE_REGIMES,
            (
                REGIME_SAME_CASE_LEGACY,
                REGIME_SAME_CASE_FULL,
                REGIME_FOLD_GLOBAL,
                REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
            ),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported candidate regime"):
            normalize_candidate_regime(REGIME_GLOBAL_SPLIT)
        self.assertEqual(
            normalize_legacy_candidate_regime(REGIME_GLOBAL_SPLIT),
            REGIME_FOLD_GLOBAL,
        )

    def test_all_four_exact_pools_gold_precedence_and_context_filter(self) -> None:
        corpus, queries = _fixture()
        data_by_regime = {
            regime: build_canonical_evaluation_data(
                all_queries=queries,
                corpus_by_passage_id=corpus,
                evaluated_case_ids=["c2", "c1"],
                role="test",
                regime_name=regime,
            )
            for regime in CANONICAL_CANDIDATE_REGIMES
        }
        legacy = data_by_regime[REGIME_SAME_CASE_LEGACY]
        self.assertEqual(legacy.case_ids, ("c1", "c2"))
        self.assertEqual([query.query_id for query in legacy.queries], ["c1::q1", "c1::q2", "c2::q1"])
        self.assertEqual(
            legacy.candidate_ids_by_query[0],
            ("c1::p1", "c1::p2", "c1::p3", "c1::p5"),
        )
        self.assertEqual(
            legacy.candidate_ids_by_query[1],
            ("c1::p2", "c1::p3", "c1::p4", "c1::p5"),
        )
        self.assertIn("c1::p2", legacy.candidate_ids_by_query[0])
        self.assertIn("c1::p2", legacy.candidate_ids_by_query[1])

        same_case = data_by_regime[REGIME_SAME_CASE_FULL]
        self.assertEqual(same_case.candidate_ids_by_query[0], tuple(sorted(corpus)[:5]))
        fold_global = data_by_regime[REGIME_FOLD_GLOBAL]
        self.assertTrue(
            all(pool == tuple(sorted(corpus)) for pool in fold_global.candidate_ids_by_query)
        )
        context = data_by_regime[REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED]
        self.assertIn("c1::p1", context.candidate_ids_by_query[0], "visible gold must win")
        self.assertNotIn("c1::p3", context.candidate_ids_by_query[0])
        self.assertNotIn("c1::p5", context.candidate_ids_by_query[1])
        self.assertNotIn("c2::p2", context.candidate_ids_by_query[2])
        for data in data_by_regime.values():
            for query, candidates in zip(data.queries, data.candidate_ids_by_query):
                self.assertTrue(set(query.gold_passage_ids).issubset(candidates))

    def test_invalid_role_inventories_and_query_sets_fail_loudly(self) -> None:
        corpus, queries = _fixture()
        kwargs = dict(
            all_queries=queries,
            corpus_by_passage_id=corpus,
            evaluated_case_ids=["c1", "c2"],
            role="test",
            regime_name=REGIME_FOLD_GLOBAL,
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            build_canonical_evaluation_data(**{**kwargs, "evaluated_case_ids": ["c1", "c1"]})
        with self.assertRaisesRegex(ValueError, "Role passages"):
            build_canonical_evaluation_data(**{**kwargs, "evaluated_case_ids": ["c1", "missing"]})
        with self.assertRaisesRegex(ValueError, "Role queries"):
            build_canonical_evaluation_data(**{**kwargs, "all_queries": queries[:2]})
        duplicate_gold = list(queries)
        duplicate_gold[0] = replace(
            duplicate_gold[0],
            positive_passage_ids=["c1::p1", "c1::p1"],
        )
        with self.assertRaisesRegex(ValueError, "unique non-empty gold"):
            build_canonical_evaluation_data(**{**kwargs, "all_queries": duplicate_gold})
        duplicate_visible = list(queries)
        duplicate_visible[0] = replace(
            duplicate_visible[0],
            visible_passage_ids=["c1::p3", "c1::p3"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate visible"):
            build_canonical_evaluation_data(**{**kwargs, "all_queries": duplicate_visible})

    def test_every_regime_filters_one_identical_source_ranking(self) -> None:
        corpus, queries = _fixture()
        data_by_regime = {
            regime: build_canonical_evaluation_data(
                all_queries=queries,
                corpus_by_passage_id=corpus,
                evaluated_case_ids=["c1", "c2"],
                role="test",
                regime_name=regime,
            )
            for regime in CANONICAL_CANDIDATE_REGIMES
        }
        passage_ids = data_by_regime[REGIME_FOLD_GLOBAL].passage_ids
        scores = torch.arange(
            len(queries) * len(passage_ids),
            dtype=torch.float32,
        ).reshape(len(queries), len(passage_ids))
        results = {
            regime: compute_canonical_retrieval_result_from_scores(
                scores=scores,
                evaluation_data=data,
            )
            for regime, data in data_by_regime.items()
        }
        self.assertEqual(
            len({result.source_ranking_sha256 for result in results.values()}),
            1,
        )
        for regime, result in results.items():
            data = data_by_regime[regime]
            for query_index, (source, final) in enumerate(
                zip(result.source_rankings, result.rankings)
            ):
                candidate_ids = set(data.candidate_ids_by_query[query_index])
                expected = [
                    dict(candidate)
                    for candidate in source["ranked_candidates"]
                    if candidate["passage_id"] in candidate_ids
                ]
                for rank, candidate in enumerate(expected, start=1):
                    candidate["rank"] = rank
                self.assertEqual(
                    [dict(candidate) for candidate in final["ranked_candidates"]],
                    expected,
                )


class CanonicalRankingMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus, self.queries = _fixture()
        self.data = build_canonical_evaluation_data(
            all_queries=self.queries,
            corpus_by_passage_id=self.corpus,
            evaluated_case_ids=["c1", "c2"],
            role="test",
            regime_name=REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
        )
        passage_position = {pid: i for i, pid in enumerate(self.data.passage_ids)}
        self.scores = torch.zeros((3, len(self.data.passage_ids)), dtype=torch.float64)
        self.scores[0, passage_position["c1::p2"]] = 2.0
        self.scores[0, passage_position["c1::p1"]] = 1.0
        self.scores[1, passage_position["c1::p4"]] = 3.0
        self.scores[2, passage_position["c2::p1"]] = -1.0
        self.result = compute_canonical_retrieval_result_from_scores(
            scores=self.scores,
            evaluation_data=self.data,
        )

    def test_stable_complete_rankings_metrics_aggregation_and_lineage(self) -> None:
        tied = rank_complete_scores(
            passage_ids=["p3", "p1", "p2"],
            scores=[1.0, 1.0, 1.0],
        )
        self.assertEqual([row["passage_id"] for row in tied], ["p1", "p2", "p3"])
        self.assertEqual([row["rank"] for row in tied], [1, 2, 3])

        payload = self.result.to_payload()
        self.assertEqual(len(payload["source_rankings"]), 3)
        self.assertEqual(
            len(payload["source_rankings"][0]["ranked_candidates"]),
            len(self.data.passage_ids),
        )
        for query_position, (source, filtered) in enumerate(
            zip(payload["source_rankings"], payload["rankings"])
        ):
            self.assertEqual(filtered["parent_ranking_sha256"], source["ranking_sha256"])
            source_by_id = {row["passage_id"]: row["score"] for row in source["ranked_candidates"]}
            expected_ids = [
                row["passage_id"]
                for row in source["ranked_candidates"]
                if row["passage_id"] in self.data.candidate_ids_by_query[query_position]
            ]
            self.assertEqual(
                [row["passage_id"] for row in filtered["ranked_candidates"]],
                expected_ids,
            )
            self.assertTrue(
                all(source_by_id[row["passage_id"]] == row["score"] for row in filtered["ranked_candidates"])
            )
        first = payload["per_query"][0]
        self.assertEqual(first["gold_passage_ids"], ["c1::p1", "c1::p2"])
        self.assertEqual(first["visible_passage_ids"], ["c1::p1", "c1::p3"])
        self.assertEqual(first["first_gold_rank"], 1)
        self.assertIs(type(first["gold_count"]), int)
        self.assertIs(type(first["first_gold_rank"]), int)
        self.assertIs(type(first["candidate_count"]), int)
        self.assertEqual(first["hit_at_1"], 1.0)
        self.assertEqual(first["set_recall_at_1"], 0.5)
        self.assertEqual(first["exact_target_recovery_at_1"], 0.0)
        self.assertEqual(first["exact_target_recovery_at_5"], 1.0)
        self.assertEqual(first["first_gold_reciprocal_rank_full_ranking"], 1.0)
        self.assertEqual(
            payload["metrics"]["query_micro_hit_at_1"],
            math.fsum(row["hit_at_1"] for row in payload["per_query"]) / 3,
        )
        case_means = [row["metrics"]["hit_at_1"] for row in payload["per_case"]]
        self.assertEqual(
            payload["metrics"]["case_macro_hit_at_1"],
            math.fsum(case_means) / 2,
        )

    def test_strict_roundtrip_and_every_malformed_class_fails(self) -> None:
        payload = self.result.to_payload()
        roundtrip = canonical_result_from_payload(payload, self.data)
        self.assertEqual(roundtrip.to_payload(), payload)

        mutations = []
        extra = copy.deepcopy(payload)
        extra["unexpected"] = True
        mutations.append(extra)
        bad_digest = copy.deepcopy(payload)
        bad_digest["ranking_sha256"] = "0" * 64
        mutations.append(bad_digest)
        truncated = copy.deepcopy(payload)
        truncated["source_rankings"][0]["ranked_candidates"].pop()
        mutations.append(truncated)
        duplicate = copy.deepcopy(payload)
        duplicate["source_rankings"][0]["ranked_candidates"][1]["passage_id"] = duplicate[
            "source_rankings"
        ][0]["ranked_candidates"][0]["passage_id"]
        mutations.append(duplicate)
        unstable = copy.deepcopy(payload)
        tied_candidates = unstable["source_rankings"][0]["ranked_candidates"]
        tied_positions = next(
            (i, i + 1)
            for i in range(len(tied_candidates) - 1)
            if tied_candidates[i]["score"] == tied_candidates[i + 1]["score"]
        )
        left, right = tied_positions
        tied_candidates[left], tied_candidates[right] = (
            tied_candidates[right],
            tied_candidates[left],
        )
        tied_candidates[left]["rank"] = left + 1
        tied_candidates[right]["rank"] = right + 1
        mutations.append(unstable)
        nonfinite = copy.deepcopy(payload)
        nonfinite["source_rankings"][0]["ranked_candidates"][0]["score"] = float("nan")
        mutations.append(nonfinite)
        missing_gold = copy.deepcopy(payload)
        gold_id = missing_gold["per_query"][0]["gold_passage_ids"][0]
        missing_gold["rankings"][0]["ranked_candidates"] = [
            row
            for row in missing_gold["rankings"][0]["ranked_candidates"]
            if row["passage_id"] != gold_id
        ]
        mutations.append(missing_gold)
        changed_metric = copy.deepcopy(payload)
        changed_metric["per_query"][0]["set_recall_at_1"] = 0.25
        mutations.append(changed_metric)
        broken_parent = copy.deepcopy(payload)
        broken_parent["rankings"][0]["parent_ranking_sha256"] = "1" * 64
        mutations.append(broken_parent)

        for mutation in mutations:
            with self.subTest(keys=sorted(mutation)):
                with self.assertRaises(RuntimeError):
                    canonical_result_from_payload(mutation, self.data)

    def test_wrong_shape_nonfinite_scores_and_changed_k_fail(self) -> None:
        with self.assertRaises(ValueError):
            compute_canonical_retrieval_result_from_scores(
                scores=self.scores[:, :-1],
                evaluation_data=self.data,
            )
        nonfinite = self.scores.clone()
        nonfinite[0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            compute_canonical_retrieval_result_from_scores(
                scores=nonfinite,
                evaluation_data=self.data,
            )
        with self.assertRaisesRegex(ValueError, "exactly"):
            compute_canonical_retrieval_result_from_scores(
                scores=self.scores,
                evaluation_data=self.data,
                ks=(1, 5, 20),
            )

    def test_score_inputs_are_canonicalized_to_cpu_float32(self) -> None:
        from_float64 = compute_canonical_retrieval_result_from_scores(
            scores=self.scores.to(dtype=torch.float64),
            evaluation_data=self.data,
        )
        from_bfloat16 = compute_canonical_retrieval_result_from_scores(
            scores=self.scores.to(dtype=torch.bfloat16),
            evaluation_data=self.data,
        )
        from_float32 = compute_canonical_retrieval_result_from_scores(
            scores=self.scores.to(dtype=torch.float32),
            evaluation_data=self.data,
        )
        self.assertEqual(from_float64.to_payload(), from_float32.to_payload())
        self.assertEqual(from_bfloat16.to_payload(), from_float32.to_payload())


class FrozenCanonicalRoleInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(DATASET_DIR)
        cls.queries = load_queries(DATASET_DIR, "all")
        cls.folds = json.loads(FOLD_CONFIG.read_text(encoding="utf-8"))

    def test_all_five_test_roles_and_four_regimes_are_exact(self) -> None:
        overlap_pairs = {
            (
                "78::ROOT=mHknhScxBS::TARGET=qWL5wevcmI::MISSING=PREMISE_GROUP_1",
                "78::SENT_00103",
            ),
            (
                "86::ROOT=TngQAxOF5Y::TARGET=bBERqsOSbo::MISSING=PREMISE_GROUP_1",
                "86::SENT_00110",
            ),
        }
        retained_overlaps = set()
        for rotation in self.folds["rotations"]:
            role = rotation["test"]
            for regime in CANONICAL_CANDIDATE_REGIMES:
                data = build_canonical_evaluation_data(
                    all_queries=self.queries,
                    corpus_by_passage_id=self.corpus,
                    evaluated_case_ids=role["case_ids"],
                    role="test",
                    regime_name=regime,
                )
                self.assertEqual(data.query_count, 98)
                self.assertEqual(data.case_count, role["num_cases"])
                self.assertEqual(data.passage_count, role["passages"])
                for query, candidates in zip(data.queries, data.candidate_ids_by_query):
                    self.assertEqual(tuple(sorted(candidates)), candidates)
                    self.assertEqual(len(candidates), len(set(candidates)))
                    self.assertTrue(set(query.gold_passage_ids).issubset(candidates))
                    if regime == REGIME_FOLD_GLOBAL:
                        self.assertEqual(candidates, data.passage_ids)
                    if regime == REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED:
                        expected = tuple(
                            passage_id
                            for passage_id in data.passage_ids
                            if passage_id
                            not in (set(query.visible_passage_ids) - set(query.gold_passage_ids))
                        )
                        self.assertEqual(candidates, expected)
                        for passage_id in set(query.visible_passage_ids) & set(
                            query.gold_passage_ids
                        ):
                            retained_overlaps.add((query.query_id, passage_id))
                            self.assertIn(passage_id, candidates)
        self.assertEqual(retained_overlaps, overlap_pairs)


if __name__ == "__main__":
    unittest.main()
