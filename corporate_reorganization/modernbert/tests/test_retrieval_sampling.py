from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from corporate_reorganization.modernbert.retriever.data import (
    CorpusPassage,
    QueryExample,
    load_candidates_by_case,
    load_corpus,
    load_queries,
)
from corporate_reorganization.modernbert.retriever.legacy_sampling import (
    MultiPositiveRetrievalTrainDataset as LegacyRetrievalTrainDataset,
    select_distractor_passage_ids,
)
from corporate_reorganization.modernbert.retriever.sampling import (
    MAX_EXPLICIT_POSITIVES,
    NUM_EXPLICIT_NEGATIVES,
    NUM_LOCAL_OTHER_CASE_NEGATIVES,
    NUM_LOCAL_SAME_CASE_NEGATIVES,
    SAMPLER_GLOBAL_UNIFORM,
    SAMPLER_LOCAL_UNIQUE,
    SELECTION_ALGORITHM,
    ControlledRetrievalTrainDataset,
    digest_ranked_sample,
    sampling_trace_checksum,
    validate_sampling_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
DATASET_DIR = REPO_ROOT / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
FOLDS_CONFIG = MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json"
EXPERIMENT_CONFIG = MODERNBERT_DIR / "experiments/retrieval_cv/configs/experiment.json"


def make_query(
    query_id: str,
    doc_id: str,
    positives: Sequence[str],
    *,
    visible: Sequence[str] = (),
) -> QueryExample:
    return QueryExample(
        query_id=query_id,
        doc_id=doc_id,
        motion_root_id="root",
        mask_parent_id="parent",
        query_text=f"structured {query_id}",
        positive_passage_ids=list(positives),
        positive_labels=["Analysis"] * len(positives),
        visible_passage_ids=list(visible),
        flat_query_text_plain=f"flat {query_id}",
        flat_query_text_masked=f"flat {query_id} [MASK]",
    )


def make_fixture(
    case_sizes: Sequence[Tuple[str, int]] = (("1", 45), ("2", 30), ("3", 10)),
) -> Tuple[Dict[str, CorpusPassage], Dict[str, List[str]], List[QueryExample]]:
    corpus: Dict[str, CorpusPassage] = {}
    candidates: Dict[str, List[str]] = {}
    for doc_id, size in case_sizes:
        passage_ids = [f"{doc_id}::p{i:03d}" for i in range(size)]
        candidates[doc_id] = passage_ids
        for passage_id in passage_ids:
            corpus[passage_id] = CorpusPassage(
                passage_id=passage_id,
                doc_id=doc_id,
                label="Analysis",
                text=f"text {passage_id}",
            )
    queries = [
        make_query(
            "q1",
            "1",
            [f"1::p{i:03d}" for i in range(5)],
            visible=["1::p006"],
        ),
        make_query("q2", "1", ["1::p005"]),
    ]
    return corpus, candidates, queries


class LegacySamplingIsolationTest(unittest.TestCase):
    def test_reconstructed_march_sample_is_frozen(self) -> None:
        queries = [
            make_query("q1", "1", ["1::p0", "1::p1"]),
            make_query("q2", "1", ["1::p2"]),
        ]
        dataset = LegacyRetrievalTrainDataset(
            queries,
            {"1": [f"1::p{i}" for i in range(10)]},
            [f"2::d{i}" for i in range(6)],
            base_seed=17,
            max_pos_per_query=4,
            num_same_case_negatives=2,
            num_distractor_negatives=2,
        )
        self.assertEqual(
            dataset[0]["candidate_passage_ids"],
            ["1::p1", "1::p0", "1::p7", "1::p8", "1::p6", "1::p4", "2::d3", "2::d5"],
        )
        self.assertNotIn("1::p2", dataset[0]["candidate_passage_ids"])

        dataset.set_epoch(3)
        self.assertEqual(
            dataset[0]["candidate_passage_ids"],
            ["1::p1", "1::p0", "1::p8", "1::p3", "1::p5", "1::p4", "2::d3", "2::d4"],
        )

    def test_reconstructed_march_replacement_path_is_retained(self) -> None:
        dataset = LegacyRetrievalTrainDataset(
            [make_query("q", "1", ["1::positive"])],
            {"1": ["1::positive", "1::negative"]},
            ["2::distractor"],
            base_seed=17,
            max_pos_per_query=4,
            num_same_case_negatives=2,
            num_distractor_negatives=2,
        )
        candidates = dataset[0]["candidate_passage_ids"]
        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates.count("1::negative"), 5)
        self.assertEqual(candidates.count("2::distractor"), 2)

    def test_reconstructed_march_all_same_case_branch_is_frozen(self) -> None:
        queries = [
            make_query("q1", "1", ["1::p0"]),
            make_query("q2", "1", ["1::p1"]),
        ]
        dataset = LegacyRetrievalTrainDataset(
            queries,
            {
                "1": ["1::p0", "1::p1", "1::p2"],
                "2": [f"2::p{i}" for i in range(5)],
            },
            ["2::d0", "2::d1"],
            base_seed=17,
            max_pos_per_query=4,
            num_same_case_negatives=-1,
            num_distractor_negatives=4,
        )
        self.assertEqual(
            dataset[0]["candidate_passage_ids"],
            [
                "1::p0",
                "1::p2",
                "1::p2",
                "1::p0",
                "1::p2",
                "2::d0",
                "2::d1",
                "2::d1",
                "2::d1",
            ],
        )
        self.assertNotIn("1::p1", dataset[0]["candidate_passage_ids"])

    def test_reconstructed_march_56_plus_4_background_configuration_is_frozen(self) -> None:
        corpus = {
            **{
                f"1::p{i}": CorpusPassage(f"1::p{i}", "1", "Analysis", f"case one {i}")
                for i in range(70)
            },
            **{
                f"2::background{i}": CorpusPassage(
                    f"2::background{i}",
                    "2",
                    "Background Facts",
                    f"background {i}",
                )
                for i in range(5)
            },
            **{
                f"2::analysis{i}": CorpusPassage(
                    f"2::analysis{i}",
                    "2",
                    "Analysis",
                    f"analysis {i}",
                )
                for i in range(5)
            },
        }
        distractors = select_distractor_passage_ids(
            corpus,
            distractor_labels=["Background Facts"],
        )
        self.assertEqual(distractors, [f"2::background{i}" for i in range(5)])
        dataset = LegacyRetrievalTrainDataset(
            [make_query("q", "1", ["1::p0", "1::p1"])],
            {"1": [f"1::p{i}" for i in range(70)]},
            distractors,
            base_seed=17,
            max_pos_per_query=4,
            num_same_case_negatives=56,
            num_distractor_negatives=4,
        )
        candidate_ids = dataset[0]["candidate_passage_ids"]
        self.assertEqual(len(candidate_ids), 64)
        self.assertEqual(set(candidate_ids[:2]), {"1::p0", "1::p1"})
        self.assertTrue(all(passage_id.startswith("1::") for passage_id in candidate_ids[2:60]))
        self.assertTrue(
            all(passage_id.startswith("2::background") for passage_id in candidate_ids[60:])
        )


class DigestRankedSelectionTest(unittest.TestCase):
    def test_selection_is_pool_order_invariant_and_seed_epoch_sensitive(self) -> None:
        pool = [f"p{i:03d}" for i in range(100)]
        kwargs = {
            "experiment_seed": 17,
            "epoch": 0,
            "query_id": "q",
            "component": "test",
        }
        selected = digest_ranked_sample(pool, 10, **kwargs)
        self.assertEqual(
            selected,
            ["p064", "p079", "p010", "p015", "p077", "p091", "p071", "p000", "p073", "p036"],
        )
        self.assertEqual(selected, digest_ranked_sample(list(reversed(pool)), 10, **kwargs))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertNotEqual(
            selected,
            digest_ranked_sample(pool, 10, **{**kwargs, "experiment_seed": 29}),
        )
        self.assertNotEqual(
            selected,
            digest_ranked_sample(pool, 10, **{**kwargs, "epoch": 1}),
        )

    def test_exact_sampling_failures_are_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "replacement is forbidden"):
            digest_ranked_sample(
                ["a"],
                2,
                experiment_seed=17,
                epoch=0,
                query_id="q",
                component="test",
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            digest_ranked_sample(
                ["a", "a"],
                1,
                experiment_seed=17,
                epoch=0,
                query_id="q",
                component="test",
            )
        with self.assertRaises(TypeError):
            digest_ranked_sample(
                ["a"],
                1,
                experiment_seed=True,
                epoch=0,
                query_id="q",
                component="test",
            )


class ControlledSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus, self.candidates, self.queries = make_fixture()

    def make_dataset(
        self,
        sampler: str,
        *,
        seed: int = 17,
        queries: Sequence[QueryExample] | None = None,
        query_view: str = "structured",
    ) -> ControlledRetrievalTrainDataset:
        return ControlledRetrievalTrainDataset(
            self.queries if queries is None else queries,
            self.corpus,
            self.candidates,
            ["1", "2", "3"],
            sampler=sampler,
            experiment_seed=seed,
            query_view=query_view,
        )

    def test_local_unique_exact_quotas_eligibility_and_union_sampling(self) -> None:
        dataset = self.make_dataset(SAMPLER_LOCAL_UNIQUE)
        sample = dataset[0]
        trace = sample["sampling_trace"]
        validate_sampling_trace(trace)
        self.assertEqual(
            sample["sampling_trace_sha256"],
            "31a20b3e8a31cd4cf0a5a15e347362f31c4d3125f391068fe47468816474c03f",
        )

        self.assertEqual(len(sample["selected_positive_passage_ids"]), MAX_EXPLICIT_POSITIVES)
        self.assertEqual(len(sample["candidate_passage_ids"]), MAX_EXPLICIT_POSITIVES + NUM_EXPLICIT_NEGATIVES)
        self.assertEqual(len(sample["candidate_passage_ids"]), len(set(sample["candidate_passage_ids"])))
        self.assertEqual(
            len(trace["negative_passage_ids_by_stratum"]["same_case"]),
            NUM_LOCAL_SAME_CASE_NEGATIVES,
        )
        self.assertEqual(
            len(trace["negative_passage_ids_by_stratum"]["other_case"]),
            NUM_LOCAL_OTHER_CASE_NEGATIVES,
        )

        all_golds = set(self.queries[0].positive_passage_ids)
        negatives = {
            passage_id
            for values in trace["negative_passage_ids_by_stratum"].values()
            for passage_id in values
        }
        self.assertFalse(all_golds.intersection(negatives))
        self.assertIn("1::p005", negatives, "another query's gold must remain eligible")
        self.assertIn("1::p006", negatives, "visible context must remain eligible")

        other_union = [*self.candidates["2"], *self.candidates["3"]]
        expected_other = digest_ranked_sample(
            other_union,
            NUM_LOCAL_OTHER_CASE_NEGATIVES,
            experiment_seed=17,
            epoch=0,
            query_id="q1",
            component="local_other_case_negative",
        )
        self.assertEqual(trace["negative_passage_ids_by_stratum"]["other_case"], expected_other)

    def test_global_uniform_uses_all_training_passages_and_exactly_sixty_negatives(self) -> None:
        dataset = self.make_dataset(SAMPLER_GLOBAL_UNIFORM)
        sample = dataset[0]
        trace = sample["sampling_trace"]
        global_negatives = trace["negative_passage_ids_by_stratum"]["global"]
        self.assertEqual(len(global_negatives), NUM_EXPLICIT_NEGATIVES)
        self.assertEqual(len(global_negatives), len(set(global_negatives)))
        self.assertTrue(any(passage_id.startswith("1::") for passage_id in global_negatives))
        self.assertTrue(any(passage_id.startswith("2::") for passage_id in global_negatives))
        self.assertTrue(any(passage_id.startswith("3::") for passage_id in global_negatives))
        self.assertFalse(set(self.queries[0].positive_passage_ids).intersection(global_negatives))

    def test_fewer_than_four_golds_do_not_trigger_negative_compensation(self) -> None:
        for sampler in (SAMPLER_LOCAL_UNIQUE, SAMPLER_GLOBAL_UNIFORM):
            with self.subTest(sampler=sampler):
                sample = self.make_dataset(sampler)[1]
                self.assertEqual(sample["selected_positive_passage_ids"], ["1::p005"])
                self.assertEqual(len(sample["candidate_passage_ids"]), 61)
                negative_count = sum(
                    len(values)
                    for values in sample["sampling_trace"]["negative_passage_ids_by_stratum"].values()
                )
                self.assertEqual(negative_count, NUM_EXPLICIT_NEGATIVES)

    def test_replay_query_order_and_matched_positive_selection(self) -> None:
        local = self.make_dataset(SAMPLER_LOCAL_UNIQUE)
        replay = self.make_dataset(SAMPLER_LOCAL_UNIQUE)
        reversed_dataset = self.make_dataset(
            SAMPLER_LOCAL_UNIQUE,
            queries=list(reversed(self.queries)),
        )
        reordered_pools = {
            doc_id: list(reversed(passage_ids))
            for doc_id, passage_ids in reversed(list(self.candidates.items()))
        }
        reordered_pool_dataset = ControlledRetrievalTrainDataset(
            self.queries,
            self.corpus,
            reordered_pools,
            ["3", "2", "1"],
            sampler=SAMPLER_LOCAL_UNIQUE,
            experiment_seed=17,
        )
        global_dataset = self.make_dataset(SAMPLER_GLOBAL_UNIFORM)
        flat_dataset = self.make_dataset(SAMPLER_LOCAL_UNIQUE, query_view="flat_masked")

        self.assertEqual(local[0], replay[0])
        self.assertEqual(local[0]["sampling_trace"], reversed_dataset[1]["sampling_trace"])
        self.assertEqual(local[0]["sampling_trace"], reordered_pool_dataset[0]["sampling_trace"])
        self.assertEqual(
            local[0]["selected_positive_passage_ids"],
            global_dataset[0]["selected_positive_passage_ids"],
        )
        self.assertEqual(
            local[0]["selected_positive_passage_ids"],
            flat_dataset[0]["selected_positive_passage_ids"],
        )
        self.assertEqual(local[0]["sampling_trace_sha256"], flat_dataset[0]["sampling_trace_sha256"])

        many_positive_reordered = copy.deepcopy(self.queries)
        many_positive_reordered[0].positive_passage_ids.reverse()
        self.assertEqual(
            replay[0],
            self.make_dataset(
                SAMPLER_LOCAL_UNIQUE,
                queries=many_positive_reordered,
            )[0],
        )

        few_positive_query = make_query("few", "1", ["1::p005", "1::p006"])
        few_positive_reordered = copy.deepcopy(few_positive_query)
        few_positive_reordered.positive_passage_ids.reverse()
        self.assertEqual(
            self.make_dataset(SAMPLER_LOCAL_UNIQUE, queries=[few_positive_query])[0],
            self.make_dataset(SAMPLER_LOCAL_UNIQUE, queries=[few_positive_reordered])[0],
        )

        local.set_epoch(1)
        self.assertNotEqual(local[0]["sampling_trace_sha256"], replay[0]["sampling_trace_sha256"])
        seed_changed = self.make_dataset(SAMPLER_LOCAL_UNIQUE, seed=29)
        self.assertNotEqual(local[0]["sampling_trace_sha256"], seed_changed[0]["sampling_trace_sha256"])

    def test_trace_tampering_is_detected(self) -> None:
        trace = copy.deepcopy(self.make_dataset(SAMPLER_LOCAL_UNIQUE)[0]["sampling_trace"])
        trace["candidate_passage_ids"][0] = "tampered"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_sampling_trace(trace)

    def test_trace_schema_rejects_malformed_types_with_valid_checksums(self) -> None:
        valid_trace = self.make_dataset(SAMPLER_LOCAL_UNIQUE)[0]["sampling_trace"]
        malformed_traces = []

        query_id_integer = copy.deepcopy(valid_trace)
        query_id_integer["query_id"] = 123
        malformed_traces.append(query_id_integer)

        empty_doc_id = copy.deepcopy(valid_trace)
        empty_doc_id["doc_id"] = ""
        malformed_traces.append(empty_doc_id)

        boolean_schema = copy.deepcopy(valid_trace)
        boolean_schema["schema_version"] = True
        malformed_traces.append(boolean_schema)

        tuple_positives = copy.deepcopy(valid_trace)
        tuple_positives["positive_passage_ids"] = tuple(tuple_positives["positive_passage_ids"])
        malformed_traces.append(tuple_positives)

        non_string_selected = copy.deepcopy(valid_trace)
        non_string_selected["selected_positive_passage_ids"][0] = 7
        malformed_traces.append(non_string_selected)

        tuple_stratum = copy.deepcopy(valid_trace)
        tuple_stratum["negative_passage_ids_by_stratum"]["same_case"] = tuple(
            tuple_stratum["negative_passage_ids_by_stratum"]["same_case"]
        )
        malformed_traces.append(tuple_stratum)

        boolean_pool_size = copy.deepcopy(valid_trace)
        boolean_pool_size["eligible_pool_sizes_by_stratum"]["same_case"] = True
        malformed_traces.append(boolean_pool_size)

        non_string_candidate = copy.deepcopy(valid_trace)
        non_string_candidate["candidate_passage_ids"][0] = 9
        malformed_traces.append(non_string_candidate)

        for malformed in malformed_traces:
            with self.subTest(field_difference=set(malformed) ^ set(valid_trace)):
                payload = dict(malformed)
                payload.pop("trace_sha256")
                malformed["trace_sha256"] = sampling_trace_checksum(payload)
                with self.assertRaises((TypeError, ValueError)):
                    validate_sampling_trace(malformed)

        bad_checksum_format = copy.deepcopy(valid_trace)
        bad_checksum_format["trace_sha256"] = "A" * 64
        with self.assertRaisesRegex(ValueError, "lowercase hexadecimal"):
            validate_sampling_trace(bad_checksum_format)

    def test_invalid_inputs_and_infeasible_quotas_fail_at_construction(self) -> None:
        small_corpus, small_candidates, small_queries = make_fixture((("1", 40), ("2", 20)))
        with self.assertRaisesRegex(ValueError, "exact quota is 40"):
            ControlledRetrievalTrainDataset(
                small_queries,
                small_corpus,
                small_candidates,
                ["1", "2"],
                sampler=SAMPLER_LOCAL_UNIQUE,
                experiment_seed=17,
            )

        incomplete = copy.deepcopy(self.candidates)
        incomplete["1"].pop()
        with self.assertRaisesRegex(ValueError, "not the complete case corpus"):
            ControlledRetrievalTrainDataset(
                self.queries,
                self.corpus,
                incomplete,
                ["1", "2", "3"],
                sampler=SAMPLER_GLOBAL_UNIFORM,
                experiment_seed=17,
            )

        with self.assertRaisesRegex(ValueError, "non-training"):
            ControlledRetrievalTrainDataset(
                [make_query("outside", "4", ["4::p000"])],
                self.corpus,
                self.candidates,
                ["1", "2", "3"],
                sampler=SAMPLER_GLOBAL_UNIFORM,
                experiment_seed=17,
            )


class FrozenDatasetSamplingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(DATASET_DIR)
        cls.queries = load_queries(DATASET_DIR, "all")
        cls.candidates = load_candidates_by_case(DATASET_DIR)
        cls.folds = json.loads(FOLDS_CONFIG.read_text(encoding="utf-8"))
        cls.experiment = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_matches_sampler_constants(self) -> None:
        samplers = self.experiment["samplers"]
        self.assertEqual(samplers["common"]["max_explicit_positives"], MAX_EXPLICIT_POSITIVES)
        self.assertEqual(samplers["common"]["explicit_negatives"], NUM_EXPLICIT_NEGATIVES)
        self.assertFalse(samplers["common"]["replacement"])
        self.assertEqual(samplers["local_unique"]["same_case_negatives"], NUM_LOCAL_SAME_CASE_NEGATIVES)
        self.assertEqual(samplers["local_unique"]["other_case_negatives"], NUM_LOCAL_OTHER_CASE_NEGATIVES)
        self.assertEqual(samplers["local_unique"]["sampling_unit"], "passage")
        self.assertEqual(samplers["global_uniform"]["sampling_unit"], "passage")
        self.assertEqual(samplers["selection"]["algorithm"], SELECTION_ALGORITHM)

    def test_every_outer_training_rotation_is_feasible_for_both_samplers(self) -> None:
        final_folds = {
            int(fold["fold_id"]): list(fold["case_ids"])
            for fold in self.folds["folds"]
        }
        for outer_fold in range(5):
            validation_fold = (outer_fold + 1) % 5
            training_doc_ids = [
                doc_id
                for fold_id in range(5)
                if fold_id not in {outer_fold, validation_fold}
                for doc_id in final_folds[fold_id]
            ]
            training_doc_id_set = set(training_doc_ids)
            training_queries = [
                query for query in self.queries if query.doc_id in training_doc_id_set
            ]
            self.assertEqual(len(training_queries), 294)
            for sampler in (SAMPLER_LOCAL_UNIQUE, SAMPLER_GLOBAL_UNIFORM):
                with self.subTest(outer_fold=outer_fold, sampler=sampler):
                    dataset = ControlledRetrievalTrainDataset(
                        training_queries,
                        self.corpus,
                        self.candidates,
                        training_doc_ids,
                        sampler=sampler,
                        experiment_seed=17,
                    )
                    self.assertEqual(len(dataset), 294)
                    validate_sampling_trace(dataset[0]["sampling_trace"])
                    validate_sampling_trace(dataset[len(dataset) - 1]["sampling_trace"])


if __name__ == "__main__":
    unittest.main()
