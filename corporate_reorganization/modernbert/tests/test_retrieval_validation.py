from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
FOLD_CONFIG = (
    MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json"
)
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.data import (  # noqa: E402
    CorpusPassage,
    PassageIndexTable,
    QueryExample,
    load_corpus,
    load_queries,
)
from retriever.evaluation import (  # noqa: E402
    VALIDATION_FORWARD_STEPS,
    VALIDATION_MAX_LEN_PASSAGE,
    VALIDATION_MAX_LEN_QUERY,
    VALIDATION_PASSAGE_BATCH_CAP,
    VALIDATION_PRIMARY_METRIC,
    VALIDATION_QUERY_BATCH_CAP,
    VALIDATION_WORLD_SIZE,
    _all_gather_positioned_embeddings,
    _balanced_nonempty_chunks,
    _result_from_payload,
    build_fold_global_validation_data,
    compute_fold_global_metrics_from_embeddings,
    evaluate_fold_global_distributed,
)
from retriever.markup import SLOT_TOKEN  # noqa: E402
from retriever.provenance import (  # noqa: E402
    EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD,
    EXPECTED_VALIDATION_IDENTITY_BY_CELL,
)


def _small_fixture():
    corpus: dict[str, CorpusPassage] = {}
    for doc_id, count in (("c1", 21), ("c2", 4)):
        for passage_number in range(count):
            passage_id = f"{doc_id}::p{passage_number:02d}"
            corpus[passage_id] = CorpusPassage(
                passage_id=passage_id,
                doc_id=doc_id,
                label="Analysis",
                text=f"text for {passage_id}",
            )
    queries = [
        QueryExample(
            query_id="c1::q1",
            doc_id="c1",
            motion_root_id="",
            mask_parent_id="",
            query_text=f"structured one {SLOT_TOKEN}",
            positive_passage_ids=[f"c1::p{index:02d}" for index in range(5)],
            positive_labels=["Analysis"],
            flat_query_text_masked=f"flat one {SLOT_TOKEN}",
        ),
        QueryExample(
            query_id="c1::q2",
            doc_id="c1",
            motion_root_id="",
            mask_parent_id="",
            query_text=f"structured two {SLOT_TOKEN}",
            positive_passage_ids=["c1::p20"],
            positive_labels=["Analysis"],
            flat_query_text_masked=f"flat two {SLOT_TOKEN}",
        ),
        QueryExample(
            query_id="c2::q1",
            doc_id="c2",
            motion_root_id="",
            mask_parent_id="",
            query_text=f"structured three {SLOT_TOKEN}",
            positive_passage_ids=["c2::p03"],
            positive_labels=["Analysis"],
            flat_query_text_masked=f"flat three {SLOT_TOKEN}",
        ),
    ]
    table = PassageIndexTable(corpus)
    data = build_fold_global_validation_data(
        all_queries=queries,
        corpus_by_passage_id=corpus,
        passage_index_table=table,
        validation_case_ids=["c2", "c1"],
        expected_query_count=3,
        expected_passage_count=25,
        query_view="structured",
    )
    return corpus, queries, table, data


class FoldGlobalValidationPureTest(unittest.TestCase):
    def test_all_gold_metrics_case_macro_stable_ties_and_immutable_result(self) -> None:
        _, _, table, data = _small_fixture()
        result = compute_fold_global_metrics_from_embeddings(
            query_embeddings=torch.ones((3, 2), dtype=torch.float32),
            passage_embeddings=torch.ones((25, 2), dtype=torch.float32),
            validation_data=data,
            passage_index_table=table,
        )

        self.assertEqual(data.case_ids, ("c1", "c2"))
        self.assertEqual(len(data.queries[0].gold_passage_indices), 5)
        self.assertEqual([row["first_gold_rank"] for row in result.per_query], [1, 21, 25])
        self.assertEqual(result.metrics["eval_validation_query_micro_hit_at_1"], 1.0 / 3.0)
        self.assertEqual(result.metrics["eval_validation_case_macro_hit_at_1"], 0.25)
        self.assertEqual(result.per_query[0]["set_recall_at_1"], 0.2)
        self.assertEqual(result.per_query[0]["exact_target_recovery_at_5"], 1.0)
        self.assertEqual(result.per_query[1]["hit_at_20"], 0.0)
        self.assertEqual(
            result.metrics[VALIDATION_PRIMARY_METRIC],
            0.25,
        )

        with self.assertRaises(TypeError):
            result.metrics[VALIDATION_PRIMARY_METRIC] = 0.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            result.per_query[0]["hit_at_1"] = 0.0  # type: ignore[index]

        independent = result.to_payload()
        independent["metrics"][VALIDATION_PRIMARY_METRIC] = -1.0
        independent["per_case"][0]["metrics"]["hit_at_1"] = -1.0
        self.assertNotEqual(
            independent["metrics"][VALIDATION_PRIMARY_METRIC],
            result.metrics[VALIDATION_PRIMARY_METRIC],
        )
        self.assertNotEqual(
            independent["per_case"][0]["metrics"]["hit_at_1"],
            result.per_case[0]["metrics"]["hit_at_1"],
        )

    def test_query_views_and_corpus_identity_are_exact(self) -> None:
        corpus, queries, table, structured = _small_fixture()
        flat = build_fold_global_validation_data(
            all_queries=queries,
            corpus_by_passage_id=corpus,
            passage_index_table=table,
            validation_case_ids=["c1", "c2"],
            expected_query_count=3,
            expected_passage_count=25,
            query_view="flat_masked",
        )
        self.assertTrue(structured.queries[0].query_text.startswith("structured"))
        self.assertTrue(flat.queries[0].query_text.startswith("flat"))
        self.assertNotEqual(structured.contract_sha256, flat.contract_sha256)

        changed_text = dict(corpus)
        source = changed_text["c1::p00"]
        changed_text["c1::p00"] = replace(source, text="changed")
        with self.assertRaisesRegex(ValueError, "text disagrees"):
            build_fold_global_validation_data(
                all_queries=queries,
                corpus_by_passage_id=changed_text,
                passage_index_table=table,
                validation_case_ids=["c1", "c2"],
                expected_query_count=3,
                expected_passage_count=25,
                query_view="structured",
            )

        changed_identity = dict(corpus)
        changed_identity["c1::p00"] = replace(source, passage_id="different")
        with self.assertRaisesRegex(ValueError, "record identity"):
            build_fold_global_validation_data(
                all_queries=queries,
                corpus_by_passage_id=changed_identity,
                passage_index_table=table,
                validation_case_ids=["c1", "c2"],
                expected_query_count=3,
                expected_passage_count=25,
                query_view="structured",
            )

    def test_contract_and_payload_are_recomputed_strictly(self) -> None:
        _, _, table, data = _small_fixture()
        query_embeddings = torch.ones((3, 2), dtype=torch.float32)
        passage_embeddings = torch.ones((25, 2), dtype=torch.float32)
        result = compute_fold_global_metrics_from_embeddings(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            validation_data=data,
            passage_index_table=table,
        )

        with self.assertRaisesRegex(ValueError, "contract digest"):
            compute_fold_global_metrics_from_embeddings(
                query_embeddings=query_embeddings,
                passage_embeddings=passage_embeddings,
                validation_data=replace(data, contract_sha256="0" * 64),
                passage_index_table=table,
            )
        duplicate_passages = replace(
            data,
            passage_indices=(data.passage_indices[0], *data.passage_indices),
            passage_doc_ids=(data.passage_doc_ids[0], *data.passage_doc_ids),
        )
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            compute_fold_global_metrics_from_embeddings(
                query_embeddings=query_embeddings,
                passage_embeddings=passage_embeddings,
                validation_data=duplicate_passages,
                passage_index_table=table,
            )

        changed_metric = result.to_payload()
        changed_metric["metrics"][VALIDATION_PRIMARY_METRIC] += 0.1
        with self.assertRaisesRegex(RuntimeError, "aggregate metric"):
            _result_from_payload(changed_metric, data, table)

        changed_query = result.to_payload()
        changed_query["per_query"][0]["set_recall_at_1"] = 0.3
        with self.assertRaisesRegex(RuntimeError, "gold-set fraction"):
            _result_from_payload(changed_query, data, table)

        changed_case = result.to_payload()
        changed_case["per_case"][0]["doc_id"] = "c2"
        with self.assertRaisesRegex(RuntimeError, "IDs or order"):
            _result_from_payload(changed_case, data, table)

    def test_nonfinite_or_wrong_shape_embeddings_fail(self) -> None:
        _, _, table, data = _small_fixture()
        with self.assertRaises(FloatingPointError):
            compute_fold_global_metrics_from_embeddings(
                query_embeddings=torch.tensor(
                    [[float("nan")], [1.0], [1.0]], dtype=torch.float32
                ),
                passage_embeddings=torch.ones((25, 1), dtype=torch.float32),
                validation_data=data,
                passage_index_table=table,
            )
        with self.assertRaises(ValueError):
            compute_fold_global_metrics_from_embeddings(
                query_embeddings=torch.ones((2, 1), dtype=torch.float32),
                passage_embeddings=torch.ones((25, 1), dtype=torch.float32),
                validation_data=data,
                passage_index_table=table,
            )


class FrozenFoldGlobalInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_corpus(DATASET_DIR)
        cls.queries = load_queries(DATASET_DIR, "all")
        cls.table = PassageIndexTable(cls.corpus)
        cls.fold_manifest = json.loads(FOLD_CONFIG.read_text(encoding="utf-8"))

    def test_all_five_validation_roles_have_exact_fold_global_inventories(self) -> None:
        expected_passage_counts = [1060, 1055, 1055, 1062, 1054]
        for rotation, expected_passages in zip(
            self.fold_manifest["rotations"],
            expected_passage_counts,
        ):
            role = rotation["validation"]
            data = build_fold_global_validation_data(
                all_queries=self.queries,
                corpus_by_passage_id=self.corpus,
                passage_index_table=self.table,
                validation_case_ids=role["case_ids"],
                expected_query_count=role["queries"],
                expected_passage_count=role["passages"],
                query_view="structured",
            )
            self.assertEqual(data.query_count, 98)
            self.assertEqual(data.passage_count, expected_passages)
            self.assertEqual(data.case_count, role["num_cases"])
            self.assertEqual(set(data.case_ids), set(role["case_ids"]))
            self.assertEqual(set(data.passage_doc_ids), set(role["case_ids"]))
            self.assertEqual(
                {query.doc_id for query in data.queries},
                set(role["case_ids"]),
            )
            candidate_indices = set(data.passage_indices)
            self.assertTrue(
                all(
                    set(query.gold_passage_indices).issubset(candidate_indices)
                    for query in data.queries
                )
            )
            for rank in range(VALIDATION_WORLD_SIZE):
                query_chunks = _balanced_nonempty_chunks(
                    tuple(range(rank, data.query_count, VALIDATION_WORLD_SIZE)),
                    chunk_count=VALIDATION_FORWARD_STEPS,
                )
                passage_chunks = _balanced_nonempty_chunks(
                    tuple(range(rank, data.passage_count, VALIDATION_WORLD_SIZE)),
                    chunk_count=VALIDATION_FORWARD_STEPS,
                )
                self.assertEqual(len(query_chunks), 7)
                self.assertEqual(len(passage_chunks), 7)
                self.assertLessEqual(
                    max(map(len, query_chunks)),
                    VALIDATION_QUERY_BATCH_CAP,
                )
                self.assertLessEqual(
                    max(map(len, passage_chunks)),
                    VALIDATION_PASSAGE_BATCH_CAP,
                )

    def test_all_frozen_training_and_validation_identity_tables_match_real_data(self) -> None:
        for rotation in self.fold_manifest["rotations"]:
            outer_fold = rotation["outer_fold"]
            train_case_ids = set(rotation["train"]["case_ids"])
            train_query_ids = sorted(
                query.query_id
                for query in self.queries
                if query.doc_id in train_case_ids
            )
            train_query_payload = json.dumps(
                train_query_ids,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.assertEqual(len(train_query_ids), rotation["train"]["queries"])
            self.assertEqual(
                hashlib.sha256(train_query_payload).hexdigest(),
                EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD[outer_fold],
            )

            validation_role = rotation["validation"]
            for query_view in ("structured", "flat_masked"):
                with self.subTest(outer_fold=outer_fold, query_view=query_view):
                    data = build_fold_global_validation_data(
                        all_queries=self.queries,
                        corpus_by_passage_id=self.corpus,
                        passage_index_table=self.table,
                        validation_case_ids=validation_role["case_ids"],
                        expected_query_count=validation_role["queries"],
                        expected_passage_count=validation_role["passages"],
                        query_view=query_view,
                    )
                    self.assertEqual(
                        {
                            "case_ids_sha256": data.case_ids_sha256,
                            "query_ids_sha256": data.query_ids_sha256,
                            "passage_ids_sha256": data.passage_ids_sha256,
                            "contract_sha256": data.contract_sha256,
                        },
                        EXPECTED_VALIDATION_IDENTITY_BY_CELL[
                            (outer_fold, query_view)
                        ],
                    )


class _ToyTokenizer:
    unk_token_id = -1

    def __init__(self, *, fail_rank: int | None = None) -> None:
        self.truncation_side = "right"
        self.fail_rank = fail_rank

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return 7 if token == SLOT_TOKEN else -1

    def __call__(self, texts, **kwargs):
        del kwargs
        if self.fail_rank is not None and dist.get_rank() == self.fail_rank:
            raise ValueError("injected tokenizer failure")
        input_ids = torch.full((len(texts), 2), 11, dtype=torch.long)
        for row, text in enumerate(texts):
            if SLOT_TOKEN in text:
                input_ids[row, 1] = 7
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class _ToyPairModel(torch.nn.Module):
    def __init__(self, *, fail_rank: int | None = None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.fail_rank = fail_rank
        self.forward_calls = 0

    def forward(
        self,
        *,
        query_input_ids,
        query_attention_mask,
        passage_input_ids,
        passage_attention_mask,
    ):
        del query_attention_mask, passage_attention_mask
        self.forward_calls += 1
        if self.fail_rank is not None and dist.get_rank() == self.fail_rank:
            raise ValueError("injected forward failure")
        return {
            "query_embeddings": torch.ones(
                (query_input_ids.shape[0], 2),
                dtype=torch.float32,
            ),
            "passage_embeddings": torch.ones(
                (passage_input_ids.shape[0], 2),
                dtype=torch.float32,
            ),
        }


def _assert_same_error_across_ranks(message: str) -> None:
    messages: list[object] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(messages, message)
    if messages != [message] * dist.get_world_size():
        raise AssertionError(f"Ranks raised different validation errors: {messages}")


def _expect_collective_error(operation, expected_fragment: str) -> str:
    try:
        operation()
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError(f"Expected collective failure containing {expected_fragment!r}")
    if expected_fragment not in message:
        raise AssertionError(f"Unexpected collective failure: {message}")
    _assert_same_error_across_ranks(message)
    return message


def _validation_worker(rank: int, init_file: str, output_dir: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
    )
    try:
        corpus = load_corpus(DATASET_DIR)
        all_queries = load_queries(DATASET_DIR, "all")
        table = PassageIndexTable(corpus)
        folds = json.loads(FOLD_CONFIG.read_text(encoding="utf-8"))
        validation_role = folds["rotations"][0]["validation"]
        data = build_fold_global_validation_data(
            all_queries=all_queries,
            corpus_by_passage_id=corpus,
            passage_index_table=table,
            validation_case_ids=validation_role["case_ids"],
            expected_query_count=validation_role["queries"],
            expected_passage_count=validation_role["passages"],
            query_view="structured",
        )

        tokenizer = _ToyTokenizer()
        model = _ToyPairModel()
        model.train()
        result = evaluate_fold_global_distributed(
            model,
            tokenizer,
            validation_data=data,
            passage_index_table=table,
            max_len_query=VALIDATION_MAX_LEN_QUERY,
            max_len_passage=VALIDATION_MAX_LEN_PASSAGE,
        )
        reference = compute_fold_global_metrics_from_embeddings(
            query_embeddings=torch.ones((data.query_count, 2), dtype=torch.float32),
            passage_embeddings=torch.ones(
                (data.passage_count, 2),
                dtype=torch.float32,
            ),
            validation_data=data,
            passage_index_table=table,
        )
        if result.to_payload() != reference.to_payload():
            raise AssertionError("Distributed validation differs from its pure reference")
        if model.forward_calls != VALIDATION_FORWARD_STEPS:
            raise AssertionError(
                f"Rank {rank} made {model.forward_calls} validation forwards"
            )
        if not model.training or tokenizer.truncation_side != "right":
            raise AssertionError("Validation did not restore model/tokenizer state")
        result_digest = hashlib.sha256(
            json.dumps(
                result.to_payload(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        result_digests: list[object] = [None for _ in range(4)]
        dist.all_gather_object(result_digests, result_digest)
        if result_digests != [result_digest] * 4:
            raise AssertionError("Ranks received different validation results")

        bad_queries = list(all_queries)
        if rank == 1:
            target_doc_ids = set(validation_role["case_ids"])
            changed_position = next(
                index
                for index, query in enumerate(bad_queries)
                if query.doc_id in target_doc_ids
            )
            source_query = bad_queries[changed_position]
            bad_queries[changed_position] = replace(
                source_query,
                query_text=source_query.query_text + " changed",
            )
        mismatched_data = build_fold_global_validation_data(
            all_queries=bad_queries,
            corpus_by_passage_id=corpus,
            passage_index_table=table,
            validation_case_ids=validation_role["case_ids"],
            expected_query_count=validation_role["queries"],
            expected_passage_count=validation_role["passages"],
            query_view="structured",
        )
        contract_model = _ToyPairModel()
        _expect_collective_error(
            lambda: evaluate_fold_global_distributed(
                contract_model,
                _ToyTokenizer(),
                validation_data=mismatched_data,
                passage_index_table=table,
                max_len_query=VALIDATION_MAX_LEN_QUERY,
                max_len_passage=VALIDATION_MAX_LEN_PASSAGE,
            ),
            "different data/runtime contracts",
        )
        if contract_model.forward_calls != 0:
            raise AssertionError("Contract mismatch reached model forward")

        forward_count_model = _ToyPairModel()
        _expect_collective_error(
            lambda: evaluate_fold_global_distributed(
                forward_count_model,
                _ToyTokenizer(),
                validation_data=data,
                passage_index_table=table,
                max_len_query=VALIDATION_MAX_LEN_QUERY,
                max_len_passage=VALIDATION_MAX_LEN_PASSAGE,
                forward_steps=VALIDATION_FORWARD_STEPS + 1,
            ),
            "requires exactly 7 forwards",
        )
        if forward_count_model.forward_calls != 0:
            raise AssertionError("Invalid forward count reached model forward")

        token_length_model = _ToyPairModel()
        _expect_collective_error(
            lambda: evaluate_fold_global_distributed(
                token_length_model,
                _ToyTokenizer(),
                validation_data=data,
                passage_index_table=table,
                max_len_query=VALIDATION_MAX_LEN_QUERY - 1,
                max_len_passage=VALIDATION_MAX_LEN_PASSAGE,
            ),
            "max query length must be 4096",
        )
        if token_length_model.forward_calls != 0:
            raise AssertionError("Invalid token length reached model forward")

        token_model = _ToyPairModel()
        token_tokenizer = _ToyTokenizer(fail_rank=1)
        _expect_collective_error(
            lambda: evaluate_fold_global_distributed(
                token_model,
                token_tokenizer,
                validation_data=data,
                passage_index_table=table,
                max_len_query=VALIDATION_MAX_LEN_QUERY,
                max_len_passage=VALIDATION_MAX_LEN_PASSAGE,
            ),
            "injected tokenizer failure",
        )
        if token_model.forward_calls != 0 or not token_model.training:
            raise AssertionError("Tokenizer failure changed model state or reached forward")
        if token_tokenizer.truncation_side != "right":
            raise AssertionError("Tokenizer failure did not restore truncation side")

        forward_model = _ToyPairModel(fail_rank=2)
        _expect_collective_error(
            lambda: evaluate_fold_global_distributed(
                forward_model,
                _ToyTokenizer(),
                validation_data=data,
                passage_index_table=table,
                max_len_query=VALIDATION_MAX_LEN_QUERY,
                max_len_passage=VALIDATION_MAX_LEN_PASSAGE,
            ),
            "injected forward failure",
        )
        if not forward_model.training:
            raise AssertionError("Forward failure did not restore model mode")

        local_positions = tuple(range(rank, 8, 4))
        local_embeddings = torch.ones((len(local_positions), 2), dtype=torch.float32)
        if rank == 3:
            local_embeddings[0, 0] = float("nan")
        _expect_collective_error(
            lambda: _all_gather_positioned_embeddings(
                local_embeddings,
                local_positions,
                total_count=8,
            ),
            "non-finite",
        )

        output_path = Path(output_dir) / f"rank-{rank}.json"
        output_path.write_text(
            json.dumps(
                {
                    "forward_calls": model.forward_calls,
                    "ranking_sha256": result.ranking_sha256,
                    "result_sha256": result_digest,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedFoldGlobalValidationTest(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "requires the PyTorch Gloo distributed backend",
    )
    def test_four_rank_validation_and_coordinated_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "results"
            output_dir.mkdir()
            mp.spawn(
                _validation_worker,
                args=(str(root / "process-group"), str(output_dir)),
                nprocs=4,
                join=True,
            )
            rows = [
                json.loads((output_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
                for rank in range(4)
            ]
            self.assertEqual({row["forward_calls"] for row in rows}, {7})
            self.assertEqual(len({row["ranking_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["result_sha256"] for row in rows}), 1)


if __name__ == "__main__":
    unittest.main()
