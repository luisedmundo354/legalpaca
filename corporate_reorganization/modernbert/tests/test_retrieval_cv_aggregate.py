from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from corporate_reorganization.modernbert.experiments.retrieval_cv import aggregate
from corporate_reorganization.modernbert.retriever.data import (
    CorpusPassage,
    PassageIndexTable,
    QueryExample,
)
from corporate_reorganization.modernbert.retriever.evaluator import (
    EvaluationIdentity,
    SystemScoreInput,
    build_canonical_evaluation_bundle,
    publish_canonical_evaluation_bundle,
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def synthetic_study() -> tuple[
    dict[str, CorpusPassage],
    list[QueryExample],
    dict[str, object],
]:
    corpus: dict[str, CorpusPassage] = {}
    queries: list[QueryExample] = []
    rotations: list[dict[str, object]] = []
    for fold in range(5):
        case_id = f"c{fold}"
        gold_id = f"p{fold}a"
        other_id = f"p{fold}b"
        corpus[gold_id] = CorpusPassage(
            passage_id=gold_id,
            doc_id=case_id,
            label="Analysis",
            text=f"Gold passage for {case_id}",
        )
        corpus[other_id] = CorpusPassage(
            passage_id=other_id,
            doc_id=case_id,
            label="Facts",
            text=f"Other passage for {case_id}",
        )
        queries.append(
            QueryExample(
                query_id=f"q{fold}",
                doc_id=case_id,
                motion_root_id=f"m{fold}",
                mask_parent_id=f"n{fold}",
                query_text=f"Structured query for {case_id}",
                positive_passage_ids=[gold_id],
                positive_labels=["Analysis"],
                visible_passage_ids=[],
                flat_query_text_plain=f"Plain query for {case_id}",
                flat_query_text_masked=f"Masked query for {case_id}",
            )
        )
        rotations.append(
            {
                "outer_fold": fold,
                "test": {
                    "case_ids": [case_id],
                    "num_cases": 1,
                    "queries": 1,
                    "passages": 2,
                },
            }
        )
    return (
        corpus,
        queries,
        {
            "totals": {"cases": 5, "queries": 5, "passages": 10},
            "rotations": rotations,
        },
    )


def write_genuine_bundle(
    root: Path,
    *,
    fold: int,
    corpus: dict[str, CorpusPassage],
    queries: list[QueryExample],
    passage_index_sha256: str,
    case_id: str | None = None,
) -> Path:
    evaluated_case_id = case_id or f"c{fold}"
    identity = EvaluationIdentity(
        experiment_id="arr_retrieval_cv_v1",
        outer_fold=fold,
        role="test",
        evaluation_plan_sha256=f"{fold + 1:064x}",
        experiment_config_sha256=aggregate.EXPECTED_EXPERIMENT_CONFIG_SHA256,
        dataset_manifest_sha256=aggregate.EXPECTED_DATASET_MANIFEST_SHA256,
        fold_manifest_sha256=aggregate.EXPECTED_FOLD_MANIFEST_SHA256,
        passage_index_sha256=passage_index_sha256,
    )
    systems = [
        SystemScoreInput(
            system_id=system_id,
            system_type=system_type,
            query_view=query_view,
            model_identity={"fixture_system_id": system_id},
            scores=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
        )
        for system_id, system_type, query_view in aggregate.EXPECTED_SYSTEM_CONTRACT
    ]
    bundle = build_canonical_evaluation_bundle(
        identity=identity,
        all_queries=queries,
        corpus_by_passage_id=corpus,
        evaluated_case_ids=[evaluated_case_id],
        systems=systems,
        runtime_identity={"fixture_runtime": "canonical"},
        torch_module=torch,
    )
    output = root / f"fold-{fold}-{evaluated_case_id}"
    publish_canonical_evaluation_bundle(bundle, output_dir=output)
    return output


def write_dummy_bundle(
    root: Path,
    *,
    passage_index_sha256: str,
) -> Path:
    output = root / "dummy-fold-0"
    output.mkdir()
    config = {
        "identity": {
            "dataset_manifest_sha256": aggregate.EXPECTED_DATASET_MANIFEST_SHA256,
            "evaluation_plan_sha256": f"{1:064x}",
            "experiment_config_sha256": aggregate.EXPECTED_EXPERIMENT_CONFIG_SHA256,
            "experiment_id": "arr_retrieval_cv_v1",
            "fold_manifest_sha256": aggregate.EXPECTED_FOLD_MANIFEST_SHA256,
            "outer_fold": 0,
            "passage_index_sha256": passage_index_sha256,
            "role": "test",
        },
        "runtime_identity": {"fixture_runtime": "canonical"},
        "case_ids": ["c0"],
        "systems": [
            {"system_id": system_id}
            for system_id in aggregate.EXPECTED_SYSTEM_IDS
        ],
    }
    payloads = {
        "evaluation_config.json": canonical(config),
        "rankings.jsonl": canonical({"ranking": ["p0a"]}),
        "results.json": canonical({"result": 1}),
    }
    records = []
    for name in sorted(payloads):
        payload = payloads[name]
        (output / name).write_bytes(payload)
        records.append(
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    (output / "artifact_manifest.json").write_bytes(
        canonical(
            {
                "schema_version": 1,
                "bundle_protocol": "canonical_complete_rankings_v1",
                "commit_marker": True,
                "files": records,
            }
        )
    )
    return output


def refresh_manifest(output: Path, changed_name: str) -> None:
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    changed_path = output / changed_name
    changed_bytes = changed_path.read_bytes()
    record = next(item for item in manifest["files"] if item["path"] == changed_name)
    record["size"] = len(changed_bytes)
    record["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    manifest_path.write_bytes(canonical(manifest))


@contextlib.contextmanager
def patched_study(
    corpus: dict[str, CorpusPassage],
    queries: list[QueryExample],
    fold_manifest: dict[str, object],
):
    passage_index_sha256 = PassageIndexTable(corpus).sha256
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                aggregate,
                "EXPECTED_TOTALS",
                {"cases": 5, "queries": 5, "passages": 10},
            )
        )
        stack.enter_context(
            mock.patch.object(
                aggregate,
                "EXPECTED_PASSAGE_INDEX_SHA256",
                passage_index_sha256,
            )
        )
        stack.enter_context(
            mock.patch.object(
                aggregate,
                "validate_staged_dataset_and_fold",
                return_value=fold_manifest,
            )
        )
        stack.enter_context(mock.patch.object(aggregate, "load_corpus", return_value=corpus))
        stack.enter_context(mock.patch.object(aggregate, "load_queries", return_value=queries))
        yield passage_index_sha256


class EvaluationIndexTest(unittest.TestCase):
    def _write_all_genuine(
        self,
        root: Path,
        corpus: dict[str, CorpusPassage],
        queries: list[QueryExample],
        passage_index_sha256: str,
    ) -> list[Path]:
        return [
            write_genuine_bundle(
                root,
                fold=fold,
                corpus=corpus,
                queries=queries,
                passage_index_sha256=passage_index_sha256,
            )
            for fold in range(5)
        ]

    def test_exact_genuine_five_fold_index_runs_strict_readback(self) -> None:
        corpus, queries, fold_manifest = synthetic_study()
        with tempfile.TemporaryDirectory() as directory, patched_study(
            corpus, queries, fold_manifest
        ) as passage_index_sha256:
            root = Path(directory)
            outputs = self._write_all_genuine(
                root, corpus, queries, passage_index_sha256
            )
            with mock.patch.object(
                aggregate,
                "validate_published_evaluation_bundle",
                wraps=aggregate.validate_published_evaluation_bundle,
            ) as strict_readback:
                index = aggregate.build_evaluation_index(
                    list(reversed(outputs)),
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )

        self.assertEqual(strict_readback.call_count, 5)
        self.assertEqual([item["outer_fold"] for item in index["folds"]], list(range(5)))
        self.assertEqual(
            [item["result_record_count"] for item in index["folds"]],
            [60] * 5,
        )
        self.assertEqual(
            [item["ranking_row_count"] for item in index["folds"]],
            [60] * 5,
        )
        self.assertIs(index["statistics_computed"], False)

    def test_self_consistent_dummy_artifacts_cannot_be_certified(self) -> None:
        corpus, queries, fold_manifest = synthetic_study()
        with tempfile.TemporaryDirectory() as directory, patched_study(
            corpus, queries, fold_manifest
        ) as passage_index_sha256:
            root = Path(directory)
            outputs = self._write_all_genuine(
                root, corpus, queries, passage_index_sha256
            )
            outputs[0] = write_dummy_bundle(
                root,
                passage_index_sha256=passage_index_sha256,
            )
            with mock.patch.object(
                aggregate,
                "validate_published_evaluation_bundle",
                wraps=aggregate.validate_published_evaluation_bundle,
            ) as strict_readback, self.assertRaises(ValueError):
                aggregate.build_evaluation_index(
                    outputs,
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )
        strict_readback.assert_called_once()

    def test_partial_duplicate_and_wrong_fold_inventory_fail(self) -> None:
        corpus, queries, fold_manifest = synthetic_study()
        with tempfile.TemporaryDirectory() as directory, patched_study(
            corpus, queries, fold_manifest
        ) as passage_index_sha256:
            root = Path(directory)
            outputs = self._write_all_genuine(
                root, corpus, queries, passage_index_sha256
            )
            with self.assertRaisesRegex(ValueError, "exactly five"):
                aggregate.build_evaluation_index(
                    outputs[:4],
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )
            with self.assertRaisesRegex(ValueError, "0..4"):
                aggregate.build_evaluation_index(
                    [outputs[0], *outputs[:4]],
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )
            wrong_fold = write_genuine_bundle(
                root,
                fold=4,
                case_id="c3",
                corpus=corpus,
                queries=queries,
                passage_index_sha256=passage_index_sha256,
            )
            with self.assertRaisesRegex(ValueError, "exact test cases"):
                aggregate.build_evaluation_index(
                    [*outputs[:4], wrong_fold],
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )

    def test_candidate_and_regime_coverage_mutations_fail_strict_readback(self) -> None:
        corpus, queries, fold_manifest = synthetic_study()
        with tempfile.TemporaryDirectory() as directory, patched_study(
            corpus, queries, fold_manifest
        ) as passage_index_sha256:
            root = Path(directory)
            outputs = self._write_all_genuine(
                root, corpus, queries, passage_index_sha256
            )
            rankings_path = outputs[0] / "rankings.jsonl"
            rows = [json.loads(line) for line in rankings_path.read_bytes().splitlines()]
            rows[0]["ranking"]["ranked_candidates"].pop()
            rankings_path.write_bytes(b"".join(canonical(row) for row in rows))
            refresh_manifest(outputs[0], "rankings.jsonl")
            with self.assertRaises((ValueError, RuntimeError)):
                aggregate.build_evaluation_index(
                    outputs,
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )

        corpus, queries, fold_manifest = synthetic_study()
        with tempfile.TemporaryDirectory() as directory, patched_study(
            corpus, queries, fold_manifest
        ) as passage_index_sha256:
            root = Path(directory)
            outputs = self._write_all_genuine(
                root, corpus, queries, passage_index_sha256
            )
            results_path = outputs[0] / "results.json"
            results = json.loads(results_path.read_bytes())
            results["result_records"][0]["regime_name"] = "fold_global"
            results_path.write_bytes(canonical(results))
            refresh_manifest(outputs[0], "results.json")
            with self.assertRaisesRegex(ValueError, "system/regime"):
                aggregate.build_evaluation_index(
                    outputs,
                    dataset_dir=root / "dataset",
                    fold_manifest_path=root / "folds.json",
                )


if __name__ == "__main__":
    unittest.main()
