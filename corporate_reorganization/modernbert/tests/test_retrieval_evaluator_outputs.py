from __future__ import annotations

import hashlib
import json
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

from retriever.data import CorpusPassage, QueryExample  # noqa: E402
from retriever.evaluator import (  # noqa: E402
    CanonicalEvaluationBundle,
    EvaluationIdentity,
    SystemScoreInput,
    build_canonical_evaluation_bundle,
    publish_and_validate_canonical_evaluation_bundle,
    publish_canonical_evaluation_bundle,
    validate_published_evaluation_bundle,
)
from retriever.markup import SLOT_TOKEN  # noqa: E402


def _fixture():
    corpus: dict[str, CorpusPassage] = {}
    for case_id, count in (("c1", 3), ("c2", 2)):
        for index in range(1, count + 1):
            passage_id = f"{case_id}::p{index}"
            corpus[passage_id] = CorpusPassage(
                passage_id=passage_id,
                doc_id=case_id,
                label="Analysis",
                text=f"text {passage_id}",
            )
    queries = [
        QueryExample(
            query_id="c1::q1",
            doc_id="c1",
            motion_root_id="",
            mask_parent_id="",
            query_text=f"structured one {SLOT_TOKEN}",
            positive_passage_ids=["c1::p1"],
            positive_labels=["Analysis"],
            visible_passage_ids=["c1::p1", "c1::p3"],
            flat_query_text_masked=f"flat one {SLOT_TOKEN}",
        ),
        QueryExample(
            query_id="c1::q2",
            doc_id="c1",
            motion_root_id="",
            mask_parent_id="",
            query_text=f"structured two {SLOT_TOKEN}",
            positive_passage_ids=["c1::p2"],
            positive_labels=["Analysis"],
            visible_passage_ids=["c1::p3"],
            flat_query_text_masked=f"flat two {SLOT_TOKEN}",
        ),
        QueryExample(
            query_id="c2::q1",
            doc_id="c2",
            motion_root_id="",
            mask_parent_id="",
            query_text=f"structured three {SLOT_TOKEN}",
            positive_passage_ids=["c2::p2"],
            positive_labels=["Analysis"],
            visible_passage_ids=["c2::p1"],
            flat_query_text_masked=f"flat three {SLOT_TOKEN}",
        ),
    ]
    scores_a = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.0, 0.0],
            [0.1, 0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    scores_b = torch.tensor(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1, 0.0],
        ],
        dtype=torch.float64,
    )
    return corpus, queries, scores_a, scores_b


def _identity() -> EvaluationIdentity:
    return EvaluationIdentity(
        experiment_id="arr_retrieval_cv_v1",
        outer_fold=0,
        role="test",
        evaluation_plan_sha256="a" * 64,
        experiment_config_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        fold_manifest_sha256="d" * 64,
        passage_index_sha256="e" * 64,
    )


def _bundle():
    corpus, queries, scores_a, scores_b = _fixture()
    bundle = build_canonical_evaluation_bundle(
        identity=_identity(),
        all_queries=queries,
        corpus_by_passage_id=corpus,
        evaluated_case_ids=("c1", "c2"),
        runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
        systems=(
            SystemScoreInput(
                system_id="a_structured",
                system_type="dual_encoder_artifact",
                query_view="structured",
                model_identity={"artifact_manifest_sha256": "1" * 64},
                scores=scores_a,
            ),
            SystemScoreInput(
                system_id="b_flat",
                system_type="dense_e5",
                query_view="flat_masked",
                model_identity={"snapshot_tree_sha256": "2" * 64},
                scores=scores_b,
            ),
        ),
        torch_module=torch,
    )
    return bundle, corpus, queries


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def _canonical_bytes(value: object) -> bytes:
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


def _refresh_manifest_record(output: Path, relative_name: str) -> None:
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    payload = (output / relative_name).read_bytes()
    record = next(record for record in manifest["files"] if record["path"] == relative_name)
    record["size"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(_canonical_bytes(manifest))


class EvaluationBundleTest(unittest.TestCase):
    def test_atomic_round_trip_and_path_independent_bytes(self) -> None:
        bundle, corpus, queries = _bundle()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output_a = parent / "result-a"
            output_b = parent / "result-b"
            record_a = publish_canonical_evaluation_bundle(bundle, output_dir=output_a)
            record_b = publish_canonical_evaluation_bundle(bundle, output_dir=output_b)
            self.assertRegex(record_a["artifact_manifest_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                record_a["artifact_manifest_sha256"],
                record_b["artifact_manifest_sha256"],
            )
            self.assertEqual(_tree_bytes(output_a), _tree_bytes(output_b))

            results = validate_published_evaluation_bundle(
                output_dir=output_a,
                identity=_identity(),
                runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                all_queries=queries,
                corpus_by_passage_id=corpus,
            )
            self.assertEqual(len(results), 8)
            self.assertEqual(
                [result.regime_name for result in results[:4]],
                [
                    "same_case_legacy",
                    "same_case_full",
                    "fold_global",
                    "fold_global_context_excluded",
                ],
            )
            ranking_lines = (output_a / "rankings.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(ranking_lines), 2 * 4 * 3)
            first = json.loads(ranking_lines[0])
            self.assertEqual(first["query_id"], "c1::q1")
            self.assertEqual(first["per_query"]["gold_passage_ids"], ["c1::p1"])
            self.assertEqual(
                first["per_query"]["visible_passage_ids"],
                ["c1::p1", "c1::p3"],
            )
            self.assertNotIn(str(parent), (output_a / "evaluation_config.json").read_text())

    def test_existing_output_and_stale_incomplete_are_rejected(self) -> None:
        bundle, _, _ = _bundle()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "result"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "absent"):
                publish_canonical_evaluation_bundle(bundle, output_dir=output)
            output.rmdir()
            (parent / ".result.incomplete").mkdir()
            with self.assertRaisesRegex(FileExistsError, "Stale incomplete"):
                publish_canonical_evaluation_bundle(bundle, output_dir=output)

    def test_failure_never_leaves_a_commit_marker(self) -> None:
        bundle, _, _ = _bundle()
        import retriever.evaluator as evaluator

        original_write = evaluator._write_new_file

        def injected_failure(path: Path, payload: bytes) -> None:
            if path.name == "results.json":
                raise RuntimeError("injected publication failure")
            original_write(path, payload)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluator,
            "_write_new_file",
            side_effect=injected_failure,
        ):
            output = Path(temporary) / "result"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                publish_canonical_evaluation_bundle(bundle, output_dir=output)
            incomplete = Path(temporary) / ".result.incomplete"
            self.assertTrue(incomplete.is_dir())
            self.assertFalse((incomplete / "artifact_manifest.json").exists())
            self.assertFalse(output.exists())

    def test_deep_readback_failure_retracts_the_success_marker(self) -> None:
        bundle, corpus, queries = _bundle()
        import retriever.evaluator as evaluator

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluator,
            "validate_published_evaluation_bundle",
            side_effect=RuntimeError("injected scientific readback failure"),
        ):
            output = Path(temporary) / "result"
            with self.assertRaisesRegex(RuntimeError, "scientific readback"):
                publish_and_validate_canonical_evaluation_bundle(
                    bundle,
                    output_dir=output,
                    identity=_identity(),
                    runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                    all_queries=queries,
                    corpus_by_passage_id=corpus,
                )
            incomplete = Path(temporary) / ".result.incomplete"
            self.assertFalse(output.exists())
            self.assertTrue(incomplete.is_dir())
            self.assertFalse((incomplete / "artifact_manifest.json").exists())

    def test_retraction_collision_still_removes_the_success_marker(self) -> None:
        bundle, corpus, queries = _bundle()
        import retriever.evaluator as evaluator

        def fail_after_creating_incomplete(*, output_dir: Path, **kwargs) -> None:
            del kwargs
            (output_dir.parent / f".{output_dir.name}.incomplete").mkdir()
            raise RuntimeError("injected scientific readback failure")

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluator,
            "validate_published_evaluation_bundle",
            side_effect=fail_after_creating_incomplete,
        ):
            output = Path(temporary) / "result"
            with self.assertRaisesRegex(RuntimeError, "existing incomplete"):
                publish_and_validate_canonical_evaluation_bundle(
                    bundle,
                    output_dir=output,
                    identity=_identity(),
                    runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                    all_queries=queries,
                    corpus_by_passage_id=corpus,
                )

            self.assertTrue(output.is_dir())
            self.assertFalse((output / "artifact_manifest.json").exists())
            self.assertTrue((Path(temporary) / ".result.incomplete").is_dir())

    def test_target_appearing_at_publication_is_never_replaced(self) -> None:
        bundle, _, _ = _bundle()
        import retriever.evaluator as evaluator

        original_rename = evaluator._rename_directory_to_absent

        def inject_target(source: Path, target: Path) -> None:
            target.mkdir()
            original_rename(source, target)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluator,
            "_rename_directory_to_absent",
            side_effect=inject_target,
        ):
            output = Path(temporary) / "result"
            with self.assertRaisesRegex(FileExistsError, "replace"):
                publish_canonical_evaluation_bundle(bundle, output_dir=output)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            incomplete = Path(temporary) / ".result.incomplete"
            self.assertTrue(incomplete.is_dir())
            self.assertFalse((incomplete / "artifact_manifest.json").exists())

    def test_publication_uses_atomic_linux_no_replace_not_os_rename(self) -> None:
        bundle, _, _ = _bundle()
        import retriever.evaluator as evaluator

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluator.os,
            "rename",
            side_effect=AssertionError("replacing rename must not be called"),
        ):
            output = Path(temporary) / "result"
            publish_canonical_evaluation_bundle(bundle, output_dir=output)
            self.assertTrue((output / "artifact_manifest.json").is_file())

    def test_manifest_or_ranking_mutation_is_rejected(self) -> None:
        bundle, corpus, queries = _bundle()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            publish_canonical_evaluation_bundle(bundle, output_dir=output)
            rankings_path = output / "rankings.jsonl"
            original = rankings_path.read_bytes()
            mutated = original.replace(
                b'"system_id":"a_structured"',
                b'"system_id":"x_structured"',
                1,
            )
            self.assertNotEqual(mutated, original)
            rankings_path.write_bytes(mutated)
            with self.assertRaisesRegex(ValueError, "size/hash"):
                validate_published_evaluation_bundle(
                    output_dir=output,
                    identity=_identity(),
                    runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                    all_queries=queries,
                    corpus_by_passage_id=corpus,
                )

    def test_refreshed_manifest_cannot_change_the_scientific_protocol(self) -> None:
        bundle, corpus, queries = _bundle()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            publish_canonical_evaluation_bundle(bundle, output_dir=output)
            config_path = output / "evaluation_config.json"
            config = json.loads(config_path.read_bytes())
            config["protocols"]["ranking"] = "score_desc_only_unstable"
            config_path.write_bytes(_canonical_bytes(config))
            _refresh_manifest_record(output, "evaluation_config.json")
            with self.assertRaisesRegex(ValueError, "inventories changed"):
                validate_published_evaluation_bundle(
                    output_dir=output,
                    identity=_identity(),
                    runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                    all_queries=queries,
                    corpus_by_passage_id=corpus,
                )

    def test_regimes_cannot_splice_independently_valid_source_rankings(self) -> None:
        bundle, corpus, queries = _bundle()
        _, _, scores_a, scores_b = _fixture()
        alternate = build_canonical_evaluation_bundle(
            identity=_identity(),
            all_queries=queries,
            corpus_by_passage_id=corpus,
            evaluated_case_ids=("c1", "c2"),
            runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
            systems=(
                SystemScoreInput(
                    system_id="a_structured",
                    system_type="dual_encoder_artifact",
                    query_view="structured",
                    model_identity={"artifact_manifest_sha256": "1" * 64},
                    scores=-scores_a,
                ),
                SystemScoreInput(
                    system_id="b_flat",
                    system_type="dense_e5",
                    query_view="flat_masked",
                    model_identity={"snapshot_tree_sha256": "2" * 64},
                    scores=scores_b,
                ),
            ),
            torch_module=torch,
        )
        records = list(bundle.result_records)
        records[1] = alternate.result_records[1]
        spliced = CanonicalEvaluationBundle(
            config=bundle.config,
            result_records=tuple(records),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            publish_canonical_evaluation_bundle(spliced, output_dir=output)
            with self.assertRaisesRegex(ValueError, "one complete source ranking"):
                validate_published_evaluation_bundle(
                    output_dir=output,
                    identity=_identity(),
                    runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                    all_queries=queries,
                    corpus_by_passage_id=corpus,
                )

    def test_system_order_and_local_paths_fail_loudly(self) -> None:
        corpus, queries, scores_a, scores_b = _fixture()
        with self.assertRaisesRegex(ValueError, "system_id order"):
            build_canonical_evaluation_bundle(
                identity=_identity(),
                all_queries=queries,
                corpus_by_passage_id=corpus,
                evaluated_case_ids=("c1", "c2"),
                runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                systems=(
                    SystemScoreInput(
                        system_id="b",
                        system_type="dense_e5",
                        query_view="flat_masked",
                        model_identity={"sha256": "2" * 64},
                        scores=scores_b,
                    ),
                    SystemScoreInput(
                        system_id="a",
                        system_type="dual_encoder_artifact",
                        query_view="structured",
                        model_identity={"sha256": "1" * 64},
                        scores=scores_a,
                    ),
                ),
                torch_module=torch,
            )
        with self.assertRaisesRegex(ValueError, "absolute path"):
            build_canonical_evaluation_bundle(
                identity=_identity(),
                all_queries=queries,
                corpus_by_passage_id=corpus,
                evaluated_case_ids=("c1", "c2"),
                runtime_identity={"device": "cpu", "runtime": "pinned-test-fixture"},
                systems=(
                    SystemScoreInput(
                        system_id="a",
                        system_type="dual_encoder_artifact",
                        query_view="structured",
                        model_identity={"artifact_dir": "/tmp/model"},
                        scores=scores_a,
                    ),
                ),
                torch_module=torch,
            )

    def test_scientific_bytes_have_stable_hashes(self) -> None:
        bundle, _, _ = _bundle()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            publish_canonical_evaluation_bundle(bundle, output_dir=output)
            hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.iterdir()
            }
            self.assertEqual(set(hashes), {
                "evaluation_config.json",
                "rankings.jsonl",
                "results.json",
                "artifact_manifest.json",
            })
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
