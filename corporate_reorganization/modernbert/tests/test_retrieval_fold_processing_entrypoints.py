from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from processing_fold_eval import evaluate_sm, inventory_sm  # noqa: E402
from retriever.data import load_queries  # noqa: E402
from retriever.evaluation import build_canonical_evaluation_data  # noqa: E402
from retriever.evaluator import (  # noqa: E402
    BM25_SYSTEM_TYPE,
    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
    E5_SYSTEM_TYPE,
    FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
    FIXED_BASE_SYSTEM_TYPE,
    LEGACY_PROCESSING_IMAGE_CONTRACT_SHA256,
    run_complete_evaluation_plan,
    run_complete_fold_evaluation_plan,
)


DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
FOLDS_PATH = MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json"


class FoldProcessingEntrypointTest(unittest.TestCase):
    def test_phase_entrypoints_reject_unknown_arguments_and_non_cuda(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as phase1_unknown:
                inventory_sm.parse_args(
                    [
                        "--archive-manifest",
                        "/manifest.json",
                        "--dataset-dir",
                        "/dataset",
                        "--fold-manifest",
                        "/folds.json",
                        "--scratch-parent",
                        "/work",
                        "--output-dir",
                        "/output",
                        "--unexpected",
                        "value",
                    ]
                )
            with self.assertRaises(SystemExit) as phase2_unknown:
                evaluate_sm.parse_args(
                    [
                        "--evaluation-plan",
                        "/plan.json",
                        "--local-bindings",
                        "/bindings.json",
                        "--output-dir",
                        "/output",
                        "--device",
                        "cuda:0",
                        "--unexpected",
                        "value",
                    ]
                )
            with self.assertRaises(SystemExit) as non_cuda:
                evaluate_sm.parse_args(
                    [
                        "--evaluation-plan",
                        "/plan.json",
                        "--local-bindings",
                        "/bindings.json",
                        "--output-dir",
                        "/output",
                        "--device",
                        "cpu",
                    ]
                )
        self.assertEqual(phase1_unknown.exception.code, 2)
        self.assertEqual(phase2_unknown.exception.code, 2)
        self.assertEqual(non_cuda.exception.code, 2)

    def test_legacy_and_fold_public_apis_dispatch_explicit_contracts(self) -> None:
        self.assertEqual(
            inspect.signature(run_complete_evaluation_plan),
            inspect.signature(run_complete_fold_evaluation_plan),
        )
        arguments = {
            "evaluation_plan_path": Path("/plan.json"),
            "local_bindings_path": Path("/bindings.json"),
            "output_dir": Path("/output"),
            "device": "cuda:0",
        }
        cases = (
            (
                run_complete_evaluation_plan,
                "processing_eval.image_smoke.validate_image_runtime",
                LEGACY_PROCESSING_IMAGE_CONTRACT_SHA256,
                None,
            ),
            (
                run_complete_fold_evaluation_plan,
                "processing_fold_eval.image_smoke.validate_image_runtime",
                FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
                "test",
            ),
        )
        for entrypoint, validator_path, expected_contract, required_role in cases:
            with self.subTest(entrypoint=entrypoint.__name__), mock.patch(
                validator_path
            ) as validator, mock.patch(
                "retriever.evaluator._run_complete_evaluation_plan",
                return_value={"ok": True},
            ) as runner:
                self.assertEqual(entrypoint(**arguments), {"ok": True})
                runner.assert_called_once_with(
                    **arguments,
                    expected_image_contract_sha256=expected_contract,
                    required_role=required_role,
                    validate_image_runtime=validator,
                )

    def test_phase1_uses_exact_phase2_fold_global_test_identity_for_all_folds(self) -> None:
        for outer_fold in range(5):
            with self.subTest(outer_fold=outer_fold):
                fold_manifest, corpus, actual = inventory_sm._fold_global_test_data(
                    input_manifest={"outer_fold": outer_fold},
                    dataset_dir=DATASET_DIR,
                    fold_manifest_path=FOLDS_PATH,
                )
                test_role = fold_manifest["rotations"][outer_fold]["test"]
                expected = build_canonical_evaluation_data(
                    all_queries=load_queries(DATASET_DIR, "all"),
                    corpus_by_passage_id=corpus,
                    evaluated_case_ids=tuple(test_role["case_ids"]),
                    role="test",
                    regime_name="fold_global",
                )
                self.assertEqual(actual.case_ids, expected.case_ids)
                self.assertEqual(actual.passage_ids, expected.passage_ids)
                self.assertEqual(actual.case_ids_sha256, expected.case_ids_sha256)
                self.assertEqual(actual.query_ids_sha256, expected.query_ids_sha256)
                self.assertEqual(actual.passage_ids_sha256, expected.passage_ids_sha256)
                self.assertEqual(
                    actual.candidate_pools_sha256,
                    expected.candidate_pools_sha256,
                )
                self.assertEqual(actual.contract_sha256, expected.contract_sha256)
                self.assertEqual(actual.query_count, test_role["queries"])
                self.assertEqual(actual.passage_count, test_role["passages"])

    def _storage_fixture(self, root: Path):
        processing = root / "processing"
        dataset = root / "dataset"
        processing.mkdir()
        dataset.mkdir()
        (dataset / "sentinel").write_bytes(b"dataset")
        folds = root / "folds.json"
        folds.write_bytes(b"{}\n")
        work = processing / "work"
        evaluation_data = SimpleNamespace(
            passage_ids=("p1", "p2"),
            query_count=3,
            passage_count=2,
            case_ids_sha256="1" * 64,
            query_ids_sha256="2" * 64,
            passage_ids_sha256="3" * 64,
            candidate_pools_sha256="4" * 64,
            contract_sha256="5" * 64,
        )
        fold_manifest = {
            "rotations": [
                {
                    "outer_fold": 0,
                    "test": {
                        "case_ids": ["1"],
                        "queries": 3,
                        "passages": 2,
                    },
                }
            ]
        }
        corpus = {
            "p1": SimpleNamespace(text="first"),
            "p2": SimpleNamespace(text="second"),
        }
        return processing, dataset, folds, work, fold_manifest, corpus, evaluation_data

    def _build_storage_receipt(self, root: Path, *, diverge: bool):
        (
            processing,
            dataset,
            folds,
            work,
            fold_manifest,
            corpus,
            evaluation_data,
        ) = self._storage_fixture(root)
        scratch_dirs = (work / "bm25-inventory-a", work / "bm25-inventory-b")
        calls: list[Path] = []

        def fake_build(*, passage_ids, passage_texts, scratch_dir):
            self.assertEqual(tuple(passage_ids), ("p1", "p2"))
            self.assertEqual(tuple(passage_texts), ("first", "second"))
            calls.append(Path(scratch_dir))
            collection = Path(scratch_dir) / "collection"
            index = Path(scratch_dir) / "index"
            collection.mkdir(parents=True)
            index.mkdir()
            (collection / "passages.jsonl").write_bytes(b"same")
            payload = b"different" if diverge and len(calls) == 2 else b"same"
            (index / "segments").write_bytes(payload)
            return index

        runtime = SimpleNamespace(to_payload=lambda: {"protocol": "pinned"})
        with (
            mock.patch.object(inventory_sm, "PROCESSING_ROOT", processing),
            mock.patch.object(inventory_sm, "WORK_PARENT", work),
            mock.patch.object(inventory_sm, "BM25_SCRATCH_DIRS", scratch_dirs),
            mock.patch.object(
                inventory_sm,
                "_fold_global_test_data",
                return_value=(fold_manifest, corpus, evaluation_data),
            ),
            mock.patch.object(inventory_sm, "validate_bm25_runtime", return_value=runtime),
            mock.patch.object(inventory_sm, "build_bm25_index", side_effect=fake_build),
            mock.patch.object(
                inventory_sm,
                "_sha256_file",
                return_value=inventory_sm.EXPECTED_FOLD_MANIFEST_SHA256,
            ),
        ):
            receipt = inventory_sm.build_bm25_storage_receipt(
                input_manifest={
                    "experiment_id": "arr_retrieval_cv_v1",
                    "outer_fold": 0,
                },
                dataset_dir=dataset,
                fold_manifest_path=folds,
                archive_inventory={
                    "input_manifest_sha256": "6" * 64,
                    "receipt_sha256": "7" * 64,
                },
                image_runtime={"image": "pinned"},
            )
        return receipt, calls

    def test_phase1_requires_two_identical_bm25_allocation_replicas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt, calls = self._build_storage_receipt(
                Path(temporary).resolve(), diverge=False
            )
        self.assertEqual([record["ordinal"] for record in receipt["bm25_replicas"]], [1, 2])
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            receipt["bm25_replicas"][0]["allocation_tree_sha256"],
            receipt["bm25_replicas"][1]["allocation_tree_sha256"],
        )
        self.assertEqual(receipt["query_ids_sha256"], "2" * 64)
        self.assertEqual(receipt["evaluation_contract_sha256"], "5" * 64)
        self.assertIn("receipt_sha256", receipt)

    def test_phase1_rejects_nonreproducible_bm25_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            RuntimeError, "different allocations"
        ):
            self._build_storage_receipt(Path(temporary).resolve(), diverge=True)

    def test_phase1_rejects_alternate_archive_root_before_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            processing = root / "processing"
            archive_root = processing / "archive"
            dataset = processing / "dataset"
            output_parent = processing / "output"
            for directory in (archive_root, dataset, output_parent):
                directory.mkdir(parents=True, exist_ok=True)
            archive_manifest = archive_root / "fold_archive_input_manifest.json"
            folds = processing / "folds.json"
            work = processing / "work"
            output = output_parent / "evidence"
            layout = {
                "archive_manifest_path": str(archive_manifest),
                "dataset_dir": str(dataset),
                "fold_manifest_path": str(folds),
                "evidence_output_dir": str(output),
                "output_parent": str(output_parent),
            }
            with (
                mock.patch.dict(inventory_sm.PROCESSING_LAYOUT, layout),
                mock.patch.object(inventory_sm, "PROCESSING_ROOT", processing),
                mock.patch.object(inventory_sm, "WORK_PARENT", work),
                mock.patch.object(
                    inventory_sm,
                    "load_fold_archive_input_manifest",
                    return_value={"archive_root": str(root / "alternate")},
                ),
                mock.patch.object(inventory_sm, "_preflight_bm25_inputs") as preflight,
                mock.patch.object(inventory_sm, "validate_image_runtime") as image_smoke,
                mock.patch.object(
                    inventory_sm, "build_fold_archive_inventory_receipt"
                ) as inventory,
                self.assertRaisesRegex(ValueError, "archive root"),
            ):
                inventory_sm.main(
                    [
                        "--archive-manifest",
                        str(archive_manifest),
                        "--dataset-dir",
                        str(dataset),
                        "--fold-manifest",
                        str(folds),
                        "--scratch-parent",
                        str(work),
                        "--output-dir",
                        str(output),
                    ]
                )
            preflight.assert_not_called()
            image_smoke.assert_not_called()
            inventory.assert_not_called()

    def test_phase1_output_is_absent_only_and_commit_marked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_parent = Path(temporary).resolve() / "output"
            output_parent.mkdir()
            output = output_parent / "evidence"
            with mock.patch.dict(
                inventory_sm.PROCESSING_LAYOUT,
                {
                    "evidence_output_dir": str(output),
                    "output_parent": str(output_parent),
                },
            ):
                manifest = inventory_sm.publish_phase1_output(
                    output_dir=output,
                    input_manifest={
                        "experiment_id": "arr_retrieval_cv_v1",
                        "outer_fold": 0,
                    },
                    archive_inventory={
                        "input_manifest_sha256": "1" * 64,
                        "receipt_sha256": "2" * 64,
                    },
                    bm25_storage={"receipt_sha256": "3" * 64},
                )
                self.assertEqual(
                    sorted(path.name for path in output.iterdir()),
                    [
                        "archive_inventory.json",
                        "artifact_manifest.json",
                        "bm25_storage.json",
                    ],
                )
                self.assertEqual(
                    json.loads((output / "artifact_manifest.json").read_bytes()),
                    manifest,
                )
                for path in output.iterdir():
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                    self.assertEqual(path.stat().st_nlink, 1)
                with self.assertRaises(FileExistsError):
                    inventory_sm.publish_phase1_output(
                        output_dir=output,
                        input_manifest={
                            "experiment_id": "arr_retrieval_cv_v1",
                            "outer_fold": 0,
                        },
                        archive_inventory={
                            "input_manifest_sha256": "1" * 64,
                            "receipt_sha256": "2" * 64,
                        },
                        bm25_storage={"receipt_sha256": "3" * 64},
                    )

    def test_phase1_output_fails_on_incomplete_write_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_parent = Path(temporary).resolve() / "output"
            output_parent.mkdir()
            output = output_parent / "evidence"
            with (
                mock.patch.dict(
                    inventory_sm.PROCESSING_LAYOUT,
                    {
                        "evidence_output_dir": str(output),
                        "output_parent": str(output_parent),
                    },
                ),
                mock.patch.object(
                    inventory_sm,
                    "_write_descriptor_exact",
                    side_effect=RuntimeError("injected short write"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected short write"),
            ):
                inventory_sm.publish_phase1_output(
                    output_dir=output,
                    input_manifest={
                        "experiment_id": "arr_retrieval_cv_v1",
                        "outer_fold": 0,
                    },
                    archive_inventory={
                        "input_manifest_sha256": "1" * 64,
                        "receipt_sha256": "2" * 64,
                    },
                    bm25_storage={"receipt_sha256": "3" * 64},
                )
            self.assertFalse(output.exists())
            self.assertTrue((output_parent / ".evidence.incomplete").is_dir())

    def test_phase2_local_binding_map_is_exact(self) -> None:
        layout = evaluate_sm.PROCESSING_LAYOUT
        systems = [
            {"system_id": "bm25_flat_plain", "system_type": BM25_SYSTEM_TYPE},
            {"system_id": "e5_base_v2_flat_plain", "system_type": E5_SYSTEM_TYPE},
            {
                "system_id": "modernbert_base_flat_masked",
                "system_type": FIXED_BASE_SYSTEM_TYPE,
            },
            {
                "system_id": "structured_local_unique_seed17",
                "system_type": CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
            },
        ]
        bound = {
            "dataset_dir": Path(layout["dataset_dir"]),
            "fold_manifest_path": Path(layout["fold_manifest_path"]),
            "experiment_config_path": Path(layout["experiment_config_path"]),
            "baseline_config_path": Path(layout["baseline_config_path"]),
            "image_contract_path": Path(layout["image_contract_path"]),
            "bm25_scratch_dir": Path(layout["bm25_scratch_dir"]),
            "systems": {
                "bm25_flat_plain": {"system_id": "bm25_flat_plain"},
                "e5_base_v2_flat_plain": {
                    "system_id": "e5_base_v2_flat_plain",
                    "snapshot_dir": Path(layout["e5_snapshot_dir"]),
                    "snapshot_manifest_path": Path(
                        layout["e5_snapshot_manifest_path"]
                    ),
                    "pack_artifact_dir": Path(layout["e5_pack_artifact_dir"]),
                },
                "modernbert_base_flat_masked": {
                    "system_id": "modernbert_base_flat_masked",
                    "artifact_dir": Path(layout["fixed_base_artifact_dir"]),
                },
                "structured_local_unique_seed17": {
                    "system_id": "structured_local_unique_seed17",
                    "artifact_dir": evaluate_sm.MATERIALIZATION_ROOT
                    / "structured_local_unique_seed17",
                },
            },
        }
        bindings = {"bm25_scratch_dir": layout["bm25_scratch_dir"]}
        patches = (
            mock.patch.object(
                evaluate_sm,
                "_load_exact_json_file_with_sha256",
                return_value=({"role": "test"}, "1" * 64),
            ),
            mock.patch.object(
                evaluate_sm,
                "_validate_complete_evaluation_plan",
                return_value=(SimpleNamespace(role="test"), (), systems),
            ),
            mock.patch.object(
                evaluate_sm, "_load_exact_json_file", return_value=bindings
            ),
            mock.patch.object(
                evaluate_sm, "_validate_complete_local_bindings", return_value=bound
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            _, normalized, actual = evaluate_sm._load_plan_and_bindings(
                Path(layout["evaluation_plan_path"]),
                Path(layout["local_bindings_path"]),
            )
            self.assertEqual(normalized, systems)
            self.assertEqual(actual, bound)
            wrong = dict(bound)
            wrong["dataset_dir"] = Path("/wrong")
            with mock.patch.object(
                evaluate_sm,
                "_validate_complete_local_bindings",
                return_value=wrong,
            ), self.assertRaisesRegex(ValueError, "common local bindings"):
                evaluate_sm._load_plan_and_bindings(
                    Path(layout["evaluation_plan_path"]),
                    Path(layout["local_bindings_path"]),
                )

    def test_phase2_rejects_validation_role_before_loading_bindings(self) -> None:
        with (
            mock.patch.object(
                evaluate_sm,
                "_load_exact_json_file_with_sha256",
                return_value=({"role": "validation"}, "1" * 64),
            ),
            mock.patch.object(
                evaluate_sm,
                "_validate_complete_evaluation_plan",
                return_value=(SimpleNamespace(role="validation"), (), []),
            ),
            mock.patch.object(evaluate_sm, "_load_exact_json_file") as load_bindings,
            self.assertRaisesRegex(ValueError, "held-out test role"),
        ):
            evaluate_sm._load_plan_and_bindings(
                Path(evaluate_sm.PROCESSING_LAYOUT["evaluation_plan_path"]),
                Path(evaluate_sm.PROCESSING_LAYOUT["local_bindings_path"]),
            )
        load_bindings.assert_not_called()

    def test_phase2_rejects_stale_incomplete_outputs_before_work(self) -> None:
        for stale_name in (".evaluation.incomplete", ".evidence.incomplete"):
            with self.subTest(stale_name=stale_name), tempfile.TemporaryDirectory() as temporary:
                output_parent = Path(temporary).resolve() / "output"
                output_parent.mkdir()
                evaluation = output_parent / "evaluation"
                evidence = output_parent / "evidence"
                (output_parent / stale_name).mkdir()
                with (
                    mock.patch.object(evaluate_sm, "OUTPUT_PARENT", output_parent),
                    mock.patch.object(
                        evaluate_sm, "EVALUATION_OUTPUT_DIR", evaluation
                    ),
                    mock.patch.object(evaluate_sm, "EVIDENCE_OUTPUT_DIR", evidence),
                    mock.patch.object(evaluate_sm, "_secure_create_work_parent") as work,
                    mock.patch.object(
                        evaluate_sm, "materialize_fold_archives"
                    ) as materialize,
                    mock.patch.object(
                        evaluate_sm, "_publish_materialization_evidence"
                    ) as publish,
                    self.assertRaisesRegex(FileExistsError, "initially absent"),
                ):
                    evaluate_sm.main(
                        [
                            "--evaluation-plan",
                            evaluate_sm.PROCESSING_LAYOUT["evaluation_plan_path"],
                            "--local-bindings",
                            evaluate_sm.PROCESSING_LAYOUT["local_bindings_path"],
                            "--output-dir",
                            str(evaluation),
                            "--device",
                            "cuda:0",
                        ]
                    )
                work.assert_not_called()
                materialize.assert_not_called()
                publish.assert_not_called()

    def test_phase2_rejects_alternate_archive_root_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output_parent = root / "output"
            output_parent.mkdir()
            evaluation = output_parent / "evaluation"
            evidence = output_parent / "evidence"
            archive_root = root / "archive"
            archive_root.mkdir()
            archive_manifest = archive_root / "fold_archive_input_manifest.json"
            with (
                mock.patch.object(evaluate_sm, "OUTPUT_PARENT", output_parent),
                mock.patch.object(evaluate_sm, "EVALUATION_OUTPUT_DIR", evaluation),
                mock.patch.object(evaluate_sm, "EVIDENCE_OUTPUT_DIR", evidence),
                mock.patch.object(
                    evaluate_sm, "ARCHIVE_MANIFEST_PATH", archive_manifest
                ),
                mock.patch.object(evaluate_sm, "_secure_create_work_parent"),
                mock.patch.object(
                    evaluate_sm,
                    "_load_plan_and_bindings",
                    return_value=({}, [], {}),
                ),
                mock.patch.object(
                    evaluate_sm,
                    "load_fold_archive_input_manifest",
                    return_value={"archive_root": str(root / "alternate")},
                ),
                mock.patch.object(
                    evaluate_sm, "load_fold_archive_inventory_receipt"
                ) as inventory,
                mock.patch.object(
                    evaluate_sm, "materialize_fold_archives"
                ) as materialize,
                self.assertRaisesRegex(ValueError, "archive root"),
            ):
                evaluate_sm.main(
                    [
                        "--evaluation-plan",
                        evaluate_sm.PROCESSING_LAYOUT["evaluation_plan_path"],
                        "--local-bindings",
                        evaluate_sm.PROCESSING_LAYOUT["local_bindings_path"],
                        "--output-dir",
                        str(evaluation),
                        "--device",
                        "cuda:0",
                    ]
                )
            inventory.assert_not_called()
            materialize.assert_not_called()

    def test_phase2_rejects_controlled_order_and_binding_root_splices(self) -> None:
        system_plans = [
            {
                "system_id": system_id,
                "system_type": CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                "expectation": {},
            }
            for system_id in ("controlled-a", "controlled-b")
        ]
        archive_manifest = {
            "experiment_id": "experiment",
            "outer_fold": 0,
            "systems": [
                {"system_id": "controlled-b"},
                {"system_id": "controlled-a"},
            ],
        }
        bound = {
            "systems": {
                system_id: {
                    "system_id": system_id,
                    "artifact_dir": evaluate_sm.MATERIALIZATION_ROOT / system_id,
                }
                for system_id in ("controlled-a", "controlled-b")
            }
        }
        with self.assertRaisesRegex(ValueError, "controlled order"):
            evaluate_sm._controlled_expectations(
                plan={"experiment_id": "experiment", "outer_fold": 0},
                system_plans=system_plans,
                bound=bound,
                archive_manifest=archive_manifest,
            )
        archive_manifest["systems"] = [
            {"system_id": "controlled-a"},
            {"system_id": "controlled-b"},
        ]
        bound["systems"]["controlled-a"]["artifact_dir"] = (
            evaluate_sm.MATERIALIZATION_ROOT / "controlled-b"
        )
        with self.assertRaisesRegex(ValueError, "left the materialization root"):
            evaluate_sm._controlled_expectations(
                plan={"experiment_id": "experiment", "outer_fold": 0},
                system_plans=system_plans,
                bound=bound,
                archive_manifest=archive_manifest,
            )

    def test_phase2_rejects_materialized_expectation_root_and_count_splices(self) -> None:
        system_id = "controlled-a"
        expectation = mock.sentinel.expected_artifact
        archive_manifest = {
            "experiment_id": "experiment",
            "outer_fold": 0,
            "systems": [
                {
                    "system_id": system_id,
                    "cell": {
                        "query_view": "flat",
                        "sampler": "global_matched",
                        "experiment_seed": 17,
                    },
                }
            ],
        }
        identity = SimpleNamespace(
            experiment_id="experiment",
            outer_fold=0,
            query_view="flat",
            sampler="global_matched",
            experiment_seed=17,
        )

        def materialization(*, root=None, artifact_expectation=expectation, artifacts=True):
            artifact = SimpleNamespace(
                root=root or evaluate_sm.MATERIALIZATION_ROOT / system_id,
                expectation=artifact_expectation,
                identity=identity,
            )
            return SimpleNamespace(
                root=evaluate_sm.MATERIALIZATION_ROOT,
                receipt={"systems": [{"system_id": system_id}]},
                artifacts=(artifact,) if artifacts else (),
            )

        with self.assertRaisesRegex(RuntimeError, "artifact count"):
            evaluate_sm._validate_materialization_result(
                archive_manifest=archive_manifest,
                expectations={system_id: expectation},
                materialization=materialization(artifacts=False),
            )
        with self.assertRaisesRegex(RuntimeError, "artifact identity"):
            evaluate_sm._validate_materialization_result(
                archive_manifest=archive_manifest,
                expectations={system_id: expectation},
                materialization=materialization(
                    root=evaluate_sm.MATERIALIZATION_ROOT / "controlled-b"
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "artifact identity"):
            evaluate_sm._validate_materialization_result(
                archive_manifest=archive_manifest,
                expectations={system_id: expectation},
                materialization=materialization(
                    artifact_expectation=mock.sentinel.spliced_expectation
                ),
            )

    def test_phase2_evidence_short_write_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_parent = Path(temporary).resolve() / "output"
            output_parent.mkdir()
            evidence = output_parent / "evidence"
            with (
                mock.patch.object(evaluate_sm, "OUTPUT_PARENT", output_parent),
                mock.patch.object(evaluate_sm, "EVIDENCE_OUTPUT_DIR", evidence),
                mock.patch.object(
                    evaluate_sm,
                    "_write_descriptor_exact",
                    side_effect=RuntimeError("injected short write"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected short write"),
            ):
                evaluate_sm._publish_materialization_evidence(
                    archive_manifest={"experiment_id": "experiment", "outer_fold": 0},
                    inventory_receipt={
                        "input_manifest_sha256": "1" * 64,
                        "receipt_sha256": "2" * 64,
                    },
                    materialization_receipt={
                        "materialization_sha256": "3" * 64,
                    },
                )
            self.assertFalse(evidence.exists())
            self.assertTrue((output_parent / ".evidence.incomplete").is_dir())


if __name__ == "__main__":
    unittest.main()
