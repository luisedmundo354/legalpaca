from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.artifacts import CONTROLLED_ARTIFACT_PROTOCOL  # noqa: E402
from retriever.data import PassageIndexTable, load_corpus  # noqa: E402
from retriever.evaluator import (  # noqa: E402
    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
    _load_exact_json_file,
    _load_exact_json_file_with_sha256,
    _runtime_identity_for_controlled_artifacts,
    _validate_controlled_artifact_uniqueness,
    _validate_controlled_evaluation_plan,
    _validate_local_bindings,
    run_local_controlled_evaluation_plan,
)


DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
FOLD_MANIFEST_PATH = (
    MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json"
)
EXPERIMENT_CONFIG_PATH = (
    MODERNBERT_DIR / "experiments/retrieval_cv/configs/experiment.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _valid_plan() -> dict[str, object]:
    fold_manifest = json.loads(FOLD_MANIFEST_PATH.read_bytes())
    role = fold_manifest["rotations"][0]["test"]
    passage_index = PassageIndexTable(load_corpus(DATASET_DIR))
    return {
        "schema_version": 2,
        "experiment_id": "arr_retrieval_cv_v1",
        "outer_fold": 0,
        "role": "test",
        "experiment_config_sha256": _sha256(EXPERIMENT_CONFIG_PATH),
        "dataset_manifest_sha256": _sha256(DATASET_DIR / "dataset_manifest.json"),
        "fold_manifest_sha256": _sha256(FOLD_MANIFEST_PATH),
        "passage_index_sha256": passage_index.sha256,
        "case_ids": role["case_ids"],
        "query_count": role["queries"],
        "passage_count": role["passages"],
        "max_len_query": 4_096,
        "max_len_passage": 500,
        "query_batch_size": 4,
        "passage_batch_size": 38,
        "runtime_identity": {"runtime": "preflight-test"},
        "systems": [
            {
                "system_id": "flat_local_seed17",
                "system_type": CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                "query_view": "flat_masked",
                "artifact_expectation": {
                    "artifact_manifest_sha256": "a" * 64,
                    "training_plan_sha256": "1" * 64,
                    "training_staging_receipt_sha256": "2" * 64,
                    "source_bundle_name": f"source-{'3' * 64}.tar.gz",
                    "source_bundle_size": 12_345,
                    "source_bundle_sha256": "3" * 64,
                    "source_bundle_inventory_sha256": "4" * 64,
                    "source_bundle_commit_epoch": 1_700_000_000,
                    "experiment_id": "arr_retrieval_cv_v1",
                    "outer_fold": 0,
                    "query_view": "flat_masked",
                    "sampler": "local_unique",
                    "experiment_seed": 17,
                    "dataset_manifest_sha256": _sha256(
                        DATASET_DIR / "dataset_manifest.json"
                    ),
                    "fold_manifest_sha256": _sha256(FOLD_MANIFEST_PATH),
                    "passage_index_sha256": passage_index.sha256,
                    "model_artifact_protocol": CONTROLLED_ARTIFACT_PROTOCOL,
                },
            }
        ],
    }


class ControlledEvaluationPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = _valid_plan()

    def test_valid_plan_is_exactly_normalized(self) -> None:
        identity, case_ids, systems = _validate_controlled_evaluation_plan(
            copy.deepcopy(self.plan),
            evaluation_plan_sha256="f" * 64,
        )
        self.assertEqual(identity.outer_fold, 0)
        self.assertEqual(identity.role, "test")
        self.assertEqual(case_ids, tuple(self.plan["case_ids"]))
        self.assertEqual([system["system_id"] for system in systems], [
            "flat_local_seed17"
        ])

    def test_plan_rejects_length_regime_baseline_and_artifact_drift(self) -> None:
        mutations = (
            ("max_len_query", 512, "4096"),
            ("max_len_passage", 600, "500"),
            ("role", "global_split", "role"),
            ("experiment_config_sha256", "0" * 64, "frozen controlled study"),
            ("dataset_manifest_sha256", "0" * 64, "frozen controlled study"),
            ("fold_manifest_sha256", "0" * 64, "frozen controlled study"),
            ("passage_index_sha256", "0" * 64, "frozen controlled study"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                plan = copy.deepcopy(self.plan)
                plan[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    _validate_controlled_evaluation_plan(
                        plan,
                        evaluation_plan_sha256="f" * 64,
                    )

        baseline = copy.deepcopy(self.plan)
        baseline["systems"][0]["system_type"] = "bm25"
        with self.assertRaisesRegex(ValueError, "Step 8"):
            _validate_controlled_evaluation_plan(
                baseline,
                evaluation_plan_sha256="f" * 64,
            )

        bad_seed = copy.deepcopy(self.plan)
        bad_seed["systems"][0]["artifact_expectation"]["experiment_seed"] = 18
        with self.assertRaisesRegex(ValueError, "experiment_seed"):
            _validate_controlled_evaluation_plan(
                bad_seed,
                evaluation_plan_sha256="f" * 64,
            )

        bad_hash = copy.deepcopy(self.plan)
        bad_hash["systems"][0]["artifact_expectation"][
            "artifact_manifest_sha256"
        ] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "artifact_manifest_sha256"):
            _validate_controlled_evaluation_plan(
                bad_hash,
                evaluation_plan_sha256="f" * 64,
            )

        noncanonical_view = copy.deepcopy(self.plan)
        noncanonical_view["systems"][0]["query_view"] = " flat_masked "
        with self.assertRaisesRegex(ValueError, "not canonical"):
            _validate_controlled_evaluation_plan(
                noncanonical_view,
                evaluation_plan_sha256="f" * 64,
            )

        mixed_source = copy.deepcopy(self.plan)
        second = copy.deepcopy(mixed_source["systems"][0])
        second["system_id"] = "structured_local_seed17"
        second["query_view"] = "structured"
        second["artifact_expectation"]["artifact_manifest_sha256"] = "b" * 64
        second["artifact_expectation"]["query_view"] = "structured"
        second["artifact_expectation"]["source_bundle_sha256"] = "5" * 64
        second["artifact_expectation"]["source_bundle_name"] = (
            f"source-{'5' * 64}.tar.gz"
        )
        mixed_source["systems"].append(second)
        with self.assertRaisesRegex(ValueError, "mixes controlled launch"):
            _validate_controlled_evaluation_plan(
                mixed_source,
                evaluation_plan_sha256="f" * 64,
            )

        mixed_ledger = copy.deepcopy(self.plan)
        second = copy.deepcopy(mixed_ledger["systems"][0])
        second["system_id"] = "structured_local_seed17"
        second["query_view"] = "structured"
        second["artifact_expectation"]["artifact_manifest_sha256"] = "b" * 64
        second["artifact_expectation"]["query_view"] = "structured"
        second["artifact_expectation"]["training_plan_sha256"] = "5" * 64
        second["artifact_expectation"]["training_staging_receipt_sha256"] = "6" * 64
        mixed_ledger["systems"].append(second)
        with self.assertRaisesRegex(ValueError, "mixes controlled launch"):
            _validate_controlled_evaluation_plan(
                mixed_ledger,
                evaluation_plan_sha256="f" * 64,
            )

    def test_local_bindings_require_absolute_exact_order_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bindings = {
                "schema_version": 1,
                "dataset_dir": str(DATASET_DIR.resolve()),
                "fold_manifest_path": str(FOLD_MANIFEST_PATH.resolve()),
                "experiment_config_path": str(EXPERIMENT_CONFIG_PATH.resolve()),
                "systems": [
                    {
                        "system_id": "flat_local_seed17",
                        "artifact_dir": str(root / "artifact"),
                    }
                ],
            }
            values = _validate_local_bindings(
                bindings,
                system_ids=("flat_local_seed17",),
            )
            self.assertEqual(values[0], DATASET_DIR.resolve())

            relative = copy.deepcopy(bindings)
            relative["systems"][0]["artifact_dir"] = "artifact"
            with self.assertRaisesRegex(ValueError, "absolute"):
                _validate_local_bindings(
                    relative,
                    system_ids=("flat_local_seed17",),
                )

            wrong_id = copy.deepcopy(bindings)
            wrong_id["systems"][0]["system_id"] = "unknown"
            with self.assertRaisesRegex(ValueError, "order/coverage"):
                _validate_local_bindings(
                    wrong_id,
                    system_ids=("flat_local_seed17",),
                )

    def test_plan_and_bindings_must_be_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(self.plan, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical"):
                _load_exact_json_file(path, name="evaluation plan", canonical=True)
            path.write_bytes(_canonical_bytes(self.plan))
            self.assertEqual(
                _load_exact_json_file(path, name="evaluation plan", canonical=True),
                self.plan,
            )

    def test_plan_value_and_sha256_come_from_exactly_one_read(self) -> None:
        first = _canonical_bytes(self.plan)
        changed = copy.deepcopy(self.plan)
        changed["runtime_identity"] = {"runtime": "changed-after-first-read"}
        second = _canonical_bytes(changed)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_bytes(first)
            with mock.patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=[first, second],
            ) as read_bytes:
                value, sha256 = _load_exact_json_file_with_sha256(
                    path,
                    name="evaluation plan",
                    canonical=True,
                )

        read_bytes.assert_called_once_with(path)
        self.assertEqual(value, self.plan)
        self.assertEqual(sha256, hashlib.sha256(first).hexdigest())

    def test_real_fold_and_dataset_preflight_reaches_only_the_pinned_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            staged_dataset = root / "mounted-data"
            staged_fold_manifest = root / "mounted-folds.json"
            staged_experiment_config = root / "mounted-experiment.json"
            shutil.copytree(DATASET_DIR, staged_dataset)
            shutil.copy2(FOLD_MANIFEST_PATH, staged_fold_manifest)
            shutil.copy2(EXPERIMENT_CONFIG_PATH, staged_experiment_config)
            plan_path = root / "plan.json"
            bindings_path = root / "bindings.json"
            output_dir = root / "result"
            plan_path.write_bytes(_canonical_bytes(self.plan))
            bindings_path.write_bytes(
                _canonical_bytes(
                    {
                        "schema_version": 1,
                        "dataset_dir": str(staged_dataset),
                        "fold_manifest_path": str(staged_fold_manifest),
                        "experiment_config_path": str(staged_experiment_config),
                        "systems": [
                            {
                                "system_id": "flat_local_seed17",
                                "artifact_dir": str(root / "artifact"),
                            }
                        ],
                    }
                )
            )
            with mock.patch(
                "retriever.artifacts.import_pinned_artifact_runtime",
                side_effect=RuntimeError("pinned runtime sentinel"),
            ), self.assertRaisesRegex(RuntimeError, "pinned runtime sentinel"):
                run_local_controlled_evaluation_plan(
                    evaluation_plan_path=plan_path,
                    local_bindings_path=bindings_path,
                    output_dir=output_dir,
                    device="cpu",
                )
            self.assertFalse(output_dir.exists())

    def test_role_case_and_count_drift_fail_before_runtime_or_artifacts(self) -> None:
        mutations = {
            "case_ids": lambda plan: plan.__setitem__(
                "case_ids", plan["case_ids"][:-1]
            ),
            "query_count": lambda plan: plan.__setitem__("query_count", 97),
            "passage_count": lambda plan: plan.__setitem__("passage_count", 1_053),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                plan = copy.deepcopy(self.plan)
                mutate(plan)
                plan_path = root / "plan.json"
                bindings_path = root / "bindings.json"
                plan_path.write_bytes(_canonical_bytes(plan))
                bindings_path.write_bytes(
                    _canonical_bytes(
                        {
                            "schema_version": 1,
                            "dataset_dir": str(DATASET_DIR.resolve()),
                            "fold_manifest_path": str(FOLD_MANIFEST_PATH.resolve()),
                            "experiment_config_path": str(
                                EXPERIMENT_CONFIG_PATH.resolve()
                            ),
                            "systems": [
                                {
                                    "system_id": "flat_local_seed17",
                                    "artifact_dir": str(root / "artifact"),
                                }
                            ],
                        }
                    )
                )
                with mock.patch(
                    "retriever.artifacts.import_pinned_artifact_runtime"
                ) as runtime_import, self.assertRaisesRegex(
                    ValueError, "role inventory"
                ):
                    run_local_controlled_evaluation_plan(
                        evaluation_plan_path=plan_path,
                        local_bindings_path=bindings_path,
                        output_dir=root / "result",
                        device="cpu",
                    )
                runtime_import.assert_not_called()

    def test_all_artifacts_are_validated_before_any_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan = copy.deepcopy(self.plan)
            second = copy.deepcopy(plan["systems"][0])
            second["system_id"] = "structured_local_seed17"
            second["query_view"] = "structured"
            second["artifact_expectation"]["artifact_manifest_sha256"] = "b" * 64
            second["artifact_expectation"]["query_view"] = "structured"
            plan["systems"].append(second)
            runtime = SimpleNamespace(torch_module=torch)
            plan["runtime_identity"] = _runtime_identity_for_controlled_artifacts(
                device="cpu",
                runtime=runtime,
            )
            plan_path = root / "plan.json"
            bindings_path = root / "bindings.json"
            plan_path.write_bytes(_canonical_bytes(plan))
            bindings_path.write_bytes(
                _canonical_bytes(
                    {
                        "schema_version": 1,
                        "dataset_dir": str(DATASET_DIR.resolve()),
                        "fold_manifest_path": str(FOLD_MANIFEST_PATH.resolve()),
                        "experiment_config_path": str(EXPERIMENT_CONFIG_PATH.resolve()),
                        "systems": [
                            {
                                "system_id": system["system_id"],
                                "artifact_dir": str(root / system["system_id"]),
                            }
                            for system in plan["systems"]
                        ],
                    }
                )
            )
            first_artifact = SimpleNamespace(
                identity=SimpleNamespace(
                    artifact_manifest_sha256="a" * 64,
                    query_view="flat_masked",
                    sampler="local_unique",
                    experiment_seed=17,
                )
            )
            with (
                mock.patch(
                    "retriever.artifacts.import_pinned_artifact_runtime",
                    return_value=runtime,
                ),
                mock.patch(
                    "retriever.artifacts.validate_controlled_artifact",
                    side_effect=[
                        first_artifact,
                        ValueError("invalid final artifact"),
                    ],
                ),
                mock.patch(
                    "retriever.evaluator.score_loaded_dual_encoder"
                ) as scorer,
                self.assertRaisesRegex(ValueError, "invalid final artifact"),
            ):
                run_local_controlled_evaluation_plan(
                    evaluation_plan_path=plan_path,
                    local_bindings_path=bindings_path,
                    output_dir=root / "result",
                    device="cpu",
                )
            scorer.assert_not_called()

    def test_artifact_aliases_and_duplicate_training_cells_are_rejected(self) -> None:
        def artifact(manifest_hash: str, view: str, sampler: str, seed: int):
            return SimpleNamespace(
                identity=SimpleNamespace(
                    artifact_manifest_sha256=manifest_hash,
                    query_view=view,
                    sampler=sampler,
                    experiment_seed=seed,
                )
            )

        first = artifact("a" * 64, "flat_masked", "local_unique", 17)
        with self.assertRaisesRegex(ValueError, "aliases"):
            _validate_controlled_artifact_uniqueness(
                (
                    first,
                    artifact("a" * 64, "structured", "local_unique", 17),
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate controlled training cells"):
            _validate_controlled_artifact_uniqueness(
                (
                    first,
                    artifact("b" * 64, "flat_masked", "local_unique", 17),
                )
            )


if __name__ == "__main__":
    unittest.main()
