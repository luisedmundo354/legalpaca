from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from corporate_reorganization.modernbert.experiments.retrieval_cv import folds


REPO_ROOT = folds._repo_root()
DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
FOLD_CONFIG = (
    REPO_ROOT
    / "corporate_reorganization/modernbert/experiments/retrieval_cv/configs/folds.json"
)
EXPERIMENT_CONFIG = (
    REPO_ROOT
    / "corporate_reorganization/modernbert/experiments/retrieval_cv/configs/experiment.json"
)

EXPECTED_PRIORITY_ORDER = [
    "49",
    "91",
    "76",
    "85",
    "86",
    "83",
    "94",
    "96",
    "75",
    "79",
    "97",
    "77",
    "87",
    "46",
    "78",
    "80",
    "74",
    "92",
    "48",
    "47",
    "42",
    "59",
    "41",
    "70",
    "63",
    "65",
    "69",
    "57",
    "73",
    "45",
    "62",
    "37",
    "38",
    "66",
    "71",
    "36",
    "60",
    "67",
    "40",
    "58",
    "68",
    "72",
]

EXPECTED_FINAL_FOLDS = [
    ["42", "49", "57", "58", "63", "71", "72", "73", "80"],
    ["38", "40", "60", "62", "68", "87", "91", "92", "96"],
    ["41", "66", "67", "69", "74", "76", "85", "97"],
    ["36", "45", "46", "65", "78", "79", "83", "94"],
    ["37", "47", "48", "59", "70", "75", "77", "86"],
]

EXPECTED_SWAPS = [
    (0, "97", 3, "41"),
    (1, "70", 4, "40"),
    (2, "77", 3, "97"),
    (0, "47", 4, "57"),
    (2, "37", 4, "66"),
    (2, "83", 3, "85"),
    (0, "41", 2, "42"),
    (3, "77", 4, "46"),
    (0, "62", 1, "73"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_float(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, float):
        test.fail(f"Manifest contains non-authoritative float: {value!r}")
    if isinstance(value, dict):
        for item in value.values():
            assert_no_float(test, item)
    elif isinstance(value, list):
        for item in value:
            assert_no_float(test, item)


class FoldAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case_loads, cls.dataset_record = folds.load_case_loads(DATASET_DIR)
        cls.greedy, cls.greedy_trace, cls.case_order = folds.assign_cases_greedily(
            cls.case_loads,
            capacities=folds.FOLD_CAPACITIES,
        )
        cls.refined, cls.swaps = folds.refine_pair_swaps(
            cls.greedy,
            case_loads=cls.case_loads,
        )
        cls.manifest = folds.build_fold_manifest(DATASET_DIR)

    def test_dataset_readback_and_hash_are_exact(self) -> None:
        self.assertEqual(len(self.case_loads), 42)
        self.assertEqual(sum(load.queries for load in self.case_loads), 490)
        self.assertEqual(sum(load.passages for load in self.case_loads), 5286)
        self.assertEqual(
            self.dataset_record["dataset_manifest_sha256"],
            "cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be",
        )

    def test_priority_and_literal_greedy_output_are_frozen(self) -> None:
        self.assertEqual(self.case_order, EXPECTED_PRIORITY_ORDER)
        self.assertEqual(
            [(fold.queries, fold.passages) for fold in self.greedy],
            [(100, 1200), (103, 1064), (95, 1035), (98, 950), (94, 1037)],
        )
        self.assertEqual(
            self.manifest["greedy"]["objective"],
            "56102533748/139767192075",
        )
        self.assertEqual(len(self.greedy_trace), 42)

    def test_approved_swap_trace_and_final_memberships_are_frozen(self) -> None:
        actual_swaps = [
            (
                swap["lower_fold_id"],
                swap["case_from_lower_fold"],
                swap["higher_fold_id"],
                swap["case_from_higher_fold"],
            )
            for swap in self.swaps
        ]
        self.assertEqual(actual_swaps, EXPECTED_SWAPS)
        self.assertEqual(
            [sorted(fold.case_ids, key=int) for fold in self.refined],
            EXPECTED_FINAL_FOLDS,
        )
        self.assertEqual(
            [(fold.queries, fold.passages) for fold in self.refined],
            [(98, 1054), (98, 1060), (98, 1055), (98, 1055), (98, 1062)],
        )
        self.assertEqual(
            self.manifest["pair_swap_refinement"]["objective"],
            "27941923/69854490",
        )

    def test_refined_folds_are_a_strict_pairwise_local_optimum(self) -> None:
        rerefined, additional_swaps = folds.refine_pair_swaps(
            self.refined,
            case_loads=self.case_loads,
        )
        self.assertEqual(additional_swaps, [])
        self.assertEqual(
            [sorted(fold.case_ids, key=int) for fold in rerefined],
            EXPECTED_FINAL_FOLDS,
        )

    def test_case_input_order_does_not_change_output(self) -> None:
        greedy, _, order = folds.assign_cases_greedily(
            list(reversed(self.case_loads)),
            capacities=folds.FOLD_CAPACITIES,
        )
        refined, swaps = folds.refine_pair_swaps(
            greedy,
            case_loads=list(reversed(self.case_loads)),
        )
        self.assertEqual(order, EXPECTED_PRIORITY_ORDER)
        self.assertEqual(
            [sorted(fold.case_ids, key=int) for fold in refined],
            EXPECTED_FINAL_FOLDS,
        )
        self.assertEqual(len(swaps), 9)

    def test_numeric_case_id_and_fold_count_ties_are_deterministic(self) -> None:
        synthetic = [
            folds.CaseLoad("10", queries=1, passages=1),
            folds.CaseLoad("2", queries=1, passages=1),
        ]
        assigned, trace, order = folds.assign_cases_greedily(
            synthetic,
            capacities=(1, 1),
        )
        self.assertEqual(order, ["2", "10"])
        self.assertEqual([fold.case_ids for fold in assigned], [["2"], ["10"]])
        self.assertEqual(
            [item["chosen_fold_id"] for item in trace],
            [0, 1],
        )

    def test_invalid_case_loads_fail_loudly(self) -> None:
        fixtures = [
            (
                [folds.CaseLoad("1", 1, 1), folds.CaseLoad("1", 1, 1)],
                (1, 1),
                "Duplicate case IDs",
            ),
            (
                [folds.CaseLoad("01", 1, 1)],
                (1,),
                "canonical non-negative integer",
            ),
            (
                [folds.CaseLoad("1", 0, 1)],
                (1,),
                "has no queries",
            ),
            (
                [folds.CaseLoad("1", 1, 1)],
                (2,),
                "capacities sum",
            ),
        ]
        for case_loads, capacities, message in fixtures:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    folds.validate_case_loads(case_loads, capacities=capacities)

    def test_refinement_rejects_inconsistent_fold_state(self) -> None:
        bad_load = [fold.copy() for fold in self.greedy]
        bad_load[0].queries += 1
        with self.assertRaisesRegex(ValueError, "stored loads do not match"):
            folds.refine_pair_swaps(bad_load, case_loads=self.case_loads)

        bad_id = [fold.copy() for fold in self.greedy]
        bad_id[0].fold_id = 4
        with self.assertRaisesRegex(ValueError, "Fold position 0"):
            folds.refine_pair_swaps(bad_id, case_loads=self.case_loads)

        duplicate_membership = [fold.copy() for fold in self.greedy]
        duplicate_membership[0].case_ids[0] = duplicate_membership[1].case_ids[0]
        with self.assertRaisesRegex(ValueError, "do not cover the same cases"):
            folds.refine_pair_swaps(
                duplicate_membership,
                case_loads=self.case_loads,
            )

    def test_rotations_are_disjoint_exhaustive_and_balanced(self) -> None:
        rotations = self.manifest["rotations"]
        self.assertEqual(len(rotations), 5)
        expected_train_cases = [24, 25, 26, 26, 25]
        expected_train_passages = [3172, 3171, 3176, 3169, 3170]
        test_occurrences = {load.case_id: 0 for load in self.case_loads}
        validation_occurrences = {load.case_id: 0 for load in self.case_loads}
        train_occurrences = {load.case_id: 0 for load in self.case_loads}
        for outer_fold, rotation in enumerate(rotations):
            self.assertEqual(rotation["outer_fold"], outer_fold)
            self.assertEqual(rotation["test"]["fold_ids"], [outer_fold])
            self.assertEqual(
                rotation["validation"]["fold_ids"],
                [(outer_fold + 1) % 5],
            )
            self.assertEqual(rotation["train"]["num_cases"], expected_train_cases[outer_fold])
            self.assertEqual(rotation["train"]["queries"], 294)
            self.assertEqual(
                rotation["train"]["passages"],
                expected_train_passages[outer_fold],
            )
            train = set(rotation["train"]["case_ids"])
            validation = set(rotation["validation"]["case_ids"])
            test = set(rotation["test"]["case_ids"])
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(len(train | validation | test), 42)
            for case_id in train:
                train_occurrences[case_id] += 1
            for case_id in validation:
                validation_occurrences[case_id] += 1
            for case_id in test:
                test_occurrences[case_id] += 1
        self.assertEqual(set(train_occurrences.values()), {3})
        self.assertEqual(set(validation_occurrences.values()), {1})
        self.assertEqual(set(test_occurrences.values()), {1})

    def test_manifest_has_only_exact_integer_or_rational_values(self) -> None:
        assert_no_float(self, self.manifest)
        self.assertEqual(
            Fraction(self.manifest["pair_swap_refinement"]["objective"]),
            Fraction(27_941_923, 69_854_490),
        )
        self.assertEqual(
            self.manifest["generator"]["source_sha256"],
            sha256(Path(folds.__file__)),
        )


class FoldManifestIOTest(unittest.TestCase):
    def test_two_freezes_are_byte_identical_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            folds.freeze_fold_manifest(dataset_dir=DATASET_DIR, output_path=first)
            folds.freeze_fold_manifest(dataset_dir=DATASET_DIR, output_path=second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            stored = folds.validate_frozen_fold_manifest(
                dataset_dir=DATASET_DIR,
                fold_manifest_path=first,
            )
            self.assertEqual(stored, json.loads(first.read_text(encoding="utf-8")))

    def test_pythonhashseed_does_not_change_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = []
            for seed in ("1", "999"):
                output = root / f"folds-{seed}.json"
                environment = dict(os.environ)
                environment["PYTHONHASHSEED"] = seed
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "corporate_reorganization.modernbert.experiments.retrieval_cv.folds",
                        "freeze",
                        "--dataset-dir",
                        str(DATASET_DIR),
                        "--output",
                        str(output),
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                outputs.append(output)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())

    def test_existing_file_directory_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_file = root / "file"
            existing_file.write_text("unchanged", encoding="utf-8")
            existing_directory = root / "directory"
            existing_directory.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(existing_file)
            for output in (existing_file, existing_directory, symlink):
                with self.subTest(output=output):
                    with self.assertRaises(FileExistsError):
                        folds.freeze_fold_manifest(
                            dataset_dir=DATASET_DIR,
                            output_path=output,
                        )
            self.assertEqual(existing_file.read_text(encoding="utf-8"), "unchanged")

    def test_stale_frozen_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "folds.json"
            folds.freeze_fold_manifest(dataset_dir=DATASET_DIR, output_path=path)
            stored = json.loads(path.read_text(encoding="utf-8"))
            stored["totals"]["queries"] = 489
            path.write_text(
                json.dumps(stored, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not exactly match"):
                folds.validate_frozen_fold_manifest(
                    dataset_dir=DATASET_DIR,
                    fold_manifest_path=path,
                )

    def test_semantically_equal_noncanonical_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "folds.json"
            folds.freeze_fold_manifest(dataset_dir=DATASET_DIR, output_path=path)
            stored = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(
                json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical deterministic bytes"):
                folds.validate_frozen_fold_manifest(
                    dataset_dir=DATASET_DIR,
                    fold_manifest_path=path,
                )

    def test_dataset_output_hash_mismatch_is_rejected(self) -> None:
        relative_paths = [
            "cases.jsonl",
            "corpus.jsonl",
            "queries/all.jsonl",
            "pools/candidates_by_case.json",
            "pools/candidates_global.json",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            records = {}
            for relative_path in relative_paths:
                path = dataset_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                records[relative_path] = {"sha256": sha256(path)}
            manifest = {"output_files": records}
            (dataset_dir / "corpus.jsonl").write_text(
                '{"changed": true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch for corpus.jsonl"):
                folds._verify_dataset_file_hashes(dataset_dir, manifest)


class FrozenScientificConfigTest(unittest.TestCase):
    def test_canonical_fold_manifest_validates_exactly(self) -> None:
        stored = folds.validate_frozen_fold_manifest(
            dataset_dir=DATASET_DIR,
            fold_manifest_path=FOLD_CONFIG,
        )
        self.assertEqual(
            stored["folds"],
            folds.build_fold_manifest(DATASET_DIR)["folds"],
        )

    def test_experiment_spec_hashes_and_matrix_are_locked(self) -> None:
        experiment = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(experiment["schema_version"], 1)
        self.assertEqual(experiment["experiment_id"], "arr_retrieval_cv_v1")
        self.assertEqual(
            experiment["dataset"]["manifest_sha256"],
            sha256(DATASET_DIR / "dataset_manifest.json"),
        )
        self.assertEqual(
            experiment["folds"]["manifest_sha256"],
            sha256(FOLD_CONFIG),
        )
        self.assertEqual(
            experiment["folds"]["generator_source_sha256"],
            sha256(Path(folds.__file__)),
        )
        matrix = experiment["run_matrix"]
        expected_runs = (
            5
            * len(matrix["seeds"])
            * len(matrix["query_views"])
            * len(matrix["samplers"])
        )
        self.assertEqual(expected_runs, 60)
        self.assertEqual(matrix["controlled_full_runs"], expected_runs)
        self.assertEqual(matrix["total_training_submissions"], 64)
        self.assertEqual(matrix["seeds"], [17, 29, 43])
        self.assertEqual(
            experiment["training"]["outer_training_case_range"],
            [24, 26],
        )
        self.assertEqual(experiment["training"]["outer_training_queries"], 294)

    def test_experiment_spec_has_no_local_paths_or_timestamps(self) -> None:
        raw = EXPERIMENT_CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("/home/", raw)
        self.assertNotIn("/tmp/", raw)
        self.assertNotIn("timestamp", raw.lower())


if __name__ == "__main__":
    unittest.main()
