from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    corrected_legacy_config,
)
from corporate_reorganization.modernbert.retriever.provenance import (
    EXPECTED_DATASET_MANIFEST_SHA256,
)
from corporate_reorganization.modernbert.retriever.staged_data import (
    validate_staged_dataset,
)


MODERNBERT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = MODERNBERT_ROOT / "experiments/retrieval_cv/configs"
CONFIG_PATH = CONFIG_DIR / "corrected_legacy.json"
MEMBERSHIP_DIR = CONFIG_DIR / "corrected_legacy_membership"
DATASET_DIR = (
    MODERNBERT_ROOT.parent
    / "data/final_annotations_gold/processed_retrieval_v2"
)


def _tracked_value() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _copy_config_inputs(destination: Path) -> Path:
    destination.mkdir(parents=True)
    shutil.copyfile(CONFIG_PATH, destination / CONFIG_PATH.name)
    shutil.copytree(MEMBERSHIP_DIR, destination / MEMBERSHIP_DIR.name)
    return destination / CONFIG_PATH.name


class CorrectedLegacyConfigTest(unittest.TestCase):
    def test_tracked_contract_and_memberships_are_exact(self) -> None:
        loaded = corrected_legacy_config.load_corrected_legacy_config(CONFIG_PATH)
        self.assertEqual(
            loaded.config_sha256,
            corrected_legacy_config.CORRECTED_LEGACY_CONFIG_SHA256,
        )
        self.assertEqual(len(loaded.memberships.train), 34)
        self.assertEqual(loaded.memberships.validation, ("45", "47", "60", "62"))
        self.assertEqual(loaded.memberships.test, ("37", "46", "65", "96"))
        self.assertEqual(
            set(loaded.memberships.train)
            | set(loaded.memberships.validation)
            | set(loaded.memberships.test),
            {
                "36", "37", "38", "40", "41", "42", "45", "46", "47",
                "48", "49", "57", "58", "59", "60", "62", "63", "65",
                "66", "67", "68", "69", "70", "71", "72", "73", "74",
                "75", "76", "77", "78", "79", "80", "83", "85", "86",
                "87", "91", "92", "94", "96", "97",
            },
        )

    def test_staged_dataset_reconstructs_exact_raw_line_subsets(self) -> None:
        validate_staged_dataset(
            dataset_dir=DATASET_DIR,
            expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        )
        loaded = corrected_legacy_config.load_corrected_legacy_config(
            CONFIG_PATH,
            dataset_dir=DATASET_DIR,
        )
        self.assertEqual(loaded.value["membership"]["train"]["query_count"], 418)
        self.assertEqual(
            loaded.value["membership"]["train"]["passage_count"], 4_307
        )

    def test_staged_dataset_binds_both_pool_files_and_exact_inventory(self) -> None:
        for relative_path in (
            "pools/candidates_by_case.json",
            "pools/candidates_global.json",
        ):
            with self.subTest(relative_path=relative_path), tempfile.TemporaryDirectory() as temporary:
                copied_dataset = Path(temporary) / "dataset"
                shutil.copytree(DATASET_DIR, copied_dataset)
                target = copied_dataset / relative_path
                target.write_bytes(target.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                    validate_staged_dataset(
                        dataset_dir=copied_dataset,
                        expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            copied_dataset = Path(temporary) / "dataset"
            shutil.copytree(DATASET_DIR, copied_dataset)
            (copied_dataset / "unexpected.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                validate_staged_dataset(
                    dataset_dir=copied_dataset,
                    expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
                )

    def test_unknown_fields_type_substitution_and_value_drift_fail(self) -> None:
        unknown = _tracked_value()
        unknown["unknown"] = "forbidden"
        with self.assertRaisesRegex(ValueError, r"unknown=\['unknown'\]"):
            corrected_legacy_config.validate_corrected_legacy_config(unknown)

        boolean_epoch = _tracked_value()
        boolean_epoch["training"]["epochs"] = True
        with self.assertRaisesRegex(TypeError, "exact int, not bool"):
            corrected_legacy_config.validate_corrected_legacy_config(boolean_epoch)

        numeric_substitution = _tracked_value()
        numeric_substitution["training"]["epochs"] = 20.0
        with self.assertRaisesRegex(TypeError, "exact int, not float"):
            corrected_legacy_config.validate_corrected_legacy_config(
                numeric_substitution
            )

        changed_replacement = _tracked_value()
        changed_replacement["candidate_sampling"]["replacement"] = "sample_anyway"
        with self.assertRaisesRegex(ValueError, "replacement changed"):
            corrected_legacy_config.validate_corrected_legacy_config(
                changed_replacement
            )

        uppercase_digest = _tracked_value()
        uppercase_digest["dataset"]["manifest_sha256"] = (
            uppercase_digest["dataset"]["manifest_sha256"].upper()
        )
        with self.assertRaisesRegex(ValueError, "manifest_sha256 changed"):
            corrected_legacy_config.validate_corrected_legacy_config(
                uppercase_digest
            )

    def test_noncanonical_duplicate_and_nonfinite_json_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(_tracked_value()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical deterministic bytes"):
                corrected_legacy_config.load_corrected_legacy_config(
                    path,
                    expected_sha256=None,
                )

            path.write_text('{"schema_version":1,"schema_version":1}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                corrected_legacy_config.load_corrected_legacy_config(
                    path,
                    expected_sha256=None,
                )

            path.write_text('{"value":NaN}\n')
            with self.assertRaisesRegex(ValueError, "Non-finite JSON number"):
                corrected_legacy_config.load_corrected_legacy_config(
                    path,
                    expected_sha256=None,
                )

    def test_config_and_membership_symlinks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_link = root / "corrected_legacy.json"
            config_link.symlink_to(CONFIG_PATH)
            with self.assertRaisesRegex(ValueError, "Symbolic links are forbidden"):
                corrected_legacy_config.load_corrected_legacy_config(config_link)

        with tempfile.TemporaryDirectory() as temporary:
            copied_path = _copy_config_inputs(Path(temporary) / "configs")
            train_path = copied_path.parent / "corrected_legacy_membership/train_cases.txt"
            train_path.unlink()
            train_path.symlink_to(MEMBERSHIP_DIR / "train_cases.txt")
            with self.assertRaisesRegex(ValueError, "Symbolic links are forbidden"):
                corrected_legacy_config.load_corrected_legacy_config(copied_path)

    def test_membership_byte_drift_and_expected_hash_format_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied_path = _copy_config_inputs(Path(temporary) / "configs")
            train_path = copied_path.parent / "corrected_legacy_membership/train_cases.txt"
            train_path.write_bytes(train_path.read_bytes() + b"98\n")
            with self.assertRaisesRegex(ValueError, "train membership hash mismatch"):
                corrected_legacy_config.load_corrected_legacy_config(copied_path)

        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            corrected_legacy_config.load_corrected_legacy_config(
                CONFIG_PATH,
                expected_sha256="A" * 64,
            )

    def test_dataset_directory_and_files_must_not_be_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_link = Path(temporary) / "dataset"
            dataset_link.symlink_to(DATASET_DIR, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symbolic links are forbidden"):
                corrected_legacy_config.load_corrected_legacy_config(
                    CONFIG_PATH,
                    dataset_dir=dataset_link,
                )

        with tempfile.TemporaryDirectory() as temporary:
            copied_dataset = Path(temporary) / "dataset"
            shutil.copytree(DATASET_DIR, copied_dataset)
            queries_path = copied_dataset / "queries/all.jsonl"
            queries_path.unlink()
            queries_path.symlink_to(DATASET_DIR / "queries/all.jsonl")
            with self.assertRaisesRegex(ValueError, "Symbolic links are forbidden"):
                corrected_legacy_config.load_corrected_legacy_config(
                    CONFIG_PATH,
                    dataset_dir=copied_dataset,
                )


if __name__ == "__main__":
    unittest.main()
