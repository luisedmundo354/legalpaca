from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from corporate_reorganization.modernbert.experiments.retrieval_cv import config


def _template(
    hyperparameters: dict[str, object],
    artifact_type: str,
    validator_version: str,
    *,
    entry_point: str = "train_sm.py",
) -> dict[str, object]:
    return {
        "entry_point": entry_point,
        "hyperparameters": hyperparameters,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "input_channels": {
            "base_model": {
                "s3_uri": "s3://ir-sagemaker/arr/base-model-" + "4" * 64,
                "identity_sha256": "4" * 64,
            },
            "data": {
                "s3_uri": "s3://ir-sagemaker/arr/data-" + "2" * 64,
                "identity_sha256": "2" * 64,
            },
        },
        "expected_artifact_identity": {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "validator_version": validator_version,
        },
    }


def valid_scientific_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "study": {
            "study_id": "arr_retrieval_cv_v1",
            "training_base_image_uri": (
                "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                "huggingface-pytorch-training@sha256:" + "9" * 64
            ),
            "experiment_config_sha256": "0" * 64,
            "fold_manifest_sha256": "1" * 64,
            "dataset_manifest_sha256": "2" * 64,
            "deepspeed_config_sha256": "3" * 64,
            "model_snapshot_tree_sha256": "4" * 64,
            "evaluation_image_uri": (
                "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@"
                "sha256:" + "5" * 64
            ),
            "evaluation_image_digest": "sha256:" + "5" * 64,
            "evaluation_image_inventory_sha256": "6" * 64,
            "training_image_uri": (
                "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@"
                "sha256:" + "7" * 64
            ),
            "training_image_digest": "sha256:" + "7" * 64,
            "training_image_inventory_sha256": "8" * 64,
        },
        "sources": {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "commit_epoch": 1_700_000_000,
            "include_paths": ["corporate_reorganization/modernbert", "requirements.txt"],
        },
        "run_templates": {
            "controlled": _template(
                {}, "controlled_retriever", "controlled_retrieval_artifact_v1"
            ),
            "legacy": _template(
                {
                    "base_seed": 17,
                    "epochs": 20,
                    "run_kind": "corrected_legacy_diagnostic",
                    "total_optimizer_updates": 80,
                },
                "corrected_legacy_diagnostic_retriever",
                "corrected_legacy_diagnostic_artifact_v1",
            ),
            "determinism_smoke": _template(
                {
                    "epochs": 2,
                    "run_kind": "determinism_smoke",
                    "total_optimizer_updates": 6,
                },
                "determinism_smoke_retriever",
                "determinism_smoke_artifact_v1",
            ),
        },
    }


def valid_aws_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "account_id": "371087393859",
        "region": "us-east-1",
        "role_arn": "arn:aws:iam::371087393859:role/AmazonSageMakerExecutionRole",
        "artifact_bucket": "ir-sagemaker",
        "artifact_root_prefix": "arr/retrieval-cv",
        "ecr_repository": "arr-retrieval-eval",
        "training_instance_type": "ml.g5.12xlarge",
        "training_instance_count": 1,
        "training_volume_size_gb": 200,
        "training_max_runtime_seconds": 86_400,
        "processing_instance_type": "ml.g5.12xlarge",
        "processing_instance_count": 1,
        "processing_volume_size_gb": 100,
        "processing_max_runtime_seconds": 3_600,
        "max_concurrent_training_jobs": 4,
        "tags": {
            "Project": "arr-retrieval-cv",
            "Study": "arr-retrieval-cv-v1",
        },
    }


class CanonicalConfigTest(unittest.TestCase):
    def test_tracked_orchestration_binds_verified_study_inputs(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/retrieval_cv/configs/orchestration.json"
        )
        tracked, digest = config.load_scientific_config(path)
        self.assertEqual(
            digest,
            "4f1ac5d4512c44234fd74cd6bae16368368aaf966ee23ecc361d8e6571d1034e",
        )
        self.assertEqual(
            {
                key: tracked["sources"][key]
                for key in ("git_commit", "git_tree", "commit_epoch")
            },
            {
                "git_commit": "6bbde30dac849d750caba4c3f350c8d52c6a4dd2",
                "git_tree": "2e425028d1a2e0316f1c0b8234f4b56cad9d0fe0",
                "commit_epoch": 1783917519,
            },
        )
        self.assertEqual(
            tracked["study"]["training_image_digest"],
            "sha256:b44c9b182a2490329b25394568299420bcfbe85a8fb17df955378b1f3630d9be",
        )
        self.assertEqual(
            tracked["study"]["evaluation_image_digest"],
            "sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2",
        )
        input_set = (
            "s3://ir-sagemaker/arr-retrieval-cv/inputs/"
            "input-set-c6aac769be56bded5ab13c6b761d2e7dfaa0c9d096b70b43250e5dcd36d42b41/"
        )
        expected_channels = {
            "base_model": (
                input_set
                + "modernbert-aca85feea4adb60c4b021eb1a439aff47c844495005f2acdee1baef9d611d63d"
            ),
            "data": (
                input_set
                + "dataset-cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be"
            ),
        }
        for template in tracked["run_templates"].values():
            self.assertEqual(
                {
                    name: record["s3_uri"]
                    for name, record in template["input_channels"].items()
                },
                expected_channels,
            )

    def test_single_read_hash_and_canonical_round_trip(self) -> None:
        value = valid_scientific_config()
        payload = config.canonical_json_bytes(value)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "orchestration.json"
            path.write_bytes(payload)
            loaded, digest = config.load_scientific_config(path)
            self.assertEqual(loaded, value)
            self.assertEqual(digest, config.sha256_bytes(payload))
            loaded_again, _ = config.load_scientific_config(
                path, expected_sha256=digest
            )
            self.assertEqual(loaded_again, value)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                config.load_scientific_config(path, expected_sha256="f" * 64)

    def test_noncanonical_duplicate_and_nonfinite_json_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(valid_scientific_config()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical deterministic bytes"):
                config.load_scientific_config(path)

            path.write_text('{"a":1,"a":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                config.load_canonical_json_object(path)

            path.write_text('{"a":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite JSON number"):
                config.load_canonical_json_object(path)

    def test_scientific_schema_rejects_unknowns_and_bool_as_int(self) -> None:
        unknown = valid_scientific_config()
        unknown["unexpected"] = "value"
        with self.assertRaisesRegex(ValueError, r"unknown=\['unexpected'\]"):
            config.validate_scientific_config(unknown)

        boolean_schema = valid_scientific_config()
        boolean_schema["schema_version"] = True
        with self.assertRaisesRegex(TypeError, "exact integer"):
            config.validate_scientific_config(boolean_schema)

        boolean_epoch = valid_scientific_config()
        boolean_epoch["sources"]["commit_epoch"] = True
        with self.assertRaisesRegex(TypeError, "exact integer"):
            config.validate_scientific_config(boolean_epoch)

        unknown_channel = valid_scientific_config()
        unknown_channel["run_templates"]["controlled"]["input_channels"]["data"][
            "unknown"
        ] = 1
        with self.assertRaisesRegex(ValueError, r"unknown=\['unknown'\]"):
            config.validate_scientific_config(unknown_channel)

    def test_generated_parameters_and_smoke_schedule_are_locked(self) -> None:
        prebound = valid_scientific_config()
        prebound["run_templates"]["controlled"]["hyperparameters"]["outer_fold"] = 0
        with self.assertRaisesRegex(ValueError, "must not pre-bind"):
            config.validate_scientific_config(prebound)

        wrong_smoke = valid_scientific_config()
        wrong_smoke["run_templates"]["determinism_smoke"]["hyperparameters"][
            "epochs"
        ] = 20
        with self.assertRaisesRegex(ValueError, "exactly two epochs"):
            config.validate_scientific_config(wrong_smoke)

        wrong_corrected_legacy = valid_scientific_config()
        wrong_corrected_legacy["run_templates"]["legacy"]["hyperparameters"][
            "total_optimizer_updates"
        ] = 60
        with self.assertRaisesRegex(
            ValueError, "Corrected legacy diagnostic template hyperparameters changed"
        ):
            config.validate_scientific_config(wrong_corrected_legacy)

        hashseed = valid_scientific_config()
        hashseed["run_templates"]["controlled"]["environment"][
            "PYTHONHASHSEED"
        ] = "17"
        with self.assertRaisesRegex(ValueError, "must not bind per-run"):
            config.validate_scientific_config(hashseed)

        mutable_image = valid_scientific_config()
        mutable_image["study"]["training_image_uri"] = (
            "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-train:latest"
        )
        with self.assertRaisesRegex(ValueError, "immutable private-ECR"):
            config.validate_scientific_config(mutable_image)

        mismatched_image = valid_scientific_config()
        mismatched_image["study"]["evaluation_image_uri"] = mismatched_image[
            "study"
        ]["evaluation_image_uri"].replace("5" * 64, "9" * 64)
        with self.assertRaisesRegex(ValueError, "separately bound digest"):
            config.validate_scientific_config(mismatched_image)

    def test_scientific_cross_field_identities_and_protocols_are_locked(self) -> None:
        mutations = []
        wrong_model = valid_scientific_config()
        wrong_model["run_templates"]["controlled"]["input_channels"]["base_model"][
            "identity_sha256"
        ] = "9" * 64
        mutations.append(wrong_model)
        wrong_channels = valid_scientific_config()
        wrong_channels["run_templates"]["legacy"]["input_channels"]["extra"] = {
            "identity_sha256": "a" * 64,
            "s3_uri": "s3://ir-sagemaker/arr/extra-" + "a" * 64,
        }
        mutations.append(wrong_channels)
        wrong_entry = valid_scientific_config()
        wrong_entry["run_templates"]["controlled"]["entry_point"] = "other.py"
        mutations.append(wrong_entry)
        wrong_artifact = valid_scientific_config()
        wrong_artifact["run_templates"]["controlled"]["expected_artifact_identity"][
            "validator_version"
        ] = "other_v1"
        mutations.append(wrong_artifact)
        wrong_environment = valid_scientific_config()
        wrong_environment["run_templates"]["controlled"]["environment"]["EXTRA"] = "1"
        mutations.append(wrong_environment)
        wrong_repository = valid_scientific_config()
        wrong_repository["study"]["training_image_uri"] = wrong_repository["study"][
            "training_image_uri"
        ].replace("arr-retrieval-eval", "arr-retrieval-train")
        mutations.append(wrong_repository)
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    config.validate_scientific_config(value)

    def test_aws_schema_is_exact_and_rejects_boolean_counts(self) -> None:
        valid = valid_aws_config()
        self.assertEqual(config.validate_aws_local_config(copy.deepcopy(valid)), valid)

        unknown = copy.deepcopy(valid)
        unknown["profile"] = "default"
        with self.assertRaisesRegex(ValueError, r"unknown=\['profile'\]"):
            config.validate_aws_local_config(unknown)

        boolean_count = copy.deepcopy(valid)
        boolean_count["training_instance_count"] = True
        with self.assertRaisesRegex(TypeError, "exact integer"):
            config.validate_aws_local_config(boolean_count)


if __name__ == "__main__":
    unittest.main()
