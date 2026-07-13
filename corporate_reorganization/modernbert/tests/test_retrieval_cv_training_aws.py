from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock, patch

MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from corporate_reorganization.modernbert import train_sm as controlled_entrypoint
from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    aws,
    manifest,
    training_aws,
)
from corporate_reorganization.modernbert.retriever import provenance
from corporate_reorganization.modernbert.tests.test_retrieval_cv_aws import (
    ACCOUNT,
    REGION,
    ROLE,
    aws_config,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_config import (
    valid_scientific_config,
)


def _scientific_config() -> dict[str, object]:
    value = valid_scientific_config()
    study = value["study"]
    study.update(
        {
            "dataset_manifest_sha256": training_aws.DATASET_MANIFEST_SHA256,
            "evaluation_image_digest": aws.EXPECTED_LOCAL_IMAGE_DIGEST,
            "evaluation_image_inventory_sha256": "6" * 64,
            "evaluation_image_uri": (
                f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/arr-retrieval-eval@"
                f"{aws.EXPECTED_LOCAL_IMAGE_DIGEST}"
            ),
            "model_snapshot_tree_sha256": provenance.EXPECTED_SNAPSHOT_TREE_SHA256,
            "training_base_image_uri": training_aws.BASE_TRAINING_IMAGE_URI,
            "training_image_digest": training_aws.TRAINING_IMAGE_DIGEST,
            "training_image_inventory_sha256": (
                training_aws.TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
            ),
            "training_image_uri": (
                f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/arr-retrieval-eval@"
                f"{training_aws.TRAINING_IMAGE_DIGEST}"
            ),
        }
    )
    value["sources"]["include_paths"] = ["train_sm.py"]
    for template in value["run_templates"].values():
        template["input_channels"] = {
            "base_model": {
                "identity_sha256": provenance.EXPECTED_SNAPSHOT_TREE_SHA256,
                "s3_uri": (
                    "s3://ir-sagemaker/arr-retrieval-cv/inputs/modernbert-"
                    + provenance.EXPECTED_SNAPSHOT_TREE_SHA256
                ),
            },
            "data": {
                "identity_sha256": training_aws.DATASET_MANIFEST_SHA256,
                "s3_uri": (
                    "s3://ir-sagemaker/arr-retrieval-cv/inputs/dataset-"
                    + training_aws.DATASET_MANIFEST_SHA256
                ),
            },
        }
    return value


def _training_plan(root: Path) -> tuple[dict[str, object], Path]:
    source = root / "source"
    source.mkdir()
    (source / "train_sm.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    temporary_bundle = manifest.build_source_bundle(
        source_root=source,
        include_paths=["train_sm.py"],
        output_path=root / "temporary.tar.gz",
        commit_epoch=1_700_000_000,
    )
    final_path = root / f"source-{temporary_bundle.sha256}.tar.gz"
    temporary_bundle.path.rename(final_path)
    bundle = dataclasses.replace(
        temporary_bundle,
        path=final_path,
        bundler_runtime=copy.deepcopy(manifest.EXPECTED_BUNDLER_RUNTIME),
    )
    plan = manifest.build_dry_manifest(
        scientific_config=_scientific_config(),
        aws_local_config=aws_config(),
        source_bundle=bundle,
    )
    return plan, final_path


def _staging_receipt(plan: dict[str, object]) -> dict[str, object]:
    bucket = plan["infrastructure"]["artifact_bucket"]
    channels = {
        name: plan["controlled_runs"][0]["input_channels"][name]["s3_uri"] + "/"
        for name in ("base_model", "data")
    }
    prefixes = {
        "base_model": channels["base_model"].removeprefix(f"s3://{bucket}/"),
        "data": channels["data"].removeprefix(f"s3://{bucket}/"),
        "source": (
            f"{plan['infrastructure']['artifact_root_prefix']}/training-inputs/"
            f"source-{plan['sources']['source_bundle_sha256']}/"
        ),
    }
    channels["source"] = f"s3://{bucket}/{prefixes['source']}"
    expected = {
        "base_model": training_aws._SNAPSHOT_FILES,
        "data": training_aws._DATASET_FILES,
        "source": {
            plan["sources"]["source_bundle_path"]: (
                plan["sources"]["source_bundle_size"],
                plan["sources"]["source_bundle_sha256"],
            )
        },
    }
    records = []
    for group, files in expected.items():
        for relative, (size, digest) in files.items():
            records.append(
                {
                    "bucket": bucket,
                    "etag": f'"{len(records):032x}"',
                    "group": group,
                    "key": prefixes[group] + relative,
                    "logical_path": relative,
                    "schema_version": 1,
                    "sha256": digest,
                    "size": size,
                    "sse": "AES256",
                    "version_id": f"version-{len(records):02d}",
                }
            )
    records.sort(key=lambda record: record["key"])
    return {
        "channels": channels,
        "input_contracts": {
            "dataset_manifest_sha256": training_aws.DATASET_MANIFEST_SHA256,
            "model_snapshot_manifest_sha256": (
                training_aws.SNAPSHOT_MANIFEST_SHA256
            ),
            "model_snapshot_tree_sha256": (
                provenance.EXPECTED_SNAPSHOT_TREE_SHA256
            ),
        },
        "objects": records,
        "plan_sha256": aws.sha256_bytes(aws.canonical_json_bytes(plan)),
        "prefixes": prefixes,
        "protocol": training_aws.TRAINING_STAGING_PROTOCOL,
        "schema_version": 1,
    }


class ControlledTrainingRequestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan, self.source_bundle = _training_plan(self.root)
        self.staging = _staging_receipt(self.plan)
        self.run = self.plan["controlled_runs"][0]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_image_and_toolkit_provenance_cross_module_bindings(self) -> None:
        self.assertEqual(
            training_aws.TRAINING_IMAGE_DIGEST,
            aws.EXPECTED_TRAINING_IMAGE_DIGEST,
        )
        self.assertEqual(
            training_aws.TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
            provenance.EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
        )
        self.assertEqual(
            training_aws.BASE_TRAINING_IMAGE_URI,
            provenance.EXPECTED_BASE_TRAINING_IMAGE,
        )
        self.assertEqual(
            training_aws.TRAINING_IMAGE_URI,
            provenance.EXPECTED_DERIVED_TRAINING_IMAGE,
        )
        self.assertEqual(
            training_aws.SNAPSHOT_MANIFEST_SHA256,
            provenance.EXPECTED_SNAPSHOT_MANIFEST_SHA256,
        )
        self.assertEqual(
            training_aws.DATASET_MANIFEST_SHA256,
            provenance.EXPECTED_DATASET_MANIFEST_SHA256,
        )

    def test_exact_toolkit_mapping_and_user_argv(self) -> None:
        rendered = training_aws.render_toolkit_hyperparameters(
            job_name=self.run["job_name"],
            region=REGION,
            logical_hyperparameters=self.run["cell"],
        )
        arguments = training_aws.toolkit_user_command_arguments(rendered)
        self.assertEqual(
            arguments,
            [
                "--experiment-seed",
                "17",
                "--outer-fold",
                "0",
                "--query-view",
                "flat_masked",
                "--sampler",
                "local_unique",
            ],
        )
        self.assertNotIn("outer_fold", rendered)
        self.assertEqual(rendered["outer-fold"], "0")
        with patch.dict(
            os.environ,
            {
                "SM_CHANNEL_BASE_MODEL": "/opt/ml/input/data/base_model",
                "SM_CHANNEL_DATA": "/opt/ml/input/data/data",
                "SM_MODEL_DIR": "/opt/ml/model",
            },
            clear=True,
        ):
            parsed = controlled_entrypoint.parse_args(arguments)
        self.assertEqual(parsed.outer_fold, 0)
        self.assertEqual(parsed.query_view, "flat_masked")
        self.assertEqual(parsed.sampler, "local_unique")
        self.assertEqual(parsed.experiment_seed, 17)
        for mutation in (
            {**rendered, "unknown": "1"},
            {key: value for key, value in rendered.items() if key != "outer-fold"},
            {**rendered, "outer-fold": "[0]"},
            {**rendered, "outer-fold": "00"},
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises((TypeError, ValueError)):
                    training_aws.toolkit_user_command_arguments(mutation)

    def test_full_golden_create_training_job_request(self) -> None:
        request = training_aws.render_controlled_training_request(
            training_plan=self.plan,
            run_id=self.run["run_id"],
            staging_receipt=self.staging,
        )
        hyperparameters = training_aws.render_toolkit_hyperparameters(
            job_name=self.run["job_name"],
            region=REGION,
            logical_hyperparameters=self.run["cell"],
        )
        expected_environment = copy.deepcopy(self.run["environment"])
        expected_environment.update(
            {
                "ARR_TRAINING_BASE_IMAGE_URI": training_aws.BASE_TRAINING_IMAGE_URI,
                "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256": (
                    training_aws.TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
                ),
                "ARR_TRAINING_IMAGE_URI": self.plan["study"]["training_image_uri"],
                "ARR_SOURCE_BUNDLE_NAME": self.plan["sources"][
                    "source_bundle_path"
                ],
                "ARR_SOURCE_BUNDLE_SHA256": self.plan["sources"][
                    "source_bundle_sha256"
                ],
                "ARR_SOURCE_BUNDLE_SIZE": str(
                    self.plan["sources"]["source_bundle_size"]
                ),
                "ARR_SOURCE_COMMIT_EPOCH": str(
                    self.plan["sources"]["commit_epoch"]
                ),
                "ARR_SOURCE_INVENTORY_SHA256": self.plan["sources"][
                    "source_inventory_sha256"
                ],
                "ARR_TRAINING_PLAN_SHA256": aws.sha256_bytes(
                    aws.canonical_json_bytes(self.plan)
                ),
                "ARR_TRAINING_STAGING_RECEIPT_SHA256": aws.sha256_bytes(
                    aws.canonical_json_bytes(self.staging)
                ),
            }
        )
        expected_channels = []
        for name in ("base_model", "data", "source"):
            expected_channels.append(
                {
                    "ChannelName": name,
                    "CompressionType": "None",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "FullyReplicated",
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.staging["channels"][name],
                        }
                    },
                    "InputMode": "File",
                    "RecordWrapperType": "None",
                }
            )
        expected = {
            "AlgorithmSpecification": {
                "EnableSageMakerMetricsTimeSeries": False,
                "TrainingImage": self.plan["study"]["training_image_uri"],
                "TrainingInputMode": "File",
            },
            "EnableManagedSpotTraining": False,
            "EnableNetworkIsolation": True,
            "Environment": expected_environment,
            "HyperParameters": hyperparameters,
            "InputDataConfig": expected_channels,
            "OutputDataConfig": {
                "CompressionType": "GZIP",
                "S3OutputPath": self.run["output_prefix"],
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.g5.12xlarge",
                "VolumeSizeInGB": 200,
            },
            "RoleArn": ROLE,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86_400},
            "Tags": [
                {"Key": key, "Value": training_aws.TRAINING_TAGS[key]}
                for key in sorted(training_aws.TRAINING_TAGS)
            ],
            "TrainingJobName": self.run["job_name"],
        }
        self.assertEqual(request, expected)
        self.assertEqual(
            request["HyperParameters"]["sagemaker_program"], '"bootstrap.py"'
        )
        self.assertEqual(
            request["HyperParameters"]["sagemaker_submit_directory"],
            '"/opt/training_bootstrap"',
        )
        self.assertTrue(
            all(
                channel["DataSource"]["S3DataSource"]["S3Uri"].endswith("/")
                for channel in request["InputDataConfig"]
            )
        )

    def test_request_is_cross_bound_to_plan_staging_and_rerendered_receipt(self) -> None:
        receipt = training_aws.build_controlled_training_request_receipt(
            training_plan=self.plan,
            run_id=self.run["run_id"],
            staging_receipt=self.staging,
        )
        self.assertEqual(
            training_aws.validate_controlled_training_request_receipt(
                receipt,
                training_plan=self.plan,
                staging_receipt=self.staging,
            ),
            receipt,
        )
        attacks = []
        changed_request = copy.deepcopy(receipt)
        changed_request["request"]["ResourceConfig"]["VolumeSizeInGB"] = 16_384
        changed_request["request_sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(changed_request["request"])
        )
        attacks.append(changed_request)
        changed_bool_type = copy.deepcopy(receipt)
        changed_bool_type["request"]["EnableNetworkIsolation"] = 1
        attacks.append(changed_bool_type)
        changed_int_type = copy.deepcopy(receipt)
        changed_int_type["request"]["ResourceConfig"]["VolumeSizeInGB"] = 200.0
        attacks.append(changed_int_type)
        changed_toolkit = copy.deepcopy(receipt)
        changed_toolkit["toolkit_provenance"]["mapping_py_sha256"] = "0" * 64
        attacks.append(changed_toolkit)
        changed_staging = copy.deepcopy(self.staging)
        changed_staging["channels"]["data"] += "-other"
        with self.assertRaises(ValueError):
            training_aws.render_controlled_training_request(
                training_plan=self.plan,
                run_id=self.run["run_id"],
                staging_receipt=changed_staging,
            )
        sibling_scope = copy.deepcopy(self.staging)
        sibling_scope["channels"]["data"] = sibling_scope["channels"][
            "data"
        ].removesuffix("/")
        with self.assertRaisesRegex(ValueError, "channel URIs"):
            training_aws.render_controlled_training_request(
                training_plan=self.plan,
                run_id=self.run["run_id"],
                staging_receipt=sibling_scope,
            )
        changed_resources = copy.deepcopy(self.plan)
        changed_resources["infrastructure"]["training_volume_size_gb"] = 16_384
        with self.assertRaisesRegex(ValueError, "frozen training request"):
            training_aws.render_controlled_training_request(
                training_plan=changed_resources,
                run_id=self.run["run_id"],
                staging_receipt=self.staging,
            )
        for attack in attacks:
            with self.subTest(attack=attack):
                with self.assertRaisesRegex(ValueError, "re-rendering"):
                    training_aws.validate_controlled_training_request_receipt(
                        attack,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    )

        class AlwaysEqual:
            def __eq__(self, other: object) -> bool:
                return True

        non_json = copy.deepcopy(receipt)
        non_json["request"] = AlwaysEqual()
        with self.assertRaisesRegex(TypeError, "non-JSON"):
            training_aws.validate_controlled_training_request_receipt(
                non_json,
                training_plan=self.plan,
                staging_receipt=self.staging,
            )

        alternate_account = copy.deepcopy(self.plan)
        alternate = "111122223333"
        alternate_account["infrastructure"]["account_id"] = alternate
        alternate_account["infrastructure"]["role_arn"] = (
            f"arn:aws:iam::{alternate}:role/AmazonSageMakerExecutionRole"
        )
        for field in ("evaluation_image_uri", "training_image_uri"):
            alternate_account["study"][field] = alternate_account["study"][
                field
            ].replace(ACCOUNT, alternate)
        alternate_staging = _staging_receipt(alternate_account)
        with self.assertRaisesRegex(ValueError, "image provenance"):
            training_aws.render_controlled_training_request(
                training_plan=alternate_account,
                run_id=self.run["run_id"],
                staging_receipt=alternate_staging,
            )

    def test_stateful_hyperparameter_mapping_is_normalized_once(self) -> None:
        class Stateful(dict):
            def __getitem__(self, key: str) -> object:
                raise AssertionError("renderer must consume a normalized plain dict")

        logical = Stateful(self.run["cell"])
        rendered = training_aws.render_toolkit_hyperparameters(
            job_name=self.run["job_name"],
            region=REGION,
            logical_hyperparameters=logical,
        )
        self.assertEqual(rendered["experiment-seed"], "17")


class CorrectedLegacyTrainingRequestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan, _ = _training_plan(self.root)
        self.staging = _staging_receipt(self.plan)
        self.runs = self.plan["auxiliary_runs"][:2]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_toolkit_mapping_and_train_sm_parser_roundtrip(self) -> None:
        expected_views = ("flat_masked", "structured")
        for run, query_view in zip(self.runs, expected_views):
            with self.subTest(run_id=run["run_id"]):
                rendered = (
                    training_aws.render_corrected_legacy_toolkit_hyperparameters(
                        job_name=run["job_name"],
                        region=REGION,
                        logical_hyperparameters=run["hyperparameters"],
                    )
                )
                self.assertEqual(
                    rendered,
                    {
                        "base-seed": "17",
                        "epochs": "20",
                        "query-view": f'"{query_view}"',
                        "run-kind": '"corrected_legacy_diagnostic"',
                        "sagemaker_container_log_level": "20",
                        "sagemaker_job_name": f'"{run["job_name"]}"',
                        "sagemaker_mpi_enabled": "true",
                        "sagemaker_mpi_num_of_processes_per_host": "4",
                        "sagemaker_program": '"bootstrap.py"',
                        "sagemaker_region": '"us-east-1"',
                        "sagemaker_submit_directory": '"/opt/training_bootstrap"',
                        "total-optimizer-updates": "80",
                    },
                )
                arguments = training_aws.toolkit_user_command_arguments(rendered)
                self.assertEqual(
                    arguments,
                    [
                        "--base-seed",
                        "17",
                        "--epochs",
                        "20",
                        "--query-view",
                        query_view,
                        "--run-kind",
                        "corrected_legacy_diagnostic",
                        "--total-optimizer-updates",
                        "80",
                    ],
                )
                with patch.dict(
                    os.environ,
                    {
                        "SM_CHANNEL_BASE_MODEL": "/opt/ml/input/data/base_model",
                        "SM_CHANNEL_DATA": "/opt/ml/input/data/data",
                        "SM_MODEL_DIR": "/opt/ml/model",
                    },
                    clear=True,
                ):
                    parsed = controlled_entrypoint.parse_args(arguments)
                self.assertEqual(parsed.base_seed, 17)
                self.assertEqual(parsed.epochs, 20)
                self.assertEqual(parsed.query_view, query_view)
                self.assertEqual(parsed.run_kind, "corrected_legacy_diagnostic")
                self.assertEqual(parsed.total_optimizer_updates, 80)
                self.assertIsNone(parsed.experiment_seed)
                self.assertIsNone(parsed.outer_fold)
                self.assertIsNone(parsed.sampler)

    def test_two_exact_non_submitting_requests_and_receipts(self) -> None:
        expected_job_names = (
            "arr-ret-cv1-corrected-legacy-flat-a1",
            "arr-ret-cv1-corrected-legacy-structured-a1",
        )
        for run, job_name in zip(self.runs, expected_job_names):
            with self.subTest(run_id=run["run_id"]):
                request = training_aws.render_corrected_legacy_training_request(
                    training_plan=self.plan,
                    run_id=run["run_id"],
                    staging_receipt=self.staging,
                )
                matching_controlled = next(
                    candidate
                    for candidate in self.plan["controlled_runs"]
                    if candidate["cell"]
                    == {
                        "outer_fold": 0,
                        "query_view": run["cell"]["query_view"],
                        "sampler": "local_unique",
                        "experiment_seed": 17,
                    }
                )
                expected = training_aws.render_controlled_training_request(
                    training_plan=self.plan,
                    run_id=matching_controlled["run_id"],
                    staging_receipt=self.staging,
                )
                expected["HyperParameters"] = (
                    training_aws.render_corrected_legacy_toolkit_hyperparameters(
                        job_name=run["job_name"],
                        region=REGION,
                        logical_hyperparameters=run["hyperparameters"],
                    )
                )
                expected["OutputDataConfig"]["S3OutputPath"] = run["output_prefix"]
                expected["TrainingJobName"] = run["job_name"]
                expected["Environment"]["ARR_TRAINING_RUN_ID"] = run["run_id"]
                expected_request_payload_sha256 = aws.sha256_bytes(
                    aws.canonical_json_bytes(expected)
                )
                expected["Environment"][
                    "ARR_TRAINING_REQUEST_PAYLOAD_SHA256"
                ] = expected_request_payload_sha256
                self.assertEqual(request, expected)
                self.assertEqual(request["TrainingJobName"], job_name)
                self.assertEqual(
                    request["OutputDataConfig"]["S3OutputPath"],
                    run["output_prefix"],
                )
                self.assertEqual(
                    [row["ChannelName"] for row in request["InputDataConfig"]],
                    ["base_model", "data", "source"],
                )
                self.assertTrue(request["EnableNetworkIsolation"])
                self.assertFalse(request["EnableManagedSpotTraining"])
                self.assertEqual(request["Environment"]["PYTHONHASHSEED"], "17")
                self.assertEqual(
                    request["Environment"]["ARR_TRAINING_REQUEST_PAYLOAD_SHA256"],
                    expected_request_payload_sha256,
                )

                receipt = (
                    training_aws.build_corrected_legacy_training_request_receipt(
                        training_plan=self.plan,
                        run_id=run["run_id"],
                        staging_receipt=self.staging,
                    )
                )
                self.assertEqual(
                    receipt["protocol"],
                    training_aws.CORRECTED_LEGACY_REQUEST_PROTOCOL,
                )
                self.assertEqual(
                    training_aws.validate_corrected_legacy_training_request_receipt(
                        receipt,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    ),
                    receipt,
                )

    def test_identity_schedule_schema_and_rerender_attacks_fail(self) -> None:
        for run_id in (
            self.plan["controlled_runs"][0]["run_id"],
            "determinism-smoke-a",
            "corrected-legacy-missing",
        ):
            with self.subTest(run_id=run_id):
                with self.assertRaisesRegex(
                    ValueError, "(?i)corrected-legacy auxiliary"
                ):
                    training_aws.render_corrected_legacy_training_request(
                        training_plan=self.plan,
                        run_id=run_id,
                        staging_receipt=self.staging,
                    )

        logical = copy.deepcopy(self.runs[0]["hyperparameters"])
        malformed = []
        for field, replacement in (
            ("base_seed", True),
            ("epochs", 20.0),
            ("query_view", "flat"),
            ("run_kind", "controlled_full"),
            ("total_optimizer_updates", 60),
        ):
            changed = copy.deepcopy(logical)
            changed[field] = replacement
            malformed.append(changed)
        malformed.extend(
            [
                {**logical, "outer_fold": 0},
                {key: value for key, value in logical.items() if key != "epochs"},
            ]
        )
        for changed in malformed:
            with self.subTest(logical=changed):
                with self.assertRaises((TypeError, ValueError)):
                    training_aws.validate_corrected_legacy_logical_hyperparameters(
                        changed
                    )

        receipt = training_aws.build_corrected_legacy_training_request_receipt(
            training_plan=self.plan,
            run_id=self.runs[0]["run_id"],
            staging_receipt=self.staging,
        )
        for field, replacement in (
            ("EnableNetworkIsolation", 1),
            ("EnableManagedSpotTraining", True),
        ):
            attack = copy.deepcopy(receipt)
            attack["request"][field] = replacement
            attack["request_sha256"] = aws.sha256_bytes(
                aws.canonical_json_bytes(attack["request"])
            )
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "re-rendering"):
                    training_aws.validate_corrected_legacy_training_request_receipt(
                        attack,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    )


class DeterminismSmokeTrainingRequestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan, _ = _training_plan(self.root)
        self.staging = _staging_receipt(self.plan)
        self.smokes = self.plan["auxiliary_runs"][2:]
        self.first, self.second = self.smokes

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_smoke_toolkit_mapping_user_argv_and_golden_request(self) -> None:
        logical = {
            "epochs": 2,
            "experiment_seed": 17,
            "outer_fold": 0,
            "query_view": "structured",
            "run_kind": "determinism_smoke",
            "sampler": "global_uniform",
            "total_optimizer_updates": 6,
        }
        rendered = training_aws.render_determinism_smoke_toolkit_hyperparameters(
            job_name=self.first["job_name"],
            region=REGION,
            logical_hyperparameters=logical,
        )
        expected_hyperparameters = {
            "epochs": "2",
            "experiment-seed": "17",
            "outer-fold": "0",
            "query-view": '"structured"',
            "run-kind": '"determinism_smoke"',
            "sagemaker_container_log_level": "20",
            "sagemaker_job_name": f'"{self.first["job_name"]}"',
            "sagemaker_mpi_enabled": "true",
            "sagemaker_mpi_num_of_processes_per_host": "4",
            "sagemaker_program": '"bootstrap.py"',
            "sagemaker_region": '"us-east-1"',
            "sagemaker_submit_directory": '"/opt/training_bootstrap"',
            "sampler": '"global_uniform"',
            "total-optimizer-updates": "6",
        }
        self.assertEqual(rendered, expected_hyperparameters)
        self.assertEqual(
            training_aws.toolkit_user_command_arguments(rendered),
            [
                "--epochs",
                "2",
                "--experiment-seed",
                "17",
                "--outer-fold",
                "0",
                "--query-view",
                "structured",
                "--run-kind",
                "determinism_smoke",
                "--sampler",
                "global_uniform",
                "--total-optimizer-updates",
                "6",
            ],
        )

        request = training_aws.render_determinism_smoke_training_request(
            training_plan=self.plan,
            run_id=self.first["run_id"],
            staging_receipt=self.staging,
        )
        matching_controlled = next(
            run
            for run in self.plan["controlled_runs"]
            if run["cell"]
            == {
                "outer_fold": 0,
                "query_view": "structured",
                "sampler": "global_uniform",
                "experiment_seed": 17,
            }
        )
        expected = training_aws.render_controlled_training_request(
            training_plan=self.plan,
            run_id=matching_controlled["run_id"],
            staging_receipt=self.staging,
        )
        expected["HyperParameters"] = expected_hyperparameters
        expected["OutputDataConfig"]["S3OutputPath"] = self.first["output_prefix"]
        expected["TrainingJobName"] = self.first["job_name"]
        self.assertEqual(request, expected)
        self.assertEqual([row["ChannelName"] for row in request["InputDataConfig"]], [
            "base_model",
            "data",
            "source",
        ])
        self.assertTrue(
            all(
                row["DataSource"]["S3DataSource"]["S3Uri"].endswith("/")
                for row in request["InputDataConfig"]
            )
        )
        self.assertTrue(request["EnableNetworkIsolation"])
        self.assertFalse(request["EnableManagedSpotTraining"])
        self.assertFalse(any("replica" in key.lower() for key in request["Environment"]))

    def test_two_receipts_are_exactly_equivalent_outside_launch_coordinates(self) -> None:
        receipts = [
            training_aws.build_determinism_smoke_training_request_receipt(
                training_plan=self.plan,
                run_id=run["run_id"],
                staging_receipt=self.staging,
            )
            for run in self.smokes
        ]
        for receipt in receipts:
            self.assertEqual(
                training_aws.validate_determinism_smoke_training_request_receipt(
                    receipt,
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                ),
                receipt,
            )
            self.assertEqual(
                receipt["protocol"],
                training_aws.DETERMINISM_SMOKE_REQUEST_PROTOCOL,
            )
            self.assertNotIn("replica_id", receipt)
            self.assertFalse(
                any("replica" in key.lower() for key in receipt["request"]["Environment"])
            )
        equivalence = training_aws.validate_determinism_smoke_request_equivalence(
            receipts[1],
            receipts[0],
            training_plan=self.plan,
            staging_receipt=self.staging,
        )
        self.assertEqual(
            equivalence["protocol"],
            training_aws.DETERMINISM_SMOKE_EQUIVALENCE_PROTOCOL,
        )
        self.assertEqual(
            [row["run_id"] for row in equivalence["launch_coordinates"]],
            ["determinism-smoke-a", "determinism-smoke-b"],
        )
        self.assertEqual(
            equivalence["user_argv"],
            training_aws.toolkit_user_command_arguments(
                receipts[0]["request"]["HyperParameters"]
            ),
        )
        first_request, second_request = [row["request"] for row in receipts]
        self.assertNotEqual(first_request["TrainingJobName"], second_request["TrainingJobName"])
        self.assertNotEqual(
            first_request["OutputDataConfig"]["S3OutputPath"],
            second_request["OutputDataConfig"]["S3OutputPath"],
        )
        self.assertNotEqual(
            first_request["HyperParameters"]["sagemaker_job_name"],
            second_request["HyperParameters"]["sagemaker_job_name"],
        )

    def test_wrong_kind_cell_schedule_replica_and_malformed_types_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "(?i)determinism-smoke auxiliary"):
            training_aws.render_determinism_smoke_training_request(
                training_plan=self.plan,
                run_id=self.plan["controlled_runs"][0]["run_id"],
                staging_receipt=self.staging,
            )
        with self.assertRaisesRegex(ValueError, "(?i)determinism-smoke auxiliary"):
            training_aws.render_determinism_smoke_training_request(
                training_plan=self.plan,
                run_id=self.plan["auxiliary_runs"][0]["run_id"],
                staging_receipt=self.staging,
            )

        mutations = []
        wrong_kind = copy.deepcopy(self.plan)
        wrong_kind["auxiliary_runs"][2]["kind"] = manifest.LEGACY_KIND
        mutations.append(wrong_kind)
        wrong_cell = copy.deepcopy(self.plan)
        wrong_cell["auxiliary_runs"][2]["cell"]["outer_fold"] = 1
        mutations.append(wrong_cell)
        wrong_schedule = copy.deepcopy(self.plan)
        wrong_schedule["auxiliary_runs"][2]["hyperparameters"]["epochs"] = 3
        mutations.append(wrong_schedule)
        wrong_replica = copy.deepcopy(self.plan)
        wrong_replica["auxiliary_runs"][2]["launch_metadata"]["replica_id"] = "b"
        mutations.append(wrong_replica)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    training_aws.render_determinism_smoke_training_request(
                        training_plan=changed,
                        run_id="determinism-smoke-a",
                        staging_receipt=self.staging,
                    )

        valid = copy.deepcopy(training_aws.DETERMINISM_SMOKE_LOGICAL_HYPERPARAMETERS)
        malformed = []
        for field, replacement in (
            ("outer_fold", False),
            ("epochs", True),
            ("total_optimizer_updates", 6.0),
            ("experiment_seed", "17"),
            ("query_view", "flat_masked"),
            ("sampler", "local_unique"),
            ("run_kind", "controlled_full"),
        ):
            changed = copy.deepcopy(valid)
            changed[field] = replacement
            malformed.append(changed)
        malformed.extend(
            [
                {**valid, "replica_id": "a"},
                {key: value for key, value in valid.items() if key != "epochs"},
            ]
        )
        for changed in malformed:
            with self.subTest(logical=changed):
                with self.assertRaises((TypeError, ValueError)):
                    training_aws.validate_determinism_smoke_logical_hyperparameters(
                        changed
                    )

    def test_replica_leakage_unexpected_differences_and_type_attacks_fail(self) -> None:
        receipts = [
            training_aws.build_determinism_smoke_training_request_receipt(
                training_plan=self.plan,
                run_id=run["run_id"],
                staging_receipt=self.staging,
            )
            for run in self.smokes
        ]
        leaked_plan = copy.deepcopy(self.plan)
        leaked_plan["auxiliary_runs"][2]["environment"]["ARR_REPLICA_ID"] = "a"
        with self.assertRaises(ValueError):
            training_aws.render_determinism_smoke_training_request(
                training_plan=leaked_plan,
                run_id="determinism-smoke-a",
                staging_receipt=self.staging,
            )

        attacks = []
        leaked_receipt = copy.deepcopy(receipts[0])
        leaked_receipt["request"]["Environment"]["ARR_REPLICA_ID"] = "a"
        leaked_receipt["request_sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(leaked_receipt["request"])
        )
        attacks.append(leaked_receipt)
        unexpected = copy.deepcopy(receipts[0])
        unexpected["request"]["ResourceConfig"]["VolumeSizeInGB"] = 201
        unexpected["request_sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(unexpected["request"])
        )
        attacks.append(unexpected)
        bool_attack = copy.deepcopy(receipts[0])
        bool_attack["request"]["EnableNetworkIsolation"] = 1
        attacks.append(bool_attack)
        float_attack = copy.deepcopy(receipts[0])
        float_attack["request"]["ResourceConfig"]["VolumeSizeInGB"] = 200.0
        attacks.append(float_attack)
        for attack in attacks:
            with self.subTest(attack=attack):
                with self.assertRaisesRegex(ValueError, "re-rendering"):
                    training_aws.validate_determinism_smoke_request_equivalence(
                        attack,
                        receipts[1],
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    )


class TrainingStagingReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.plan, _ = _training_plan(Path(self.temporary.name))
        self.receipt = _staging_receipt(self.plan)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _remote(self) -> Mock:
        s3 = Mock()
        records = {record["key"]: record for record in self.receipt["objects"]}

        def list_versions(**request: object) -> dict[str, object]:
            prefix = request["Prefix"]
            return {
                "DeleteMarkers": [],
                "IsTruncated": False,
                "MaxKeys": 1000,
                "Name": "ir-sagemaker",
                "Prefix": prefix,
                "Versions": [
                    {
                        "ETag": record["etag"],
                        "IsLatest": True,
                        "Key": record["key"],
                        "Size": record["size"],
                        "VersionId": record["version_id"],
                    }
                    for record in records.values()
                    if record["key"].startswith(prefix)
                ],
            }

        def head(**request: object) -> dict[str, object]:
            record = records[request["Key"]]
            import base64

            return {
                "ChecksumSHA256": base64.b64encode(
                    bytes.fromhex(record["sha256"])
                ).decode("ascii"),
                "ContentLength": record["size"],
                "ETag": record["etag"],
                "Metadata": {"sha256": record["sha256"]},
                "ServerSideEncryption": "AES256",
                "VersionId": record["version_id"],
            }

        s3.list_object_versions.side_effect = list_versions
        s3.head_object.side_effect = head
        return s3

    def test_receipt_and_remote_version_history_are_exact(self) -> None:
        validated = training_aws.validate_training_staging_receipt(
            self.receipt, training_plan=self.plan
        )
        self.assertEqual(validated, self.receipt)
        s3 = self._remote()
        self.assertEqual(
            training_aws.verify_remote_training_staging(
                s3,
                training_plan=self.plan,
                staging_receipt=self.receipt,
                deep_read=False,
            ),
            self.receipt,
        )
        self.assertEqual(s3.list_object_versions.call_count, 3)
        self.assertEqual(s3.head_object.call_count, 12)

    def test_staging_coordinator_checks_all_prefixes_before_exactly_twelve_puts(self) -> None:
        descriptors = [
            {
                "group": record["group"],
                "key": record["key"],
                "logical_path": record["logical_path"],
                "path": Path("/not-read-by-this-unit-test") / record["logical_path"],
            }
            for record in self.receipt["objects"]
        ]
        staged_by_key = {
            record["key"]: {
                key: record[key]
                for key in (
                    "bucket",
                    "etag",
                    "key",
                    "schema_version",
                    "sha256",
                    "size",
                    "sse",
                    "version_id",
                )
            }
            for record in self.receipt["objects"]
        }
        unused = Mock()

        def stage_once(_s3: object, **arguments: object) -> dict[str, object]:
            self.assertEqual(unused.call_count, 3)
            self.assertEqual(arguments["expected_bucket_owner"], ACCOUNT)
            return copy.deepcopy(staged_by_key[arguments["key"]])

        with (
            patch.object(
                training_aws,
                "_staging_descriptors",
                return_value=(self.receipt["prefixes"], descriptors),
            ),
            patch.object(aws, "validate_artifact_bucket") as validate_bucket,
            patch.object(aws, "assert_unused_versioned_prefix", unused),
            patch.object(aws, "stage_file_once", side_effect=stage_once) as stage,
        ):
            actual = training_aws.stage_training_inputs_once(
                self._remote(),
                training_plan=self.plan,
                source_bundle_path=Path("source.tar.gz"),
                dataset_dir=Path("dataset"),
                base_model_dir=Path("base-model"),
                snapshot_manifest_path=Path("snapshot.json"),
            )
        self.assertEqual(actual, self.receipt)
        validate_bucket.assert_called_once_with(
            ANY,
            bucket="ir-sagemaker",
            region="us-east-1",
        )
        self.assertEqual(unused.call_count, 3)
        self.assertEqual(stage.call_count, 12)

    def test_receipt_and_remote_drift_fail_loudly(self) -> None:
        for field, value in (
            ("sha256", "0" * 64),
            ("size", 1),
        ):
            changed = copy.deepcopy(self.receipt)
            changed["objects"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    training_aws.validate_training_staging_receipt(
                        changed, training_plan=self.plan
                    )

        for field, value in (
            ("version_id", "changed"),
            ("etag", '"ffffffffffffffffffffffffffffffff"'),
        ):
            changed = copy.deepcopy(self.receipt)
            changed["objects"][0][field] = value
            with self.subTest(remote_field=field):
                with self.assertRaises(RuntimeError):
                    training_aws.verify_remote_training_staging(
                        self._remote(),
                        training_plan=self.plan,
                        staging_receipt=changed,
                        deep_read=False,
                    )

        extra = self._remote()
        original = extra.list_object_versions.side_effect

        def with_extra(**request: object) -> dict[str, object]:
            response = original(**request)
            if request["Prefix"] == self.receipt["prefixes"]["source"]:
                response["Versions"].append(
                    {
                        "ETag": '"00000000000000000000000000000000"',
                        "IsLatest": True,
                        "Key": request["Prefix"] + "extra",
                        "Size": 1,
                        "VersionId": "extra",
                    }
                )
            return response

        extra.list_object_versions.side_effect = with_extra
        with self.assertRaisesRegex(RuntimeError, "version count"):
            training_aws.verify_remote_training_staging(
                extra,
                training_plan=self.plan,
                staging_receipt=self.receipt,
                deep_read=False,
            )

    def test_version_pagination_and_deep_read_byte_drift_helpers(self) -> None:
        s3 = Mock()
        prefix = self.receipt["prefixes"]["source"]
        record = next(
            row for row in self.receipt["objects"] if row["group"] == "source"
        )
        s3.list_object_versions.side_effect = [
            {
                "IsTruncated": True,
                "MaxKeys": 1000,
                "Name": "ir-sagemaker",
                "NextKeyMarker": record["key"],
                "NextVersionIdMarker": record["version_id"],
                "Prefix": prefix,
                "Versions": [],
            },
            {
                "IsTruncated": False,
                "MaxKeys": 1000,
                "Name": "ir-sagemaker",
                "Prefix": prefix,
                "Versions": [
                    {
                        "ETag": record["etag"],
                        "IsLatest": True,
                        "Key": record["key"],
                        "Size": record["size"],
                        "VersionId": record["version_id"],
                    }
                ],
            },
        ]
        versions, delete_markers = training_aws._list_prefix_versions(
            s3,
            bucket="ir-sagemaker",
            prefix=prefix,
            expected_bucket_owner=ACCOUNT,
        )
        self.assertEqual(len(versions), 1)
        self.assertEqual(delete_markers, [])
        self.assertEqual(
            s3.list_object_versions.call_args_list[1].kwargs["KeyMarker"],
            record["key"],
        )
        payload = b"verified staged bytes\n"
        training_aws._verify_readback_stream(
            io.BytesIO(payload),
            expected_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            name="fixture",
        )
        with self.assertRaisesRegex(RuntimeError, "bytes changed"):
            training_aws._verify_readback_stream(
                io.BytesIO(payload + b"changed"),
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                name="fixture",
            )


if __name__ == "__main__":
    unittest.main()
