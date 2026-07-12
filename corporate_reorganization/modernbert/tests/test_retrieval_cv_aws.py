from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from corporate_reorganization.modernbert.experiments.retrieval_cv import aws


ACCOUNT = "371087393859"
REGION = "us-east-1"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/AmazonSageMakerExecutionRole"
IMAGE_URI = (
    f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/arr-retrieval-eval@"
    f"{aws.EXPECTED_LOCAL_IMAGE_DIGEST}"
)


def aws_config() -> dict[str, object]:
    return {
        "account_id": ACCOUNT,
        "artifact_bucket": "ir-sagemaker",
        "artifact_root_prefix": "arr-retrieval-cv/abc123",
        "ecr_repository": "arr-retrieval-eval",
        "max_concurrent_training_jobs": 4,
        "processing_instance_count": 1,
        "processing_instance_type": "ml.g5.12xlarge",
        "processing_max_runtime_seconds": 3600,
        "processing_volume_size_gb": 100,
        "region": REGION,
        "role_arn": ROLE,
        "schema_version": 1,
        "tags": {
            "Experiment": "arr_retrieval_cv_v1",
            "ManagedBy": "arr-retrieval-cv",
            "Purpose": "evaluation-image-runtime-smoke",
        },
        "training_instance_count": 1,
        "training_instance_type": "ml.g5.12xlarge",
        "training_max_runtime_seconds": 86400,
        "training_volume_size_gb": 200,
    }


def repository() -> dict[str, object]:
    return {
        "repositoryName": aws.ECR_REPOSITORY_NAME,
        "imageTagMutability": "IMMUTABLE",
        "imageScanningConfiguration": {"scanOnPush": True},
        "encryptionConfiguration": {"encryptionType": "AES256"},
    }


class AwsConfigAndRenderingTest(unittest.TestCase):
    def test_runtime_identity_ledger_is_exact_canonical_image_output(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments/retrieval_cv/configs/evaluation_runtime_identity.json"
        )
        raw = path.read_bytes()
        identity = json.loads(raw)
        self.assertEqual(raw, aws.canonical_json_bytes(identity))
        self.assertEqual(
            aws.sha256_bytes(raw),
            aws.EXPECTED_RUNTIME_IDENTITY_SHA256,
        )

    @unittest.skipUnless(
        os.environ.get("ARR_RUN_REAL_RETRIEVAL_IMAGE") == "1",
        "set ARR_RUN_REAL_RETRIEVAL_IMAGE=1 for the exact local-image binding",
    )
    def test_real_image_matches_runtime_identity_ledger_under_request_environment(self) -> None:
        request = aws.render_runtime_smoke_request(
            aws_config(),
            remote_image_uri=IMAGE_URI,
            job_name="arr-ret-cv1-runtime-smoke-a1",
        )
        app = request["AppSpecification"]
        command = ["docker", "run", "--rm", "--network", "none"]
        for key, value in sorted(request["Environment"].items()):
            command.extend(("-e", f"{key}={value}"))
        command.extend(
            (
                "--entrypoint",
                app["ContainerEntrypoint"][0],
                f"arr-retrieval-eval@{aws.EXPECTED_LOCAL_IMAGE_DIGEST}",
                *app["ContainerArguments"],
            )
        )
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )
        identity = json.loads(completed.stdout)
        self.assertEqual(
            aws.sha256_bytes(aws.canonical_json_bytes(identity)),
            aws.EXPECTED_RUNTIME_IDENTITY_SHA256,
        )
        ledger = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "experiments/retrieval_cv/configs/evaluation_runtime_identity.json"
            ).read_bytes()
        )
        self.assertEqual(identity, ledger)

    def test_config_and_request_are_exact(self) -> None:
        config = aws.validate_aws_config(aws_config())
        request = aws.render_runtime_smoke_request(
            config,
            remote_image_uri=IMAGE_URI,
            job_name="arr-ret-cv1-runtime-smoke-a1",
        )
        self.assertEqual(request["AppSpecification"]["ImageUri"], IMAGE_URI)
        self.assertEqual(
            request["AppSpecification"]["ContainerEntrypoint"],
            ["/opt/conda/bin/python"],
        )
        self.assertIn("image_smoke.py", request["AppSpecification"]["ContainerArguments"][0])
        self.assertTrue(request["NetworkConfig"]["EnableNetworkIsolation"])
        self.assertNotIn("ProcessingInputs", request)
        self.assertNotIn("ProcessingOutputConfig", request)
        self.assertEqual(
            request["Tags"],
            sorted(request["Tags"], key=lambda item: item["Key"]),
        )

    def test_unknown_bool_and_mutable_inputs_fail_loudly(self) -> None:
        mutations = []
        extra = aws_config()
        extra["unexpected"] = 1
        mutations.append(extra)
        boolean_volume = aws_config()
        boolean_volume["processing_volume_size_gb"] = True
        mutations.append(boolean_volume)
        wrong_bucket = aws_config()
        wrong_bucket["artifact_bucket"] = "sagemaker-us-east-1-371087393859"
        mutations.append(wrong_bucket)
        wrong_instance = aws_config()
        wrong_instance["processing_instance_type"] = "ml.m5.xlarge"
        mutations.append(wrong_instance)
        wrong_volume = aws_config()
        wrong_volume["processing_volume_size_gb"] = 16_384
        mutations.append(wrong_volume)
        wrong_runtime = aws_config()
        wrong_runtime["processing_max_runtime_seconds"] = 86_400
        mutations.append(wrong_runtime)
        wrong_training_volume = aws_config()
        wrong_training_volume["training_volume_size_gb"] = 1_000
        mutations.append(wrong_training_volume)
        wrong_training_runtime = aws_config()
        wrong_training_runtime["training_max_runtime_seconds"] = 172_800
        mutations.append(wrong_training_runtime)
        wrong_tags = aws_config()
        wrong_tags["tags"]["Purpose"] = "other"
        mutations.append(wrong_tags)
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    aws.validate_aws_config(value)

        with self.assertRaisesRegex(ValueError, "frozen account-local evaluation digest"):
            aws.render_runtime_smoke_request(
                aws_config(),
                remote_image_uri=IMAGE_URI.replace("@sha256:", ":latest-"),
                job_name="arr-ret-cv1-runtime-smoke-a1",
            )
        with self.assertRaisesRegex(ValueError, "frozen account-local evaluation digest"):
            aws.render_runtime_smoke_request(
                aws_config(),
                remote_image_uri=IMAGE_URI.replace(
                    aws.EXPECTED_LOCAL_IMAGE_DIGEST,
                    "sha256:" + "0" * 64,
                ),
                job_name="arr-ret-cv1-runtime-smoke-a1",
            )

    def test_local_image_identity_is_bound(self) -> None:
        fixture = {
            "Descriptor": {
                "annotations": {"config.digest": aws.EXPECTED_CONFIG_DIGEST},
                "digest": aws.EXPECTED_LOCAL_IMAGE_DIGEST,
                "mediaType": aws.ECR_MEDIA_TYPE,
                "size": 14665,
            },
            "Config": {
                "Labels": {
                    "io.arr-retrieval.build-identity-sha256": aws.EXPECTED_BUILD_IDENTITY,
                    "io.arr-retrieval.source-parent-commit": aws.EXPECTED_SOURCE_PARENT_COMMIT,
                    "io.arr-retrieval.source-parent-epoch": aws.EXPECTED_SOURCE_PARENT_EPOCH,
                    "io.arr-retrieval.source-parent-rfc3339": aws.EXPECTED_SOURCE_PARENT_RFC3339,
                }
            },
        }
        with patch.object(aws, "_docker_inspect", return_value=fixture):
            identity = aws.validate_local_evaluation_image()
        self.assertEqual(identity["manifest_digest"], aws.EXPECTED_LOCAL_IMAGE_DIGEST)
        changed = copy.deepcopy(fixture)
        changed["Descriptor"]["digest"] = "sha256:" + "0" * 64
        with patch.object(aws, "_docker_inspect", return_value=changed):
            with self.assertRaisesRegex(ValueError, "manifest digest"):
                aws.validate_local_evaluation_image()


class AwsOneShotFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk_patcher = patch.object(
            aws,
            "validate_aws_sdk_versions",
            return_value=copy.deepcopy(aws.EXPECTED_AWS_SDK_VERSIONS),
        )
        self.sdk_patcher.start()

    def tearDown(self) -> None:
        self.sdk_patcher.stop()

    def clients(self) -> aws.AwsClients:
        sts = Mock()
        sts.get_caller_identity.return_value = {
            "Account": ACCOUNT,
            "Arn": f"arn:aws:iam::{ACCOUNT}:user/lbrenap1",
        }
        iam = Mock()
        iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            }
        }
        raw_manifest = json.dumps(
            {"schemaVersion": 2, "test": "fixture"},
            sort_keys=True,
            separators=(",", ":"),
        )
        ecr = Mock()
        ecr.describe_repositories.return_value = {"repositories": [repository()]}
        ecr.batch_get_image.return_value = {
            "failures": [],
            "images": [{"imageManifest": raw_manifest}],
        }
        s3 = Mock()
        quotas = Mock()
        quotas.get_service_quota.return_value = {"Quota": {"Value": 1.0}}
        ec2 = Mock()
        ec2.describe_instance_type_offerings.return_value = {
            "InstanceTypeOfferings": [{"InstanceType": "g5.12xlarge"}]
        }
        sagemaker = Mock()
        sagemaker.list_processing_jobs.return_value = {"ProcessingJobSummaries": []}
        logs = Mock()
        return aws.AwsClients(sts, iam, ecr, s3, quotas, ec2, sagemaker, logs)

    def preflight(self, clients: aws.AwsClients) -> dict[str, object]:
        with patch.object(
            aws,
            "_raw_ecr_manifest_digest",
            return_value=aws.EXPECTED_LOCAL_IMAGE_DIGEST,
        ):
            return aws.preflight_runtime_smoke(
                clients,
                aws_config(),
                remote_image_uri=IMAGE_URI,
                job_name="arr-ret-cv1-runtime-smoke-a1",
            )

    def test_preflight_and_submit_make_one_create_call(self) -> None:
        clients = self.clients()
        receipt = self.preflight(clients)
        clients.sagemaker.create_processing_job.return_value = {
            "ProcessingJobArn": (
                f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/"
                "arr-ret-cv1-runtime-smoke-a1"
            )
        }
        submission = aws.submit_runtime_smoke(
            clients,
            preflight_receipt=receipt,
        )
        self.assertEqual(submission["protocol"], aws.RUNTIME_SMOKE_PROTOCOL)
        clients.sagemaker.create_processing_job.assert_called_once_with(**receipt["request"])

        changed = copy.deepcopy(receipt)
        changed["request"]["StoppingCondition"]["MaxRuntimeInSeconds"] += 1
        changed["request_sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(changed["request"])
        )
        with self.assertRaisesRegex(ValueError, "re-rendered"):
            aws.submit_runtime_smoke(clients, preflight_receipt=changed)
        self.assertEqual(clients.sagemaker.create_processing_job.call_count, 1)

    def test_submit_rejects_self_hashed_resource_role_image_and_schema_attacks(self) -> None:
        clients = self.clients()
        receipt = self.preflight(clients)
        attacks = []
        for path, value in (
            (("request", "AppSpecification", "ImageUri"), IMAGE_URI.replace(
                aws.EXPECTED_LOCAL_IMAGE_DIGEST, "sha256:" + "0" * 64
            )),
            (("request", "RoleArn"), f"arn:aws:iam::{ACCOUNT}:role/foreign"),
            (("request", "ProcessingResources", "ClusterConfig", "InstanceType"), "ml.p5.48xlarge"),
            (("request", "ProcessingResources", "ClusterConfig", "InstanceCount"), 99),
            (("request", "ProcessingResources", "ClusterConfig", "VolumeSizeInGB"), 16_384),
        ):
            changed = copy.deepcopy(receipt)
            target = changed
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = value
            changed["request_sha256"] = aws.sha256_bytes(
                aws.canonical_json_bytes(changed["request"])
            )
            attacks.append(changed)
        extra = copy.deepcopy(receipt)
        extra["retry"] = True
        attacks.append(extra)
        for attack in attacks:
            with self.subTest(attack=attack):
                with self.assertRaises((TypeError, ValueError)):
                    aws.submit_runtime_smoke(clients, preflight_receipt=attack)
        clients.sagemaker.create_processing_job.assert_not_called()

    def test_preflight_rejects_collision_quota_and_wrong_receipt_kind(self) -> None:
        clients = self.clients()
        clients.sagemaker.list_processing_jobs.return_value = {
            "ProcessingJobSummaries": [
                {"ProcessingJobName": "arr-ret-cv1-runtime-smoke-a1"}
            ]
        }
        with patch.object(
            aws,
            "_raw_ecr_manifest_digest",
            return_value=aws.EXPECTED_LOCAL_IMAGE_DIGEST,
        ):
            with self.assertRaises(FileExistsError):
                aws.preflight_runtime_smoke(
                    clients,
                    aws_config(),
                    remote_image_uri=IMAGE_URI,
                    job_name="arr-ret-cv1-runtime-smoke-a1",
                )
        clients.service_quotas.get_service_quota.return_value = {"Quota": {"Value": 0.0}}
        clients.sagemaker.list_processing_jobs.return_value = {"ProcessingJobSummaries": []}
        with patch.object(
            aws,
            "_raw_ecr_manifest_digest",
            return_value=aws.EXPECTED_LOCAL_IMAGE_DIGEST,
        ):
            with self.assertRaisesRegex(RuntimeError, "quota"):
                aws.preflight_runtime_smoke(
                    clients,
                    aws_config(),
                    remote_image_uri=IMAGE_URI,
                    job_name="arr-ret-cv1-runtime-smoke-a1",
                )
        with self.assertRaisesRegex(ValueError, "schema changed"):
            aws.submit_runtime_smoke(
                clients,
                preflight_receipt={"protocol": "fold_evaluation_v1"},
            )

    def test_verify_requires_submission_and_reads_exact_paginated_stream(self) -> None:
        clients = self.clients()
        preflight = self.preflight(clients)
        clients.sagemaker.create_processing_job.return_value = {
            "ProcessingJobArn": (
                f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:processing-job/"
                "arr-ret-cv1-runtime-smoke-a1"
            )
        }
        submission = aws.submit_runtime_smoke(
            clients,
            preflight_receipt=preflight,
        )
        request = preflight["request"]
        clients.sagemaker.describe_processing_job.return_value = {
            **copy.deepcopy(request),
            "ProcessingJobArn": submission["job_arn"],
            "ProcessingJobStatus": "Completed",
        }
        clients.logs.describe_log_streams.side_effect = [
            {"logStreams": [], "nextToken": "page-2"},
            {
                "logStreams": [
                    {"logStreamName": "arr-ret-cv1-runtime-smoke-a1/algo-1"}
                ]
            },
        ]
        identity = {
            "build_context": {},
            "image_contract_sha256": "x",
            "neural_runtime": {},
            "platform": "linux/amd64",
            "sparse_runtime": {},
        }
        clients.logs.get_log_events.side_effect = [
            {"events": [], "nextForwardToken": "events-2"},
            {
                "events": [{"message": json.dumps(identity)}],
                "nextForwardToken": "events-2",
            },
        ]
        expected_identity_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(identity))
        with patch.object(
            aws,
            "EXPECTED_RUNTIME_IDENTITY_SHA256",
            expected_identity_sha256,
        ):
            verified = aws.verify_completed_runtime_smoke(
                clients,
                preflight_receipt=preflight,
                submission_receipt=submission,
            )
        self.assertEqual(verified["runtime_identity_sha256"], expected_identity_sha256)
        self.assertEqual(clients.logs.describe_log_streams.call_count, 2)
        self.assertEqual(clients.logs.get_log_events.call_count, 2)
        self.assertEqual(
            clients.logs.describe_log_streams.call_args_list[0].kwargs[
                "logStreamNamePrefix"
            ],
            "arr-ret-cv1-runtime-smoke-a1/",
        )

        changed_submission = copy.deepcopy(submission)
        changed_submission["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from preflight"):
            aws.verify_completed_runtime_smoke(
                clients,
                preflight_receipt=preflight,
                submission_receipt=changed_submission,
            )

    def test_s3_staging_is_conditional_versioned_and_read_back(self) -> None:
        s3 = Mock()
        body = b"exact staged bytes\n"
        import base64
        import hashlib

        checksum = base64.b64encode(hashlib.sha256(body).digest()).decode("ascii")
        s3.put_object.return_value = {
            "VersionId": "version-1",
            "ServerSideEncryption": "AES256",
            "ChecksumSHA256": checksum,
        }
        s3.head_object.return_value = {
            "VersionId": "version-1",
            "ContentLength": len(body),
            "ServerSideEncryption": "AES256",
            "ChecksumSHA256": checksum,
            "Metadata": {"sha256": hashlib.sha256(body).hexdigest()},
        }
        response_body = Mock()
        response_body.read.return_value = body
        s3.get_object.return_value = {"Body": response_body}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tar.gz"
            path.write_bytes(body)
            receipt = aws.stage_file_once(
                s3,
                source_path=path,
                bucket="ir-sagemaker",
                key="arr-retrieval-cv/commit/inputs/source.tar.gz",
            )
        self.assertEqual(receipt["version_id"], "version-1")
        self.assertEqual(receipt["sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(s3.put_object.call_count, 1)
        self.assertEqual(s3.put_object.call_args.kwargs["IfNoneMatch"], "*")
        s3.get_object.assert_called_once_with(
            Bucket="ir-sagemaker",
            Key="arr-retrieval-cv/commit/inputs/source.tar.gz",
            VersionId="version-1",
        )

    def test_unused_prefix_checks_versions_and_delete_markers(self) -> None:
        s3 = Mock()
        s3.list_object_versions.return_value = {
            "Versions": [],
            "DeleteMarkers": [],
            "IsTruncated": False,
        }
        aws.assert_unused_versioned_prefix(
            s3,
            bucket="ir-sagemaker",
            prefix="arr-retrieval-cv/commit/",
        )
        s3.list_object_versions.return_value["DeleteMarkers"] = [{"Key": "old"}]
        with self.assertRaises(FileExistsError):
            aws.assert_unused_versioned_prefix(
                s3,
                bucket="ir-sagemaker",
                prefix="arr-retrieval-cv/commit/",
            )


if __name__ == "__main__":
    unittest.main()
