from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, call, patch

from botocore.exceptions import ClientError

MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    aws,
    manifest,
    training_aws,
    training_launch,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_aws import (
    ACCOUNT,
    REGION,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_training_aws import (
    _staging_receipt,
    _training_plan,
)


def _resource_not_found() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ResourceNotFound", "Message": "not found"}},
        "DescribeTrainingJob",
    )


def _client_error(code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "failure"}},
        "DescribeTrainingJob",
    )


def _live_missing_training_job_error() -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Requested resource not found.",
            }
        },
        "DescribeTrainingJob",
    )


class TrainingLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan, _ = _training_plan(self.root)
        self.staging = _staging_receipt(self.plan)
        self.run_id = self.plan["controlled_runs"][0]["run_id"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _clients(self) -> aws.AwsClients:
        sts = Mock()
        sts.get_caller_identity.return_value = {
            "Account": ACCOUNT,
            "Arn": f"arn:aws:iam::{ACCOUNT}:user/retrieval-launcher",
            "UserId": "AIDAEXAMPLE",
        }
        iam = Mock()
        iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Action": "sts:AssumeRole",
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                        }
                    ],
                }
            }
        }
        ecr = Mock()
        ecr.describe_repositories.return_value = {
            "repositories": [
                {
                    "encryptionConfiguration": {"encryptionType": "AES256"},
                    "imageScanningConfiguration": {"scanOnPush": True},
                    "imageTagMutability": "IMMUTABLE",
                    "repositoryName": aws.ECR_REPOSITORY_NAME,
                }
            ]
        }
        ecr.batch_get_image.return_value = {
            "failures": [],
            "images": [
                {
                    "imageId": {
                        "imageDigest": training_aws.TRAINING_IMAGE_DIGEST
                    },
                    "imageManifest": "exact manifest bytes",
                    "imageManifestMediaType": aws.ECR_MEDIA_TYPE,
                    "registryId": ACCOUNT,
                    "repositoryName": aws.ECR_REPOSITORY_NAME,
                }
            ],
        }
        s3 = Mock()
        s3.get_bucket_location.return_value = {"LocationConstraint": None}
        s3.get_bucket_versioning.return_value = {"Status": "Enabled"}
        s3.get_bucket_encryption.return_value = {
            "ServerSideEncryptionConfiguration": {
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        },
                        "BucketKeyEnabled": True,
                    }
                ]
            }
        }
        s3.get_public_access_block.return_value = {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            }
        }
        s3.get_bucket_ownership_controls.return_value = {
            "OwnershipControls": {
                "Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]
            }
        }

        def unused_prefix(**arguments: object) -> dict[str, object]:
            return {
                "IsTruncated": False,
                "MaxKeys": arguments["MaxKeys"],
                "Name": arguments["Bucket"],
                "Prefix": arguments["Prefix"],
            }

        s3.list_object_versions.side_effect = unused_prefix
        service_quotas = Mock()
        service_quotas.get_service_quota.return_value = {
            "Quota": {
                "QuotaCode": training_launch.TRAINING_QUOTA_CODE,
                "QuotaName": training_launch.TRAINING_QUOTA_NAME,
                "ServiceCode": "sagemaker",
                "Value": 4.0,
            }
        }
        ec2 = Mock()
        ec2.describe_instance_type_offerings.return_value = {
            "InstanceTypeOfferings": [
                {
                    "InstanceType": training_launch.EC2_TRAINING_INSTANCE_TYPE,
                    "Location": REGION,
                    "LocationType": "region",
                }
            ]
        }
        sagemaker = Mock()
        sagemaker.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        sagemaker.describe_training_job.side_effect = _resource_not_found()
        return aws.AwsClients(
            sts=sts,
            iam=iam,
            ecr=ecr,
            s3=s3,
            service_quotas=service_quotas,
            ec2=ec2,
            sagemaker=sagemaker,
            logs=Mock(),
        )

    @contextmanager
    def _remote_dependencies(self):
        with (
            patch.object(
                training_launch.training_aws,
                "verify_remote_training_staging",
                side_effect=lambda *args, **kwargs: copy.deepcopy(self.staging),
            ) as staging_verify,
            patch.object(
                training_launch.aws,
                "validate_aws_sdk_versions",
                return_value=copy.deepcopy(aws.EXPECTED_AWS_SDK_VERSIONS),
            ),
            patch.object(
                training_launch.aws,
                "_raw_ecr_manifest_digest",
                return_value=training_aws.TRAINING_IMAGE_DIGEST,
            ),
        ):
            yield staging_verify

    def _preflight(
        self, clients: aws.AwsClients, *, run_id: str | None = None
    ) -> dict[str, object]:
        return training_launch.preflight_training_job(
            clients,
            training_plan=self.plan,
            staging_receipt=self.staging,
            run_id=self.run_id if run_id is None else run_id,
        )

    def _describe_response(
        self,
        preflight: dict[str, object],
        *,
        status: str = "InProgress",
    ) -> dict[str, object]:
        request = preflight["request_receipt"]["request"]
        response = {
            field: copy.deepcopy(request[field])
            for field in (
                "AlgorithmSpecification",
                "EnableManagedSpotTraining",
                "EnableNetworkIsolation",
                "Environment",
                "HyperParameters",
                "InputDataConfig",
                "OutputDataConfig",
                "ResourceConfig",
                "RoleArn",
                "StoppingCondition",
            )
        }
        created = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        response.update(
            {
                "CreationTime": created,
                "LastModifiedTime": created + timedelta(seconds=1),
                "ModelArtifacts": {"S3ModelArtifacts": ""},
                "SecondaryStatus": "Starting",
                "TrainingJobArn": (
                    f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/"
                    f"{request['TrainingJobName']}"
                ),
                "TrainingJobName": request["TrainingJobName"],
                "TrainingJobStatus": status,
            }
        )
        if status in {"Completed", "Failed", "Stopped"}:
            response.update(
                {
                    "BillableTimeInSeconds": 121,
                    "SecondaryStatus": status,
                    "TrainingEndTime": created + timedelta(seconds=122),
                    "TrainingStartTime": created + timedelta(seconds=2),
                    "TrainingTimeInSeconds": 120,
                }
            )
        if status == "Completed":
            response["ModelArtifacts"] = {
                "S3ModelArtifacts": (
                    f"{request['OutputDataConfig']['S3OutputPath']}/"
                    f"{request['TrainingJobName']}/output/model.tar.gz"
                )
            }
        if status == "Failed":
            response["FailureReason"] = "AlgorithmError: exact failure"
        if status == "Stopped":
            response["FailureReason"] = "Stopped by the operator"
        return response

    @staticmethod
    def _tag_pages(preflight: dict[str, object]) -> list[dict[str, object]]:
        tags = copy.deepcopy(preflight["request_receipt"]["request"]["Tags"])
        return [
            {"NextToken": "page-2", "Tags": tags[:1]},
            {"Tags": list(reversed(tags[1:]))},
        ]

    def _submit_in_progress(
        self, clients: aws.AwsClients
    ) -> tuple[dict[str, object], dict[str, object]]:
        preflight = self._preflight(clients)
        clients.sagemaker.reset_mock()
        job_arn = (
            f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/"
            f"{preflight['job_name']}"
        )
        clients.sagemaker.describe_training_job.side_effect = [
            _resource_not_found(),
            self._describe_response(preflight),
        ]
        clients.sagemaker.create_training_job.return_value = {
            "TrainingJobArn": job_arn
        }
        clients.sagemaker.list_tags.side_effect = self._tag_pages(preflight)
        submission = training_launch.submit_training_job_once(
            clients,
            training_plan=self.plan,
            staging_receipt=self.staging,
            preflight_receipt=preflight,
        )
        return preflight, submission

    def test_preflight_supports_all_three_exact_run_kinds(self) -> None:
        run_ids = (
            self.plan["controlled_runs"][0]["run_id"],
            next(
                run["run_id"]
                for run in self.plan["auxiliary_runs"]
                if run["kind"] == manifest.LEGACY_KIND
            ),
            next(
                run["run_id"]
                for run in self.plan["auxiliary_runs"]
                if run["kind"] == manifest.SMOKE_KIND
            ),
        )
        with self._remote_dependencies() as staging_verify:
            for run_id in run_ids:
                with self.subTest(run_id=run_id):
                    clients = self._clients()
                    receipt = self._preflight(clients, run_id=run_id)
                    self.assertEqual(
                        receipt["protocol"],
                        training_launch.TRAINING_PREFLIGHT_PROTOCOL,
                    )
                    self.assertEqual(receipt["run_id"], run_id)
                    training_launch.validate_training_preflight_receipt(
                        receipt,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    )
                    clients.sagemaker.create_training_job.assert_not_called()
            self.assertEqual(staging_verify.call_count, 3)
            self.assertTrue(
                all(
                    called.kwargs["deep_read"] is True
                    for called in staging_verify.call_args_list
                )
            )

    def test_preflight_uses_exact_quota_offering_and_collision_requests(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            receipt = self._preflight(clients)
        clients.service_quotas.get_service_quota.assert_called_once_with(
            ServiceCode="sagemaker",
            QuotaCode=training_launch.TRAINING_QUOTA_CODE,
        )
        clients.ec2.describe_instance_type_offerings.assert_called_once_with(
            Filters=[
                {
                    "Name": "instance-type",
                    "Values": [training_launch.EC2_TRAINING_INSTANCE_TYPE],
                }
            ],
            LocationType="region",
            MaxResults=100,
        )
        self.assertEqual(
            receipt["instance_offering"],
            {
                "ec2_instance_type": training_launch.EC2_TRAINING_INSTANCE_TYPE,
                "location": REGION,
                "location_type": "region",
                "sagemaker_instance_type": training_launch.TRAINING_INSTANCE_TYPE,
            },
        )
        clients.sagemaker.describe_training_job.assert_called_once_with(
            TrainingJobName=receipt["job_name"]
        )
        self.assertEqual(receipt["quota"]["value"], 4)
        self.assertEqual(
            clients.sagemaker.list_training_jobs.call_args_list,
            [
                call(
                    MaxResults=100,
                    SortBy="Name",
                    SortOrder="Ascending",
                    StatusEquals="InProgress",
                ),
                call(
                    MaxResults=100,
                    SortBy="Name",
                    SortOrder="Ascending",
                    StatusEquals="Stopping",
                ),
            ],
        )
        clients.sagemaker.create_training_job.assert_not_called()

    def test_active_planned_concurrency_paginates_and_allows_three(self) -> None:
        clients = self._clients()
        names = [run["job_name"] for run in self.plan["controlled_runs"][1:4]]

        def pages(**arguments: object) -> dict[str, object]:
            status = arguments["StatusEquals"]
            if status == "Stopping":
                return {"TrainingJobSummaries": []}
            if "NextToken" not in arguments:
                return {
                    "NextToken": "page-2",
                    "TrainingJobSummaries": [
                        {
                            "TrainingJobName": name,
                            "TrainingJobStatus": "InProgress",
                        }
                        for name in names[:2]
                    ],
                }
            self.assertEqual(arguments["NextToken"], "page-2")
            return {
                "TrainingJobSummaries": [
                    {
                        "TrainingJobName": names[2],
                        "TrainingJobStatus": "InProgress",
                    }
                ]
            }

        clients.sagemaker.list_training_jobs.side_effect = pages
        with self._remote_dependencies():
            receipt = self._preflight(clients)
        self.assertEqual(
            receipt["active_planned_jobs"],
            {"count": 3, "job_names": sorted(names)},
        )
        clients.sagemaker.create_training_job.assert_not_called()

    def test_active_planned_concurrency_four_fails_before_create(self) -> None:
        clients = self._clients()
        names = [run["job_name"] for run in self.plan["controlled_runs"][1:5]]
        clients.sagemaker.list_training_jobs.side_effect = lambda **arguments: {
            "TrainingJobSummaries": (
                [
                    {
                        "TrainingJobName": name,
                        "TrainingJobStatus": "InProgress",
                    }
                    for name in names
                ]
                if arguments["StatusEquals"] == "InProgress"
                else []
            )
        }
        with self._remote_dependencies():
            with self.assertRaisesRegex(RuntimeError, "frozen maximum"):
                self._preflight(clients)
        clients.sagemaker.describe_training_job.assert_not_called()
        clients.sagemaker.create_training_job.assert_not_called()

    def test_ecr_accepts_optional_tag_but_rejects_digest_drift(self) -> None:
        clients = self._clients()
        clients.ecr.batch_get_image.return_value["images"][0]["imageId"][
            "imageTag"
        ] = "frozen-build"
        with self._remote_dependencies():
            self._preflight(clients)
        clients = self._clients()
        clients.ecr.batch_get_image.return_value["images"][0]["imageId"][
            "imageDigest"
        ] = "sha256:" + "0" * 64
        with self._remote_dependencies():
            with self.assertRaises(ValueError):
                self._preflight(clients)
        clients.sagemaker.create_training_job.assert_not_called()

    def test_only_documented_or_exact_live_not_found_means_unused_job_name(self) -> None:
        with self._remote_dependencies():
            clients = self._clients()
            clients.sagemaker.describe_training_job.side_effect = (
                _live_missing_training_job_error()
            )
            receipt = self._preflight(clients)
            self.assertEqual(receipt["run_id"], self.run_id)
            clients.sagemaker.create_training_job.assert_not_called()

            for side_effect in (
                _client_error("AccessDeniedException"),
                _client_error("ValidationException"),
                ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": "Requested resource not found",
                        }
                    },
                    "DescribeTrainingJob",
                ),
            ):
                with self.subTest(error=side_effect.response["Error"]):
                    clients = self._clients()
                    clients.sagemaker.describe_training_job.side_effect = side_effect
                    with self.assertRaises(ClientError):
                        self._preflight(clients)
                    clients.sagemaker.create_training_job.assert_not_called()
            clients = self._clients()
            clients.sagemaker.describe_training_job.side_effect = None
            clients.sagemaker.describe_training_job.return_value = {
                "TrainingJobName": "already-present"
            }
            with self.assertRaises(FileExistsError):
                self._preflight(clients)
            clients.sagemaker.create_training_job.assert_not_called()

    def test_preflight_attacks_never_create_a_job(self) -> None:
        def staging_failure(clients: aws.AwsClients) -> None:
            del clients

        def caller_failure(clients: aws.AwsClients) -> None:
            clients.sts.get_caller_identity.return_value["Account"] = "0" * 12

        def role_failure(clients: aws.AwsClients) -> None:
            clients.iam.get_role.return_value["Role"][
                "AssumeRolePolicyDocument"
            ]["Statement"][0]["Principal"] = {"Service": "ec2.amazonaws.com"}

        def bucket_failure(clients: aws.AwsClients) -> None:
            clients.s3.get_bucket_versioning.return_value = {"Status": "Suspended"}

        def ecr_failure(clients: aws.AwsClients) -> None:
            clients.ecr.batch_get_image.return_value = {
                "failures": [{"failureCode": "ImageNotFound"}],
                "images": [],
            }

        def quota_failure(clients: aws.AwsClients) -> None:
            clients.service_quotas.get_service_quota.return_value["Quota"][
                "Value"
            ] = 3.0

        def offering_failure(clients: aws.AwsClients) -> None:
            clients.ec2.describe_instance_type_offerings.return_value[
                "InstanceTypeOfferings"
            ] = []

        def output_failure(clients: aws.AwsClients) -> None:
            def used(**arguments: object) -> dict[str, object]:
                return {
                    "IsTruncated": False,
                    "MaxKeys": arguments["MaxKeys"],
                    "Name": arguments["Bucket"],
                    "Prefix": arguments["Prefix"],
                    "Versions": [{"VersionId": "used"}],
                }

            clients.s3.list_object_versions.side_effect = used

        attacks = (
            ("caller", caller_failure),
            ("role", role_failure),
            ("bucket", bucket_failure),
            ("ecr", ecr_failure),
            ("quota", quota_failure),
            ("offering", offering_failure),
            ("output", output_failure),
        )
        del staging_failure
        with self._remote_dependencies():
            for name, mutate in attacks:
                with self.subTest(attack=name):
                    clients = self._clients()
                    mutate(clients)
                    with self.assertRaises((ValueError, FileExistsError)):
                        self._preflight(clients)
                    clients.sagemaker.create_training_job.assert_not_called()

    def test_deep_staging_failure_stops_before_other_remote_gates(self) -> None:
        clients = self._clients()
        with (
            patch.object(
                training_launch.training_aws,
                "verify_remote_training_staging",
                side_effect=RuntimeError("staging changed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "staging changed"):
                self._preflight(clients)
        clients.sts.get_caller_identity.assert_not_called()
        clients.sagemaker.create_training_job.assert_not_called()

    def test_quota_rejects_bool_fraction_nonfinite_and_wrong_name(self) -> None:
        with self._remote_dependencies():
            for value in (True, 4.5, math.inf, math.nan):
                with self.subTest(value=value):
                    clients = self._clients()
                    clients.service_quotas.get_service_quota.return_value[
                        "Quota"
                    ]["Value"] = value
                    with self.assertRaises(ValueError):
                        self._preflight(clients)
                    clients.sagemaker.create_training_job.assert_not_called()
            clients = self._clients()
            clients.service_quotas.get_service_quota.return_value["Quota"][
                "QuotaName"
            ] += "s"
            with self.assertRaises(ValueError):
                self._preflight(clients)
            clients.sagemaker.create_training_job.assert_not_called()

    def test_preflight_receipt_rejects_extra_key_and_type_attacks(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            receipt = self._preflight(clients)
        mutations = []
        extra = copy.deepcopy(receipt)
        extra["unexpected"] = 1
        mutations.append(extra)
        bool_schema = copy.deepcopy(receipt)
        bool_schema["schema_version"] = True
        mutations.append(bool_schema)
        float_quota = copy.deepcopy(receipt)
        float_quota["quota"]["value"] = 4.0
        mutations.append(float_quota)
        bad_hash = copy.deepcopy(receipt)
        bad_hash["receipt_sha256"] = "0" * 64
        mutations.append(bad_hash)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises((TypeError, ValueError)):
                    training_launch.validate_training_preflight_receipt(
                        mutation,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    )

    def test_preflight_receipt_rejects_resealed_cross_account_caller_arn(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            receipt = self._preflight(clients)
        receipt["caller_arn"] = (
            "arn:aws:sts::000000000000:assumed-role/launcher/session"
        )
        payload = {
            key: copy.deepcopy(value)
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
        receipt["receipt_sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(payload)
        )
        with self.assertRaisesRegex(ValueError, "planned AWS account"):
            training_launch.validate_training_preflight_receipt(
                receipt,
                training_plan=self.plan,
                staging_receipt=self.staging,
            )

    def test_submission_creates_once_then_immediately_describes_and_lists_tags(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            preflight, submission = self._submit_in_progress(clients)
        request = preflight["request_receipt"]["request"]
        self.assertNotIn("RetryStrategy", request)
        self.assertEqual(
            request["StoppingCondition"],
            {
                "MaxPendingTimeInSeconds": 7_200,
                "MaxRuntimeInSeconds": 86_400,
            },
        )
        clients.sagemaker.create_training_job.assert_called_once_with(**request)
        self.assertEqual(clients.sagemaker.describe_training_job.call_count, 2)
        self.assertEqual(clients.sagemaker.list_tags.call_count, 2)
        method_calls = clients.sagemaker.method_calls
        create_index = method_calls.index(call.create_training_job(**request))
        self.assertEqual(
            method_calls[create_index + 1],
            call.describe_training_job(TrainingJobName=preflight["job_name"]),
        )
        self.assertEqual(
            submission["snapshot"]["training_job_status"], "InProgress"
        )
        training_launch.validate_training_submission_receipt(
            submission,
            training_plan=self.plan,
            staging_receipt=self.staging,
            preflight_receipt=preflight,
        )

    def test_ambiguous_create_exception_propagates_without_reconciliation(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            preflight = self._preflight(clients)
            clients.sagemaker.reset_mock()
            clients.sagemaker.describe_training_job.side_effect = _resource_not_found()
            clients.sagemaker.create_training_job.side_effect = TimeoutError(
                "ambiguous transport failure"
            )
            with self.assertRaisesRegex(TimeoutError, "ambiguous"):
                training_launch.submit_training_job_once(
                    clients,
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    preflight_receipt=preflight,
                )
        clients.sagemaker.create_training_job.assert_called_once()
        clients.sagemaker.describe_training_job.assert_called_once_with(
            TrainingJobName=preflight["job_name"]
        )
        clients.sagemaker.list_tags.assert_not_called()

    def test_tampered_saved_preflight_causes_zero_remote_create(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            preflight = self._preflight(clients)
            clients.sagemaker.reset_mock()
            preflight["quota"]["value"] = 5
            with self.assertRaises(ValueError):
                training_launch.submit_training_job_once(
                    clients,
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    preflight_receipt=preflight,
                )
        clients.sagemaker.describe_training_job.assert_not_called()
        clients.sagemaker.create_training_job.assert_not_called()

    def test_create_arn_mismatch_stops_before_describe(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            preflight = self._preflight(clients)
            clients.sagemaker.reset_mock()
            clients.sagemaker.describe_training_job.side_effect = _resource_not_found()
            clients.sagemaker.create_training_job.return_value = {
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:000000000000:training-job/wrong"
            }
            with self.assertRaises(ValueError):
                training_launch.submit_training_job_once(
                    clients,
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    preflight_receipt=preflight,
                )
        clients.sagemaker.create_training_job.assert_called_once()
        clients.sagemaker.describe_training_job.assert_called_once()
        clients.sagemaker.list_tags.assert_not_called()

    def test_post_create_describe_mutations_fail_loudly(self) -> None:
        mutations = ("request", "arn", "retry", "tags")
        with self._remote_dependencies():
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    clients = self._clients()
                    preflight = self._preflight(clients)
                    clients.sagemaker.reset_mock()
                    described = self._describe_response(preflight)
                    if mutation == "request":
                        described["ResourceConfig"]["VolumeSizeInGB"] += 1
                    elif mutation == "arn":
                        described["TrainingJobArn"] += "-wrong"
                    elif mutation == "retry":
                        described["RetryStrategy"] = {"MaximumRetryAttempts": 1}
                    job_arn = (
                        f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/"
                        f"{preflight['job_name']}"
                    )
                    clients.sagemaker.describe_training_job.side_effect = [
                        _resource_not_found(),
                        described,
                    ]
                    clients.sagemaker.create_training_job.return_value = {
                        "TrainingJobArn": job_arn
                    }
                    tag_pages = self._tag_pages(preflight)
                    if mutation == "tags":
                        tag_pages[-1]["Tags"][0]["Value"] += "-wrong"
                    clients.sagemaker.list_tags.side_effect = tag_pages
                    with self.assertRaises(ValueError):
                        training_launch.submit_training_job_once(
                            clients,
                            training_plan=self.plan,
                            staging_receipt=self.staging,
                            preflight_receipt=preflight,
                        )
                    clients.sagemaker.create_training_job.assert_called_once()

    def test_status_receipt_deeply_revalidates_inputs_without_rerunning_preflight(self) -> None:
        clients = self._clients()
        with self._remote_dependencies() as staging_verify:
            preflight, submission = self._submit_in_progress(clients)
            prior_deep_reads = staging_verify.call_count
            clients.sagemaker.reset_mock()
            clients.sagemaker.describe_training_job.side_effect = None
            clients.sagemaker.describe_training_job.return_value = (
                self._describe_response(preflight, status="Completed")
            )
            clients.sagemaker.list_tags.side_effect = self._tag_pages(preflight)
            status = training_launch.describe_training_job_status(
                clients,
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=preflight,
                submission_receipt=submission,
            )
            self.assertEqual(staging_verify.call_count, prior_deep_reads + 1)
        self.assertEqual(status["snapshot"]["training_job_status"], "Completed")
        self.assertEqual(status["snapshot"]["training_time_seconds"], 120)
        training_launch.validate_training_status_receipt(
            status,
            training_plan=self.plan,
            staging_receipt=self.staging,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
        clients.sagemaker.describe_training_job.assert_called_once()

    def test_status_input_drift_returns_no_receipt_and_makes_no_describe(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            preflight, submission = self._submit_in_progress(clients)
        clients.sagemaker.reset_mock()
        with patch.object(
            training_launch.training_aws,
            "verify_remote_training_staging",
            side_effect=RuntimeError("remote input drift"),
        ):
            with self.assertRaisesRegex(RuntimeError, "remote input drift"):
                training_launch.describe_training_job_status(
                    clients,
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                )
        clients.sagemaker.describe_training_job.assert_not_called()
        clients.sagemaker.list_tags.assert_not_called()

    def test_terminal_states_and_nonterminal_rejection(self) -> None:
        with self._remote_dependencies() as staging_verify:
            for status_name, succeeded in (
                ("Completed", True),
                ("Failed", False),
                ("Stopped", False),
            ):
                with self.subTest(status=status_name):
                    clients = self._clients()
                    preflight, submission = self._submit_in_progress(clients)
                    clients.sagemaker.reset_mock()
                    clients.sagemaker.describe_training_job.side_effect = None
                    clients.sagemaker.describe_training_job.return_value = (
                        self._describe_response(preflight, status=status_name)
                    )
                    clients.sagemaker.list_tags.side_effect = self._tag_pages(
                        preflight
                    )
                    terminal = training_launch.verify_terminal_training_job(
                        clients,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                        preflight_receipt=preflight,
                        submission_receipt=submission,
                    )
                    self.assertIs(terminal["succeeded"], succeeded)
                    self.assertEqual(terminal["terminal_status"], status_name)
                    training_launch.validate_training_terminal_receipt(
                        terminal,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                        preflight_receipt=preflight,
                        submission_receipt=submission,
                    )
            for status_name in ("InProgress", "Stopping"):
                with self.subTest(status=status_name):
                    clients = self._clients()
                    preflight, submission = self._submit_in_progress(clients)
                    clients.sagemaker.reset_mock()
                    clients.sagemaker.describe_training_job.side_effect = None
                    clients.sagemaker.describe_training_job.return_value = (
                        self._describe_response(preflight, status=status_name)
                    )
                    clients.sagemaker.list_tags.side_effect = self._tag_pages(
                        preflight
                    )
                    with self.assertRaisesRegex(RuntimeError, "not terminal"):
                        training_launch.verify_terminal_training_job(
                            clients,
                            training_plan=self.plan,
                            staging_receipt=self.staging,
                            preflight_receipt=preflight,
                            submission_receipt=submission,
                        )
            self.assertGreaterEqual(staging_verify.call_count, 5)

    def test_failed_snapshot_requires_reason_and_completed_requires_model(self) -> None:
        clients = self._clients()
        with self._remote_dependencies():
            preflight, submission = self._submit_in_progress(clients)
            for status_name, field in (
                ("Failed", "FailureReason"),
                ("Completed", "ModelArtifacts"),
            ):
                with self.subTest(status=status_name):
                    clients.sagemaker.reset_mock()
                    described = self._describe_response(
                        preflight, status=status_name
                    )
                    if field == "FailureReason":
                        described.pop(field)
                    else:
                        described[field] = {"S3ModelArtifacts": ""}
                    clients.sagemaker.describe_training_job.side_effect = None
                    clients.sagemaker.describe_training_job.return_value = described
                    clients.sagemaker.list_tags.side_effect = self._tag_pages(
                        preflight
                    )
                    with self.assertRaises(ValueError):
                        training_launch.describe_training_job_status(
                            clients,
                            training_plan=self.plan,
                            staging_receipt=self.staging,
                            preflight_receipt=preflight,
                            submission_receipt=submission,
                        )


if __name__ == "__main__":
    unittest.main()
