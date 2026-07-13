"""Strict one-shot coordination for retrieval-CV SageMaker training jobs.

The coordinator is intentionally host-only.  It accepts only a validated,
ready training plan and its exact staging receipt, performs a complete remote
preflight, and submits at most one ``CreateTrainingJob`` request.  It has no
retry, reconciliation, waiter, polling, or resource-selection fallback.
"""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import aws
from . import config as strict_config
from . import manifest
from . import training_aws


TRAINING_PREFLIGHT_PROTOCOL = "retrieval_cv_training_preflight_v1"
TRAINING_SUBMISSION_PROTOCOL = "retrieval_cv_training_submission_v1"
TRAINING_STATUS_PROTOCOL = "retrieval_cv_training_status_v1"
TRAINING_TERMINAL_PROTOCOL = "retrieval_cv_training_terminal_v1"
TRAINING_QUOTA_CODE = "L-C6383286"
TRAINING_QUOTA_NAME = "ml.g5.12xlarge for training job usage"
TRAINING_INSTANCE_TYPE = "ml.g5.12xlarge"
EC2_TRAINING_INSTANCE_TYPE = "g5.12xlarge"
TRAINING_MAX_PENDING_SECONDS = 7_200

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"\.[0-9]{6}Z\Z"
)
_TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}
_NONTERMINAL_STATUSES = {"InProgress", "Stopping"}
_REQUEST_ECHO_FIELDS = (
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


def _require_plain_json(value: object, *, name: str) -> None:
    """Reject equality tricks and values that cannot be canonical evidence."""

    def visit(current: object, path: str) -> None:
        if type(current) is dict:
            for key, nested in current.items():
                if type(key) is not str:
                    raise TypeError(f"{path} contains a non-string object key")
                visit(nested, f"{path}.{key}")
            return
        if type(current) is list:
            for index, nested in enumerate(current):
                visit(nested, f"{path}[{index}]")
            return
        if current is None or type(current) in {str, bool, int}:
            return
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError(f"{path} contains a non-finite float")
            return
        raise TypeError(f"{path} contains a non-JSON type: {type(current).__name__}")

    visit(value, name)


def _exact_object(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one non-empty exact string")
    return value


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be one non-negative exact integer")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _seal_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    _require_plain_json(payload, name="receipt payload")
    if "receipt_sha256" in payload:
        raise ValueError("Receipt payload already contains receipt_sha256")
    receipt = copy.deepcopy(payload)
    receipt["receipt_sha256"] = _document_sha256(payload)
    return receipt


def _validate_self_hash(receipt: Mapping[str, Any], *, name: str) -> None:
    actual = _exact_sha256(receipt["receipt_sha256"], name=f"{name}.receipt_sha256")
    payload = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    if actual != _document_sha256(payload):
        raise ValueError(f"{name} self-hash changed")


def _validated_context(
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(training_plan) is not dict:
        raise TypeError("training_plan must be one exact object")
    if type(staging_receipt) is not dict:
        raise TypeError("staging_receipt must be one exact object")
    _require_plain_json(training_plan, name="training_plan")
    _require_plain_json(staging_receipt, name="staging_receipt")
    plan = manifest.validate_dry_manifest(copy.deepcopy(training_plan))
    execution = plan["execution"]
    if (
        type(execution["blockers"]) is not list
        or execution["blockers"]
        or type(execution["status"]) is not str
        or execution["status"] != "ready"
        or execution["submittable"] is not True
    ):
        raise ValueError("Training plan is not exactly ready and submittable")
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(staging_receipt), training_plan=plan
    )
    return plan, staged


def _find_run(plan: Mapping[str, Any], run_id: object) -> dict[str, Any]:
    selected = _exact_string(run_id, name="run_id")
    matches = [
        run
        for run in (*plan["controlled_runs"], *plan["auxiliary_runs"])
        if run["run_id"] == selected
    ]
    if len(matches) != 1:
        raise ValueError("run_id must select exactly one planned training run")
    run = matches[0]
    if run["kind"] not in {
        manifest.CONTROLLED_KIND,
        manifest.LEGACY_KIND,
        manifest.SMOKE_KIND,
    }:
        raise ValueError("Selected training run kind is unsupported")
    return run


def _build_request_receipt(
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    run = _find_run(plan, run_id)
    common = {
        "training_plan": plan,
        "run_id": run_id,
        "staging_receipt": staged,
    }
    if run["kind"] == manifest.CONTROLLED_KIND:
        receipt = training_aws.build_controlled_training_request_receipt(**common)
        return training_aws.validate_controlled_training_request_receipt(
            receipt, training_plan=plan, staging_receipt=staged
        )
    if run["kind"] == manifest.LEGACY_KIND:
        receipt = training_aws.build_corrected_legacy_training_request_receipt(
            **common
        )
        return training_aws.validate_corrected_legacy_training_request_receipt(
            receipt, training_plan=plan, staging_receipt=staged
        )
    receipt = training_aws.build_determinism_smoke_training_request_receipt(
        **common
    )
    return training_aws.validate_determinism_smoke_training_request_receipt(
        receipt, training_plan=plan, staging_receipt=staged
    )


def _validate_request_receipt(
    value: object,
    *,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("training request receipt must be one exact object")
    _require_plain_json(value, name="training request receipt")
    run_id = value.get("run_id")
    expected = _build_request_receipt(plan, staged, _exact_string(run_id, name="run_id"))
    if aws.canonical_json_bytes(value) != aws.canonical_json_bytes(expected):
        raise ValueError("Training request receipt differs from a fresh rendering")
    return copy.deepcopy(expected)


def _validate_launch_request(
    request_receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    request = request_receipt["request"]
    _require_plain_json(request, name="training request")
    if type(request) is not dict:
        raise TypeError("Training request must be one exact object")
    if "RetryStrategy" in request:
        raise ValueError("Training request must omit RetryStrategy")
    stopping = _exact_object(
        request.get("StoppingCondition"),
        {"MaxPendingTimeInSeconds", "MaxRuntimeInSeconds"},
        name="training request StoppingCondition",
    )
    if (
        type(stopping["MaxPendingTimeInSeconds"]) is not int
        or stopping["MaxPendingTimeInSeconds"] != TRAINING_MAX_PENDING_SECONDS
        or type(stopping["MaxRuntimeInSeconds"]) is not int
        or stopping["MaxRuntimeInSeconds"]
        != plan["infrastructure"]["training_max_runtime_seconds"]
    ):
        raise ValueError("Training request stopping condition changed")
    resource = request.get("ResourceConfig")
    if (
        type(resource) is not dict
        or resource.get("InstanceType") != TRAINING_INSTANCE_TYPE
        or type(resource.get("InstanceCount")) is not int
        or resource["InstanceCount"] != 1
    ):
        raise ValueError("Training request resource contract changed")
    return copy.deepcopy(request)


def _expected_job_arn(plan: Mapping[str, Any], job_name: str) -> str:
    infrastructure = plan["infrastructure"]
    return (
        f"arn:aws:sagemaker:{infrastructure['region']}:"
        f"{infrastructure['account_id']}:training-job/{job_name}"
    )


def _validate_caller_arn(value: object, *, account_id: str) -> str:
    arn = _exact_string(value, name="STS caller ARN")
    parts = arn.split(":", 5)
    if (
        len(parts) != 6
        or parts[0] != "arn"
        or parts[1] != "aws"
        or parts[2] not in {"iam", "sts"}
        or parts[3] != ""
        or parts[4] != account_id
        or not parts[5]
    ):
        raise ValueError("STS caller ARN is outside the planned AWS account")
    return arn


def _verify_training_image(
    ecr: object,
    *,
    account_id: str,
) -> str:
    aws.ensure_evaluation_repository(ecr, create_if_absent=False)
    response = ecr.batch_get_image(
        registryId=account_id,
        repositoryName=aws.ECR_REPOSITORY_NAME,
        imageIds=[{"imageDigest": training_aws.TRAINING_IMAGE_DIGEST}],
        acceptedMediaTypes=[aws.ECR_MEDIA_TYPE],
    )
    if type(response) is not dict:
        raise RuntimeError("ECR image lookup returned a non-object")
    failures = response.get("failures")
    images = response.get("images")
    if type(failures) is not list or failures or type(images) is not list or len(images) != 1:
        raise ValueError("Digest-addressed training image is not uniquely readable")
    image = images[0]
    if type(image) is not dict:
        raise RuntimeError("ECR training image record is malformed")
    image_id = image.get("imageId")
    if type(image_id) is not dict or not set(image_id).issubset(
        {"imageDigest", "imageTag"}
    ):
        raise ValueError("ECR training image identifier is malformed")
    if "imageTag" in image_id:
        _exact_string(image_id["imageTag"], name="ECR training image tag")
    if (
        image.get("registryId") != account_id
        or image.get("repositoryName") != aws.ECR_REPOSITORY_NAME
        or image_id.get("imageDigest") != training_aws.TRAINING_IMAGE_DIGEST
        or image.get("imageManifestMediaType") != aws.ECR_MEDIA_TYPE
    ):
        raise ValueError("ECR training image response identity changed")
    raw_manifest = image.get("imageManifest")
    if (
        type(raw_manifest) is not str
        or aws._raw_ecr_manifest_digest(raw_manifest)
        != training_aws.TRAINING_IMAGE_DIGEST
    ):
        raise ValueError("ECR raw training image manifest digest changed")
    return training_aws.TRAINING_IMAGE_DIGEST


def _verify_training_quota(service_quotas: object) -> dict[str, Any]:
    response = service_quotas.get_service_quota(
        ServiceCode="sagemaker", QuotaCode=TRAINING_QUOTA_CODE
    )
    if type(response) is not dict or type(response.get("Quota")) is not dict:
        raise RuntimeError("Service Quotas returned a malformed response")
    quota = response["Quota"]
    value = quota.get("Value")
    if type(value) is int:
        exact_value = value
    elif (
        type(value) is float
        and math.isfinite(value)
        and value.is_integer()
    ):
        exact_value = int(value)
    else:
        raise ValueError("SageMaker training quota value is not an exact integer")
    if (
        quota.get("ServiceCode") != "sagemaker"
        or quota.get("QuotaCode") != TRAINING_QUOTA_CODE
        or quota.get("QuotaName") != TRAINING_QUOTA_NAME
    ):
        raise ValueError("SageMaker training quota identity or value changed")
    if exact_value < 4:
        raise ValueError("SageMaker ml.g5.12xlarge training quota is below four")
    return {
        "code": TRAINING_QUOTA_CODE,
        "name": TRAINING_QUOTA_NAME,
        "value": exact_value,
    }


def _verify_regional_offering(ec2: object, *, region: str) -> dict[str, str]:
    if TRAINING_INSTANCE_TYPE != f"ml.{EC2_TRAINING_INSTANCE_TYPE}":
        raise RuntimeError("SageMaker-to-EC2 training instance mapping changed")
    offerings: list[dict[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        request: dict[str, Any] = {
            "Filters": [
                {
                    "Name": "instance-type",
                    "Values": [EC2_TRAINING_INSTANCE_TYPE],
                }
            ],
            "LocationType": "region",
            "MaxResults": 100,
        }
        if token is not None:
            request["NextToken"] = token
        response = ec2.describe_instance_type_offerings(**request)
        if type(response) is not dict or type(response.get("InstanceTypeOfferings")) is not list:
            raise RuntimeError("EC2 instance-offering lookup returned a malformed response")
        offerings.extend(response["InstanceTypeOfferings"])
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen_tokens
            or next_token == token
        ):
            raise RuntimeError("EC2 instance-offering pagination did not advance")
        seen_tokens.add(next_token)
        token = next_token
    if len(offerings) != 1:
        raise ValueError("EC2 did not return exactly one regional instance offering")
    offering = _exact_object(
        offerings[0],
        {"InstanceType", "LocationType", "Location"},
        name="EC2 instance offering",
    )
    if offering != {
        "InstanceType": EC2_TRAINING_INSTANCE_TYPE,
        "LocationType": "region",
        "Location": region,
    }:
        raise ValueError("Regional ml.g5.12xlarge offering changed")
    return {
        "ec2_instance_type": EC2_TRAINING_INSTANCE_TYPE,
        "location": region,
        "location_type": "region",
        "sagemaker_instance_type": TRAINING_INSTANCE_TYPE,
    }


def _assert_unused_job_name(sagemaker: object, *, job_name: str) -> None:
    from botocore.exceptions import ClientError

    try:
        sagemaker.describe_training_job(TrainingJobName=job_name)
    except ClientError as error:
        error_body = error.response.get("Error")
        if type(error_body) is not dict:
            raise
        code = error_body.get("Code")
        if code == "ResourceNotFound":
            return
        if (
            code == "ValidationException"
            and error_body.get("Message") == "Requested resource not found."
        ):
            return
        raise
    raise FileExistsError(f"Training job name is already used: {job_name}")


def _active_planned_jobs(
    sagemaker: object,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    planned_names = {
        run["job_name"]
        for run in (*plan["controlled_runs"], *plan["auxiliary_runs"])
    }
    if len(planned_names) != 64:
        raise ValueError("Training plan does not contain 64 unique job names")
    observed_names: set[str] = set()
    active_planned_names: set[str] = set()
    for status in ("InProgress", "Stopping"):
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            request: dict[str, Any] = {
                "MaxResults": 100,
                "SortBy": "Name",
                "SortOrder": "Ascending",
                "StatusEquals": status,
            }
            if token is not None:
                request["NextToken"] = token
            response = sagemaker.list_training_jobs(**request)
            if (
                type(response) is not dict
                or type(response.get("TrainingJobSummaries")) is not list
            ):
                raise RuntimeError("ListTrainingJobs returned a malformed response")
            for index, summary in enumerate(response["TrainingJobSummaries"]):
                if type(summary) is not dict:
                    raise RuntimeError(
                        f"ListTrainingJobs summary {index} is not an object"
                    )
                name = _exact_string(
                    summary.get("TrainingJobName"),
                    name=f"ListTrainingJobs[{status}][{index}].TrainingJobName",
                )
                if summary.get("TrainingJobStatus") != status:
                    raise ValueError("ListTrainingJobs returned a different status")
                if name in observed_names:
                    raise ValueError("ListTrainingJobs returned a duplicate job name")
                observed_names.add(name)
                if name in planned_names:
                    active_planned_names.add(name)
            next_token = response.get("NextToken")
            if next_token is None:
                break
            if (
                type(next_token) is not str
                or not next_token
                or next_token in seen_tokens
                or next_token == token
            ):
                raise RuntimeError("ListTrainingJobs pagination did not advance")
            seen_tokens.add(next_token)
            token = next_token
    names = sorted(active_planned_names)
    maximum = plan["infrastructure"]["max_concurrent_training_jobs"]
    if type(maximum) is not int or maximum != 4:
        raise ValueError("Planned maximum training concurrency changed")
    if len(names) >= maximum:
        raise RuntimeError(
            "Planned training concurrency is already at the frozen maximum"
        )
    return {"count": len(names), "job_names": names}


def _output_version_prefix(
    request: Mapping[str, Any],
    *,
    expected_bucket: str,
) -> str:
    output = request["OutputDataConfig"]["S3OutputPath"]
    bucket, key = strict_config._s3_uri_coordinates(output)
    if bucket != expected_bucket or output.endswith("/") or key.endswith("/"):
        raise ValueError("Training output path is not one exact unused run prefix")
    return key + "/"


def _normalize_datetime(value: object, *, name: str) -> str:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise TypeError(f"{name} must be one timezone-aware datetime")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: object, *, name: str, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{name} must be one canonical UTC timestamp")
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    canonical = (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    if canonical != value:
        raise ValueError(f"{name} is not canonical")
    return value


def _optional_remote_datetime(response: Mapping[str, Any], name: str) -> str | None:
    if name not in response or response[name] is None:
        return None
    return _normalize_datetime(response[name], name=name)


def _optional_remote_string(response: Mapping[str, Any], name: str) -> str | None:
    if name not in response or response[name] is None or response[name] == "":
        return None
    return _exact_string(response[name], name=name)


def _optional_remote_int(response: Mapping[str, Any], name: str) -> int | None:
    if name not in response or response[name] is None:
        return None
    return _exact_nonnegative_int(response[name], name=name)


def _list_exact_tags(
    sagemaker: object,
    *,
    job_arn: str,
    expected_tags: object,
) -> list[dict[str, str]]:
    if type(expected_tags) is not list:
        raise TypeError("Training request Tags must be one exact list")
    normalized_expected: list[dict[str, str]] = []
    expected_keys: set[str] = set()
    for index, raw_tag in enumerate(expected_tags):
        tag = _exact_object(raw_tag, {"Key", "Value"}, name=f"request Tags[{index}]")
        key = _exact_string(tag["Key"], name=f"request Tags[{index}].Key")
        value = _exact_string(tag["Value"], name=f"request Tags[{index}].Value")
        if key in expected_keys:
            raise ValueError("Training request contains duplicate tag keys")
        expected_keys.add(key)
        normalized_expected.append({"Key": key, "Value": value})
    normalized_expected.sort(key=lambda tag: (tag["Key"], tag["Value"]))

    observed: list[dict[str, str]] = []
    observed_keys: set[str] = set()
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        request: dict[str, Any] = {"MaxResults": 100, "ResourceArn": job_arn}
        if token is not None:
            request["NextToken"] = token
        response = sagemaker.list_tags(**request)
        if type(response) is not dict or type(response.get("Tags")) is not list:
            raise RuntimeError("SageMaker ListTags returned a malformed response")
        for index, raw_tag in enumerate(response["Tags"]):
            tag = _exact_object(
                raw_tag, {"Key", "Value"}, name=f"observed Tags[{index}]"
            )
            key = _exact_string(tag["Key"], name=f"observed Tags[{index}].Key")
            value = _exact_string(tag["Value"], name=f"observed Tags[{index}].Value")
            if key in observed_keys:
                raise ValueError("SageMaker returned a duplicate tag key")
            observed_keys.add(key)
            observed.append({"Key": key, "Value": value})
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if (
            type(next_token) is not str
            or not next_token
            or next_token in seen_tokens
            or next_token == token
        ):
            raise RuntimeError("SageMaker ListTags pagination did not advance")
        seen_tokens.add(next_token)
        token = next_token
    observed.sort(key=lambda tag: (tag["Key"], tag["Value"]))
    if observed != normalized_expected:
        raise ValueError("SageMaker training-job tags differ from the request")
    return observed


def _validate_snapshot(
    value: object,
    *,
    request_receipt: Mapping[str, Any],
    job_arn: str,
) -> dict[str, Any]:
    snapshot = _exact_object(
        value,
        {
            "billable_time_seconds",
            "creation_time",
            "failure_reason",
            "job_arn",
            "job_name",
            "last_modified_time",
            "model_artifact_s3_uri",
            "request_sha256",
            "secondary_status",
            "tags",
            "training_end_time",
            "training_job_status",
            "training_start_time",
            "training_time_seconds",
        },
        name="training status snapshot",
    )
    _require_plain_json(snapshot, name="training status snapshot")
    request = request_receipt["request"]
    if (
        snapshot["job_name"] != request["TrainingJobName"]
        or snapshot["job_arn"] != job_arn
        or snapshot["request_sha256"] != request_receipt["request_sha256"]
    ):
        raise ValueError("Training status snapshot identity changed")
    _exact_sha256(snapshot["request_sha256"], name="snapshot.request_sha256")
    status = _exact_string(
        snapshot["training_job_status"], name="snapshot.training_job_status"
    )
    if status not in _TERMINAL_STATUSES | _NONTERMINAL_STATUSES:
        raise ValueError("Training status snapshot has an unsupported primary status")
    _exact_string(snapshot["secondary_status"], name="snapshot.secondary_status")
    failure = snapshot["failure_reason"]
    if failure is not None:
        _exact_string(failure, name="snapshot.failure_reason")
    model_uri = snapshot["model_artifact_s3_uri"]
    if model_uri is not None:
        _exact_string(model_uri, name="snapshot.model_artifact_s3_uri")
        strict_config._s3_uri_coordinates(model_uri)
        expected_model_uri = (
            f"{request['OutputDataConfig']['S3OutputPath']}/"
            f"{request['TrainingJobName']}/output/model.tar.gz"
        )
        if model_uri != expected_model_uri:
            raise ValueError("Training model-artifact URI changed")
    creation = _validate_timestamp(
        snapshot["creation_time"], name="snapshot.creation_time", optional=False
    )
    start = _validate_timestamp(
        snapshot["training_start_time"],
        name="snapshot.training_start_time",
        optional=True,
    )
    end = _validate_timestamp(
        snapshot["training_end_time"],
        name="snapshot.training_end_time",
        optional=True,
    )
    modified = _validate_timestamp(
        snapshot["last_modified_time"],
        name="snapshot.last_modified_time",
        optional=True,
    )
    parsed_creation = datetime.fromisoformat(creation.removesuffix("Z") + "+00:00")
    parsed_start = (
        None if start is None else datetime.fromisoformat(start.removesuffix("Z") + "+00:00")
    )
    parsed_end = (
        None if end is None else datetime.fromisoformat(end.removesuffix("Z") + "+00:00")
    )
    parsed_modified = (
        None
        if modified is None
        else datetime.fromisoformat(modified.removesuffix("Z") + "+00:00")
    )
    if parsed_start is not None and parsed_start < parsed_creation:
        raise ValueError("Training start time precedes creation")
    if parsed_end is not None and (parsed_start is None or parsed_end < parsed_start):
        raise ValueError("Training end time precedes or lacks a start time")
    if parsed_modified is not None and parsed_modified < parsed_creation:
        raise ValueError("Training last-modified time precedes creation")
    training_seconds = snapshot["training_time_seconds"]
    billable_seconds = snapshot["billable_time_seconds"]
    if training_seconds is not None:
        _exact_nonnegative_int(training_seconds, name="snapshot.training_time_seconds")
    if billable_seconds is not None:
        _exact_nonnegative_int(billable_seconds, name="snapshot.billable_time_seconds")
    expected_tags = sorted(
        copy.deepcopy(request["Tags"]), key=lambda tag: (tag["Key"], tag["Value"])
    )
    if snapshot["tags"] != expected_tags:
        raise ValueError("Training status snapshot tags changed")
    if status == "Completed":
        if (
            failure is not None
            or model_uri is None
            or start is None
            or end is None
            or training_seconds is None
            or billable_seconds is None
        ):
            raise ValueError("Completed training snapshot lacks exact success evidence")
    elif status == "Failed" and failure is None:
        raise ValueError("Failed training snapshot lacks a failure reason")
    return copy.deepcopy(snapshot)


def _snapshot_from_remote(
    response: object,
    *,
    tags: list[dict[str, str]],
    request_receipt: Mapping[str, Any],
    job_arn: str,
) -> dict[str, Any]:
    if type(response) is not dict:
        raise RuntimeError("DescribeTrainingJob returned a non-object")
    request = request_receipt["request"]
    if (
        response.get("TrainingJobName") != request["TrainingJobName"]
        or response.get("TrainingJobArn") != job_arn
    ):
        raise ValueError("DescribeTrainingJob returned a different job identity")
    if "RetryStrategy" in response:
        raise ValueError("DescribeTrainingJob unexpectedly contains RetryStrategy")
    for field in _REQUEST_ECHO_FIELDS:
        if field not in response:
            raise ValueError(f"DescribeTrainingJob omitted request field {field}")
        _require_plain_json(response[field], name=f"DescribeTrainingJob.{field}")
        if aws.canonical_json_bytes(response[field]) != aws.canonical_json_bytes(
            request[field]
        ):
            raise ValueError(f"DescribeTrainingJob changed request field {field}")
    artifacts = _exact_object(
        response.get("ModelArtifacts"),
        {"S3ModelArtifacts"},
        name="DescribeTrainingJob.ModelArtifacts",
    )
    snapshot = {
        "billable_time_seconds": _optional_remote_int(
            response, "BillableTimeInSeconds"
        ),
        "creation_time": _normalize_datetime(
            response.get("CreationTime"), name="CreationTime"
        ),
        "failure_reason": _optional_remote_string(response, "FailureReason"),
        "job_arn": job_arn,
        "job_name": request["TrainingJobName"],
        "last_modified_time": _optional_remote_datetime(
            response, "LastModifiedTime"
        ),
        "model_artifact_s3_uri": (
            None
            if artifacts["S3ModelArtifacts"] in {None, ""}
            else _exact_string(
                artifacts["S3ModelArtifacts"], name="ModelArtifacts.S3ModelArtifacts"
            )
        ),
        "request_sha256": request_receipt["request_sha256"],
        "secondary_status": _exact_string(
            response.get("SecondaryStatus"), name="SecondaryStatus"
        ),
        "tags": copy.deepcopy(tags),
        "training_end_time": _optional_remote_datetime(response, "TrainingEndTime"),
        "training_job_status": _exact_string(
            response.get("TrainingJobStatus"), name="TrainingJobStatus"
        ),
        "training_start_time": _optional_remote_datetime(
            response, "TrainingStartTime"
        ),
        "training_time_seconds": _optional_remote_int(
            response, "TrainingTimeInSeconds"
        ),
    }
    return _validate_snapshot(
        snapshot, request_receipt=request_receipt, job_arn=job_arn
    )


def _describe_verified_snapshot(
    clients: aws.AwsClients,
    *,
    request_receipt: Mapping[str, Any],
    job_arn: str,
) -> dict[str, Any]:
    request = request_receipt["request"]
    response = clients.sagemaker.describe_training_job(
        TrainingJobName=request["TrainingJobName"]
    )
    tags = _list_exact_tags(
        clients.sagemaker,
        job_arn=job_arn,
        expected_tags=request["Tags"],
    )
    return _snapshot_from_remote(
        response,
        tags=tags,
        request_receipt=request_receipt,
        job_arn=job_arn,
    )


def preflight_training_job(
    clients: aws.AwsClients,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Perform the entire remote gate for one exact, still-absent job."""

    plan, staged = _validated_context(training_plan, staging_receipt)
    run = _find_run(plan, run_id)
    request_receipt = _build_request_receipt(plan, staged, run_id)
    request = _validate_launch_request(request_receipt, plan=plan)
    verified_staging = training_aws.verify_remote_training_staging(
        clients.s3,
        training_plan=plan,
        staging_receipt=staged,
        deep_read=True,
    )
    if aws.canonical_json_bytes(verified_staging) != aws.canonical_json_bytes(staged):
        raise RuntimeError("Deep staging verification returned different evidence")
    sdk_versions = aws.validate_aws_sdk_versions()
    if sdk_versions != aws.EXPECTED_AWS_SDK_VERSIONS:
        raise RuntimeError("Pinned AWS SDK inventory changed")
    caller = clients.sts.get_caller_identity()
    if type(caller) is not dict:
        raise RuntimeError("STS caller identity response is malformed")
    account_id = plan["infrastructure"]["account_id"]
    if caller.get("Account") != account_id:
        raise ValueError("Active AWS account differs from the training plan")
    caller_arn = _validate_caller_arn(caller.get("Arn"), account_id=account_id)
    aws._assert_role_trust(clients.iam, plan["infrastructure"]["role_arn"])
    aws.validate_artifact_bucket(
        clients.s3,
        bucket=plan["infrastructure"]["artifact_bucket"],
        region=plan["infrastructure"]["region"],
    )
    image_manifest_digest = _verify_training_image(
        clients.ecr, account_id=account_id
    )
    quota = _verify_training_quota(clients.service_quotas)
    offering = _verify_regional_offering(
        clients.ec2, region=plan["infrastructure"]["region"]
    )
    active_planned_jobs = _active_planned_jobs(clients.sagemaker, plan=plan)
    _assert_unused_job_name(clients.sagemaker, job_name=run["job_name"])
    output_prefix = _output_version_prefix(
        request,
        expected_bucket=plan["infrastructure"]["artifact_bucket"],
    )
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=plan["infrastructure"]["artifact_bucket"],
        prefix=output_prefix,
        expected_bucket_owner=account_id,
    )
    return _seal_receipt(
        {
            "account_id": account_id,
            "active_planned_jobs": active_planned_jobs,
            "caller_arn": caller_arn,
            "image_manifest_digest": image_manifest_digest,
            "instance_offering": offering,
            "job_name": run["job_name"],
            "output_version_prefix": output_prefix,
            "plan_sha256": _document_sha256(plan),
            "protocol": TRAINING_PREFLIGHT_PROTOCOL,
            "quota": quota,
            "region": plan["infrastructure"]["region"],
            "request_receipt": request_receipt,
            "request_receipt_sha256": _document_sha256(request_receipt),
            "run_id": run_id,
            "schema_version": 1,
            "sdk_versions": sdk_versions,
            "staging_receipt_sha256": _document_sha256(staged),
        }
    )


def validate_training_preflight_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan, staged = _validated_context(training_plan, staging_receipt)
    if type(value) is not dict:
        raise TypeError("training preflight receipt must be one exact object")
    _require_plain_json(value, name="training preflight receipt")
    receipt = _exact_object(
        value,
        {
            "account_id",
            "active_planned_jobs",
            "caller_arn",
            "image_manifest_digest",
            "instance_offering",
            "job_name",
            "output_version_prefix",
            "plan_sha256",
            "protocol",
            "quota",
            "receipt_sha256",
            "region",
            "request_receipt",
            "request_receipt_sha256",
            "run_id",
            "schema_version",
            "sdk_versions",
            "staging_receipt_sha256",
        },
        name="training preflight receipt",
    )
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("Training preflight schema version changed")
    if receipt["protocol"] != TRAINING_PREFLIGHT_PROTOCOL:
        raise ValueError("Training preflight protocol changed")
    run = _find_run(plan, receipt["run_id"])
    request_receipt = _validate_request_receipt(
        receipt["request_receipt"], plan=plan, staged=staged
    )
    request = _validate_launch_request(request_receipt, plan=plan)
    expected_output_prefix = _output_version_prefix(
        request, expected_bucket=plan["infrastructure"]["artifact_bucket"]
    )
    if (
        receipt["plan_sha256"] != _document_sha256(plan)
        or receipt["staging_receipt_sha256"] != _document_sha256(staged)
        or receipt["request_receipt_sha256"]
        != _document_sha256(request_receipt)
        or receipt["job_name"] != run["job_name"]
        or receipt["account_id"] != plan["infrastructure"]["account_id"]
        or receipt["region"] != plan["infrastructure"]["region"]
        or receipt["image_manifest_digest"] != training_aws.TRAINING_IMAGE_DIGEST
        or receipt["output_version_prefix"] != expected_output_prefix
        or receipt["sdk_versions"] != aws.EXPECTED_AWS_SDK_VERSIONS
    ):
        raise ValueError("Training preflight evidence identity changed")
    for field in (
        "plan_sha256",
        "staging_receipt_sha256",
        "request_receipt_sha256",
    ):
        _exact_sha256(receipt[field], name=f"preflight.{field}")
    _validate_caller_arn(
        receipt["caller_arn"],
        account_id=plan["infrastructure"]["account_id"],
    )
    offering = _exact_object(
        receipt["instance_offering"],
        {
            "ec2_instance_type",
            "location",
            "location_type",
            "sagemaker_instance_type",
        },
        name="preflight.instance_offering",
    )
    if offering != {
        "ec2_instance_type": EC2_TRAINING_INSTANCE_TYPE,
        "location": plan["infrastructure"]["region"],
        "location_type": "region",
        "sagemaker_instance_type": TRAINING_INSTANCE_TYPE,
    }:
        raise ValueError("Training preflight instance offering changed")
    quota = _exact_object(
        receipt["quota"], {"code", "name", "value"}, name="preflight.quota"
    )
    if (
        quota["code"] != TRAINING_QUOTA_CODE
        or quota["name"] != TRAINING_QUOTA_NAME
        or type(quota["value"]) is not int
        or quota["value"] < 4
    ):
        raise ValueError("Training preflight quota evidence changed")
    active = _exact_object(
        receipt["active_planned_jobs"],
        {"count", "job_names"},
        name="preflight.active_planned_jobs",
    )
    active_names = active["job_names"]
    planned_names = {
        run["job_name"]
        for run in (*plan["controlled_runs"], *plan["auxiliary_runs"])
    }
    if (
        type(active["count"]) is not int
        or type(active_names) is not list
        or any(type(name) is not str for name in active_names)
        or active_names != sorted(set(active_names))
        or not set(active_names).issubset(planned_names)
        or active["count"] != len(active_names)
        or active["count"] >= 4
    ):
        raise ValueError("Training preflight active-planned-job evidence changed")
    _validate_self_hash(receipt, name="training preflight receipt")
    return copy.deepcopy(receipt)


def submit_training_job_once(
    clients: aws.AwsClients,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-run preflight and issue exactly one un-reconciled create request."""

    saved = validate_training_preflight_receipt(
        preflight_receipt,
        training_plan=training_plan,
        staging_receipt=staging_receipt,
    )
    fresh = preflight_training_job(
        clients,
        training_plan=training_plan,
        staging_receipt=staging_receipt,
        run_id=saved["run_id"],
    )
    if aws.canonical_json_bytes(fresh) != aws.canonical_json_bytes(saved):
        raise RuntimeError("Fresh preflight differs from the approved saved receipt")
    request_receipt = fresh["request_receipt"]
    request = _validate_launch_request(
        request_receipt,
        plan=manifest.validate_dry_manifest(copy.deepcopy(training_plan)),
    )
    create_response = clients.sagemaker.create_training_job(**copy.deepcopy(request))
    if type(create_response) is not dict:
        raise RuntimeError("CreateTrainingJob returned a non-object")
    plan = manifest.validate_dry_manifest(copy.deepcopy(training_plan))
    job_arn = _expected_job_arn(plan, request["TrainingJobName"])
    if create_response.get("TrainingJobArn") != job_arn:
        raise ValueError("CreateTrainingJob returned a different training-job ARN")
    snapshot = _describe_verified_snapshot(
        clients,
        request_receipt=request_receipt,
        job_arn=job_arn,
    )
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(staging_receipt), training_plan=plan
    )
    return _seal_receipt(
        {
            "job_arn": job_arn,
            "job_name": request["TrainingJobName"],
            "plan_sha256": _document_sha256(plan),
            "preflight_receipt_sha256": _document_sha256(saved),
            "protocol": TRAINING_SUBMISSION_PROTOCOL,
            "request_receipt_sha256": _document_sha256(request_receipt),
            "run_id": saved["run_id"],
            "schema_version": 1,
            "snapshot": snapshot,
            "staging_receipt_sha256": _document_sha256(staged),
        }
    )


def validate_training_submission_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan, staged = _validated_context(training_plan, staging_receipt)
    preflight = validate_training_preflight_receipt(
        preflight_receipt, training_plan=plan, staging_receipt=staged
    )
    if type(value) is not dict:
        raise TypeError("training submission receipt must be one exact object")
    _require_plain_json(value, name="training submission receipt")
    receipt = _exact_object(
        value,
        {
            "job_arn",
            "job_name",
            "plan_sha256",
            "preflight_receipt_sha256",
            "protocol",
            "receipt_sha256",
            "request_receipt_sha256",
            "run_id",
            "schema_version",
            "snapshot",
            "staging_receipt_sha256",
        },
        name="training submission receipt",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != TRAINING_SUBMISSION_PROTOCOL
    ):
        raise ValueError("Training submission receipt identity changed")
    job_arn = _expected_job_arn(plan, preflight["job_name"])
    if (
        receipt["plan_sha256"] != _document_sha256(plan)
        or receipt["staging_receipt_sha256"] != _document_sha256(staged)
        or receipt["preflight_receipt_sha256"] != _document_sha256(preflight)
        or receipt["request_receipt_sha256"]
        != _document_sha256(preflight["request_receipt"])
        or receipt["run_id"] != preflight["run_id"]
        or receipt["job_name"] != preflight["job_name"]
        or receipt["job_arn"] != job_arn
    ):
        raise ValueError("Training submission evidence binding changed")
    for field in (
        "plan_sha256",
        "staging_receipt_sha256",
        "preflight_receipt_sha256",
        "request_receipt_sha256",
    ):
        _exact_sha256(receipt[field], name=f"submission.{field}")
    _validate_snapshot(
        receipt["snapshot"],
        request_receipt=preflight["request_receipt"],
        job_arn=job_arn,
    )
    _validate_self_hash(receipt, name="training submission receipt")
    return copy.deepcopy(receipt)


def describe_training_job_status(
    clients: aws.AwsClients,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Read one strict status snapshot; do not wait or poll."""

    plan, staged = _validated_context(training_plan, staging_receipt)
    preflight = validate_training_preflight_receipt(
        preflight_receipt, training_plan=plan, staging_receipt=staged
    )
    submission = validate_training_submission_receipt(
        submission_receipt,
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight,
    )
    verified_staging = training_aws.verify_remote_training_staging(
        clients.s3,
        training_plan=plan,
        staging_receipt=staged,
        deep_read=True,
    )
    if aws.canonical_json_bytes(verified_staging) != aws.canonical_json_bytes(staged):
        raise RuntimeError("Status staging verification returned different evidence")
    snapshot = _describe_verified_snapshot(
        clients,
        request_receipt=preflight["request_receipt"],
        job_arn=submission["job_arn"],
    )
    return _seal_receipt(
        {
            "job_arn": submission["job_arn"],
            "job_name": submission["job_name"],
            "plan_sha256": _document_sha256(plan),
            "preflight_receipt_sha256": _document_sha256(preflight),
            "protocol": TRAINING_STATUS_PROTOCOL,
            "request_receipt_sha256": _document_sha256(
                preflight["request_receipt"]
            ),
            "run_id": submission["run_id"],
            "schema_version": 1,
            "snapshot": snapshot,
            "staging_receipt_sha256": _document_sha256(staged),
            "submission_receipt_sha256": _document_sha256(submission),
        }
    )


def validate_training_status_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan, staged = _validated_context(training_plan, staging_receipt)
    preflight = validate_training_preflight_receipt(
        preflight_receipt, training_plan=plan, staging_receipt=staged
    )
    submission = validate_training_submission_receipt(
        submission_receipt,
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight,
    )
    if type(value) is not dict:
        raise TypeError("training status receipt must be one exact object")
    _require_plain_json(value, name="training status receipt")
    receipt = _exact_object(
        value,
        {
            "job_arn",
            "job_name",
            "plan_sha256",
            "preflight_receipt_sha256",
            "protocol",
            "receipt_sha256",
            "request_receipt_sha256",
            "run_id",
            "schema_version",
            "snapshot",
            "staging_receipt_sha256",
            "submission_receipt_sha256",
        },
        name="training status receipt",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != TRAINING_STATUS_PROTOCOL
        or receipt["job_arn"] != submission["job_arn"]
        or receipt["job_name"] != submission["job_name"]
        or receipt["run_id"] != submission["run_id"]
        or receipt["plan_sha256"] != _document_sha256(plan)
        or receipt["staging_receipt_sha256"] != _document_sha256(staged)
        or receipt["preflight_receipt_sha256"] != _document_sha256(preflight)
        or receipt["submission_receipt_sha256"] != _document_sha256(submission)
        or receipt["request_receipt_sha256"]
        != _document_sha256(preflight["request_receipt"])
    ):
        raise ValueError("Training status evidence binding changed")
    for field in (
        "plan_sha256",
        "staging_receipt_sha256",
        "preflight_receipt_sha256",
        "submission_receipt_sha256",
        "request_receipt_sha256",
    ):
        _exact_sha256(receipt[field], name=f"status.{field}")
    _validate_snapshot(
        receipt["snapshot"],
        request_receipt=preflight["request_receipt"],
        job_arn=submission["job_arn"],
    )
    _validate_self_hash(receipt, name="training status receipt")
    return copy.deepcopy(receipt)


def verify_terminal_training_job(
    clients: aws.AwsClients,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reverify immutable inputs and capture exactly one terminal observation."""

    plan, staged = _validated_context(training_plan, staging_receipt)
    status_receipt = describe_training_job_status(
        clients,
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
    )
    status = status_receipt["snapshot"]["training_job_status"]
    if status in _NONTERMINAL_STATUSES:
        raise RuntimeError(f"Training job is not terminal: {status}")
    if status not in _TERMINAL_STATUSES:
        raise ValueError(f"Unsupported terminal training status: {status}")
    snapshot = status_receipt["snapshot"]
    return _seal_receipt(
        {
            "billable_time_seconds": snapshot["billable_time_seconds"],
            "failure_reason": snapshot["failure_reason"],
            "job_arn": status_receipt["job_arn"],
            "job_name": status_receipt["job_name"],
            "model_artifact_s3_uri": snapshot["model_artifact_s3_uri"],
            "plan_sha256": status_receipt["plan_sha256"],
            "preflight_receipt_sha256": status_receipt[
                "preflight_receipt_sha256"
            ],
            "protocol": TRAINING_TERMINAL_PROTOCOL,
            "run_id": status_receipt["run_id"],
            "schema_version": 1,
            "staging_receipt_sha256": status_receipt[
                "staging_receipt_sha256"
            ],
            "status_receipt": status_receipt,
            "status_receipt_sha256": _document_sha256(status_receipt),
            "succeeded": status == "Completed",
            "submission_receipt_sha256": status_receipt[
                "submission_receipt_sha256"
            ],
            "terminal_status": status,
            "training_time_seconds": snapshot["training_time_seconds"],
        }
    )


def validate_training_terminal_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan, staged = _validated_context(training_plan, staging_receipt)
    preflight = validate_training_preflight_receipt(
        preflight_receipt, training_plan=plan, staging_receipt=staged
    )
    submission = validate_training_submission_receipt(
        submission_receipt,
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight,
    )
    if type(value) is not dict:
        raise TypeError("training terminal receipt must be one exact object")
    _require_plain_json(value, name="training terminal receipt")
    receipt = _exact_object(
        value,
        {
            "billable_time_seconds",
            "failure_reason",
            "job_arn",
            "job_name",
            "model_artifact_s3_uri",
            "plan_sha256",
            "preflight_receipt_sha256",
            "protocol",
            "receipt_sha256",
            "run_id",
            "schema_version",
            "staging_receipt_sha256",
            "status_receipt",
            "status_receipt_sha256",
            "succeeded",
            "submission_receipt_sha256",
            "terminal_status",
            "training_time_seconds",
        },
        name="training terminal receipt",
    )
    status_receipt = validate_training_status_receipt(
        receipt["status_receipt"],
        training_plan=plan,
        staging_receipt=staged,
        preflight_receipt=preflight,
        submission_receipt=submission,
    )
    snapshot = status_receipt["snapshot"]
    terminal_status = snapshot["training_job_status"]
    if terminal_status not in _TERMINAL_STATUSES:
        raise ValueError("Terminal receipt contains a nonterminal status")
    expected = {
        "billable_time_seconds": snapshot["billable_time_seconds"],
        "failure_reason": snapshot["failure_reason"],
        "job_arn": submission["job_arn"],
        "job_name": submission["job_name"],
        "model_artifact_s3_uri": snapshot["model_artifact_s3_uri"],
        "plan_sha256": _document_sha256(plan),
        "preflight_receipt_sha256": _document_sha256(preflight),
        "protocol": TRAINING_TERMINAL_PROTOCOL,
        "run_id": submission["run_id"],
        "schema_version": 1,
        "staging_receipt_sha256": _document_sha256(staged),
        "status_receipt": status_receipt,
        "status_receipt_sha256": _document_sha256(status_receipt),
        "succeeded": terminal_status == "Completed",
        "submission_receipt_sha256": _document_sha256(submission),
        "terminal_status": terminal_status,
        "training_time_seconds": snapshot["training_time_seconds"],
    }
    actual_payload = {
        key: copy.deepcopy(nested)
        for key, nested in receipt.items()
        if key != "receipt_sha256"
    }
    if aws.canonical_json_bytes(actual_payload) != aws.canonical_json_bytes(expected):
        raise ValueError("Training terminal receipt differs from its terminal evidence")
    if type(receipt["succeeded"]) is not bool:
        raise TypeError("terminal.succeeded must be one exact bool")
    _validate_self_hash(receipt, name="training terminal receipt")
    return copy.deepcopy(receipt)


__all__: Sequence[str] = (
    "EC2_TRAINING_INSTANCE_TYPE",
    "TRAINING_INSTANCE_TYPE",
    "TRAINING_MAX_PENDING_SECONDS",
    "TRAINING_PREFLIGHT_PROTOCOL",
    "TRAINING_QUOTA_CODE",
    "TRAINING_QUOTA_NAME",
    "TRAINING_STATUS_PROTOCOL",
    "TRAINING_SUBMISSION_PROTOCOL",
    "TRAINING_TERMINAL_PROTOCOL",
    "describe_training_job_status",
    "preflight_training_job",
    "submit_training_job_once",
    "validate_training_preflight_receipt",
    "validate_training_status_receipt",
    "validate_training_submission_receipt",
    "validate_training_terminal_receipt",
    "verify_terminal_training_job",
)
