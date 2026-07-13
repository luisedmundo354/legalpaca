"""Strict AWS rendering and one-shot execution for the retrieval study.

The functions in this module deliberately use the low-level service clients.
They never retry at the orchestration layer, never choose another resource,
and never turn a runtime smoke into scientific evaluation evidence.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import config as strict_config

RUNTIME_SMOKE_PROTOCOL = "evaluation_image_runtime_smoke_v1"
ECR_REPOSITORY_NAME = "arr-retrieval-eval"
ECR_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
EXPECTED_LOCAL_IMAGE_DIGEST = (
    "sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2"
)
EXPECTED_CONFIG_DIGEST = (
    "sha256:76c29a7f5ca0a1a36d0f8b53fe1e49f40ab199f8ff1bc594ddbb09107c7749e8"
)
EXPECTED_BUILD_IDENTITY = (
    "249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8"
)
EXPECTED_RUNTIME_IDENTITY_SHA256 = (
    "75c1d8fd9a012419772a2b23948694e4def868c0d4b1af0b9a253a7a2c0057a6"
)
EXPECTED_SOURCE_PARENT_COMMIT = "4b4f26852c59f809591edfced61bfc1d13650021"
EXPECTED_SOURCE_PARENT_EPOCH = "1783881756"
EXPECTED_SOURCE_PARENT_RFC3339 = "2026-07-12T18:42:36Z"
EXPECTED_LOCAL_IMAGE_REF = (
    "arr-retrieval-eval:"
    "build-sha256-249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8-build1"
)
EXPECTED_TRAINING_IMAGE_REF = (
    "arr-retrieval-train:step10a-bootstrap-build1"
)
EXPECTED_TRAINING_IMAGE_DIGEST = (
    "sha256:b44c9b182a2490329b25394568299420bcfbe85a8fb17df955378b1f3630d9be"
)
EXPECTED_TRAINING_CONFIG_DIGEST = (
    "sha256:24784672e3d1f8004fe6577069d6f01393239310276a570f5e8d0db1fe13b85f"
)
EXPECTED_TRAINING_CONTRACT_SHA256 = (
    "db4b2b307a56686054c2c04fbcebf5c133077765074ceef61a613c183a4b04ef"
)
EXPECTED_TRAINING_RUNTIME_INVENTORY_SHA256 = (
    "1151907eb4c0c63a6a317ae11b909ceb7bbbe29d4a56c46d8bec91d8424d795c"
)
EXPECTED_AWS_SDK_VERSIONS = {
    "boto3": "1.39.12",
    "botocore": "1.39.12",
    "s3transfer": "0.13.1",
    "sagemaker": "2.248.2",
    "sagemaker-core": "1.0.46",
}
EXPECTED_SDK_TRAINING_IMAGE_TAG = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:"
    "2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04"
)
EXPECTED_SDK_TRAINING_IMAGE_DIGEST_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training@"
    "sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9"
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ACCOUNT = re.compile(r"[0-9]{12}\Z")
_JOB_NAME = re.compile(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}\Z")


def canonical_json_bytes(value: object) -> bytes:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _raw_ecr_manifest_digest(raw_manifest: str) -> str:
    if type(raw_manifest) is not str or not raw_manifest:
        raise ValueError("ECR raw manifest must be one non-empty string")
    return "sha256:" + sha256_bytes(raw_manifest.encode("utf-8"))


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != expected:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one non-empty exact string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def validate_aws_config(value: object) -> dict[str, Any]:
    config = strict_config.validate_aws_local_config(value)
    account = config["account_id"]
    region = config["region"]
    role = config["role_arn"]
    expected_role = f"arn:aws:iam::{account}:role/AmazonSageMakerExecutionRole"
    if role != expected_role:
        raise ValueError(f"role_arn must equal {expected_role}")
    if region != "us-east-1":
        raise ValueError("The frozen study region is us-east-1")
    if config["artifact_bucket"] != "ir-sagemaker":
        raise ValueError("The immutable study bucket must be ir-sagemaker")
    prefix = config["artifact_root_prefix"]
    if (
        prefix.startswith("/")
        or prefix.endswith("/")
        or "//" in prefix
        or not prefix.startswith("arr-retrieval-cv/")
    ):
        raise ValueError(
            "artifact_root_prefix must be a normalized ARR prefix without slashes at ends"
        )
    if config["ecr_repository"] != ECR_REPOSITORY_NAME:
        raise ValueError("The frozen evaluation repository is arr-retrieval-eval")
    if config["processing_instance_type"] != "ml.g5.12xlarge":
        raise ValueError("The production-shaped Processing pilot requires ml.g5.12xlarge")
    if config["training_instance_type"] != "ml.g5.12xlarge":
        raise ValueError("Controlled training requires ml.g5.12xlarge")
    if config["processing_instance_count"] != 1 or config["training_instance_count"] != 1:
        raise ValueError("The frozen study uses one instance per job")
    if (
        config["processing_volume_size_gb"] != 100
        or config["processing_max_runtime_seconds"] != 3_600
    ):
        raise ValueError("The bounded Processing pilot requires 100 GB and 3600 seconds")
    if (
        config["training_volume_size_gb"] != 200
        or config["training_max_runtime_seconds"] != 86_400
    ):
        raise ValueError("The frozen training request requires 200 GB and 86400 seconds")
    if config["max_concurrent_training_jobs"] != 4:
        raise ValueError("The frozen maximum training concurrency is four")
    if config["tags"] != {
        "Experiment": "arr_retrieval_cv_v1",
        "ManagedBy": "arr-retrieval-cv",
        "Purpose": "evaluation-image-runtime-smoke",
    }:
        raise ValueError("The frozen AWS resource tags changed")
    return config


def no_retry_botocore_config() -> object:
    """Construct a Botocore config with one total request attempt."""

    from botocore.config import Config

    return Config(
        retries={"mode": "standard", "total_max_attempts": 1},
        user_agent_extra="arr-retrieval-cv/1",
    )


def validate_aws_sdk_versions() -> dict[str, str]:
    actual: dict[str, str] = {}
    for distribution, expected in EXPECTED_AWS_SDK_VERSIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required AWS SDK distribution is absent: {distribution}") from exc
        actual[distribution] = version
        if version != expected:
            raise RuntimeError(
                f"AWS SDK distribution changed: {distribution}={version}, expected={expected}"
            )
    return actual


def resolve_sdk_training_image() -> dict[str, str]:
    """Resolve the frozen DLC selector and bind it to the already proven digest."""

    validate_aws_sdk_versions()
    from sagemaker import image_uris

    selected = image_uris.retrieve(
        framework="huggingface",
        region="us-east-1",
        version="4.49.0",
        py_version="py311",
        instance_type="ml.g5.12xlarge",
        image_scope="training",
        base_framework_version="pytorch2.5.1",
    )
    if selected != EXPECTED_SDK_TRAINING_IMAGE_TAG:
        raise RuntimeError(
            f"SageMaker SDK selected a different training image: {selected!r}"
        )
    base = _docker_inspect(EXPECTED_SDK_TRAINING_IMAGE_DIGEST_URI)
    repo_digests = base.get("RepoDigests")
    if (
        type(repo_digests) is not list
        or EXPECTED_SDK_TRAINING_IMAGE_DIGEST_URI not in repo_digests
    ):
        raise RuntimeError("Local SDK-selected base image is not bound to the frozen digest")
    return {
        "resolved_tag_uri": selected,
        "resolved_digest_uri": EXPECTED_SDK_TRAINING_IMAGE_DIGEST_URI,
    }


@dataclass(frozen=True)
class AwsClients:
    sts: object
    iam: object
    ecr: object
    s3: object
    service_quotas: object
    ec2: object
    sagemaker: object
    logs: object


def make_clients(*, region: str) -> AwsClients:
    import boto3

    client_config = no_retry_botocore_config()
    session = boto3.Session(region_name=region)
    return AwsClients(
        sts=session.client("sts", config=client_config),
        iam=session.client("iam", config=client_config),
        ecr=session.client("ecr", config=client_config),
        s3=session.client("s3", config=client_config),
        service_quotas=session.client("service-quotas", config=client_config),
        ec2=session.client("ec2", config=client_config),
        sagemaker=session.client("sagemaker", config=client_config),
        logs=session.client("logs", config=client_config),
    )


def validate_artifact_bucket(s3: object, *, bucket: str, region: str) -> None:
    if bucket != "ir-sagemaker" or region != "us-east-1":
        raise ValueError("Artifact bucket/region changed")
    location = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
    if location not in {None, "us-east-1"}:
        raise ValueError("Artifact bucket is outside us-east-1")
    if s3.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
        raise ValueError("Artifact bucket versioning is not enabled")
    encryption = s3.get_bucket_encryption(Bucket=bucket).get(
        "ServerSideEncryptionConfiguration"
    )
    expected_encryption = {
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                "BucketKeyEnabled": True,
            }
        ]
    }
    if encryption != expected_encryption:
        raise ValueError("Artifact bucket encryption contract changed")
    public = s3.get_public_access_block(Bucket=bucket).get(
        "PublicAccessBlockConfiguration"
    )
    if public != {
        "BlockPublicAcls": True,
        "IgnorePublicAcls": True,
        "BlockPublicPolicy": True,
        "RestrictPublicBuckets": True,
    }:
        raise ValueError("Artifact bucket public-access block changed")
    ownership = s3.get_bucket_ownership_controls(Bucket=bucket).get("OwnershipControls")
    if ownership != {"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]}:
        raise ValueError("Artifact bucket ownership contract changed")


def assert_unused_versioned_prefix(
    s3: object,
    *,
    bucket: str,
    prefix: str,
    expected_bucket_owner: str,
) -> None:
    if not prefix or prefix.startswith("/") or not prefix.endswith("/") or "//" in prefix:
        raise ValueError("S3 check requires one normalized non-root prefix ending in slash")
    if type(expected_bucket_owner) is not str or _ACCOUNT.fullmatch(
        expected_bucket_owner
    ) is None:
        raise ValueError("S3 prefix check requires one exact expected bucket owner")
    response = s3.list_object_versions(
        Bucket=bucket,
        ExpectedBucketOwner=expected_bucket_owner,
        Prefix=prefix,
        MaxKeys=2,
    )
    if (
        type(response) is not dict
        or response.get("Name") != bucket
        or response.get("Prefix") != prefix
        or response.get("MaxKeys") != 2
        or type(response.get("IsTruncated")) is not bool
    ):
        raise RuntimeError("Unused-prefix probe returned an invalid response identity")
    raw_versions = response.get("Versions")
    raw_delete_markers = response.get("DeleteMarkers")
    if raw_versions is not None and type(raw_versions) is not list:
        raise RuntimeError("Unused-prefix probe Versions must be an optional exact list")
    if raw_delete_markers is not None and type(raw_delete_markers) is not list:
        raise RuntimeError("Unused-prefix probe DeleteMarkers must be an optional exact list")
    versions = [] if raw_versions is None else raw_versions
    delete_markers = [] if raw_delete_markers is None else raw_delete_markers
    if versions or delete_markers:
        raise FileExistsError(
            f"Refusing used immutable S3 prefix, including historical versions: s3://{bucket}/{prefix}"
        )
    if response["IsTruncated"]:
        raise RuntimeError("Unused-prefix probe was unexpectedly truncated")


def stage_file_once(
    s3: object,
    *,
    source_path: Path,
    bucket: str,
    key: str,
    expected_bucket_owner: str,
) -> dict[str, Any]:
    """Conditionally publish one exact object and verify its versioned readback."""

    source_path = Path(source_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"Staged source must be one regular non-symlink file: {source_path}")
    if not key or key.startswith("/") or key.endswith("/") or "//" in key:
        raise ValueError("Staged S3 key must be normalized and identify one object")
    if type(expected_bucket_owner) is not str or _ACCOUNT.fullmatch(
        expected_bucket_owner
    ) is None:
        raise ValueError("Staged S3 object requires one exact expected bucket owner")
    size = source_path.stat().st_size
    if size < 1:
        raise ValueError("Refusing to stage an empty source file")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with source_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    digest_bytes = sha256.digest()
    digest_hex = digest_bytes.hex()
    checksum = base64.b64encode(digest_bytes).decode("ascii")
    expected_etag = f'"{md5.hexdigest()}"'
    with source_path.open("rb") as source:
        response = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=source,
            ContentLength=size,
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum,
            ExpectedBucketOwner=expected_bucket_owner,
            IfNoneMatch="*",
            Metadata={"sha256": digest_hex},
            ServerSideEncryption="AES256",
        )
    version_id = response.get("VersionId")
    if (
        type(version_id) is not str
        or not version_id
        or response.get("ServerSideEncryption") != "AES256"
        or response.get("ChecksumSHA256") != checksum
        or response.get("ETag") != expected_etag
    ):
        raise RuntimeError("S3 conditional publication returned incomplete identity")
    head = s3.head_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ChecksumMode="ENABLED",
        ExpectedBucketOwner=expected_bucket_owner,
    )
    if (
        head.get("ContentLength") != size
        or head.get("ChecksumSHA256") != checksum
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("Metadata") != {"sha256": digest_hex}
        or head.get("VersionId") != version_id
        or head.get("ETag") != expected_etag
    ):
        raise RuntimeError("S3 staged object metadata changed on readback")
    body = s3.get_object(
        Bucket=bucket,
        ExpectedBucketOwner=expected_bucket_owner,
        Key=key,
        VersionId=version_id,
    )["Body"]
    readback_sha256 = hashlib.sha256()
    readback_size = 0
    while True:
        chunk = body.read(1024 * 1024)
        if type(chunk) is not bytes:
            raise RuntimeError("S3 staged object readback returned non-bytes")
        if not chunk:
            break
        readback_size += len(chunk)
        readback_sha256.update(chunk)
    if readback_size != size or readback_sha256.hexdigest() != digest_hex:
        raise RuntimeError("S3 staged object bytes changed on versioned readback")
    return {
        "bucket": bucket,
        "etag": expected_etag,
        "key": key,
        "schema_version": 1,
        "sha256": digest_hex,
        "size": size,
        "sse": "AES256",
        "version_id": version_id,
    }


def _docker_inspect(local_image_ref: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "image", "inspect", local_image_ref],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker image inspect failed: {completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise RuntimeError("docker image inspect returned an unexpected payload")
    return value[0]


def validate_local_evaluation_image(
    local_image_ref: str = EXPECTED_LOCAL_IMAGE_REF,
) -> dict[str, str]:
    if local_image_ref != EXPECTED_LOCAL_IMAGE_REF:
        raise ValueError("Local evaluation image reference changed")
    image = _docker_inspect(local_image_ref)
    descriptor = _exact_keys(
        image.get("Descriptor"),
        {"annotations", "digest", "mediaType", "size"},
        name="local image descriptor",
    )
    if descriptor["mediaType"] != ECR_MEDIA_TYPE:
        raise ValueError("Local evaluation image is not one Docker-v2 manifest")
    if descriptor["digest"] != EXPECTED_LOCAL_IMAGE_DIGEST:
        raise ValueError("Local evaluation image manifest digest changed")
    annotations = descriptor["annotations"]
    if type(annotations) is not dict or annotations.get("config.digest") != EXPECTED_CONFIG_DIGEST:
        raise ValueError("Local evaluation image config digest changed")
    labels = image.get("Config", {}).get("Labels")
    expected_labels = {
        "io.arr-retrieval.build-identity-sha256": EXPECTED_BUILD_IDENTITY,
        "io.arr-retrieval.source-parent-commit": EXPECTED_SOURCE_PARENT_COMMIT,
        "io.arr-retrieval.source-parent-epoch": EXPECTED_SOURCE_PARENT_EPOCH,
        "io.arr-retrieval.source-parent-rfc3339": EXPECTED_SOURCE_PARENT_RFC3339,
    }
    if type(labels) is not dict or any(labels.get(key) != item for key, item in expected_labels.items()):
        raise ValueError("Local evaluation image provenance labels changed")
    return {
        "build_identity_sha256": EXPECTED_BUILD_IDENTITY,
        "config_digest": EXPECTED_CONFIG_DIGEST,
        "manifest_digest": EXPECTED_LOCAL_IMAGE_DIGEST,
        "media_type": ECR_MEDIA_TYPE,
        "source_parent_commit": EXPECTED_SOURCE_PARENT_COMMIT,
        "source_parent_epoch": EXPECTED_SOURCE_PARENT_EPOCH,
        "source_parent_rfc3339": EXPECTED_SOURCE_PARENT_RFC3339,
    }


def validate_local_training_image(
    local_image_ref: str = EXPECTED_TRAINING_IMAGE_REF,
) -> dict[str, str]:
    if local_image_ref != EXPECTED_TRAINING_IMAGE_REF:
        raise ValueError("Local training image reference changed")
    image = _docker_inspect(local_image_ref)
    descriptor = image.get("Descriptor")
    if (
        type(descriptor) is not dict
        or descriptor.get("mediaType") != ECR_MEDIA_TYPE
        or descriptor.get("digest") != EXPECTED_TRAINING_IMAGE_DIGEST
        or descriptor.get("annotations", {}).get("config.digest")
        != EXPECTED_TRAINING_CONFIG_DIGEST
    ):
        raise ValueError("Local training image descriptor changed")
    config_value = image.get("Config")
    if (
        type(config_value) is not dict
        or config_value.get("Entrypoint") != ["bash", "-m", "start_with_right_hostname.sh"]
        or config_value.get("Cmd") != ["/bin/bash"]
    ):
        raise ValueError("Local training image SageMaker entrypoint changed")
    labels = config_value.get("Labels")
    if (
        type(labels) is not dict
        or labels.get("org.opencontainers.image.training-contract.sha256")
        != EXPECTED_TRAINING_CONTRACT_SHA256
        or labels.get("org.opencontainers.image.base.digest")
        != EXPECTED_SDK_TRAINING_IMAGE_DIGEST_URI.rsplit("@", 1)[1]
    ):
        raise ValueError("Local training image provenance labels changed")
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--pull=never",
            "--entrypoint",
            "/opt/conda/bin/python",
            f"arr-retrieval-train@{EXPECTED_TRAINING_IMAGE_DIGEST}",
            "/opt/training_image/runtime_contract.py",
            "--contract",
            "/opt/training_image/image_contract.json",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Training image runtime contract failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    inventory = json.loads(completed.stdout)
    inventory_sha256 = sha256_bytes(canonical_json_bytes(inventory))
    if inventory_sha256 != EXPECTED_TRAINING_RUNTIME_INVENTORY_SHA256:
        raise RuntimeError("Training image runtime inventory hash changed")
    return {
        "config_digest": EXPECTED_TRAINING_CONFIG_DIGEST,
        "contract_sha256": EXPECTED_TRAINING_CONTRACT_SHA256,
        "manifest_digest": EXPECTED_TRAINING_IMAGE_DIGEST,
        "media_type": ECR_MEDIA_TYPE,
        "runtime_inventory_sha256": inventory_sha256,
    }


def _validate_repository(repository: Mapping[str, Any]) -> None:
    if repository.get("repositoryName") != ECR_REPOSITORY_NAME:
        raise ValueError("ECR repository name changed")
    if repository.get("imageTagMutability") != "IMMUTABLE":
        raise ValueError("ECR repository tags are not immutable")
    if repository.get("imageScanningConfiguration") != {"scanOnPush": True}:
        raise ValueError("ECR repository scan-on-push configuration changed")
    if repository.get("encryptionConfiguration") != {"encryptionType": "AES256"}:
        raise ValueError("ECR repository encryption configuration changed")


def ensure_evaluation_repository(ecr: object, *, create_if_absent: bool) -> dict[str, Any]:
    """Validate the repository or create it exactly once with the frozen contract."""

    from botocore.exceptions import ClientError

    try:
        response = ecr.describe_repositories(repositoryNames=[ECR_REPOSITORY_NAME])
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code != "RepositoryNotFoundException" or not create_if_absent:
            raise
        created = ecr.create_repository(
            repositoryName=ECR_REPOSITORY_NAME,
            imageTagMutability="IMMUTABLE",
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
            tags=[
                {"Key": "Experiment", "Value": "arr_retrieval_cv_v1"},
                {"Key": "ManagedBy", "Value": "arr-retrieval-cv"},
            ],
        )
        repository = created.get("repository")
    else:
        repositories = response.get("repositories")
        if type(repositories) is not list or len(repositories) != 1:
            raise RuntimeError("ECR repository lookup did not return exactly one repository")
        repository = repositories[0]
    if type(repository) is not dict:
        raise RuntimeError("ECR repository response is malformed")
    _validate_repository(repository)
    return dict(repository)


def _remote_image_identity(account_id: str, region: str) -> tuple[str, str, str]:
    if _ACCOUNT.fullmatch(account_id) is None or region != "us-east-1":
        raise ValueError("Remote image account/region changed")
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    repository_uri = f"{registry}/{ECR_REPOSITORY_NAME}"
    content_tag = f"build-{EXPECTED_BUILD_IDENTITY}"
    if len(content_tag) > 128:
        raise AssertionError("Content-derived ECR tag exceeds the service limit")
    return registry, repository_uri, content_tag


def _run_docker(arguments: Sequence[str], *, stdin: bytes | None = None) -> None:
    completed = subprocess.run(
        ["docker", *arguments],
        input=stdin,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"docker {' '.join(arguments[:2])} failed: {stderr}")


def _publish_verified_image_once(
    *,
    ecr: object,
    account_id: str,
    region: str,
    local_image_ref: str,
    expected_manifest_digest: str,
    content_tag: str,
    protocol: str,
    identity: Mapping[str, str],
) -> dict[str, Any]:
    ensure_evaluation_repository(ecr, create_if_absent=False)
    registry, repository_uri, _ = _remote_image_identity(account_id, region)
    if (
        not content_tag
        or len(content_tag) > 128
        or re.fullmatch(r"[0-9A-Za-z_.-]+", content_tag) is None
    ):
        raise ValueError("Content-derived ECR tag is invalid")
    remote_tag = f"{repository_uri}:{content_tag}"
    existing = ecr.batch_get_image(
        registryId=account_id,
        repositoryName=ECR_REPOSITORY_NAME,
        imageIds=[{"imageTag": content_tag}],
        acceptedMediaTypes=[ECR_MEDIA_TYPE],
    )
    if existing.get("images"):
        raise FileExistsError(f"Refusing to republish existing immutable ECR tag: {remote_tag}")
    failures = existing.get("failures")
    if (
        type(failures) is not list
        or len(failures) != 1
        or failures[0].get("failureCode") != "ImageNotFound"
    ):
        raise RuntimeError(f"ECR immutable-tag absence probe was inconclusive: {failures}")

    authorization = ecr.get_authorization_token().get("authorizationData")
    if type(authorization) is not list or len(authorization) != 1:
        raise RuntimeError("ECR authorization did not return exactly one token")
    decoded = base64.b64decode(authorization[0]["authorizationToken"], validate=True)
    username, separator, password = decoded.partition(b":")
    if username != b"AWS" or separator != b":" or not password:
        raise RuntimeError("ECR authorization token has an unexpected structure")
    _run_docker(("login", "--username", "AWS", "--password-stdin", registry), stdin=password)
    _run_docker(("tag", local_image_ref, remote_tag))
    _run_docker(("push", remote_tag))

    response = ecr.batch_get_image(
        registryId=account_id,
        repositoryName=ECR_REPOSITORY_NAME,
        imageIds=[{"imageTag": content_tag}],
        acceptedMediaTypes=[ECR_MEDIA_TYPE],
    )
    if response.get("failures"):
        raise RuntimeError(f"ECR readback failed: {response['failures']}")
    images = response.get("images")
    if type(images) is not list or len(images) != 1:
        raise RuntimeError("ECR readback did not return exactly one image")
    image = images[0]
    remote_digest = image.get("imageId", {}).get("imageDigest")
    raw_manifest = image.get("imageManifest")
    if type(raw_manifest) is not str:
        raise RuntimeError("ECR did not return the raw image manifest")
    raw_digest = "sha256:" + sha256_bytes(raw_manifest.encode("utf-8"))
    if remote_digest != expected_manifest_digest or raw_digest != expected_manifest_digest:
        raise RuntimeError(
            "Published image digest differs from the verified local manifest: "
            f"local={expected_manifest_digest}, ECR={remote_digest}, raw={raw_digest}"
        )
    remote_digest_uri = f"{repository_uri}@{remote_digest}"
    _run_docker(("pull", remote_digest_uri))
    pulled = _docker_inspect(remote_digest_uri)
    if pulled.get("Descriptor", {}).get("digest") != remote_digest:
        raise RuntimeError("Digest-addressed ECR pull did not preserve the manifest digest")
    return {
        "content_tag": content_tag,
        "identity": dict(identity),
        "manifest_digest": remote_digest,
        "media_type": ECR_MEDIA_TYPE,
        "protocol": protocol,
        "raw_manifest_sha256": raw_digest.removeprefix("sha256:"),
        "remote_digest_uri": remote_digest_uri,
        "remote_tag_uri": remote_tag,
    }


def publish_evaluation_image_once(
    *,
    ecr: object,
    account_id: str,
    region: str,
    local_image_ref: str = EXPECTED_LOCAL_IMAGE_REF,
) -> dict[str, Any]:
    """Push the already verified evaluation manifest once and prove identity."""

    local = validate_local_evaluation_image(local_image_ref)
    return _publish_verified_image_once(
        ecr=ecr,
        account_id=account_id,
        region=region,
        local_image_ref=local_image_ref,
        expected_manifest_digest=local["manifest_digest"],
        content_tag=f"build-{EXPECTED_BUILD_IDENTITY}",
        protocol="immutable_ecr_evaluation_image_publication_v1",
        identity=local,
    )


def publish_training_image_once(
    *,
    ecr: object,
    account_id: str,
    region: str,
    local_image_ref: str = EXPECTED_TRAINING_IMAGE_REF,
) -> dict[str, Any]:
    """Push the twice-reproduced training manifest once and prove identity."""

    local = validate_local_training_image(local_image_ref)
    return _publish_verified_image_once(
        ecr=ecr,
        account_id=account_id,
        region=region,
        local_image_ref=local_image_ref,
        expected_manifest_digest=local["manifest_digest"],
        content_tag=f"training-{EXPECTED_TRAINING_CONTRACT_SHA256}",
        protocol="immutable_ecr_training_image_publication_v1",
        identity=local,
    )


def render_runtime_smoke_request(
    aws_config: Mapping[str, Any],
    *,
    remote_image_uri: str,
    job_name: str,
) -> dict[str, Any]:
    config = validate_aws_config(dict(aws_config))
    expected_image_uri = (
        f"{config['account_id']}.dkr.ecr.{config['region']}.amazonaws.com/"
        f"{ECR_REPOSITORY_NAME}@{EXPECTED_LOCAL_IMAGE_DIGEST}"
    )
    if remote_image_uri != expected_image_uri:
        raise ValueError(
            "Runtime smoke image must equal the frozen account-local evaluation digest"
        )
    if not _JOB_NAME.fullmatch(job_name) or "runtime-smoke" not in job_name:
        raise ValueError("Runtime smoke job name is invalid or insufficiently explicit")
    tags = [
        {"Key": key, "Value": config["tags"][key]}
        for key in sorted(config["tags"])
    ]
    return {
        "AppSpecification": {
            "ContainerArguments": [
                "/opt/program/modernbert/processing_eval/image_smoke.py",
                "--contract",
                "/opt/program/modernbert/processing_eval/image_contract.json",
                "--build-manifest",
                "/opt/program/modernbert/processing_eval/build_context_manifest.json",
                "--expected-build-identity-sha256",
                EXPECTED_BUILD_IDENTITY,
                "--expected-source-parent-commit",
                EXPECTED_SOURCE_PARENT_COMMIT,
                "--expected-source-parent-epoch",
                EXPECTED_SOURCE_PARENT_EPOCH,
                "--expected-source-parent-rfc3339",
                EXPECTED_SOURCE_PARENT_RFC3339,
            ],
            "ContainerEntrypoint": ["/opt/conda/bin/python"],
            "ImageUri": remote_image_uri,
        },
        "Environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "17",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "NetworkConfig": {
            "EnableInterContainerTrafficEncryption": False,
            "EnableNetworkIsolation": True,
        },
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": config["processing_instance_type"],
                "VolumeSizeInGB": config["processing_volume_size_gb"],
            }
        },
        "RoleArn": config["role_arn"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": config["processing_max_runtime_seconds"]
        },
        "Tags": tags,
    }


def _assert_role_trust(iam: object, role_arn: str) -> None:
    role_name = role_arn.rsplit("/", 1)[1]
    role = iam.get_role(RoleName=role_name).get("Role")
    if type(role) is not dict:
        raise RuntimeError("IAM role lookup is malformed")
    trust = role.get("AssumeRolePolicyDocument")
    expected = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    if trust != expected:
        raise ValueError("SageMaker execution-role trust policy changed")


def preflight_runtime_smoke(
    clients: AwsClients,
    aws_config: Mapping[str, Any],
    *,
    remote_image_uri: str,
    job_name: str,
) -> dict[str, Any]:
    config = validate_aws_config(dict(aws_config))
    sdk_versions = validate_aws_sdk_versions()
    request = render_runtime_smoke_request(
        config,
        remote_image_uri=remote_image_uri,
        job_name=job_name,
    )
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != config["account_id"]:
        raise ValueError("Active AWS account differs from the AWS-local contract")
    _assert_role_trust(clients.iam, config["role_arn"])
    ensure_evaluation_repository(clients.ecr, create_if_absent=False)
    image_digest = remote_image_uri.rsplit("@", 1)[1]
    image_response = clients.ecr.batch_get_image(
        registryId=config["account_id"],
        repositoryName=ECR_REPOSITORY_NAME,
        imageIds=[{"imageDigest": image_digest}],
        acceptedMediaTypes=[ECR_MEDIA_TYPE],
    )
    if image_response.get("failures") or len(image_response.get("images", [])) != 1:
        raise ValueError("Digest-addressed evaluation image is not readable from ECR")
    raw = image_response["images"][0].get("imageManifest")
    if type(raw) is not str or _raw_ecr_manifest_digest(raw) != image_digest:
        raise ValueError("ECR raw manifest differs from its requested digest")
    quota = clients.service_quotas.get_service_quota(
        ServiceCode="sagemaker",
        QuotaCode="L-B013C051",
    ).get("Quota", {})
    if quota.get("Value", 0) < 1:
        raise RuntimeError("Processing ml.g5.12xlarge quota is below one")
    offerings = clients.ec2.describe_instance_type_offerings(
        LocationType="region",
        Filters=[{"Name": "instance-type", "Values": ["g5.12xlarge"]}],
    ).get("InstanceTypeOfferings", [])
    if not any(item.get("InstanceType") == "g5.12xlarge" for item in offerings):
        raise RuntimeError("g5.12xlarge is not offered in the configured region")
    existing = clients.sagemaker.list_processing_jobs(
        NameContains=job_name,
        MaxResults=100,
        SortBy="Name",
        SortOrder="Ascending",
    ).get("ProcessingJobSummaries", [])
    if any(item.get("ProcessingJobName") == job_name for item in existing):
        raise FileExistsError(f"SageMaker Processing job name already exists: {job_name}")
    receipt = {
        "account_id": config["account_id"],
        "aws_config": config,
        "aws_config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "caller_arn": caller.get("Arn"),
        "image_manifest_digest": image_digest,
        "job_name": job_name,
        "processing_quota": int(quota["Value"]),
        "protocol": RUNTIME_SMOKE_PROTOCOL,
        "region": config["region"],
        "request": request,
        "request_sha256": sha256_bytes(canonical_json_bytes(request)),
        "sdk_versions": sdk_versions,
    }
    return validate_runtime_smoke_preflight_receipt(receipt)


def validate_runtime_smoke_preflight_receipt(
    value: object,
) -> dict[str, Any]:
    receipt = _exact_keys(
        value,
        {
            "account_id",
            "aws_config",
            "aws_config_sha256",
            "caller_arn",
            "image_manifest_digest",
            "job_name",
            "processing_quota",
            "protocol",
            "region",
            "request",
            "request_sha256",
            "sdk_versions",
        },
        name="runtime-smoke preflight receipt",
    )
    if receipt["protocol"] != RUNTIME_SMOKE_PROTOCOL:
        raise ValueError("A scientific/fold receipt cannot be used for a runtime smoke")
    config = validate_aws_config(receipt["aws_config"])
    config_sha256 = sha256_bytes(canonical_json_bytes(config))
    if receipt["aws_config_sha256"] != config_sha256:
        raise ValueError("Runtime-smoke AWS configuration hash changed")
    if (
        receipt["account_id"] != config["account_id"]
        or receipt["region"] != config["region"]
    ):
        raise ValueError("Runtime-smoke receipt account/region differs from its AWS contract")
    caller_arn = _string(receipt["caller_arn"], name="runtime-smoke caller ARN")
    if not caller_arn.startswith((
        f"arn:aws:iam::{config['account_id']}:",
        f"arn:aws:sts::{config['account_id']}:",
    )):
        raise ValueError("Runtime-smoke caller ARN differs from its AWS account")
    if receipt["image_manifest_digest"] != EXPECTED_LOCAL_IMAGE_DIGEST:
        raise ValueError("Runtime-smoke receipt is not bound to the frozen evaluation digest")
    job_name = _string(receipt["job_name"], name="runtime-smoke job name")
    quota = _positive_int(receipt["processing_quota"], name="runtime-smoke quota")
    if quota < 1:
        raise AssertionError("positive quota validation failed")
    if receipt["sdk_versions"] != EXPECTED_AWS_SDK_VERSIONS:
        raise ValueError("Runtime-smoke AWS SDK inventory changed")
    remote_image_uri = (
        f"{config['account_id']}.dkr.ecr.{config['region']}.amazonaws.com/"
        f"{ECR_REPOSITORY_NAME}@{EXPECTED_LOCAL_IMAGE_DIGEST}"
    )
    expected_request = render_runtime_smoke_request(
        config,
        remote_image_uri=remote_image_uri,
        job_name=job_name,
    )
    if receipt["request"] != expected_request:
        raise ValueError("Runtime-smoke request differs from the re-rendered frozen request")
    request_sha256 = sha256_bytes(canonical_json_bytes(expected_request))
    if receipt["request_sha256"] != request_sha256:
        raise ValueError("Runtime-smoke request hash changed after preflight")
    return dict(receipt)


def submit_runtime_smoke(
    clients: AwsClients,
    *,
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = validate_runtime_smoke_preflight_receipt(preflight_receipt)
    caller = clients.sts.get_caller_identity()
    if (
        caller.get("Account") != receipt["account_id"]
        or caller.get("Arn") != receipt["caller_arn"]
    ):
        raise ValueError("Active AWS caller differs from the preflight receipt")
    request = receipt["request"]
    request_sha256 = receipt["request_sha256"]
    response = clients.sagemaker.create_processing_job(**request)
    arn = response.get("ProcessingJobArn")
    expected_arn = (
        f"arn:aws:sagemaker:{receipt['region']}:{receipt['account_id']}:"
        f"processing-job/{request['ProcessingJobName']}"
    )
    if arn != expected_arn:
        raise RuntimeError("CreateProcessingJob returned an unexpected ARN")
    return {
        "job_arn": arn,
        "job_name": request["ProcessingJobName"],
        "preflight_receipt_sha256": sha256_bytes(
            canonical_json_bytes(receipt)
        ),
        "protocol": RUNTIME_SMOKE_PROTOCOL,
        "request_sha256": request_sha256,
    }


def validate_runtime_smoke_submission_receipt(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_runtime_smoke_preflight_receipt(preflight_receipt)
    receipt = _exact_keys(
        value,
        {
            "job_arn",
            "job_name",
            "preflight_receipt_sha256",
            "protocol",
            "request_sha256",
        },
        name="runtime-smoke submission receipt",
    )
    if receipt["protocol"] != RUNTIME_SMOKE_PROTOCOL:
        raise ValueError("Submission receipt is not a runtime smoke")
    if (
        receipt["job_name"] != preflight["job_name"]
        or receipt["request_sha256"] != preflight["request_sha256"]
        or receipt["preflight_receipt_sha256"]
        != sha256_bytes(canonical_json_bytes(preflight))
    ):
        raise ValueError("Runtime-smoke submission receipt differs from preflight")
    expected_arn = (
        f"arn:aws:sagemaker:{preflight['region']}:{preflight['account_id']}:"
        f"processing-job/{preflight['job_name']}"
    )
    if receipt["job_arn"] != expected_arn:
        raise ValueError("Runtime-smoke submission job ARN changed")
    return dict(receipt)


def describe_runtime_smoke(
    sagemaker: object,
    *,
    job_name: str,
) -> dict[str, Any]:
    response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
    return {
        "app_specification": response.get("AppSpecification"),
        "failure_reason": response.get("FailureReason"),
        "job_arn": response.get("ProcessingJobArn"),
        "job_name": response.get("ProcessingJobName"),
        "network_config": response.get("NetworkConfig"),
        "processing_resources": response.get("ProcessingResources"),
        "protocol": RUNTIME_SMOKE_PROTOCOL,
        "role_arn": response.get("RoleArn"),
        "status": response.get("ProcessingJobStatus"),
        "stopping_condition": response.get("StoppingCondition"),
    }


def verify_completed_runtime_smoke(
    clients: AwsClients,
    *,
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_runtime_smoke_preflight_receipt(preflight_receipt)
    submission = validate_runtime_smoke_submission_receipt(
        submission_receipt,
        preflight_receipt=preflight,
    )
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != preflight["account_id"]:
        raise ValueError("Active AWS account differs from the runtime-smoke receipt")
    request = preflight["request"]
    job_name = request["ProcessingJobName"]
    response = clients.sagemaker.describe_processing_job(ProcessingJobName=job_name)
    if response.get("ProcessingJobStatus") != "Completed" or response.get("FailureReason"):
        raise RuntimeError(
            f"Runtime smoke is not cleanly complete: "
            f"status={response.get('ProcessingJobStatus')}, "
            f"reason={response.get('FailureReason')!r}"
        )
    if (
        response.get("ProcessingJobName") != job_name
        or response.get("ProcessingJobArn") != submission["job_arn"]
    ):
        raise RuntimeError("DescribeProcessingJob identity differs from submission")
    expected_fields = {
        "AppSpecification": request["AppSpecification"],
        "Environment": request["Environment"],
        "NetworkConfig": request["NetworkConfig"],
        "ProcessingResources": request["ProcessingResources"],
        "RoleArn": request["RoleArn"],
        "StoppingCondition": request["StoppingCondition"],
    }
    for key, expected in expected_fields.items():
        if response.get(key) != expected:
            raise RuntimeError(f"DescribeProcessingJob {key} differs from the frozen request")

    stream_prefix = job_name + "/"
    streams: list[dict[str, Any]] = []
    next_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        arguments: dict[str, Any] = {
            "logGroupName": "/aws/sagemaker/ProcessingJobs",
            "logStreamNamePrefix": stream_prefix,
            "orderBy": "LogStreamName",
            "descending": False,
            "limit": 50,
        }
        if next_token is not None:
            arguments["nextToken"] = next_token
        page = clients.logs.describe_log_streams(**arguments)
        page_streams = page.get("logStreams", [])
        if type(page_streams) is not list:
            raise RuntimeError("CloudWatch returned malformed log streams")
        streams.extend(page_streams)
        candidate = page.get("nextToken")
        if candidate is None:
            break
        if type(candidate) is not str or not candidate or candidate in seen_tokens:
            raise RuntimeError("CloudWatch log-stream pagination did not advance")
        seen_tokens.add(candidate)
        next_token = candidate
    if type(streams) is not list or not streams:
        raise RuntimeError("Runtime smoke has no CloudWatch log stream")
    payloads: list[dict[str, Any]] = []
    for stream in streams:
        name = stream.get("logStreamName")
        if type(name) is not str or not name.startswith(stream_prefix):
            raise RuntimeError("CloudWatch returned a foreign log-stream name")
        event_token: str | None = None
        seen_event_tokens: set[str] = set()
        while True:
            event_arguments: dict[str, Any] = {
                "logGroupName": "/aws/sagemaker/ProcessingJobs",
                "logStreamName": name,
                "startFromHead": True,
            }
            if event_token is not None:
                event_arguments["nextToken"] = event_token
            event_page = clients.logs.get_log_events(**event_arguments)
            events = event_page.get("events", [])
            if type(events) is not list:
                raise RuntimeError("CloudWatch returned malformed log events")
            for event in events:
                message = event.get("message")
                if type(message) is not str or not message.startswith("{"):
                    continue
                try:
                    value = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if type(value) is dict and set(value) == {
                    "build_context",
                    "image_contract_sha256",
                    "neural_runtime",
                    "platform",
                    "sparse_runtime",
                }:
                    payloads.append(value)
            candidate = event_page.get("nextForwardToken")
            if type(candidate) is not str or not candidate:
                raise RuntimeError("CloudWatch log-event page has no forward token")
            if candidate == event_token:
                break
            if candidate in seen_event_tokens:
                raise RuntimeError("CloudWatch log-event pagination cycled")
            seen_event_tokens.add(candidate)
            event_token = candidate
    if len(payloads) != 1:
        raise RuntimeError(
            f"Expected exactly one canonical runtime identity in CloudWatch; got {len(payloads)}"
        )
    identity = payloads[0]
    runtime_identity_sha256 = sha256_bytes(canonical_json_bytes(identity))
    if runtime_identity_sha256 != EXPECTED_RUNTIME_IDENTITY_SHA256:
        raise RuntimeError("Runtime-smoke complete runtime identity changed")
    return {
        "job_arn": response["ProcessingJobArn"],
        "job_name": job_name,
        "protocol": RUNTIME_SMOKE_PROTOCOL,
        "request_sha256": preflight["request_sha256"],
        "runtime_identity": identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "status": "Completed",
    }
