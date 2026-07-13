"""Strict, dependency-free configuration primitives for retrieval CV orchestration.

The launch layer treats configuration bytes as scientific inputs.  This module
therefore accepts one canonical JSON representation, rejects duplicate or
unknown fields, and uses exact JSON types (in particular, ``bool`` is never an
integer).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = 1
CONTROLLED_FOLDS = (0, 1, 2, 3, 4)
CONTROLLED_QUERY_VIEWS = ("flat_masked", "structured")
CONTROLLED_SAMPLERS = ("local_unique", "global_uniform")
CONTROLLED_SEEDS = (17, 29, 43)
REQUIRED_ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_ACCOUNT_RE = re.compile(r"[0-9]{12}\Z")
_REGION_RE = re.compile(r"[a-z]{2}(?:-gov)?-[a-z]+-[0-9]\Z")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_ECR_REPOSITORY_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*\Z")
_ECR_IMAGE_RE = re.compile(
    r"(?P<account>[0-9]{12})\.dkr\.ecr\."
    r"(?P<region>[a-z]{2}(?:-gov)?-[a-z]+-[0-9])\.amazonaws\.com/"
    r"(?P<repository>[a-z0-9]+(?:[._/-][a-z0-9]+)*)@"
    r"(?P<digest>sha256:[0-9a-f]{64})\Z"
)
_CHANNEL_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole accepted UTF-8 JSON representation, including newline."""

    _validate_json_value(value, name="JSON value")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise TypeError("sha256_bytes input must be exact bytes")
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"Non-finite JSON number is forbidden: {token}")


def _read_regular_file_once(path: Path) -> bytes:
    path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_canonical_json_object(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Read once, hash those exact bytes, and parse one canonical JSON object."""

    if expected_sha256 is not None:
        _require_sha256("expected_sha256", expected_sha256)
    raw = _read_regular_file_once(Path(path))
    actual_sha256 = sha256_bytes(raw)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Canonical JSON hash mismatch for {path}: "
            f"actual={actual_sha256}, expected={expected_sha256}"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"Canonical JSON is not UTF-8: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"Canonical JSON must contain one object: {path}")
    if raw != canonical_json_bytes(value):
        raise ValueError(f"JSON does not use canonical deterministic bytes: {path}")
    return value, actual_sha256


def _validate_json_value(value: object, *, name: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} contains a non-string object key")
            _validate_json_value(item, name=f"{name}.{key}")
        return
    raise TypeError(f"{name} contains unsupported JSON type {type(value).__name__}")


def _require_object(
    name: str,
    value: object,
    *,
    keys: set[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ValueError(f"{name} keys mismatch: missing={missing}, unknown={unknown}")
    return value


def _require_string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _require_image_digest(name: str, value: object) -> str:
    if type(value) is not str or not value.startswith("sha256:"):
        raise ValueError(f"{name} must be an immutable sha256: image digest")
    _require_sha256(name, value.removeprefix("sha256:"))
    return value


def _require_ecr_image_uri(
    name: str,
    value: object,
    *,
    expected_digest: str,
) -> str:
    text = _require_string(name, value)
    match = _ECR_IMAGE_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"{name} must be one immutable private-ECR @sha256 URI")
    if match.group("digest") != expected_digest:
        raise ValueError(f"{name} digest does not match its separately bound digest")
    return text


def _ecr_image_coordinates(value: str) -> tuple[str, str, str, str]:
    match = _ECR_IMAGE_RE.fullmatch(value)
    if match is None:
        raise ValueError("Expected one previously validated immutable ECR image URI")
    return (
        match.group("account"),
        match.group("region"),
        match.group("repository"),
        match.group("digest"),
    )


def _s3_uri_coordinates(value: str) -> tuple[str, str]:
    _require_s3_uri("S3 URI", value)
    bucket, key = value.removeprefix("s3://").split("/", 1)
    return bucket, key


def _require_git_object(name: str, value: object) -> str:
    if type(value) is not str or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase Git object ID")
    return value


def _require_posix_relative_path(name: str, value: object) -> str:
    text = _require_string(name, value)
    if "\\" in text:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or text != path.as_posix() or text in {".", ".."}:
        raise ValueError(f"{name} must be one normalized relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must not contain empty, dot, or parent components")
    return text


def _require_s3_uri(name: str, value: object) -> str:
    text = _require_string(name, value)
    if not text.startswith("s3://") or text.endswith("/"):
        raise ValueError(f"{name} must be a non-directory s3:// URI")
    bucket_and_key = text[5:]
    if "/" not in bucket_and_key:
        raise ValueError(f"{name} must include an S3 object key")
    bucket, key = bucket_and_key.split("/", 1)
    if _BUCKET_RE.fullmatch(bucket) is None or not key or "//" in key:
        raise ValueError(f"{name} is not a normalized S3 object URI")
    _require_posix_relative_path(f"{name} key", key)
    return text


def _validate_artifact_identity(name: str, value: object) -> dict[str, Any]:
    record = _require_object(
        name,
        value,
        keys={"schema_version", "artifact_type", "validator_version"},
    )
    _require_int(f"{name}.schema_version", record["schema_version"], minimum=1)
    _require_string(f"{name}.artifact_type", record["artifact_type"])
    _require_string(f"{name}.validator_version", record["validator_version"])
    return record


def _validate_environment(name: str, value: object) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise TypeError(f"{name} must be one non-empty exact JSON object")
    for key, item in value.items():
        if type(key) is not str or not key or type(item) is not str or not item:
            raise TypeError(f"{name} must map non-empty strings to non-empty strings")
    for key, expected in REQUIRED_ENVIRONMENT.items():
        if value.get(key) != expected:
            raise ValueError(f"{name}.{key} must equal {expected!r}")
    if "PYTHONHASHSEED" in value:
        raise ValueError(f"{name} must not bind per-run PYTHONHASHSEED")
    return value


def _validate_string_map(name: str, value: object) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise TypeError(f"{name} must be one non-empty exact JSON object")
    for key, item in value.items():
        if type(key) is not str or not key or type(item) is not str or not item:
            raise TypeError(f"{name} must map non-empty strings to non-empty strings")
    return value


def _validate_hyperparameters(name: str, value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be one exact JSON object")
    _validate_json_value(value, name=name)
    for key, item in value.items():
        if type(key) is not str or not key:
            raise TypeError(f"{name} has an invalid parameter name")
        if type(item) not in {str, bool, int, float}:
            raise TypeError(f"{name}.{key} must be one JSON scalar")
    return value


def validate_input_channels(name: str, value: object) -> dict[str, dict[str, Any]]:
    if type(value) is not dict or not value:
        raise TypeError(f"{name} must contain at least one input channel")
    for channel, raw_record in value.items():
        if type(channel) is not str or _CHANNEL_RE.fullmatch(channel) is None:
            raise ValueError(f"{name} has invalid channel name {channel!r}")
        record = _require_object(
            f"{name}.{channel}",
            raw_record,
            keys={"s3_uri", "identity_sha256"},
        )
        _require_s3_uri(f"{name}.{channel}.s3_uri", record["s3_uri"])
        _require_sha256(
            f"{name}.{channel}.identity_sha256", record["identity_sha256"]
        )
    return value


def _validate_run_template(name: str, value: object) -> dict[str, Any]:
    template = _require_object(
        name,
        value,
        keys={
            "entry_point",
            "hyperparameters",
            "environment",
            "input_channels",
            "expected_artifact_identity",
        },
    )
    _require_posix_relative_path(f"{name}.entry_point", template["entry_point"])
    _validate_hyperparameters(f"{name}.hyperparameters", template["hyperparameters"])
    _validate_environment(f"{name}.environment", template["environment"])
    validate_input_channels(f"{name}.input_channels", template["input_channels"])
    _validate_artifact_identity(
        f"{name}.expected_artifact_identity",
        template["expected_artifact_identity"],
    )
    return template


def validate_scientific_config(value: object) -> dict[str, Any]:
    """Validate the tracked scientific orchestration configuration schema v1."""

    config = _require_object(
        "scientific config",
        value,
        keys={"schema_version", "study", "sources", "run_templates"},
    )
    if _require_int("scientific config.schema_version", config["schema_version"], minimum=1) != 1:
        raise ValueError("Unsupported scientific config schema_version")

    study = _require_object(
        "scientific config.study",
        config["study"],
        keys={
            "study_id",
            "experiment_config_sha256",
            "fold_manifest_sha256",
            "dataset_manifest_sha256",
            "deepspeed_config_sha256",
            "model_snapshot_tree_sha256",
            "evaluation_image_uri",
            "evaluation_image_digest",
            "evaluation_image_inventory_sha256",
            "training_image_uri",
            "training_image_digest",
            "training_image_inventory_sha256",
            "training_base_image_uri",
        },
    )
    _require_string("scientific config.study.study_id", study["study_id"])
    for key in (
        "experiment_config_sha256",
        "fold_manifest_sha256",
        "dataset_manifest_sha256",
        "deepspeed_config_sha256",
        "model_snapshot_tree_sha256",
        "evaluation_image_inventory_sha256",
        "training_image_inventory_sha256",
    ):
        _require_sha256(f"scientific config.study.{key}", study[key])
    _require_image_digest(
        "scientific config.study.evaluation_image_digest",
        study["evaluation_image_digest"],
    )
    _require_image_digest(
        "scientific config.study.training_image_digest",
        study["training_image_digest"],
    )
    _require_ecr_image_uri(
        "scientific config.study.evaluation_image_uri",
        study["evaluation_image_uri"],
        expected_digest=study["evaluation_image_digest"],
    )
    _require_ecr_image_uri(
        "scientific config.study.training_image_uri",
        study["training_image_uri"],
        expected_digest=study["training_image_digest"],
    )
    training_base_uri = _require_string(
        "scientific config.study.training_base_image_uri",
        study["training_base_image_uri"],
    )
    if _ECR_IMAGE_RE.fullmatch(training_base_uri) is None:
        raise ValueError(
            "scientific config.study.training_base_image_uri must be immutable ECR @sha256"
        )

    sources = _require_object(
        "scientific config.sources",
        config["sources"],
        keys={"git_commit", "git_tree", "commit_epoch", "include_paths"},
    )
    _require_git_object("scientific config.sources.git_commit", sources["git_commit"])
    _require_git_object("scientific config.sources.git_tree", sources["git_tree"])
    _require_int("scientific config.sources.commit_epoch", sources["commit_epoch"])
    include_paths = sources["include_paths"]
    if type(include_paths) is not list or not include_paths:
        raise TypeError("scientific config.sources.include_paths must be a non-empty list")
    normalized_paths = [
        _require_posix_relative_path(
            f"scientific config.sources.include_paths[{index}]", item
        )
        for index, item in enumerate(include_paths)
    ]
    if normalized_paths != sorted(normalized_paths) or len(normalized_paths) != len(
        set(normalized_paths)
    ):
        raise ValueError("scientific config source paths must be sorted and unique")

    templates = _require_object(
        "scientific config.run_templates",
        config["run_templates"],
        keys={"controlled", "legacy", "determinism_smoke"},
    )
    for key in ("controlled", "legacy", "determinism_smoke"):
        _validate_run_template(f"scientific config.run_templates.{key}", templates[key])

    evaluation_coordinates = _ecr_image_coordinates(study["evaluation_image_uri"])
    training_coordinates = _ecr_image_coordinates(study["training_image_uri"])
    if evaluation_coordinates[:3] != training_coordinates[:3]:
        raise ValueError(
            "Scientific evaluation and training images must share one account/region/repository"
        )

    expected_templates = {
        "controlled": {
            "entry_point": "train_sm.py",
            "artifact": {
                "artifact_type": "controlled_retriever",
                "schema_version": 1,
                "validator_version": "controlled_retrieval_artifact_v1",
            },
        },
        "legacy": {
            "entry_point": "train_sm.py",
            "artifact": {
                "artifact_type": "corrected_legacy_diagnostic_retriever",
                "schema_version": 1,
                "validator_version": "corrected_legacy_diagnostic_artifact_v1",
            },
        },
        "determinism_smoke": {
            "entry_point": "train_sm.py",
            "artifact": {
                "artifact_type": "determinism_smoke_retriever",
                "schema_version": 1,
                "validator_version": "determinism_smoke_artifact_v1",
            },
        },
    }
    expected_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FLASH_ATTENTION_DETERMINISTIC": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    channel_buckets: set[str] = set()
    for template_name, expected in expected_templates.items():
        template = templates[template_name]
        if template["entry_point"] != expected["entry_point"]:
            raise ValueError(f"{template_name} entry point changed")
        if template["expected_artifact_identity"] != expected["artifact"]:
            raise ValueError(f"{template_name} artifact protocol changed")
        if template["environment"] != expected_environment:
            raise ValueError(f"{template_name} environment contract changed")
        channels = template["input_channels"]
        if set(channels) != {"base_model", "data"}:
            raise ValueError(
                f"{template_name} must contain exactly data and base_model channels"
            )
        expected_identities = {
            "base_model": study["model_snapshot_tree_sha256"],
            "data": study["dataset_manifest_sha256"],
        }
        for channel, expected_identity in expected_identities.items():
            record = channels[channel]
            if record["identity_sha256"] != expected_identity:
                raise ValueError(
                    f"{template_name}.{channel} identity differs from the study"
                )
            bucket, key = _s3_uri_coordinates(record["s3_uri"])
            channel_buckets.add(bucket)
            if not key.endswith(expected_identity):
                raise ValueError(
                    f"{template_name}.{channel} URI is not content-addressed by its identity"
                )
    if len(channel_buckets) != 1:
        raise ValueError("Scientific input channels must share one artifact bucket")

    generated_keys = {"outer_fold", "query_view", "sampler", "experiment_seed"}
    controlled_parameters = templates["controlled"]["hyperparameters"]
    smoke_parameters = templates["determinism_smoke"]["hyperparameters"]
    if generated_keys & set(controlled_parameters):
        raise ValueError("Controlled template must not pre-bind generated cell parameters")
    if generated_keys & set(smoke_parameters):
        raise ValueError("Smoke template must not pre-bind generated cell parameters")
    if "query_view" in templates["legacy"]["hyperparameters"]:
        raise ValueError("Legacy template must not pre-bind generated query_view")
    legacy_parameters = templates["legacy"]["hyperparameters"]
    legacy_seed = legacy_parameters.get("base_seed")
    if type(legacy_seed) is not int or legacy_seed < 0:
        raise TypeError("Legacy template must bind non-negative exact integer base_seed")
    if controlled_parameters != {}:
        raise ValueError("Controlled template hyperparameters changed")
    if legacy_parameters != {
        "base_seed": 17,
        "epochs": 20,
        "run_kind": "corrected_legacy_diagnostic",
        "total_optimizer_updates": 80,
    }:
        raise ValueError("Corrected legacy diagnostic template hyperparameters changed")
    if smoke_parameters.get("run_kind") != "determinism_smoke":
        raise ValueError("Smoke template run_kind must be 'determinism_smoke'")
    if type(smoke_parameters.get("epochs")) is not int or smoke_parameters["epochs"] != 2:
        raise ValueError("Smoke template must bind exactly two epochs")
    if (
        type(smoke_parameters.get("total_optimizer_updates")) is not int
        or smoke_parameters["total_optimizer_updates"] != 6
    ):
        raise ValueError("Smoke template must bind exactly six optimizer updates")
    if smoke_parameters != {
        "epochs": 2,
        "run_kind": "determinism_smoke",
        "total_optimizer_updates": 6,
    }:
        raise ValueError("Smoke template hyperparameters changed")
    return config


def load_scientific_config(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    value, digest = load_canonical_json_object(path, expected_sha256=expected_sha256)
    return validate_scientific_config(value), digest


def validate_aws_local_config(value: object) -> dict[str, Any]:
    """Validate the ignored, machine/account-local AWS configuration schema v1."""

    config = _require_object(
        "AWS local config",
        value,
        keys={
            "schema_version",
            "account_id",
            "region",
            "role_arn",
            "artifact_bucket",
            "artifact_root_prefix",
            "ecr_repository",
            "training_instance_type",
            "training_instance_count",
            "training_volume_size_gb",
            "training_max_runtime_seconds",
            "processing_instance_type",
            "processing_instance_count",
            "processing_volume_size_gb",
            "processing_max_runtime_seconds",
            "max_concurrent_training_jobs",
            "tags",
        },
    )
    if _require_int("AWS local config.schema_version", config["schema_version"], minimum=1) != 1:
        raise ValueError("Unsupported AWS local config schema_version")
    account_id = _require_string("AWS local config.account_id", config["account_id"])
    if _ACCOUNT_RE.fullmatch(account_id) is None:
        raise ValueError("AWS local config.account_id must contain exactly 12 digits")
    region = _require_string("AWS local config.region", config["region"])
    if _REGION_RE.fullmatch(region) is None:
        raise ValueError("AWS local config.region is invalid")
    role_arn = _require_string("AWS local config.role_arn", config["role_arn"])
    expected_prefix = f"arn:aws:iam::{account_id}:role/"
    if not role_arn.startswith(expected_prefix) or role_arn == expected_prefix:
        raise ValueError("AWS local config.role_arn does not match account_id")
    bucket = _require_string("AWS local config.artifact_bucket", config["artifact_bucket"])
    if _BUCKET_RE.fullmatch(bucket) is None:
        raise ValueError("AWS local config.artifact_bucket is invalid")
    _require_posix_relative_path(
        "AWS local config.artifact_root_prefix", config["artifact_root_prefix"]
    )
    repository = _require_string(
        "AWS local config.ecr_repository", config["ecr_repository"]
    )
    if _ECR_REPOSITORY_RE.fullmatch(repository) is None:
        raise ValueError("AWS local config.ecr_repository is invalid")
    for key in ("training_instance_type", "processing_instance_type"):
        value_text = _require_string(f"AWS local config.{key}", config[key])
        if not value_text.startswith("ml."):
            raise ValueError(f"AWS local config.{key} must be a SageMaker ml.* type")
    for key in (
        "training_instance_count",
        "training_volume_size_gb",
        "training_max_runtime_seconds",
        "processing_instance_count",
        "processing_volume_size_gb",
        "processing_max_runtime_seconds",
        "max_concurrent_training_jobs",
    ):
        _require_int(f"AWS local config.{key}", config[key], minimum=1)
    _validate_string_map("AWS local config.tags", config["tags"])
    return config


def load_aws_local_config(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    value, digest = load_canonical_json_object(path, expected_sha256=expected_sha256)
    return validate_aws_local_config(value), digest
