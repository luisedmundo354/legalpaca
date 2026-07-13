"""Exact, non-submitting SageMaker training request and input staging.

This module deliberately has no training submission function.  It can stage a
previously absent, versioned set of inputs and render a request only from one
validated training-plan run plus the matching staging receipt.  A later launch
step must reverify the remote versions before it may submit that request.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import aws
from . import config as strict_config
from . import manifest


TRAINING_STAGING_PROTOCOL = "retrieval_cv_training_input_staging_v1"
CONTROLLED_REQUEST_PROTOCOL = "retrieval_cv_controlled_training_request_v1"
DETERMINISM_SMOKE_REQUEST_PROTOCOL = (
    "retrieval_cv_determinism_smoke_training_request_v1"
)
CORRECTED_LEGACY_REQUEST_PROTOCOL = (
    "retrieval_cv_corrected_legacy_diagnostic_training_request_v1"
)
DETERMINISM_SMOKE_EQUIVALENCE_PROTOCOL = (
    "retrieval_cv_determinism_smoke_request_equivalence_v1"
)
TRAINING_TOOLKIT_VERSION = "5.0.0"
TRAINING_TOOLKIT_MAPPING_SHA256 = (
    "3fd30fe8dcb3925d4c31807a916296956e5019c9875249a8299cc25d40aa176f"
)
TRAINING_IMAGE_DIGEST = aws.EXPECTED_TRAINING_IMAGE_DIGEST
TRAINING_IMAGE_URI = (
    "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@"
    + TRAINING_IMAGE_DIGEST
)
TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256 = (
    aws.EXPECTED_TRAINING_RUNTIME_INVENTORY_SHA256
)
BASE_TRAINING_IMAGE_URI = aws.EXPECTED_SDK_TRAINING_IMAGE_DIGEST_URI
TRAINING_IMAGE_CONTRACT_SHA256 = aws.EXPECTED_TRAINING_CONTRACT_SHA256
BOOTSTRAP_PROGRAM = "bootstrap.py"
BOOTSTRAP_SUBMIT_DIRECTORY = "/opt/training_bootstrap"
SOURCE_CHANNEL_DIRECTORY = "/opt/ml/input/data/source"
SNAPSHOT_MANIFEST_SHA256 = (
    "0807d16ba5b49a5e30c8b09b72acef7d8c6326823a850640027cc1363ee446b5"
)
DATASET_MANIFEST_SHA256 = (
    "cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be"
)
TRAINING_TAGS = {
    "Experiment": "arr_retrieval_cv_v1",
    "ManagedBy": "arr-retrieval-cv",
    "Purpose": "controlled-retrieval-training",
}
CONTROLLED_LOGICAL_TO_CLI = {
    "experiment_seed": "experiment-seed",
    "outer_fold": "outer-fold",
    "query_view": "query-view",
    "sampler": "sampler",
}
DETERMINISM_SMOKE_LOGICAL_TO_CLI = {
    "epochs": "epochs",
    "experiment_seed": "experiment-seed",
    "outer_fold": "outer-fold",
    "query_view": "query-view",
    "run_kind": "run-kind",
    "sampler": "sampler",
    "total_optimizer_updates": "total-optimizer-updates",
}
DETERMINISM_SMOKE_LOGICAL_HYPERPARAMETERS = {
    "epochs": 2,
    "experiment_seed": 17,
    "outer_fold": 0,
    "query_view": "structured",
    "run_kind": manifest.SMOKE_KIND,
    "sampler": "global_uniform",
    "total_optimizer_updates": 6,
}
CORRECTED_LEGACY_LOGICAL_TO_CLI = {
    "base_seed": "base-seed",
    "epochs": "epochs",
    "query_view": "query-view",
    "run_kind": "run-kind",
    "total_optimizer_updates": "total-optimizer-updates",
}
CORRECTED_LEGACY_FIXED_LOGICAL_HYPERPARAMETERS = {
    "base_seed": 17,
    "epochs": 20,
    "run_kind": manifest.LEGACY_KIND,
    "total_optimizer_updates": 80,
}
CONTROLLED_QUERY_VIEWS = {"flat_masked", "structured"}
CONTROLLED_SAMPLERS = {"global_uniform", "local_unique"}
CONTROLLED_SEEDS = {17, 29, 43}
_RESERVED_HYPERPARAMETERS = {
    "sagemaker_container_log_level",
    "sagemaker_job_name",
    "sagemaker_mpi_enabled",
    "sagemaker_mpi_num_of_processes_per_host",
    "sagemaker_program",
    "sagemaker_region",
    "sagemaker_submit_directory",
}
_ETAG = re.compile(r'"[0-9a-f]{32}"\Z')

_DATASET_FILES = {
    "cases.jsonl": (
        13_344,
        "313b53fe32be512c7a4a94ecf9a21b718fa1ee50b92b6877a11c1c89289f443f",
    ),
    "corpus.jsonl": (
        1_834_004,
        "f0abc16886727a3c818201fc4888224edf281c3c711b15685d86fd5d63137474",
    ),
    "dataset_manifest.json": (10_823, DATASET_MANIFEST_SHA256),
    "pools/candidates_by_case.json": (
        116_883,
        "75c33c3fa56e7983532f54e3ac2f6969648c9363bb09cd5a1812073c542b3c5f",
    ),
    "pools/candidates_global.json": (
        105_723,
        "39fb2f3360c66ac33cb1aca6cded3f192c15cb06d4494333ebd2a17d1ffc894d",
    ),
    "queries/all.jsonl": (
        11_913_964,
        "bcc6e7573009329f50aaa42a483981e9e30c6e3060984dd840f1c0d7e6f66279",
    ),
}
_SNAPSHOT_FILES = {
    "config.json": (
        1_193,
        "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
    ),
    "model.safetensors": (
        598_635_032,
        "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
    ),
    "special_tokens_map.json": (
        694,
        "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
    ),
    "tokenizer.json": (
        2_132_967,
        "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
    ),
    "tokenizer_config.json": (
        20_810,
        "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
    ),
}


def _exact_keys(value: object, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _require_plain_json(value: object, *, name: str) -> None:
    """Reject Python equality/type tricks before exact canonical comparison."""

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


def _json_scalar(value: object, *, name: str) -> str:
    if value is None or type(value) not in {str, bool, int, float}:
        raise TypeError(f"{name} must be one exact non-null JSON scalar")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> tuple[int, str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Input must be one regular non-symlink file: {path}")
    size = path.stat().st_size
    if size < 1:
        raise ValueError(f"Input file must be non-empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return size, digest.hexdigest()


def _load_json_object(path: Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one regular non-symlink file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 changed: actual={digest}, expected={expected_sha256}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _validate_directory_inventory(
    root: Path,
    expected: Mapping[str, tuple[int, str]],
    *,
    name: str,
) -> None:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{name} must be one real directory: {root}")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"{name} forbids symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"{name} has an unsupported entry: {relative}")
    expected_directories = {
        PurePosixPath(relative).parent.as_posix()
        for relative in expected
        if PurePosixPath(relative).parent.as_posix() != "."
    }
    if actual_files != set(expected) or actual_directories != expected_directories:
        raise ValueError(
            f"{name} inventory changed: files={sorted(actual_files)}, "
            f"directories={sorted(actual_directories)}"
        )
    for relative, (expected_size, expected_digest) in sorted(expected.items()):
        size, digest = _sha256_file(root / relative)
        if size != expected_size or digest != expected_digest:
            raise ValueError(f"{name} file changed: {relative}")


def _validated_training_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("training_plan must be one mapping")
    _require_plain_json(value, name="training_plan")
    plan = manifest.validate_dry_manifest(copy.deepcopy(dict(value)))
    aws.validate_aws_config(
        {"schema_version": 1, **copy.deepcopy(plan["infrastructure"])}
    )
    return plan


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(plan))


def validate_controlled_logical_hyperparameters(value: object) -> dict[str, Any]:
    logical = _exact_keys(
        value,
        set(CONTROLLED_LOGICAL_TO_CLI),
        name="controlled logical hyperparameters",
    )
    if type(logical["outer_fold"]) is not int or logical["outer_fold"] not in range(5):
        raise ValueError("outer_fold must be one exact fold integer 0..4")
    if logical["query_view"] not in CONTROLLED_QUERY_VIEWS:
        raise ValueError("query_view changed")
    if logical["sampler"] not in CONTROLLED_SAMPLERS:
        raise ValueError("sampler changed")
    if (
        type(logical["experiment_seed"]) is not int
        or logical["experiment_seed"] not in CONTROLLED_SEEDS
    ):
        raise ValueError("experiment_seed changed")
    return copy.deepcopy(logical)


def validate_determinism_smoke_logical_hyperparameters(
    value: object,
) -> dict[str, Any]:
    """Validate the one frozen two-epoch distributed determinism cell."""

    logical = _exact_keys(
        value,
        set(DETERMINISM_SMOKE_LOGICAL_TO_CLI),
        name="determinism-smoke logical hyperparameters",
    )
    for name, expected in DETERMINISM_SMOKE_LOGICAL_HYPERPARAMETERS.items():
        actual = logical[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "Determinism-smoke logical hyperparameter changed: "
                f"{name}={actual!r}, expected={expected!r}"
            )
    return copy.deepcopy(logical)


def validate_corrected_legacy_logical_hyperparameters(
    value: object,
) -> dict[str, Any]:
    """Validate one exact corrected-legacy diagnostic schedule."""

    logical = _exact_keys(
        value,
        set(CORRECTED_LEGACY_LOGICAL_TO_CLI),
        name="corrected-legacy logical hyperparameters",
    )
    for name, expected in CORRECTED_LEGACY_FIXED_LOGICAL_HYPERPARAMETERS.items():
        actual = logical[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "Corrected-legacy logical hyperparameter changed: "
                f"{name}={actual!r}, expected={expected!r}"
            )
    if type(logical["query_view"]) is not str or (
        logical["query_view"] not in CONTROLLED_QUERY_VIEWS
    ):
        raise ValueError("Corrected-legacy query_view changed")
    return copy.deepcopy(logical)


def _render_exact_toolkit_hyperparameters(
    *,
    job_name: str,
    region: str,
    logical: Mapping[str, Any],
    logical_to_cli: Mapping[str, str],
) -> dict[str, str]:
    if type(job_name) is not str or aws._JOB_NAME.fullmatch(job_name) is None:
        raise ValueError("Training job name is invalid")
    if region != "us-east-1":
        raise ValueError("Training region changed")
    rendered = {
        logical_to_cli[key]: _json_scalar(
            logical[key], name=f"logical_hyperparameters.{key}"
        )
        for key in sorted(logical_to_cli)
    }
    rendered.update(
        {
            "sagemaker_container_log_level": _json_scalar(
                20, name="sagemaker_container_log_level"
            ),
            "sagemaker_job_name": _json_scalar(job_name, name="sagemaker_job_name"),
            "sagemaker_mpi_enabled": _json_scalar(
                True, name="sagemaker_mpi_enabled"
            ),
            "sagemaker_mpi_num_of_processes_per_host": _json_scalar(
                4, name="sagemaker_mpi_num_of_processes_per_host"
            ),
            "sagemaker_program": _json_scalar(
                BOOTSTRAP_PROGRAM, name="sagemaker_program"
            ),
            "sagemaker_region": _json_scalar(region, name="sagemaker_region"),
            "sagemaker_submit_directory": _json_scalar(
                BOOTSTRAP_SUBMIT_DIRECTORY, name="sagemaker_submit_directory"
            ),
        }
    )
    return {key: rendered[key] for key in sorted(rendered)}


def render_toolkit_hyperparameters(
    *,
    job_name: str,
    region: str,
    logical_hyperparameters: Mapping[str, Any],
) -> dict[str, str]:
    """Render the exact strings consumed by training-toolkit 5.0.0."""

    logical = validate_controlled_logical_hyperparameters(
        dict(logical_hyperparameters)
    )
    if type(job_name) is not str or aws._JOB_NAME.fullmatch(job_name) is None:
        raise ValueError("Controlled training job name is invalid")
    if region != "us-east-1":
        raise ValueError("Controlled training region changed")
    return _render_exact_toolkit_hyperparameters(
        job_name=job_name,
        region=region,
        logical=logical,
        logical_to_cli=CONTROLLED_LOGICAL_TO_CLI,
    )


def render_determinism_smoke_toolkit_hyperparameters(
    *,
    job_name: str,
    region: str,
    logical_hyperparameters: Mapping[str, Any],
) -> dict[str, str]:
    """Render the exact pinned-toolkit strings for one determinism smoke."""

    logical = validate_determinism_smoke_logical_hyperparameters(
        dict(logical_hyperparameters)
    )
    return _render_exact_toolkit_hyperparameters(
        job_name=job_name,
        region=region,
        logical=logical,
        logical_to_cli=DETERMINISM_SMOKE_LOGICAL_TO_CLI,
    )


def render_corrected_legacy_toolkit_hyperparameters(
    *,
    job_name: str,
    region: str,
    logical_hyperparameters: Mapping[str, Any],
) -> dict[str, str]:
    """Render exact pinned-toolkit strings for one corrected diagnostic."""

    logical = validate_corrected_legacy_logical_hyperparameters(
        dict(logical_hyperparameters)
    )
    return _render_exact_toolkit_hyperparameters(
        job_name=job_name,
        region=region,
        logical=logical,
        logical_to_cli=CORRECTED_LEGACY_LOGICAL_TO_CLI,
    )


def toolkit_user_command_arguments(
    hyperparameters: Mapping[str, str],
) -> list[str]:
    """Reproduce the pinned toolkit mapping after exact schema validation."""

    controlled_keys = {
        *CONTROLLED_LOGICAL_TO_CLI.values(),
        *_RESERVED_HYPERPARAMETERS,
    }
    smoke_keys = {
        *DETERMINISM_SMOKE_LOGICAL_TO_CLI.values(),
        *_RESERVED_HYPERPARAMETERS,
    }
    corrected_legacy_keys = {
        *CORRECTED_LEGACY_LOGICAL_TO_CLI.values(),
        *_RESERVED_HYPERPARAMETERS,
    }
    actual_keys = (
        frozenset(hyperparameters) if type(hyperparameters) is dict else frozenset()
    )
    if type(hyperparameters) is not dict or actual_keys not in {
        frozenset(controlled_keys),
        frozenset(smoke_keys),
        frozenset(corrected_legacy_keys),
    }:
        raise ValueError("Rendered training hyperparameter schema changed")
    decoded: dict[str, object] = {}
    for key in sorted(hyperparameters):
        raw_value = hyperparameters[key]
        if type(raw_value) is not str:
            raise TypeError("Toolkit hyperparameters must map strings to strings")
        try:
            value = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Toolkit hyperparameter is not exact JSON: {key}") from error
        if value is None or type(value) not in {str, bool, int, float}:
            raise TypeError(f"Toolkit hyperparameter is not a JSON scalar: {key}")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"Toolkit hyperparameter is non-finite: {key}")
        if _json_scalar(value, name=key) != raw_value:
            raise ValueError(f"Toolkit hyperparameter is not canonically encoded: {key}")
        decoded[key] = value
    if (
        decoded["sagemaker_container_log_level"] != 20
        or type(decoded["sagemaker_container_log_level"]) is not int
        or decoded["sagemaker_mpi_enabled"] is not True
        or decoded["sagemaker_mpi_num_of_processes_per_host"] != 4
        or type(decoded["sagemaker_mpi_num_of_processes_per_host"]) is not int
        or decoded["sagemaker_program"] != BOOTSTRAP_PROGRAM
        or decoded["sagemaker_region"] != "us-east-1"
    ):
        raise ValueError("Pinned toolkit reserved hyperparameters changed")
    if (
        type(decoded["sagemaker_job_name"]) is not str
        or aws._JOB_NAME.fullmatch(decoded["sagemaker_job_name"]) is None
        or decoded["sagemaker_submit_directory"] != BOOTSTRAP_SUBMIT_DIRECTORY
    ):
        raise ValueError("Pinned toolkit job/source hyperparameters changed")
    if actual_keys == frozenset(controlled_keys):
        logical_to_cli = CONTROLLED_LOGICAL_TO_CLI
        logical_validator = validate_controlled_logical_hyperparameters
    elif actual_keys == frozenset(smoke_keys):
        logical_to_cli = DETERMINISM_SMOKE_LOGICAL_TO_CLI
        logical_validator = validate_determinism_smoke_logical_hyperparameters
    else:
        logical_to_cli = CORRECTED_LEGACY_LOGICAL_TO_CLI
        logical_validator = validate_corrected_legacy_logical_hyperparameters
    user_decoded = {
        key: decoded[cli_name] for key, cli_name in logical_to_cli.items()
    }
    logical_validator(user_decoded)
    arguments: list[str] = []
    for key in sorted(set(decoded) - _RESERVED_HYPERPARAMETERS):
        arguments.extend((f"--{key}", str(decoded[key])))
    return arguments


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _channel_coordinates(plan: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    first = plan["controlled_runs"][0]["input_channels"]
    coordinates: dict[str, tuple[str, str]] = {}
    for name in ("base_model", "data"):
        coordinates[name] = strict_config._s3_uri_coordinates(first[name]["s3_uri"])
    return coordinates


def _staging_descriptors(
    plan: Mapping[str, Any],
    *,
    source_bundle_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    snapshot_manifest_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    source_bundle_path = Path(source_bundle_path)
    dataset_dir = Path(dataset_dir)
    base_model_dir = Path(base_model_dir)
    source_size, source_sha256 = _sha256_file(source_bundle_path)
    sources = plan["sources"]
    if (
        source_size != sources["source_bundle_size"]
        or source_sha256 != sources["source_bundle_sha256"]
        or source_bundle_path.name != sources["source_bundle_path"]
    ):
        raise ValueError("Local source bundle differs from the validated training plan")
    _validate_directory_inventory(dataset_dir, _DATASET_FILES, name="Dataset input")
    dataset_manifest = _load_json_object(
        dataset_dir / "dataset_manifest.json",
        expected_sha256=DATASET_MANIFEST_SHA256,
        name="Dataset manifest",
    )
    if (
        dataset_manifest.get("schema_version") != 2
        or type(dataset_manifest.get("schema_version")) is not int
        or dataset_manifest.get("counts", {}).get("cases") != 42
        or dataset_manifest.get("counts", {}).get("queries") != 490
        or dataset_manifest.get("counts", {}).get("passages") != 5_286
    ):
        raise ValueError("Dataset manifest counts/schema changed")
    _validate_directory_inventory(
        base_model_dir, _SNAPSHOT_FILES, name="Base-model input"
    )
    snapshot_manifest = _load_json_object(
        snapshot_manifest_path,
        expected_sha256=SNAPSHOT_MANIFEST_SHA256,
        name="Base-model snapshot manifest",
    )
    expected_snapshot_records = [
        {"path": path, "sha256": digest, "size": size}
        for path, (size, digest) in sorted(_SNAPSHOT_FILES.items())
    ]
    if (
        snapshot_manifest.get("schema_version") != 1
        or type(snapshot_manifest.get("schema_version")) is not int
        or snapshot_manifest.get("manifest_type") != "huggingface_model_snapshot"
        or snapshot_manifest.get("tree_sha256")
        != plan["study"]["model_snapshot_tree_sha256"]
        or snapshot_manifest.get("files") != expected_snapshot_records
    ):
        raise ValueError("Base-model snapshot manifest identity changed")
    if plan["study"]["dataset_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise ValueError("Training plan dataset identity changed")
    channel_coordinates = _channel_coordinates(plan)
    for name, expected_identity in {
        "base_model": plan["study"]["model_snapshot_tree_sha256"],
        "data": DATASET_MANIFEST_SHA256,
    }.items():
        bucket, prefix = channel_coordinates[name]
        if bucket != plan["infrastructure"]["artifact_bucket"]:
            raise ValueError(f"{name} channel bucket changed")
        if not prefix.endswith(expected_identity):
            raise ValueError(f"{name} channel identity suffix changed")

    source_prefix = (
        f"{plan['infrastructure']['artifact_root_prefix']}/training-inputs/"
        f"source-{source_sha256}/"
    )
    source_key = source_prefix + source_bundle_path.name
    prefixes = {
        "base_model": channel_coordinates["base_model"][1] + "/",
        "data": channel_coordinates["data"][1] + "/",
        "source": source_prefix,
    }
    if len(set(prefixes.values())) != 3 or any(
        left.startswith(right) or right.startswith(left)
        for index, left in enumerate(prefixes.values())
        for position, right in enumerate(prefixes.values())
        if index < position
    ):
        raise ValueError("Training staging prefixes overlap")

    descriptors = [
        {
            "group": "source",
            "key": source_key,
            "logical_path": source_bundle_path.name,
            "path": source_bundle_path,
        }
    ]
    for group, root, expected in (
        ("base_model", base_model_dir, _SNAPSHOT_FILES),
        ("data", dataset_dir, _DATASET_FILES),
    ):
        prefix = prefixes[group]
        for relative in sorted(expected):
            descriptors.append(
                {
                    "group": group,
                    "key": prefix + relative,
                    "logical_path": relative,
                    "path": root / relative,
                }
            )
    descriptors.sort(key=lambda record: record["key"])
    return prefixes, descriptors


def _expected_staged_object_identity(
    plan: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[int, str]:
    group = record["group"]
    logical_path = record["logical_path"]
    if group == "source":
        if logical_path != plan["sources"]["source_bundle_path"]:
            raise ValueError("Staged source logical path changed")
        return (
            plan["sources"]["source_bundle_size"],
            plan["sources"]["source_bundle_sha256"],
        )
    if group == "data" and logical_path in _DATASET_FILES:
        return _DATASET_FILES[logical_path]
    if group == "base_model" and logical_path in _SNAPSHOT_FILES:
        return _SNAPSHOT_FILES[logical_path]
    raise ValueError("Staged object group/logical path changed")


def validate_training_staging_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validated_training_plan(training_plan)
    receipt = _exact_keys(
        value,
        {
            "channels",
            "input_contracts",
            "objects",
            "plan_sha256",
            "prefixes",
            "protocol",
            "schema_version",
        },
        name="training staging receipt",
    )
    _require_plain_json(receipt, name="training staging receipt")
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != TRAINING_STAGING_PROTOCOL
        or receipt["plan_sha256"] != _plan_sha256(plan)
    ):
        raise ValueError("Training staging receipt identity changed")
    channels = _exact_keys(
        receipt["channels"],
        {"base_model", "data", "source"},
        name="staged channels",
    )
    bucket = plan["infrastructure"]["artifact_bucket"]
    expected_source_prefix = (
        f"{plan['infrastructure']['artifact_root_prefix']}/training-inputs/"
        f"source-{plan['sources']['source_bundle_sha256']}/"
    )
    expected_channels = {
        name: plan["controlled_runs"][0]["input_channels"][name]["s3_uri"] + "/"
        for name in ("base_model", "data")
    }
    expected_channels["source"] = _s3_uri(bucket, expected_source_prefix)
    if channels != expected_channels:
        raise ValueError("Staged channel URIs differ from the training plan")
    input_contracts = _exact_keys(
        receipt["input_contracts"],
        {
            "dataset_manifest_sha256",
            "model_snapshot_manifest_sha256",
            "model_snapshot_tree_sha256",
        },
        name="staged input contracts",
    )
    if input_contracts != {
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "model_snapshot_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
        "model_snapshot_tree_sha256": plan["study"]["model_snapshot_tree_sha256"],
    }:
        raise ValueError("Staged input contracts changed")
    prefixes = _exact_keys(
        receipt["prefixes"], {"base_model", "data", "source"}, name="staged prefixes"
    )
    for name in ("base_model", "data"):
        expected_bucket, expected_prefix = strict_config._s3_uri_coordinates(
            channels[name].removesuffix("/")
        )
        if expected_bucket != bucket or prefixes[name] != expected_prefix + "/":
            raise ValueError(f"Staged {name} prefix changed")
    if prefixes["source"] != expected_source_prefix:
        raise ValueError("Staged source prefix changed")
    objects = receipt["objects"]
    if type(objects) is not list or len(objects) != 12:
        raise ValueError("Training staging receipt must contain exactly 12 objects")
    observed_keys: list[str] = []
    observed_pairs: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(objects):
        record = _exact_keys(
            raw_record,
            {
                "bucket",
                "etag",
                "group",
                "key",
                "logical_path",
                "schema_version",
                "sha256",
                "size",
                "sse",
                "version_id",
            },
            name=f"staged objects[{index}]",
        )
        group = record["group"]
        if type(group) is not str or group not in prefixes:
            raise ValueError(f"Staged objects[{index}] group changed")
        if (
            record["schema_version"] != 1
            or type(record["schema_version"]) is not int
            or record["bucket"] != bucket
            or type(record["key"]) is not str
            or not record["key"].startswith(prefixes[group])
            or record["key"] != prefixes[group] + record["logical_path"]
            or record["sse"] != "AES256"
            or type(record["version_id"]) is not str
            or not record["version_id"]
            or type(record["etag"]) is not str
            or _ETAG.fullmatch(record["etag"]) is None
        ):
            raise ValueError(f"Staged objects[{index}] identity changed")
        expected_size, expected_sha256 = _expected_staged_object_identity(plan, record)
        if (
            record["size"] != expected_size
            or type(record["size"]) is not int
            or record["sha256"] != expected_sha256
        ):
            raise ValueError(f"Staged objects[{index}] content identity changed")
        pair = (record["group"], record["logical_path"])
        if pair in observed_pairs:
            raise ValueError("Training staging receipt contains duplicate logical objects")
        observed_pairs.add(pair)
        observed_keys.append(record["key"])
    expected_pairs = {
        ("source", plan["sources"]["source_bundle_path"]),
        *(("data", path) for path in _DATASET_FILES),
        *(("base_model", path) for path in _SNAPSHOT_FILES),
    }
    if observed_pairs != expected_pairs or observed_keys != sorted(observed_keys):
        raise ValueError("Training staging receipt object coverage/order changed")
    return copy.deepcopy(receipt)


def _list_prefix_versions(
    s3: object,
    *,
    bucket: str,
    prefix: str,
    expected_bucket_owner: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    versions: list[dict[str, Any]] = []
    delete_markers: list[dict[str, Any]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    while True:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "ExpectedBucketOwner": expected_bucket_owner,
            "MaxKeys": 1000,
            "Prefix": prefix,
        }
        if key_marker is not None:
            request["KeyMarker"] = key_marker
            request["VersionIdMarker"] = version_marker
        response = s3.list_object_versions(**request)
        if (
            type(response) is not dict
            or response.get("Name") != bucket
            or response.get("Prefix") != prefix
            or response.get("MaxKeys") != 1000
            or type(response.get("IsTruncated")) is not bool
        ):
            raise RuntimeError("S3 version listing returned an invalid response identity")
        raw_versions = response.get("Versions")
        raw_delete_markers = response.get("DeleteMarkers")
        if raw_versions is not None and type(raw_versions) is not list:
            raise RuntimeError("S3 version listing Versions must be an optional exact list")
        if raw_delete_markers is not None and type(raw_delete_markers) is not list:
            raise RuntimeError(
                "S3 version listing DeleteMarkers must be an optional exact list"
            )
        if raw_versions is None:
            raw_versions = []
        if raw_delete_markers is None:
            raw_delete_markers = []
        if type(raw_versions) is not list or type(raw_delete_markers) is not list:
            raise RuntimeError("S3 version listing returned an invalid collection")
        versions.extend(raw_versions)
        delete_markers.extend(raw_delete_markers)
        is_truncated = response["IsTruncated"]
        if not is_truncated:
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if (
            type(next_key) is not str
            or not next_key
            or type(next_version) is not str
            or not next_version
            or (next_key, next_version) == (key_marker, version_marker)
        ):
            raise RuntimeError("S3 version listing pagination did not advance")
        key_marker, version_marker = next_key, next_version
    return versions, delete_markers


def _verify_readback_stream(
    stream: object,
    *,
    expected_size: int,
    expected_sha256: str,
    name: str,
) -> None:
    if type(expected_size) is not int or expected_size < 1:
        raise ValueError("Readback expected_size must be one positive exact integer")
    if type(expected_sha256) is not str or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError("Readback expected_sha256 must be lowercase SHA-256")
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if type(chunk) is not bytes:
            raise RuntimeError(f"{name} readback returned non-bytes")
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"{name} readback bytes changed")


def verify_remote_training_staging(
    s3: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    deep_read: bool,
) -> dict[str, Any]:
    """Verify exact current/history state, optionally rehashing versioned bytes."""

    if type(deep_read) is not bool:
        raise TypeError("deep_read must be one exact bool")
    plan = _validated_training_plan(training_plan)
    receipt = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    bucket = plan["infrastructure"]["artifact_bucket"]
    account = plan["infrastructure"]["account_id"]
    expected_by_key = {record["key"]: record for record in receipt["objects"]}
    for prefix in receipt["prefixes"].values():
        versions, delete_markers = _list_prefix_versions(
            s3,
            bucket=bucket,
            prefix=prefix,
            expected_bucket_owner=account,
        )
        expected_keys = {
            key for key in expected_by_key if key.startswith(prefix)
        }
        if delete_markers:
            raise RuntimeError(f"Training staging prefix has delete markers: {prefix}")
        if len(versions) != len(expected_keys):
            raise RuntimeError(f"Training staging prefix version count changed: {prefix}")
        observed_keys: set[str] = set()
        for version in versions:
            if type(version) is not dict:
                raise RuntimeError("S3 version listing record is not an object")
            key = version.get("Key")
            if key not in expected_keys or key in observed_keys:
                raise RuntimeError(f"Training staging prefix contains an extra version: {key}")
            expected = expected_by_key[key]
            if (
                version.get("VersionId") != expected["version_id"]
                or version.get("ETag") != expected["etag"]
                or version.get("Size") != expected["size"]
                or version.get("IsLatest") is not True
            ):
                raise RuntimeError(f"Training staged version identity changed: {key}")
            observed_keys.add(key)
        if observed_keys != expected_keys:
            raise RuntimeError(f"Training staging prefix object coverage changed: {prefix}")
    for record in receipt["objects"]:
        checksum = base64.b64encode(bytes.fromhex(record["sha256"])).decode("ascii")
        head = s3.head_object(
            Bucket=bucket,
            Key=record["key"],
            VersionId=record["version_id"],
            ChecksumMode="ENABLED",
            ExpectedBucketOwner=account,
        )
        if (
            head.get("VersionId") != record["version_id"]
            or head.get("ContentLength") != record["size"]
            or head.get("ChecksumSHA256") != checksum
            or head.get("ServerSideEncryption") != "AES256"
            or head.get("Metadata") != {"sha256": record["sha256"]}
            or head.get("ETag") != record["etag"]
        ):
            raise RuntimeError(f"Training staged object metadata changed: {record['key']}")
        if deep_read:
            stream = s3.get_object(
                Bucket=bucket,
                ExpectedBucketOwner=account,
                Key=record["key"],
                VersionId=record["version_id"],
            )["Body"]
            _verify_readback_stream(
                stream,
                expected_size=record["size"],
                expected_sha256=record["sha256"],
                name=f"Training staged object {record['key']}",
            )
    return receipt


def stage_training_inputs_once(
    s3: object,
    *,
    training_plan: Mapping[str, Any],
    source_bundle_path: Path,
    dataset_dir: Path,
    base_model_dir: Path,
    snapshot_manifest_path: Path,
) -> dict[str, Any]:
    """Stage all 12 exact inputs once; a partial failure permanently taints the prefix."""

    plan = _validated_training_plan(training_plan)
    infrastructure = plan["infrastructure"]
    bucket = infrastructure["artifact_bucket"]
    aws.validate_artifact_bucket(
        s3, bucket=bucket, region=infrastructure["region"]
    )
    prefixes, descriptors = _staging_descriptors(
        plan,
        source_bundle_path=source_bundle_path,
        dataset_dir=dataset_dir,
        base_model_dir=base_model_dir,
        snapshot_manifest_path=snapshot_manifest_path,
    )
    for prefix in sorted(prefixes.values()):
        aws.assert_unused_versioned_prefix(
            s3,
            bucket=bucket,
            prefix=prefix,
            expected_bucket_owner=infrastructure["account_id"],
        )
    objects: list[dict[str, Any]] = []
    for descriptor in descriptors:
        staged = aws.stage_file_once(
            s3,
            source_path=descriptor["path"],
            bucket=bucket,
            key=descriptor["key"],
            expected_bucket_owner=infrastructure["account_id"],
        )
        objects.append(
            {
                **staged,
                "group": descriptor["group"],
                "logical_path": descriptor["logical_path"],
            }
        )
    objects.sort(key=lambda record: record["key"])
    channels = {
        name: plan["controlled_runs"][0]["input_channels"][name]["s3_uri"] + "/"
        for name in ("base_model", "data")
    }
    channels["source"] = _s3_uri(bucket, prefixes["source"])
    receipt = {
        "channels": channels,
        "input_contracts": {
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "model_snapshot_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
            "model_snapshot_tree_sha256": plan["study"][
                "model_snapshot_tree_sha256"
            ],
        },
        "objects": objects,
        "plan_sha256": _plan_sha256(plan),
        "prefixes": prefixes,
        "protocol": TRAINING_STAGING_PROTOCOL,
        "schema_version": 1,
    }
    return verify_remote_training_staging(
        s3,
        training_plan=plan,
        staging_receipt=receipt,
        deep_read=False,
    )


def _training_channel(name: str, s3_uri: str) -> dict[str, Any]:
    if name not in {"base_model", "data", "source"}:
        raise ValueError("Training channel name changed")
    if type(s3_uri) is not str or not s3_uri.endswith("/"):
        raise ValueError("Training S3Prefix URI must end at an exact directory boundary")
    strict_config._s3_uri_coordinates(s3_uri.removesuffix("/"))
    return {
        "ChannelName": name,
        "CompressionType": "None",
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": s3_uri,
            }
        },
        "InputMode": "File",
        "RecordWrapperType": "None",
    }


def _find_controlled_run(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be one non-empty string")
    matches = [run for run in plan["controlled_runs"] if run["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("run_id does not select exactly one controlled plan run")
    return matches[0]


def _find_determinism_smoke_run(
    plan: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be one non-empty string")
    matches = [
        run for run in plan["auxiliary_runs"] if run.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "run_id does not select exactly one determinism-smoke auxiliary run"
        )
    run = matches[0]
    expected_replica_by_run = {
        "determinism-smoke-a": "a",
        "determinism-smoke-b": "b",
    }
    replica_id = expected_replica_by_run.get(run_id)
    if (
        replica_id is None
        or run.get("kind") != manifest.SMOKE_KIND
        or run.get("entry_point") != "train_sm.py"
        or run.get("cell")
        != {
            "outer_fold": 0,
            "query_view": "structured",
            "sampler": "global_uniform",
            "experiment_seed": 17,
        }
        or run.get("expected_artifact_identity")
        != {
            "artifact_type": "determinism_smoke_retriever",
            "schema_version": 1,
            "validator_version": "determinism_smoke_artifact_v1",
        }
        or run.get("launch_metadata") != {"replica_id": replica_id}
    ):
        raise ValueError("Determinism-smoke auxiliary run identity changed")
    logical = validate_determinism_smoke_logical_hyperparameters(
        run.get("hyperparameters")
    )
    if logical != DETERMINISM_SMOKE_LOGICAL_HYPERPARAMETERS:
        raise ValueError("Determinism-smoke run hyperparameters changed")
    return run


def _find_corrected_legacy_run(
    plan: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be one non-empty string")
    matches = [
        run for run in plan["auxiliary_runs"] if run.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "run_id does not select exactly one corrected-legacy auxiliary run"
        )
    run = matches[0]
    expected_view_by_run = {
        "corrected-legacy-flat": "flat_masked",
        "corrected-legacy-structured": "structured",
    }
    query_view = expected_view_by_run.get(run_id)
    if (
        query_view is None
        or run.get("kind") != manifest.LEGACY_KIND
        or run.get("entry_point") != "train_sm.py"
        or run.get("cell") != {"query_view": query_view}
        or run.get("expected_artifact_identity")
        != {
            "artifact_type": "corrected_legacy_diagnostic_retriever",
            "schema_version": 1,
            "validator_version": "corrected_legacy_diagnostic_artifact_v1",
        }
        or run.get("launch_metadata") != {"replica_id": None}
    ):
        raise ValueError("Corrected-legacy auxiliary run identity changed")
    logical = validate_corrected_legacy_logical_hyperparameters(
        run.get("hyperparameters")
    )
    if logical["query_view"] != query_view:
        raise ValueError("Corrected-legacy run query_view changed")
    return run


def render_controlled_training_request(
    *,
    training_plan: Mapping[str, Any],
    run_id: str,
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Render only from a validated plan cell and its exact staged inputs."""

    plan = _validated_training_plan(training_plan)
    staged = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    run = _find_controlled_run(plan, run_id)
    logical = validate_controlled_logical_hyperparameters(run["hyperparameters"])
    if run["cell"] != logical:
        raise ValueError("Controlled run cell and logical hyperparameters differ")
    if run["source_bundle_sha256"] != plan["sources"]["source_bundle_sha256"]:
        raise ValueError("Controlled run source identity changed")
    expected_scientific_channels = {
        name: run["input_channels"][name]["s3_uri"]
        for name in ("base_model", "data")
    }
    expected_staged_channels = {
        name: expected_scientific_channels[name] + "/"
        for name in ("base_model", "data")
    }
    expected_staged_channels["source"] = _s3_uri(
        plan["infrastructure"]["artifact_bucket"], staged["prefixes"]["source"]
    )
    if staged["channels"] != expected_staged_channels:
        raise ValueError("Controlled run channels differ from staged inputs")
    image_uri = plan["study"]["training_image_uri"]
    if (
        image_uri != TRAINING_IMAGE_URI
        or image_uri.rsplit("@", 1)[1] != TRAINING_IMAGE_DIGEST
        or plan["study"]["training_image_inventory_sha256"]
        != TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        or plan["study"]["training_base_image_uri"] != BASE_TRAINING_IMAGE_URI
    ):
        raise ValueError("Controlled training image provenance changed")
    hyperparameters = render_toolkit_hyperparameters(
        job_name=run["job_name"],
        region=plan["infrastructure"]["region"],
        logical_hyperparameters=logical,
    )
    toolkit_user_command_arguments(hyperparameters)
    environment = copy.deepcopy(run["environment"])
    if environment["PYTHONHASHSEED"] != str(logical["experiment_seed"]):
        raise ValueError("Controlled PYTHONHASHSEED differs from the plan cell")
    environment.update(
        {
            "ARR_TRAINING_BASE_IMAGE_URI": BASE_TRAINING_IMAGE_URI,
            "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256": (
                TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
            ),
            "ARR_TRAINING_IMAGE_URI": image_uri,
            "ARR_SOURCE_BUNDLE_NAME": plan["sources"]["source_bundle_path"],
            "ARR_SOURCE_BUNDLE_SHA256": plan["sources"]["source_bundle_sha256"],
            "ARR_SOURCE_BUNDLE_SIZE": str(plan["sources"]["source_bundle_size"]),
            "ARR_SOURCE_COMMIT_EPOCH": str(plan["sources"]["commit_epoch"]),
            "ARR_SOURCE_INVENTORY_SHA256": plan["sources"][
                "source_inventory_sha256"
            ],
            "ARR_TRAINING_PLAN_SHA256": _plan_sha256(plan),
            "ARR_TRAINING_STAGING_RECEIPT_SHA256": aws.sha256_bytes(
                aws.canonical_json_bytes(staged)
            ),
        }
    )
    infrastructure = plan["infrastructure"]
    tags = [
        {"Key": key, "Value": TRAINING_TAGS[key]}
        for key in sorted(TRAINING_TAGS)
    ]
    return {
        "AlgorithmSpecification": {
            "EnableSageMakerMetricsTimeSeries": False,
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "EnableManagedSpotTraining": False,
        "EnableNetworkIsolation": True,
        "Environment": environment,
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            _training_channel("base_model", expected_staged_channels["base_model"]),
            _training_channel("data", expected_staged_channels["data"]),
            _training_channel("source", expected_staged_channels["source"]),
        ],
        "OutputDataConfig": {
            "CompressionType": "GZIP",
            "S3OutputPath": run["output_prefix"],
        },
        "ResourceConfig": {
            "InstanceCount": infrastructure["training_instance_count"],
            "InstanceType": infrastructure["training_instance_type"],
            "VolumeSizeInGB": infrastructure["training_volume_size_gb"],
        },
        "RoleArn": infrastructure["role_arn"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": infrastructure["training_max_runtime_seconds"]
        },
        "Tags": tags,
        "TrainingJobName": run["job_name"],
    }


def render_determinism_smoke_training_request(
    *,
    training_plan: Mapping[str, Any],
    run_id: str,
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Render one non-submitting request for a frozen determinism replica."""

    plan = _validated_training_plan(training_plan)
    staged = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    run = _find_determinism_smoke_run(plan, run_id)
    logical = validate_determinism_smoke_logical_hyperparameters(
        run["hyperparameters"]
    )
    expected_scientific_channels = {
        name: run["input_channels"][name]["s3_uri"]
        for name in ("base_model", "data")
    }
    expected_staged_channels = {
        name: expected_scientific_channels[name] + "/"
        for name in ("base_model", "data")
    }
    expected_staged_channels["source"] = _s3_uri(
        plan["infrastructure"]["artifact_bucket"], staged["prefixes"]["source"]
    )
    if staged["channels"] != expected_staged_channels:
        raise ValueError("Determinism-smoke channels differ from staged inputs")
    image_uri = plan["study"]["training_image_uri"]
    if (
        image_uri != TRAINING_IMAGE_URI
        or image_uri.rsplit("@", 1)[1] != TRAINING_IMAGE_DIGEST
        or plan["study"]["training_image_inventory_sha256"]
        != TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        or plan["study"]["training_base_image_uri"] != BASE_TRAINING_IMAGE_URI
    ):
        raise ValueError("Determinism-smoke training image provenance changed")
    hyperparameters = render_determinism_smoke_toolkit_hyperparameters(
        job_name=run["job_name"],
        region=plan["infrastructure"]["region"],
        logical_hyperparameters=logical,
    )
    user_arguments = toolkit_user_command_arguments(hyperparameters)
    expected_user_arguments = []
    for key in sorted(DETERMINISM_SMOKE_LOGICAL_TO_CLI.values()):
        expected_user_arguments.extend(
            (f"--{key}", str(json.loads(hyperparameters[key])))
        )
    if user_arguments != expected_user_arguments or any(
        "replica" in argument.lower() for argument in user_arguments
    ):
        raise RuntimeError("Determinism-smoke scientific user argv changed")
    environment = copy.deepcopy(run["environment"])
    if environment["PYTHONHASHSEED"] != "17" or any(
        "replica" in name.lower() or "replica" in value.lower()
        for name, value in environment.items()
    ):
        raise ValueError("Determinism-smoke environment contains launch identity")
    environment.update(
        {
            "ARR_TRAINING_BASE_IMAGE_URI": BASE_TRAINING_IMAGE_URI,
            "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256": (
                TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
            ),
            "ARR_TRAINING_IMAGE_URI": image_uri,
            "ARR_SOURCE_BUNDLE_NAME": plan["sources"]["source_bundle_path"],
            "ARR_SOURCE_BUNDLE_SHA256": plan["sources"]["source_bundle_sha256"],
            "ARR_SOURCE_BUNDLE_SIZE": str(plan["sources"]["source_bundle_size"]),
            "ARR_SOURCE_COMMIT_EPOCH": str(plan["sources"]["commit_epoch"]),
            "ARR_SOURCE_INVENTORY_SHA256": plan["sources"][
                "source_inventory_sha256"
            ],
            "ARR_TRAINING_PLAN_SHA256": _plan_sha256(plan),
            "ARR_TRAINING_STAGING_RECEIPT_SHA256": aws.sha256_bytes(
                aws.canonical_json_bytes(staged)
            ),
        }
    )
    if any(
        "replica" in name.lower() or "replica" in value.lower()
        for name, value in environment.items()
    ):
        raise ValueError("Determinism-smoke environment contains launch identity")
    infrastructure = plan["infrastructure"]
    tags = [
        {"Key": key, "Value": TRAINING_TAGS[key]}
        for key in sorted(TRAINING_TAGS)
    ]
    return {
        "AlgorithmSpecification": {
            "EnableSageMakerMetricsTimeSeries": False,
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "EnableManagedSpotTraining": False,
        "EnableNetworkIsolation": True,
        "Environment": environment,
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            _training_channel("base_model", expected_staged_channels["base_model"]),
            _training_channel("data", expected_staged_channels["data"]),
            _training_channel("source", expected_staged_channels["source"]),
        ],
        "OutputDataConfig": {
            "CompressionType": "GZIP",
            "S3OutputPath": run["output_prefix"],
        },
        "ResourceConfig": {
            "InstanceCount": infrastructure["training_instance_count"],
            "InstanceType": infrastructure["training_instance_type"],
            "VolumeSizeInGB": infrastructure["training_volume_size_gb"],
        },
        "RoleArn": infrastructure["role_arn"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": infrastructure["training_max_runtime_seconds"]
        },
        "Tags": tags,
        "TrainingJobName": run["job_name"],
    }


def render_corrected_legacy_training_request(
    *,
    training_plan: Mapping[str, Any],
    run_id: str,
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Render one exact, non-submitting corrected-legacy diagnostic request."""

    plan = _validated_training_plan(training_plan)
    staged = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    run = _find_corrected_legacy_run(plan, run_id)
    logical = validate_corrected_legacy_logical_hyperparameters(
        run["hyperparameters"]
    )
    expected_scientific_channels = {
        name: run["input_channels"][name]["s3_uri"]
        for name in ("base_model", "data")
    }
    expected_staged_channels = {
        name: expected_scientific_channels[name] + "/"
        for name in ("base_model", "data")
    }
    expected_staged_channels["source"] = _s3_uri(
        plan["infrastructure"]["artifact_bucket"], staged["prefixes"]["source"]
    )
    if staged["channels"] != expected_staged_channels:
        raise ValueError("Corrected-legacy channels differ from staged inputs")
    image_uri = plan["study"]["training_image_uri"]
    if (
        image_uri != TRAINING_IMAGE_URI
        or image_uri.rsplit("@", 1)[1] != TRAINING_IMAGE_DIGEST
        or plan["study"]["training_image_inventory_sha256"]
        != TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        or plan["study"]["training_base_image_uri"] != BASE_TRAINING_IMAGE_URI
    ):
        raise ValueError("Corrected-legacy training image provenance changed")
    hyperparameters = render_corrected_legacy_toolkit_hyperparameters(
        job_name=run["job_name"],
        region=plan["infrastructure"]["region"],
        logical_hyperparameters=logical,
    )
    user_arguments = toolkit_user_command_arguments(hyperparameters)
    expected_user_arguments = []
    for key in sorted(CORRECTED_LEGACY_LOGICAL_TO_CLI.values()):
        expected_user_arguments.extend(
            (f"--{key}", str(json.loads(hyperparameters[key])))
        )
    if user_arguments != expected_user_arguments:
        raise RuntimeError("Corrected-legacy scientific user argv changed")
    environment = copy.deepcopy(run["environment"])
    if environment["PYTHONHASHSEED"] != str(logical["base_seed"]):
        raise ValueError("Corrected-legacy PYTHONHASHSEED differs from base_seed")
    environment.update(
        {
            "ARR_TRAINING_BASE_IMAGE_URI": BASE_TRAINING_IMAGE_URI,
            "ARR_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256": (
                TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
            ),
            "ARR_TRAINING_IMAGE_URI": image_uri,
            "ARR_SOURCE_BUNDLE_NAME": plan["sources"]["source_bundle_path"],
            "ARR_SOURCE_BUNDLE_SHA256": plan["sources"]["source_bundle_sha256"],
            "ARR_SOURCE_BUNDLE_SIZE": str(plan["sources"]["source_bundle_size"]),
            "ARR_SOURCE_COMMIT_EPOCH": str(plan["sources"]["commit_epoch"]),
            "ARR_SOURCE_INVENTORY_SHA256": plan["sources"][
                "source_inventory_sha256"
            ],
            "ARR_TRAINING_PLAN_SHA256": _plan_sha256(plan),
            "ARR_TRAINING_RUN_ID": run_id,
            "ARR_TRAINING_STAGING_RECEIPT_SHA256": aws.sha256_bytes(
                aws.canonical_json_bytes(staged)
            ),
        }
    )
    infrastructure = plan["infrastructure"]
    tags = [
        {"Key": key, "Value": TRAINING_TAGS[key]}
        for key in sorted(TRAINING_TAGS)
    ]
    request = {
        "AlgorithmSpecification": {
            "EnableSageMakerMetricsTimeSeries": False,
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "EnableManagedSpotTraining": False,
        "EnableNetworkIsolation": True,
        "Environment": environment,
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            _training_channel("base_model", expected_staged_channels["base_model"]),
            _training_channel("data", expected_staged_channels["data"]),
            _training_channel("source", expected_staged_channels["source"]),
        ],
        "OutputDataConfig": {
            "CompressionType": "GZIP",
            "S3OutputPath": run["output_prefix"],
        },
        "ResourceConfig": {
            "InstanceCount": infrastructure["training_instance_count"],
            "InstanceType": infrastructure["training_instance_type"],
            "VolumeSizeInGB": infrastructure["training_volume_size_gb"],
        },
        "RoleArn": infrastructure["role_arn"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": infrastructure["training_max_runtime_seconds"]
        },
        "Tags": tags,
        "TrainingJobName": run["job_name"],
    }
    request_payload_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(request))
    request["Environment"]["ARR_TRAINING_REQUEST_PAYLOAD_SHA256"] = (
        request_payload_sha256
    )
    return request


def build_controlled_training_request_receipt(
    *,
    training_plan: Mapping[str, Any],
    run_id: str,
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validated_training_plan(training_plan)
    staged = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    request = render_controlled_training_request(
        training_plan=plan,
        run_id=run_id,
        staging_receipt=staged,
    )
    return {
        "plan_sha256": _plan_sha256(plan),
        "protocol": CONTROLLED_REQUEST_PROTOCOL,
        "request": request,
        "request_sha256": aws.sha256_bytes(aws.canonical_json_bytes(request)),
        "run_id": run_id,
        "schema_version": 1,
        "staging_receipt_sha256": aws.sha256_bytes(
            aws.canonical_json_bytes(staged)
        ),
        "toolkit_provenance": {
            "mapping_py_sha256": TRAINING_TOOLKIT_MAPPING_SHA256,
            "sagemaker_training_version": TRAINING_TOOLKIT_VERSION,
        },
    }


def validate_controlled_training_request_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require_plain_json(value, name="controlled training request receipt")
    receipt = _exact_keys(
        value,
        {
            "plan_sha256",
            "protocol",
            "request",
            "request_sha256",
            "run_id",
            "schema_version",
            "staging_receipt_sha256",
            "toolkit_provenance",
        },
        name="controlled training request receipt",
    )
    expected = build_controlled_training_request_receipt(
        training_plan=training_plan,
        run_id=receipt["run_id"],
        staging_receipt=staging_receipt,
    )
    if aws.canonical_json_bytes(receipt) != aws.canonical_json_bytes(expected):
        raise ValueError("Controlled training request receipt differs from re-rendering")
    return copy.deepcopy(receipt)


def build_determinism_smoke_training_request_receipt(
    *,
    training_plan: Mapping[str, Any],
    run_id: str,
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validated_training_plan(training_plan)
    staged = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    request = render_determinism_smoke_training_request(
        training_plan=plan,
        run_id=run_id,
        staging_receipt=staged,
    )
    return {
        "plan_sha256": _plan_sha256(plan),
        "protocol": DETERMINISM_SMOKE_REQUEST_PROTOCOL,
        "request": request,
        "request_sha256": aws.sha256_bytes(aws.canonical_json_bytes(request)),
        "run_id": run_id,
        "schema_version": 1,
        "staging_receipt_sha256": aws.sha256_bytes(
            aws.canonical_json_bytes(staged)
        ),
        "toolkit_provenance": {
            "mapping_py_sha256": TRAINING_TOOLKIT_MAPPING_SHA256,
            "sagemaker_training_version": TRAINING_TOOLKIT_VERSION,
        },
    }


def validate_determinism_smoke_training_request_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require_plain_json(value, name="determinism-smoke training request receipt")
    receipt = _exact_keys(
        value,
        {
            "plan_sha256",
            "protocol",
            "request",
            "request_sha256",
            "run_id",
            "schema_version",
            "staging_receipt_sha256",
            "toolkit_provenance",
        },
        name="determinism-smoke training request receipt",
    )
    expected = build_determinism_smoke_training_request_receipt(
        training_plan=training_plan,
        run_id=receipt["run_id"],
        staging_receipt=staging_receipt,
    )
    if aws.canonical_json_bytes(receipt) != aws.canonical_json_bytes(expected):
        raise ValueError(
            "Determinism-smoke training request receipt differs from re-rendering"
        )
    return copy.deepcopy(receipt)


def build_corrected_legacy_training_request_receipt(
    *,
    training_plan: Mapping[str, Any],
    run_id: str,
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validated_training_plan(training_plan)
    staged = validate_training_staging_receipt(
        dict(staging_receipt), training_plan=plan
    )
    request = render_corrected_legacy_training_request(
        training_plan=plan,
        run_id=run_id,
        staging_receipt=staged,
    )
    return {
        "plan_sha256": _plan_sha256(plan),
        "protocol": CORRECTED_LEGACY_REQUEST_PROTOCOL,
        "request": request,
        "request_sha256": aws.sha256_bytes(aws.canonical_json_bytes(request)),
        "run_id": run_id,
        "schema_version": 1,
        "staging_receipt_sha256": aws.sha256_bytes(
            aws.canonical_json_bytes(staged)
        ),
        "toolkit_provenance": {
            "mapping_py_sha256": TRAINING_TOOLKIT_MAPPING_SHA256,
            "sagemaker_training_version": TRAINING_TOOLKIT_VERSION,
        },
    }


def validate_corrected_legacy_training_request_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require_plain_json(value, name="corrected-legacy training request receipt")
    receipt = _exact_keys(
        value,
        {
            "plan_sha256",
            "protocol",
            "request",
            "request_sha256",
            "run_id",
            "schema_version",
            "staging_receipt_sha256",
            "toolkit_provenance",
        },
        name="corrected-legacy training request receipt",
    )
    expected = build_corrected_legacy_training_request_receipt(
        training_plan=training_plan,
        run_id=receipt["run_id"],
        staging_receipt=staging_receipt,
    )
    if aws.canonical_json_bytes(receipt) != aws.canonical_json_bytes(expected):
        raise ValueError(
            "Corrected-legacy training request receipt differs from re-rendering"
        )
    return copy.deepcopy(receipt)


def _normalized_determinism_smoke_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(request))
    normalized.pop("TrainingJobName")
    normalized["OutputDataConfig"].pop("S3OutputPath")
    normalized["HyperParameters"].pop("sagemaker_job_name")
    return normalized


def validate_determinism_smoke_request_equivalence(
    first_value: object,
    second_value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove two smoke requests differ only in their three launch coordinates."""

    first = validate_determinism_smoke_training_request_receipt(
        first_value,
        training_plan=training_plan,
        staging_receipt=staging_receipt,
    )
    second = validate_determinism_smoke_training_request_receipt(
        second_value,
        training_plan=training_plan,
        staging_receipt=staging_receipt,
    )
    by_run = {first["run_id"]: first, second["run_id"]: second}
    expected_run_ids = {"determinism-smoke-a", "determinism-smoke-b"}
    if set(by_run) != expected_run_ids or len(by_run) != 2:
        raise ValueError("Smoke equivalence requires exactly replicas a and b")
    ordered = [by_run[run_id] for run_id in sorted(expected_run_ids)]
    requests = [receipt["request"] for receipt in ordered]
    launch_coordinates = [
        {
            "run_id": receipt["run_id"],
            "training_job_name": request["TrainingJobName"],
            "s3_output_path": request["OutputDataConfig"]["S3OutputPath"],
            "toolkit_job_name": json.loads(
                request["HyperParameters"]["sagemaker_job_name"]
            ),
        }
        for receipt, request in zip(ordered, requests)
    ]
    for coordinate in launch_coordinates:
        if coordinate["training_job_name"] != coordinate["toolkit_job_name"]:
            raise ValueError("Smoke request job-name launch coordinates disagree")
    for field in ("training_job_name", "s3_output_path", "toolkit_job_name"):
        if launch_coordinates[0][field] == launch_coordinates[1][field]:
            raise ValueError(f"Smoke launch coordinate did not vary: {field}")

    normalized = [
        _normalized_determinism_smoke_request(request) for request in requests
    ]
    normalized_bytes = [aws.canonical_json_bytes(value) for value in normalized]
    if normalized_bytes[0] != normalized_bytes[1]:
        raise ValueError(
            "Determinism-smoke requests differ outside their launch coordinates"
        )
    user_argv = [
        toolkit_user_command_arguments(request["HyperParameters"])
        for request in requests
    ]
    if user_argv[0] != user_argv[1] or any(
        "replica" in argument.lower() for argument in user_argv[0]
    ):
        raise ValueError("Determinism-smoke scientific user argv differs by replica")
    user_argv_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(user_argv[0]))
    return {
        "launch_coordinates": launch_coordinates,
        "normalized_request_sha256": aws.sha256_bytes(normalized_bytes[0]),
        "protocol": DETERMINISM_SMOKE_EQUIVALENCE_PROTOCOL,
        "request_sha256_by_run": {
            receipt["run_id"]: receipt["request_sha256"] for receipt in ordered
        },
        "schema_version": 1,
        "user_argv": user_argv[0],
        "user_argv_sha256": user_argv_sha256,
    }


__all__: Sequence[str] = (
    "BASE_TRAINING_IMAGE_URI",
    "CONTROLLED_REQUEST_PROTOCOL",
    "CORRECTED_LEGACY_REQUEST_PROTOCOL",
    "DETERMINISM_SMOKE_EQUIVALENCE_PROTOCOL",
    "DETERMINISM_SMOKE_REQUEST_PROTOCOL",
    "TRAINING_IMAGE_DIGEST",
    "TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256",
    "TRAINING_STAGING_PROTOCOL",
    "TRAINING_TOOLKIT_MAPPING_SHA256",
    "TRAINING_TOOLKIT_VERSION",
    "build_controlled_training_request_receipt",
    "build_corrected_legacy_training_request_receipt",
    "build_determinism_smoke_training_request_receipt",
    "render_controlled_training_request",
    "render_corrected_legacy_toolkit_hyperparameters",
    "render_corrected_legacy_training_request",
    "render_determinism_smoke_toolkit_hyperparameters",
    "render_determinism_smoke_training_request",
    "render_toolkit_hyperparameters",
    "stage_training_inputs_once",
    "toolkit_user_command_arguments",
    "validate_controlled_logical_hyperparameters",
    "validate_controlled_training_request_receipt",
    "validate_corrected_legacy_logical_hyperparameters",
    "validate_corrected_legacy_training_request_receipt",
    "validate_determinism_smoke_logical_hyperparameters",
    "validate_determinism_smoke_request_equivalence",
    "validate_determinism_smoke_training_request_receipt",
    "validate_training_staging_receipt",
    "verify_remote_training_staging",
)
