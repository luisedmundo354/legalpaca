"""Pure provenance-complete gate for two determinism-smoke acquisitions.

The gate performs no AWS, network, submission, acquisition, or publication
operation.  Each input path must name a complete acquisition receipt whose
loader recursively revalidates the embedded launch chain and the fixed local
bundle.  Request receipts, artifact roots, and artifact hashes are never
accepted as independent caller-supplied coordinates.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ...retriever.determinism import (
    SMOKE_COMPARISON_PROTOCOL,
    SMOKE_RUN_KIND,
    compare_smoke_scientific_evidence,
)
from ...retriever.determinism_artifacts import (
    DeterminismSmokeArtifactIdentity,
    ValidatedDeterminismSmokeArtifact,
)

from . import aws, manifest, training_artifacts, training_aws


DETERMINISM_GATE_PROTOCOL = "retrieval_cv_two_replica_determinism_gate_v2"
SMOKE_RUN_IDS = ("determinism-smoke-a", "determinism-smoke-b")

_ARTIFACT_IDENTITY_FIELDS = tuple(
    field.name for field in dataclasses.fields(DeterminismSmokeArtifactIdentity)
)
_GATE_KEYS = {
    "schema_version",
    "protocol",
    "plan_sha256",
    "staging_receipt_sha256",
    "request_equivalence",
    "replicas",
    "scientific_comparison",
    "exact_match",
    "receipt_sha256",
}
_REPLICA_KEYS = {
    "run_id",
    "acquisition_receipt_path",
    "receipt_hashes",
    "job",
    "archive",
    "artifact",
}
_RECEIPT_HASH_KEYS = {
    "request_sha256",
    "request_receipt_sha256",
    "preflight_receipt_sha256",
    "submission_receipt_sha256",
    "status_receipt_sha256",
    "terminal_receipt_sha256",
    "acquisition_receipt_sha256",
}
_JOB_KEYS = {"job_name", "job_arn", "model_artifact_s3_uri"}
_ARCHIVE_KEYS = {"size", "sha256", "remote_object"}
_ARTIFACT_KEYS = {
    "bundle_root",
    "artifact_root",
    "artifact_manifest_sha256",
    "file_count",
    "total_size",
    "inventory_sha256",
    "identity",
}
_REMOTE_OBJECT_KEYS = {
    "bucket",
    "key",
    "s3_uri",
    "version_id",
    "size",
    "sha256",
    "etag",
    "last_modified",
    "storage_class",
    "owner_id",
    "multipart_part_count",
    "checksum",
    "encryption",
    "content_type",
    "metadata",
}
_CHECKSUM_KEYS = {"algorithm", "type", "value"}
_ENCRYPTION_KEYS = {"algorithm", "kms_key_id", "bucket_key_enabled"}
_REQUEST_EQUIVALENCE_KEYS = {
    "launch_coordinates",
    "normalized_request_sha256",
    "protocol",
    "request_sha256_by_run",
    "schema_version",
    "user_argv",
    "user_argv_sha256",
}
_LAUNCH_COORDINATE_KEYS = {
    "run_id",
    "training_job_name",
    "s3_output_path",
    "toolkit_job_name",
}
_SCIENTIFIC_COMPARISON_KEYS = {
    "schema_version",
    "protocol",
    "run_kind",
    "scientific_identity_sha256",
    "replicas",
    "exact_match",
    "sha256",
}


def _exact_dict(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be one exact dict")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} fields changed: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one non-empty exact string")
    return value


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be one exact integer >= {minimum}")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _self_hash(payload: Mapping[str, Any]) -> str:
    return _document_sha256(dict(payload))


def _determinism_self_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_run_mapping(value: object, *, name: str) -> dict[str, Any]:
    mapping = _exact_dict(value, set(SMOKE_RUN_IDS), name=name)
    return {run_id: mapping[run_id] for run_id in SMOKE_RUN_IDS}


def _normalized_absolute_path_text(value: object, *, name: str) -> str:
    text = _exact_string(value, name=name)
    path = Path(text)
    if (
        not path.is_absolute()
        or text == "/"
        or text.startswith("//")
        or text != PurePosixPath(text).as_posix()
        or ".." in path.parts
    ):
        raise ValueError(f"{name} must be one normalized absolute POSIX path")
    return text


def _validate_receipt_path(value: object, *, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    text = _normalized_absolute_path_text(value.as_posix(), name=name)
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"{name} must be one existing regular non-symlink file")
    resolved = value.resolve(strict=True)
    if resolved.as_posix() != text:
        raise ValueError(f"{name} must already be its strict-resolved path")
    return resolved


def _validate_request_equivalence(value: object) -> dict[str, Any]:
    receipt = _exact_dict(
        value, _REQUEST_EQUIVALENCE_KEYS, name="request equivalence receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"]
        != training_aws.DETERMINISM_SMOKE_EQUIVALENCE_PROTOCOL
    ):
        raise ValueError("Request equivalence receipt identity changed")
    _sha256(
        receipt["normalized_request_sha256"],
        name="request equivalence normalized_request_sha256",
    )
    _sha256(
        receipt["user_argv_sha256"],
        name="request equivalence user_argv_sha256",
    )
    user_argv = receipt["user_argv"]
    if (
        type(user_argv) is not list
        or not user_argv
        or any(type(argument) is not str or not argument for argument in user_argv)
        or any("replica" in argument.lower() for argument in user_argv)
    ):
        raise ValueError("Request equivalence user argv is invalid")
    if _document_sha256(user_argv) != receipt["user_argv_sha256"]:
        raise ValueError("Request equivalence user argv hash changed")

    request_hashes = _exact_run_mapping(
        receipt["request_sha256_by_run"],
        name="request equivalence request_sha256_by_run",
    )
    for run_id in SMOKE_RUN_IDS:
        _sha256(request_hashes[run_id], name=f"request hash for {run_id}")

    coordinates = receipt["launch_coordinates"]
    if type(coordinates) is not list or len(coordinates) != 2:
        raise ValueError("Request equivalence must contain exactly two launch coordinates")
    for index, (raw, run_id) in enumerate(zip(coordinates, SMOKE_RUN_IDS)):
        coordinate = _exact_dict(
            raw,
            _LAUNCH_COORDINATE_KEYS,
            name=f"launch coordinate {index}",
        )
        if coordinate["run_id"] != run_id:
            raise ValueError("Request equivalence launch coordinates are out of order")
        for field in ("training_job_name", "s3_output_path", "toolkit_job_name"):
            _exact_string(coordinate[field], name=f"launch coordinate {index}.{field}")
        if coordinate["training_job_name"] != coordinate["toolkit_job_name"]:
            raise ValueError("Request equivalence launch job names disagree")
    for field in ("training_job_name", "s3_output_path", "toolkit_job_name"):
        if coordinates[0][field] == coordinates[1][field]:
            raise ValueError(f"Request equivalence launch coordinate did not vary: {field}")
    return copy.deepcopy(receipt)


def _validate_scientific_comparison(value: object) -> dict[str, Any]:
    receipt = _exact_dict(
        value,
        _SCIENTIFIC_COMPARISON_KEYS,
        name="scientific comparison receipt",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["protocol"] != SMOKE_COMPARISON_PROTOCOL
        or receipt["run_kind"] != SMOKE_RUN_KIND
        or type(receipt["replicas"]) is not int
        or receipt["replicas"] != 2
        or type(receipt["exact_match"]) is not bool
        or receipt["exact_match"] is not True
    ):
        raise ValueError("Scientific comparison receipt identity changed")
    _sha256(
        receipt["scientific_identity_sha256"],
        name="scientific comparison identity",
    )
    _sha256(receipt["sha256"], name="scientific comparison self hash")
    payload = {key: receipt[key] for key in receipt if key != "sha256"}
    if _determinism_self_hash(payload) != receipt["sha256"]:
        raise ValueError("Scientific comparison receipt self hash changed")
    return copy.deepcopy(receipt)


def _validate_remote_object(value: object, *, name: str) -> dict[str, Any]:
    remote = _exact_dict(value, _REMOTE_OBJECT_KEYS, name=name)
    for field in (
        "bucket",
        "key",
        "s3_uri",
        "version_id",
        "etag",
        "last_modified",
        "storage_class",
        "owner_id",
        "content_type",
    ):
        _exact_string(remote[field], name=f"{name}.{field}")
    _positive_int(remote["size"], name=f"{name}.size")
    _sha256(remote["sha256"], name=f"{name}.sha256")
    part_count = _positive_int(
        remote["multipart_part_count"],
        name=f"{name}.multipart_part_count",
        minimum=2,
    )
    checksum = _exact_dict(remote["checksum"], _CHECKSUM_KEYS, name=f"{name}.checksum")
    if checksum["algorithm"] != "CRC32" or checksum["type"] != "COMPOSITE":
        raise ValueError(f"{name}.checksum wire contract changed")
    checksum_value = _exact_string(checksum["value"], name=f"{name}.checksum.value")
    if not checksum_value.endswith(f"-{part_count}"):
        raise ValueError(f"{name}.checksum multipart suffix changed")
    encryption = _exact_dict(
        remote["encryption"], _ENCRYPTION_KEYS, name=f"{name}.encryption"
    )
    if (
        encryption["algorithm"] != "aws:kms"
        or type(encryption["bucket_key_enabled"]) is not bool
        or encryption["bucket_key_enabled"] is not True
    ):
        raise ValueError(f"{name}.encryption wire contract changed")
    _exact_string(encryption["kms_key_id"], name=f"{name}.encryption.kms_key_id")
    metadata = remote["metadata"]
    if type(metadata) is not dict or any(
        type(key) is not str or type(nested) is not str
        for key, nested in metadata.items()
    ):
        raise TypeError(f"{name}.metadata must be one exact string mapping")
    if remote["s3_uri"] != f"s3://{remote['bucket']}/{remote['key']}":
        raise ValueError(f"{name}.s3_uri differs from its bucket/key")
    return copy.deepcopy(remote)


def _remote_wire_class(remote: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "bucket": remote["bucket"],
        "storage_class": remote["storage_class"],
        "owner_id": remote["owner_id"],
        "multipart_part_count": remote["multipart_part_count"],
        "checksum": {
            "algorithm": remote["checksum"]["algorithm"],
            "type": remote["checksum"]["type"],
        },
        "encryption": copy.deepcopy(remote["encryption"]),
        "content_type": remote["content_type"],
        "metadata": copy.deepcopy(remote["metadata"]),
    }


def _validate_gate_shape(value: object) -> dict[str, Any]:
    receipt = _exact_dict(value, _GATE_KEYS, name="determinism gate receipt")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 2
        or receipt["protocol"] != DETERMINISM_GATE_PROTOCOL
        or type(receipt["exact_match"]) is not bool
        or receipt["exact_match"] is not True
    ):
        raise ValueError("Determinism gate receipt identity changed")
    _sha256(receipt["plan_sha256"], name="determinism gate plan_sha256")
    _sha256(
        receipt["staging_receipt_sha256"],
        name="determinism gate staging_receipt_sha256",
    )
    equivalence = _validate_request_equivalence(receipt["request_equivalence"])
    comparison = _validate_scientific_comparison(receipt["scientific_comparison"])

    replicas = receipt["replicas"]
    if type(replicas) is not list or len(replicas) != 2:
        raise ValueError("Gate receipt must contain exactly two ordered replicas")
    receipt_paths: list[str] = []
    bundle_roots: list[str] = []
    artifact_roots: list[str] = []
    job_names: list[str] = []
    job_arns: list[str] = []
    model_uris: list[str] = []
    remote_coordinates: list[tuple[str, str, str]] = []
    remote_classes: list[dict[str, Any]] = []
    evidence_hashes: list[str] = []
    for index, (raw, run_id) in enumerate(zip(replicas, SMOKE_RUN_IDS)):
        replica = _exact_dict(raw, _REPLICA_KEYS, name=f"gate replica {index}")
        if replica["run_id"] != run_id:
            raise ValueError("Gate replicas are out of order")
        receipt_path = _normalized_absolute_path_text(
            replica["acquisition_receipt_path"],
            name=f"gate replica {run_id}.acquisition_receipt_path",
        )
        receipt_paths.append(receipt_path)

        hashes = _exact_dict(
            replica["receipt_hashes"],
            _RECEIPT_HASH_KEYS,
            name=f"gate replica {run_id}.receipt_hashes",
        )
        for field in _RECEIPT_HASH_KEYS:
            _sha256(hashes[field], name=f"gate replica {run_id}.{field}")
        if hashes["request_sha256"] != equivalence["request_sha256_by_run"][run_id]:
            raise ValueError("Gate request hash differs from request equivalence")

        job = _exact_dict(replica["job"], _JOB_KEYS, name=f"gate replica {run_id}.job")
        for field in _JOB_KEYS:
            _exact_string(job[field], name=f"gate replica {run_id}.job.{field}")
        coordinate = equivalence["launch_coordinates"][index]
        if (
            job["job_name"] != coordinate["training_job_name"]
            or job["job_name"] != coordinate["toolkit_job_name"]
        ):
            raise ValueError("Gate job differs from request-equivalence coordinates")
        job_names.append(job["job_name"])
        job_arns.append(job["job_arn"])
        model_uris.append(job["model_artifact_s3_uri"])

        archive = _exact_dict(
            replica["archive"], _ARCHIVE_KEYS, name=f"gate replica {run_id}.archive"
        )
        archive_size = _positive_int(
            archive["size"], name=f"gate replica {run_id}.archive.size"
        )
        archive_sha256 = _sha256(
            archive["sha256"], name=f"gate replica {run_id}.archive.sha256"
        )
        remote = _validate_remote_object(
            archive["remote_object"], name=f"gate replica {run_id}.remote_object"
        )
        if (
            remote["size"] != archive_size
            or remote["sha256"] != archive_sha256
            or remote["s3_uri"] != job["model_artifact_s3_uri"]
        ):
            raise ValueError("Gate archive differs from its remote/job coordinates")
        remote_coordinates.append(
            (remote["bucket"], remote["key"], remote["version_id"])
        )
        remote_classes.append(_remote_wire_class(remote))

        artifact = _exact_dict(
            replica["artifact"],
            _ARTIFACT_KEYS,
            name=f"gate replica {run_id}.artifact",
        )
        bundle_root = _normalized_absolute_path_text(
            artifact["bundle_root"], name=f"gate replica {run_id}.bundle_root"
        )
        artifact_root = _normalized_absolute_path_text(
            artifact["artifact_root"], name=f"gate replica {run_id}.artifact_root"
        )
        if Path(receipt_path) != Path(bundle_root) / "acquisition_receipt.json":
            raise ValueError("Gate acquisition receipt is outside its fixed bundle path")
        if Path(artifact_root) != Path(bundle_root) / "artifact":
            raise ValueError("Gate artifact root is outside its fixed bundle path")
        bundle_roots.append(bundle_root)
        artifact_roots.append(artifact_root)
        manifest_sha256 = _sha256(
            artifact["artifact_manifest_sha256"],
            name=f"gate replica {run_id}.artifact_manifest_sha256",
        )
        _positive_int(artifact["file_count"], name=f"gate replica {run_id}.file_count")
        _positive_int(artifact["total_size"], name=f"gate replica {run_id}.total_size")
        _sha256(
            artifact["inventory_sha256"],
            name=f"gate replica {run_id}.inventory_sha256",
        )
        identity = _exact_dict(
            artifact["identity"],
            set(_ARTIFACT_IDENTITY_FIELDS),
            name=f"gate replica {run_id}.identity",
        )
        for field in _ARTIFACT_IDENTITY_FIELDS:
            _sha256(identity[field], name=f"gate replica {run_id}.identity.{field}")
        if identity["artifact_manifest_sha256"] != manifest_sha256:
            raise ValueError("Gate artifact identity differs from its commit marker")
        evidence_hashes.append(identity["scientific_evidence_sha256"])

    for values, name in (
        (receipt_paths, "acquisition receipt paths"),
        (bundle_roots, "bundle roots"),
        (artifact_roots, "artifact roots"),
        (job_names, "job names"),
        (job_arns, "job ARNs"),
        (model_uris, "model artifact URIs"),
        (remote_coordinates, "version-addressed remote objects"),
    ):
        if len(set(values)) != len(SMOKE_RUN_IDS):
            raise ValueError(f"Gate replicas must use two distinct {name}")
    if remote_classes[0] != remote_classes[1]:
        raise ValueError("Gate replica remote-object wire contracts differ")
    if any(
        digest != comparison["scientific_identity_sha256"]
        for digest in evidence_hashes
    ):
        raise ValueError("Artifact evidence identities differ from scientific comparison")

    _sha256(receipt["receipt_sha256"], name="determinism gate self hash")
    payload = {
        key: copy.deepcopy(nested)
        for key, nested in receipt.items()
        if key != "receipt_sha256"
    }
    if _self_hash(payload) != receipt["receipt_sha256"]:
        raise ValueError("Determinism gate receipt self hash changed")
    return copy.deepcopy(receipt)


def _validated_context(
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(training_plan) is not dict:
        raise TypeError("training_plan must be one exact dict")
    plan = manifest.validate_dry_manifest(copy.deepcopy(training_plan))
    if type(staging_receipt) is not dict:
        raise TypeError("staging_receipt must be one exact dict")
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(staging_receipt), training_plan=plan
    )
    return plan, staged


def _load_acquisitions(
    paths_by_run: Mapping[str, Path],
    *,
    plan: Mapping[str, Any],
    staged: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = _exact_run_mapping(
        paths_by_run, name="acquisition_receipt_paths_by_run"
    )
    paths = {
        run_id: _validate_receipt_path(
            supplied[run_id],
            name=f"acquisition_receipt_paths_by_run[{run_id!r}]",
        )
        for run_id in SMOKE_RUN_IDS
    }
    if len(set(paths.values())) != len(SMOKE_RUN_IDS):
        raise ValueError("Determinism replicas must use distinct acquisition receipts")

    acquisitions: dict[str, Any] = {}
    for run_id in SMOKE_RUN_IDS:
        validated = training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
            paths[run_id],
            training_plan=plan,
            staging_receipt=staged,
        )
        required_attributes = (
            "receipt",
            "receipt_path",
            "bundle_root",
            "artifact_root",
            "archive_sha256",
            "archive_size",
            "inventory_sha256",
            "file_count",
            "total_size",
            "remote_object",
            "request_receipt",
            "preflight_receipt",
            "submission_receipt",
            "terminal_receipt",
            "validated_artifact",
        )
        if any(not hasattr(validated, field) for field in required_attributes):
            raise TypeError("Acquisition loader returned an unexpected result type")
        if validated.receipt_path != paths[run_id]:
            raise RuntimeError("Acquisition loader returned a different receipt path")
        if type(validated.receipt) is not dict:
            raise TypeError("Validated acquisition receipt must be one exact dict")
        for name in (
            "request_receipt",
            "preflight_receipt",
            "submission_receipt",
            "terminal_receipt",
        ):
            if type(getattr(validated, name)) is not dict:
                raise TypeError(f"Validated acquisition {name} must be one exact dict")
        chain_run_ids = (
            validated.receipt.get("run_id"),
            validated.request_receipt.get("run_id"),
            validated.preflight_receipt.get("run_id"),
            validated.submission_receipt.get("run_id"),
            validated.terminal_receipt.get("run_id"),
        )
        if any(value != run_id for value in chain_run_ids):
            raise ValueError(f"Acquisition launch chain is not keyed by {run_id}")
        artifact = validated.validated_artifact
        if not isinstance(artifact, ValidatedDeterminismSmokeArtifact):
            raise TypeError("Acquisition loader returned an unexpected artifact type")
        if (
            not isinstance(validated.bundle_root, Path)
            or not isinstance(validated.artifact_root, Path)
            or validated.artifact_root != artifact.root
            or validated.bundle_root / "artifact" != validated.artifact_root
            or validated.bundle_root / "acquisition_receipt.json" != paths[run_id]
        ):
            raise RuntimeError("Acquisition loader returned different local coordinates")
        if (
            type(validated.file_count) is not int
            or validated.file_count != len(artifact.files)
            or type(validated.total_size) is not int
            or validated.total_size != sum(record.size for record in artifact.files)
        ):
            raise RuntimeError("Acquisition loader returned a different artifact inventory")
        remote = _validate_remote_object(
            validated.remote_object,
            name=f"validated acquisition {run_id}.remote_object",
        )
        if (
            validated.archive_size != remote["size"]
            or validated.archive_sha256 != remote["sha256"]
        ):
            raise RuntimeError("Acquisition loader returned different archive coordinates")
        _sha256(validated.inventory_sha256, name=f"acquisition {run_id} inventory")
        acquisitions[run_id] = validated
    return acquisitions


def _build_determinism_gate_receipt(
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    acquisition_receipt_paths_by_run: Mapping[str, Path],
) -> dict[str, Any]:
    plan, staged = _validated_context(training_plan, staging_receipt)
    acquisitions = _load_acquisitions(
        acquisition_receipt_paths_by_run,
        plan=plan,
        staged=staged,
    )
    request_receipts = {
        run_id: acquisitions[run_id].request_receipt for run_id in SMOKE_RUN_IDS
    }
    request_equivalence = training_aws.validate_determinism_smoke_request_equivalence(
        request_receipts[SMOKE_RUN_IDS[0]],
        request_receipts[SMOKE_RUN_IDS[1]],
        training_plan=plan,
        staging_receipt=staged,
    )
    request_equivalence = _validate_request_equivalence(request_equivalence)

    scientific_comparison = compare_smoke_scientific_evidence(
        acquisitions[SMOKE_RUN_IDS[0]].validated_artifact.scientific_evidence,
        acquisitions[SMOKE_RUN_IDS[1]].validated_artifact.scientific_evidence,
    )
    scientific_comparison = _validate_scientific_comparison(scientific_comparison)

    replicas = []
    for run_id in SMOKE_RUN_IDS:
        acquisition = acquisitions[run_id]
        terminal = acquisition.terminal_receipt
        status = terminal.get("status_receipt")
        if type(status) is not dict:
            raise TypeError("Validated terminal receipt must embed one status receipt")
        if (
            terminal.get("terminal_status") != "Completed"
            or terminal.get("succeeded") is not True
        ):
            raise ValueError("Determinism gate requires two successful Completed jobs")
        remote = _validate_remote_object(
            acquisition.remote_object,
            name=f"acquisition {run_id}.remote_object",
        )
        if terminal.get("model_artifact_s3_uri") != remote["s3_uri"]:
            raise ValueError("Terminal and acquisition model-artifact URIs differ")
        identity = dataclasses.asdict(acquisition.validated_artifact.identity)
        replicas.append(
            {
                "run_id": run_id,
                "acquisition_receipt_path": acquisition.receipt_path.as_posix(),
                "receipt_hashes": {
                    "request_sha256": acquisition.request_receipt["request_sha256"],
                    "request_receipt_sha256": _document_sha256(
                        acquisition.request_receipt
                    ),
                    "preflight_receipt_sha256": _document_sha256(
                        acquisition.preflight_receipt
                    ),
                    "submission_receipt_sha256": _document_sha256(
                        acquisition.submission_receipt
                    ),
                    "status_receipt_sha256": _document_sha256(status),
                    "terminal_receipt_sha256": _document_sha256(terminal),
                    "acquisition_receipt_sha256": _document_sha256(
                        acquisition.receipt
                    ),
                },
                "job": {
                    "job_name": terminal["job_name"],
                    "job_arn": terminal["job_arn"],
                    "model_artifact_s3_uri": terminal["model_artifact_s3_uri"],
                },
                "archive": {
                    "size": acquisition.archive_size,
                    "sha256": acquisition.archive_sha256,
                    "remote_object": remote,
                },
                "artifact": {
                    "bundle_root": acquisition.bundle_root.as_posix(),
                    "artifact_root": acquisition.artifact_root.as_posix(),
                    "artifact_manifest_sha256": identity[
                        "artifact_manifest_sha256"
                    ],
                    "file_count": acquisition.file_count,
                    "total_size": acquisition.total_size,
                    "inventory_sha256": acquisition.inventory_sha256,
                    "identity": identity,
                },
            }
        )

    payload = {
        "schema_version": 2,
        "protocol": DETERMINISM_GATE_PROTOCOL,
        "plan_sha256": _document_sha256(plan),
        "staging_receipt_sha256": _document_sha256(staged),
        "request_equivalence": request_equivalence,
        "replicas": replicas,
        "scientific_comparison": scientific_comparison,
        "exact_match": True,
    }
    return _validate_gate_shape(
        {**payload, "receipt_sha256": _self_hash(payload)}
    )


def run_determinism_gate(
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    acquisition_receipt_paths_by_run: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate both complete acquisition chains and exact scientific equality."""

    return _build_determinism_gate_receipt(
        training_plan=training_plan,
        staging_receipt=staging_receipt,
        acquisition_receipt_paths_by_run=acquisition_receipt_paths_by_run,
    )


def validate_determinism_gate_receipt(
    value: object,
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and validate one gate receipt from its recorded acquisitions."""

    supplied = _validate_gate_shape(value)
    paths_by_run = {
        run_id: Path(supplied["replicas"][index]["acquisition_receipt_path"])
        for index, run_id in enumerate(SMOKE_RUN_IDS)
    }
    expected = _build_determinism_gate_receipt(
        training_plan=training_plan,
        staging_receipt=staging_receipt,
        acquisition_receipt_paths_by_run=paths_by_run,
    )
    if aws.canonical_json_bytes(supplied) != aws.canonical_json_bytes(expected):
        raise ValueError("Determinism gate receipt differs from complete revalidation")
    return copy.deepcopy(expected)


__all__: Sequence[str] = (
    "DETERMINISM_GATE_PROTOCOL",
    "SMOKE_RUN_IDS",
    "run_determinism_gate",
    "validate_determinism_gate_receipt",
)
