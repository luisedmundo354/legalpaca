"""Pure two-replica determinism gate for completed retrieval smoke artifacts.

This module performs no AWS, network, submission, or publication operation.
Artifact expectations are constructed exclusively from validated launch inputs
and externally supplied commit-marker digests; artifact metadata never defines
its own expected identity.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from retriever.determinism import (
    SMOKE_COMPARISON_PROTOCOL,
    SMOKE_RUN_KIND,
    compare_smoke_scientific_evidence,
)
from retriever.determinism_artifacts import (
    DeterminismSmokeArtifactExpectation,
    DeterminismSmokeArtifactIdentity,
    ValidatedDeterminismSmokeArtifact,
    validate_determinism_smoke_artifact,
)

from . import aws, manifest, training_aws


DETERMINISM_GATE_PROTOCOL = "retrieval_cv_two_replica_determinism_gate_v1"
SMOKE_RUN_IDS = ("determinism-smoke-a", "determinism-smoke-b")

_ARTIFACT_IDENTITY_FIELDS = tuple(
    field.name for field in dataclasses.fields(DeterminismSmokeArtifactIdentity)
)
_GATE_KEYS = {
    "schema_version",
    "protocol",
    "plan_sha256",
    "staging_receipt_sha256",
    "launch",
    "artifacts",
    "scientific_comparison",
    "exact_match",
    "sha256",
}
_LAUNCH_KEYS = {"request_equivalence", "request_receipts"}
_REQUEST_COORDINATE_KEYS = {"run_id", "request_sha256"}
_ARTIFACT_COORDINATE_KEYS = {
    "run_id",
    "artifact_root",
    "artifact_manifest_sha256",
    "identity",
}
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


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _self_hash(payload: Mapping[str, Any]) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(dict(payload)))


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


def _validate_artifact_root(value: object, *, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    text = value.as_posix()
    if (
        not value.is_absolute()
        or text == "/"
        or text.startswith("//")
        or text != PurePosixPath(text).as_posix()
        or ".." in value.parts
    ):
        raise ValueError(f"{name} must be one normalized absolute path")
    if value.is_symlink() or not value.is_dir():
        raise ValueError(f"{name} must be one existing real directory")
    resolved = value.resolve(strict=True)
    if text != resolved.as_posix():
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
    if aws.sha256_bytes(aws.canonical_json_bytes(user_argv)) != receipt[
        "user_argv_sha256"
    ]:
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
            if type(coordinate[field]) is not str or not coordinate[field]:
                raise TypeError(f"launch coordinate {index}.{field} must be a string")
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


def validate_determinism_gate_receipt(value: object) -> dict[str, Any]:
    """Validate the exact outer receipt schema and every cross-binding."""

    receipt = _exact_dict(value, _GATE_KEYS, name="determinism gate receipt")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
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

    launch = _exact_dict(receipt["launch"], _LAUNCH_KEYS, name="gate launch record")
    equivalence = _validate_request_equivalence(launch["request_equivalence"])
    request_coordinates = launch["request_receipts"]
    if type(request_coordinates) is not list or len(request_coordinates) != 2:
        raise ValueError("Gate launch record must contain exactly two request receipts")
    for index, (raw, run_id) in enumerate(zip(request_coordinates, SMOKE_RUN_IDS)):
        coordinate = _exact_dict(
            raw,
            _REQUEST_COORDINATE_KEYS,
            name=f"gate request receipt {index}",
        )
        if coordinate["run_id"] != run_id:
            raise ValueError("Gate request receipts are out of order")
        request_sha256 = _sha256(
            coordinate["request_sha256"], name=f"gate request hash for {run_id}"
        )
        if equivalence["request_sha256_by_run"][run_id] != request_sha256:
            raise ValueError("Gate request hash differs from request equivalence")

    artifacts = receipt["artifacts"]
    if type(artifacts) is not list or len(artifacts) != 2:
        raise ValueError("Gate receipt must contain exactly two artifact coordinates")
    artifact_evidence_hashes: list[str] = []
    artifact_roots: list[str] = []
    for index, (raw, run_id) in enumerate(zip(artifacts, SMOKE_RUN_IDS)):
        coordinate = _exact_dict(
            raw,
            _ARTIFACT_COORDINATE_KEYS,
            name=f"artifact coordinate {index}",
        )
        if coordinate["run_id"] != run_id:
            raise ValueError("Artifact coordinates are out of order")
        root_text = coordinate["artifact_root"]
        if type(root_text) is not str:
            raise TypeError("Artifact root coordinate must be a string")
        if root_text != Path(root_text).as_posix():
            raise ValueError("Artifact root coordinate is not a normalized POSIX path")
        resolved_root = _validate_artifact_root(
            Path(root_text), name=f"artifact root for {run_id}"
        )
        artifact_roots.append(resolved_root.as_posix())
        artifact_manifest_sha256 = _sha256(
            coordinate["artifact_manifest_sha256"],
            name=f"artifact manifest for {run_id}",
        )
        identity = _exact_dict(
            coordinate["identity"],
            set(_ARTIFACT_IDENTITY_FIELDS),
            name=f"artifact identity for {run_id}",
        )
        for field in _ARTIFACT_IDENTITY_FIELDS:
            _sha256(identity[field], name=f"artifact identity {run_id}.{field}")
        if identity["artifact_manifest_sha256"] != artifact_manifest_sha256:
            raise ValueError("Artifact identity differs from external manifest coordinate")
        artifact_evidence_hashes.append(identity["scientific_evidence_sha256"])
    if len(set(artifact_roots)) != len(SMOKE_RUN_IDS):
        raise ValueError(
            "Determinism gate receipt must contain two distinct artifact roots"
        )

    comparison = _validate_scientific_comparison(receipt["scientific_comparison"])
    if any(
        evidence_sha256 != comparison["scientific_identity_sha256"]
        for evidence_sha256 in artifact_evidence_hashes
    ):
        raise ValueError("Artifact evidence identities differ from scientific comparison")

    _sha256(receipt["sha256"], name="determinism gate self hash")
    payload = {key: receipt[key] for key in receipt if key != "sha256"}
    if _self_hash(payload) != receipt["sha256"]:
        raise ValueError("Determinism gate receipt self hash changed")
    return copy.deepcopy(receipt)


def run_determinism_gate(
    *,
    training_plan: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    request_receipts_by_run: Mapping[str, Mapping[str, Any]],
    artifact_roots_by_run: Mapping[str, Path],
    artifact_manifest_sha256_by_run: Mapping[str, str],
) -> dict[str, Any]:
    """Validate launch equivalence, two artifacts, and exact scientific equality."""

    if not isinstance(training_plan, Mapping):
        raise TypeError("training_plan must be one mapping")
    plan = manifest.validate_dry_manifest(copy.deepcopy(dict(training_plan)))
    if not isinstance(staging_receipt, Mapping):
        raise TypeError("staging_receipt must be one mapping")
    staged = training_aws.validate_training_staging_receipt(
        copy.deepcopy(dict(staging_receipt)), training_plan=plan
    )
    request_receipts = _exact_run_mapping(
        request_receipts_by_run, name="request_receipts_by_run"
    )
    artifact_roots = _exact_run_mapping(
        artifact_roots_by_run, name="artifact_roots_by_run"
    )
    manifest_hashes = _exact_run_mapping(
        artifact_manifest_sha256_by_run,
        name="artifact_manifest_sha256_by_run",
    )

    for run_id in SMOKE_RUN_IDS:
        request_receipt = request_receipts[run_id]
        if type(request_receipt) is not dict or request_receipt.get("run_id") != run_id:
            raise ValueError(f"Request receipt is not keyed by its own run ID: {run_id}")
        artifact_roots[run_id] = _validate_artifact_root(
            artifact_roots[run_id], name=f"artifact_roots_by_run[{run_id!r}]"
        )
        _sha256(
            manifest_hashes[run_id],
            name=f"artifact_manifest_sha256_by_run[{run_id!r}]",
        )
    resolved_artifact_roots = {
        run_id: artifact_roots[run_id].expanduser().resolve(strict=True)
        for run_id in SMOKE_RUN_IDS
    }
    if len(set(resolved_artifact_roots.values())) != len(SMOKE_RUN_IDS):
        raise ValueError(
            "Determinism replicas must use two distinct resolved artifact roots"
        )

    request_equivalence = training_aws.validate_determinism_smoke_request_equivalence(
        request_receipts[SMOKE_RUN_IDS[0]],
        request_receipts[SMOKE_RUN_IDS[1]],
        training_plan=plan,
        staging_receipt=staged,
    )
    request_equivalence = _validate_request_equivalence(request_equivalence)

    plan_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(plan))
    staging_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(staged))
    if any(
        request_receipts[run_id].get("plan_sha256") != plan_sha256
        or request_receipts[run_id].get("staging_receipt_sha256")
        != staging_sha256
        for run_id in SMOKE_RUN_IDS
    ):
        raise ValueError("Request receipt plan/staging coordinates changed")

    source = plan["sources"]
    validated_artifacts: dict[str, ValidatedDeterminismSmokeArtifact] = {}
    for run_id in SMOKE_RUN_IDS:
        expectation = DeterminismSmokeArtifactExpectation(
            artifact_manifest_sha256=manifest_hashes[run_id],
            training_plan_sha256=plan_sha256,
            training_staging_receipt_sha256=staging_sha256,
            source_bundle_name=source["source_bundle_path"],
            source_bundle_size=source["source_bundle_size"],
            source_bundle_sha256=source["source_bundle_sha256"],
            source_bundle_inventory_sha256=source["source_inventory_sha256"],
            source_bundle_commit_epoch=source["commit_epoch"],
        )
        validated = validate_determinism_smoke_artifact(
            artifact_roots[run_id], expectation=expectation
        )
        if not isinstance(validated, ValidatedDeterminismSmokeArtifact):
            raise TypeError("Artifact validator returned an unexpected result type")
        if validated.expectation != expectation:
            raise RuntimeError("Artifact validator returned a different expectation")
        expected_root = resolved_artifact_roots[run_id]
        if validated.root != expected_root:
            raise RuntimeError("Artifact validator returned a different artifact root")
        validated_artifacts[run_id] = validated

    scientific_comparison = compare_smoke_scientific_evidence(
        validated_artifacts[SMOKE_RUN_IDS[0]].scientific_evidence,
        validated_artifacts[SMOKE_RUN_IDS[1]].scientific_evidence,
    )
    scientific_comparison = _validate_scientific_comparison(scientific_comparison)

    artifact_coordinates = []
    for run_id in SMOKE_RUN_IDS:
        validated = validated_artifacts[run_id]
        identity = dataclasses.asdict(validated.identity)
        artifact_coordinates.append(
            {
                "run_id": run_id,
                "artifact_root": validated.root.as_posix(),
                "artifact_manifest_sha256": manifest_hashes[run_id],
                "identity": identity,
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": DETERMINISM_GATE_PROTOCOL,
        "plan_sha256": plan_sha256,
        "staging_receipt_sha256": staging_sha256,
        "launch": {
            "request_equivalence": request_equivalence,
            "request_receipts": [
                {
                    "run_id": run_id,
                    "request_sha256": request_receipts[run_id]["request_sha256"],
                }
                for run_id in SMOKE_RUN_IDS
            ],
        },
        "artifacts": artifact_coordinates,
        "scientific_comparison": scientific_comparison,
        "exact_match": True,
    }
    return validate_determinism_gate_receipt(
        {**payload, "sha256": _self_hash(payload)}
    )


__all__: Sequence[str] = (
    "DETERMINISM_GATE_PROTOCOL",
    "SMOKE_RUN_IDS",
    "run_determinism_gate",
    "validate_determinism_gate_receipt",
)
