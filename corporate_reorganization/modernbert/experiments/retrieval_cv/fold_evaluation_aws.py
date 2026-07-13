"""Exact host control plane for Phase-2 retrieval-fold evaluation.

Phase 2 consumes only evidence sealed by :mod:`fold_processing_aws`.  It
renders one immutable control bundle, stages that bundle once, re-derives the
measured storage proof, and launches one network-isolated SageMaker Processing
job.  Every write is absent-only; there is no retry, adoption, reconciliation,
fallback, overwrite, or cleanup path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ...retriever.artifacts import CONTROLLED_ARTIFACT_PROTOCOL
from ...retriever.baseline_artifacts import (
    E5_MODEL_ID,
    E5_REVISION,
    E5_SNAPSHOT_MANIFEST_SHA256,
    E5_SNAPSHOT_TREE_SHA256,
    FIXED_BASE_ARTIFACT_PROTOCOL,
    FIXED_BASE_SEED,
    MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
    MODERNBERT_SNAPSHOT_TREE_SHA256,
)
from ...retriever.bm25 import (
    ANSERINI_JAR_SHA256,
    BM25_B,
    BM25_K1,
    BM25_RUNTIME_PROTOCOL,
    PYJNIUS_VERSION,
    PYSERINI_VERSION,
)
from ...retriever.e5_pack_artifact import E5_PACK_ARTIFACT_PROTOCOL
from ...retriever.evaluator import (
    BM25_SYSTEM_TYPE,
    COMPLETE_EVALUATION_PLAN_SCHEMA_VERSION,
    COMPLETE_LOCAL_BINDINGS_SCHEMA_VERSION,
    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
    E5_SYSTEM_TYPE,
    FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
    FIXED_BASE_SYSTEM_TYPE,
    _validate_complete_evaluation_plan,
)
from ...retriever.query_packing import FOCUS_PRESERVING_PACK_PROTOCOL
from . import aws
from . import config as strict_config
from . import controlled_supervisor
from . import fold_processing_aws as phase1
from . import training_aws


PHASE2_CONTROL_BUNDLE_PROTOCOL = "retrieval_cv_fold_evaluation_controls_v1"
PHASE2_CONTROL_STAGING_PROTOCOL = "retrieval_cv_fold_evaluation_control_staging_v1"
PHASE2_CONTROL_STAGING_INTENT_PROTOCOL = (
    "retrieval_cv_fold_evaluation_control_staging_intent_v1"
)
PHASE2_PREFLIGHT_PROTOCOL = "retrieval_cv_fold_evaluation_preflight_v1"
PHASE2_SUBMISSION_PROTOCOL = "retrieval_cv_fold_evaluation_submission_v1"
PHASE2_TERMINAL_PROTOCOL = "retrieval_cv_fold_evaluation_terminal_v1"
PHASE2_ACQUISITION_PROTOCOL = "retrieval_cv_fold_evaluation_acquisition_v1"
PHASE2_CONTROL_MANIFEST_NAME = "phase2_control_staging_manifest.json"
PHASE2_PROCESSING_MAX_RUNTIME_SECONDS = 86_400
PHASE2_DEVICE = "cuda:0"
PHASE2_OVERLAY_PUBLICATION_PROTOCOL = (
    "immutable_ecr_fold_phase2_evaluation_image_publication_v1"
)
PHASE2_OVERLAY_IMAGE_DIGEST = (
    "sha256:b7306c60c4104154d7a3e0d373d2114ab18e2f1e7e2881d81e8c806e1a5cd57f"
)
PHASE2_OVERLAY_CONFIG_DIGEST = (
    "sha256:58394c4bb1be62a1bc2715e920e034a3b69a9311ffd45ea31122dbe15f8001fe"
)
PHASE2_OVERLAY_BUILD_IDENTITY = (
    "bc9c10989af547247f1282d458608e855f67ba72ca296b1ea1bf31d820940a0a"
)
PHASE2_OVERLAY_FILES_IDENTITY = (
    "bfaf1390dd4d41fe181cfca7850ed3708f586943227bfba12c3d739c3bda010c"
)
PHASE2_OVERLAY_LOCAL_IMAGE_IDENTITY = (
    "acec7e5901e73fcb2d5c9642b99db7fb15794473ffd3c25786d0ea0b33aa5da8"
)
PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256 = (
    "4bee17b1ea26f7ef9a3800ba4f67d5e36962e2c44f2256088ee2bb52e550843b"
)
PHASE2_OVERLAY_CONTENT_TAG = (
    "fold-phase2-build-sha256-"
    "bc9c10989af547247f1282d458608e855f67ba72ca296b1ea1bf31d820940a0a"
)

CONTROL_FILE_NAMES = (
    "e5_snapshot.json",
    "evaluation_baselines.json",
    "evaluation_plan.json",
    "experiment.json",
    "folds.json",
    "local_bindings.json",
)
GENERATED_CONTROL_FILE_NAMES = ("evaluation_plan.json", "local_bindings.json")
PHASE2_OUTPUT_PATHS = (
    "evaluation/artifact_manifest.json",
    "evaluation/evaluation_config.json",
    "evaluation/rankings.jsonl",
    "evaluation/results.json",
    "evidence/artifact_manifest.json",
    "evidence/materialization_receipt.json",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ETAG = re.compile(r'"[0-9a-f]{32}(?:-[1-9][0-9]*)?"\Z')
_JOB_NAME = re.compile(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}\Z")

_CONTROL_BUNDLE_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "completed_fold_evidence_sha256",
    "archive_copy_receipt_sha256",
    "static_staging_receipt_sha256",
    "inventory_acquisition_receipt_sha256",
    "phase1_overlay_publication_sha256",
    "phase2_overlay_publication_sha256",
    "phase2_runtime_identity_sha256",
    "files",
    "evaluation_plan_sha256",
    "local_bindings_sha256",
    "receipt_sha256",
}
_CONTROL_STAGING_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "control_bundle_receipt_sha256",
    "destination_prefix",
    "input_prefix",
    "files",
    "manifest_object",
    "receipt_sha256",
}
_PREFLIGHT_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "account_id",
    "region",
    "caller_arn",
    "job_name",
    "output_prefix",
    "image_uri",
    "completed_fold_evidence_sha256",
    "archive_copy_receipt_sha256",
    "static_staging_receipt_sha256",
    "inventory_acquisition_receipt_sha256",
    "control_bundle_receipt_sha256",
    "control_staging_receipt_sha256",
    "storage_proof_sha256",
    "archive_verification",
    "static_verification",
    "control_verification",
    "phase1_output_verification",
    "request",
    "request_sha256",
    "sdk_versions",
    "processing_quota",
    "receipt_sha256",
}
_SUBMISSION_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "job_name",
    "job_arn",
    "preflight_receipt_sha256",
    "request_sha256",
    "receipt_sha256",
}
_TERMINAL_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "job_name",
    "job_arn",
    "preflight_receipt_sha256",
    "submission_receipt_sha256",
    "request_sha256",
    "status",
    "failure_reason",
    "processing_start_time",
    "processing_end_time",
    "processing_time_microseconds",
    "exit_message",
    "receipt_sha256",
}
_ACQUISITION_KEYS = {
    "schema_version",
    "protocol",
    "outer_fold",
    "terminal_receipt_sha256",
    "control_bundle_receipt_sha256",
    "output_prefix",
    "remote_objects",
    "files",
    "evaluation_artifact_manifest_sha256",
    "materialization_artifact_manifest_sha256",
    "receipt_sha256",
}


def _canonical_bytes(value: object) -> bytes:
    return aws.canonical_json_bytes(value)


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(_canonical_bytes(value))


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in payload:
        raise ValueError("Receipt payload already contains receipt_sha256")
    result = copy.deepcopy(dict(payload))
    result["receipt_sha256"] = _document_sha256(payload)
    return result


def _exact_object(value: object, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = sorted(value) if type(value) is dict else type(value).__name__
        raise ValueError(f"{name} schema changed: {actual}")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _normalized_utc_datetime(value: object, *, name: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{name} must be one normalized UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"{name} must be one normalized UTC timestamp") from error
    if phase1._normalize_datetime(parsed, name=name) != value:
        raise ValueError(f"{name} must be one normalized UTC timestamp")
    return parsed


def _validate_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    actual = _exact_sha256(value["receipt_sha256"], name=f"{name}.receipt_sha256")
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != "receipt_sha256"}
    if actual != _document_sha256(payload):
        raise ValueError(f"{name} self-hash changed")


def _read_exact_json(payload: bytes, *, compact: bool, name: str) -> dict[str, Any]:
    value = _read_strict_json(payload, name=name)
    expected = _canonical_bytes(value) if compact else strict_config.canonical_json_bytes(value)
    if payload != expected:
        raise ValueError(f"{name} is not in its exact canonical representation")
    return value


def _read_strict_json(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"{name} contains non-finite number {value!r}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain one object")
    return value


def _read_regular(path: Path, *, name: str) -> bytes:
    path = phase1._canonical_absolute(Path(path), name=name)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{name} must be one singly-linked regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_nlink,
        ):
            raise RuntimeError(f"{name} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_phase2_overlay_publication(
    value: object,
    *,
    account_id: str,
    region: str,
) -> dict[str, Any]:
    publication = _exact_object(
        value,
        {
            "content_tag",
            "identity",
            "manifest_digest",
            "media_type",
            "protocol",
            "raw_manifest_sha256",
            "remote_digest_uri",
            "remote_tag_uri",
        },
        name="Phase-2 fold overlay publication",
    )
    identity = _exact_object(
        publication["identity"],
        {
            "build_context_files_sha256",
            "build_context_identity_sha256",
            "config_digest",
            "image_digest",
            "image_runtime_identity",
            "local_image_identity_sha256",
            "manifest_media_type",
            "offline_smoke_sha256",
        },
        name="Phase-2 fold overlay publication identity",
    )
    runtime_identity = identity["image_runtime_identity"]
    if type(runtime_identity) is not dict or not runtime_identity:
        raise ValueError("Phase-2 runtime identity must be one non-empty object")
    try:
        runtime_sha256 = hashlib.sha256(_canonical_bytes(runtime_identity)[:-1]).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError("Phase-2 runtime identity is not strict JSON") from error
    repository = f"{account_id}.dkr.ecr.{region}.amazonaws.com/arr-retrieval-eval"
    expected_digest_uri = f"{repository}@{PHASE2_OVERLAY_IMAGE_DIGEST}"
    expected_tag_uri = f"{repository}:{PHASE2_OVERLAY_CONTENT_TAG}"
    if (
        publication["protocol"] != PHASE2_OVERLAY_PUBLICATION_PROTOCOL
        or publication["content_tag"] != PHASE2_OVERLAY_CONTENT_TAG
        or publication["manifest_digest"] != PHASE2_OVERLAY_IMAGE_DIGEST
        or publication["raw_manifest_sha256"]
        != PHASE2_OVERLAY_IMAGE_DIGEST.removeprefix("sha256:")
        or publication["remote_digest_uri"] != expected_digest_uri
        or publication["remote_tag_uri"] != expected_tag_uri
        or publication["media_type"] != aws.ECR_MEDIA_TYPE
        or identity["build_context_files_sha256"]
        != PHASE2_OVERLAY_FILES_IDENTITY
        or identity["build_context_identity_sha256"]
        != PHASE2_OVERLAY_BUILD_IDENTITY
        or identity["config_digest"] != PHASE2_OVERLAY_CONFIG_DIGEST
        or identity["image_digest"] != PHASE2_OVERLAY_IMAGE_DIGEST
        or identity["local_image_identity_sha256"]
        != PHASE2_OVERLAY_LOCAL_IMAGE_IDENTITY
        or identity["manifest_media_type"] != aws.ECR_MEDIA_TYPE
        or identity["offline_smoke_sha256"]
        != PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256
        or runtime_sha256 != PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256
        or runtime_identity.get("runtime_identity_protocol")
        != "arr_retrieval_fold_runtime_identity_v2"
    ):
        raise ValueError("Phase-2 fold overlay publication identity changed")
    return copy.deepcopy(publication)


def _validate_context(
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    archive = phase1.validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt), completed_fold_evidence=completed
    )
    static = phase1.validate_static_evaluation_staging_receipt(
        copy.deepcopy(static_staging_receipt), completed_fold_evidence=completed
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    phase1_publication = phase1._validate_overlay_publication(
        copy.deepcopy(phase1_overlay_publication_receipt),
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    phase2_publication = _validate_phase2_overlay_publication(
        copy.deepcopy(phase2_overlay_publication_receipt),
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    preflight = phase1.validate_fold_inventory_preflight_receipt(
        copy.deepcopy(phase1_preflight_receipt),
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        overlay_publication_receipt=phase1_publication,
    )
    submission = phase1.validate_fold_inventory_submission_receipt(
        copy.deepcopy(phase1_submission_receipt),
        preflight_receipt=preflight,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        overlay_publication_receipt=phase1_publication,
    )
    terminal = phase1.validate_fold_inventory_terminal_receipt(
        copy.deepcopy(phase1_terminal_receipt),
        preflight_receipt=preflight,
        submission_receipt=submission,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        overlay_publication_receipt=phase1_publication,
    )
    acquisition = phase1.validate_fold_inventory_acquisition_receipt(
        copy.deepcopy(phase1_acquisition_receipt),
        terminal_receipt=terminal,
        preflight_receipt=preflight,
        submission_receipt=submission,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        overlay_publication_receipt=phase1_publication,
    )
    return (
        completed,
        archive,
        static,
        phase1_publication,
        phase2_publication,
        preflight,
        submission,
        terminal,
        acquisition,
    )


def _baseline_systems() -> list[dict[str, Any]]:
    return [
        {
            "system_id": "bm25_flat_plain",
            "system_type": BM25_SYSTEM_TYPE,
            "query_view": "flat_plain",
            "expectation": {
                "baseline_config_sha256": phase1.EXPECTED_BASELINE_CONFIG_SHA256,
                "runtime_protocol": BM25_RUNTIME_PROTOCOL,
                "pyserini_version": PYSERINI_VERSION,
                "pyjnius_version": PYJNIUS_VERSION,
                "anserini_jar_sha256": ANSERINI_JAR_SHA256,
                "k1": BM25_K1,
                "b": BM25_B,
            },
        },
        {
            "system_id": "e5_base_v2_flat_plain",
            "system_type": E5_SYSTEM_TYPE,
            "query_view": "flat_plain",
            "expectation": {
                "baseline_config_sha256": phase1.EXPECTED_BASELINE_CONFIG_SHA256,
                "model_id": E5_MODEL_ID,
                "revision": E5_REVISION,
                "snapshot_manifest_sha256": E5_SNAPSHOT_MANIFEST_SHA256,
                "snapshot_tree_sha256": E5_SNAPSHOT_TREE_SHA256,
                "pack_artifact_protocol": E5_PACK_ARTIFACT_PROTOCOL,
                "pack_manifest_sha256": phase1.EXPECTED_E5_PACK_MANIFEST_SHA256,
                "packed_query_inventory_sha256": phase1.EXPECTED_E5_PACK_INVENTORY_SHA256,
                "packing_protocol": FOCUS_PRESERVING_PACK_PROTOCOL,
                "weight_dtype": "float32",
                "attention_implementation": "eager",
                "pooling": "attention_masked_mean_then_l2_normalize_v1",
                "max_positions": 512,
                "max_passage_tokens": 500,
                "passage_truncation": "right",
                "token_type_ids": "explicit_all_zero",
            },
        },
        {
            "system_id": "modernbert_base_flat_masked",
            "system_type": FIXED_BASE_SYSTEM_TYPE,
            "query_view": "flat_masked",
            "expectation": {
                "baseline_config_sha256": phase1.EXPECTED_BASELINE_CONFIG_SHA256,
                "artifact_manifest_sha256": phase1.EXPECTED_FIXED_BASE_ARTIFACT_MANIFEST_SHA256,
                "model_artifact_protocol": FIXED_BASE_ARTIFACT_PROTOCOL,
                "fixed_initialization_seed": FIXED_BASE_SEED,
                "model_sha256": phase1.EXPECTED_FIXED_BASE_MODEL_SHA256,
                "new_embedding_rows_sha256": phase1.EXPECTED_FIXED_BASE_NEW_ROWS_SHA256,
                "state_key_sha256": phase1.EXPECTED_FIXED_BASE_STATE_KEYS_SHA256,
                "snapshot_manifest_sha256": MODERNBERT_SNAPSHOT_MANIFEST_SHA256,
                "snapshot_tree_sha256": MODERNBERT_SNAPSHOT_TREE_SHA256,
                "weight_dtype": "bfloat16",
            },
        },
    ]


def _phase2_execution_runtime_identity(
    *, publication: Mapping[str, Any]
) -> dict[str, Any]:
    portable = publication["identity"]["image_runtime_identity"]
    if type(portable) is not dict or not portable:
        raise ValueError("Phase-2 portable image runtime identity is invalid")
    reserved = {"device", "image_uri"}.intersection(portable)
    if reserved:
        raise ValueError(
            "Phase-2 portable image runtime identity contains reserved execution "
            f"fields: {sorted(reserved)}"
        )
    image_uri = publication["remote_digest_uri"]
    if type(image_uri) is not str or not image_uri:
        raise ValueError("Phase-2 execution image URI is invalid")
    return {
        "device": PHASE2_DEVICE,
        "image_uri": image_uri,
        **copy.deepcopy(portable),
    }


def _render_controls(
    *,
    completed: Mapping[str, Any],
    publication: Mapping[str, Any],
    archive_inventory: Mapping[str, Any],
    bm25_storage: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = completed["source_bundle"]
    controlled: list[dict[str, Any]] = []
    archive_system_ids = [
        record["system_id"] for record in archive_inventory["systems"]
    ]
    if len(archive_system_ids) != 12 or len(set(archive_system_ids)) != 12:
        raise ValueError("Phase-2 archive system inventory changed")
    for record in archive_inventory["systems"]:
        cell = record["cell"]
        controlled.append(
            {
                "system_id": record["system_id"],
                "system_type": CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                "query_view": cell["query_view"],
                "expectation": {
                    "artifact_manifest_sha256": record["archive_evidence"]["artifact"][
                        "artifact_manifest_sha256"
                    ],
                    "training_plan_sha256": completed["training_plan_sha256"],
                    "training_staging_receipt_sha256": completed[
                        "training_staging_receipt_sha256"
                    ],
                    "source_bundle_name": source["name"],
                    "source_bundle_size": source["size"],
                    "source_bundle_sha256": source["sha256"],
                    "source_bundle_inventory_sha256": source["inventory_sha256"],
                    "source_bundle_commit_epoch": source["commit_epoch"],
                    "experiment_id": "arr_retrieval_cv_v1",
                    "outer_fold": completed["outer_fold"],
                    "query_view": cell["query_view"],
                    "sampler": cell["sampler"],
                    "experiment_seed": cell["experiment_seed"],
                    "dataset_manifest_sha256": bm25_storage[
                        "dataset_manifest_sha256"
                    ],
                    "fold_manifest_sha256": bm25_storage["fold_manifest_sha256"],
                    "passage_index_sha256": bm25_storage["passage_index_sha256"],
                    "model_artifact_protocol": CONTROLLED_ARTIFACT_PROTOCOL,
                },
            }
        )
    systems = sorted([*_baseline_systems(), *controlled], key=lambda row: row["system_id"])
    planned_controlled_ids = [
        record["system_id"]
        for record in systems
        if record["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
    ]
    if planned_controlled_ids != archive_system_ids:
        raise ValueError("Phase-2 plan/archive controlled-system order changed")
    plan = {
        "schema_version": COMPLETE_EVALUATION_PLAN_SCHEMA_VERSION,
        "experiment_id": "arr_retrieval_cv_v1",
        "outer_fold": completed["outer_fold"],
        "role": "test",
        "experiment_config_sha256": phase1.EXPECTED_EXPERIMENT_CONFIG_SHA256,
        "dataset_manifest_sha256": bm25_storage["dataset_manifest_sha256"],
        "fold_manifest_sha256": bm25_storage["fold_manifest_sha256"],
        "passage_index_sha256": bm25_storage["passage_index_sha256"],
        "baseline_config_sha256": phase1.EXPECTED_BASELINE_CONFIG_SHA256,
        "image_contract_sha256": FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
        "image_uri": publication["remote_digest_uri"],
        "case_ids": copy.deepcopy(bm25_storage["case_ids"]),
        "query_count": bm25_storage["query_count"],
        "passage_count": bm25_storage["passage_count"],
        "controlled_max_len_query": 4_096,
        "controlled_max_len_passage": 500,
        "e5_max_len_passage": 500,
        "query_batch_size": 4,
        "passage_batch_size": 38,
        "runtime_identity": _phase2_execution_runtime_identity(
            publication=publication
        ),
        "systems": systems,
    }
    plan_sha256 = hashlib.sha256(_canonical_bytes(plan)).hexdigest()
    identity, case_ids, validated_systems = _validate_complete_evaluation_plan(
        copy.deepcopy(plan),
        evaluation_plan_sha256=plan_sha256,
        expected_image_contract_sha256=FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
    )
    if (
        identity.outer_fold != completed["outer_fold"]
        or tuple(plan["case_ids"]) != case_ids
        or [row["system_id"] for row in validated_systems]
        != [row["system_id"] for row in systems]
    ):
        raise RuntimeError("Rendered complete evaluation plan changed on validation")

    local_systems: list[dict[str, Any]] = []
    for system in systems:
        system_id = system["system_id"]
        system_type = system["system_type"]
        if system_type == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE:
            record = {
                "system_id": system_id,
                "artifact_dir": f"/opt/ml/processing/work/materialized/{system_id}",
            }
        elif system_type == FIXED_BASE_SYSTEM_TYPE:
            record = {
                "system_id": system_id,
                "artifact_dir": "/opt/ml/processing/input/fixed-base",
            }
        elif system_type == E5_SYSTEM_TYPE:
            record = {
                "system_id": system_id,
                "snapshot_dir": "/opt/ml/processing/input/e5-snapshot",
                "snapshot_manifest_path": "/opt/ml/processing/input/control/e5_snapshot.json",
                "pack_artifact_dir": "/opt/ml/processing/input/e5-pack",
            }
        elif system_type == BM25_SYSTEM_TYPE:
            record = {"system_id": system_id}
        else:
            raise RuntimeError(f"Unexpected Phase-2 system type: {system_type}")
        local_systems.append(record)
    bindings = {
        "schema_version": COMPLETE_LOCAL_BINDINGS_SCHEMA_VERSION,
        "dataset_dir": "/opt/ml/processing/input/dataset",
        "fold_manifest_path": "/opt/ml/processing/input/control/folds.json",
        "experiment_config_path": "/opt/ml/processing/input/control/experiment.json",
        "baseline_config_path": "/opt/ml/processing/input/control/evaluation_baselines.json",
        "image_contract_path": "/opt/program/modernbert/processing_fold_eval/image_contract.json",
        "bm25_scratch_dir": "/opt/ml/processing/work/bm25-evaluation",
        "systems": local_systems,
    }
    return plan, bindings


def validate_phase2_control_bundle_receipt(
    value: object,
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    archive = phase1.validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt), completed_fold_evidence=completed
    )
    static = phase1.validate_static_evaluation_staging_receipt(
        copy.deepcopy(static_staging_receipt), completed_fold_evidence=completed
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    phase1_publication = phase1._validate_overlay_publication(
        copy.deepcopy(phase1_overlay_publication_receipt),
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    phase2_publication = _validate_phase2_overlay_publication(
        copy.deepcopy(phase2_overlay_publication_receipt),
        account_id=infrastructure["account_id"],
        region=infrastructure["region"],
    )
    receipt = _exact_object(value, _CONTROL_BUNDLE_KEYS, name="Phase-2 control bundle receipt")
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != PHASE2_CONTROL_BUNDLE_PROTOCOL
        or type(receipt["outer_fold"]) is not int
        or receipt["outer_fold"] != completed["outer_fold"]
        or receipt["completed_fold_evidence_sha256"] != _document_sha256(completed)
        or receipt["archive_copy_receipt_sha256"] != _document_sha256(archive)
        or receipt["static_staging_receipt_sha256"] != _document_sha256(static)
        or receipt["inventory_acquisition_receipt_sha256"]
        != _document_sha256(phase1_acquisition_receipt)
        or receipt["phase1_overlay_publication_sha256"]
        != _document_sha256(phase1_publication)
        or receipt["phase2_overlay_publication_sha256"]
        != _document_sha256(phase2_publication)
        or receipt["phase2_runtime_identity_sha256"]
        != PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256
    ):
        raise ValueError("Phase-2 control bundle evidence binding changed")
    files = receipt["files"]
    if (
        type(files) is not list
        or [record.get("path") for record in files] != list(CONTROL_FILE_NAMES)
    ):
        raise ValueError("Phase-2 control file inventory changed")
    for record in files:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Phase-2 control file record schema changed")
        if type(record["size"]) is not int or not 0 < record["size"] <= 16 * 1024 * 1024:
            raise ValueError("Phase-2 control file size changed")
        _exact_sha256(record["sha256"], name="Phase-2 control file SHA-256")
    by_name = {record["path"]: record for record in files}
    if (
        receipt["evaluation_plan_sha256"] != by_name["evaluation_plan.json"]["sha256"]
        or receipt["local_bindings_sha256"] != by_name["local_bindings.json"]["sha256"]
    ):
        raise ValueError("Phase-2 generated control hashes changed")
    _validate_self_hash(receipt, name="Phase-2 control bundle receipt")
    return copy.deepcopy(receipt)


def _load_control_bundle(
    control_bundle_dir: Path,
    receipt: Mapping[str, Any],
) -> dict[str, bytes]:
    root = phase1._real_directory(Path(control_bundle_dir), name="Phase-2 control bundle")
    expected = {*CONTROL_FILE_NAMES, "control_bundle_receipt.json"}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("Phase-2 control bundle directory inventory changed")
    saved = _read_regular(root / "control_bundle_receipt.json", name="saved control bundle receipt")
    if saved != strict_config.canonical_json_bytes(receipt):
        raise ValueError("Saved Phase-2 control bundle receipt changed")
    payloads: dict[str, bytes] = {}
    records = {record["path"]: record for record in receipt["files"]}
    for name in CONTROL_FILE_NAMES:
        payload = _read_regular(root / name, name=f"Phase-2 control {name}")
        record = records[name]
        if len(payload) != record["size"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Phase-2 control bytes changed: {name}")
        if name in GENERATED_CONTROL_FILE_NAMES:
            _read_exact_json(
                payload,
                compact=True,
                name=f"Phase-2 control {name}",
            )
        else:
            _read_strict_json(payload, name=f"Phase-2 frozen control {name}")
        payloads[name] = payload
    return payloads


def _validate_phase2_control_content(
    *,
    control_bundle_dir: Path,
    bundle: Mapping[str, Any],
    completed: Mapping[str, Any],
    archive: Mapping[str, Any],
    static: Mapping[str, Any],
    phase2_publication: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    phase1_acquisition_dir: Path,
) -> dict[str, bytes]:
    """Independently re-render generated controls from sealed Phase-1 evidence."""

    payloads = _load_control_bundle(Path(control_bundle_dir), bundle)
    documents = phase1._load_phase1_acquisition_files(
        Path(phase1_acquisition_dir), acquisition
    )
    inventory, storage, _ = phase1._validate_phase1_documents(
        archive_inventory=documents["archive_inventory.json"],
        bm25_storage=documents["bm25_storage.json"],
        artifact_manifest=documents["artifact_manifest.json"],
        archive_copy=archive,
        completed=completed,
    )
    expected_plan, expected_bindings = _render_controls(
        completed=completed,
        publication=phase2_publication,
        archive_inventory=inventory,
        bm25_storage=storage,
    )
    expected_generated = {
        "evaluation_plan.json": _canonical_bytes(expected_plan),
        "local_bindings.json": _canonical_bytes(expected_bindings),
    }
    for name, expected in expected_generated.items():
        if payloads[name] != expected:
            raise ValueError(f"Phase-2 generated control differs from re-rendering: {name}")

    frozen_names = {
        "e5_snapshot.json",
        "evaluation_baselines.json",
        "experiment.json",
        "folds.json",
    }
    static_control = phase1._static_asset(static, "control")
    static_by_name = {record["path"]: record for record in static_control["files"]}
    if set(static_by_name) != frozen_names:
        raise ValueError("Frozen Phase-2 static control inventory changed")
    for name in sorted(frozen_names):
        record = static_by_name[name]
        payload = payloads[name]
        if (
            type(record) is not dict
            or type(record.get("size")) is not int
            or record["size"] != len(payload)
            or record.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"Frozen Phase-2 control differs from staging evidence: {name}")
    return payloads


def build_phase2_control_bundle(
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    static_control_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render and atomically publish the exact six-file Phase-2 controls."""

    (
        completed,
        archive,
        static,
        phase1_publication,
        phase2_publication,
        preflight,
        submission,
        terminal,
        acquisition,
    ) = _validate_context(
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    documents = phase1._load_phase1_acquisition_files(phase1_acquisition_dir, acquisition)
    inventory, storage, _ = phase1._validate_phase1_documents(
        archive_inventory=documents["archive_inventory.json"],
        bm25_storage=documents["bm25_storage.json"],
        artifact_manifest=documents["artifact_manifest.json"],
        archive_copy=archive,
        completed=completed,
    )
    plan, bindings = _render_controls(
        completed=completed,
        publication=phase2_publication,
        archive_inventory=inventory,
        bm25_storage=storage,
    )
    static_control = phase1._static_asset(static, "control")
    static_root = phase1._real_directory(
        Path(static_control_dir), name="frozen Phase-2 control source"
    )
    source_names = {
        "e5_snapshot.json",
        "evaluation_baselines.json",
        "experiment.json",
        "folds.json",
    }
    payloads: dict[str, bytes] = {
        "evaluation_plan.json": _canonical_bytes(plan),
        "local_bindings.json": _canonical_bytes(bindings),
    }
    static_by_name = {record["path"]: record for record in static_control["files"]}
    if set(static_by_name) != source_names:
        raise ValueError("Frozen Phase-2 static control inventory changed")
    for name in sorted(source_names):
        payload = _read_regular(static_root / name, name=f"frozen control {name}")
        record = static_by_name[name]
        if len(payload) != record["size"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen control differs from static staging receipt: {name}")
        _read_strict_json(payload, name=f"frozen control {name}")
        payloads[name] = payload
    files = [
        {
            "path": name,
            "size": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in CONTROL_FILE_NAMES
    ]
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_CONTROL_BUNDLE_PROTOCOL,
            "outer_fold": completed["outer_fold"],
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "archive_copy_receipt_sha256": _document_sha256(archive),
            "static_staging_receipt_sha256": _document_sha256(static),
            "inventory_acquisition_receipt_sha256": _document_sha256(acquisition),
            "phase1_overlay_publication_sha256": _document_sha256(
                phase1_publication
            ),
            "phase2_overlay_publication_sha256": _document_sha256(
                phase2_publication
            ),
            "phase2_runtime_identity_sha256": (
                PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256
            ),
            "files": files,
            "evaluation_plan_sha256": hashlib.sha256(payloads["evaluation_plan.json"]).hexdigest(),
            "local_bindings_sha256": hashlib.sha256(payloads["local_bindings.json"]).hexdigest(),
        }
    )
    receipt = validate_phase2_control_bundle_receipt(
        receipt,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    output = phase1._canonical_absolute(Path(output_dir), name="Phase-2 control output")
    parent = phase1._real_directory(output.parent, name="Phase-2 control output parent")
    incomplete = output.with_name(f".{output.name}.incomplete")
    if output.exists() or output.is_symlink() or incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError("Phase-2 control output must be initially absent")
    os.mkdir(incomplete, mode=0o700)
    try:
        for name in CONTROL_FILE_NAMES:
            phase1._write_bytes_at(incomplete, name, payloads[name])
        phase1._write_bytes_at(
            incomplete,
            "control_bundle_receipt.json",
            strict_config.canonical_json_bytes(receipt),
        )
        descriptor = os.open(incomplete, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            phase1._rename_no_replace(parent_descriptor, incomplete.name, output.name)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        # Preserve the incomplete control tree as a fail-loud taint marker.
        raise
    _validate_phase2_control_content(
        control_bundle_dir=output,
        bundle=receipt,
        completed=completed,
        archive=archive,
        static=static,
        phase2_publication=phase2_publication,
        acquisition=acquisition,
        phase1_acquisition_dir=phase1_acquisition_dir,
    )
    return receipt


def validate_phase2_control_staging_receipt(
    value: object,
    *,
    control_bundle_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    receipt = _exact_object(
        value, _CONTROL_STAGING_KEYS, name="Phase-2 control staging receipt"
    )
    prefix = phase1._normalized_prefix(
        receipt["destination_prefix"], name="Phase-2 control destination prefix"
    )
    input_prefix = phase1._normalized_prefix(
        receipt["input_prefix"], name="Phase-2 control input prefix"
    )
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != PHASE2_CONTROL_STAGING_PROTOCOL
        or type(receipt["outer_fold"]) is not int
        or receipt["outer_fold"] != completed["outer_fold"]
        or receipt["control_bundle_receipt_sha256"] != _document_sha256(bundle)
        or input_prefix != prefix + "input/"
    ):
        raise ValueError("Phase-2 control staging evidence binding changed")
    files = receipt["files"]
    bundle_files = {record["path"]: record for record in bundle["files"]}
    bucket = completed["training_plan"]["infrastructure"]["artifact_bucket"]
    if type(files) is not list or len(files) != len(CONTROL_FILE_NAMES):
        raise ValueError("Phase-2 staged control file count changed")
    for index, (record, name) in enumerate(zip(files, CONTROL_FILE_NAMES)):
        record = _exact_object(
            record,
            {"path", "size", "sha256", "bucket", "key", "version_id", "etag", "sse"},
            name=f"Phase-2 staged control file {index}",
        )
        source = bundle_files[name]
        if (
            record["path"] != name
            or type(record["size"]) is not int
            or record["size"] != source["size"]
            or record["sha256"] != source["sha256"]
            or record["bucket"] != bucket
            or record["key"] != input_prefix + name
            or record["sse"] != "AES256"
            or _ETAG.fullmatch(record["etag"] if type(record["etag"]) is str else "") is None
        ):
            raise ValueError("Phase-2 staged control identity changed")
        if type(record["version_id"]) is not str or not record["version_id"]:
            raise ValueError("Phase-2 staged control VersionId changed")
    manifest = _exact_object(
        receipt["manifest_object"],
        {"bucket", "key", "version_id", "size", "sha256", "etag", "sse"},
        name="Phase-2 control staging manifest",
    )
    if (
        manifest["bucket"] != bucket
        or manifest["key"] != prefix + PHASE2_CONTROL_MANIFEST_NAME
        or manifest["sse"] != "AES256"
        or type(manifest["size"]) is not int
        or manifest["size"] < 1
        or type(manifest["version_id"]) is not str
        or not manifest["version_id"]
        or _ETAG.fullmatch(manifest["etag"] if type(manifest["etag"]) is str else "") is None
    ):
        raise ValueError("Phase-2 control staging manifest identity changed")
    _exact_sha256(manifest["sha256"], name="Phase-2 control manifest SHA-256")
    _validate_self_hash(receipt, name="Phase-2 control staging receipt")
    return copy.deepcopy(receipt)


def _phase2_control_manifest_payload(
    *,
    bundle: Mapping[str, Any],
    prefix: str,
    input_prefix: str,
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PHASE2_CONTROL_STAGING_PROTOCOL,
        "outer_fold": bundle["outer_fold"],
        "control_bundle_receipt_sha256": _document_sha256(bundle),
        "destination_prefix": prefix,
        "input_prefix": input_prefix,
        "files": copy.deepcopy(list(files)),
    }


def stage_phase2_controls_once(
    clients: aws.AwsClients,
    *,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    destination_prefix: str,
    state_dir: Path,
) -> dict[str, Any]:
    """Stage one exact six-file control prefix and publish its manifest last."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    aws.validate_aws_sdk_versions()
    (
        completed,
        archive,
        static,
        phase1_publication,
        phase2_publication,
        _,
        _,
        _,
        acquisition,
    ) = _validate_context(
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    _validate_phase2_control_content(
        control_bundle_dir=control_bundle_dir,
        bundle=bundle,
        completed=completed,
        archive=archive,
        static=static,
        phase2_publication=phase2_publication,
        acquisition=acquisition,
        phase1_acquisition_dir=phase1_acquisition_dir,
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    bucket = infrastructure["artifact_bucket"]
    account = infrastructure["account_id"]
    prefix = phase1._normalized_prefix(
        destination_prefix, name="Phase-2 control destination prefix"
    )
    input_prefix = prefix + "input/"
    aws.validate_artifact_bucket(
        clients.s3, bucket=bucket, region=infrastructure["region"]
    )
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=bucket,
        prefix=prefix,
        expected_bucket_owner=account,
    )
    state = phase1._create_state_directory(
        Path(state_dir), protocol=PHASE2_CONTROL_STAGING_INTENT_PROTOCOL
    )
    intent = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_CONTROL_STAGING_INTENT_PROTOCOL,
            "outer_fold": completed["outer_fold"],
            "control_bundle_receipt_sha256": _document_sha256(bundle),
            "destination_prefix": prefix,
            "input_prefix": input_prefix,
            "files": copy.deepcopy(bundle["files"]),
        }
    )
    phase1._publish_json_absent(state / "intent.json", intent)
    staged: list[dict[str, Any]] = []
    root = phase1._real_directory(Path(control_bundle_dir), name="Phase-2 control bundle")
    for index, source in enumerate(bundle["files"]):
        name = source["path"]
        object_receipt = aws.stage_file_once(
            clients.s3,
            source_path=root / name,
            bucket=bucket,
            key=input_prefix + name,
            expected_bucket_owner=account,
        )
        record = phase1._static_file_record(source=source, staged=object_receipt)
        staged.append(record)
        phase1._publish_json_absent(
            state / f"file-{index:02d}.json",
            _seal(
                {
                    "schema_version": 1,
                    "protocol": PHASE2_CONTROL_STAGING_PROTOCOL,
                    "ordinal": index,
                    "record": record,
                }
            ),
        )
    manifest_payload = _phase2_control_manifest_payload(
        bundle=bundle,
        prefix=prefix,
        input_prefix=input_prefix,
        files=staged,
    )
    manifest_object = phase1._put_json_object_once(
        clients.s3,
        payload=manifest_payload,
        bucket=bucket,
        key=prefix + PHASE2_CONTROL_MANIFEST_NAME,
        expected_bucket_owner=account,
    )
    receipt = _seal(
        {
            **manifest_payload,
            "manifest_object": manifest_object,
        }
    )
    validated = validate_phase2_control_staging_receipt(
        receipt,
        control_bundle_receipt=bundle,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    verify_remote_phase2_controls(
        clients,
        receipt=validated,
        control_bundle_receipt=bundle,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    phase1._publish_json_absent(state / "staging.json", validated)
    return validated


def verify_remote_phase2_controls(
    clients: aws.AwsClients,
    *,
    receipt: Mapping[str, Any],
    control_bundle_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    staged = validate_phase2_control_staging_receipt(
        copy.deepcopy(receipt),
        control_bundle_receipt=control_bundle_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    completed = controlled_supervisor.validate_completed_fold_evidence(
        copy.deepcopy(completed_fold_evidence)
    )
    account = completed["training_plan"]["infrastructure"]["account_id"]
    for record in staged["files"]:
        phase1._verify_aes256_object(
            clients.s3, record=record, expected_bucket_owner=account
        )
    phase1._verify_aes256_object(
        clients.s3,
        record=staged["manifest_object"],
        expected_bucket_owner=account,
    )
    expected_manifest = _phase2_control_manifest_payload(
        bundle=control_bundle_receipt,
        prefix=staged["destination_prefix"],
        input_prefix=staged["input_prefix"],
        files=staged["files"],
    )
    if phase1._read_exact_object(
        clients.s3,
        record=staged["manifest_object"],
        expected_bucket_owner=account,
    ) != _canonical_bytes(expected_manifest):
        raise RuntimeError("Remote Phase-2 control manifest content changed")
    phase1._require_exact_copy_history(
        phase1._list_prefix_history(
            clients.s3,
            bucket=staged["manifest_object"]["bucket"],
            prefix=staged["destination_prefix"],
            expected_bucket_owner=account,
        ),
        expected_objects=[*staged["files"], staged["manifest_object"]],
    )
    return _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_CONTROL_STAGING_PROTOCOL,
            "staging_receipt_sha256": _document_sha256(staged),
            "verified_object_versions": len(staged["files"]) + 1,
            "verified": True,
        }
    )


def build_phase2_storage_proof(
    *,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    phase2_output_reserve_bytes: int,
    safety_reserve_bytes: int,
    acquisition_receipt: Mapping[str, Any],
    acquisition_dir: Path,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive all six control-file sizes from the sealed bundle and prove fit."""

    (
        completed,
        archive,
        static,
        phase1_publication,
        phase2_publication,
        phase1_preflight,
        phase1_submission,
        phase1_terminal,
        acquisition,
    ) = _validate_context(
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=preflight_receipt,
        phase1_submission_receipt=submission_receipt,
        phase1_terminal_receipt=terminal_receipt,
        phase1_acquisition_receipt=acquisition_receipt,
    )
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    _validate_phase2_control_content(
        control_bundle_dir=control_bundle_dir,
        bundle=bundle,
        completed=completed,
        archive=archive,
        static=static,
        phase2_publication=phase2_publication,
        acquisition=acquisition,
        phase1_acquisition_dir=acquisition_dir,
    )
    sizes = {record["path"]: record["size"] for record in bundle["files"]}
    if set(sizes) != set(CONTROL_FILE_NAMES):
        raise ValueError("Phase-2 control-bundle coverage changed")
    return phase1.build_fold_storage_proof(
        acquisition_receipt=acquisition,
        acquisition_dir=acquisition_dir,
        terminal_receipt=phase1_terminal,
        preflight_receipt=phase1_preflight,
        submission_receipt=phase1_submission,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        overlay_publication_receipt=phase1_publication,
        phase2_control_file_sizes=[sizes[name] for name in CONTROL_FILE_NAMES],
        phase2_output_reserve_bytes=phase2_output_reserve_bytes,
        safety_reserve_bytes=safety_reserve_bytes,
    )


def _processing_input(*, name: str, s3_uri: str, local_path: str) -> dict[str, Any]:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": s3_uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }


def _render_phase2_request(
    *,
    completed: Mapping[str, Any],
    archive_copy: Mapping[str, Any],
    static_staging: Mapping[str, Any],
    publication: Mapping[str, Any],
    phase1_preflight: Mapping[str, Any],
    control_staging: Mapping[str, Any],
    job_name: str,
    output_prefix: str,
) -> dict[str, Any]:
    expected_job_name = (
        f"arr-ret-cv1-f{completed['outer_fold']}-evaluate-{completed['attempt_id']}"
    )
    if (
        type(job_name) is not str
        or _JOB_NAME.fullmatch(job_name) is None
        or job_name != expected_job_name
    ):
        raise ValueError(f"Phase-2 job name must equal {expected_job_name}")
    output_prefix = phase1._normalized_prefix(
        output_prefix, name="Phase-2 output prefix"
    )
    plan = completed["training_plan"]
    infrastructure = plan["infrastructure"]
    bucket = infrastructure["artifact_bucket"]
    dataset_uris = {
        run["input_channels"]["data"]["s3_uri"]
        for run in plan["controlled_runs"]
    }
    if len(dataset_uris) != 1:
        raise ValueError("Controlled plan does not share one corrected dataset input")
    dataset_uri = next(iter(dataset_uris)).rstrip("/") + "/"
    dataset_bucket, dataset_prefix = dataset_uri.removeprefix("s3://").split("/", 1)
    dataset_prefix = phase1._normalized_prefix(
        dataset_prefix, name="corrected dataset prefix"
    )
    archive_prefix = phase1._normalized_prefix(
        archive_copy["destination_prefix"], name="fold archive prefix"
    )
    inventory_prefix = phase1._normalized_prefix(
        phase1_preflight["output_prefix"] + "evidence/",
        name="Phase-1 evidence prefix",
    )
    control_prefix = phase1._normalized_prefix(
        control_staging["input_prefix"], name="Phase-2 control input prefix"
    )
    e5_snapshot = phase1._static_asset(static_staging, "e5-snapshot")
    e5_pack = phase1._static_asset(static_staging, "e5-pack")
    fixed_base = phase1._static_asset(static_staging, "fixed-base")
    input_coordinates = (
        ("fold archives", bucket, archive_prefix),
        ("corrected dataset", dataset_bucket, dataset_prefix),
        ("Phase-1 evidence", bucket, inventory_prefix),
        ("Phase-2 controls", bucket, control_prefix),
        ("E5 snapshot", bucket, e5_snapshot["s3_prefix"]),
        ("E5 focus pack", bucket, e5_pack["s3_prefix"]),
        ("fixed base", bucket, fixed_base["s3_prefix"]),
    )
    for name, input_bucket, input_prefix in input_coordinates:
        if input_bucket == bucket and phase1._prefixes_overlap(output_prefix, input_prefix):
            raise ValueError(f"Phase-2 output prefix overlaps the {name} input prefix")
    kms_keys = {
        record["destination_object"]["encryption"]["kms_key_id"]
        for record in archive_copy["copy_set_receipt"]["systems"]
    }
    if len(kms_keys) != 1:
        raise ValueError("Fold archives do not share one output KMS identity")
    return {
        "AppSpecification": {
            "ContainerArguments": [
                "--evaluation-plan",
                "/opt/ml/processing/input/control/evaluation_plan.json",
                "--local-bindings",
                "/opt/ml/processing/input/control/local_bindings.json",
                "--output-dir",
                "/opt/ml/processing/output/evaluation",
                "--device",
                PHASE2_DEVICE,
            ],
            "ContainerEntrypoint": [
                "/opt/conda/bin/python",
                "/opt/program/modernbert/processing_fold_eval/evaluate_sm.py",
            ],
            "ImageUri": publication["remote_digest_uri"],
        },
        "Environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "17",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "NetworkConfig": {
            "EnableInterContainerTrafficEncryption": False,
            "EnableNetworkIsolation": True,
        },
        "ProcessingInputs": [
            _processing_input(
                name="fold-archives",
                s3_uri=f"s3://{bucket}/{archive_prefix}",
                local_path="/opt/ml/processing/input/fold-archives",
            ),
            _processing_input(
                name="dataset",
                s3_uri=dataset_uri,
                local_path="/opt/ml/processing/input/dataset",
            ),
            _processing_input(
                name="fold-inventory",
                s3_uri=f"s3://{bucket}/{inventory_prefix}",
                local_path="/opt/ml/processing/input/fold-inventory",
            ),
            _processing_input(
                name="control",
                s3_uri=f"s3://{bucket}/{control_prefix}",
                local_path="/opt/ml/processing/input/control",
            ),
            _processing_input(
                name="e5-snapshot",
                s3_uri=f"s3://{bucket}/{e5_snapshot['s3_prefix']}",
                local_path="/opt/ml/processing/input/e5-snapshot",
            ),
            _processing_input(
                name="e5-pack",
                s3_uri=f"s3://{bucket}/{e5_pack['s3_prefix']}",
                local_path="/opt/ml/processing/input/e5-pack",
            ),
            _processing_input(
                name="fixed-base",
                s3_uri=f"s3://{bucket}/{fixed_base['s3_prefix']}",
                local_path="/opt/ml/processing/input/fixed-base",
            ),
        ],
        "ProcessingJobName": job_name,
        "ProcessingOutputConfig": {
            "KmsKeyId": next(iter(kms_keys)),
            "Outputs": [
                {
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{output_prefix}",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": infrastructure["processing_instance_count"],
                "InstanceType": infrastructure["processing_instance_type"],
                "VolumeSizeInGB": infrastructure["processing_volume_size_gb"],
            }
        },
        "RoleArn": infrastructure["role_arn"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": PHASE2_PROCESSING_MAX_RUNTIME_SECONDS
        },
        "Tags": [
            {"Key": "Experiment", "Value": "arr_retrieval_cv_v1"},
            {"Key": "ManagedBy", "Value": "arr-retrieval-cv"},
            {"Key": "Purpose", "Value": "fold-evaluation"},
        ],
    }


def _verify_phase1_output_versions(
    clients: aws.AwsClients,
    *,
    acquisition: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    bucket = preflight["request"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
        "S3Uri"
    ].split("/", 3)[2]
    account = preflight["account_id"]
    history = phase1._list_prefix_history(
        clients.s3,
        bucket=bucket,
        prefix=preflight["output_prefix"],
        expected_bucket_owner=account,
    )
    if history["delete_markers"] or len(history["versions"]) != 3:
        raise RuntimeError("Phase-1 output prefix changed after acquisition")
    expected = {
        (record["key"], record["version_id"]): record
        for record in acquisition["remote_objects"]
    }
    observed = {
        (record.get("Key"), record.get("VersionId")): record
        for record in history["versions"]
    }
    if set(observed) != set(expected):
        raise RuntimeError("Phase-1 output version identities changed")
    for coordinate, record in expected.items():
        version = observed[coordinate]
        head = clients.s3.head_object(
            Bucket=record["bucket"],
            Key=record["key"],
            VersionId=record["version_id"],
            ExpectedBucketOwner=account,
        )
        if (
            version.get("IsLatest") is not True
            or version.get("Size") != record["size"]
            or version.get("ETag") != record["etag"]
            or head.get("ContentLength") != record["size"]
            or head.get("ETag") != record["etag"]
            or head.get("VersionId") != record["version_id"]
            or head.get("ServerSideEncryption") != "aws:kms"
            or head.get("SSEKMSKeyId") != record["encryption"]["kms_key_id"]
            or head.get("BucketKeyEnabled") is not True
        ):
            raise RuntimeError("Phase-1 output metadata changed after acquisition")
        response = clients.s3.get_object(
            Bucket=record["bucket"],
            Key=record["key"],
            VersionId=record["version_id"],
            ExpectedBucketOwner=account,
        )
        body = response.get("Body")
        if body is None:
            raise RuntimeError("Phase-1 output readback omitted body")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = body.read(1024 * 1024)
            if type(chunk) is not bytes:
                raise RuntimeError("Phase-1 output body returned non-bytes")
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        if size != record["size"] or digest.hexdigest() != record["sha256"]:
            raise RuntimeError("Phase-1 output bytes changed after acquisition")
    return _seal(
        {
            "schema_version": 1,
            "protocol": phase1.FOLD_INVENTORY_ACQUISITION_PROTOCOL,
            "acquisition_receipt_sha256": _document_sha256(acquisition),
            "verified_object_versions": 3,
            "verified": True,
        }
    )


def _validate_storage_proof_for_bundle(
    *,
    proof: Mapping[str, Any],
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    acquisition_receipt: Mapping[str, Any],
    acquisition_dir: Path,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validated = phase1.validate_fold_storage_proof(copy.deepcopy(proof))
    components = validated["components"]
    expected = build_phase2_storage_proof(
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        phase2_output_reserve_bytes=components["phase2_output_reserve_bytes"],
        safety_reserve_bytes=components["safety_reserve_bytes"],
        acquisition_receipt=acquisition_receipt,
        acquisition_dir=acquisition_dir,
        terminal_receipt=terminal_receipt,
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
    )
    if validated != expected:
        raise ValueError("Phase-2 storage proof differs from exact re-derivation")
    return validated


def validate_phase2_preflight_receipt(
    value: object,
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
) -> dict[str, Any]:
    (
        completed,
        archive,
        static,
        phase1_publication,
        phase2_publication,
        phase1_preflight,
        phase1_submission,
        phase1_terminal,
        acquisition,
    ) = _validate_context(
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    control_staging = validate_phase2_control_staging_receipt(
        copy.deepcopy(control_staging_receipt),
        control_bundle_receipt=bundle,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    proof = _validate_storage_proof_for_bundle(
        proof=storage_proof,
        control_bundle_receipt=bundle,
        control_bundle_dir=control_bundle_dir,
        acquisition_receipt=acquisition,
        acquisition_dir=phase1_acquisition_dir,
        terminal_receipt=phase1_terminal,
        preflight_receipt=phase1_preflight,
        submission_receipt=phase1_submission,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
    )
    receipt = _exact_object(value, _PREFLIGHT_KEYS, name="Phase-2 preflight receipt")
    infrastructure = completed["training_plan"]["infrastructure"]
    request = _render_phase2_request(
        completed=completed,
        archive_copy=archive,
        static_staging=static,
        publication=phase2_publication,
        phase1_preflight=phase1_preflight,
        control_staging=control_staging,
        job_name=receipt["job_name"],
        output_prefix=receipt["output_prefix"],
    )
    static_count = sum(len(asset["files"]) for asset in static["assets"]) + 1
    expected_archive = _seal(
        {
            "schema_version": 1,
            "protocol": phase1.FOLD_ARCHIVE_COPY_PROTOCOL,
            "copy_receipt_sha256": _document_sha256(archive),
            "verified_source_versions": 12,
            "verified_destination_versions": 13,
            "verified": True,
        }
    )
    expected_static = _seal(
        {
            "schema_version": 1,
            "protocol": phase1.STATIC_STAGING_PROTOCOL,
            "staging_receipt_sha256": _document_sha256(static),
            "verified_object_versions": static_count,
            "verified": True,
        }
    )
    expected_control = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_CONTROL_STAGING_PROTOCOL,
            "staging_receipt_sha256": _document_sha256(control_staging),
            "verified_object_versions": len(CONTROL_FILE_NAMES) + 1,
            "verified": True,
        }
    )
    expected_phase1 = _seal(
        {
            "schema_version": 1,
            "protocol": phase1.FOLD_INVENTORY_ACQUISITION_PROTOCOL,
            "acquisition_receipt_sha256": _document_sha256(acquisition),
            "verified_object_versions": 3,
            "verified": True,
        }
    )
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != PHASE2_PREFLIGHT_PROTOCOL
        or type(receipt["outer_fold"]) is not int
        or receipt["outer_fold"] != completed["outer_fold"]
        or receipt["account_id"] != infrastructure["account_id"]
        or receipt["region"] != infrastructure["region"]
        or receipt["image_uri"] != phase2_publication["remote_digest_uri"]
        or receipt["completed_fold_evidence_sha256"] != _document_sha256(completed)
        or receipt["archive_copy_receipt_sha256"] != _document_sha256(archive)
        or receipt["static_staging_receipt_sha256"] != _document_sha256(static)
        or receipt["inventory_acquisition_receipt_sha256"] != _document_sha256(acquisition)
        or receipt["control_bundle_receipt_sha256"] != _document_sha256(bundle)
        or receipt["control_staging_receipt_sha256"] != _document_sha256(control_staging)
        or receipt["storage_proof_sha256"] != _document_sha256(proof)
        or receipt["archive_verification"] != expected_archive
        or receipt["static_verification"] != expected_static
        or receipt["control_verification"] != expected_control
        or receipt["phase1_output_verification"] != expected_phase1
        or receipt["request"] != request
        or receipt["request_sha256"] != _document_sha256(request)
        or receipt["sdk_versions"] != aws.EXPECTED_AWS_SDK_VERSIONS
        or type(receipt["processing_quota"]) is not int
        or receipt["processing_quota"] < 1
    ):
        raise ValueError("Phase-2 preflight evidence binding changed")
    caller = receipt["caller_arn"]
    if type(caller) is not str or not caller.startswith(
        (
            f"arn:aws:iam::{receipt['account_id']}:",
            f"arn:aws:sts::{receipt['account_id']}:",
        )
    ):
        raise ValueError("Phase-2 caller ARN differs from its account")
    _validate_self_hash(receipt, name="Phase-2 preflight receipt")
    return copy.deepcopy(receipt)


def preflight_phase2_evaluation(
    clients: aws.AwsClients,
    *,
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
    job_name: str,
    output_prefix: str,
) -> dict[str, Any]:
    """Revalidate every local/remote input and freeze one Phase-2 request."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    sdk_versions = aws.validate_aws_sdk_versions()
    (
        completed,
        archive,
        static,
        phase1_publication,
        phase2_publication,
        phase1_preflight,
        phase1_submission,
        phase1_terminal,
        acquisition,
    ) = _validate_context(
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    control_staging = validate_phase2_control_staging_receipt(
        copy.deepcopy(control_staging_receipt),
        control_bundle_receipt=bundle,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    proof = _validate_storage_proof_for_bundle(
        proof=storage_proof,
        control_bundle_receipt=bundle,
        control_bundle_dir=control_bundle_dir,
        acquisition_receipt=acquisition,
        acquisition_dir=phase1_acquisition_dir,
        terminal_receipt=phase1_terminal,
        preflight_receipt=phase1_preflight,
        submission_receipt=phase1_submission,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
    )
    request = _render_phase2_request(
        completed=completed,
        archive_copy=archive,
        static_staging=static,
        publication=phase2_publication,
        phase1_preflight=phase1_preflight,
        control_staging=control_staging,
        job_name=job_name,
        output_prefix=output_prefix,
    )
    infrastructure = completed["training_plan"]["infrastructure"]
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != infrastructure["account_id"]:
        raise ValueError("Active AWS account differs from the completed-fold contract")
    aws._assert_role_trust(clients.iam, infrastructure["role_arn"])
    training_aws.verify_remote_training_staging(
        clients.s3,
        training_plan=completed["training_plan"],
        staging_receipt=completed["training_staging_receipt"],
        deep_read=False,
    )
    archive_verification = phase1.verify_remote_fold_archives(
        clients, receipt=archive, completed_fold_evidence=completed
    )
    static_verification = phase1.verify_remote_static_evaluation_inputs(
        clients, receipt=static, completed_fold_evidence=completed
    )
    control_verification = verify_remote_phase2_controls(
        clients,
        receipt=control_staging,
        control_bundle_receipt=bundle,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_acquisition_receipt=acquisition,
    )
    phase1_output_verification = _verify_phase1_output_versions(
        clients, acquisition=acquisition, preflight=phase1_preflight
    )
    image_response = clients.ecr.batch_get_image(
        registryId=infrastructure["account_id"],
        repositoryName="arr-retrieval-eval",
        imageIds=[{"imageDigest": PHASE2_OVERLAY_IMAGE_DIGEST}],
        acceptedMediaTypes=[aws.ECR_MEDIA_TYPE],
    )
    if image_response.get("failures") or len(image_response.get("images", [])) != 1:
        raise ValueError("Fold overlay image is not readable by exact digest")
    raw_manifest = image_response["images"][0].get("imageManifest")
    if (
        type(raw_manifest) is not str
        or "sha256:" + hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()
        != PHASE2_OVERLAY_IMAGE_DIGEST
    ):
        raise ValueError("Fold overlay ECR manifest differs from its digest")
    quota_response = clients.service_quotas.get_service_quota(
        ServiceCode="sagemaker", QuotaCode="L-B013C051"
    )
    quota = quota_response.get("Quota") if type(quota_response) is dict else None
    quota_value = quota.get("Value") if type(quota) is dict else None
    if type(quota_value) is int:
        exact_quota = quota_value
    elif type(quota_value) is float and math.isfinite(quota_value) and quota_value.is_integer():
        exact_quota = int(quota_value)
    else:
        raise ValueError("Processing quota value is not one exact integer")
    if exact_quota < 1:
        raise RuntimeError("Processing ml.g5.12xlarge quota is below one")
    offerings = clients.ec2.describe_instance_type_offerings(
        LocationType="region",
        Filters=[{"Name": "instance-type", "Values": ["g5.12xlarge"]}],
    ).get("InstanceTypeOfferings", [])
    if not any(record.get("InstanceType") == "g5.12xlarge" for record in offerings):
        raise RuntimeError("g5.12xlarge is not offered in the configured region")
    existing = clients.sagemaker.list_processing_jobs(
        NameContains=job_name,
        MaxResults=100,
        SortBy="Name",
        SortOrder="Ascending",
    ).get("ProcessingJobSummaries", [])
    if any(record.get("ProcessingJobName") == job_name for record in existing):
        raise FileExistsError(f"Processing job name already exists: {job_name}")
    normalized_output = phase1._normalized_prefix(
        output_prefix, name="Phase-2 output prefix"
    )
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=infrastructure["artifact_bucket"],
        prefix=normalized_output,
        expected_bucket_owner=infrastructure["account_id"],
    )
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_PREFLIGHT_PROTOCOL,
            "outer_fold": completed["outer_fold"],
            "account_id": infrastructure["account_id"],
            "region": infrastructure["region"],
            "caller_arn": caller.get("Arn"),
            "job_name": job_name,
            "output_prefix": normalized_output,
            "image_uri": phase2_publication["remote_digest_uri"],
            "completed_fold_evidence_sha256": _document_sha256(completed),
            "archive_copy_receipt_sha256": _document_sha256(archive),
            "static_staging_receipt_sha256": _document_sha256(static),
            "inventory_acquisition_receipt_sha256": _document_sha256(acquisition),
            "control_bundle_receipt_sha256": _document_sha256(bundle),
            "control_staging_receipt_sha256": _document_sha256(control_staging),
            "storage_proof_sha256": _document_sha256(proof),
            "archive_verification": archive_verification,
            "static_verification": static_verification,
            "control_verification": control_verification,
            "phase1_output_verification": phase1_output_verification,
            "request": request,
            "request_sha256": _document_sha256(request),
            "sdk_versions": sdk_versions,
            "processing_quota": exact_quota,
        }
    )
    return validate_phase2_preflight_receipt(
        receipt,
        completed_fold_evidence=completed,
        archive_copy_receipt=archive,
        static_staging_receipt=static,
        phase1_overlay_publication_receipt=phase1_publication,
        phase2_overlay_publication_receipt=phase2_publication,
        phase1_preflight_receipt=phase1_preflight,
        phase1_submission_receipt=phase1_submission,
        phase1_terminal_receipt=phase1_terminal,
        phase1_acquisition_receipt=acquisition,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=bundle,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging,
        storage_proof=proof,
    )


def validate_phase2_submission_receipt(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_phase2_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    receipt = _exact_object(value, _SUBMISSION_KEYS, name="Phase-2 submission receipt")
    expected_arn = (
        f"arn:aws:sagemaker:{preflight['region']}:{preflight['account_id']}:"
        f"processing-job/{preflight['job_name']}"
    )
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != PHASE2_SUBMISSION_PROTOCOL
        or type(receipt["outer_fold"]) is not int
        or receipt["outer_fold"] != preflight["outer_fold"]
        or receipt["job_name"] != preflight["job_name"]
        or receipt["job_arn"] != expected_arn
        or receipt["preflight_receipt_sha256"] != _document_sha256(preflight)
        or receipt["request_sha256"] != preflight["request_sha256"]
    ):
        raise ValueError("Phase-2 submission evidence binding changed")
    _validate_self_hash(receipt, name="Phase-2 submission receipt")
    return copy.deepcopy(receipt)


def submit_phase2_evaluation_once(
    clients: aws.AwsClients,
    *,
    preflight_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
    state_dir: Path,
) -> dict[str, Any]:
    """Persist the exact intent, then call CreateProcessingJob once."""

    if not isinstance(clients, aws.AwsClients):
        raise TypeError("clients must be one AwsClients bundle")
    preflight = validate_phase2_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    fresh = preflight_phase2_evaluation(
        clients,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
        job_name=preflight["job_name"],
        output_prefix=preflight["output_prefix"],
    )
    if fresh != preflight:
        raise RuntimeError("Phase-2 preflight is no longer exactly reproducible")
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != preflight["account_id"] or caller.get("Arn") != preflight["caller_arn"]:
        raise ValueError("Active AWS caller differs from the Phase-2 preflight")
    existing = clients.sagemaker.list_processing_jobs(
        NameContains=preflight["job_name"],
        MaxResults=100,
        SortBy="Name",
        SortOrder="Ascending",
    ).get("ProcessingJobSummaries", [])
    if any(record.get("ProcessingJobName") == preflight["job_name"] for record in existing):
        raise FileExistsError("Phase-2 Processing job already exists")
    bucket = preflight["request"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
        "S3Uri"
    ].split("/", 3)[2]
    aws.assert_unused_versioned_prefix(
        clients.s3,
        bucket=bucket,
        prefix=preflight["output_prefix"],
        expected_bucket_owner=preflight["account_id"],
    )
    state = phase1._create_state_directory(
        Path(state_dir), protocol=PHASE2_SUBMISSION_PROTOCOL
    )
    intent = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_SUBMISSION_PROTOCOL,
            "outer_fold": preflight["outer_fold"],
            "job_name": preflight["job_name"],
            "preflight_receipt_sha256": _document_sha256(preflight),
            "request": preflight["request"],
            "request_sha256": preflight["request_sha256"],
        }
    )
    phase1._publish_json_absent(state / "create-intent.json", intent)
    response = clients.sagemaker.create_processing_job(**preflight["request"])
    expected_arn = (
        f"arn:aws:sagemaker:{preflight['region']}:{preflight['account_id']}:"
        f"processing-job/{preflight['job_name']}"
    )
    if response.get("ProcessingJobArn") != expected_arn:
        raise RuntimeError("CreateProcessingJob returned an unexpected Phase-2 ARN")
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_SUBMISSION_PROTOCOL,
            "outer_fold": preflight["outer_fold"],
            "job_name": preflight["job_name"],
            "job_arn": expected_arn,
            "preflight_receipt_sha256": _document_sha256(preflight),
            "request_sha256": preflight["request_sha256"],
        }
    )
    validated = validate_phase2_submission_receipt(
        receipt,
        preflight_receipt=preflight,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    phase1._publish_json_absent(state / "submission.json", validated)
    return validated


def describe_phase2_evaluation(sagemaker: object, *, job_name: str) -> dict[str, Any]:
    if type(job_name) is not str or _JOB_NAME.fullmatch(job_name) is None:
        raise ValueError("Phase-2 job name is invalid")
    response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
    return {
        "schema_version": 1,
        "protocol": PHASE2_TERMINAL_PROTOCOL,
        "job_name": response.get("ProcessingJobName"),
        "job_arn": response.get("ProcessingJobArn"),
        "status": response.get("ProcessingJobStatus"),
        "failure_reason": response.get("FailureReason"),
        "exit_message": response.get("ExitMessage"),
        "processing_start_time": (
            None
            if response.get("ProcessingStartTime") is None
            else phase1._normalize_datetime(
                response["ProcessingStartTime"], name="ProcessingStartTime"
            )
        ),
        "processing_end_time": (
            None
            if response.get("ProcessingEndTime") is None
            else phase1._normalize_datetime(
                response["ProcessingEndTime"], name="ProcessingEndTime"
            )
        ),
    }


def validate_phase2_terminal_receipt(
    value: object,
    *,
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_phase2_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    submission = validate_phase2_submission_receipt(
        copy.deepcopy(submission_receipt),
        preflight_receipt=preflight,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    receipt = _exact_object(value, _TERMINAL_KEYS, name="Phase-2 terminal receipt")
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != PHASE2_TERMINAL_PROTOCOL
        or type(receipt["outer_fold"]) is not int
        or receipt["outer_fold"] != preflight["outer_fold"]
        or receipt["job_name"] != preflight["job_name"]
        or receipt["job_arn"] != submission["job_arn"]
        or receipt["preflight_receipt_sha256"] != _document_sha256(preflight)
        or receipt["submission_receipt_sha256"] != _document_sha256(submission)
        or receipt["request_sha256"] != preflight["request_sha256"]
        or receipt["status"] != "Completed"
        or receipt["failure_reason"] is not None
        or type(receipt["processing_time_microseconds"]) is not int
        or receipt["processing_time_microseconds"] < 0
        or (
            receipt["exit_message"] is not None
            and type(receipt["exit_message"]) is not str
        )
    ):
        raise ValueError("Phase-2 terminal evidence changed")
    start = _normalized_utc_datetime(
        receipt["processing_start_time"], name="Phase-2 processing start time"
    )
    end = _normalized_utc_datetime(
        receipt["processing_end_time"], name="Phase-2 processing end time"
    )
    elapsed = end - start
    elapsed_microseconds = (
        elapsed.days * 86_400_000_000
        + elapsed.seconds * 1_000_000
        + elapsed.microseconds
    )
    if end < start or receipt["processing_time_microseconds"] != elapsed_microseconds:
        raise ValueError("Phase-2 terminal timing evidence changed")
    _validate_self_hash(receipt, name="Phase-2 terminal receipt")
    return copy.deepcopy(receipt)


def verify_completed_phase2_evaluation(
    clients: aws.AwsClients,
    *,
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal terminal evidence only after exact clean service readback."""

    preflight = validate_phase2_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    submission = validate_phase2_submission_receipt(
        copy.deepcopy(submission_receipt),
        preflight_receipt=preflight,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    caller = clients.sts.get_caller_identity()
    if caller.get("Account") != preflight["account_id"]:
        raise ValueError("Active AWS account differs from the Phase-2 receipt")
    response = clients.sagemaker.describe_processing_job(
        ProcessingJobName=preflight["job_name"]
    )
    if response.get("ProcessingJobStatus") != "Completed" or response.get("FailureReason"):
        raise RuntimeError(
            "Fold evaluation is not cleanly complete: "
            f"status={response.get('ProcessingJobStatus')}, "
            f"reason={response.get('FailureReason')!r}"
        )
    if (
        response.get("ProcessingJobName") != preflight["job_name"]
        or response.get("ProcessingJobArn") != submission["job_arn"]
    ):
        raise RuntimeError("DescribeProcessingJob identity differs from Phase-2 submission")
    request = preflight["request"]
    for field in (
        "AppSpecification",
        "Environment",
        "NetworkConfig",
        "ProcessingResources",
        "RoleArn",
        "StoppingCondition",
    ):
        if response.get(field) != request[field]:
            raise RuntimeError(f"DescribeProcessingJob {field} differs from request")
    phase1._validate_processing_io_readback(response, request=request)
    start = response.get("ProcessingStartTime")
    end = response.get("ProcessingEndTime")
    if (
        type(start) is not datetime
        or type(end) is not datetime
        or start.tzinfo is None
        or end.tzinfo is None
        or end < start
    ):
        raise RuntimeError("Completed Phase-2 timing evidence is invalid")
    elapsed = end - start
    elapsed_microseconds = (
        elapsed.days * 86_400_000_000
        + elapsed.seconds * 1_000_000
        + elapsed.microseconds
    )
    receipt = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE2_TERMINAL_PROTOCOL,
            "outer_fold": preflight["outer_fold"],
            "job_name": preflight["job_name"],
            "job_arn": submission["job_arn"],
            "preflight_receipt_sha256": _document_sha256(preflight),
            "submission_receipt_sha256": _document_sha256(submission),
            "request_sha256": preflight["request_sha256"],
            "status": "Completed",
            "failure_reason": None,
            "processing_start_time": phase1._normalize_datetime(
                start, name="ProcessingStartTime"
            ),
            "processing_end_time": phase1._normalize_datetime(
                end, name="ProcessingEndTime"
            ),
            "processing_time_microseconds": elapsed_microseconds,
            "exit_message": response.get("ExitMessage"),
        }
    )
    return validate_phase2_terminal_receipt(
        receipt,
        preflight_receipt=preflight,
        submission_receipt=submission,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )


def _validate_phase2_output_tree(
    root: Path,
    *,
    bundle: Mapping[str, Any],
    control_bundle_dir: Path,
    archive_copy: Mapping[str, Any],
    phase1_acquisition: Mapping[str, Any],
) -> tuple[str, str]:
    root = phase1._real_directory(root, name="Phase-2 acquisition staging tree")
    evaluation = phase1._real_directory(root / "evaluation", name="acquired evaluation")
    evidence = phase1._real_directory(root / "evidence", name="acquired materialization evidence")
    if {path.name for path in root.iterdir()} != {"evaluation", "evidence"}:
        raise ValueError("Phase-2 acquired output-root inventory changed")
    if {path.name for path in evaluation.iterdir()} != {
        "artifact_manifest.json",
        "evaluation_config.json",
        "rankings.jsonl",
        "results.json",
    }:
        raise ValueError("Phase-2 acquired evaluation inventory changed")
    if {path.name for path in evidence.iterdir()} != {
        "artifact_manifest.json",
        "materialization_receipt.json",
    }:
        raise ValueError("Phase-2 acquired evidence inventory changed")
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Phase-2 acquisition contains a symlink")
        if path.is_file() and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1):
            raise ValueError("Phase-2 acquisition contains an unsafe file")

    evaluation_manifest_payload = _read_regular(
        evaluation / "artifact_manifest.json", name="evaluation artifact manifest"
    )
    evaluation_manifest = _read_exact_json(
        evaluation_manifest_payload, compact=True, name="evaluation artifact manifest"
    )
    if (
        set(evaluation_manifest)
        != {"schema_version", "bundle_protocol", "commit_marker", "files"}
        or type(evaluation_manifest["schema_version"]) is not int
        or evaluation_manifest["schema_version"] != 1
        or evaluation_manifest["bundle_protocol"] != "canonical_complete_rankings_v1"
        or evaluation_manifest["commit_marker"] is not True
        or [record.get("path") for record in evaluation_manifest["files"]]
        != ["evaluation_config.json", "rankings.jsonl", "results.json"]
    ):
        raise ValueError("Evaluation artifact manifest schema changed")
    evaluation_payloads: dict[str, bytes] = {}
    for record in evaluation_manifest["files"]:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Evaluation artifact manifest file record changed")
        payload = _read_regular(evaluation / record["path"], name="evaluation artifact")
        if (
            type(record["size"]) is not int
            or record["size"] < 1
            or len(payload) != record["size"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError("Evaluation artifact manifest file identity changed")
        evaluation_payloads[record["path"]] = payload

    controls = _load_control_bundle(Path(control_bundle_dir), bundle)
    plan = _read_exact_json(
        controls["evaluation_plan.json"], compact=True, name="Phase-2 evaluation plan"
    )
    plan_record = next(
        record for record in bundle["files"] if record["path"] == "evaluation_plan.json"
    )
    config_payload = evaluation_payloads["evaluation_config.json"]
    config = _read_exact_json(config_payload, compact=True, name="evaluation configuration")
    identity = config.get("identity")
    if (
        type(identity) is not dict
        or identity.get("experiment_id") != "arr_retrieval_cv_v1"
        or identity.get("outer_fold") != bundle["outer_fold"]
        or identity.get("role") != "test"
        or identity.get("evaluation_plan_sha256") != plan_record["sha256"]
        or identity.get("experiment_config_sha256")
        != plan["experiment_config_sha256"]
        or identity.get("dataset_manifest_sha256")
        != plan["dataset_manifest_sha256"]
        or identity.get("fold_manifest_sha256") != plan["fold_manifest_sha256"]
        or identity.get("passage_index_sha256") != plan["passage_index_sha256"]
        or config.get("runtime_identity") != plan["runtime_identity"]
        or config.get("case_ids") != plan["case_ids"]
        or type(config.get("systems")) is not list
        or len(config["systems"]) != 15
        or type(config.get("query_ids")) is not list
        or len(config["query_ids"]) != plan["query_count"]
        or type(config.get("passage_ids")) is not list
        or len(config["passage_ids"]) != plan["passage_count"]
    ):
        raise ValueError("Evaluation configuration left the Phase-2 control identity")
    system_ids = [record.get("system_id") for record in config["systems"]]
    planned_system_ids = [record["system_id"] for record in plan["systems"]]
    if (
        any(type(system_id) is not str for system_id in system_ids)
        or system_ids != planned_system_ids
        or len(set(system_ids)) != 15
    ):
        raise ValueError("Evaluation configuration system inventory changed")
    query_ids = config["query_ids"]
    regimes = [record.get("regime_name") for record in config.get("regimes", [])]
    if len(regimes) != 4 or len(set(regimes)) != 4:
        raise ValueError("Evaluation configuration regime inventory changed")

    results_payload = evaluation_payloads["results.json"]
    results = _read_exact_json(results_payload, compact=True, name="evaluation results")
    result_records = results.get("result_records")
    expected_pairs = {(system_id, regime) for system_id in system_ids for regime in regimes}
    observed_pairs = (
        {
            (record.get("system_id"), record.get("regime_name"))
            for record in result_records
        }
        if type(result_records) is list
        else set()
    )
    if (
        type(results.get("schema_version")) is not int
        or results.get("schema_version") != 1
        or results.get("bundle_protocol") != "canonical_complete_rankings_v1"
        or type(result_records) is not list
        or len(result_records) != 60
        or observed_pairs != expected_pairs
    ):
        raise ValueError("Evaluation result-summary coverage changed")

    rankings_payload = evaluation_payloads["rankings.jsonl"]
    observed_rows: set[tuple[str, str, str]] = set()
    line_count = 0
    for raw in rankings_payload.splitlines(keepends=True):
        line_count += 1
        if not raw.endswith(b"\n"):
            raise ValueError("Evaluation rankings contain an unterminated line")
        row = _read_exact_json(raw, compact=True, name=f"ranking line {line_count}")
        key = (row.get("system_id"), row.get("regime_name"), row.get("query_id"))
        if key in observed_rows:
            raise ValueError("Evaluation rankings contain a duplicate row")
        observed_rows.add(key)
    expected_rows = {
        (system_id, regime, query_id)
        for system_id in system_ids
        for regime in regimes
        for query_id in query_ids
    }
    if line_count != 60 * len(query_ids) or observed_rows != expected_rows:
        raise ValueError("Evaluation ranking coverage changed")

    materialization_payload = _read_regular(
        evidence / "materialization_receipt.json", name="materialization receipt"
    )
    materialization = _read_exact_json(
        materialization_payload, compact=True, name="materialization receipt"
    )
    materialization_payload_without_hash = {
        key: copy.deepcopy(value)
        for key, value in materialization.items()
        if key != "materialization_sha256"
    }
    expected_manifest_sha = _document_sha256(archive_copy["fold_archive_input_manifest"])
    if (
        set(materialization)
        != {
            "schema_version",
            "protocol",
            "input_manifest_sha256",
            "inventory_receipt_sha256",
            "experiment_id",
            "outer_fold",
            "artifact_root",
            "systems",
            "materialization_sha256",
        }
        or type(materialization["schema_version"]) is not int
        or materialization["schema_version"] != 1
        or materialization["protocol"] != "retrieval_cv_fold_archive_materialization_v1"
        or materialization["input_manifest_sha256"] != expected_manifest_sha
        or materialization["inventory_receipt_sha256"]
        != phase1_acquisition["archive_inventory_receipt_sha256"]
        or materialization["experiment_id"] != "arr_retrieval_cv_v1"
        or type(materialization["outer_fold"]) is not int
        or materialization["outer_fold"] != bundle["outer_fold"]
        or materialization["artifact_root"] != "/opt/ml/processing/work/materialized"
        or type(materialization["systems"]) is not list
        or len(materialization["systems"]) != 12
        or [record.get("system_id") for record in materialization["systems"]]
        != [record["system_id"] for record in archive_copy["copy_set_receipt"]["systems"]]
        or materialization["materialization_sha256"]
        != _document_sha256(materialization_payload_without_hash)
    ):
        raise ValueError("Materialization receipt identity changed")

    evidence_manifest_payload = _read_regular(
        evidence / "artifact_manifest.json", name="materialization artifact manifest"
    )
    evidence_manifest = _read_exact_json(
        evidence_manifest_payload, compact=True, name="materialization artifact manifest"
    )
    evidence_without_hash = {
        key: copy.deepcopy(value)
        for key, value in evidence_manifest.items()
        if key != "artifact_manifest_sha256"
    }
    if (
        set(evidence_manifest)
        != {
            "schema_version",
            "protocol",
            "experiment_id",
            "outer_fold",
            "archive_input_manifest_sha256",
            "archive_inventory_receipt_sha256",
            "materialization_receipt_sha256",
            "files",
            "artifact_manifest_sha256",
        }
        or type(evidence_manifest.get("schema_version")) is not int
        or evidence_manifest.get("schema_version") != 1
        or evidence_manifest.get("protocol") != "retrieval_cv_fold_materialization_output_v1"
        or evidence_manifest.get("experiment_id") != "arr_retrieval_cv_v1"
        or type(evidence_manifest.get("outer_fold")) is not int
        or evidence_manifest.get("outer_fold") != bundle["outer_fold"]
        or evidence_manifest.get("archive_input_manifest_sha256") != expected_manifest_sha
        or evidence_manifest.get("archive_inventory_receipt_sha256")
        != phase1_acquisition["archive_inventory_receipt_sha256"]
        or evidence_manifest.get("materialization_receipt_sha256")
        != materialization["materialization_sha256"]
        or evidence_manifest.get("files")
        != [
            {
                "path": "materialization_receipt.json",
                "sha256": hashlib.sha256(materialization_payload).hexdigest(),
                "size": len(materialization_payload),
            }
        ]
        or evidence_manifest.get("artifact_manifest_sha256")
        != _document_sha256(evidence_without_hash)
    ):
        raise ValueError("Materialization artifact manifest identity changed")
    return (
        hashlib.sha256(evaluation_manifest_payload).hexdigest(),
        hashlib.sha256(evidence_manifest_payload).hexdigest(),
    )


def validate_phase2_acquisition_receipt(
    value: object,
    *,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = validate_phase2_terminal_receipt(
        copy.deepcopy(terminal_receipt),
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    preflight = validate_phase2_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    receipt = _exact_object(value, _ACQUISITION_KEYS, name="Phase-2 acquisition receipt")
    if (
        receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["protocol"] != PHASE2_ACQUISITION_PROTOCOL
        or type(receipt["outer_fold"]) is not int
        or receipt["outer_fold"] != preflight["outer_fold"]
        or receipt["terminal_receipt_sha256"] != _document_sha256(terminal)
        or receipt["control_bundle_receipt_sha256"] != _document_sha256(bundle)
        or receipt["output_prefix"] != preflight["output_prefix"]
    ):
        raise ValueError("Phase-2 acquisition evidence binding changed")
    files = receipt["files"]
    if type(files) is not list or [record.get("path") for record in files] != list(PHASE2_OUTPUT_PATHS):
        raise ValueError("Phase-2 acquisition file inventory changed")
    file_by_path: dict[str, dict[str, Any]] = {}
    for record in files:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Phase-2 acquired-file schema changed")
        if type(record["size"]) is not int or record["size"] < 1:
            raise ValueError("Phase-2 acquired-file size changed")
        _exact_sha256(record["sha256"], name="Phase-2 acquired-file SHA-256")
        file_by_path[record["path"]] = record
    remote = receipt["remote_objects"]
    if type(remote) is not list or len(remote) != len(PHASE2_OUTPUT_PATHS):
        raise ValueError("Phase-2 remote output inventory changed")
    expected_bucket = preflight["request"]["ProcessingOutputConfig"]["Outputs"][0][
        "S3Output"
    ]["S3Uri"].split("/", 3)[2]
    expected_kms = preflight["request"]["ProcessingOutputConfig"]["KmsKeyId"]
    for record, path in zip(remote, PHASE2_OUTPUT_PATHS):
        if type(record) is not dict or set(record) != {
            "bucket",
            "key",
            "version_id",
            "size",
            "etag",
            "sha256",
            "encryption",
        }:
            raise ValueError("Phase-2 remote-object schema changed")
        local = file_by_path[path]
        if (
            record["bucket"] != expected_bucket
            or record["key"] != preflight["output_prefix"] + path
            or type(record["version_id"]) is not str
            or not record["version_id"]
            or type(record["size"]) is not int
            or record["size"] != local["size"]
            or record["sha256"] != local["sha256"]
            or _ETAG.fullmatch(record["etag"] if type(record["etag"]) is str else "") is None
            or record["encryption"]
            != {
                "algorithm": "aws:kms",
                "kms_key_id": expected_kms,
                "bucket_key_enabled": True,
            }
        ):
            raise ValueError("Phase-2 remote-object identity changed")
    by_path = {record["path"]: record for record in files}
    if (
        receipt["evaluation_artifact_manifest_sha256"]
        != by_path["evaluation/artifact_manifest.json"]["sha256"]
        or receipt["materialization_artifact_manifest_sha256"]
        != by_path["evidence/artifact_manifest.json"]["sha256"]
    ):
        raise ValueError("Phase-2 output manifest identities changed")
    _validate_self_hash(receipt, name="Phase-2 acquisition receipt")
    return copy.deepcopy(receipt)


def acquire_phase2_evaluation_once(
    clients: aws.AwsClients,
    *,
    terminal_receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    completed_fold_evidence: Mapping[str, Any],
    archive_copy_receipt: Mapping[str, Any],
    static_staging_receipt: Mapping[str, Any],
    phase1_overlay_publication_receipt: Mapping[str, Any],
    phase2_overlay_publication_receipt: Mapping[str, Any],
    phase1_preflight_receipt: Mapping[str, Any],
    phase1_submission_receipt: Mapping[str, Any],
    phase1_terminal_receipt: Mapping[str, Any],
    phase1_acquisition_receipt: Mapping[str, Any],
    phase1_acquisition_dir: Path,
    control_bundle_receipt: Mapping[str, Any],
    control_bundle_dir: Path,
    control_staging_receipt: Mapping[str, Any],
    storage_proof: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Stream exactly six compact result/evidence objects to one absent tree."""

    terminal = validate_phase2_terminal_receipt(
        copy.deepcopy(terminal_receipt),
        preflight_receipt=preflight_receipt,
        submission_receipt=submission_receipt,
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    preflight = validate_phase2_preflight_receipt(
        copy.deepcopy(preflight_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_preflight_receipt=phase1_preflight_receipt,
        phase1_submission_receipt=phase1_submission_receipt,
        phase1_terminal_receipt=phase1_terminal_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
        phase1_acquisition_dir=phase1_acquisition_dir,
        control_bundle_receipt=control_bundle_receipt,
        control_bundle_dir=control_bundle_dir,
        control_staging_receipt=control_staging_receipt,
        storage_proof=storage_proof,
    )
    bundle = validate_phase2_control_bundle_receipt(
        copy.deepcopy(control_bundle_receipt),
        completed_fold_evidence=completed_fold_evidence,
        archive_copy_receipt=archive_copy_receipt,
        static_staging_receipt=static_staging_receipt,
        phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
        phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
        phase1_acquisition_receipt=phase1_acquisition_receipt,
    )
    archive = phase1.validate_fold_archive_copy_receipt(
        copy.deepcopy(archive_copy_receipt),
        completed_fold_evidence=completed_fold_evidence,
    )
    proof = phase1.validate_fold_storage_proof(copy.deepcopy(storage_proof))
    bucket = preflight["request"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
        "S3Uri"
    ].split("/", 3)[2]
    prefix = preflight["output_prefix"]
    account = preflight["account_id"]
    history = phase1._list_prefix_history(
        clients.s3, bucket=bucket, prefix=prefix, expected_bucket_owner=account
    )
    if history["delete_markers"] or len(history["versions"]) != len(PHASE2_OUTPUT_PATHS):
        raise RuntimeError("Phase-2 output prefix must contain exactly six versions")
    by_path: dict[str, Mapping[str, Any]] = {}
    for version in history["versions"]:
        key = version.get("Key")
        if type(key) is not str or not key.startswith(prefix):
            raise RuntimeError("Phase-2 output key escaped its prefix")
        relative = key.removeprefix(prefix)
        if relative not in PHASE2_OUTPUT_PATHS or relative in by_path:
            raise RuntimeError("Phase-2 output key inventory changed")
        if (
            version.get("IsLatest") is not True
            or type(version.get("VersionId")) is not str
            or type(version.get("Size")) is not int
            or version["Size"] < 1
            or _ETAG.fullmatch(version.get("ETag", "")) is None
        ):
            raise RuntimeError("Phase-2 output version identity changed")
        by_path[relative] = version
    if set(by_path) != set(PHASE2_OUTPUT_PATHS):
        raise RuntimeError("Phase-2 output file coverage changed")
    total_size = sum(version["Size"] for version in by_path.values())
    if total_size > proof["components"]["phase2_output_reserve_bytes"]:
        raise RuntimeError("Phase-2 outputs exceed the proven output reserve")

    output = phase1._canonical_absolute(Path(output_dir), name="Phase-2 acquisition output")
    parent = phase1._real_directory(output.parent, name="Phase-2 acquisition parent")
    incomplete = output.with_name(f".{output.name}.incomplete")
    if output.exists() or output.is_symlink() or incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError("Phase-2 acquisition output must be initially absent")
    os.mkdir(incomplete, mode=0o700)
    os.mkdir(incomplete / "evaluation", mode=0o700)
    os.mkdir(incomplete / "evidence", mode=0o700)
    files: list[dict[str, Any]] = []
    remote_objects: list[dict[str, Any]] = []
    expected_kms = preflight["request"]["ProcessingOutputConfig"]["KmsKeyId"]
    try:
        for relative in PHASE2_OUTPUT_PATHS:
            version = by_path[relative]
            key = version["Key"]
            version_id = version["VersionId"]
            head = clients.s3.head_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ExpectedBucketOwner=account,
            )
            if (
                head.get("ContentLength") != version["Size"]
                or head.get("ETag") != version["ETag"]
                or head.get("VersionId") != version_id
                or head.get("ServerSideEncryption") != "aws:kms"
                or head.get("SSEKMSKeyId") != expected_kms
                or head.get("BucketKeyEnabled") is not True
            ):
                raise RuntimeError("Phase-2 output object metadata changed")
            response = clients.s3.get_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ExpectedBucketOwner=account,
            )
            body = response.get("Body")
            if body is None:
                raise RuntimeError("Phase-2 output GetObject omitted body")
            target = incomplete / PurePosixPath(relative)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
            digest = hashlib.sha256()
            size = 0
            try:
                while True:
                    chunk = body.read(1024 * 1024)
                    if type(chunk) is not bytes:
                        raise RuntimeError("Phase-2 output body returned non-bytes")
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    position = 0
                    while position < len(chunk):
                        written = os.write(descriptor, chunk[position:])
                        if written < 1:
                            raise RuntimeError("Phase-2 output write made no progress")
                        position += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if size != version["Size"]:
                raise RuntimeError("Phase-2 output size changed during download")
            sha256 = digest.hexdigest()
            files.append({"path": relative, "size": size, "sha256": sha256})
            remote_objects.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "version_id": version_id,
                    "size": size,
                    "etag": version["ETag"],
                    "sha256": sha256,
                    "encryption": {
                        "algorithm": "aws:kms",
                        "kms_key_id": expected_kms,
                        "bucket_key_enabled": True,
                    },
                }
            )
        evaluation_hash, evidence_hash = _validate_phase2_output_tree(
            incomplete,
            bundle=bundle,
            control_bundle_dir=control_bundle_dir,
            archive_copy=archive,
            phase1_acquisition=phase1_acquisition_receipt,
        )
        final_history = phase1._list_prefix_history(
            clients.s3, bucket=bucket, prefix=prefix, expected_bucket_owner=account
        )
        if final_history != history:
            raise RuntimeError("Phase-2 output prefix changed during acquisition")
        receipt = _seal(
            {
                "schema_version": 1,
                "protocol": PHASE2_ACQUISITION_PROTOCOL,
                "outer_fold": preflight["outer_fold"],
                "terminal_receipt_sha256": _document_sha256(terminal),
                "control_bundle_receipt_sha256": _document_sha256(bundle),
                "output_prefix": prefix,
                "remote_objects": remote_objects,
                "files": files,
                "evaluation_artifact_manifest_sha256": evaluation_hash,
                "materialization_artifact_manifest_sha256": evidence_hash,
            }
        )
        receipt = validate_phase2_acquisition_receipt(
            receipt,
            terminal_receipt=terminal,
            preflight_receipt=preflight,
            submission_receipt=submission_receipt,
            completed_fold_evidence=completed_fold_evidence,
            archive_copy_receipt=archive,
            static_staging_receipt=static_staging_receipt,
            phase1_overlay_publication_receipt=phase1_overlay_publication_receipt,
            phase2_overlay_publication_receipt=phase2_overlay_publication_receipt,
            phase1_preflight_receipt=phase1_preflight_receipt,
            phase1_submission_receipt=phase1_submission_receipt,
            phase1_terminal_receipt=phase1_terminal_receipt,
            phase1_acquisition_receipt=phase1_acquisition_receipt,
            phase1_acquisition_dir=phase1_acquisition_dir,
            control_bundle_receipt=bundle,
            control_bundle_dir=control_bundle_dir,
            control_staging_receipt=control_staging_receipt,
            storage_proof=storage_proof,
        )
        phase1._write_bytes_at(
            incomplete,
            "acquisition_receipt.json",
            strict_config.canonical_json_bytes(receipt),
        )
        for directory in (incomplete / "evaluation", incomplete / "evidence", incomplete):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            phase1._rename_no_replace(parent_descriptor, incomplete.name, output.name)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        # Preserve the incomplete acquisition tree as a fail-loud taint marker.
        raise
    return receipt
