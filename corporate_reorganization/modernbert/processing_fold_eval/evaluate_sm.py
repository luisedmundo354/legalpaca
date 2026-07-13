"""Strict Phase-2 archive materializer and complete fold-evaluation wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from processing_fold_eval.archive_bridge import (  # noqa: E402
    _file_identity,
    _open_absent_regular,
    _open_directory_snapshot,
    _read_descriptor_exact,
    _rename_no_replace,
    _write_descriptor_exact,
    load_fold_archive_input_manifest,
    load_fold_archive_inventory_receipt,
    materialize_fold_archives,
)
from processing_fold_eval.image_smoke import PROCESSING_LAYOUT  # noqa: E402
from retriever.artifacts import ControlledArtifactExpectation  # noqa: E402
from retriever.evaluator import (  # noqa: E402
    BM25_SYSTEM_TYPE,
    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
    E5_SYSTEM_TYPE,
    FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
    FIXED_BASE_SYSTEM_TYPE,
    _load_exact_json_file,
    _load_exact_json_file_with_sha256,
    _validate_complete_evaluation_plan,
    _validate_complete_local_bindings,
    run_complete_fold_evaluation_plan,
)


PHASE2_EVIDENCE_PROTOCOL = "retrieval_cv_fold_materialization_output_v1"
ARCHIVE_MANIFEST_PATH = Path(PROCESSING_LAYOUT["archive_manifest_path"])
ARCHIVE_RECEIPT_PATH = Path(PROCESSING_LAYOUT["archive_receipt_path"])
WORK_PARENT = Path(PROCESSING_LAYOUT["work_parent"])
MATERIALIZATION_ROOT = Path(PROCESSING_LAYOUT["materialization_root"])
OUTPUT_PARENT = Path(PROCESSING_LAYOUT["output_parent"])
EVALUATION_OUTPUT_DIR = Path(PROCESSING_LAYOUT["evaluation_output_dir"])
EVIDENCE_OUTPUT_DIR = Path(PROCESSING_LAYOUT["evidence_output_dir"])
MATERIALIZATION_RECEIPT_NAME = "materialization_receipt.json"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"


def _canonical_bytes(value: object) -> bytes:
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


def _document_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve(strict=False) != path
        or path.as_posix().startswith("//")
    ):
        raise ValueError(f"{name} must be one canonical absolute path")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{name} contains a symlink component: {current}")
    if not stat.S_ISDIR(path.stat().st_mode):
        raise ValueError(f"{name} must be one real directory")
    return path


def _secure_create_work_parent() -> Path:
    processing_root = _require_real_directory(WORK_PARENT.parent, name="Processing root")
    if WORK_PARENT.exists() or WORK_PARENT.is_symlink():
        raise FileExistsError("Phase-2 work parent must be initially absent")
    parent_descriptor = os.open(
        processing_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.mkdir(WORK_PARENT.name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        metadata = os.stat(
            WORK_PARENT.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink != 2:
            raise RuntimeError("Phase-2 work-parent creation was not stable")
    finally:
        os.close(parent_descriptor)
    live = WORK_PARENT.lstat()
    if (
        stat.S_ISLNK(live.st_mode)
        or not stat.S_ISDIR(live.st_mode)
        or live.st_dev != metadata.st_dev
        or live.st_ino != metadata.st_ino
    ):
        raise RuntimeError("Phase-2 work-parent path changed after creation")
    return WORK_PARENT


def _load_plan_and_bindings(
    evaluation_plan_path: Path,
    local_bindings_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan, plan_sha256 = _load_exact_json_file_with_sha256(
        evaluation_plan_path,
        name="complete fold evaluation plan",
        canonical=True,
    )
    identity, _, system_plans = _validate_complete_evaluation_plan(
        plan,
        evaluation_plan_sha256=plan_sha256,
        expected_image_contract_sha256=FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
    )
    if identity.role != "test":
        raise ValueError("Phase-2 evaluation plan must target the held-out test role")
    bindings = _load_exact_json_file(
        local_bindings_path,
        name="complete fold local bindings",
        canonical=True,
    )
    if bindings.get("bm25_scratch_dir") != PROCESSING_LAYOUT["bm25_scratch_dir"]:
        raise ValueError("Phase-2 BM25 scratch path left the contract-bound work root")
    bound = _validate_complete_local_bindings(bindings, system_plans=system_plans)
    expected_common = {
        "dataset_dir": Path(PROCESSING_LAYOUT["dataset_dir"]),
        "fold_manifest_path": Path(PROCESSING_LAYOUT["fold_manifest_path"]),
        "experiment_config_path": Path(PROCESSING_LAYOUT["experiment_config_path"]),
        "baseline_config_path": Path(PROCESSING_LAYOUT["baseline_config_path"]),
        "image_contract_path": Path(PROCESSING_LAYOUT["image_contract_path"]),
        "bm25_scratch_dir": Path(PROCESSING_LAYOUT["bm25_scratch_dir"]),
    }
    if any(bound[name] != expected for name, expected in expected_common.items()):
        raise ValueError("Phase-2 common local bindings left the exact path map")
    for system in system_plans:
        system_id = system["system_id"]
        local = bound["systems"][system_id]
        if system["system_type"] == BM25_SYSTEM_TYPE:
            expected_local: dict[str, object] = {"system_id": system_id}
        elif system["system_type"] == E5_SYSTEM_TYPE:
            expected_local = {
                "system_id": system_id,
                "snapshot_dir": Path(PROCESSING_LAYOUT["e5_snapshot_dir"]),
                "snapshot_manifest_path": Path(
                    PROCESSING_LAYOUT["e5_snapshot_manifest_path"]
                ),
                "pack_artifact_dir": Path(
                    PROCESSING_LAYOUT["e5_pack_artifact_dir"]
                ),
            }
        elif system["system_type"] == FIXED_BASE_SYSTEM_TYPE:
            expected_local = {
                "system_id": system_id,
                "artifact_dir": Path(
                    PROCESSING_LAYOUT["fixed_base_artifact_dir"]
                ),
            }
        elif system["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE:
            expected_local = {
                "system_id": system_id,
                "artifact_dir": MATERIALIZATION_ROOT / system_id,
            }
        else:
            raise RuntimeError(f"Unexpected Phase-2 system type: {system['system_type']!r}")
        if local != expected_local:
            raise ValueError(f"Phase-2 local binding changed for {system_id!r}")
    return plan, system_plans, bound


def _controlled_expectations(
    *,
    plan: Mapping[str, Any],
    system_plans: Sequence[Mapping[str, Any]],
    bound: Mapping[str, Any],
    archive_manifest: Mapping[str, Any],
) -> dict[str, ControlledArtifactExpectation]:
    if (
        archive_manifest["experiment_id"] != plan["experiment_id"]
        or archive_manifest["outer_fold"] != plan["outer_fold"]
    ):
        raise ValueError("Phase-2 archive manifest differs from the evaluation fold")
    controlled = {
        record["system_id"]: record
        for record in system_plans
        if record["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
    }
    archive_ids = [record["system_id"] for record in archive_manifest["systems"]]
    if list(controlled) != archive_ids:
        raise ValueError("Phase-2 plan and archive manifest controlled order changed")
    expectations: dict[str, ControlledArtifactExpectation] = {}
    for system_id in archive_ids:
        local = bound["systems"][system_id]
        expected_root = MATERIALIZATION_ROOT / system_id
        if local.get("artifact_dir") != expected_root:
            raise ValueError(
                f"Controlled binding {system_id!r} left the materialization root"
            )
        expectation = ControlledArtifactExpectation(
            **controlled[system_id]["expectation"]
        )
        expectations[system_id] = expectation
    materialization_resolved = MATERIALIZATION_ROOT.resolve(strict=False)
    for record in system_plans:
        if record["system_id"] in controlled:
            continue
        local = bound["systems"][record["system_id"]]
        for name, value in local.items():
            if name == "system_id" or not isinstance(value, Path):
                continue
            resolved = value.resolve(strict=False)
            if resolved == materialization_resolved or resolved.is_relative_to(
                materialization_resolved
            ):
                raise ValueError(
                    f"Non-controlled binding {record['system_id']!r}.{name} "
                    "entered the materialization root"
                )
    return expectations


def _validate_materialization_result(
    *,
    archive_manifest: Mapping[str, Any],
    expectations: Mapping[str, ControlledArtifactExpectation],
    materialization: Any,
) -> None:
    expected_system_ids = list(expectations)
    archive_systems = archive_manifest["systems"]
    if [record["system_id"] for record in archive_systems] != expected_system_ids:
        raise RuntimeError("Phase-2 archive order changed after expectation binding")
    if materialization.root != MATERIALIZATION_ROOT:
        raise RuntimeError("Phase-2 materialization result changed identity or order")
    receipt_systems = materialization.receipt["systems"]
    if [record["system_id"] for record in receipt_systems] != expected_system_ids:
        raise RuntimeError("Phase-2 materialization receipt order changed")
    if len(materialization.artifacts) != len(expected_system_ids):
        raise RuntimeError("Phase-2 materialized artifact count changed")
    for ordinal, system_id in enumerate(expected_system_ids):
        expectation = expectations[system_id]
        artifact = materialization.artifacts[ordinal]
        archive_system = archive_systems[ordinal]
        cell = archive_system["cell"]
        if (
            archive_system["system_id"] != system_id
            or artifact.root != MATERIALIZATION_ROOT / system_id
            or artifact.expectation != expectation
            or artifact.identity.experiment_id != archive_manifest["experiment_id"]
            or artifact.identity.outer_fold != archive_manifest["outer_fold"]
            or artifact.identity.query_view != cell["query_view"]
            or artifact.identity.sampler != cell["sampler"]
            or artifact.identity.experiment_seed != cell["experiment_seed"]
        ):
            raise RuntimeError(
                f"Phase-2 materialized artifact identity changed: {system_id!r}"
            )


def _write_payload_at(stage, name: str, payload: bytes) -> None:
    descriptor = _open_absent_regular(stage.descriptor, name)
    try:
        _write_descriptor_exact(descriptor, payload, name=f"Phase-2 evidence {name}")
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        identity = _file_identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError(f"Phase-2 evidence became unsafe: {name}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if _read_descriptor_exact(
            descriptor, len(payload), name=f"Phase-2 evidence readback {name}"
        ) != payload or _file_identity(os.fstat(descriptor)) != identity:
            raise RuntimeError(f"Phase-2 evidence readback changed: {name}")
        stage.assert_stable()
        child = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
        if _file_identity(child) != identity:
            raise RuntimeError(f"Phase-2 evidence entry changed: {name}")
    finally:
        os.close(descriptor)


def _verify_payload_at(stage, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=stage.descriptor,
    )
    try:
        metadata = os.fstat(descriptor)
        identity = _file_identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError(f"Published Phase-2 evidence became unsafe: {name}")
        if _read_descriptor_exact(
            descriptor, len(payload), name=f"Published Phase-2 evidence {name}"
        ) != payload:
            raise RuntimeError(f"Published Phase-2 evidence changed: {name}")
        stage.assert_stable()
        child = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
        if _file_identity(child) != identity:
            raise RuntimeError(f"Published Phase-2 evidence entry changed: {name}")
    finally:
        os.close(descriptor)


def _publish_payloads(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    incomplete = output_dir.with_name(f".{output_dir.name}.incomplete")
    with _open_directory_snapshot(
        output_dir.parent, name="Phase-2 evidence parent"
    ) as publication_parent:
        os.mkdir(incomplete.name, mode=0o700, dir_fd=publication_parent.descriptor)
        created = os.stat(
            incomplete.name,
            dir_fd=publication_parent.descriptor,
            follow_symlinks=False,
        )
        created_identity = (created.st_dev, created.st_ino, created.st_mode)
        if not stat.S_ISDIR(created.st_mode) or stat.S_IMODE(created.st_mode) != 0o700:
            raise RuntimeError("Incomplete Phase-2 evidence creation changed")
        publication_parent.assert_stable()
        with _open_directory_snapshot(
            incomplete, name="Incomplete Phase-2 evidence"
        ) as staging:
            if staging.identity != created_identity:
                raise RuntimeError("Incomplete Phase-2 evidence was replaced before open")
            for name, payload in payloads.items():
                publication_parent.assert_stable()
                staging.assert_stable()
                _write_payload_at(staging, name, payload)
            if set(os.listdir(staging.descriptor)) != set(payloads):
                raise RuntimeError("Incomplete Phase-2 evidence inventory changed")
            os.fsync(staging.descriptor)
            publication_parent.assert_stable()
            staging.assert_stable()
            _rename_no_replace(
                publication_parent.descriptor,
                incomplete.name,
                output_dir.name,
            )
            staging.rebind(output_dir, name="Published Phase-2 evidence")
            os.fsync(publication_parent.descriptor)
            publication_parent.assert_stable()
            if set(os.listdir(staging.descriptor)) != set(payloads):
                raise RuntimeError("Published Phase-2 evidence inventory changed")
            for name, payload in payloads.items():
                _verify_payload_at(staging, name, payload)


def _publish_materialization_evidence(
    *,
    archive_manifest: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _require_real_directory(OUTPUT_PARENT, name="Phase-2 output parent")
    if EVIDENCE_OUTPUT_DIR.parent != parent:
        raise RuntimeError("Phase-2 evidence path left the output parent")
    incomplete = EVIDENCE_OUTPUT_DIR.with_name(f".{EVIDENCE_OUTPUT_DIR.name}.incomplete")
    if (
        EVIDENCE_OUTPUT_DIR.exists()
        or EVIDENCE_OUTPUT_DIR.is_symlink()
        or incomplete.exists()
        or incomplete.is_symlink()
    ):
        raise FileExistsError("Phase-2 evidence output must be initially absent")
    receipt_payload = _canonical_bytes(materialization_receipt)
    artifact_manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PHASE2_EVIDENCE_PROTOCOL,
        "experiment_id": archive_manifest["experiment_id"],
        "outer_fold": archive_manifest["outer_fold"],
        "archive_input_manifest_sha256": inventory_receipt[
            "input_manifest_sha256"
        ],
        "archive_inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
        "materialization_receipt_sha256": materialization_receipt[
            "materialization_sha256"
        ],
        "files": [
            {
                "path": MATERIALIZATION_RECEIPT_NAME,
                "sha256": hashlib.sha256(receipt_payload).hexdigest(),
                "size": len(receipt_payload),
            }
        ],
    }
    artifact_manifest["artifact_manifest_sha256"] = _document_sha256(
        artifact_manifest
    )
    _publish_payloads(
        EVIDENCE_OUTPUT_DIR,
        {
            MATERIALIZATION_RECEIPT_NAME: receipt_payload,
            ARTIFACT_MANIFEST_NAME: _canonical_bytes(artifact_manifest),
        },
    )
    return artifact_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize 12 fold archives and run one complete evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--local-bindings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evaluation_plan_path = Path(args.evaluation_plan)
    local_bindings_path = Path(args.local_bindings)
    if (
        not evaluation_plan_path.is_absolute()
        or not local_bindings_path.is_absolute()
    ):
        raise ValueError("Phase-2 plan and bindings paths must be absolute")
    if evaluation_plan_path != Path(PROCESSING_LAYOUT["evaluation_plan_path"]):
        raise ValueError("Phase-2 evaluation plan left the exact control path")
    if local_bindings_path != Path(PROCESSING_LAYOUT["local_bindings_path"]):
        raise ValueError("Phase-2 local bindings left the exact control path")
    if Path(args.output_dir) != EVALUATION_OUTPUT_DIR:
        raise ValueError("Phase-2 evaluation output left its contract-bound path")
    _require_real_directory(OUTPUT_PARENT, name="Phase-2 output parent")
    bound_outputs = (EVALUATION_OUTPUT_DIR, EVIDENCE_OUTPUT_DIR)
    for path in bound_outputs:
        incomplete = path.with_name(f".{path.name}.incomplete")
        if (
            path.exists()
            or path.is_symlink()
            or incomplete.exists()
            or incomplete.is_symlink()
        ):
            raise FileExistsError("Phase-2 outputs must be initially absent")
    _secure_create_work_parent()
    plan, system_plans, bound = _load_plan_and_bindings(
        evaluation_plan_path,
        local_bindings_path,
    )
    archive_manifest = load_fold_archive_input_manifest(ARCHIVE_MANIFEST_PATH)
    if Path(archive_manifest["archive_root"]) != ARCHIVE_MANIFEST_PATH.parent:
        raise ValueError("Phase-2 archive root left the contract-bound mount")
    inventory_receipt = load_fold_archive_inventory_receipt(
        ARCHIVE_RECEIPT_PATH,
        input_manifest=archive_manifest,
    )
    expectations = _controlled_expectations(
        plan=plan,
        system_plans=system_plans,
        bound=bound,
        archive_manifest=archive_manifest,
    )
    materialization = materialize_fold_archives(
        archive_manifest,
        inventory_receipt,
        output_root=MATERIALIZATION_ROOT,
        expectations=expectations,
    )
    _validate_materialization_result(
        archive_manifest=archive_manifest,
        expectations=expectations,
        materialization=materialization,
    )
    _publish_materialization_evidence(
        archive_manifest=archive_manifest,
        inventory_receipt=inventory_receipt,
        materialization_receipt=materialization.receipt,
    )
    run_complete_fold_evaluation_plan(
        evaluation_plan_path=evaluation_plan_path,
        local_bindings_path=local_bindings_path,
        output_dir=EVALUATION_OUTPUT_DIR,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
