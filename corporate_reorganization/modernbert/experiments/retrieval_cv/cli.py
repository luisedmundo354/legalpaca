"""Fail-loud command line entry point for immutable retrieval-CV orchestration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from . import (
    aggregate,
    aws,
    config,
    controlled_supervisor,
    determinism_gate,
    fold_evaluation_aws,
    fold_processing_aws,
    folds,
    manifest,
    training_artifacts,
    training_aws,
    training_launch,
)


def _add_phase1_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--completed-evidence", type=Path, required=True)
    parser.add_argument("--archive-copy-receipt", type=Path, required=True)
    parser.add_argument("--static-staging-receipt", type=Path, required=True)
    parser.add_argument("--overlay-publication-receipt", type=Path, required=True)
    parser.add_argument("--phase1-preflight-receipt", type=Path, required=True)
    parser.add_argument("--phase1-submission-receipt", type=Path, required=True)
    parser.add_argument("--phase1-terminal-receipt", type=Path, required=True)
    parser.add_argument("--phase1-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--phase1-acquisition-dir", type=Path, required=True)


def _add_phase2_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    _add_phase1_evidence_arguments(parser)
    parser.add_argument(
        "--phase2-overlay-publication-receipt", type=Path, required=True
    )


def _add_phase2_context_arguments(parser: argparse.ArgumentParser) -> None:
    _add_phase2_evidence_arguments(parser)
    parser.add_argument("--control-bundle-receipt", type=Path, required=True)
    parser.add_argument("--control-bundle-dir", type=Path, required=True)
    parser.add_argument("--control-staging-receipt", type=Path, required=True)
    parser.add_argument("--storage-proof", type=Path, required=True)


def _absolute(path: Path, *, name: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    return path


def _require_absent_output(path: Path) -> None:
    path = _absolute(path, name="output")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"Output parent must be a real directory: {path.parent}")
    incomplete = path.with_name(f".{path.name}.incomplete")
    if incomplete.exists() or incomplete.is_symlink():
        raise FileExistsError(f"Refusing stale incomplete output: {incomplete}")


def _require_disjoint_paths(
    first: Path,
    second: Path,
    *,
    name: str,
) -> None:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    if (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    ):
        raise ValueError(f"{name} must be disjoint")


def _publish_json(path: Path, value: object) -> None:
    _require_absent_output(path)
    payload = config.canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.incomplete")
    published = False
    try:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
        published = True
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        loaded, _ = config.load_canonical_json_object(path)
        if loaded != value:
            raise RuntimeError("Published JSON changed on exact readback")
    except BaseException:
        if published and (path.exists() or path.is_symlink()):
            path.unlink()
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Immutable ARR retrieval-CV orchestration",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build_data = commands.add_parser("build-data", allow_abbrev=False)
    build_data.add_argument("--raw-dir", type=Path, required=True)
    build_data.add_argument("--tokenizer-dir", type=Path, required=True)
    build_data.add_argument("--output-dir", type=Path, required=True)

    freeze = commands.add_parser("freeze-folds", allow_abbrev=False)
    freeze.add_argument("--dataset-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    build_image = commands.add_parser("build-eval-image", allow_abbrev=False)
    build_image.add_argument("--frozen-context", type=Path, required=True)
    build_image.add_argument("--metadata-file", type=Path, required=True)
    build_image.add_argument("--build-replica", type=int, choices=(1, 2), required=True)
    build_image.add_argument("--receipt-output", type=Path, required=True)

    publish_image = commands.add_parser("publish-eval-image", allow_abbrev=False)
    publish_image.add_argument("--aws-config", type=Path, required=True)
    publish_image.add_argument("--receipt-output", type=Path, required=True)

    publish_training = commands.add_parser("publish-training-image", allow_abbrev=False)
    publish_training.add_argument("--aws-config", type=Path, required=True)
    publish_training.add_argument("--receipt-output", type=Path, required=True)

    freeze_manifest = commands.add_parser("freeze-training-plan", allow_abbrev=False)
    freeze_manifest.add_argument("--scientific-config", type=Path, required=True)
    freeze_manifest.add_argument("--aws-config", type=Path, required=True)
    freeze_manifest.add_argument("--source-root", type=Path, required=True)
    freeze_manifest.add_argument("--source-bundle-output", type=Path, required=True)
    freeze_manifest.add_argument("--manifest-output", type=Path, required=True)
    freeze_manifest.add_argument("--attempt-id", default="a1")
    freeze_manifest.add_argument("--parent-manifest", type=Path)

    stage = commands.add_parser("stage", allow_abbrev=False)
    stage_modes = stage.add_subparsers(dest="stage_mode", required=True)
    training_inputs = stage_modes.add_parser("training-inputs", allow_abbrev=False)
    training_inputs.add_argument("--manifest", type=Path, required=True)
    training_inputs.add_argument("--source-bundle", type=Path, required=True)
    training_inputs.add_argument("--dataset-dir", type=Path, required=True)
    training_inputs.add_argument("--base-model-dir", type=Path, required=True)
    training_inputs.add_argument("--snapshot-manifest", type=Path, required=True)
    training_inputs.add_argument("--receipt-output", type=Path, required=True)

    preflight = commands.add_parser("preflight", allow_abbrev=False)
    preflight_modes = preflight.add_subparsers(dest="preflight_mode", required=True)
    runtime_preflight = preflight_modes.add_parser("runtime-smoke", allow_abbrev=False)
    runtime_preflight.add_argument("--aws-config", type=Path, required=True)
    runtime_preflight.add_argument("--image-uri", required=True)
    runtime_preflight.add_argument("--job-name", required=True)
    runtime_preflight.add_argument("--receipt-output", type=Path, required=True)
    training_preflight = preflight_modes.add_parser("training", allow_abbrev=False)
    training_preflight.add_argument("--manifest", type=Path, required=True)
    training_preflight.add_argument("--staging-receipt", type=Path, required=True)
    training_preflight.add_argument("--run-id", required=True)
    training_preflight.add_argument("--receipt-output", type=Path, required=True)

    submit = commands.add_parser("submit", allow_abbrev=False)
    submit_modes = submit.add_subparsers(dest="submit_mode", required=True)
    runtime_submit = submit_modes.add_parser("runtime-smoke", allow_abbrev=False)
    runtime_submit.add_argument("--preflight-receipt", type=Path, required=True)
    runtime_submit.add_argument("--receipt-output", type=Path, required=True)
    training_submit = submit_modes.add_parser("training", allow_abbrev=False)
    training_submit.add_argument("--manifest", type=Path, required=True)
    training_submit.add_argument("--staging-receipt", type=Path, required=True)
    training_submit.add_argument("--preflight-receipt", type=Path, required=True)
    training_submit.add_argument("--receipt-output", type=Path, required=True)

    status = commands.add_parser("status", allow_abbrev=False)
    status_modes = status.add_subparsers(dest="status_mode", required=True)
    runtime_status = status_modes.add_parser("runtime-smoke", allow_abbrev=False)
    runtime_status.add_argument("--job-name", required=True)
    runtime_status.add_argument("--region", choices=("us-east-1",), required=True)
    runtime_status.add_argument("--output", type=Path, required=True)
    training_status = status_modes.add_parser("training", allow_abbrev=False)
    training_status.add_argument("--manifest", type=Path, required=True)
    training_status.add_argument("--staging-receipt", type=Path, required=True)
    training_status.add_argument("--preflight-receipt", type=Path, required=True)
    training_status.add_argument("--submission-receipt", type=Path, required=True)
    training_status.add_argument("--output", type=Path, required=True)

    acquire = commands.add_parser("acquire", allow_abbrev=False)
    acquire_modes = acquire.add_subparsers(dest="acquire_mode", required=True)
    acquire_smoke = acquire_modes.add_parser(
        "determinism-smoke", allow_abbrev=False
    )
    acquire_smoke.add_argument("--manifest", type=Path, required=True)
    acquire_smoke.add_argument("--staging-receipt", type=Path, required=True)
    acquire_smoke.add_argument("--preflight-receipt", type=Path, required=True)
    acquire_smoke.add_argument("--submission-receipt", type=Path, required=True)
    acquire_smoke.add_argument("--terminal-receipt", type=Path, required=True)
    acquire_smoke.add_argument("--output-dir", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate_modes = evaluate.add_subparsers(dest="evaluate_mode", required=True)
    runtime_evaluate = evaluate_modes.add_parser("runtime-smoke", allow_abbrev=False)
    runtime_evaluate.add_argument("--preflight-receipt", type=Path, required=True)
    runtime_evaluate.add_argument("--submission-receipt", type=Path, required=True)
    runtime_evaluate.add_argument("--region", choices=("us-east-1",), required=True)
    runtime_evaluate.add_argument("--output", type=Path, required=True)

    aggregate_parser = commands.add_parser("aggregate", allow_abbrev=False)
    aggregate_parser.add_argument(
        "--evaluation-dir", type=Path, action="append", required=True
    )
    aggregate_parser.add_argument("--dataset-dir", type=Path, required=True)
    aggregate_parser.add_argument("--fold-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", allow_abbrev=False)
    verify_modes = verify.add_subparsers(dest="verify_mode", required=True)
    verify_manifest = verify_modes.add_parser("training-plan", allow_abbrev=False)
    verify_manifest.add_argument("--manifest", type=Path, required=True)
    verify_manifest.add_argument("--output", type=Path, required=True)
    verify_training = verify_modes.add_parser("training", allow_abbrev=False)
    verify_training.add_argument("--manifest", type=Path, required=True)
    verify_training.add_argument("--staging-receipt", type=Path, required=True)
    verify_training.add_argument("--preflight-receipt", type=Path, required=True)
    verify_training.add_argument("--submission-receipt", type=Path, required=True)
    verify_training.add_argument("--output", type=Path, required=True)
    verify_determinism = verify_modes.add_parser(
        "determinism-smoke", allow_abbrev=False
    )
    verify_determinism.add_argument("--manifest", type=Path, required=True)
    verify_determinism.add_argument("--staging-receipt", type=Path, required=True)
    verify_determinism.add_argument(
        "--acquisition-receipt-a", type=Path, required=True
    )
    verify_determinism.add_argument(
        "--acquisition-receipt-b", type=Path, required=True
    )
    verify_determinism.add_argument("--output", type=Path, required=True)

    fold_processing = commands.add_parser("fold-processing", allow_abbrev=False)
    fold_modes = fold_processing.add_subparsers(dest="fold_mode", required=True)

    completed_evidence = fold_modes.add_parser("completed-evidence", allow_abbrev=False)
    completed_evidence.add_argument("--supervisor-state-dir", type=Path, required=True)
    completed_evidence.add_argument(
        "--outer-fold", type=int, choices=range(5), required=True
    )
    completed_evidence.add_argument("--output", type=Path, required=True)

    stage_static = fold_modes.add_parser("stage-static", allow_abbrev=False)
    stage_static.add_argument("--completed-evidence", type=Path, required=True)
    stage_static.add_argument("--e5-snapshot-dir", type=Path, required=True)
    stage_static.add_argument("--e5-snapshot-manifest", type=Path, required=True)
    stage_static.add_argument("--e5-pack-dir", type=Path, required=True)
    stage_static.add_argument("--fixed-base-dir", type=Path, required=True)
    stage_static.add_argument("--destination-prefix", required=True)
    stage_static.add_argument("--state-dir", type=Path, required=True)
    stage_static.add_argument("--receipt-output", type=Path, required=True)

    copy_archives = fold_modes.add_parser("copy-archives", allow_abbrev=False)
    copy_archives.add_argument("--completed-evidence", type=Path, required=True)
    copy_archives.add_argument("--destination-prefix", required=True)
    copy_archives.add_argument("--state-dir", type=Path, required=True)
    copy_archives.add_argument("--receipt-output", type=Path, required=True)

    phase1_preflight = fold_modes.add_parser("phase1-preflight", allow_abbrev=False)
    phase1_preflight.add_argument("--completed-evidence", type=Path, required=True)
    phase1_preflight.add_argument("--archive-copy-receipt", type=Path, required=True)
    phase1_preflight.add_argument("--static-staging-receipt", type=Path, required=True)
    phase1_preflight.add_argument(
        "--overlay-publication-receipt", type=Path, required=True
    )
    phase1_preflight.add_argument("--job-name", required=True)
    phase1_preflight.add_argument("--output-prefix", required=True)
    phase1_preflight.add_argument("--receipt-output", type=Path, required=True)

    phase1_submit = fold_modes.add_parser("phase1-submit", allow_abbrev=False)
    phase1_submit.add_argument("--completed-evidence", type=Path, required=True)
    phase1_submit.add_argument("--archive-copy-receipt", type=Path, required=True)
    phase1_submit.add_argument("--static-staging-receipt", type=Path, required=True)
    phase1_submit.add_argument(
        "--overlay-publication-receipt", type=Path, required=True
    )
    phase1_submit.add_argument("--preflight-receipt", type=Path, required=True)
    phase1_submit.add_argument("--state-dir", type=Path, required=True)
    phase1_submit.add_argument("--receipt-output", type=Path, required=True)

    phase1_status = fold_modes.add_parser("phase1-status", allow_abbrev=False)
    phase1_status.add_argument("--completed-evidence", type=Path, required=True)
    phase1_status.add_argument("--archive-copy-receipt", type=Path, required=True)
    phase1_status.add_argument("--static-staging-receipt", type=Path, required=True)
    phase1_status.add_argument(
        "--overlay-publication-receipt", type=Path, required=True
    )
    phase1_status.add_argument("--preflight-receipt", type=Path, required=True)
    phase1_status.add_argument("--output", type=Path, required=True)

    phase1_verify = fold_modes.add_parser("phase1-verify", allow_abbrev=False)
    phase1_verify.add_argument("--completed-evidence", type=Path, required=True)
    phase1_verify.add_argument("--archive-copy-receipt", type=Path, required=True)
    phase1_verify.add_argument("--static-staging-receipt", type=Path, required=True)
    phase1_verify.add_argument(
        "--overlay-publication-receipt", type=Path, required=True
    )
    phase1_verify.add_argument("--preflight-receipt", type=Path, required=True)
    phase1_verify.add_argument("--submission-receipt", type=Path, required=True)
    phase1_verify.add_argument("--receipt-output", type=Path, required=True)

    phase1_acquire = fold_modes.add_parser("phase1-acquire", allow_abbrev=False)
    phase1_acquire.add_argument("--completed-evidence", type=Path, required=True)
    phase1_acquire.add_argument("--archive-copy-receipt", type=Path, required=True)
    phase1_acquire.add_argument("--static-staging-receipt", type=Path, required=True)
    phase1_acquire.add_argument(
        "--overlay-publication-receipt", type=Path, required=True
    )
    phase1_acquire.add_argument("--preflight-receipt", type=Path, required=True)
    phase1_acquire.add_argument("--submission-receipt", type=Path, required=True)
    phase1_acquire.add_argument("--terminal-receipt", type=Path, required=True)
    phase1_acquire.add_argument("--output-dir", type=Path, required=True)

    storage_proof = fold_modes.add_parser("storage-proof", allow_abbrev=False)
    storage_proof.add_argument("--completed-evidence", type=Path, required=True)
    storage_proof.add_argument("--archive-copy-receipt", type=Path, required=True)
    storage_proof.add_argument("--static-staging-receipt", type=Path, required=True)
    storage_proof.add_argument(
        "--overlay-publication-receipt", type=Path, required=True
    )
    storage_proof.add_argument(
        "--phase2-overlay-publication-receipt", type=Path, required=True
    )
    storage_proof.add_argument("--preflight-receipt", type=Path, required=True)
    storage_proof.add_argument("--submission-receipt", type=Path, required=True)
    storage_proof.add_argument("--terminal-receipt", type=Path, required=True)
    storage_proof.add_argument("--acquisition-receipt", type=Path, required=True)
    storage_proof.add_argument("--acquisition-dir", type=Path, required=True)
    storage_proof.add_argument("--control-bundle-receipt", type=Path, required=True)
    storage_proof.add_argument("--control-bundle-dir", type=Path, required=True)
    storage_proof.add_argument("--phase2-output-reserve-bytes", type=int, required=True)
    storage_proof.add_argument("--safety-reserve-bytes", type=int, required=True)
    storage_proof.add_argument("--output", type=Path, required=True)

    phase2_controls = fold_modes.add_parser("phase2-controls", allow_abbrev=False)
    _add_phase2_evidence_arguments(phase2_controls)
    phase2_controls.add_argument("--static-control-dir", type=Path, required=True)
    phase2_controls.add_argument("--output-dir", type=Path, required=True)

    phase2_stage = fold_modes.add_parser("phase2-stage-controls", allow_abbrev=False)
    _add_phase2_evidence_arguments(phase2_stage)
    phase2_stage.add_argument("--control-bundle-receipt", type=Path, required=True)
    phase2_stage.add_argument("--control-bundle-dir", type=Path, required=True)
    phase2_stage.add_argument("--destination-prefix", required=True)
    phase2_stage.add_argument("--state-dir", type=Path, required=True)
    phase2_stage.add_argument("--receipt-output", type=Path, required=True)

    phase2_preflight = fold_modes.add_parser("phase2-preflight", allow_abbrev=False)
    _add_phase2_context_arguments(phase2_preflight)
    phase2_preflight.add_argument("--job-name", required=True)
    phase2_preflight.add_argument("--output-prefix", required=True)
    phase2_preflight.add_argument("--receipt-output", type=Path, required=True)

    phase2_submit = fold_modes.add_parser("phase2-submit", allow_abbrev=False)
    _add_phase2_context_arguments(phase2_submit)
    phase2_submit.add_argument("--preflight-receipt", type=Path, required=True)
    phase2_submit.add_argument("--state-dir", type=Path, required=True)
    phase2_submit.add_argument("--receipt-output", type=Path, required=True)

    phase2_status = fold_modes.add_parser("phase2-status", allow_abbrev=False)
    _add_phase2_context_arguments(phase2_status)
    phase2_status.add_argument("--preflight-receipt", type=Path, required=True)
    phase2_status.add_argument("--output", type=Path, required=True)

    phase2_verify = fold_modes.add_parser("phase2-verify", allow_abbrev=False)
    _add_phase2_context_arguments(phase2_verify)
    phase2_verify.add_argument("--preflight-receipt", type=Path, required=True)
    phase2_verify.add_argument("--submission-receipt", type=Path, required=True)
    phase2_verify.add_argument("--receipt-output", type=Path, required=True)

    phase2_acquire = fold_modes.add_parser("phase2-acquire", allow_abbrev=False)
    _add_phase2_context_arguments(phase2_acquire)
    phase2_acquire.add_argument("--preflight-receipt", type=Path, required=True)
    phase2_acquire.add_argument("--submission-receipt", type=Path, required=True)
    phase2_acquire.add_argument("--terminal-receipt", type=Path, required=True)
    phase2_acquire.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _load_receipt(path: Path) -> dict[str, Any]:
    path = _absolute(path, name="receipt")
    raw = config._read_regular_file_once(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=config._reject_duplicate_keys,
            parse_constant=config._reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"Canonical JSON is not UTF-8: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"Canonical receipt must contain one object: {path}")
    if raw not in {
        config.canonical_json_bytes(value),
        aws.canonical_json_bytes(value),
    }:
        raise ValueError(
            "Receipt is neither repository-canonical nor compact-canonical JSON: "
            f"{path}"
        )
    return value


def _load_completed_fold_evidence(path: Path) -> dict[str, Any]:
    return controlled_supervisor.validate_completed_fold_evidence(_load_receipt(path))


def _clients_for_completed_fold(
    completed_fold_evidence: dict[str, Any],
) -> aws.AwsClients:
    region = completed_fold_evidence["training_plan"]["infrastructure"]["region"]
    return aws.make_clients(region=region)


def _load_phase1_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _load_completed_fold_evidence(args.completed_evidence),
        _load_receipt(args.archive_copy_receipt),
        _load_receipt(args.static_staging_receipt),
        _load_receipt(args.overlay_publication_receipt),
    )


def _load_phase2_evidence(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "completed_fold_evidence": _load_completed_fold_evidence(
            args.completed_evidence
        ),
        "archive_copy_receipt": _load_receipt(args.archive_copy_receipt),
        "static_staging_receipt": _load_receipt(args.static_staging_receipt),
        "phase1_overlay_publication_receipt": _load_receipt(
            args.overlay_publication_receipt
        ),
        "phase2_overlay_publication_receipt": _load_receipt(
            args.phase2_overlay_publication_receipt
        ),
        "phase1_preflight_receipt": _load_receipt(
            args.phase1_preflight_receipt
        ),
        "phase1_submission_receipt": _load_receipt(
            args.phase1_submission_receipt
        ),
        "phase1_terminal_receipt": _load_receipt(args.phase1_terminal_receipt),
        "phase1_acquisition_receipt": _load_receipt(
            args.phase1_acquisition_receipt
        ),
        "phase1_acquisition_dir": _absolute(
            args.phase1_acquisition_dir, name="phase1-acquisition-dir"
        ),
    }


def _load_phase2_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_load_phase2_evidence(args),
        "control_bundle_receipt": _load_receipt(args.control_bundle_receipt),
        "control_bundle_dir": _absolute(
            args.control_bundle_dir, name="control-bundle-dir"
        ),
        "control_staging_receipt": _load_receipt(args.control_staging_receipt),
        "storage_proof": _load_receipt(args.storage_proof),
    }


def _load_training_plan(path: Path) -> dict[str, Any]:
    value, _ = manifest.read_manifest(_absolute(path, name="manifest"))
    return value


def _parent_manifest_sha256(
    *,
    attempt_id: str,
    parent_manifest: Path | None,
) -> str | None:
    attempt_number = manifest._attempt_number(attempt_id)
    if attempt_number == 1:
        if parent_manifest is not None:
            raise ValueError("Attempt a1 must not name a parent manifest")
        return None
    if parent_manifest is None:
        raise ValueError(f"Attempt {attempt_id} requires its parent manifest")
    parent, parent_sha256 = manifest.read_manifest(
        _absolute(parent_manifest, name="parent-manifest")
    )
    expected_parent = f"a{attempt_number - 1}"
    actual_parent = parent["attempt"]["attempt_id"]
    if actual_parent != expected_parent:
        raise ValueError(
            f"Attempt {attempt_id} requires parent {expected_parent}, got {actual_parent}"
        )
    return parent_sha256


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-data":
        from ...data_prep.build_final_annotations_gold_dataset import build_dataset

        build_dataset(
            raw_dir=_absolute(args.raw_dir, name="raw-dir"),
            tokenizer_dir=_absolute(args.tokenizer_dir, name="tokenizer-dir"),
            processed_dir=_absolute(args.output_dir, name="output-dir"),
        )
        return 0
    if args.command == "freeze-folds":
        folds.freeze_fold_manifest(
            dataset_dir=_absolute(args.dataset_dir, name="dataset-dir"),
            output_path=_absolute(args.output, name="output"),
        )
        return 0
    if args.command == "build-eval-image":
        from ...processing_eval.build_context import build_frozen_image

        receipt_output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(receipt_output)
        receipt = build_frozen_image(
            _absolute(args.frozen_context, name="frozen-context"),
            _absolute(args.metadata_file, name="metadata-file"),
            build_replica=args.build_replica,
        )
        _publish_json(receipt_output, receipt)
        return 0
    if args.command == "publish-eval-image":
        receipt_output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(receipt_output)
        aws_config, _ = config.load_aws_local_config(
            _absolute(args.aws_config, name="aws-config")
        )
        validated = aws.validate_aws_config(aws_config)
        clients = aws.make_clients(region=validated["region"])
        caller = clients.sts.get_caller_identity()
        if caller.get("Account") != validated["account_id"]:
            raise ValueError("Active AWS account differs from the AWS-local contract")
        aws.ensure_evaluation_repository(clients.ecr, create_if_absent=True)
        receipt = aws.publish_evaluation_image_once(
            ecr=clients.ecr,
            account_id=validated["account_id"],
            region=validated["region"],
        )
        _publish_json(receipt_output, receipt)
        return 0
    if args.command == "publish-training-image":
        receipt_output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(receipt_output)
        aws_config, _ = config.load_aws_local_config(
            _absolute(args.aws_config, name="aws-config")
        )
        validated = aws.validate_aws_config(aws_config)
        clients = aws.make_clients(region=validated["region"])
        caller = clients.sts.get_caller_identity()
        if caller.get("Account") != validated["account_id"]:
            raise ValueError("Active AWS account differs from the AWS-local contract")
        aws.ensure_evaluation_repository(clients.ecr, create_if_absent=False)
        receipt = aws.publish_training_image_once(
            ecr=clients.ecr,
            account_id=validated["account_id"],
            region=validated["region"],
        )
        _publish_json(receipt_output, receipt)
        return 0
    if args.command == "freeze-training-plan":
        source_output = _absolute(
            args.source_bundle_output, name="source-bundle-output"
        )
        manifest_output = _absolute(args.manifest_output, name="manifest-output")
        _require_absent_output(source_output)
        _require_absent_output(manifest_output)
        scientific, _ = config.load_scientific_config(
            _absolute(args.scientific_config, name="scientific-config")
        )
        aws_config, _ = config.load_aws_local_config(
            _absolute(args.aws_config, name="aws-config")
        )
        parent_manifest_sha256 = _parent_manifest_sha256(
            attempt_id=args.attempt_id,
            parent_manifest=args.parent_manifest,
        )
        source_root = _absolute(args.source_root, name="source-root")
        manifest.validate_scientific_source_claims(source_root, scientific)
        manifest.validate_clean_source_checkout(
            source_root,
            expected_git_commit=scientific["sources"]["git_commit"],
            expected_git_tree=scientific["sources"]["git_tree"],
            expected_commit_epoch=scientific["sources"]["commit_epoch"],
        )
        bundle = manifest.build_source_bundle(
            source_root=source_root,
            include_paths=scientific["sources"]["include_paths"],
            output_path=source_output,
            commit_epoch=scientific["sources"]["commit_epoch"],
            expected_git_commit=scientific["sources"]["git_commit"],
            expected_bundler_runtime=manifest.EXPECTED_BUNDLER_RUNTIME,
        )
        try:
            manifest.validate_clean_source_checkout(
                source_root,
                expected_git_commit=scientific["sources"]["git_commit"],
                expected_git_tree=scientific["sources"]["git_tree"],
                expected_commit_epoch=scientific["sources"]["commit_epoch"],
            )
            dry = manifest.build_dry_manifest(
                scientific_config=scientific,
                aws_local_config=aws_config,
                source_bundle=bundle,
                attempt_id=args.attempt_id,
                parent_manifest_sha256=parent_manifest_sha256,
            )
            manifest.publish_manifest_absent(manifest_output, dry)
        except BaseException:
            if source_output.exists() and not source_output.is_symlink():
                source_output.unlink()
                descriptor = os.open(
                    source_output.parent, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            raise
        return 0
    if args.command == "stage" and args.stage_mode == "training-inputs":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        clients = aws.make_clients(region=plan["infrastructure"]["region"])
        receipt = training_aws.stage_training_inputs_once(
            clients.s3,
            training_plan=plan,
            source_bundle_path=_absolute(args.source_bundle, name="source-bundle"),
            dataset_dir=_absolute(args.dataset_dir, name="dataset-dir"),
            base_model_dir=_absolute(args.base_model_dir, name="base-model-dir"),
            snapshot_manifest_path=_absolute(
                args.snapshot_manifest, name="snapshot-manifest"
            ),
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "preflight" and args.preflight_mode == "runtime-smoke":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        aws_config, _ = config.load_aws_local_config(
            _absolute(args.aws_config, name="aws-config")
        )
        clients = aws.make_clients(region=aws_config["region"])
        receipt = aws.preflight_runtime_smoke(
            clients,
            aws_config,
            remote_image_uri=args.image_uri,
            job_name=args.job_name,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "preflight" and args.preflight_mode == "training":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        staging = _load_receipt(args.staging_receipt)
        clients = aws.make_clients(region=plan["infrastructure"]["region"])
        receipt = training_launch.preflight_training_job(
            clients,
            training_plan=plan,
            staging_receipt=staging,
            run_id=args.run_id,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "submit" and args.submit_mode == "runtime-smoke":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        receipt = _load_receipt(args.preflight_receipt)
        region = receipt.get("region")
        if region != "us-east-1":
            raise ValueError("Runtime-smoke receipt region changed")
        clients = aws.make_clients(region=region)
        submission = aws.submit_runtime_smoke(
            clients,
            preflight_receipt=receipt,
        )
        _publish_json(output, submission)
        return 0
    if args.command == "submit" and args.submit_mode == "training":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        staging = _load_receipt(args.staging_receipt)
        preflight = _load_receipt(args.preflight_receipt)
        clients = aws.make_clients(region=plan["infrastructure"]["region"])
        submission = training_launch.submit_training_job_once(
            clients,
            training_plan=plan,
            staging_receipt=staging,
            preflight_receipt=preflight,
        )
        _publish_json(output, submission)
        return 0
    if args.command == "status" and args.status_mode == "runtime-smoke":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        clients = aws.make_clients(region=args.region)
        _publish_json(
            output,
            aws.describe_runtime_smoke(clients.sagemaker, job_name=args.job_name),
        )
        return 0
    if args.command == "status" and args.status_mode == "training":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        staging = _load_receipt(args.staging_receipt)
        preflight = _load_receipt(args.preflight_receipt)
        submission = _load_receipt(args.submission_receipt)
        clients = aws.make_clients(region=plan["infrastructure"]["region"])
        status = training_launch.describe_training_job_status(
            clients,
            training_plan=plan,
            staging_receipt=staging,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
        _publish_json(output, status)
        return 0
    if args.command == "evaluate" and args.evaluate_mode == "runtime-smoke":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        receipt = _load_receipt(args.preflight_receipt)
        submission = _load_receipt(args.submission_receipt)
        clients = aws.make_clients(region=args.region)
        verified = aws.verify_completed_runtime_smoke(
            clients,
            preflight_receipt=receipt,
            submission_receipt=submission,
        )
        _publish_json(output, verified)
        return 0
    if args.command == "acquire" and args.acquire_mode == "determinism-smoke":
        output = _absolute(args.output_dir, name="output-dir")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        staging = _load_receipt(args.staging_receipt)
        preflight = _load_receipt(args.preflight_receipt)
        submission = _load_receipt(args.submission_receipt)
        terminal = _load_receipt(args.terminal_receipt)
        clients = aws.make_clients(region=plan["infrastructure"]["region"])
        training_artifacts.acquire_completed_determinism_smoke_artifact(
            clients.s3,
            training_plan=plan,
            staging_receipt=staging,
            preflight_receipt=preflight,
            submission_receipt=submission,
            terminal_receipt=terminal,
            output_bundle=output,
        )
        return 0
    if args.command == "fold-processing" and args.fold_mode == "completed-evidence":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        evidence = controlled_supervisor.load_completed_fold_evidence(
            state_dir=_absolute(
                args.supervisor_state_dir,
                name="supervisor-state-dir",
            ),
            outer_fold=args.outer_fold,
        )
        _publish_json(output, evidence)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "stage-static":
        output = _absolute(args.receipt_output, name="receipt-output")
        state_dir = _absolute(args.state_dir, name="state-dir")
        _require_absent_output(output)
        _require_absent_output(state_dir)
        _require_disjoint_paths(
            output,
            state_dir,
            name="Static staging state and receipt outputs",
        )
        completed = _load_completed_fold_evidence(args.completed_evidence)
        clients = _clients_for_completed_fold(completed)
        receipt = fold_processing_aws.stage_static_evaluation_inputs_once(
            clients,
            completed_fold_evidence=completed,
            e5_snapshot_dir=_absolute(
                args.e5_snapshot_dir,
                name="e5-snapshot-dir",
            ),
            e5_snapshot_manifest_path=_absolute(
                args.e5_snapshot_manifest,
                name="e5-snapshot-manifest",
            ),
            e5_pack_dir=_absolute(args.e5_pack_dir, name="e5-pack-dir"),
            fixed_base_dir=_absolute(
                args.fixed_base_dir,
                name="fixed-base-dir",
            ),
            destination_prefix=args.destination_prefix,
            state_dir=state_dir,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "copy-archives":
        output = _absolute(args.receipt_output, name="receipt-output")
        state_dir = _absolute(args.state_dir, name="state-dir")
        _require_absent_output(output)
        _require_absent_output(state_dir)
        _require_disjoint_paths(
            output,
            state_dir,
            name="Archive-copy state and receipt outputs",
        )
        completed = _load_completed_fold_evidence(args.completed_evidence)
        clients = _clients_for_completed_fold(completed)
        receipt = fold_processing_aws.copy_completed_fold_archives_once(
            clients,
            completed_fold_evidence=completed,
            destination_prefix=args.destination_prefix,
            state_dir=state_dir,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase1-preflight":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        completed, archive_copy, static_staging, publication = _load_phase1_context(
            args
        )
        clients = _clients_for_completed_fold(completed)
        receipt = fold_processing_aws.preflight_fold_inventory(
            clients,
            completed_fold_evidence=completed,
            archive_copy_receipt=archive_copy,
            static_staging_receipt=static_staging,
            overlay_publication_receipt=publication,
            job_name=args.job_name,
            output_prefix=args.output_prefix,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase1-submit":
        output = _absolute(args.receipt_output, name="receipt-output")
        state_dir = _absolute(args.state_dir, name="state-dir")
        _require_absent_output(output)
        _require_absent_output(state_dir)
        _require_disjoint_paths(
            output,
            state_dir,
            name="Phase-1 submission state and receipt outputs",
        )
        completed, archive_copy, static_staging, publication = _load_phase1_context(
            args
        )
        preflight = _load_receipt(args.preflight_receipt)
        clients = _clients_for_completed_fold(completed)
        receipt = fold_processing_aws.submit_fold_inventory_once(
            clients,
            preflight_receipt=preflight,
            completed_fold_evidence=completed,
            archive_copy_receipt=archive_copy,
            static_staging_receipt=static_staging,
            overlay_publication_receipt=publication,
            state_dir=state_dir,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase1-status":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        completed, archive_copy, static_staging, publication = _load_phase1_context(
            args
        )
        preflight = fold_processing_aws.validate_fold_inventory_preflight_receipt(
            _load_receipt(args.preflight_receipt),
            completed_fold_evidence=completed,
            archive_copy_receipt=archive_copy,
            static_staging_receipt=static_staging,
            overlay_publication_receipt=publication,
        )
        clients = _clients_for_completed_fold(completed)
        status = fold_processing_aws.describe_fold_inventory(
            clients.sagemaker,
            job_name=preflight["job_name"],
        )
        _publish_json(output, status)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase1-verify":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        completed, archive_copy, static_staging, publication = _load_phase1_context(
            args
        )
        preflight = _load_receipt(args.preflight_receipt)
        submission = _load_receipt(args.submission_receipt)
        clients = _clients_for_completed_fold(completed)
        terminal = fold_processing_aws.verify_completed_fold_inventory(
            clients,
            preflight_receipt=preflight,
            submission_receipt=submission,
            completed_fold_evidence=completed,
            archive_copy_receipt=archive_copy,
            static_staging_receipt=static_staging,
            overlay_publication_receipt=publication,
        )
        _publish_json(output, terminal)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase1-acquire":
        output = _absolute(args.output_dir, name="output-dir")
        _require_absent_output(output)
        completed, archive_copy, static_staging, publication = _load_phase1_context(
            args
        )
        preflight = _load_receipt(args.preflight_receipt)
        submission = _load_receipt(args.submission_receipt)
        terminal = _load_receipt(args.terminal_receipt)
        clients = _clients_for_completed_fold(completed)
        fold_processing_aws.acquire_fold_inventory_once(
            clients,
            terminal_receipt=terminal,
            preflight_receipt=preflight,
            submission_receipt=submission,
            completed_fold_evidence=completed,
            archive_copy_receipt=archive_copy,
            static_staging_receipt=static_staging,
            overlay_publication_receipt=publication,
            output_dir=output,
        )
        return 0
    if args.command == "fold-processing" and args.fold_mode == "storage-proof":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        completed, archive_copy, static_staging, publication = _load_phase1_context(
            args
        )
        proof = fold_evaluation_aws.build_phase2_storage_proof(
            control_bundle_receipt=_load_receipt(args.control_bundle_receipt),
            control_bundle_dir=_absolute(
                args.control_bundle_dir,
                name="control-bundle-dir",
            ),
            acquisition_receipt=_load_receipt(args.acquisition_receipt),
            acquisition_dir=_absolute(
                args.acquisition_dir,
                name="acquisition-dir",
            ),
            terminal_receipt=_load_receipt(args.terminal_receipt),
            preflight_receipt=_load_receipt(args.preflight_receipt),
            submission_receipt=_load_receipt(args.submission_receipt),
            completed_fold_evidence=completed,
            archive_copy_receipt=archive_copy,
            static_staging_receipt=static_staging,
            phase1_overlay_publication_receipt=publication,
            phase2_overlay_publication_receipt=_load_receipt(
                args.phase2_overlay_publication_receipt
            ),
            phase2_output_reserve_bytes=args.phase2_output_reserve_bytes,
            safety_reserve_bytes=args.safety_reserve_bytes,
        )
        _publish_json(output, proof)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-controls":
        output = _absolute(args.output_dir, name="output-dir")
        _require_absent_output(output)
        context = _load_phase2_evidence(args)
        fold_evaluation_aws.build_phase2_control_bundle(
            **context,
            static_control_dir=_absolute(
                args.static_control_dir,
                name="static-control-dir",
            ),
            output_dir=output,
        )
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-stage-controls":
        output = _absolute(args.receipt_output, name="receipt-output")
        state = _absolute(args.state_dir, name="state-dir")
        _require_absent_output(output)
        _require_disjoint_paths(
            output,
            state,
            name="Phase-2 control staging state and receipt outputs",
        )
        context = _load_phase2_evidence(args)
        fold_evaluation_aws._validate_context(
            **{
                key: value
                for key, value in context.items()
                if key != "phase1_acquisition_dir"
            }
        )
        clients = _clients_for_completed_fold(context["completed_fold_evidence"])
        receipt = fold_evaluation_aws.stage_phase2_controls_once(
            clients,
            control_bundle_receipt=_load_receipt(args.control_bundle_receipt),
            control_bundle_dir=_absolute(
                args.control_bundle_dir, name="control-bundle-dir"
            ),
            completed_fold_evidence=context["completed_fold_evidence"],
            archive_copy_receipt=context["archive_copy_receipt"],
            static_staging_receipt=context["static_staging_receipt"],
            phase1_overlay_publication_receipt=context[
                "phase1_overlay_publication_receipt"
            ],
            phase2_overlay_publication_receipt=context[
                "phase2_overlay_publication_receipt"
            ],
            phase1_preflight_receipt=context["phase1_preflight_receipt"],
            phase1_submission_receipt=context["phase1_submission_receipt"],
            phase1_terminal_receipt=context["phase1_terminal_receipt"],
            phase1_acquisition_receipt=context["phase1_acquisition_receipt"],
            phase1_acquisition_dir=context["phase1_acquisition_dir"],
            destination_prefix=args.destination_prefix,
            state_dir=state,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-preflight":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        context = _load_phase2_context(args)
        clients = _clients_for_completed_fold(context["completed_fold_evidence"])
        receipt = fold_evaluation_aws.preflight_phase2_evaluation(
            clients,
            **context,
            job_name=args.job_name,
            output_prefix=args.output_prefix,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-submit":
        output = _absolute(args.receipt_output, name="receipt-output")
        state = _absolute(args.state_dir, name="state-dir")
        _require_absent_output(output)
        _require_disjoint_paths(
            output,
            state,
            name="Phase-2 submission state and receipt outputs",
        )
        context = _load_phase2_context(args)
        clients = _clients_for_completed_fold(context["completed_fold_evidence"])
        receipt = fold_evaluation_aws.submit_phase2_evaluation_once(
            clients,
            preflight_receipt=_load_receipt(args.preflight_receipt),
            **context,
            state_dir=state,
        )
        _publish_json(output, receipt)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-status":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        context = _load_phase2_context(args)
        preflight = fold_evaluation_aws.validate_phase2_preflight_receipt(
            _load_receipt(args.preflight_receipt), **context
        )
        clients = _clients_for_completed_fold(context["completed_fold_evidence"])
        status = fold_evaluation_aws.describe_phase2_evaluation(
            clients.sagemaker, job_name=preflight["job_name"]
        )
        _publish_json(output, status)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-verify":
        output = _absolute(args.receipt_output, name="receipt-output")
        _require_absent_output(output)
        context = _load_phase2_context(args)
        clients = _clients_for_completed_fold(context["completed_fold_evidence"])
        terminal = fold_evaluation_aws.verify_completed_phase2_evaluation(
            clients,
            preflight_receipt=_load_receipt(args.preflight_receipt),
            submission_receipt=_load_receipt(args.submission_receipt),
            **context,
        )
        _publish_json(output, terminal)
        return 0
    if args.command == "fold-processing" and args.fold_mode == "phase2-acquire":
        output = _absolute(args.output_dir, name="output-dir")
        _require_absent_output(output)
        context = _load_phase2_context(args)
        clients = _clients_for_completed_fold(context["completed_fold_evidence"])
        fold_evaluation_aws.acquire_phase2_evaluation_once(
            clients,
            terminal_receipt=_load_receipt(args.terminal_receipt),
            preflight_receipt=_load_receipt(args.preflight_receipt),
            submission_receipt=_load_receipt(args.submission_receipt),
            **context,
            output_dir=output,
        )
        return 0
    if args.command == "aggregate":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        directories = [
            _absolute(path, name="evaluation-dir") for path in args.evaluation_dir
        ]
        _publish_json(
            output,
            aggregate.build_evaluation_index(
                directories,
                dataset_dir=_absolute(args.dataset_dir, name="dataset-dir"),
                fold_manifest_path=_absolute(
                    args.fold_manifest, name="fold-manifest"
                ),
            ),
        )
        return 0
    if args.command == "verify" and args.verify_mode == "training-plan":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        value, digest = manifest.read_manifest(
            _absolute(args.manifest, name="manifest")
        )
        _publish_json(
            output,
            {
                "controlled_runs": len(value["controlled_runs"]),
                "manifest_sha256": digest,
                "manifest_type": value["manifest_type"],
                "protocol": "retrieval_cv_training_plan_verification_v1",
                "total_runs": len(value["controlled_runs"])
                + len(value["auxiliary_runs"]),
            },
        )
        return 0
    if args.command == "verify" and args.verify_mode == "training":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        staging = _load_receipt(args.staging_receipt)
        preflight = _load_receipt(args.preflight_receipt)
        submission = _load_receipt(args.submission_receipt)
        clients = aws.make_clients(region=plan["infrastructure"]["region"])
        terminal = training_launch.verify_terminal_training_job(
            clients,
            training_plan=plan,
            staging_receipt=staging,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
        _publish_json(output, terminal)
        if terminal["succeeded"] is not True:
            raise RuntimeError(
                "Training job ended unsuccessfully: "
                f"{terminal['terminal_status']}"
            )
        return 0
    if args.command == "verify" and args.verify_mode == "determinism-smoke":
        output = _absolute(args.output, name="output")
        _require_absent_output(output)
        plan = _load_training_plan(args.manifest)
        staging = _load_receipt(args.staging_receipt)
        receipt = determinism_gate.run_determinism_gate(
            training_plan=plan,
            staging_receipt=staging,
            acquisition_receipt_paths_by_run={
                "determinism-smoke-a": _absolute(
                    args.acquisition_receipt_a,
                    name="acquisition-receipt-a",
                ),
                "determinism-smoke-b": _absolute(
                    args.acquisition_receipt_b,
                    name="acquisition-receipt-b",
                ),
            },
        )
        _publish_json(output, receipt)
        return 0
    raise ValueError(f"Unsupported command/mode: {args}")


if __name__ == "__main__":
    raise SystemExit(main())
