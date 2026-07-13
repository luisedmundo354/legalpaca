"""Fail-loud command line entry point for immutable retrieval-CV orchestration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from . import aggregate, aws, config, folds, manifest, training_aws, training_launch


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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _load_receipt(path: Path) -> dict[str, Any]:
    value, _ = config.load_canonical_json_object(_absolute(path, name="receipt"))
    return value


def _load_training_plan(path: Path) -> dict[str, Any]:
    value, _ = manifest.read_manifest(_absolute(path, name="manifest"))
    return value


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
    raise ValueError(f"Unsupported command/mode: {args}")


if __name__ == "__main__":
    raise SystemExit(main())
