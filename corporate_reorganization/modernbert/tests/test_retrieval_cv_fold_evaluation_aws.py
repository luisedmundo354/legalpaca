from __future__ import annotations

import copy
import hashlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from corporate_reorganization.modernbert.experiments.retrieval_cv import aws
from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    fold_evaluation_aws as phase2,
)
from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    fold_processing_aws as phase1,
)
from corporate_reorganization.modernbert.tests.test_processing_fold_image_contract import (
    _portable_runtime_identity,
)


ACCOUNT_ID = "371087393859"
REGION = "us-east-1"
REPOSITORY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/arr-retrieval-eval"


def _phase2_runtime_identity() -> dict[str, object]:
    identity = _portable_runtime_identity(
        context_identity=phase2.PHASE2_OVERLAY_BUILD_IDENTITY,
        files_sha256=phase2.PHASE2_OVERLAY_FILES_IDENTITY,
    )
    identity["build_context"].update(
        source_parent_commit="dfa1a52833df88b78b4b4349bec8b7ab3ead9ab8",
        source_parent_epoch=1_783_977_591,
        source_parent_rfc3339="2026-07-13T21:19:51Z",
        toolchain={
            "builder_driver": "docker",
            "buildkit_version": "v0.29.0",
            "buildx_version": "v0.33.0",
        },
    )
    return identity


def _phase2_publication() -> dict[str, object]:
    runtime_identity = _phase2_runtime_identity()
    return {
        "content_tag": phase2.PHASE2_OVERLAY_CONTENT_TAG,
        "identity": {
            "build_context_files_sha256": phase2.PHASE2_OVERLAY_FILES_IDENTITY,
            "build_context_identity_sha256": (
                phase2.PHASE2_OVERLAY_BUILD_IDENTITY
            ),
            "config_digest": phase2.PHASE2_OVERLAY_CONFIG_DIGEST,
            "image_digest": phase2.PHASE2_OVERLAY_IMAGE_DIGEST,
            "image_runtime_identity": runtime_identity,
            "local_image_identity_sha256": (
                phase2.PHASE2_OVERLAY_LOCAL_IMAGE_IDENTITY
            ),
            "manifest_media_type": aws.ECR_MEDIA_TYPE,
            "offline_smoke_sha256": (
                phase2.PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256
            ),
        },
        "manifest_digest": phase2.PHASE2_OVERLAY_IMAGE_DIGEST,
        "media_type": aws.ECR_MEDIA_TYPE,
        "protocol": phase2.PHASE2_OVERLAY_PUBLICATION_PROTOCOL,
        "raw_manifest_sha256": phase2.PHASE2_OVERLAY_IMAGE_DIGEST.removeprefix(
            "sha256:"
        ),
        "remote_digest_uri": f"{REPOSITORY}@{phase2.PHASE2_OVERLAY_IMAGE_DIGEST}",
        "remote_tag_uri": f"{REPOSITORY}:{phase2.PHASE2_OVERLAY_CONTENT_TAG}",
    }


def _phase1_publication() -> dict[str, object]:
    content_tag = f"fold-build-sha256-{phase1.FOLD_OVERLAY_BUILD_IDENTITY}"
    return {
        "content_tag": content_tag,
        "identity": {
            "build_context_files_sha256": (
                "e1dcee02f089d396ce74934b271dc90f2de8567cedfa6070077d6b85a86b29aa"
            ),
            "build_context_identity_sha256": phase1.FOLD_OVERLAY_BUILD_IDENTITY,
            "config_digest": phase1.FOLD_OVERLAY_CONFIG_DIGEST,
            "image_digest": phase1.FOLD_OVERLAY_IMAGE_DIGEST,
            "local_image_identity_sha256": (
                "6efc58408e86e7a464f39b851f1c370d0b33754a4deceabc5123fe33ad9218dc"
            ),
            "manifest_media_type": aws.ECR_MEDIA_TYPE,
            "offline_smoke_sha256": phase1.FOLD_OVERLAY_OFFLINE_SMOKE_SHA256,
        },
        "manifest_digest": phase1.FOLD_OVERLAY_IMAGE_DIGEST,
        "media_type": aws.ECR_MEDIA_TYPE,
        "protocol": phase1.FOLD_OVERLAY_PUBLICATION_PROTOCOL,
        "raw_manifest_sha256": phase1.FOLD_OVERLAY_IMAGE_DIGEST.removeprefix(
            "sha256:"
        ),
        "remote_digest_uri": f"{REPOSITORY}@{phase1.FOLD_OVERLAY_IMAGE_DIGEST}",
        "remote_tag_uri": f"{REPOSITORY}:{content_tag}",
    }


def _clients() -> aws.AwsClients:
    return aws.AwsClients(
        sts=Mock(),
        iam=Mock(),
        ecr=Mock(),
        s3=Mock(),
        service_quotas=Mock(),
        ec2=Mock(),
        sagemaker=Mock(),
        logs=Mock(),
    )


def _phase2_context_kwargs() -> dict[str, object]:
    return {
        "completed_fold_evidence": {},
        "archive_copy_receipt": {},
        "static_staging_receipt": {},
        "phase1_overlay_publication_receipt": {},
        "phase2_overlay_publication_receipt": {},
        "phase1_preflight_receipt": {},
        "phase1_submission_receipt": {},
        "phase1_terminal_receipt": {},
        "phase1_acquisition_receipt": {},
        "phase1_acquisition_dir": Path("/phase1-acquisition"),
        "control_bundle_receipt": {},
        "control_bundle_dir": Path("/phase2-controls"),
        "control_staging_receipt": {},
        "storage_proof": {},
    }


class RetrievalCvFoldEvaluationAwsTest(unittest.TestCase):
    def test_phase_publications_are_exact_and_cross_phase_rejected(self) -> None:
        old = _phase1_publication()
        new = _phase2_publication()
        self.assertEqual(
            phase1._validate_overlay_publication(
                copy.deepcopy(old), account_id=ACCOUNT_ID, region=REGION
            ),
            old,
        )
        self.assertEqual(
            phase2._validate_phase2_overlay_publication(
                copy.deepcopy(new), account_id=ACCOUNT_ID, region=REGION
            ),
            new,
        )
        with self.assertRaises(ValueError):
            phase2._validate_phase2_overlay_publication(
                copy.deepcopy(old), account_id=ACCOUNT_ID, region=REGION
            )
        with self.assertRaises(ValueError):
            phase1._validate_overlay_publication(
                copy.deepcopy(new), account_id=ACCOUNT_ID, region=REGION
            )

    def test_phase2_runtime_identity_rejects_rehashed_splices(self) -> None:
        publication = _phase2_publication()
        runtime = publication["identity"]["image_runtime_identity"]
        runtime_hash = hashlib.sha256(phase2._canonical_bytes(runtime)[:-1]).hexdigest()
        self.assertEqual(runtime_hash, phase2.PHASE2_OVERLAY_OFFLINE_SMOKE_SHA256)

        for label, mutate in (
            (
                "raw-path",
                lambda value: value["inherited_runtime"]["sparse_runtime"].__setitem__(
                    "java_home", "/etc/passwd"
                ),
            ),
            (
                "extra-field",
                lambda value: value.__setitem__("unexpected", "same-in-both"),
            ),
            (
                "nested-build",
                lambda value: value["build_context"].__setitem__(
                    "build_identity_sha256", "f" * 64
                ),
            ),
        ):
            changed = copy.deepcopy(publication)
            changed_runtime = changed["identity"]["image_runtime_identity"]
            mutate(changed_runtime)
            changed["identity"]["offline_smoke_sha256"] = hashlib.sha256(
                phase2._canonical_bytes(changed_runtime)[:-1]
            ).hexdigest()
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "publication identity changed"
            ):
                phase2._validate_phase2_overlay_publication(
                    changed, account_id=ACCOUNT_ID, region=REGION
                )

    def test_validate_context_keeps_phase1_and_phase2_publications_disjoint(
        self,
    ) -> None:
        raw_completed = {"raw": "completed"}
        completed = {
            "training_plan": {
                "infrastructure": {
                    "account_id": ACCOUNT_ID,
                    "region": REGION,
                }
            }
        }
        raw_old = {"raw": "old-publication"}
        raw_new = {"raw": "new-publication"}
        old = {"validated": "old-publication"}
        new = {"validated": "new-publication"}
        archive = {"validated": "archive"}
        static = {"validated": "static"}
        preflight = {"validated": "phase1-preflight"}
        submission = {"validated": "phase1-submission"}
        terminal = {"validated": "phase1-terminal"}
        acquisition = {"validated": "phase1-acquisition"}

        with (
            patch.object(
                phase2.controlled_supervisor,
                "validate_completed_fold_evidence",
                return_value=completed,
            ),
            patch.object(
                phase1,
                "validate_fold_archive_copy_receipt",
                return_value=archive,
            ),
            patch.object(
                phase1,
                "validate_static_evaluation_staging_receipt",
                return_value=static,
            ),
            patch.object(
                phase1,
                "_validate_overlay_publication",
                return_value=old,
            ) as validate_old,
            patch.object(
                phase2,
                "_validate_phase2_overlay_publication",
                return_value=new,
            ) as validate_new,
            patch.object(
                phase1,
                "validate_fold_inventory_preflight_receipt",
                return_value=preflight,
            ) as validate_preflight,
            patch.object(
                phase1,
                "validate_fold_inventory_submission_receipt",
                return_value=submission,
            ) as validate_submission,
            patch.object(
                phase1,
                "validate_fold_inventory_terminal_receipt",
                return_value=terminal,
            ) as validate_terminal,
            patch.object(
                phase1,
                "validate_fold_inventory_acquisition_receipt",
                return_value=acquisition,
            ) as validate_acquisition,
        ):
            result = phase2._validate_context(
                completed_fold_evidence=raw_completed,
                archive_copy_receipt={"raw": "archive"},
                static_staging_receipt={"raw": "static"},
                phase1_overlay_publication_receipt=raw_old,
                phase2_overlay_publication_receipt=raw_new,
                phase1_preflight_receipt={"raw": "preflight"},
                phase1_submission_receipt={"raw": "submission"},
                phase1_terminal_receipt={"raw": "terminal"},
                phase1_acquisition_receipt={"raw": "acquisition"},
            )

        self.assertEqual(result[3:5], (old, new))
        validate_old.assert_called_once_with(
            raw_old, account_id=ACCOUNT_ID, region=REGION
        )
        validate_new.assert_called_once_with(
            raw_new, account_id=ACCOUNT_ID, region=REGION
        )
        for validator in (
            validate_preflight,
            validate_submission,
            validate_terminal,
            validate_acquisition,
        ):
            self.assertIs(
                validator.call_args.kwargs["overlay_publication_receipt"], old
            )
            self.assertNotIn(
                "phase2_overlay_publication_receipt",
                validator.call_args.kwargs,
            )

    def test_phase2_request_has_exact_image_mounts_and_isolation(self) -> None:
        completed = {
            "outer_fold": 0,
            "attempt_id": "a3",
            "training_plan": {
                "infrastructure": {
                    "artifact_bucket": "ir-sagemaker",
                    "processing_instance_count": 1,
                    "processing_instance_type": "ml.g5.12xlarge",
                    "processing_volume_size_gb": 100,
                    "role_arn": (
                        "arn:aws:iam::371087393859:role/"
                        "AmazonSageMakerExecutionRole"
                    ),
                },
                "controlled_runs": [
                    {
                        "input_channels": {
                            "data": {
                                "s3_uri": (
                                    "s3://ir-sagemaker/arr-retrieval-cv/"
                                    "corrected-data/"
                                )
                            }
                        }
                    }
                ],
            },
        }
        archive = {
            "destination_prefix": "arr-retrieval-cv/fold-0/archives/",
            "copy_set_receipt": {
                "systems": [
                    {
                        "destination_object": {
                            "encryption": {"kms_key_id": "kms-key"}
                        }
                    }
                    for _ in range(12)
                ]
            },
        }
        static = {
            "assets": [
                {
                    "name": name,
                    "s3_prefix": f"arr-retrieval-cv/static/{name}/",
                }
                for name in ("e5-snapshot", "e5-pack", "fixed-base")
            ]
        }
        publication = _phase2_publication()
        request = phase2._render_phase2_request(
            completed=completed,
            archive_copy=archive,
            static_staging=static,
            publication=publication,
            phase1_preflight={
                "output_prefix": "arr-retrieval-cv/fold-0/inventory/"
            },
            control_staging={
                "input_prefix": "arr-retrieval-cv/fold-0/phase2-controls/input/"
            },
            job_name="arr-ret-cv1-f0-evaluate-a3",
            output_prefix="arr-retrieval-cv/fold-0/evaluation/",
        )

        self.assertEqual(
            request["AppSpecification"]["ImageUri"],
            publication["remote_digest_uri"],
        )
        self.assertEqual(
            request["AppSpecification"]["ContainerArguments"],
            [
                "--evaluation-plan",
                "/opt/ml/processing/input/control/evaluation_plan.json",
                "--local-bindings",
                "/opt/ml/processing/input/control/local_bindings.json",
                "--output-dir",
                "/opt/ml/processing/output/evaluation",
                "--device",
                "cuda:0",
            ],
        )
        self.assertEqual(
            request["AppSpecification"]["ContainerEntrypoint"],
            [
                "/opt/conda/bin/python",
                "/opt/program/modernbert/processing_fold_eval/evaluate_sm.py",
            ],
        )
        self.assertEqual(
            [record["InputName"] for record in request["ProcessingInputs"]],
            [
                "fold-archives",
                "dataset",
                "fold-inventory",
                "control",
                "e5-snapshot",
                "e5-pack",
                "fixed-base",
            ],
        )
        self.assertEqual(
            [record["S3Input"]["LocalPath"] for record in request["ProcessingInputs"]],
            [
                "/opt/ml/processing/input/fold-archives",
                "/opt/ml/processing/input/dataset",
                "/opt/ml/processing/input/fold-inventory",
                "/opt/ml/processing/input/control",
                "/opt/ml/processing/input/e5-snapshot",
                "/opt/ml/processing/input/e5-pack",
                "/opt/ml/processing/input/fixed-base",
            ],
        )
        self.assertIs(request["NetworkConfig"]["EnableNetworkIsolation"], True)
        self.assertEqual(
            request["ProcessingOutputConfig"]["Outputs"],
            [
                {
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri": (
                            "s3://ir-sagemaker/arr-retrieval-cv/"
                            "fold-0/evaluation/"
                        ),
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "fold archives"):
            phase2._render_phase2_request(
                completed=completed,
                archive_copy=archive,
                static_staging=static,
                publication=publication,
                phase1_preflight={
                    "output_prefix": "arr-retrieval-cv/fold-0/inventory/"
                },
                control_staging={
                    "input_prefix": (
                        "arr-retrieval-cv/fold-0/phase2-controls/input/"
                    )
                },
                job_name="arr-ret-cv1-f0-evaluate-a3",
                output_prefix="arr-retrieval-cv/fold-0/archives/nested/",
            )

    def test_phase2_submission_is_one_shot_and_persists_intent_first(self) -> None:
        clients = _clients()
        job_name = "arr-ret-cv1-f0-evaluate-a3"
        job_arn = (
            "arn:aws:sagemaker:us-east-1:371087393859:"
            f"processing-job/{job_name}"
        )
        preflight = {
            "outer_fold": 0,
            "account_id": ACCOUNT_ID,
            "region": REGION,
            "caller_arn": f"arn:aws:iam::{ACCOUNT_ID}:user/tester",
            "job_name": job_name,
            "output_prefix": "arr-retrieval-cv/fold-0/evaluation/",
            "request": {
                "ProcessingJobName": job_name,
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "S3Output": {
                                "S3Uri": (
                                    "s3://ir-sagemaker/arr-retrieval-cv/"
                                    "fold-0/evaluation/"
                                )
                            }
                        }
                    ]
                },
            },
            "request_sha256": "a" * 64,
        }
        clients.sts.get_caller_identity.return_value = {
            "Account": ACCOUNT_ID,
            "Arn": preflight["caller_arn"],
        }
        clients.sagemaker.list_processing_jobs.return_value = {
            "ProcessingJobSummaries": []
        }
        clients.sagemaker.create_processing_job.return_value = {
            "ProcessingJobArn": job_arn
        }
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary).resolve() / "submission"
            with (
                patch.object(
                    phase2,
                    "validate_phase2_preflight_receipt",
                    return_value=preflight,
                ),
                patch.object(
                    phase2,
                    "preflight_phase2_evaluation",
                    return_value=preflight,
                ),
                patch.object(
                    phase2,
                    "validate_phase2_submission_receipt",
                    side_effect=lambda value, **_: copy.deepcopy(value),
                ),
                patch.object(phase2.aws, "assert_unused_versioned_prefix"),
            ):
                receipt = phase2.submit_phase2_evaluation_once(
                    clients,
                    preflight_receipt=preflight,
                    **_phase2_context_kwargs(),
                    state_dir=state,
                )

            self.assertEqual(receipt["job_arn"], job_arn)
            clients.sagemaker.create_processing_job.assert_called_once_with(
                **preflight["request"]
            )
            self.assertEqual(
                sorted(path.name for path in state.iterdir()),
                ["create-intent.json", "state.json", "submission.json"],
            )

    def test_generated_controls_are_independently_rerendered(self) -> None:
        expected_plan = {"plan": "expected"}
        expected_bindings = {"bindings": "expected"}
        frozen_names = {
            "e5_snapshot.json",
            "evaluation_baselines.json",
            "experiment.json",
            "folds.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            def write_bundle(plan: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
                payloads = {
                    name: phase2._canonical_bytes({"frozen": name})
                    for name in frozen_names
                }
                payloads.update(
                    {
                        "evaluation_plan.json": phase2._canonical_bytes(plan),
                        "local_bindings.json": phase2._canonical_bytes(
                            expected_bindings
                        ),
                    }
                )
                files = [
                    {
                        "path": name,
                        "size": len(payloads[name]),
                        "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                    }
                    for name in phase2.CONTROL_FILE_NAMES
                ]
                bundle = {"files": files}
                for name, payload in payloads.items():
                    (root / name).write_bytes(payload)
                (root / "control_bundle_receipt.json").write_bytes(
                    phase2.strict_config.canonical_json_bytes(bundle)
                )
                static = {
                    "assets": [
                        {
                            "name": "control",
                            "files": [
                                copy.deepcopy(record)
                                for record in files
                                if record["path"] in frozen_names
                            ],
                        }
                    ]
                }
                return bundle, static

            bundle, static = write_bundle(expected_plan)
            with (
                patch.object(
                    phase1,
                    "_load_phase1_acquisition_files",
                    return_value={
                        "archive_inventory.json": {},
                        "bm25_storage.json": {},
                        "artifact_manifest.json": {},
                    },
                ),
                patch.object(
                    phase1,
                    "_validate_phase1_documents",
                    return_value=({}, {}, {}),
                ),
                patch.object(
                    phase2,
                    "_render_controls",
                    return_value=(expected_plan, expected_bindings),
                ),
            ):
                phase2._validate_phase2_control_content(
                    control_bundle_dir=root,
                    bundle=bundle,
                    completed={},
                    archive={},
                    static=static,
                    phase2_publication={
                        "identity": {"image_runtime_identity": {}}
                    },
                    acquisition={},
                    phase1_acquisition_dir=Path("/phase1-acquisition"),
                )

                changed_bundle, changed_static = write_bundle(
                    {"plan": "edited-and-rehashed"}
                )
                with self.assertRaisesRegex(ValueError, "re-rendering"):
                    phase2._validate_phase2_control_content(
                        control_bundle_dir=root,
                        bundle=changed_bundle,
                        completed={},
                        archive={},
                        static=changed_static,
                        phase2_publication={
                            "identity": {"image_runtime_identity": {}}
                        },
                        acquisition={},
                        phase1_acquisition_dir=Path("/phase1-acquisition"),
                    )

    def test_phase2_terminal_readback_and_timing_are_exact(self) -> None:
        clients = _clients()
        job_name = "arr-ret-cv1-f0-evaluate-a3"
        job_arn = (
            "arn:aws:sagemaker:us-east-1:371087393859:"
            f"processing-job/{job_name}"
        )
        request = {
            "AppSpecification": {"ImageUri": "image@sha256:digest"},
            "Environment": {"HF_HUB_OFFLINE": "1"},
            "NetworkConfig": {"EnableNetworkIsolation": True},
            "ProcessingInputs": [
                {
                    "InputName": "control",
                    "S3Input": {
                        "S3Uri": "s3://bucket/control/",
                        "LocalPath": "/opt/ml/processing/input/control",
                    },
                }
            ],
            "ProcessingOutputConfig": {
                "KmsKeyId": "kms-key",
                "Outputs": [
                    {
                        "OutputName": "results",
                        "S3Output": {
                            "S3Uri": "s3://bucket/output/",
                            "LocalPath": "/opt/ml/processing/output",
                        },
                    }
                ],
            },
            "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1}},
            "RoleArn": "role",
            "StoppingCondition": {"MaxRuntimeInSeconds": 86_400},
        }
        preflight = {
            "outer_fold": 0,
            "account_id": ACCOUNT_ID,
            "job_name": job_name,
            "request": request,
            "request_sha256": "b" * 64,
        }
        submission = {"job_arn": job_arn}
        start = datetime(2026, 7, 13, 20, 0, 0, 123_456, tzinfo=timezone.utc)
        end = datetime(2026, 7, 13, 20, 0, 2, 358_023, tzinfo=timezone.utc)
        response = {
            "ProcessingJobName": job_name,
            "ProcessingJobArn": job_arn,
            "ProcessingJobStatus": "Completed",
            "FailureReason": None,
            "ExitMessage": "complete",
            "ProcessingStartTime": start,
            "ProcessingEndTime": end,
            **{
                field: copy.deepcopy(request[field])
                for field in (
                    "AppSpecification",
                    "Environment",
                    "NetworkConfig",
                    "ProcessingResources",
                    "RoleArn",
                    "StoppingCondition",
                )
            },
        }
        response["ProcessingInputs"] = copy.deepcopy(request["ProcessingInputs"])
        response["ProcessingInputs"][0]["AppManaged"] = False
        response["ProcessingOutputConfig"] = copy.deepcopy(
            request["ProcessingOutputConfig"]
        )
        response["ProcessingOutputConfig"]["Outputs"][0]["AppManaged"] = False
        clients.sts.get_caller_identity.return_value = {"Account": ACCOUNT_ID}
        clients.sagemaker.describe_processing_job.return_value = response
        with (
            patch.object(
                phase2,
                "validate_phase2_preflight_receipt",
                return_value=preflight,
            ),
            patch.object(
                phase2,
                "validate_phase2_submission_receipt",
                return_value=submission,
            ),
            patch.object(
                phase2,
                "validate_phase2_terminal_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
        ):
            terminal = phase2.verify_completed_phase2_evaluation(
                clients,
                preflight_receipt=preflight,
                submission_receipt=submission,
                **_phase2_context_kwargs(),
            )
            self.assertEqual(terminal["processing_time_microseconds"], 2_234_567)

            changed = copy.deepcopy(response)
            changed["ProcessingInputs"][0]["AppManaged"] = True
            clients.sagemaker.describe_processing_job.return_value = changed
            with self.assertRaisesRegex(RuntimeError, "unexpected service default"):
                phase2.verify_completed_phase2_evaluation(
                    clients,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    **_phase2_context_kwargs(),
                )

        exact_terminal = phase2._seal(
            {
                "schema_version": 1,
                "protocol": phase2.PHASE2_TERMINAL_PROTOCOL,
                "outer_fold": 0,
                "job_name": job_name,
                "job_arn": job_arn,
                "preflight_receipt_sha256": phase2._document_sha256(preflight),
                "submission_receipt_sha256": phase2._document_sha256(submission),
                "request_sha256": preflight["request_sha256"],
                "status": "Completed",
                "failure_reason": None,
                "processing_start_time": phase1._normalize_datetime(
                    start, name="start"
                ),
                "processing_end_time": phase1._normalize_datetime(end, name="end"),
                "processing_time_microseconds": 2_234_567,
                "exit_message": "complete",
            }
        )
        with (
            patch.object(
                phase2,
                "validate_phase2_preflight_receipt",
                return_value=preflight,
            ),
            patch.object(
                phase2,
                "validate_phase2_submission_receipt",
                return_value=submission,
            ),
        ):
            self.assertEqual(
                phase2.validate_phase2_terminal_receipt(
                    exact_terminal,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    **_phase2_context_kwargs(),
                ),
                exact_terminal,
            )
            changed = copy.deepcopy(exact_terminal)
            changed["processing_time_microseconds"] += 1
            changed = phase2._seal(
                {key: value for key, value in changed.items() if key != "receipt_sha256"}
            )
            with self.assertRaisesRegex(ValueError, "timing evidence"):
                phase2.validate_phase2_terminal_receipt(
                    changed,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    **_phase2_context_kwargs(),
                )

    def test_phase2_acquisition_streams_exact_six_objects_atomically(self) -> None:
        clients = _clients()
        bucket = "ir-sagemaker"
        prefix = "arr-retrieval-cv/fold-0/evaluation/"
        kms_key = "kms-key"
        payloads = {
            path: phase2._canonical_bytes({"path": path})
            for path in phase2.PHASE2_OUTPUT_PATHS
        }
        versions = [
            {
                "Key": prefix + path,
                "VersionId": f"version-{index}",
                "Size": len(payloads[path]),
                "ETag": '"' + f"{index:032x}" + '"',
                "IsLatest": True,
            }
            for index, path in enumerate(phase2.PHASE2_OUTPUT_PATHS)
        ]
        history = {"versions": versions, "delete_markers": []}
        by_key = {record["Key"]: record for record in versions}

        def head_object(**request: object) -> dict[str, object]:
            record = by_key[request["Key"]]
            return {
                "ContentLength": record["Size"],
                "ETag": record["ETag"],
                "VersionId": record["VersionId"],
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": kms_key,
                "BucketKeyEnabled": True,
            }

        def get_object(**request: object) -> dict[str, object]:
            relative = str(request["Key"]).removeprefix(prefix)
            return {"Body": io.BytesIO(payloads[relative])}

        clients.s3.head_object.side_effect = head_object
        clients.s3.get_object.side_effect = get_object
        preflight = {
            "outer_fold": 0,
            "account_id": ACCOUNT_ID,
            "output_prefix": prefix,
            "request": {
                "ProcessingOutputConfig": {
                    "KmsKeyId": kms_key,
                    "Outputs": [
                        {
                            "S3Output": {
                                "S3Uri": f"s3://{bucket}/{prefix}"
                            }
                        }
                    ],
                }
            },
        }
        terminal = {"terminal": "validated"}
        bundle = {"outer_fold": 0}
        proof = {
            "components": {
                "phase2_output_reserve_bytes": sum(map(len, payloads.values()))
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "acquired"
            with (
                patch.object(
                    phase2,
                    "validate_phase2_terminal_receipt",
                    return_value=terminal,
                ),
                patch.object(
                    phase2,
                    "validate_phase2_preflight_receipt",
                    return_value=preflight,
                ),
                patch.object(
                    phase2,
                    "validate_phase2_control_bundle_receipt",
                    return_value=bundle,
                ),
                patch.object(
                    phase1,
                    "validate_fold_archive_copy_receipt",
                    return_value={"archive": "validated"},
                ),
                patch.object(
                    phase1,
                    "validate_fold_storage_proof",
                    return_value=proof,
                ),
                patch.object(
                    phase1,
                    "_list_prefix_history",
                    side_effect=[copy.deepcopy(history), copy.deepcopy(history)],
                ),
                patch.object(
                    phase2,
                    "_validate_phase2_output_tree",
                    return_value=("1" * 64, "2" * 64),
                ) as validate_tree,
                patch.object(
                    phase2,
                    "validate_phase2_acquisition_receipt",
                    side_effect=lambda value, **_: copy.deepcopy(value),
                ),
            ):
                receipt = phase2.acquire_phase2_evaluation_once(
                    clients,
                    terminal_receipt={},
                    preflight_receipt={},
                    submission_receipt={},
                    **_phase2_context_kwargs(),
                    output_dir=output,
                )

            self.assertEqual(
                [record["path"] for record in receipt["files"]],
                list(phase2.PHASE2_OUTPUT_PATHS),
            )
            self.assertEqual(len(receipt["remote_objects"]), 6)
            self.assertEqual(clients.s3.head_object.call_count, 6)
            self.assertEqual(clients.s3.get_object.call_count, 6)
            validate_tree.assert_called_once()
            for path, payload in payloads.items():
                self.assertEqual((output / path).read_bytes(), payload)
            self.assertTrue((output / "acquisition_receipt.json").is_file())
            self.assertFalse(
                output.with_name(f".{output.name}.incomplete").exists()
            )


if __name__ == "__main__":
    unittest.main()
