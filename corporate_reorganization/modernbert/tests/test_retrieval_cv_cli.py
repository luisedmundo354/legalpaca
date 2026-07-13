from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from corporate_reorganization.modernbert.experiments.retrieval_cv import cli, config


class RetrievalCvCliTest(unittest.TestCase):
    def test_required_commands_and_runtime_modes_parse(self) -> None:
        fixtures = [
            [
                "build-data",
                "--raw-dir",
                "/raw",
                "--tokenizer-dir",
                "/tokenizer",
                "--output-dir",
                "/output",
            ],
            [
                "freeze-folds",
                "--dataset-dir",
                "/data",
                "--output",
                "/folds.json",
            ],
            [
                "build-eval-image",
                "--frozen-context",
                "/context",
                "--metadata-file",
                "/metadata.json",
                "--build-replica",
                "1",
                "--receipt-output",
                "/receipt.json",
            ],
            [
                "publish-eval-image",
                "--aws-config",
                "/aws.json",
                "--receipt-output",
                "/receipt.json",
            ],
            [
                "publish-training-image",
                "--aws-config",
                "/aws.json",
                "--receipt-output",
                "/receipt.json",
            ],
            [
                "freeze-training-plan",
                "--scientific-config",
                "/scientific.json",
                "--aws-config",
                "/aws.json",
                "--source-root",
                "/source",
                "--source-bundle-output",
                "/source.tar.gz",
                "--manifest-output",
                "/launch.json",
                "--attempt-id",
                "a2",
                "--parent-manifest",
                "/a1.json",
            ],
            [
                "stage",
                "training-inputs",
                "--manifest",
                "/launch.json",
                "--source-bundle",
                "/source.tar.gz",
                "--dataset-dir",
                "/dataset",
                "--base-model-dir",
                "/base-model",
                "--snapshot-manifest",
                "/snapshot.json",
                "--receipt-output",
                "/staging.json",
            ],
            [
                "preflight",
                "runtime-smoke",
                "--aws-config",
                "/aws.json",
                "--image-uri",
                "example@sha256:" + "a" * 64,
                "--job-name",
                "arr-ret-runtime-smoke-a1",
                "--receipt-output",
                "/receipt.json",
            ],
            [
                "preflight",
                "training",
                "--manifest",
                "/launch.json",
                "--staging-receipt",
                "/staging.json",
                "--run-id",
                "determinism-smoke-a",
                "--receipt-output",
                "/preflight.json",
            ],
            [
                "submit",
                "runtime-smoke",
                "--preflight-receipt",
                "/preflight.json",
                "--receipt-output",
                "/submit.json",
            ],
            [
                "submit",
                "training",
                "--manifest",
                "/launch.json",
                "--staging-receipt",
                "/staging.json",
                "--preflight-receipt",
                "/preflight.json",
                "--receipt-output",
                "/submission.json",
            ],
            [
                "status",
                "runtime-smoke",
                "--job-name",
                "arr-ret-runtime-smoke-a1",
                "--region",
                "us-east-1",
                "--output",
                "/status.json",
            ],
            [
                "status",
                "training",
                "--manifest",
                "/launch.json",
                "--staging-receipt",
                "/staging.json",
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--output",
                "/status.json",
            ],
            [
                "acquire",
                "determinism-smoke",
                "--manifest",
                "/launch.json",
                "--staging-receipt",
                "/staging.json",
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--terminal-receipt",
                "/terminal.json",
                "--output-dir",
                "/artifact",
            ],
            [
                "evaluate",
                "runtime-smoke",
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--region",
                "us-east-1",
                "--output",
                "/evaluation.json",
            ],
            [
                "aggregate",
                *sum(
                    (["--evaluation-dir", f"/fold-{fold}"] for fold in range(5)),
                    [],
                ),
                "--dataset-dir",
                "/dataset",
                "--fold-manifest",
                "/folds.json",
                "--output",
                "/index.json",
            ],
            [
                "verify",
                "training-plan",
                "--manifest",
                "/launch.json",
                "--output",
                "/verified.json",
            ],
            [
                "verify",
                "training",
                "--manifest",
                "/launch.json",
                "--staging-receipt",
                "/staging.json",
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--output",
                "/terminal.json",
            ],
            [
                "verify",
                "determinism-smoke",
                "--manifest",
                "/launch.json",
                "--staging-receipt",
                "/staging.json",
                "--acquisition-receipt-a",
                "/artifact-a/acquisition_receipt.json",
                "--acquisition-receipt-b",
                "/artifact-b/acquisition_receipt.json",
                "--output",
                "/determinism.json",
            ],
        ]
        for arguments in fixtures:
            with self.subTest(command=arguments[:2]):
                parsed = cli.parse_args(arguments)
                self.assertEqual(parsed.command, arguments[0])

        first_attempt = cli.parse_args(
            [
                "freeze-training-plan",
                "--scientific-config",
                "/scientific.json",
                "--aws-config",
                "/aws.json",
                "--source-root",
                "/source",
                "--source-bundle-output",
                "/source.tar.gz",
                "--manifest-output",
                "/launch.json",
            ]
        )
        self.assertEqual(first_attempt.attempt_id, "a1")
        self.assertIsNone(first_attempt.parent_manifest)

    def test_fold_processing_modes_parse_explicit_contract(self) -> None:
        context = [
            "--completed-evidence",
            "/fold.json",
            "--archive-copy-receipt",
            "/archives.json",
            "--static-staging-receipt",
            "/static.json",
            "--overlay-publication-receipt",
            "/overlay.json",
        ]
        fixtures = [
            [
                "completed-evidence",
                "--supervisor-state-dir",
                "/supervisor",
                "--outer-fold",
                "0",
                "--output",
                "/fold.json",
            ],
            [
                "stage-static",
                "--completed-evidence",
                "/fold.json",
                "--e5-snapshot-dir",
                "/e5",
                "--e5-snapshot-manifest",
                "/controls/e5_snapshot.json",
                "--e5-pack-dir",
                "/e5-pack",
                "--fixed-base-dir",
                "/fixed",
                "--destination-prefix",
                "arr/static/f0/",
                "--state-dir",
                "/state/static",
                "--receipt-output",
                "/static.json",
            ],
            [
                "copy-archives",
                "--completed-evidence",
                "/fold.json",
                "--destination-prefix",
                "arr/archives/f0/",
                "--state-dir",
                "/state/archives",
                "--receipt-output",
                "/archives.json",
            ],
            [
                "phase1-preflight",
                *context,
                "--job-name",
                "arr-ret-cv1-f0-inventory-a3",
                "--output-prefix",
                "arr/inventory/f0/",
                "--receipt-output",
                "/preflight.json",
            ],
            [
                "phase1-submit",
                *context,
                "--preflight-receipt",
                "/preflight.json",
                "--state-dir",
                "/state/submission",
                "--receipt-output",
                "/submission.json",
            ],
            [
                "phase1-status",
                *context,
                "--preflight-receipt",
                "/preflight.json",
                "--output",
                "/status.json",
            ],
            [
                "phase1-verify",
                *context,
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--receipt-output",
                "/terminal.json",
            ],
            [
                "phase1-acquire",
                *context,
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--terminal-receipt",
                "/terminal.json",
                "--output-dir",
                "/acquisition",
            ],
            [
                "storage-proof",
                *context,
                "--preflight-receipt",
                "/preflight.json",
                "--submission-receipt",
                "/submission.json",
                "--terminal-receipt",
                "/terminal.json",
                "--acquisition-receipt",
                "/acquisition/acquisition_receipt.json",
                "--acquisition-dir",
                "/acquisition",
                "--phase2-generated-control-file-size",
                "4096",
                "--phase2-generated-control-file-size",
                "8192",
                "--phase2-output-reserve-bytes",
                "1073741824",
                "--safety-reserve-bytes",
                "2147483648",
                "--output",
                "/storage-proof.json",
            ],
        ]
        for arguments in fixtures:
            with self.subTest(mode=arguments[0]):
                parsed = cli.parse_args(["fold-processing", *arguments])
                self.assertEqual(parsed.command, "fold-processing")
                self.assertEqual(parsed.fold_mode, arguments[0])

        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "fold-processing",
                    "completed-evidence",
                    "--supervisor-state-dir",
                    "/supervisor",
                    "--outer-fold",
                    "5",
                    "--output",
                    "/fold.json",
                ]
            )

    def test_attempt_parent_is_explicit_and_immediately_previous(self) -> None:
        self.assertIsNone(
            cli._parent_manifest_sha256(
                attempt_id="a1",
                parent_manifest=None,
            )
        )
        with self.assertRaisesRegex(ValueError, "must not name"):
            cli._parent_manifest_sha256(
                attempt_id="a1",
                parent_manifest=Path("/a0.json"),
            )
        with self.assertRaisesRegex(ValueError, "requires its parent"):
            cli._parent_manifest_sha256(
                attempt_id="a2",
                parent_manifest=None,
            )

        digest = "a" * 64
        with patch.object(
            cli.manifest,
            "read_manifest",
            return_value=(
                {
                    "attempt": {
                        "attempt_id": "a1",
                        "parent_manifest_sha256": None,
                    }
                },
                digest,
            ),
        ) as read:
            self.assertEqual(
                cli._parent_manifest_sha256(
                    attempt_id="a2",
                    parent_manifest=Path("/a1.json"),
                ),
                digest,
            )
        read.assert_called_once_with(Path("/a1.json"))

        with patch.object(
            cli.manifest,
            "read_manifest",
            return_value=(
                {
                    "attempt": {
                        "attempt_id": "a1",
                        "parent_manifest_sha256": None,
                    }
                },
                digest,
            ),
        ), self.assertRaisesRegex(ValueError, "requires parent a2"):
            cli._parent_manifest_sha256(
                attempt_id="a3",
                parent_manifest=Path("/a1.json"),
            )

    def test_freeze_dispatch_binds_attempt_and_parent_plan(self) -> None:
        scientific = {
            "sources": {
                "commit_epoch": 1_700_000_000,
                "git_commit": "1" * 40,
                "git_tree": "2" * 40,
                "include_paths": ["train_sm.py"],
            }
        }
        aws_config = {"region": "us-east-1"}
        bundle = Mock(name="source_bundle")
        dry = {"manifest_type": "retrieval_cv_training_plan"}
        arguments = [
            "freeze-training-plan",
            "--scientific-config",
            "/scientific.json",
            "--aws-config",
            "/aws.json",
            "--source-root",
            "/source",
            "--source-bundle-output",
            "/source.tar.gz",
            "--manifest-output",
            "/launch.json",
            "--attempt-id",
            "a2",
            "--parent-manifest",
            "/a1.json",
        ]
        with (
            patch.object(
                cli.config,
                "load_scientific_config",
                return_value=(scientific, "1" * 64),
            ),
            patch.object(
                cli.config,
                "load_aws_local_config",
                return_value=(aws_config, "2" * 64),
            ),
            patch.object(
                cli,
                "_parent_manifest_sha256",
                return_value="3" * 64,
            ) as parent,
            patch.object(cli.manifest, "validate_scientific_source_claims"),
            patch.object(cli.manifest, "validate_clean_source_checkout"),
            patch.object(
                cli.manifest,
                "build_source_bundle",
                return_value=bundle,
            ),
            patch.object(
                cli.manifest,
                "build_dry_manifest",
                return_value=dry,
            ) as build,
            patch.object(cli.manifest, "publish_manifest_absent") as publish,
        ):
            self.assertEqual(cli.main(arguments), 0)
        parent.assert_called_once_with(
            attempt_id="a2",
            parent_manifest=Path("/a1.json"),
        )
        build.assert_called_once_with(
            scientific_config=scientific,
            aws_local_config=aws_config,
            source_bundle=bundle,
            attempt_id="a2",
            parent_manifest_sha256="3" * 64,
        )
        publish.assert_called_once_with(Path("/launch.json"), dry)

    def test_abbreviations_unknown_modes_and_relative_runtime_paths_fail(self) -> None:
        invalid = [
            ["build-d"],
            ["preflight", "runtime"],
            ["submit", "fold"],
            ["status", "runtime-smoke", "--job-name", "x", "--region", "us-west-2"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    cli.parse_args(arguments)
        with self.assertRaisesRegex(ValueError, "absolute"):
            cli._absolute(Path("relative.json"), name="fixture")

    def test_json_publication_is_canonical_absent_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            value = {"schema_version": 1, "status": "ready"}
            cli._publish_json(output, value)
            loaded, _ = config.load_canonical_json_object(output)
            self.assertEqual(loaded, value)
            self.assertFalse((output.parent / ".receipt.json.incomplete").exists())
            with self.assertRaises(FileExistsError):
                cli._publish_json(output, value)

    def test_receipt_loader_accepts_only_the_two_exact_canonical_encodings(
        self,
    ) -> None:
        value = {"nested": {"ready": True}, "schema_version": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretty = root / "pretty.json"
            compact = root / "compact.json"
            duplicate = root / "duplicate.json"
            noncanonical = root / "noncanonical.json"
            pretty.write_bytes(config.canonical_json_bytes(value))
            compact.write_bytes(cli.aws.canonical_json_bytes(value))
            duplicate.write_bytes(b'{"schema_version":1,"schema_version":2}\n')
            noncanonical.write_bytes(b'{"schema_version": 1}\n')

            self.assertEqual(cli._load_receipt(pretty), value)
            self.assertEqual(cli._load_receipt(compact), value)
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                cli._load_receipt(duplicate)
            with self.assertRaisesRegex(ValueError, "neither repository-canonical"):
                cli._load_receipt(noncanonical)

    def test_fold_clients_use_only_completed_evidence_plan_region(self) -> None:
        completed = {
            "training_plan": {
                "infrastructure": {"region": "us-east-1"},
            }
        }
        clients = Mock()
        with patch.object(cli.aws, "make_clients", return_value=clients) as make:
            self.assertIs(cli._clients_for_completed_fold(completed), clients)
        make.assert_called_once_with(region="us-east-1")

    def test_phase1_status_validates_context_and_uses_preflight_job_name(
        self,
    ) -> None:
        completed = {
            "training_plan": {
                "infrastructure": {"region": "us-east-1"},
            }
        }
        archive = {"protocol": "archives"}
        static = {"protocol": "static"}
        publication = {"protocol": "overlay"}
        preflight = {"job_name": "arr-ret-cv1-f0-inventory-a3"}
        clients = Mock()
        status = {"schema_version": 1, "status": "InProgress"}
        arguments = [
            "fold-processing",
            "phase1-status",
            "--completed-evidence",
            "/fold.json",
            "--archive-copy-receipt",
            "/archives.json",
            "--static-staging-receipt",
            "/static.json",
            "--overlay-publication-receipt",
            "/overlay.json",
            "--preflight-receipt",
            "/preflight.json",
            "--output",
            "/status.json",
        ]
        with (
            patch.object(cli, "_require_absent_output") as absent,
            patch.object(
                cli,
                "_load_phase1_context",
                return_value=(completed, archive, static, publication),
            ),
            patch.object(cli, "_load_receipt", return_value={"raw": "preflight"}),
            patch.object(
                cli.fold_processing_aws,
                "validate_fold_inventory_preflight_receipt",
                return_value=preflight,
            ) as validate,
            patch.object(
                cli,
                "_clients_for_completed_fold",
                return_value=clients,
            ) as make_clients,
            patch.object(
                cli.fold_processing_aws,
                "describe_fold_inventory",
                return_value=status,
            ) as describe,
            patch.object(cli, "_publish_json") as publish,
        ):
            self.assertEqual(cli.main(arguments), 0)
        absent.assert_called_once_with(Path("/status.json"))
        validate.assert_called_once_with(
            {"raw": "preflight"},
            completed_fold_evidence=completed,
            archive_copy_receipt=archive,
            static_staging_receipt=static,
            overlay_publication_receipt=publication,
        )
        make_clients.assert_called_once_with(completed)
        describe.assert_called_once_with(
            clients.sagemaker,
            job_name="arr-ret-cv1-f0-inventory-a3",
        )
        publish.assert_called_once_with(Path("/status.json"), status)

    def test_existing_fold_output_stops_before_evidence_or_aws_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "static.json"
            output.write_bytes(b"occupied")
            with (
                patch.object(cli, "_load_completed_fold_evidence") as load,
                patch.object(cli.aws, "make_clients") as make_clients,
            ):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    cli.main(
                        [
                            "fold-processing",
                            "stage-static",
                            "--completed-evidence",
                            "/fold.json",
                            "--e5-snapshot-dir",
                            "/e5",
                            "--e5-snapshot-manifest",
                            "/controls/e5_snapshot.json",
                            "--e5-pack-dir",
                            "/e5-pack",
                            "--fixed-base-dir",
                            "/fixed",
                            "--destination-prefix",
                            "arr/static/f0/",
                            "--state-dir",
                            str(root / "state"),
                            "--receipt-output",
                            str(output),
                        ]
                    )
            load.assert_not_called()
            make_clients.assert_not_called()

    def test_fold_state_and_receipt_paths_must_be_resolved_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            invalid = [
                ("equal", root / "same", root / "same"),
                ("first ancestor", root / "parent", root / "parent/child"),
                ("second ancestor", root / "parent/child", root / "parent"),
                ("resolved equal", real / "same", alias / "same"),
                (
                    "resolved first ancestor",
                    real / "parent",
                    alias / "parent/child",
                ),
                (
                    "resolved second ancestor",
                    real / "parent/child",
                    alias / "parent",
                ),
            ]
            for label, first, second in invalid:
                with self.subTest(case=label):
                    with self.assertRaisesRegex(ValueError, "must be disjoint"):
                        cli._require_disjoint_paths(
                            first,
                            second,
                            name="Fold outputs",
                        )

            cli._require_disjoint_paths(
                real / "left",
                alias / "right",
                name="Fold outputs",
            )

    def test_each_stateful_fold_dispatch_rejects_equal_outputs_before_load(
        self,
    ) -> None:
        context = [
            "--completed-evidence",
            "/fold.json",
            "--archive-copy-receipt",
            "/archives.json",
            "--static-staging-receipt",
            "/static.json",
            "--overlay-publication-receipt",
            "/overlay.json",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = [
                [
                    "stage-static",
                    "--completed-evidence",
                    "/fold.json",
                    "--e5-snapshot-dir",
                    "/e5",
                    "--e5-snapshot-manifest",
                    "/controls/e5_snapshot.json",
                    "--e5-pack-dir",
                    "/e5-pack",
                    "--fixed-base-dir",
                    "/fixed",
                    "--destination-prefix",
                    "arr/static/f0/",
                ],
                [
                    "copy-archives",
                    "--completed-evidence",
                    "/fold.json",
                    "--destination-prefix",
                    "arr/archives/f0/",
                ],
                [
                    "phase1-submit",
                    *context,
                    "--preflight-receipt",
                    "/preflight.json",
                ],
            ]
            for arguments in fixtures:
                output = root / f"{arguments[0]}.output"
                command = [
                    "fold-processing",
                    *arguments,
                    "--state-dir",
                    str(output),
                    "--receipt-output",
                    str(output),
                ]
                with self.subTest(mode=arguments[0]):
                    with (
                        patch.object(
                            cli,
                            "_load_completed_fold_evidence",
                        ) as load_completed,
                        patch.object(
                            cli,
                            "_load_phase1_context",
                        ) as load_context,
                        patch.object(cli.aws, "make_clients") as make_clients,
                    ):
                        with self.assertRaisesRegex(ValueError, "must be disjoint"):
                            cli.main(command)
                    load_completed.assert_not_called()
                    load_context.assert_not_called()
                    make_clients.assert_not_called()

    def test_training_commands_check_absent_output_before_remote_operation(self) -> None:
        plan = {"infrastructure": {"region": "us-east-1"}}
        clients = Mock()
        fixtures = [
            (
                [
                    "stage",
                    "training-inputs",
                    "--manifest",
                    "/plan.json",
                    "--source-bundle",
                    "/source.tar.gz",
                    "--dataset-dir",
                    "/dataset",
                    "--base-model-dir",
                    "/base-model",
                    "--snapshot-manifest",
                    "/snapshot.json",
                    "--receipt-output",
                    "/receipt.json",
                ],
                cli.training_aws,
                "stage_training_inputs_once",
            ),
            (
                [
                    "preflight",
                    "training",
                    "--manifest",
                    "/plan.json",
                    "--staging-receipt",
                    "/staging.json",
                    "--run-id",
                    "determinism-smoke-a",
                    "--receipt-output",
                    "/receipt.json",
                ],
                cli.training_launch,
                "preflight_training_job",
            ),
            (
                [
                    "submit",
                    "training",
                    "--manifest",
                    "/plan.json",
                    "--staging-receipt",
                    "/staging.json",
                    "--preflight-receipt",
                    "/preflight.json",
                    "--receipt-output",
                    "/receipt.json",
                ],
                cli.training_launch,
                "submit_training_job_once",
            ),
            (
                [
                    "status",
                    "training",
                    "--manifest",
                    "/plan.json",
                    "--staging-receipt",
                    "/staging.json",
                    "--preflight-receipt",
                    "/preflight.json",
                    "--submission-receipt",
                    "/submission.json",
                    "--output",
                    "/receipt.json",
                ],
                cli.training_launch,
                "describe_training_job_status",
            ),
            (
                [
                    "verify",
                    "training",
                    "--manifest",
                    "/plan.json",
                    "--staging-receipt",
                    "/staging.json",
                    "--preflight-receipt",
                    "/preflight.json",
                    "--submission-receipt",
                    "/submission.json",
                    "--output",
                    "/receipt.json",
                ],
                cli.training_launch,
                "verify_terminal_training_job",
            ),
        ]
        for arguments, module, function_name in fixtures:
            with self.subTest(command=arguments[:2]):
                order: list[str] = []

                def absent(_path: Path) -> None:
                    order.append("absent")

                def remote(*_args: object, **_kwargs: object) -> dict[str, object]:
                    order.append("remote")
                    return {"schema_version": 1, "succeeded": True}

                with (
                    patch.object(cli, "_require_absent_output", side_effect=absent),
                    patch.object(cli, "_load_training_plan", return_value=plan),
                    patch.object(
                        cli, "_load_receipt", return_value={"schema_version": 1}
                    ),
                    patch.object(cli.aws, "make_clients", return_value=clients),
                    patch.object(module, function_name, side_effect=remote) as operation,
                    patch.object(cli, "_publish_json") as publish,
                ):
                    self.assertEqual(cli.main(arguments), 0)
                self.assertEqual(order, ["absent", "remote"])
                operation.assert_called_once()
                publish.assert_called_once()

    def test_existing_training_submit_output_prevents_client_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            output.write_bytes(b"occupied")
            with patch.object(cli.aws, "make_clients") as make_clients:
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    cli.main(
                        [
                            "submit",
                            "training",
                            "--manifest",
                            "/plan.json",
                            "--staging-receipt",
                            "/staging.json",
                            "--preflight-receipt",
                            "/preflight.json",
                            "--receipt-output",
                            str(output),
                        ]
                    )
            make_clients.assert_not_called()

    def test_acquisition_checks_absent_output_before_remote_operation(self) -> None:
        plan = {"infrastructure": {"region": "us-east-1"}}
        arguments = [
            "acquire",
            "determinism-smoke",
            "--manifest",
            "/plan.json",
            "--staging-receipt",
            "/staging.json",
            "--preflight-receipt",
            "/preflight.json",
            "--submission-receipt",
            "/submission.json",
            "--terminal-receipt",
            "/terminal.json",
            "--output-dir",
            "/artifact",
        ]
        order: list[str] = []

        def absent(_path: Path) -> None:
            order.append("absent")

        def acquire(*_args: object, **_kwargs: object) -> dict[str, object]:
            order.append("remote")
            return {"schema_version": 1}

        with (
            patch.object(cli, "_require_absent_output", side_effect=absent),
            patch.object(cli, "_load_training_plan", return_value=plan),
            patch.object(cli, "_load_receipt", return_value={"schema_version": 1}),
            patch.object(cli.aws, "make_clients", return_value=Mock()),
            patch.object(
                cli.training_artifacts,
                "acquire_completed_determinism_smoke_artifact",
                side_effect=acquire,
            ) as operation,
            patch.object(cli, "_publish_json") as publish,
        ):
            self.assertEqual(cli.main(arguments), 0)
        self.assertEqual(order, ["absent", "remote"])
        operation.assert_called_once()
        publish.assert_not_called()

    def test_determinism_verify_is_local_and_publishes_once(self) -> None:
        plan = {"manifest_type": "retrieval_cv_training_plan"}
        staging = {"protocol": "retrieval_cv_training_input_staging_v2"}
        receipt = {"schema_version": 3, "exact_match": True}
        arguments = [
            "verify",
            "determinism-smoke",
            "--manifest",
            "/plan.json",
            "--staging-receipt",
            "/staging.json",
            "--acquisition-receipt-a",
            "/artifact-a/acquisition_receipt.json",
            "--acquisition-receipt-b",
            "/artifact-b/acquisition_receipt.json",
            "--output",
            "/determinism.json",
        ]
        with (
            patch.object(cli, "_require_absent_output") as absent,
            patch.object(cli, "_load_training_plan", return_value=plan),
            patch.object(cli, "_load_receipt", return_value=staging),
            patch.object(cli.aws, "make_clients") as make_clients,
            patch.object(
                cli.determinism_gate,
                "run_determinism_gate",
                return_value=receipt,
            ) as gate,
            patch.object(cli, "_publish_json") as publish,
        ):
            self.assertEqual(cli.main(arguments), 0)
        absent.assert_called_once_with(Path("/determinism.json"))
        make_clients.assert_not_called()
        gate.assert_called_once_with(
            training_plan=plan,
            staging_receipt=staging,
            acquisition_receipt_paths_by_run={
                "determinism-smoke-a": Path(
                    "/artifact-a/acquisition_receipt.json"
                ),
                "determinism-smoke-b": Path(
                    "/artifact-b/acquisition_receipt.json"
                ),
            },
        )
        publish.assert_called_once_with(Path("/determinism.json"), receipt)

    def test_training_verify_publishes_failure_evidence_then_fails_loudly(
        self,
    ) -> None:
        plan = {"infrastructure": {"region": "us-east-1"}}
        terminal = {
            "schema_version": 1,
            "succeeded": False,
            "terminal_status": "Failed",
        }
        arguments = [
            "verify",
            "training",
            "--manifest",
            "/plan.json",
            "--staging-receipt",
            "/staging.json",
            "--preflight-receipt",
            "/preflight.json",
            "--submission-receipt",
            "/submission.json",
            "--output",
            "/terminal.json",
        ]
        with (
            patch.object(cli, "_require_absent_output"),
            patch.object(cli, "_load_training_plan", return_value=plan),
            patch.object(cli, "_load_receipt", return_value={"schema_version": 1}),
            patch.object(cli.aws, "make_clients", return_value=Mock()),
            patch.object(
                cli.training_launch,
                "verify_terminal_training_job",
                return_value=terminal,
            ),
            patch.object(cli, "_publish_json") as publish,
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed"):
                cli.main(arguments)
        publish.assert_called_once_with(Path("/terminal.json"), terminal)


if __name__ == "__main__":
    unittest.main()
