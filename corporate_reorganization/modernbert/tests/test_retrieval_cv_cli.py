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
        ]
        for arguments in fixtures:
            with self.subTest(command=arguments[:2]):
                parsed = cli.parse_args(arguments)
                self.assertEqual(parsed.command, arguments[0])

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
