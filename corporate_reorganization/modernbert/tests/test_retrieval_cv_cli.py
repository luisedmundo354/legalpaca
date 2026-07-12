from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
                "submit",
                "runtime-smoke",
                "--preflight-receipt",
                "/preflight.json",
                "--receipt-output",
                "/submit.json",
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


if __name__ == "__main__":
    unittest.main()
