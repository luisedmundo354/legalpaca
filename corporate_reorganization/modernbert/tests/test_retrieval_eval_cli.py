from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import eval_retriever  # noqa: E402
from processing_eval import evaluate_sm  # noqa: E402


class EvaluationCliTest(unittest.TestCase):
    def test_exact_required_arguments_parse(self) -> None:
        args = eval_retriever.parse_args(
            [
                "--evaluation-plan",
                "plan.json",
                "--local-bindings",
                "bindings.json",
                "--output-dir",
                "result",
                "--device",
                "cuda",
            ]
        )
        self.assertEqual(args.evaluation_plan, Path("plan.json"))
        self.assertEqual(args.local_bindings, Path("bindings.json"))
        self.assertEqual(args.output_dir, Path("result"))
        self.assertEqual(args.device, "cuda")

    def test_missing_abbreviated_unknown_and_retired_options_are_rejected(self) -> None:
        valid = [
            "--evaluation-plan",
            "plan.json",
            "--local-bindings",
            "bindings.json",
            "--output-dir",
            "result",
            "--device",
            "cuda",
        ]
        invalid_argvs = (
            valid[:-2],
            ["--evaluation", "plan.json", *valid[2:]],
            [*valid, "--write_rankings_top_n", "20"],
            [*valid, "--model_s3_uri", "s3://bucket/model.tar.gz"],
            [*valid, "--systems", "bm25_flat"],
        )
        for argv in invalid_argvs:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    eval_retriever.parse_args(argv)

    def test_main_delegates_once_and_prints_canonical_publication_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            bindings = root / "bindings.json"
            output = root / "output"
            stdout = io.StringIO()
            with mock.patch.object(
                eval_retriever,
                "run_local_controlled_evaluation_plan",
                return_value=MappingProxyType(
                    {
                        "output_name": "output",
                        "artifact_manifest_sha256": "a" * 64,
                        "files": (
                            MappingProxyType(
                                {
                                    "path": "results.json",
                                    "size": 12,
                                    "sha256": "b" * 64,
                                }
                            ),
                        ),
                    }
                ),
            ) as runner, contextlib.redirect_stdout(stdout):
                status = eval_retriever.main(
                    [
                        "--evaluation-plan",
                        str(plan),
                        "--local-bindings",
                        str(bindings),
                        "--output-dir",
                        str(output),
                        "--device",
                        "cuda:0",
                    ]
                )
            self.assertEqual(status, 0)
            runner.assert_called_once_with(
                evaluation_plan_path=plan.resolve(),
                local_bindings_path=bindings.resolve(),
                output_dir=output.resolve(),
                device="cuda:0",
            )
            self.assertEqual(
                json.loads(stdout.getvalue()),
                {
                    "artifact_manifest_sha256": "a" * 64,
                    "files": [
                        {
                            "path": "results.json",
                            "sha256": "b" * 64,
                            "size": 12,
                        }
                    ],
                    "output_name": "output",
                },
            )

    def test_processing_entrypoint_has_one_strict_complete_evaluation_cli(self) -> None:
        args = evaluate_sm.parse_args(
            [
                "--evaluation-plan",
                "plan.json",
                "--local-bindings",
                "bindings.json",
                "--output-dir",
                "result",
                "--device",
                "cuda:0",
            ]
        )
        self.assertEqual(args.device, "cuda:0")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_sm.parse_args(
                [
                    "--evaluation-plan",
                    "plan.json",
                    "--local-bindings",
                    "bindings.json",
                    "--output-dir",
                    "result",
                    "--device",
                    "cpu",
                ]
            )


if __name__ == "__main__":
    unittest.main()
