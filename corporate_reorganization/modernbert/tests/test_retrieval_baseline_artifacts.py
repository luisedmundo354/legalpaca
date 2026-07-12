from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.artifacts import import_pinned_artifact_runtime  # noqa: E402
from retriever.baseline_artifacts import (  # noqa: E402
    FixedBaseArtifactExpectation,
    _load_fixed_base_build_contract,
    _publish_fixed_base_staging,
    _validate_fixed_base_output_paths,
    load_e5_encoder,
    validate_fixed_base_artifact,
)


CONFIG_DIR = MODERNBERT_DIR / "experiments/retrieval_cv/configs"
BASELINE_CONFIG_SHA256 = "714b8c18e9e32130ebf3358a72d9c6aceceeb1e14ee0d76270306d901b81f33a"
FIXED_MANIFEST_SHA256 = "ccff3fa4c141290ef9383992a4d3de2b8cfa5e50d02c4cd06e3fe52e92d0202b"


class BaselineArtifactContractTest(unittest.TestCase):
    def test_fixed_base_publication_is_no_replace_and_retracts_failures(self) -> None:
        payload = b"{}\n"
        payload_sha256 = hashlib.sha256(payload).hexdigest()

        def staged(root: Path) -> tuple[Path, Path]:
            staging = root / "artifact.incomplete"
            output = root / "artifact"
            staging.mkdir()
            (staging / "payload.bin").write_bytes(b"fixed-base-fixture\n")
            return staging, output

        expected = {
            "model_sha256": "1" * 64,
            "state_key_sha256": "2" * 64,
            "new_embedding_rows_sha256": "3" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            staging, output = staged(Path(temporary))
            output.mkdir()
            (output / "sentinel").write_text("preserve me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _publish_fixed_base_staging(
                    staging_dir=staging,
                    output_dir=output,
                    manifest_payload=payload,
                    expected_manifest_sha256=payload_sha256,
                    expected_baseline_config_sha256=BASELINE_CONFIG_SHA256,
                    expected_model_sha256=expected["model_sha256"],
                    expected_state_key_sha256=expected["state_key_sha256"],
                    expected_new_rows_sha256=expected["new_embedding_rows_sha256"],
                    precommit_validator=lambda: None,
                )
            self.assertTrue(staging.is_dir())
            self.assertTrue(output.is_dir())
            self.assertEqual(
                (output / "sentinel").read_text(encoding="utf-8"),
                "preserve me",
            )
            self.assertFalse((staging / "artifact_manifest.json").exists())

        for failure_target in ("precommit", "marker", "readback"):
            with self.subTest(failure_target=failure_target), tempfile.TemporaryDirectory() as temporary:
                staging, output = staged(Path(temporary))
                precommit_validator = lambda: None
                if failure_target == "precommit":
                    failure_patch = contextlib.nullcontext()
                    precommit_validator = mock.Mock(
                        side_effect=RuntimeError("precommit validation failed")
                    )
                elif failure_target == "marker":
                    failure_patch = mock.patch(
                        "retriever.baseline_artifacts._new_file",
                        side_effect=OSError("marker write failed"),
                    )
                else:
                    failure_patch = mock.patch(
                        "retriever.baseline_artifacts.validate_fixed_base_artifact",
                        side_effect=ValueError("readback failed"),
                    )
                with failure_patch, self.assertRaisesRegex(
                    (OSError, RuntimeError, ValueError), "failed"
                ):
                    _publish_fixed_base_staging(
                        staging_dir=staging,
                        output_dir=output,
                        manifest_payload=payload,
                        expected_manifest_sha256=payload_sha256,
                        expected_baseline_config_sha256=BASELINE_CONFIG_SHA256,
                        expected_model_sha256=expected["model_sha256"],
                        expected_state_key_sha256=expected["state_key_sha256"],
                        expected_new_rows_sha256=expected["new_embedding_rows_sha256"],
                        precommit_validator=precommit_validator,
                    )
                self.assertTrue(staging.is_dir())
                self.assertFalse(output.exists())
                self.assertFalse((staging / "artifact_manifest.json").exists())

        with tempfile.TemporaryDirectory() as temporary:
            staging, output = staged(Path(temporary))
            validated = SimpleNamespace(**expected)
            with (
                mock.patch(
                    "retriever.baseline_artifacts.validate_fixed_base_artifact",
                    return_value=validated,
                ) as validator,
            ):
                _publish_fixed_base_staging(
                    staging_dir=staging,
                    output_dir=output,
                    manifest_payload=payload,
                    expected_manifest_sha256=payload_sha256,
                    expected_baseline_config_sha256=BASELINE_CONFIG_SHA256,
                    expected_model_sha256=expected["model_sha256"],
                    expected_state_key_sha256=expected["state_key_sha256"],
                    expected_new_rows_sha256=expected["new_embedding_rows_sha256"],
                    precommit_validator=lambda: None,
                )
            validator.assert_called_once()
            self.assertFalse(staging.exists())
            self.assertEqual((output / "artifact_manifest.json").read_bytes(), payload)

    def test_fixed_base_build_contract_and_output_boundaries_are_exact(self) -> None:
        contract_path = CONFIG_DIR / "fixed_base_artifact.json"
        contract = _load_fixed_base_build_contract(contract_path)
        self.assertEqual(contract["artifact_manifest_sha256"], FIXED_MANIFEST_SHA256)
        self.assertEqual(contract["baseline_config_sha256"], BASELINE_CONFIG_SHA256)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            protected = root / "protected"
            protected.mkdir()
            output = protected / "nested-output"
            with self.assertRaisesRegex(ValueError, "overlaps"):
                _validate_fixed_base_output_paths(
                    output_dir=output,
                    protected_inputs=(protected,),
                )

        with tempfile.TemporaryDirectory() as temporary:
            mutated_path = Path(temporary) / "contract.json"
            mutated = dict(contract)
            mutated["fixed_initialization_seed"] = 18
            mutated_path.write_bytes(
                (
                    json.dumps(
                        mutated,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            with self.assertRaisesRegex(ValueError, "fixed_initialization_seed"):
                _load_fixed_base_build_contract(mutated_path)

    def test_committed_snapshot_and_fixed_artifact_identities_are_exact(self) -> None:
        baseline_path = CONFIG_DIR / "evaluation_baselines.json"
        self.assertEqual(hashlib.sha256(baseline_path.read_bytes()).hexdigest(), BASELINE_CONFIG_SHA256)
        baseline = json.loads(baseline_path.read_bytes())
        fixed = json.loads((CONFIG_DIR / "fixed_base_artifact.json").read_bytes())
        e5 = json.loads((CONFIG_DIR / "e5_snapshot.json").read_bytes())
        self.assertEqual(fixed["artifact_manifest_sha256"], FIXED_MANIFEST_SHA256)
        self.assertEqual(fixed["baseline_config_sha256"], BASELINE_CONFIG_SHA256)
        self.assertEqual(fixed["fixed_initialization_seed"], 17)
        self.assertEqual(
            fixed["model_sha256"],
            "a2822fd04d0ba9b5df5289d9384e89740d113ddd68810a8d05ba6dbefbc33300",
        )
        self.assertEqual(
            fixed["new_embedding_rows_sha256"],
            "6dba50931329f2bea4618616ba222440488b776dd1216a2a61279f83f9e9a26b",
        )
        self.assertEqual(e5["tree_sha256"], baseline["e5_base_v2"]["snapshot_tree_sha256"])
        self.assertEqual(e5["revision"], baseline["e5_base_v2"]["revision"])

    @unittest.skipUnless(
        os.environ.get("ARR_FIXED_BASE_ARTIFACT_DIR"),
        "ARR_FIXED_BASE_ARTIFACT_DIR is absent",
    )
    def test_real_fixed_base_artifact_validates_strictly(self) -> None:
        artifact = validate_fixed_base_artifact(
            Path(os.environ["ARR_FIXED_BASE_ARTIFACT_DIR"]),
            expectation=FixedBaseArtifactExpectation(
                artifact_manifest_sha256=FIXED_MANIFEST_SHA256,
                baseline_config_sha256=BASELINE_CONFIG_SHA256,
            ),
        )
        self.assertEqual(
            artifact.model_sha256,
            "a2822fd04d0ba9b5df5289d9384e89740d113ddd68810a8d05ba6dbefbc33300",
        )

    @unittest.skipUnless(
        os.environ.get("ARR_FIXED_BASE_ARTIFACT_DIR")
        and os.environ.get("ARR_FIXED_BASE_ARTIFACT_DIR_SECOND"),
        "two fixed-base artifact directories are required",
    )
    def test_two_fresh_fixed_base_builds_are_byte_identical(self) -> None:
        first = Path(os.environ["ARR_FIXED_BASE_ARTIFACT_DIR"]).resolve()
        second = Path(os.environ["ARR_FIXED_BASE_ARTIFACT_DIR_SECOND"]).resolve()
        self.assertNotEqual(first, second, "fresh builds must use distinct roots")
        first_files = {
            path.relative_to(first).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first.rglob("*")
            if path.is_file()
        }
        second_files = {
            path.relative_to(second).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in second.rglob("*")
            if path.is_file()
        }
        self.assertEqual(first_files, second_files)

    @unittest.skipUnless(
        os.environ.get("ARR_E5_SNAPSHOT_DIR"),
        "ARR_E5_SNAPSHOT_DIR is absent",
    )
    def test_real_e5_snapshot_loads_strictly_with_position_id_accounted_for(self) -> None:
        loaded = load_e5_encoder(
            snapshot_dir=Path(os.environ["ARR_E5_SNAPSHOT_DIR"]),
            manifest_path=CONFIG_DIR / "e5_snapshot.json",
            device="cpu",
            runtime=import_pinned_artifact_runtime(),
        )
        self.assertEqual(len(loaded.model.state_dict()), 200)
        self.assertEqual(loaded.model.config._attn_implementation, "eager")
        self.assertEqual(len(loaded.tokenizer), 30_522)
        with (
            mock.patch(
                "retriever.baseline_artifacts.validate_snapshot",
                side_effect=[
                    loaded.snapshot_identity,
                    ValueError("snapshot changed during load"),
                ],
            ) as validator,
            self.assertRaisesRegex(ValueError, "changed during load"),
        ):
            load_e5_encoder(
                snapshot_dir=Path(os.environ["ARR_E5_SNAPSHOT_DIR"]),
                manifest_path=CONFIG_DIR / "e5_snapshot.json",
                device="cpu",
                runtime=import_pinned_artifact_runtime(),
            )
        self.assertEqual(validator.call_count, 2)


if __name__ == "__main__":
    unittest.main()
