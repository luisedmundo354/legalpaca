from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    aws,
    determinism_gate,
    training_aws,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_training_aws import (
    _staging_receipt,
    _training_plan,
)
from retriever.determinism import SMOKE_COMPARISON_PROTOCOL, SMOKE_RUN_KIND
from retriever.determinism_artifacts import (
    DeterminismSmokeArtifactIdentity,
    ValidatedDeterminismSmokeArtifact,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _comparison(scientific_sha256: str) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "protocol": SMOKE_COMPARISON_PROTOCOL,
        "run_kind": SMOKE_RUN_KIND,
        "scientific_identity_sha256": scientific_sha256,
        "replicas": 2,
        "exact_match": True,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


class DeterminismGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan, _ = _training_plan(self.root)
        self.staging = _staging_receipt(self.plan)
        smokes = self.plan["auxiliary_runs"][2:]
        self.requests = {
            run["run_id"]: training_aws.build_determinism_smoke_training_request_receipt(
                training_plan=self.plan,
                run_id=run["run_id"],
                staging_receipt=self.staging,
            )
            for run in smokes
        }
        self.roots = {
            run_id: self.root / f"artifact-{run_id[-1]}"
            for run_id in determinism_gate.SMOKE_RUN_IDS
        }
        for root in self.roots.values():
            root.mkdir()
        self.manifest_hashes = {
            run_id: _digest(f"manifest:{run_id}")
            for run_id in determinism_gate.SMOKE_RUN_IDS
        }
        self.scientific_sha256 = _digest("identical-scientific-evidence")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validated_artifact(self, root: Path, expectation):
        identity_values = {
            field: _digest(f"shared:{field}")
            for field in DeterminismSmokeArtifactIdentity.__dataclass_fields__
        }
        identity_values["artifact_manifest_sha256"] = (
            expectation.artifact_manifest_sha256
        )
        identity_values["scientific_evidence_sha256"] = self.scientific_sha256
        return ValidatedDeterminismSmokeArtifact(
            root=root.resolve(strict=True),
            expectation=expectation,
            identity=DeterminismSmokeArtifactIdentity(**identity_values),
            files=(),
            scientific_evidence={"fixture": "same"},
            run_path=root / "determinism_smoke_run.json",
            model_path=root / "model.safetensors",
        )

    def _run(self):
        artifact_validator = Mock(
            side_effect=lambda root, *, expectation: self._validated_artifact(
                root, expectation
            )
        )
        evidence_comparator = Mock(
            return_value=_comparison(self.scientific_sha256)
        )
        with patch.object(
            determinism_gate,
            "validate_determinism_smoke_artifact",
            artifact_validator,
        ), patch.object(
            determinism_gate,
            "compare_smoke_scientific_evidence",
            evidence_comparator,
        ):
            receipt = determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                request_receipts_by_run=self.requests,
                artifact_roots_by_run=self.roots,
                artifact_manifest_sha256_by_run=self.manifest_hashes,
            )
        return receipt, artifact_validator, evidence_comparator

    def test_gate_derives_external_expectations_and_separates_coordinates(self) -> None:
        receipt, artifact_validator, evidence_comparator = self._run()
        self.assertEqual(receipt["protocol"], determinism_gate.DETERMINISM_GATE_PROTOCOL)
        self.assertIs(receipt["exact_match"], True)
        self.assertEqual(
            [row["run_id"] for row in receipt["launch"]["request_receipts"]],
            list(determinism_gate.SMOKE_RUN_IDS),
        )
        self.assertEqual(
            [row["run_id"] for row in receipt["artifacts"]],
            list(determinism_gate.SMOKE_RUN_IDS),
        )
        self.assertNotIn("artifact_root", receipt["launch"])
        self.assertNotIn("training_job_name", receipt["artifacts"][0])
        self.assertEqual(
            determinism_gate.validate_determinism_gate_receipt(receipt), receipt
        )
        self.assertEqual(artifact_validator.call_count, 2)
        plan_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(self.plan))
        staging_sha256 = aws.sha256_bytes(aws.canonical_json_bytes(self.staging))
        for call, run_id in zip(
            artifact_validator.call_args_list, determinism_gate.SMOKE_RUN_IDS
        ):
            expectation = call.kwargs["expectation"]
            self.assertEqual(call.args[0], self.roots[run_id])
            self.assertEqual(
                expectation.artifact_manifest_sha256,
                self.manifest_hashes[run_id],
            )
            self.assertEqual(expectation.training_plan_sha256, plan_sha256)
            self.assertEqual(
                expectation.training_staging_receipt_sha256, staging_sha256
            )
            self.assertEqual(
                expectation.source_bundle_sha256,
                self.plan["sources"]["source_bundle_sha256"],
            )
            self.assertEqual(
                expectation.source_bundle_inventory_sha256,
                self.plan["sources"]["source_inventory_sha256"],
            )
        evidence_comparator.assert_called_once_with(
            {"fixture": "same"}, {"fixture": "same"}
        )

    def test_exact_keying_paths_and_external_hashes_are_required(self) -> None:
        missing = copy.deepcopy(self.requests)
        missing.pop("determinism-smoke-b")
        extra = copy.deepcopy(self.requests)
        extra["determinism-smoke-c"] = extra["determinism-smoke-a"]
        swapped = {
            "determinism-smoke-a": self.requests["determinism-smoke-b"],
            "determinism-smoke-b": self.requests["determinism-smoke-a"],
        }
        for invalid in (missing, extra, swapped):
            with self.subTest(requests=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    determinism_gate.run_determinism_gate(
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                        request_receipts_by_run=invalid,
                        artifact_roots_by_run=self.roots,
                        artifact_manifest_sha256_by_run=self.manifest_hashes,
                    )

        string_root = dict(self.roots)
        string_root["determinism-smoke-a"] = str(
            string_root["determinism-smoke-a"]
        )
        with self.assertRaisesRegex(TypeError, "pathlib.Path"):
            determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                request_receipts_by_run=self.requests,
                artifact_roots_by_run=string_root,
                artifact_manifest_sha256_by_run=self.manifest_hashes,
            )

        boolean_hash = dict(self.manifest_hashes)
        boolean_hash["determinism-smoke-a"] = True
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                request_receipts_by_run=self.requests,
                artifact_roots_by_run=self.roots,
                artifact_manifest_sha256_by_run=boolean_hash,
            )

        duplicate_roots = dict(self.roots)
        duplicate_roots["determinism-smoke-b"] = duplicate_roots[
            "determinism-smoke-a"
        ]
        duplicate_hashes = dict(self.manifest_hashes)
        duplicate_hashes["determinism-smoke-b"] = duplicate_hashes[
            "determinism-smoke-a"
        ]
        with self.assertRaisesRegex(ValueError, "distinct resolved artifact roots"):
            determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                request_receipts_by_run=self.requests,
                artifact_roots_by_run=duplicate_roots,
                artifact_manifest_sha256_by_run=duplicate_hashes,
            )

    def test_scientific_mismatch_stops_without_a_receipt(self) -> None:
        artifact_validator = Mock(
            side_effect=lambda root, *, expectation: self._validated_artifact(
                root, expectation
            )
        )
        with patch.object(
            determinism_gate,
            "validate_determinism_smoke_artifact",
            artifact_validator,
        ), patch.object(
            determinism_gate,
            "compare_smoke_scientific_evidence",
            side_effect=RuntimeError("scientific mismatch"),
        ):
            with self.assertRaisesRegex(RuntimeError, "scientific mismatch"):
                determinism_gate.run_determinism_gate(
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    request_receipts_by_run=self.requests,
                    artifact_roots_by_run=self.roots,
                    artifact_manifest_sha256_by_run=self.manifest_hashes,
                )

    def test_receipt_schema_type_order_and_self_hash_are_strict(self) -> None:
        receipt, _, _ = self._run()
        mutations = []
        extra = copy.deepcopy(receipt)
        extra["retry"] = False
        mutations.append(extra)
        boolean_schema = copy.deepcopy(receipt)
        boolean_schema["schema_version"] = True
        mutations.append(boolean_schema)
        swapped = copy.deepcopy(receipt)
        swapped["artifacts"].reverse()
        swapped["sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(
                {key: value for key, value in swapped.items() if key != "sha256"}
            )
        )
        mutations.append(swapped)
        malformed_path = copy.deepcopy(receipt)
        malformed_path["artifacts"][0]["artifact_root"] = "relative/artifact"
        malformed_path["sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(
                {
                    key: value
                    for key, value in malformed_path.items()
                    if key != "sha256"
                }
            )
        )
        mutations.append(malformed_path)
        changed_hash = copy.deepcopy(receipt)
        changed_hash["plan_sha256"] = "f" * 64
        mutations.append(changed_hash)
        duplicate_artifact = copy.deepcopy(receipt)
        duplicate_artifact["artifacts"][1].update(
            {
                key: copy.deepcopy(duplicate_artifact["artifacts"][0][key])
                for key in (
                    "artifact_root",
                    "artifact_manifest_sha256",
                    "identity",
                )
            }
        )
        duplicate_artifact["sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(
                {
                    key: value
                    for key, value in duplicate_artifact.items()
                    if key != "sha256"
                }
            )
        )
        mutations.append(duplicate_artifact)
        alias_root = self.root / "artifact-alias"
        alias_root.symlink_to(
            self.roots["determinism-smoke-a"],
            target_is_directory=True,
        )
        aliased_artifact = copy.deepcopy(receipt)
        aliased_artifact["artifacts"][1].update(
            {
                "artifact_root": alias_root.as_posix(),
                "artifact_manifest_sha256": aliased_artifact["artifacts"][0][
                    "artifact_manifest_sha256"
                ],
                "identity": copy.deepcopy(
                    aliased_artifact["artifacts"][0]["identity"]
                ),
            }
        )
        aliased_artifact["sha256"] = aws.sha256_bytes(
            aws.canonical_json_bytes(
                {
                    key: value
                    for key, value in aliased_artifact.items()
                    if key != "sha256"
                }
            )
        )
        mutations.append(aliased_artifact)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises((TypeError, ValueError)):
                    determinism_gate.validate_determinism_gate_receipt(mutation)


if __name__ == "__main__":
    unittest.main()
