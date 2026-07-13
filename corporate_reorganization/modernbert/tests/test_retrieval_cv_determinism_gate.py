from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from corporate_reorganization.modernbert.retriever.artifacts import ArtifactFileRecord
from corporate_reorganization.modernbert.retriever.determinism import (
    SMOKE_COMPARISON_PROTOCOL,
    SMOKE_RUN_KIND,
)
from corporate_reorganization.modernbert.retriever.determinism_artifacts import (
    DeterminismSmokeArtifactIdentity,
    ValidatedDeterminismSmokeArtifact,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


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


def _reseal_gate(receipt: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(receipt)
    payload = {key: nested for key, nested in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = _document_sha256(payload)
    return value


class DeterminismGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)
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
        self.scientific_sha256 = _digest("identical-scientific-evidence")
        self.paths: dict[str, Path] = {}
        self.acquisitions: dict[str, SimpleNamespace] = {}
        for index, run_id in enumerate(determinism_gate.SMOKE_RUN_IDS):
            bundle_root = self.root / f"bundle-{index}"
            artifact_root = bundle_root / "artifact"
            artifact_root.mkdir(parents=True)
            receipt_path = bundle_root / "acquisition_receipt.json"
            receipt_path.write_bytes(b"{}")
            self.paths[run_id] = receipt_path
            request_receipt = self.requests[run_id]
            request = request_receipt["request"]
            job_name = request["TrainingJobName"]
            account_id = self.plan["infrastructure"]["account_id"]
            region = self.plan["infrastructure"]["region"]
            job_arn = (
                f"arn:aws:sagemaker:{region}:{account_id}:training-job/{job_name}"
            )
            bucket = self.plan["infrastructure"]["artifact_bucket"]
            key = f"outputs/{job_name}/output/model.tar.gz"
            model_uri = f"s3://{bucket}/{key}"
            archive_size = 1_000 + index
            archive_sha256 = _digest(f"archive:{run_id}")
            remote_object = {
                "bucket": bucket,
                "key": key,
                "s3_uri": model_uri,
                "version_id": f"version-{index}",
                "size": archive_size,
                "sha256": archive_sha256,
                "etag": f'"etag-{index}-2"',
                "last_modified": f"2026-07-13T12:4{index}:00.000000Z",
                "storage_class": "STANDARD",
                "owner_id": "owner-id",
                "multipart_part_count": 2,
                "checksum": {
                    "algorithm": "CRC32",
                    "type": "COMPOSITE",
                    "value": f"checksum-{index}-2",
                },
                "encryption": {
                    "algorithm": "aws:kms",
                    "kms_key_id": (
                        f"arn:aws:kms:{region}:{account_id}:key/example-key"
                    ),
                    "bucket_key_enabled": True,
                },
                "content_type": "application/x-tar",
                "metadata": {"experiment": "retrieval-cv"},
            }
            preflight = {
                "run_id": run_id,
                "request_receipt": copy.deepcopy(request_receipt),
                "receipt_sha256": _digest(f"preflight-self:{run_id}"),
            }
            submission = {
                "run_id": run_id,
                "job_name": job_name,
                "job_arn": job_arn,
                "receipt_sha256": _digest(f"submission-self:{run_id}"),
            }
            status = {
                "run_id": run_id,
                "job_name": job_name,
                "job_arn": job_arn,
                "receipt_sha256": _digest(f"status-self:{run_id}"),
            }
            terminal = {
                "run_id": run_id,
                "job_name": job_name,
                "job_arn": job_arn,
                "model_artifact_s3_uri": model_uri,
                "status_receipt": status,
                "terminal_status": "Completed",
                "succeeded": True,
                "receipt_sha256": _digest(f"terminal-self:{run_id}"),
            }
            acquisition_receipt = {
                "schema_version": 1,
                "run_id": run_id,
                "receipt_sha256": _digest(f"acquisition-self:{run_id}"),
            }
            identity_values = {
                field: _digest(f"shared:{field}")
                for field in DeterminismSmokeArtifactIdentity.__dataclass_fields__
            }
            identity_values["artifact_manifest_sha256"] = _digest(
                f"manifest:{run_id}"
            )
            identity_values["scientific_evidence_sha256"] = self.scientific_sha256
            file_record = ArtifactFileRecord(
                path="artifact_manifest.json",
                size=100 + index,
                sha256=identity_values["artifact_manifest_sha256"],
            )
            artifact = ValidatedDeterminismSmokeArtifact(
                root=artifact_root,
                expectation=Mock(),
                identity=DeterminismSmokeArtifactIdentity(**identity_values),
                files=(file_record,),
                scientific_evidence={"fixture": "same"},
                run_path=artifact_root / "determinism_smoke_run.json",
                model_path=artifact_root / "model.safetensors",
            )
            self.acquisitions[run_id] = SimpleNamespace(
                receipt=acquisition_receipt,
                receipt_path=receipt_path,
                bundle_root=bundle_root,
                artifact_root=artifact_root,
                archive_sha256=archive_sha256,
                archive_size=archive_size,
                inventory_sha256=_digest(f"inventory:{run_id}"),
                file_count=1,
                total_size=file_record.size,
                remote_object=remote_object,
                request_receipt=request_receipt,
                preflight_receipt=preflight,
                submission_receipt=submission,
                terminal_receipt=terminal,
                validated_artifact=artifact,
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _loader(self, path: Path, **_kwargs: object) -> SimpleNamespace:
        matches = [
            acquisition
            for acquisition in self.acquisitions.values()
            if acquisition.receipt_path == path
        ]
        if len(matches) != 1:
            raise ValueError("fixture acquisition path is not unique")
        return matches[0]

    def _run(self, *, validate: bool = False):
        loader = Mock(side_effect=self._loader)
        comparator = Mock(return_value=_comparison(self.scientific_sha256))
        with (
            patch.object(
                determinism_gate.training_artifacts,
                "load_and_validate_determinism_smoke_acquisition_receipt",
                loader,
            ),
            patch.object(
                determinism_gate,
                "compare_smoke_scientific_evidence",
                comparator,
            ),
            patch.object(determinism_gate.aws, "make_clients", create=True) as clients,
        ):
            receipt = determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                acquisition_receipt_paths_by_run=self.paths,
            )
            if validate:
                self.assertEqual(
                    determinism_gate.validate_determinism_gate_receipt(
                        receipt,
                        training_plan=self.plan,
                        staging_receipt=self.staging,
                    ),
                    receipt,
                )
            clients.assert_not_called()
        return receipt, loader, comparator

    def test_gate_binds_complete_acquisitions_and_derives_requests(self) -> None:
        receipt, loader, comparator = self._run(validate=True)
        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(receipt["protocol"], determinism_gate.DETERMINISM_GATE_PROTOCOL)
        self.assertIs(receipt["exact_match"], True)
        self.assertEqual(
            receipt["receipt_sha256"],
            _document_sha256(
                {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            ),
        )
        self.assertEqual(
            [row["run_id"] for row in receipt["replicas"]],
            list(determinism_gate.SMOKE_RUN_IDS),
        )
        self.assertEqual(loader.call_count, 4)
        self.assertEqual(comparator.call_count, 2)
        for call in loader.call_args_list:
            self.assertIn(call.args[0], self.paths.values())
            self.assertEqual(call.kwargs["training_plan"], self.plan)
            self.assertEqual(call.kwargs["staging_receipt"], self.staging)
        for row, run_id in zip(receipt["replicas"], determinism_gate.SMOKE_RUN_IDS):
            acquisition = self.acquisitions[run_id]
            hashes = row["receipt_hashes"]
            self.assertEqual(
                hashes["request_sha256"],
                acquisition.request_receipt["request_sha256"],
            )
            self.assertEqual(
                hashes["request_receipt_sha256"],
                _document_sha256(acquisition.request_receipt),
            )
            self.assertEqual(
                hashes["preflight_receipt_sha256"],
                _document_sha256(acquisition.preflight_receipt),
            )
            self.assertEqual(
                hashes["submission_receipt_sha256"],
                _document_sha256(acquisition.submission_receipt),
            )
            self.assertEqual(
                hashes["status_receipt_sha256"],
                _document_sha256(acquisition.terminal_receipt["status_receipt"]),
            )
            self.assertEqual(
                hashes["terminal_receipt_sha256"],
                _document_sha256(acquisition.terminal_receipt),
            )
            self.assertEqual(
                hashes["acquisition_receipt_sha256"],
                _document_sha256(acquisition.receipt),
            )
            self.assertEqual(row["archive"]["sha256"], acquisition.archive_sha256)
            self.assertEqual(
                row["archive"]["remote_object"]["version_id"],
                acquisition.remote_object["version_id"],
            )
            self.assertEqual(
                row["artifact"]["inventory_sha256"], acquisition.inventory_sha256
            )

        # Transport coordinates are recorded and may differ; their wire class is exact.
        self.assertNotEqual(
            receipt["replicas"][0]["archive"]["sha256"],
            receipt["replicas"][1]["archive"]["sha256"],
        )
        self.assertNotEqual(
            receipt["replicas"][0]["archive"]["remote_object"]["version_id"],
            receipt["replicas"][1]["archive"]["remote_object"]["version_id"],
        )

    def test_mapping_paths_are_exact_and_old_loose_api_is_gone(self) -> None:
        missing = dict(self.paths)
        missing.pop("determinism-smoke-b")
        extra = dict(self.paths)
        extra["determinism-smoke-c"] = extra["determinism-smoke-a"]
        swapped = {
            "determinism-smoke-a": self.paths["determinism-smoke-b"],
            "determinism-smoke-b": self.paths["determinism-smoke-a"],
        }
        duplicate = dict(self.paths)
        duplicate["determinism-smoke-b"] = duplicate["determinism-smoke-a"]
        invalid_values = dict(self.paths)
        invalid_values["determinism-smoke-a"] = str(
            invalid_values["determinism-smoke-a"]
        )
        relative = dict(self.paths)
        relative["determinism-smoke-a"] = Path("acquisition_receipt.json")
        for invalid in (missing, extra, swapped, duplicate, invalid_values, relative):
            with self.subTest(paths=invalid):
                with patch.object(
                    determinism_gate.training_artifacts,
                    "load_and_validate_determinism_smoke_acquisition_receipt",
                    side_effect=self._loader,
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        determinism_gate.run_determinism_gate(
                            training_plan=self.plan,
                            staging_receipt=self.staging,
                            acquisition_receipt_paths_by_run=invalid,
                        )

        alias = self.root / "receipt-alias.json"
        alias.symlink_to(self.paths["determinism-smoke-a"])
        aliased = dict(self.paths)
        aliased["determinism-smoke-a"] = alias
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                acquisition_receipt_paths_by_run=aliased,
            )

        with self.assertRaises(TypeError):
            determinism_gate.run_determinism_gate(
                training_plan=self.plan,
                staging_receipt=self.staging,
                request_receipts_by_run=self.requests,
                artifact_roots_by_run={},
                artifact_manifest_sha256_by_run={},
            )

    def test_loader_result_cross_bindings_and_chain_replay_are_rejected(self) -> None:
        run_a, run_b = determinism_gate.SMOKE_RUN_IDS
        original = self.acquisitions[run_a]
        mutations: list[tuple[str, object]] = []

        wrong_path = copy.copy(original)
        wrong_path.receipt_path = self.paths[run_b]
        mutations.append(("receipt path", wrong_path))

        wrong_chain = copy.copy(original)
        wrong_chain.terminal_receipt = copy.deepcopy(original.terminal_receipt)
        wrong_chain.terminal_receipt["run_id"] = run_b
        mutations.append(("cross-run chain", wrong_chain))

        wrong_root = copy.copy(original)
        wrong_root.artifact_root = self.acquisitions[run_b].artifact_root
        mutations.append(("artifact root", wrong_root))

        wrong_inventory = copy.copy(original)
        wrong_inventory.total_size += 1
        mutations.append(("inventory", wrong_inventory))

        wrong_archive = copy.copy(original)
        wrong_archive.archive_sha256 = _digest("different archive")
        mutations.append(("archive", wrong_archive))

        wrong_type = copy.copy(original)
        wrong_type.validated_artifact = object()
        mutations.append(("artifact type", wrong_type))

        for label, mutation in mutations:
            with self.subTest(label=label):
                def loader(path: Path, **_kwargs: object):
                    if path == self.paths[run_a]:
                        return mutation
                    return self.acquisitions[run_b]

                with patch.object(
                    determinism_gate.training_artifacts,
                    "load_and_validate_determinism_smoke_acquisition_receipt",
                    side_effect=loader,
                ):
                    with self.assertRaises((TypeError, ValueError, RuntimeError)):
                        determinism_gate.run_determinism_gate(
                            training_plan=self.plan,
                            staging_receipt=self.staging,
                            acquisition_receipt_paths_by_run=self.paths,
                        )

    def test_remote_wire_class_mismatch_and_coordinate_replay_are_rejected(self) -> None:
        run_a, run_b = determinism_gate.SMOKE_RUN_IDS
        original = self.acquisitions[run_b]
        changed_wire = copy.copy(original)
        changed_wire.remote_object = copy.deepcopy(original.remote_object)
        changed_wire.remote_object["content_type"] = "application/octet-stream"
        self.acquisitions[run_b] = changed_wire
        with (
            patch.object(
                determinism_gate.training_artifacts,
                "load_and_validate_determinism_smoke_acquisition_receipt",
                side_effect=self._loader,
            ),
            patch.object(
                determinism_gate,
                "compare_smoke_scientific_evidence",
                return_value=_comparison(self.scientific_sha256),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "wire contracts differ"):
                determinism_gate.run_determinism_gate(
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    acquisition_receipt_paths_by_run=self.paths,
                )

        replayed = copy.copy(original)
        replayed.remote_object = copy.deepcopy(
            self.acquisitions[run_a].remote_object
        )
        replayed.archive_size = replayed.remote_object["size"]
        replayed.archive_sha256 = replayed.remote_object["sha256"]
        replayed.terminal_receipt = copy.deepcopy(original.terminal_receipt)
        replayed.terminal_receipt["model_artifact_s3_uri"] = replayed.remote_object[
            "s3_uri"
        ]
        self.acquisitions[run_b] = replayed
        with (
            patch.object(
                determinism_gate.training_artifacts,
                "load_and_validate_determinism_smoke_acquisition_receipt",
                side_effect=self._loader,
            ),
            patch.object(
                determinism_gate,
                "compare_smoke_scientific_evidence",
                return_value=_comparison(self.scientific_sha256),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "distinct model artifact URIs"):
                determinism_gate.run_determinism_gate(
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    acquisition_receipt_paths_by_run=self.paths,
                )

    def test_scientific_mismatch_stops_without_a_gate_receipt(self) -> None:
        with (
            patch.object(
                determinism_gate.training_artifacts,
                "load_and_validate_determinism_smoke_acquisition_receipt",
                side_effect=self._loader,
            ),
            patch.object(
                determinism_gate,
                "compare_smoke_scientific_evidence",
                side_effect=RuntimeError("scientific mismatch"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "scientific mismatch"):
                determinism_gate.run_determinism_gate(
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    acquisition_receipt_paths_by_run=self.paths,
                )

    def test_contextual_receipt_validation_rejects_resealed_attacks_and_v1(self) -> None:
        receipt, _, _ = self._run()
        mutations: list[dict[str, object]] = []

        extra = copy.deepcopy(receipt)
        extra["retry"] = False
        mutations.append(extra)

        boolean_schema = copy.deepcopy(receipt)
        boolean_schema["schema_version"] = True
        mutations.append(boolean_schema)

        old_protocol = copy.deepcopy(receipt)
        old_protocol["schema_version"] = 1
        old_protocol["protocol"] = "retrieval_cv_two_replica_determinism_gate_v1"
        mutations.append(_reseal_gate(old_protocol))

        swapped = copy.deepcopy(receipt)
        swapped["replicas"].reverse()
        mutations.append(_reseal_gate(swapped))

        relative = copy.deepcopy(receipt)
        relative["replicas"][0]["acquisition_receipt_path"] = "relative.json"
        mutations.append(_reseal_gate(relative))

        changed_acquisition_hash = copy.deepcopy(receipt)
        changed_acquisition_hash["replicas"][0]["receipt_hashes"][
            "acquisition_receipt_sha256"
        ] = "f" * 64
        mutations.append(_reseal_gate(changed_acquisition_hash))

        changed_version = copy.deepcopy(receipt)
        changed_version["replicas"][0]["archive"]["remote_object"][
            "version_id"
        ] = "different-version"
        mutations.append(_reseal_gate(changed_version))

        changed_internal_self_hash = copy.deepcopy(receipt)
        changed_internal_self_hash["replicas"][0]["receipt_hashes"][
            "preflight_receipt_sha256"
        ] = self.acquisitions["determinism-smoke-a"].preflight_receipt[
            "receipt_sha256"
        ]
        mutations.append(_reseal_gate(changed_internal_self_hash))

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with (
                    patch.object(
                        determinism_gate.training_artifacts,
                        "load_and_validate_determinism_smoke_acquisition_receipt",
                        side_effect=self._loader,
                    ),
                    patch.object(
                        determinism_gate,
                        "compare_smoke_scientific_evidence",
                        return_value=_comparison(self.scientific_sha256),
                    ),
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        determinism_gate.validate_determinism_gate_receipt(
                            mutation,
                            training_plan=self.plan,
                            staging_receipt=self.staging,
                        )


if __name__ == "__main__":
    unittest.main()
