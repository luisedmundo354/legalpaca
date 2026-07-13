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
from corporate_reorganization.modernbert.tests import (
    test_retrieval_determinism as determinism_test_helpers,
)
from corporate_reorganization.modernbert.retriever.artifacts import ArtifactFileRecord
from corporate_reorganization.modernbert.retriever.determinism import (
    SMOKE_EXPECTED_MODEL_TENSOR_COUNT,
    SMOKE_MODEL_STATE_PROTOCOL,
    SMOKE_RUN_KIND,
)
from corporate_reorganization.modernbert.retriever.determinism_artifacts import (
    DeterminismSmokeArtifactIdentity,
    ValidatedDeterminismSmokeArtifact,
)
from corporate_reorganization.modernbert.retriever import determinism_artifacts


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_SYNTHETIC_MODEL_INVENTORY = {
    f"tensor_{index:03d}": (1,) for index in range(SMOKE_EXPECTED_MODEL_TENSOR_COUNT)
}


def _write_synthetic_safetensors(path: Path) -> None:
    data = b""
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "source": "fresh_best_engine_zero3_gathered_16bit_state",
            "weight_dtype": "bfloat16",
        }
    }
    for tensor_name in sorted(_SYNTHETIC_MODEL_INVENTORY):
        start = len(data)
        data += b"\x80\x3f"
        header[tensor_name] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [start, len(data)],
        }
    _write_safetensors_parts(path, header, data)


def _safetensors_parts(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    header_length = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_length].rstrip(b" "))
    return header, raw[8 + header_length :]


def _write_safetensors_parts(
    path: Path,
    header: dict[str, object],
    data: bytes,
    *,
    extra_header_padding: int = 0,
    trailing: bytes = b"",
) -> None:
    raw_header = json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    raw_header += b" " * extra_header_padding
    path.write_bytes(
        len(raw_header).to_bytes(8, "little") + raw_header + data + trailing
    )


def _comparison(
    scientific_sha256: str,
    *,
    evidence_sha256: list[str],
    model_file_sha256: list[str],
    model_state_sha256: str,
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "protocol": determinism_gate.DETERMINISM_SCIENTIFIC_COMPARISON_PROTOCOL,
        "run_kind": SMOKE_RUN_KIND,
        "normalized_scientific_identity_sha256": scientific_sha256,
        "replica_evidence_sha256": evidence_sha256,
        "model_serialization": {
            "protocol": (
                determinism_gate.DETERMINISM_MODEL_SERIALIZATION_COMPARISON_PROTOCOL
            ),
            "replica_model_file_sha256": model_file_sha256,
            "common_serialization_semantics": {
                "protocol": (
                    determinism_gate.DETERMINISM_SAFETENSORS_SERIALIZATION_PROTOCOL
                ),
                "file_size": 1_024,
                "header_length": 512,
                "canonical_parsed_header_sha256": _digest(
                    "common parsed safetensors header"
                ),
                "metadata_order_normalized_raw_header_sha256": _digest(
                    "common metadata-order-normalized raw header"
                ),
            },
            "canonical_model_state": {
                "protocol": SMOKE_MODEL_STATE_PROTOCOL,
                "tensor_count": SMOKE_EXPECTED_MODEL_TENSOR_COUNT,
                "sha256": model_state_sha256,
            },
        },
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


class SemanticExactScientificComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        determinism_test_helpers.SmokeScientificEvidenceTest.setUpClass()
        cls.evidence_factory = determinism_test_helpers.SmokeScientificEvidenceTest(
            "test_exact_evidence_roundtrip_and_comparison_receipt"
        )

    def _evidence(
        self,
        *,
        model_file_sha256: str,
        model_state: dict[str, object],
        changed_final_field: str | None = None,
    ) -> dict[str, object]:
        baseline = self.evidence_factory.make_evidence()
        final_artifacts = copy.deepcopy(baseline["final_artifacts"])
        final_artifacts["model_sha256"] = model_file_sha256
        if changed_final_field is not None:
            final_artifacts[changed_final_field] = _digest(
                f"changed:{changed_final_field}"
            )
        return self.evidence_factory.make_evidence(
            selected_model_state=model_state,
            roundtrip_model_state=copy.deepcopy(model_state),
            final_artifacts=final_artifacts,
        )

    def _artifact(
        self,
        *,
        evidence: dict[str, object],
        model_path: Path,
        model_state_sha256: str,
    ) -> ValidatedDeterminismSmokeArtifact:
        identity_values = {
            field: _digest(f"semantic-comparison:{field}")
            for field in DeterminismSmokeArtifactIdentity.__dataclass_fields__
        }
        identity_values["scientific_evidence_sha256"] = evidence["sha256"]
        identity_values["model_file_sha256"] = evidence["final_artifacts"][
            "model_sha256"
        ]
        identity_values["model_state_sha256"] = model_state_sha256
        return ValidatedDeterminismSmokeArtifact(
            root=model_path.parent,
            expectation=Mock(),
            identity=DeterminismSmokeArtifactIdentity(**identity_values),
            files=(),
            scientific_evidence=evidence,
            run_path=model_path.parent / "determinism_smoke_run.json",
            model_path=model_path,
        )

    def test_metadata_key_order_is_the_only_normalized_file_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.safetensors"
            second_path = root / "second.safetensors"
            with patch.object(
                determinism_artifacts,
                "_EXPECTED_MODEL_TENSOR_SHAPES",
                _SYNTHETIC_MODEL_INVENTORY,
            ):
                _write_synthetic_safetensors(first_path)
                first_state = (
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        first_path
                    )
                )
                header, data = _safetensors_parts(first_path)
                header["__metadata__"] = {
                    "source": "fresh_best_engine_zero3_gathered_16bit_state",
                    "weight_dtype": "bfloat16",
                    "format": "pt",
                }
                _write_safetensors_parts(second_path, header, data)
                second_state = (
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        second_path
                    )
                )

            first_raw_sha256 = _file_sha256(first_path)
            second_raw_sha256 = _file_sha256(second_path)
            self.assertNotEqual(first_raw_sha256, second_raw_sha256)
            self.assertEqual(first_path.stat().st_size, second_path.stat().st_size)
            self.assertEqual(first_state, second_state)
            first_evidence = self._evidence(
                model_file_sha256=first_raw_sha256,
                model_state=first_state,
            )
            second_evidence = self._evidence(
                model_file_sha256=second_raw_sha256,
                model_state=second_state,
            )
            first_artifact = self._artifact(
                evidence=first_evidence,
                model_path=first_path,
                model_state_sha256=first_state["sha256"],
            )
            second_artifact = self._artifact(
                evidence=second_evidence,
                model_path=second_path,
                model_state_sha256=second_state["sha256"],
            )

            receipt = determinism_gate._compare_semantic_exact_scientific_evidence(
                first_artifact,
                second_artifact,
            )
            self.assertIs(receipt["exact_match"], True)
            self.assertNotEqual(first_evidence["sha256"], second_evidence["sha256"])
            self.assertEqual(
                receipt["replica_evidence_sha256"],
                [first_evidence["sha256"], second_evidence["sha256"]],
            )
            self.assertEqual(
                receipt["model_serialization"]["replica_model_file_sha256"],
                [first_raw_sha256, second_raw_sha256],
            )
            self.assertEqual(
                receipt["model_serialization"]["canonical_model_state"],
                first_state,
            )
            semantics = receipt["model_serialization"][
                "common_serialization_semantics"
            ]
            self.assertEqual(semantics["file_size"], first_path.stat().st_size)
            self.assertEqual(
                semantics["header_length"],
                int.from_bytes(first_path.read_bytes()[:8], "little"),
            )
            reverse = determinism_gate._compare_semantic_exact_scientific_evidence(
                second_artifact,
                first_artifact,
            )
            self.assertEqual(
                receipt["normalized_scientific_identity_sha256"],
                reverse["normalized_scientific_identity_sha256"],
            )

    def test_nonmetadata_header_changes_are_not_normalized(self) -> None:
        attacks = (
            "header-length",
            "descriptor-offsets",
            "descriptor-key-order",
            "top-level-key-order",
        )
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first_path = root / "first.safetensors"
                second_path = root / "second.safetensors"
                with patch.object(
                    determinism_artifacts,
                    "_EXPECTED_MODEL_TENSOR_SHAPES",
                    _SYNTHETIC_MODEL_INVENTORY,
                ):
                    _write_synthetic_safetensors(first_path)
                    first_state = (
                        determinism_artifacts._canonical_bf16_safetensors_identity(
                            first_path
                        )
                    )
                    header, data = _safetensors_parts(first_path)
                    if attack == "header-length":
                        _write_safetensors_parts(
                            second_path,
                            header,
                            data,
                            extra_header_padding=8,
                        )
                    elif attack == "descriptor-offsets":
                        first_offsets = header["tensor_000"]["data_offsets"]
                        second_offsets = header["tensor_001"]["data_offsets"]
                        header["tensor_000"]["data_offsets"] = second_offsets
                        header["tensor_001"]["data_offsets"] = first_offsets
                        _write_safetensors_parts(second_path, header, data)
                    elif attack == "descriptor-key-order":
                        descriptor = header["tensor_000"]
                        header["tensor_000"] = {
                            "shape": descriptor["shape"],
                            "data_offsets": descriptor["data_offsets"],
                            "dtype": descriptor["dtype"],
                        }
                        _write_safetensors_parts(second_path, header, data)
                    else:
                        first_descriptor = header.pop("tensor_000")
                        header["tensor_000"] = first_descriptor
                        _write_safetensors_parts(second_path, header, data)
                    second_state = (
                        determinism_artifacts._canonical_bf16_safetensors_identity(
                            second_path
                        )
                    )
                self.assertEqual(first_state, second_state)
                first_evidence = self._evidence(
                    model_file_sha256=_file_sha256(first_path),
                    model_state=first_state,
                )
                second_evidence = self._evidence(
                    model_file_sha256=_file_sha256(second_path),
                    model_state=second_state,
                )
                first_artifact = self._artifact(
                    evidence=first_evidence,
                    model_path=first_path,
                    model_state_sha256=first_state["sha256"],
                )
                second_artifact = self._artifact(
                    evidence=second_evidence,
                    model_path=second_path,
                    model_state_sha256=second_state["sha256"],
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "serialization semantics differ",
                ):
                    determinism_gate._compare_semantic_exact_scientific_evidence(
                        first_artifact,
                        second_artifact,
                    )

    def test_equal_length_metadata_whitespace_change_is_not_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.safetensors"
            second_path = root / "second.safetensors"
            with patch.object(
                determinism_artifacts,
                "_EXPECTED_MODEL_TENSOR_SHAPES",
                _SYNTHETIC_MODEL_INVENTORY,
            ):
                _write_synthetic_safetensors(first_path)
                header, data = _safetensors_parts(first_path)
                _write_safetensors_parts(
                    first_path,
                    header,
                    data,
                    extra_header_padding=8,
                )
                raw = first_path.read_bytes()
                header_length = int.from_bytes(raw[:8], "little")
                header_raw = raw[8 : 8 + header_length]
                data = raw[8 + header_length :]
                self.assertTrue(header_raw.endswith(b" "))
                changed_header = header_raw.replace(
                    b',"source"', b', "source"', 1
                )[:-1]
                self.assertEqual(len(changed_header), len(header_raw))
                second_path.write_bytes(raw[:8] + changed_header + data)
                first_state = (
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        first_path
                    )
                )
                second_state = (
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        second_path
                    )
                )
            self.assertEqual(first_state, second_state)
            first_evidence = self._evidence(
                model_file_sha256=_file_sha256(first_path),
                model_state=first_state,
            )
            second_evidence = self._evidence(
                model_file_sha256=_file_sha256(second_path),
                model_state=second_state,
            )
            first_artifact = self._artifact(
                evidence=first_evidence,
                model_path=first_path,
                model_state_sha256=first_state["sha256"],
            )
            second_artifact = self._artifact(
                evidence=second_evidence,
                model_path=second_path,
                model_state_sha256=second_state["sha256"],
            )
            with self.assertRaisesRegex(
                ValueError,
                "differs beyond member ordering",
            ):
                determinism_gate._compare_semantic_exact_scientific_evidence(
                    first_artifact,
                    second_artifact,
                )

    def test_model_path_swap_after_open_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.safetensors"
            replacement_path = root / "replacement.safetensors"
            opened_path = root / "opened.safetensors"
            _write_synthetic_safetensors(model_path)
            _write_synthetic_safetensors(replacement_path)
            replacement_path.write_bytes(replacement_path.read_bytes() + b"different")
            expected_sha256 = _file_sha256(model_path)
            real_open = determinism_gate.os.open

            def swap_after_open(raw_path: object, flags: int) -> int:
                file_descriptor = real_open(raw_path, flags)
                model_path.rename(opened_path)
                replacement_path.rename(model_path)
                return file_descriptor

            with (
                patch.object(
                    determinism_gate.os,
                    "open",
                    side_effect=swap_after_open,
                ),
                self.assertRaisesRegex(RuntimeError, "path identity changed"),
            ):
                determinism_gate._safetensors_serialization_semantics(
                    model_path,
                    expected_raw_sha256=expected_sha256,
                )

    def test_metadata_descriptor_and_file_size_changes_fail_artifact_parsing(
        self,
    ) -> None:
        attacks = ("metadata-value", "metadata-key", "descriptor", "file-size")
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                model_path = Path(temporary) / "model.safetensors"
                with patch.object(
                    determinism_artifacts,
                    "_EXPECTED_MODEL_TENSOR_SHAPES",
                    _SYNTHETIC_MODEL_INVENTORY,
                ):
                    _write_synthetic_safetensors(model_path)
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        model_path
                    )
                    header, data = _safetensors_parts(model_path)
                    trailing = b""
                    if attack == "metadata-value":
                        header["__metadata__"]["format"] = "different"
                    elif attack == "metadata-key":
                        header["__metadata__"]["extra"] = "different"
                    elif attack == "descriptor":
                        header["tensor_000"]["shape"] = [2]
                    else:
                        trailing = b"\x00"
                    _write_safetensors_parts(
                        model_path,
                        header,
                        data,
                        trailing=trailing,
                    )
                    with self.assertRaises(ValueError):
                        determinism_artifacts._canonical_bf16_safetensors_identity(
                            model_path
                        )

    def test_tensor_byte_and_other_final_artifact_changes_fail_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.safetensors"
            second_path = root / "second.safetensors"
            with patch.object(
                determinism_artifacts,
                "_EXPECTED_MODEL_TENSOR_SHAPES",
                _SYNTHETIC_MODEL_INVENTORY,
            ):
                _write_synthetic_safetensors(first_path)
                first_state = (
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        first_path
                    )
                )
                header, data = _safetensors_parts(first_path)
                _write_safetensors_parts(
                    second_path,
                    header,
                    data[:-1] + bytes([data[-1] ^ 1]),
                )
                second_state = (
                    determinism_artifacts._canonical_bf16_safetensors_identity(
                        second_path
                    )
                )
            self.assertNotEqual(first_state, second_state)
            first_evidence = self._evidence(
                model_file_sha256=_file_sha256(first_path),
                model_state=first_state,
            )
            second_evidence = self._evidence(
                model_file_sha256=_file_sha256(second_path),
                model_state=second_state,
            )
            first_artifact = self._artifact(
                evidence=first_evidence,
                model_path=first_path,
                model_state_sha256=first_state["sha256"],
            )
            second_artifact = self._artifact(
                evidence=second_evidence,
                model_path=second_path,
                model_state_sha256=second_state["sha256"],
            )
            with self.assertRaisesRegex(RuntimeError, "model states differ"):
                determinism_gate._compare_semantic_exact_scientific_evidence(
                    first_artifact,
                    second_artifact,
                )

            changed_final_evidence = self._evidence(
                model_file_sha256=_file_sha256(first_path),
                model_state=first_state,
                changed_final_field="tokenizer_inventory_sha256",
            )
            changed_final_artifact = self._artifact(
                evidence=changed_final_evidence,
                model_path=first_path,
                model_state_sha256=first_state["sha256"],
            )
            with self.assertRaisesRegex(RuntimeError, "final_artifacts"):
                determinism_gate._compare_semantic_exact_scientific_evidence(
                    first_artifact,
                    changed_final_artifact,
                )


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
        self.model_state_sha256 = _digest("identical-model-state")
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
            identity_values["scientific_evidence_sha256"] = _digest(
                f"scientific-evidence:{run_id}"
            )
            identity_values["model_file_sha256"] = _digest(
                f"raw-model-file:{run_id}"
            )
            identity_values["model_state_sha256"] = self.model_state_sha256
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
        comparator = Mock(return_value=self._comparison())
        with (
            patch.object(
                determinism_gate.training_artifacts,
                "load_and_validate_determinism_smoke_acquisition_receipt",
                loader,
            ),
            patch.object(
                determinism_gate,
                "_compare_semantic_exact_scientific_evidence",
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

    def _comparison(self) -> dict[str, object]:
        identities = [
            self.acquisitions[run_id].validated_artifact.identity
            for run_id in determinism_gate.SMOKE_RUN_IDS
        ]
        return _comparison(
            self.scientific_sha256,
            evidence_sha256=[
                identity.scientific_evidence_sha256 for identity in identities
            ],
            model_file_sha256=[identity.model_file_sha256 for identity in identities],
            model_state_sha256=self.model_state_sha256,
        )

    def test_gate_binds_complete_acquisitions_and_derives_requests(self) -> None:
        receipt, loader, comparator = self._run(validate=True)
        self.assertEqual(receipt["schema_version"], 3)
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
        comparison = receipt["scientific_comparison"]
        identities = [
            self.acquisitions[run_id].validated_artifact.identity
            for run_id in determinism_gate.SMOKE_RUN_IDS
        ]
        self.assertEqual(
            comparison["replica_evidence_sha256"],
            [identity.scientific_evidence_sha256 for identity in identities],
        )
        self.assertEqual(
            comparison["model_serialization"]["replica_model_file_sha256"],
            [identity.model_file_sha256 for identity in identities],
        )
        self.assertEqual(
            comparison["model_serialization"]["canonical_model_state"]["sha256"],
            self.model_state_sha256,
        )
        self.assertNotEqual(
            identities[0].scientific_evidence_sha256,
            identities[1].scientific_evidence_sha256,
        )
        self.assertNotEqual(
            identities[0].model_file_sha256,
            identities[1].model_file_sha256,
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
                "_compare_semantic_exact_scientific_evidence",
                return_value=self._comparison(),
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
                "_compare_semantic_exact_scientific_evidence",
                return_value=self._comparison(),
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
                "_compare_semantic_exact_scientific_evidence",
                side_effect=RuntimeError("scientific mismatch"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "scientific mismatch"):
                determinism_gate.run_determinism_gate(
                    training_plan=self.plan,
                    staging_receipt=self.staging,
                    acquisition_receipt_paths_by_run=self.paths,
                )

    def test_contextual_receipt_validation_rejects_resealed_attacks_and_v2(self) -> None:
        receipt, _, _ = self._run()
        mutations: list[dict[str, object]] = []

        extra = copy.deepcopy(receipt)
        extra["retry"] = False
        mutations.append(extra)

        boolean_schema = copy.deepcopy(receipt)
        boolean_schema["schema_version"] = True
        mutations.append(boolean_schema)

        old_protocol = copy.deepcopy(receipt)
        old_protocol["schema_version"] = 2
        old_protocol["protocol"] = "retrieval_cv_two_replica_determinism_gate_v2"
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

        changed_evidence_binding = copy.deepcopy(receipt)
        changed_evidence_binding["replicas"][0]["artifact"]["identity"][
            "scientific_evidence_sha256"
        ] = "e" * 64
        mutations.append(_reseal_gate(changed_evidence_binding))

        changed_model_file_binding = copy.deepcopy(receipt)
        changed_model_file_binding["replicas"][1]["artifact"]["identity"][
            "model_file_sha256"
        ] = "e" * 64
        mutations.append(_reseal_gate(changed_model_file_binding))

        changed_model_state_binding = copy.deepcopy(receipt)
        changed_model_state_binding["replicas"][1]["artifact"]["identity"][
            "model_state_sha256"
        ] = "e" * 64
        mutations.append(_reseal_gate(changed_model_state_binding))

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
                        "_compare_semantic_exact_scientific_evidence",
                        return_value=self._comparison(),
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
