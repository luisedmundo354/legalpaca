from __future__ import annotations

import copy
import base64
import hashlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import Mock, patch


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))


from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    aws,
    controlled_supervisor,
    fold_processing_aws,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_controlled_supervisor import (
    _LaunchHarness,
    _determinism_gate_receipt,
    _identity_validator,
)
from corporate_reorganization.modernbert.tests.test_retrieval_cv_training_aws import (
    _staging_receipt,
    _training_plan,
)


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    payload = {
        key: copy.deepcopy(nested)
        for key, nested in value.items()
        if key != "receipt_sha256"
    }
    payload["receipt_sha256"] = _document_sha256(payload)
    return payload


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _static_inventory_expectations(
    assets: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        asset["name"]: {
            "inventory_sha256": _document_sha256(asset["files"]),
            "file_count": len(asset["files"]),
            "total_size": sum(record["size"] for record in asset["files"]),
        }
        for asset in assets
    }


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(controlled_supervisor.strict_config.canonical_json_bytes(value))


class CompletedFoldEvidenceTest(unittest.TestCase):
    """The AWS boundary may consume only a complete immutable fold view."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.plan, _ = _training_plan(self.root / "plan")
        self.staging = _staging_receipt(self.plan)
        self.gate = _determinism_gate_receipt(self.plan, self.staging)
        self.state_dir = self.root / "controlled-supervisor"
        self.harness = _LaunchHarness(self.plan)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @contextmanager
    def _runtime(self):
        with (
            patch.object(
                controlled_supervisor.determinism_gate,
                "validate_determinism_gate_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "preflight_training_job",
                side_effect=self.harness.preflight_training_job,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "submit_training_job_once",
                side_effect=self.harness.submit_training_job_once,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "describe_training_job_status",
                side_effect=self.harness.describe_training_job_status,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "verify_terminal_training_job",
                side_effect=self.harness.verify_terminal_training_job,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_preflight_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_submission_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_status_receipt",
                side_effect=_identity_validator,
            ),
            patch.object(
                controlled_supervisor.training_launch,
                "validate_training_terminal_receipt",
                side_effect=_identity_validator,
            ),
        ):
            yield

    def _initialize(self) -> controlled_supervisor.ControlledTrainingSupervisor:
        controlled_supervisor.initialize_controlled_supervisor_state(
            state_dir=self.state_dir,
            training_plan=self.plan,
            staging_receipt=self.staging,
            determinism_gate_receipt=self.gate,
        )
        return controlled_supervisor.ControlledTrainingSupervisor(
            self.harness.clients,
            state_dir=self.state_dir,
        )

    def _complete_fold_zero(self) -> dict[str, object]:
        supervisor = self._initialize()
        supervisor.advance_once()
        self.harness.default_status = "Completed"
        supervisor.advance_once()
        supervisor.advance_once()
        supervisor.advance_once()
        return controlled_supervisor.load_completed_fold_evidence(
            state_dir=self.state_dir,
            outer_fold=0,
        )

    def test_exact_twelve_completed_cells_are_loaded_in_canonical_order(self) -> None:
        with self._runtime():
            evidence = self._complete_fold_zero()
            validated = controlled_supervisor.validate_completed_fold_evidence(evidence)

        self.assertEqual(
            evidence["protocol"],
            controlled_supervisor.COMPLETED_FOLD_EVIDENCE_PROTOCOL,
        )
        self.assertEqual(evidence["outer_fold"], 0)
        self.assertEqual(len(evidence["systems"]), 12)
        self.assertEqual(
            [
                (
                    system["cell"]["experiment_seed"],
                    system["cell"]["query_view"],
                    system["cell"]["sampler"],
                )
                for system in evidence["systems"]
            ],
            sorted(
                [
                    (seed, view, sampler)
                    for seed in (17, 29, 43)
                    for view in ("flat_masked", "structured")
                    for sampler in ("local_unique", "global_uniform")
                ],
                key=lambda item: f"{item[1]}_{item[2]}_seed{item[0]}",
            ),
        )
        self.assertEqual(
            evidence["training_plan_sha256"],
            _document_sha256(self.plan),
        )
        self.assertEqual(
            evidence["training_staging_receipt_sha256"],
            _document_sha256(self.staging),
        )
        self.assertEqual(evidence, validated)

    def test_eleven_completed_cells_do_not_form_a_fold(self) -> None:
        missing = "controlled-f0-struct-global-s43"
        with self._runtime():
            supervisor = self._initialize()
            supervisor.advance_once()
            self.harness.default_status = "Completed"
            supervisor.advance_once()
            supervisor.advance_once()
            self.harness.status_by_run_id[missing] = "InProgress"
            supervisor.advance_once()
            with self.assertRaisesRegex(RuntimeError, "all twelve successful"):
                controlled_supervisor.load_completed_fold_evidence(
                    state_dir=self.state_dir,
                    outer_fold=0,
                )

        missing_entry = next(
            entry
            for entry in supervisor._supervisor["schedule"]
            if entry["run_id"] == missing
        )
        missing_root = self.state_dir / "runs" / (
            f"{missing_entry['queue_index']:02d}-{missing}"
        )
        self.assertFalse((missing_root / "terminal.json").exists())

    def test_resealed_terminal_splice_is_rejected(self) -> None:
        with self._runtime():
            evidence = self._complete_fold_zero()
            changed = copy.deepcopy(evidence)
            changed["systems"][0]["terminal_receipt"]["run_id"] = changed[
                "systems"
            ][1]["run_id"]
            changed = _reseal(changed)
            with self.assertRaisesRegex(ValueError, "receipt chain changed"):
                controlled_supervisor.validate_completed_fold_evidence(changed)

    def test_resealed_queue_index_swap_is_rejected(self) -> None:
        with self._runtime():
            evidence = self._complete_fold_zero()
            changed = copy.deepcopy(evidence)
            changed["systems"][0]["queue_index"], changed["systems"][1][
                "queue_index"
            ] = (
                changed["systems"][1]["queue_index"],
                changed["systems"][0]["queue_index"],
            )
            changed = _reseal(changed)
            with self.assertRaisesRegex(ValueError, "launch identity changed"):
                controlled_supervisor.validate_completed_fold_evidence(changed)

    def test_resealed_source_bundle_splice_is_rejected(self) -> None:
        with self._runtime():
            evidence = self._complete_fold_zero()
            changed = copy.deepcopy(evidence)
            changed["source_bundle"]["sha256"] = "f" * 64
            changed = _reseal(changed)
            with self.assertRaisesRegex(
                ValueError,
                "plan/staging/source binding changed",
            ):
                controlled_supervisor.validate_completed_fold_evidence(changed)

    def test_supervisor_manifest_mutation_during_load_is_rejected(self) -> None:
        with self._runtime():
            supervisor = self._initialize()
            supervisor.advance_once()
            self.harness.default_status = "Completed"
            supervisor.advance_once()
            supervisor.advance_once()
            supervisor.advance_once()

            real_loader = controlled_supervisor._load_supervisor_manifest
            calls = 0

            def load_then_change(state_dir: Path, *, deep_gate: bool):
                nonlocal calls
                calls += 1
                loaded, digest = real_loader(state_dir, deep_gate=deep_gate)
                if calls == 2:
                    loaded = copy.deepcopy(loaded)
                    loaded["max_active"] = 3
                return loaded, digest

            with patch.object(
                controlled_supervisor,
                "_load_supervisor_manifest",
                side_effect=load_then_change,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "manifest changed during completed-fold loading",
                ):
                    controlled_supervisor.load_completed_fold_evidence(
                        state_dir=self.state_dir,
                        outer_fold=0,
                    )

    def test_completed_fold_arguments_are_strict(self) -> None:
        for outer_fold in (-1, 5, True, 0.0, "0"):
            with self.subTest(outer_fold=outer_fold):
                with self.assertRaises((TypeError, ValueError)):
                    controlled_supervisor.load_completed_fold_evidence(
                        state_dir=self.state_dir,
                        outer_fold=outer_fold,
                    )


class StaticSourceValidationTest(unittest.TestCase):
    """Tiny trees exercise the immutable-source boundary without model downloads."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.snapshot_dir = self.root / "e5-snapshot"
        self.pack_dir = self.root / "e5-pack"
        self.fixed_dir = self.root / "fixed-base"
        for directory in (self.snapshot_dir, self.pack_dir, self.fixed_dir):
            directory.mkdir()

        (self.snapshot_dir / "config.json").write_bytes(b"snapshot config\n")
        (self.snapshot_dir / "model.safetensors").write_bytes(b"snapshot model\n")
        self.snapshot_manifest_path = self.root / "e5_snapshot.json"
        _write_canonical(self.snapshot_manifest_path, {"fixture": "snapshot"})
        _write_canonical(
            self.root / "evaluation_baselines.json",
            {"fixture": "baselines"},
        )
        _write_canonical(self.root / "experiment.json", {"fixture": "experiment"})
        _write_canonical(self.root / "folds.json", {"fixture": "folds"})
        self.control_hashes = {
            name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            for name in (
                "e5_snapshot.json",
                "evaluation_baselines.json",
                "experiment.json",
                "folds.json",
            )
        }

        (self.pack_dir / "packed_queries.jsonl").write_bytes(b'{"query_id":"q1"}\n')
        packed_record = _file_record(
            self.pack_dir / "packed_queries.jsonl",
            relative_to=self.pack_dir,
        )
        packed_record["records"] = 490
        self.pack_inventory_sha256 = "9" * 64
        _write_canonical(
            self.pack_dir / "manifest.json",
            {
                "packed_queries_file": packed_record,
                "packed_query_inventory_sha256": self.pack_inventory_sha256,
            },
        )
        self.pack_manifest_sha256 = hashlib.sha256(
            (self.pack_dir / "manifest.json").read_bytes()
        ).hexdigest()

        (self.fixed_dir / "artifact_manifest.json").write_bytes(b"fixed manifest\n")
        (self.fixed_dir / "model.safetensors").write_bytes(b"fixed model\n")
        self.static_inventory_expectations = _static_inventory_expectations(
            [
                {
                    "name": "e5-snapshot",
                    "files": fold_processing_aws._directory_inventory(
                        self.snapshot_dir,
                        name="synthetic E5 snapshot",
                    ),
                },
                {
                    "name": "e5-pack",
                    "files": fold_processing_aws._directory_inventory(
                        self.pack_dir,
                        name="synthetic E5 pack",
                    ),
                },
                {
                    "name": "fixed-base",
                    "files": fold_processing_aws._directory_inventory(
                        self.fixed_dir,
                        name="synthetic fixed base",
                    ),
                },
                {
                    "name": "control",
                    "files": [
                        _file_record(self.root / name, relative_to=self.root)
                        for name in (
                            "e5_snapshot.json",
                            "evaluation_baselines.json",
                            "experiment.json",
                            "folds.json",
                        )
                    ],
                },
            ]
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _snapshot_identity(self, *, files=None):
        records = fold_processing_aws._directory_inventory(
            self.snapshot_dir,
            name="synthetic E5 snapshot",
        )
        return SimpleNamespace(
            files=(
                tuple((row["path"], row["size"], row["sha256"]) for row in records)
                if files is None
                else files
            ),
            manifest_sha256=self.control_hashes["e5_snapshot.json"],
            model_id=fold_processing_aws.E5_MODEL_ID,
            revision=fold_processing_aws.E5_REVISION,
            tree_sha256=fold_processing_aws.E5_SNAPSHOT_TREE_SHA256,
        )

    def _fixed_identity(self):
        return SimpleNamespace(
            manifest_sha256="4" * 64,
            model_sha256=fold_processing_aws.EXPECTED_FIXED_BASE_MODEL_SHA256,
            state_key_sha256=fold_processing_aws.EXPECTED_FIXED_BASE_STATE_KEYS_SHA256,
            new_embedding_rows_sha256=(
                fold_processing_aws.EXPECTED_FIXED_BASE_NEW_ROWS_SHA256
            ),
        )

    @contextmanager
    def _patched_validators(self, *, snapshot=None, fixed=None):
        with (
            patch.object(
                fold_processing_aws,
                "validate_snapshot",
                return_value=snapshot or self._snapshot_identity(),
            ) as validate_snapshot,
            patch.object(
                fold_processing_aws,
                "validate_fixed_base_artifact",
                return_value=fixed or self._fixed_identity(),
            ) as validate_fixed,
            patch.object(
                fold_processing_aws,
                "EXPECTED_E5_PACK_MANIFEST_SHA256",
                self.pack_manifest_sha256,
            ),
            patch.object(
                fold_processing_aws,
                "EXPECTED_E5_PACK_INVENTORY_SHA256",
                self.pack_inventory_sha256,
            ),
            patch.object(
                fold_processing_aws,
                "E5_SNAPSHOT_MANIFEST_SHA256",
                self.control_hashes["e5_snapshot.json"],
            ),
            patch.object(
                fold_processing_aws,
                "EXPECTED_BASELINE_CONFIG_SHA256",
                self.control_hashes["evaluation_baselines.json"],
            ),
            patch.object(
                fold_processing_aws,
                "EXPECTED_EXPERIMENT_CONFIG_SHA256",
                self.control_hashes["experiment.json"],
            ),
            patch.object(
                fold_processing_aws,
                "EXPECTED_FOLD_MANIFEST_SHA256",
                self.control_hashes["folds.json"],
            ),
            patch.object(
                fold_processing_aws,
                "EXPECTED_STATIC_INVENTORIES",
                copy.deepcopy(self.static_inventory_expectations),
            ),
        ):
            yield validate_snapshot, validate_fixed

    def test_tiny_exact_sources_produce_four_canonical_assets(self) -> None:
        with self._patched_validators() as (validate_snapshot, validate_fixed):
            assets = fold_processing_aws._validate_static_sources(
                e5_snapshot_dir=self.snapshot_dir,
                e5_snapshot_manifest_path=self.snapshot_manifest_path,
                e5_pack_dir=self.pack_dir,
                fixed_base_dir=self.fixed_dir,
            )

        self.assertEqual(
            [asset["name"] for asset in assets],
            ["e5-snapshot", "e5-pack", "fixed-base", "control"],
        )
        self.assertEqual(
            [row["path"] for row in assets[0]["files"]],
            ["config.json", "model.safetensors"],
        )
        self.assertEqual(
            [row["path"] for row in assets[1]["files"]],
            ["manifest.json", "packed_queries.jsonl"],
        )
        self.assertEqual(
            [row["path"] for row in assets[3]["files"]],
            [
                "e5_snapshot.json",
                "evaluation_baselines.json",
                "experiment.json",
                "folds.json",
            ],
        )
        validate_snapshot.assert_called_once_with(
            snapshot_dir=self.snapshot_dir,
            manifest_path=self.snapshot_manifest_path,
            expected_manifest_sha256=self.control_hashes["e5_snapshot.json"],
            expected_model_id=fold_processing_aws.E5_MODEL_ID,
            expected_revision=fold_processing_aws.E5_REVISION,
            expected_tree_sha256=fold_processing_aws.E5_SNAPSHOT_TREE_SHA256,
        )
        validate_fixed.assert_called_once()

    def test_snapshot_validator_and_staging_inventory_must_agree(self) -> None:
        changed_files = tuple(
            (path, size, "f" * 64)
            for path, size, _ in self._snapshot_identity().files
        )
        with self._patched_validators(
            snapshot=self._snapshot_identity(files=changed_files)
        ):
            with self.assertRaisesRegex(RuntimeError, "validator and staged inventory"):
                fold_processing_aws._validate_static_sources(
                    e5_snapshot_dir=self.snapshot_dir,
                    e5_snapshot_manifest_path=self.snapshot_manifest_path,
                    e5_pack_dir=self.pack_dir,
                    fixed_base_dir=self.fixed_dir,
                )

    def test_pack_extra_file_and_manifest_byte_drift_fail_loudly(self) -> None:
        extra = self.pack_dir / "extra.json"
        extra.write_bytes(b"{}\n")
        with self._patched_validators():
            with self.assertRaisesRegex(ValueError, "file inventory changed"):
                fold_processing_aws._validate_static_sources(
                    e5_snapshot_dir=self.snapshot_dir,
                    e5_snapshot_manifest_path=self.snapshot_manifest_path,
                    e5_pack_dir=self.pack_dir,
                    fixed_base_dir=self.fixed_dir,
                )
        extra.unlink()

        (self.pack_dir / "manifest.json").write_bytes(b'{"changed":true}\n')
        with self._patched_validators():
            with self.assertRaisesRegex(ValueError, "manifest hash changed"):
                fold_processing_aws._validate_static_sources(
                    e5_snapshot_dir=self.snapshot_dir,
                    e5_snapshot_manifest_path=self.snapshot_manifest_path,
                    e5_pack_dir=self.pack_dir,
                    fixed_base_dir=self.fixed_dir,
                )

    def test_fixed_scientific_identity_drift_is_rejected(self) -> None:
        fixed = self._fixed_identity()
        fixed.model_sha256 = "0" * 64
        with self._patched_validators(fixed=fixed):
            with self.assertRaisesRegex(ValueError, "scientific identity changed"):
                fold_processing_aws._validate_static_sources(
                    e5_snapshot_dir=self.snapshot_dir,
                    e5_snapshot_manifest_path=self.snapshot_manifest_path,
                    e5_pack_dir=self.pack_dir,
                    fixed_base_dir=self.fixed_dir,
                )

    def test_directory_inventory_rejects_empty_symlink_hardlink_and_fifo(self) -> None:
        variants: list[tuple[str, callable]] = []

        def empty(root: Path) -> None:
            (root / "empty").write_bytes(b"")

        def symlink(root: Path) -> None:
            target = root / "target"
            target.write_bytes(b"target\n")
            (root / "link").symlink_to(target)

        def hardlink(root: Path) -> None:
            target = root / "target"
            target.write_bytes(b"target\n")
            os.link(target, root / "alias")

        def fifo(root: Path) -> None:
            os.mkfifo(root / "pipe")

        variants.extend(
            (
                ("empty", empty),
                ("symlink", symlink),
                ("hardlink", hardlink),
                ("fifo", fifo),
            )
        )
        for label, build in variants:
            with self.subTest(label=label):
                root = self.root / f"invalid-{label}"
                root.mkdir()
                build(root)
                with self.assertRaises(ValueError):
                    fold_processing_aws._directory_inventory(root, name=label)

    def test_path_replacement_during_hash_is_rejected(self) -> None:
        path = self.root / "race.txt"
        replacement = self.root / "replacement.txt"
        displaced = self.root / "displaced.txt"
        path.write_bytes(b"original bytes\n")
        replacement.write_bytes(b"replacement!!\n")
        real_read = fold_processing_aws.os.read
        swapped = False

        def read_then_swap(descriptor: int, count: int) -> bytes:
            nonlocal swapped
            chunk = real_read(descriptor, count)
            if chunk and not swapped:
                swapped = True
                path.rename(displaced)
                replacement.rename(path)
            return chunk

        with patch.object(fold_processing_aws.os, "read", side_effect=read_then_swap):
            with self.assertRaisesRegex(RuntimeError, "changed while hashed"):
                fold_processing_aws._sha256_file(path)
        self.assertTrue(swapped)


class _ManifestS3:
    def __init__(self, events: list[tuple[str, str]], *, corrupt_head: bool = False):
        self.events = events
        self.corrupt_head = corrupt_head
        self.payload_by_version: dict[str, tuple[str, bytes, str, str]] = {}
        self.versions: list[dict[str, object]] = []
        self.delete_markers: list[dict[str, object]] = []
        self.extra_versions: list[dict[str, object]] = []

    def record_version(
        self,
        *,
        key: str,
        version_id: str,
        size: int,
        etag: str,
    ) -> None:
        for record in self.versions:
            if record["Key"] == key:
                record["IsLatest"] = False
        self.versions.append(
            {
                "ChecksumAlgorithm": ["SHA256"],
                "ChecksumType": "FULL_OBJECT",
                "ETag": etag,
                "IsLatest": True,
                "Key": key,
                "Size": size,
                "StorageClass": "STANDARD",
                "VersionId": version_id,
            }
        )

    def put_object(self, **request: object) -> dict[str, object]:
        key = request["Key"]
        body = request["Body"]
        if type(key) is not str or type(body) is not bytes:
            raise AssertionError("Synthetic manifest publication changed shape")
        version = f"manifest-version-{len(self.payload_by_version)}"
        digest = hashlib.sha256(body).digest()
        checksum = __import__("base64").b64encode(digest).decode("ascii")
        etag = f'"{hashlib.md5(body, usedforsecurity=False).hexdigest()}"'
        self.payload_by_version[version] = (key, body, checksum, etag)
        self.record_version(
            key=key,
            version_id=version,
            size=len(body),
            etag=etag,
        )
        self.events.append(("put-manifest", key))
        return {
            "ChecksumSHA256": checksum,
            "ETag": etag,
            "ServerSideEncryption": "AES256",
            "VersionId": version,
        }

    def head_object(self, **request: object) -> dict[str, object]:
        key, body, checksum, etag = self.payload_by_version[request["VersionId"]]
        if request["Key"] != key:
            raise AssertionError("Manifest head selected another key")
        return {
            "ChecksumSHA256": ("A" * 43 + "=" if self.corrupt_head else checksum),
            "ContentLength": len(body),
            "ContentType": "application/json",
            "ETag": etag,
            "Metadata": {"sha256": hashlib.sha256(body).hexdigest()},
            "ServerSideEncryption": "AES256",
            "VersionId": request["VersionId"],
        }

    def get_object(self, **request: object) -> dict[str, object]:
        key, body, _, _ = self.payload_by_version[request["VersionId"]]
        if request["Key"] != key:
            raise AssertionError("Manifest get selected another key")
        return {"Body": io.BytesIO(body)}

    def list_object_versions(self, **request: object) -> dict[str, object]:
        prefix = request["Prefix"]
        return {
            "DeleteMarkers": [
                copy.deepcopy(record)
                for record in self.delete_markers
                if record["Key"].startswith(prefix)
            ],
            "IsTruncated": False,
            "MaxKeys": request["MaxKeys"],
            "Name": request["Bucket"],
            "Prefix": prefix,
            "Versions": [
                copy.deepcopy(record)
                for record in [*self.versions, *self.extra_versions]
                if record["Key"].startswith(prefix)
            ],
        }


class StaticStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.events: list[tuple[str, str]] = []
        self.s3 = _ManifestS3(self.events)
        self.clients = aws.AwsClients(
            sts=Mock(),
            iam=Mock(),
            ecr=Mock(),
            s3=self.s3,
            service_quotas=Mock(),
            ec2=Mock(),
            sagemaker=Mock(),
            logs=Mock(),
        )
        self.completed = {
            "training_plan": {
                "infrastructure": {
                    "account_id": "371087393859",
                    "artifact_bucket": "ir-sagemaker",
                    "region": "us-east-1",
                }
            },
            "training_plan_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
        }
        self.destination_prefix = "arr-retrieval-cv/fold-eval/static-fixture/"
        self.state_dir = self.root / "static-state"
        self.sources = self._sources()
        self.static_inventory_expectations = _static_inventory_expectations(
            self.sources
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _sources(self) -> list[dict[str, object]]:
        identities = {
            "e5-snapshot": {
                "manifest_sha256": fold_processing_aws.E5_SNAPSHOT_MANIFEST_SHA256,
                "model_id": fold_processing_aws.E5_MODEL_ID,
                "revision": fold_processing_aws.E5_REVISION,
                "tree_sha256": fold_processing_aws.E5_SNAPSHOT_TREE_SHA256,
            },
            "e5-pack": {
                "manifest_sha256": (
                    fold_processing_aws.EXPECTED_E5_PACK_MANIFEST_SHA256
                ),
                "packed_query_inventory_sha256": (
                    fold_processing_aws.EXPECTED_E5_PACK_INVENTORY_SHA256
                ),
            },
            "fixed-base": {
                "artifact_manifest_sha256": (
                    fold_processing_aws.EXPECTED_FIXED_BASE_ARTIFACT_MANIFEST_SHA256
                ),
                "model_sha256": fold_processing_aws.EXPECTED_FIXED_BASE_MODEL_SHA256,
                "new_embedding_rows_sha256": (
                    fold_processing_aws.EXPECTED_FIXED_BASE_NEW_ROWS_SHA256
                ),
                "state_key_sha256": (
                    fold_processing_aws.EXPECTED_FIXED_BASE_STATE_KEYS_SHA256
                ),
            },
            "control": {
                "baseline_config_sha256": (
                    fold_processing_aws.EXPECTED_BASELINE_CONFIG_SHA256
                ),
                "e5_snapshot_manifest_sha256": (
                    fold_processing_aws.E5_SNAPSHOT_MANIFEST_SHA256
                ),
                "experiment_config_sha256": (
                    fold_processing_aws.EXPECTED_EXPERIMENT_CONFIG_SHA256
                ),
                "fold_manifest_sha256": (
                    fold_processing_aws.EXPECTED_FOLD_MANIFEST_SHA256
                ),
            },
        }
        assets: list[dict[str, object]] = []
        for asset_index, name in enumerate(
            ("e5-snapshot", "e5-pack", "fixed-base", "control")
        ):
            root = self.root / name
            root.mkdir()
            files: list[dict[str, object]] = []
            relatives = (
                (
                    "e5_snapshot.json",
                    "evaluation_baselines.json",
                    "experiment.json",
                    "folds.json",
                )
                if name == "control"
                else ("a.bin", "nested/b.bin")
            )
            for file_index, relative in enumerate(relatives):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    f"{name}:{asset_index}:{file_index}\n".encode("ascii")
                )
                files.append(_file_record(path, relative_to=root))
            assets.append(
                {
                    "name": name,
                    "root": root,
                    "source_identity": identities[name],
                    "files": files,
                }
            )
        return assets

    @contextmanager
    def _runtime(self, *, stage_side_effect=None, unused_side_effect=None):
        staged_count = 0

        def stage_file_once(_s3: object, **request: object) -> dict[str, object]:
            nonlocal staged_count
            if stage_side_effect is not None:
                return stage_side_effect(_s3, **request)
            source_path = request["source_path"]
            key = request["key"]
            payload = source_path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            etag = f'"{staged_count:032x}"'
            staged_count += 1
            self.events.append(("stage-file", key))
            record = {
                "bucket": request["bucket"],
                "etag": etag,
                "key": key,
                "schema_version": 1,
                "sha256": digest,
                "size": len(payload),
                "sse": "AES256",
                "version_id": f"static-version-{staged_count}",
            }
            self.s3.record_version(
                key=record["key"],
                version_id=record["version_id"],
                size=record["size"],
                etag=record["etag"],
            )
            return record

        with (
            patch.object(
                fold_processing_aws.aws,
                "validate_aws_sdk_versions",
                return_value=copy.deepcopy(aws.EXPECTED_AWS_SDK_VERSIONS),
            ),
            patch.object(
                fold_processing_aws.controlled_supervisor,
                "validate_completed_fold_evidence",
                side_effect=lambda value: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "_validate_static_sources",
                return_value=copy.deepcopy(self.sources),
            ),
            patch.object(
                fold_processing_aws,
                "EXPECTED_STATIC_INVENTORIES",
                copy.deepcopy(self.static_inventory_expectations),
            ),
            patch.object(fold_processing_aws.aws, "validate_artifact_bucket") as bucket,
            patch.object(
                fold_processing_aws.aws,
                "assert_unused_versioned_prefix",
                side_effect=unused_side_effect,
            ) as unused,
            patch.object(
                fold_processing_aws.aws,
                "stage_file_once",
                side_effect=stage_file_once,
            ) as stage,
        ):
            yield bucket, unused, stage

    def _stage(self, *, state_dir: Path | None = None) -> dict[str, object]:
        return fold_processing_aws.stage_static_evaluation_inputs_once(
            self.clients,
            completed_fold_evidence=self.completed,
            e5_snapshot_dir=self.root / "not-read-snapshot",
            e5_snapshot_manifest_path=self.root / "not-read-manifest.json",
            e5_pack_dir=self.root / "not-read-pack",
            fixed_base_dir=self.root / "not-read-fixed",
            destination_prefix=self.destination_prefix,
            state_dir=state_dir or self.state_dir,
        )

    def test_stages_all_files_then_publishes_manifest_last(self) -> None:
        with self._runtime() as (bucket, unused, stage):
            receipt = self._stage()
            validated = fold_processing_aws.validate_static_evaluation_staging_receipt(
                receipt,
                completed_fold_evidence=self.completed,
            )

        self.assertEqual(receipt, validated)
        self.assertEqual(stage.call_count, 10)
        self.assertEqual([event[0] for event in self.events], ["stage-file"] * 10 + ["put-manifest"])
        self.assertTrue(self.events[-1][1].endswith(fold_processing_aws.STATIC_MANIFEST_NAME))
        bucket.assert_called_once_with(
            self.s3,
            bucket="ir-sagemaker",
            region="us-east-1",
        )
        unused.assert_called_once_with(
            self.s3,
            bucket="ir-sagemaker",
            prefix=self.destination_prefix,
            expected_bucket_owner="371087393859",
        )
        self.assertEqual(
            sorted(path.name for path in self.state_dir.iterdir()),
            [
                "intent.json",
                "manifest-intent.json",
                *(f"object-{index:03d}.json" for index in range(10)),
                "receipt.json",
                "state.json",
            ],
        )

    def test_partial_staging_is_not_retried_or_resumed(self) -> None:
        calls = 0

        def fail_second(_s3: object, **request: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("synthetic lost PutObject response")
            source = request["source_path"].read_bytes()
            return {
                "bucket": request["bucket"],
                "etag": '"00000000000000000000000000000001"',
                "key": request["key"],
                "schema_version": 1,
                "sha256": hashlib.sha256(source).hexdigest(),
                "size": len(source),
                "sse": "AES256",
                "version_id": "partial-version",
            }

        with self._runtime(stage_side_effect=fail_second):
            with self.assertRaisesRegex(TimeoutError, "lost PutObject"):
                self._stage()
            with self.assertRaisesRegex(FileExistsError, "state"):
                self._stage()
        self.assertEqual(calls, 2)
        self.assertTrue((self.state_dir / "intent.json").is_file())
        self.assertTrue((self.state_dir / "object-000.json").is_file())
        self.assertFalse((self.state_dir / "receipt.json").exists())
        self.assertFalse(self.s3.payload_by_version)

    def test_sdk_drift_stops_before_source_or_aws_writes(self) -> None:
        with (
            patch.object(
                fold_processing_aws.aws,
                "validate_aws_sdk_versions",
                side_effect=RuntimeError("synthetic SDK drift"),
            ),
            patch.object(fold_processing_aws, "_validate_static_sources") as sources,
        ):
            with self.assertRaisesRegex(RuntimeError, "SDK drift"):
                self._stage()
        sources.assert_not_called()
        self.assertFalse(self.state_dir.exists())
        self.assertFalse(self.events)

    def test_wrong_staged_hash_taints_state_without_manifest(self) -> None:
        def wrong_hash(_s3: object, **request: object) -> dict[str, object]:
            source = request["source_path"].read_bytes()
            return {
                "bucket": request["bucket"],
                "etag": '"00000000000000000000000000000001"',
                "key": request["key"],
                "schema_version": 1,
                "sha256": "0" * 64,
                "size": len(source),
                "sse": "AES256",
                "version_id": "wrong-hash-version",
            }

        with self._runtime(stage_side_effect=wrong_hash):
            with self.assertRaisesRegex(RuntimeError, "differs from its source"):
                self._stage()
        self.assertTrue((self.state_dir / "intent.json").is_file())
        self.assertFalse((self.state_dir / "object-000.json").exists())
        self.assertFalse(self.s3.payload_by_version)

    def test_manifest_head_mismatch_leaves_no_success_receipt(self) -> None:
        self.s3 = _ManifestS3(self.events, corrupt_head=True)
        self.clients = copy.copy(self.clients)
        object.__setattr__(self.clients, "s3", self.s3)
        with self._runtime():
            with self.assertRaisesRegex(RuntimeError, "metadata changed"):
                self._stage()
        self.assertEqual([event[0] for event in self.events], ["stage-file"] * 10 + ["put-manifest"])
        self.assertTrue((self.state_dir / "manifest-intent.json").is_file())
        self.assertFalse((self.state_dir / "receipt.json").exists())

    def test_final_prefix_audit_rejects_extra_versions_and_delete_markers(self) -> None:
        for mutation in ("extra-version", "delete-marker"):
            with self.subTest(mutation=mutation):
                self.events = []
                self.s3 = _ManifestS3(self.events)
                self.clients = copy.copy(self.clients)
                object.__setattr__(self.clients, "s3", self.s3)
                self.state_dir = self.root / f"static-state-{mutation}"
                if mutation == "extra-version":
                    self.s3.extra_versions = [
                        {
                            "ChecksumAlgorithm": ["SHA256"],
                            "ChecksumType": "FULL_OBJECT",
                            "ETag": '"ffffffffffffffffffffffffffffffff"',
                            "IsLatest": True,
                            "Key": self.destination_prefix + "unexpected.bin",
                            "Size": 1,
                            "StorageClass": "STANDARD",
                            "VersionId": "unexpected-version",
                        }
                    ]
                    message = "version count changed"
                else:
                    self.s3.delete_markers = [
                        {
                            "IsLatest": True,
                            "Key": self.destination_prefix + "deleted.bin",
                            "LastModified": _COPY_TIME,
                            "Owner": {
                                "DisplayName": "fixture",
                                "ID": "f" * 64,
                            },
                            "VersionId": "delete-version",
                        }
                    ]
                    message = "delete marker"
                with self._runtime():
                    with self.assertRaisesRegex(RuntimeError, message):
                        self._stage()
                self.assertTrue(
                    (self.state_dir / "manifest-intent.json").is_file()
                )
                self.assertFalse((self.state_dir / "receipt.json").exists())

    def test_resealed_receipt_splices_and_aliases_are_rejected(self) -> None:
        with self._runtime():
            receipt = self._stage()
            mutations = []
            changed = copy.deepcopy(receipt)
            changed["assets"][0]["source_identity"]["tree_sha256"] = "0" * 64
            mutations.append(changed)
            changed = copy.deepcopy(receipt)
            changed["assets"][0]["files"][0]["version_id"] = changed["assets"][1][
                "files"
            ][0]["version_id"]
            changed["assets"][0]["files"][0]["bucket"] = changed["assets"][1][
                "files"
            ][0]["bucket"]
            changed["assets"][0]["files"][0]["key"] = changed["assets"][1][
                "files"
            ][0]["key"]
            mutations.append(changed)
            for changed in mutations:
                changed = _reseal(changed)
                with self.assertRaises(ValueError):
                    fold_processing_aws.validate_static_evaluation_staging_receipt(
                        changed,
                        completed_fold_evidence=self.completed,
                    )


_KMS_KEY = (
    "arn:aws:kms:us-east-1:371087393859:"
    "key/d83fe50a-4d76-45cf-8b6c-bb45d04fddda"
)
_COPY_TIME = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)


class _FoldCopyS3(_ManifestS3):
    def __init__(
        self,
        events: list[tuple[str, str]],
        *,
        page_size: int = 1000,
        fail_copy_index: int | None = None,
        fail_after_remote_write: bool = False,
        corrupt_copy_response: str | None = None,
        corrupt_copy_head: str | None = None,
        inject_extra_after_copy: int | None = None,
    ) -> None:
        super().__init__(events)
        self.page_size = page_size
        self.fail_copy_index = fail_copy_index
        self.fail_after_remote_write = fail_after_remote_write
        self.corrupt_copy_response = corrupt_copy_response
        self.corrupt_copy_head = corrupt_copy_head
        self.inject_extra_after_copy = inject_extra_after_copy
        self.copy_requests: list[dict[str, object]] = []
        self.versions: list[dict[str, object]] = []
        self.delete_markers: list[dict[str, object]] = []
        self.copy_metadata: dict[str, dict[str, object]] = {}

    def _add_destination(
        self,
        *,
        key: str,
        size: int,
        etag: str,
        version_id: str,
        checksum: str,
        content_type: str,
        encryption: str,
        kms_key: str | None,
        metadata: dict[str, str],
    ) -> None:
        for record in self.versions:
            if record["Key"] == key:
                record["IsLatest"] = False
        self.versions.append(
            {
                "ChecksumAlgorithm": ["SHA256"],
                "ChecksumType": "FULL_OBJECT",
                "ETag": etag,
                "IsLatest": True,
                "Key": key,
                "Size": size,
                "StorageClass": "STANDARD",
                "VersionId": version_id,
            }
        )
        self.copy_metadata[version_id] = {
            "BucketKeyEnabled": encryption == "aws:kms",
            "ChecksumSHA256": checksum,
            "ChecksumType": "FULL_OBJECT",
            "ContentLength": size,
            "ContentType": content_type,
            "ETag": etag,
            "LastModified": _COPY_TIME,
            "Metadata": metadata,
            "SSEKMSKeyId": kms_key,
            "ServerSideEncryption": encryption,
            "VersionId": version_id,
        }

    def copy_object(self, **request: object) -> dict[str, object]:
        index = len(self.copy_requests)
        self.copy_requests.append(copy.deepcopy(request))
        key = request["Key"]
        self.events.append(("copy", key))
        if self.fail_copy_index == index and not self.fail_after_remote_write:
            raise TimeoutError("synthetic CopyObject failure before response")
        raw_digest = hashlib.sha256(f"archive-{index}".encode("ascii")).digest()
        checksum = base64.b64encode(raw_digest).decode("ascii")
        etag = f'"{index + 100:032x}"'
        version = f"copy-version-{index}"
        self._add_destination(
            key=key,
            size=100,
            etag=etag,
            version_id=version,
            checksum=checksum,
            content_type="application/gzip",
            encryption="aws:kms",
            kms_key=request["SSEKMSKeyId"],
            metadata={},
        )
        if self.inject_extra_after_copy == index:
            self.versions.append(
                {
                    "ChecksumAlgorithm": ["SHA256"],
                    "ChecksumType": "FULL_OBJECT",
                    "ETag": '"ffffffffffffffffffffffffffffffff"',
                    "IsLatest": True,
                    "Key": key + ".unexpected",
                    "Size": 1,
                    "StorageClass": "STANDARD",
                    "VersionId": "unexpected-version",
                }
            )
        if self.fail_copy_index == index and self.fail_after_remote_write:
            raise TimeoutError("synthetic lost CopyObject response")
        result = {
            "ChecksumSHA256": checksum,
            "ChecksumType": "FULL_OBJECT",
            "ETag": etag,
            "LastModified": _COPY_TIME,
        }
        response: dict[str, object] = {
            "BucketKeyEnabled": True,
            "CopyObjectResult": result,
            "SSEKMSKeyId": request["SSEKMSKeyId"],
            "ServerSideEncryption": "aws:kms",
            "VersionId": version,
        }
        if self.corrupt_copy_response == "checksum":
            result["ChecksumSHA256"] = "not-base64"
        elif self.corrupt_copy_response == "checksum_type":
            result["ChecksumType"] = "COMPOSITE"
        elif self.corrupt_copy_response == "kms":
            response["SSEKMSKeyId"] = _KMS_KEY + "-changed"
        elif self.corrupt_copy_response == "version":
            response.pop("VersionId")
        return response

    def put_object(self, **request: object) -> dict[str, object]:
        return super().put_object(**request)

    def head_object(self, **request: object) -> dict[str, object]:
        version = request["VersionId"]
        if version.startswith("manifest-version-"):
            metadata = super().head_object(**request)
        else:
            metadata = copy.deepcopy(self.copy_metadata[version])
            self.events.append(("head-copy", request["Key"]))
        if self.corrupt_copy_head == "checksum" and not version.startswith(
            "manifest-version-"
        ):
            metadata["ChecksumSHA256"] = "A" * 43 + "="
        elif self.corrupt_copy_head == "kms" and not version.startswith(
            "manifest-version-"
        ):
            metadata["SSEKMSKeyId"] = _KMS_KEY + "-changed"
        elif self.corrupt_copy_head == "size" and not version.startswith(
            "manifest-version-"
        ):
            metadata["ContentLength"] += 1
        return metadata

    def list_object_versions(self, **request: object) -> dict[str, object]:
        prefix = request["Prefix"]
        records: list[tuple[str, str, str, dict[str, object]]] = []
        records.extend(
            (record["Key"], record["VersionId"], "version", record)
            for record in self.versions
            if record["Key"].startswith(prefix)
        )
        records.extend(
            (record["Key"], record["VersionId"], "delete", record)
            for record in self.delete_markers
            if record["Key"].startswith(prefix)
        )
        records.sort(key=lambda item: (item[0], item[1], item[2]))
        marker = (request.get("KeyMarker"), request.get("VersionIdMarker"))
        start = 0
        if marker[0] is not None:
            for position, row in enumerate(records):
                if (row[0], row[1]) == marker:
                    start = position + 1
                    break
            else:
                raise AssertionError("Synthetic pagination marker was not found")
        limit = min(request["MaxKeys"], self.page_size)
        page = records[start : start + limit]
        truncated = start + len(page) < len(records)
        response: dict[str, object] = {
            "DeleteMarkers": [row[3] for row in page if row[2] == "delete"],
            "IsTruncated": truncated,
            "MaxKeys": request["MaxKeys"],
            "Name": request["Bucket"],
            "Prefix": prefix,
            "Versions": [row[3] for row in page if row[2] == "version"],
        }
        if truncated:
            response["NextKeyMarker"] = page[-1][0]
            response["NextVersionIdMarker"] = page[-1][1]
        self.events.append(("list", f"{prefix}:{start}"))
        return response


class FoldArchiveCopyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.events: list[tuple[str, str]] = []
        self.s3 = _FoldCopyS3(self.events)
        self.clients = self._clients(self.s3)
        self.destination_prefix = "arr-retrieval-cv/fold-eval/f0-copy-fixture/"
        self.state_dir = self.root / "copy-state"
        self.completed = self._completed_evidence()
        self.remote_by_run_id = {
            system["run_id"]: self._remote(system["run_id"], ordinal)
            for ordinal, system in enumerate(self.completed["systems"])
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _clients(s3: object) -> aws.AwsClients:
        return aws.AwsClients(
            sts=Mock(),
            iam=Mock(),
            ecr=Mock(),
            s3=s3,
            service_quotas=Mock(),
            ec2=Mock(),
            sagemaker=Mock(),
            logs=Mock(),
        )

    @staticmethod
    def _completed_evidence() -> dict[str, object]:
        systems: list[dict[str, object]] = []
        controlled_runs: list[dict[str, object]] = []
        for ordinal, cell in enumerate(controlled_supervisor._completed_fold_order(0)):
            view = {"flat_masked": "flat", "structured": "struct"}[
                cell["query_view"]
            ]
            sampler = {"local_unique": "local", "global_uniform": "global"}[
                cell["sampler"]
            ]
            run_id = (
                f"controlled-f0-{view}-{sampler}-s{cell['experiment_seed']}"
            )
            systems.append(
                {
                    "ordinal": ordinal,
                    "cell": copy.deepcopy(cell),
                    "run_id": run_id,
                    "job_name": "arr-ret-cv1-" + run_id.removeprefix("controlled-") + "-a3",
                    "preflight_receipt": {"run_id": run_id},
                    "terminal_receipt": {
                        "run_id": run_id,
                        "model_artifact_s3_uri": (
                            "s3://ir-sagemaker/arr-retrieval-cv/outputs/"
                            f"{run_id}/job/output/model.tar.gz"
                        ),
                    },
                    "terminal_receipt_sha256": hashlib.sha256(
                        f"terminal-{run_id}".encode("ascii")
                    ).hexdigest(),
                    "request_receipt_sha256": hashlib.sha256(
                        f"request-{run_id}".encode("ascii")
                    ).hexdigest(),
                }
            )
            controlled_runs.append(
                {
                    "run_id": run_id,
                    "input_channels": {
                        "data": {
                            "s3_uri": (
                                "s3://ir-sagemaker/arr-retrieval-cv/inputs/"
                                "dataset-cce04197/"
                            )
                        }
                    },
                }
            )
        return {
            "outer_fold": 0,
            "attempt_id": "a3",
            "training_plan": {
                "infrastructure": {
                    "account_id": "371087393859",
                    "artifact_bucket": "ir-sagemaker",
                    "processing_instance_count": 1,
                    "processing_instance_type": "ml.g5.12xlarge",
                    "processing_max_runtime_seconds": 3_600,
                    "processing_volume_size_gb": 100,
                    "region": "us-east-1",
                    "role_arn": (
                        "arn:aws:iam::371087393859:"
                        "role/AmazonSageMakerExecutionRole"
                    ),
                },
                "controlled_runs": controlled_runs,
            },
            "training_plan_sha256": "a" * 64,
            "training_staging_receipt": {"fixture": "training-staging"},
            "training_staging_receipt_sha256": "b" * 64,
            "source_bundle": {
                "name": "source-" + "c" * 64 + ".tar.gz",
                "size": 400_089,
                "sha256": "c" * 64,
                "inventory_sha256": "d" * 64,
                "commit_epoch": 1_783_917_519,
            },
            "systems": systems,
            "receipt_sha256": "e" * 64,
        }

    @staticmethod
    def _remote(run_id: str, ordinal: int) -> dict[str, object]:
        return {
            "bucket": "ir-sagemaker",
            "key": f"arr-retrieval-cv/outputs/{run_id}/job/output/model.tar.gz",
            "version_id": f"source-version-{ordinal}",
            "size": 100,
            "etag": f'"{ordinal + 1:032x}-2"',
            "checksum": {
                "algorithm": "CRC32",
                "type": "COMPOSITE",
                "value": "AAAAAA==-2",
            },
            "encryption": {
                "algorithm": "aws:kms",
                "kms_key_id": _KMS_KEY,
                "bucket_key_enabled": True,
            },
        }

    @contextmanager
    def _runtime(self, *, inspect_side_effect=None):
        def inspect(_s3: object, **request: object) -> dict[str, object]:
            run_id = request["preflight"]["run_id"]
            if inspect_side_effect is not None:
                return inspect_side_effect(run_id)
            return copy.deepcopy(self.remote_by_run_id[run_id])

        def coordinates(*, plan: object, preflight: object, terminal: object):
            del plan, terminal
            remote = self.remote_by_run_id[preflight["run_id"]]
            prefix = remote["key"].removesuffix("model.tar.gz")
            return (
                remote["bucket"],
                remote["key"],
                prefix,
                f"s3://{remote['bucket']}/{remote['key']}",
            )

        with (
            patch.object(
                fold_processing_aws.aws,
                "validate_aws_sdk_versions",
                return_value=copy.deepcopy(aws.EXPECTED_AWS_SDK_VERSIONS),
            ),
            patch.object(
                fold_processing_aws.controlled_supervisor,
                "validate_completed_fold_evidence",
                side_effect=lambda value: copy.deepcopy(value),
            ),
            patch.object(fold_processing_aws.aws, "validate_artifact_bucket"),
            patch.object(
                fold_processing_aws.training_artifacts,
                "_inspect_remote_output",
                side_effect=inspect,
            ) as inspect_remote,
            patch.object(
                fold_processing_aws.training_artifacts,
                "_expected_remote_coordinates",
                side_effect=coordinates,
            ),
        ):
            yield inspect_remote

    def _copy(self, *, state_dir: Path | None = None) -> dict[str, object]:
        return fold_processing_aws.copy_completed_fold_archives_once(
            self.clients,
            completed_fold_evidence=self.completed,
            destination_prefix=self.destination_prefix,
            state_dir=state_dir or self.state_dir,
        )

    def test_exact_version_kms_sha256_copies_and_manifest_last(self) -> None:
        with self._runtime() as inspect:
            receipt = self._copy()
            validated = fold_processing_aws.validate_fold_archive_copy_receipt(
                receipt,
                completed_fold_evidence=self.completed,
            )

        self.assertEqual(receipt, validated)
        self.assertEqual(inspect.call_count, 24)
        self.assertEqual(len(self.s3.copy_requests), 12)
        for ordinal, (request, system) in enumerate(
            zip(self.s3.copy_requests, self.completed["systems"])
        ):
            remote = self.remote_by_run_id[system["run_id"]]
            self.assertEqual(
                request["CopySource"],
                {
                    "Bucket": remote["bucket"],
                    "Key": remote["key"],
                    "VersionId": remote["version_id"],
                },
            )
            self.assertEqual(request["ChecksumAlgorithm"], "SHA256")
            self.assertEqual(request["ServerSideEncryption"], "aws:kms")
            self.assertEqual(request["SSEKMSKeyId"], _KMS_KEY)
            self.assertIs(request["BucketKeyEnabled"], True)
            self.assertEqual(request["ExpectedBucketOwner"], "371087393859")
            self.assertEqual(request["ExpectedSourceBucketOwner"], "371087393859")
            self.assertTrue(
                request["Key"].endswith(
                    f"{ordinal:02d}-"
                    f"{system['cell']['query_view']}_{system['cell']['sampler']}_"
                    f"seed{system['cell']['experiment_seed']}.model.tar.gz"
                )
            )
        writes = [event[0] for event in self.events if event[0] in {"copy", "put-manifest"}]
        self.assertEqual(writes, ["copy"] * 12 + ["put-manifest"])
        self.assertEqual(
            receipt["fold_archive_input_manifest"]["copy_set_receipt_sha256"],
            fold_processing_aws._document_sha256(receipt["copy_set_receipt"]),
        )
        self.assertTrue((self.state_dir / "receipt.json").is_file())

    def test_every_copy_has_complete_history_audit_before_and_after(self) -> None:
        with self._runtime():
            self._copy()
        for position, (kind, key) in enumerate(self.events):
            if kind != "copy":
                continue
            previous_list = max(
                index
                for index in range(position)
                if self.events[index][0] == "list"
            )
            head = next(
                index
                for index in range(position + 1, len(self.events))
                if self.events[index] == ("head-copy", key)
            )
            following_list = next(
                index
                for index in range(head + 1, len(self.events))
                if self.events[index][0] == "list"
            )
            self.assertLess(previous_list, position)
            self.assertLess(position, head)
            self.assertLess(head, following_list)

    def test_history_pagination_and_delete_or_extra_rejection(self) -> None:
        self.s3.page_size = 1
        expected = []
        for index in range(3):
            checksum = base64.b64encode(hashlib.sha256(str(index).encode()).digest()).decode()
            key = self.destination_prefix + f"{index}.gz"
            version = f"v{index}"
            etag = f'"{index + 1:032x}"'
            self.s3._add_destination(
                key=key,
                size=10,
                etag=etag,
                version_id=version,
                checksum=checksum,
                content_type="application/gzip",
                encryption="aws:kms",
                kms_key=_KMS_KEY,
                metadata={},
            )
            expected.append(
                {"key": key, "version_id": version, "size": 10, "etag": etag}
            )
        history = fold_processing_aws._list_prefix_history(
            self.s3,
            bucket="ir-sagemaker",
            prefix=self.destination_prefix,
            expected_bucket_owner="371087393859",
        )
        self.assertEqual(len(history["versions"]), 3)
        fold_processing_aws._require_exact_copy_history(
            history,
            expected_objects=expected,
        )
        changed = copy.deepcopy(history)
        changed["delete_markers"] = [
            {"Key": self.destination_prefix + "deleted", "VersionId": "delete-v"}
        ]
        with self.assertRaisesRegex(RuntimeError, "delete marker"):
            fold_processing_aws._require_exact_copy_history(
                changed,
                expected_objects=expected,
            )
        with self.assertRaisesRegex(RuntimeError, "version count"):
            fold_processing_aws._require_exact_copy_history(
                history,
                expected_objects=expected[:-1],
            )

    def test_copy_response_and_head_mismatches_fail_loudly(self) -> None:
        source = self.remote_by_run_id[self.completed["systems"][0]["run_id"]]
        for location, mutation in (
            ("response", "checksum"),
            ("response", "checksum_type"),
            ("response", "kms"),
            ("response", "version"),
            ("head", "checksum"),
            ("head", "kms"),
            ("head", "size"),
        ):
            with self.subTest(location=location, mutation=mutation):
                s3 = _FoldCopyS3(
                    [],
                    corrupt_copy_response=(mutation if location == "response" else None),
                    corrupt_copy_head=(mutation if location == "head" else None),
                )
                with self.assertRaises((RuntimeError, ValueError)):
                    fold_processing_aws._copy_archive_object_once(
                        s3,
                        source=source,
                        destination_bucket="ir-sagemaker",
                        destination_key=self.destination_prefix + "archive.tar.gz",
                        expected_bucket_owner="371087393859",
                    )

    def test_source_mutation_before_copy_taints_state_and_stops(self) -> None:
        counts: dict[str, int] = {}
        target = self.completed["systems"][2]["run_id"]

        def mutate_third(run_id: str) -> dict[str, object]:
            counts[run_id] = counts.get(run_id, 0) + 1
            remote = copy.deepcopy(self.remote_by_run_id[run_id])
            if run_id == target and counts[run_id] == 2:
                remote["version_id"] = "changed-source-version"
            return remote

        with self._runtime(inspect_side_effect=mutate_third):
            with self.assertRaisesRegex(RuntimeError, "changed before CopyObject"):
                self._copy()
        self.assertEqual(len(self.s3.copy_requests), 2)
        self.assertTrue((self.state_dir / "intent.json").is_file())
        self.assertTrue((self.state_dir / "copy-01-receipt.json").is_file())
        self.assertFalse((self.state_dir / "copy-02-intent.json").exists())

    def test_lost_copy_response_permanently_taints_prefix_without_retry(self) -> None:
        self.s3.fail_copy_index = 1
        self.s3.fail_after_remote_write = True
        with self._runtime():
            with self.assertRaisesRegex(TimeoutError, "lost CopyObject"):
                self._copy()
            with self.assertRaises(FileExistsError):
                self._copy(state_dir=self.root / "second-state")
        self.assertEqual(len(self.s3.copy_requests), 2)
        self.assertEqual(len(self.s3.versions), 2)
        self.assertFalse((self.state_dir / "copy-01-receipt.json").exists())
        self.assertFalse((self.state_dir / "receipt.json").exists())

    def test_extra_version_after_copy_stops_before_next_copy(self) -> None:
        self.s3.inject_extra_after_copy = 0
        with self._runtime():
            with self.assertRaisesRegex(RuntimeError, "version count"):
                self._copy()
        self.assertEqual(len(self.s3.copy_requests), 1)
        self.assertFalse((self.state_dir / "copy-01-intent.json").exists())

    def test_fully_resealed_cross_cell_source_splice_is_rejected(self) -> None:
        with self._runtime():
            receipt = self._copy()
            changed = copy.deepcopy(receipt)
            systems = changed["copy_set_receipt"]["systems"]
            systems[0]["source_object"], systems[1]["source_object"] = (
                systems[1]["source_object"],
                systems[0]["source_object"],
            )
            changed["copy_set_receipt"] = _reseal(changed["copy_set_receipt"])
            changed["fold_archive_input_manifest"] = (
                fold_processing_aws._expected_fold_manifest(
                    completed=self.completed,
                    copy_set=changed["copy_set_receipt"],
                )
            )
            manifest_raw = fold_processing_aws._canonical_bytes(
                changed["fold_archive_input_manifest"]
            )
            changed["manifest_object"]["size"] = len(manifest_raw)
            changed["manifest_object"]["sha256"] = hashlib.sha256(
                manifest_raw
            ).hexdigest()
            changed = _reseal(changed)
            with self.assertRaises(ValueError):
                fold_processing_aws.validate_fold_archive_copy_receipt(
                    changed,
                    completed_fold_evidence=self.completed,
                )


class FoldInventoryPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.completed = FoldArchiveCopyTest._completed_evidence()
        self.archive = {
            "destination_prefix": "arr-retrieval-cv/fold-eval/f0-archives/",
            "copy_set_receipt": {
                "systems": [
                    {
                        "destination_object": {
                            "encryption": {
                                "algorithm": "aws:kms",
                                "kms_key_id": _KMS_KEY,
                                "bucket_key_enabled": True,
                            }
                        }
                    }
                    for _ in range(12)
                ]
            },
        }
        self.static = {
            "destination_prefix": "arr-retrieval-cv/fold-eval/static/",
            "assets": [
                {
                    "name": "control",
                    "s3_prefix": "arr-retrieval-cv/fold-eval/static/control/",
                    "files": [{} for _ in range(10)],
                }
            ]
        }
        self.raw_manifest = '{"schemaVersion":2,"fixture":"fold-overlay"}'
        self.image_digest = "sha256:" + hashlib.sha256(
            self.raw_manifest.encode("utf-8")
        ).hexdigest()
        repository = (
            "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval"
        )
        self.publication = {
            "content_tag": "fold-build-fixture",
            "identity": {
                "build_context_files_sha256": "1" * 64,
                "build_context_identity_sha256": (
                    fold_processing_aws.FOLD_OVERLAY_BUILD_IDENTITY
                ),
                "config_digest": fold_processing_aws.FOLD_OVERLAY_CONFIG_DIGEST,
                "image_digest": self.image_digest,
                "local_image_identity_sha256": "2" * 64,
                "manifest_media_type": aws.ECR_MEDIA_TYPE,
                "offline_smoke_sha256": (
                    fold_processing_aws.FOLD_OVERLAY_OFFLINE_SMOKE_SHA256
                ),
            },
            "manifest_digest": self.image_digest,
            "media_type": aws.ECR_MEDIA_TYPE,
            "protocol": fold_processing_aws.FOLD_OVERLAY_PUBLICATION_PROTOCOL,
            "raw_manifest_sha256": self.image_digest.removeprefix("sha256:"),
            "remote_digest_uri": f"{repository}@{self.image_digest}",
            "remote_tag_uri": f"{repository}:fold-build-fixture",
        }
        self.job_name = "arr-ret-cv1-f0-inventory-a3"
        self.output_prefix = "arr-retrieval-cv/fold-eval/f0-inventory-output/"
        self.clients = self._clients()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _clients(self) -> aws.AwsClients:
        sts = Mock()
        sts.get_caller_identity.return_value = {
            "Account": "371087393859",
            "Arn": "arn:aws:iam::371087393859:user/tester",
        }
        ecr = Mock()
        ecr.batch_get_image.return_value = {
            "failures": [],
            "images": [{"imageManifest": self.raw_manifest}],
        }
        quotas = Mock()
        quotas.get_service_quota.return_value = {"Quota": {"Value": 1.0}}
        ec2 = Mock()
        ec2.describe_instance_type_offerings.return_value = {
            "InstanceTypeOfferings": [{"InstanceType": "g5.12xlarge"}]
        }
        sagemaker = Mock()
        sagemaker.list_processing_jobs.return_value = {"ProcessingJobSummaries": []}
        return aws.AwsClients(
            sts=sts,
            iam=Mock(),
            ecr=ecr,
            s3=Mock(),
            service_quotas=quotas,
            ec2=ec2,
            sagemaker=sagemaker,
            logs=Mock(),
        )

    @contextmanager
    def _runtime(self, *, sdk_error=None, existing_job: bool = False):
        if existing_job:
            self.clients.sagemaker.list_processing_jobs.return_value = {
                "ProcessingJobSummaries": [
                    {"ProcessingJobName": self.job_name}
                ]
            }
        archive_verification = fold_processing_aws._seal(
            {
                "schema_version": 1,
                "protocol": fold_processing_aws.FOLD_ARCHIVE_COPY_PROTOCOL,
                "copy_receipt_sha256": fold_processing_aws._document_sha256(
                    self.archive
                ),
                "verified_source_versions": 12,
                "verified_destination_versions": 13,
                "verified": True,
            }
        )
        static_verification = fold_processing_aws._seal(
            {
                "schema_version": 1,
                "protocol": fold_processing_aws.STATIC_STAGING_PROTOCOL,
                "staging_receipt_sha256": fold_processing_aws._document_sha256(
                    self.static
                ),
                "verified_object_versions": 11,
                "verified": True,
            }
        )
        sdk = (
            RuntimeError("synthetic SDK drift")
            if sdk_error
            else copy.deepcopy(aws.EXPECTED_AWS_SDK_VERSIONS)
        )
        with (
            patch.object(
                fold_processing_aws,
                "FOLD_OVERLAY_IMAGE_DIGEST",
                self.image_digest,
            ),
            patch.object(
                fold_processing_aws.controlled_supervisor,
                "validate_completed_fold_evidence",
                side_effect=lambda value: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "validate_fold_archive_copy_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "validate_static_evaluation_staging_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws.aws,
                "validate_aws_sdk_versions",
                side_effect=(sdk if isinstance(sdk, BaseException) else None),
                return_value=(None if isinstance(sdk, BaseException) else sdk),
            ),
            patch.object(fold_processing_aws.aws, "_assert_role_trust") as role,
            patch.object(
                fold_processing_aws.training_aws,
                "verify_remote_training_staging",
            ) as training,
            patch.object(
                fold_processing_aws,
                "verify_remote_fold_archives",
                return_value=archive_verification,
            ) as archives,
            patch.object(
                fold_processing_aws,
                "verify_remote_static_evaluation_inputs",
                return_value=static_verification,
            ) as static,
            patch.object(
                fold_processing_aws.aws,
                "assert_unused_versioned_prefix",
            ) as unused,
        ):
            yield role, training, archives, static, unused

    def _preflight(
        self,
        *,
        output_prefix: str | None = None,
    ) -> dict[str, object]:
        return fold_processing_aws.preflight_fold_inventory(
            self.clients,
            completed_fold_evidence=self.completed,
            archive_copy_receipt=self.archive,
            static_staging_receipt=self.static,
            overlay_publication_receipt=self.publication,
            job_name=self.job_name,
            output_prefix=output_prefix or self.output_prefix,
        )

    def test_exact_three_channel_network_isolated_request(self) -> None:
        with self._runtime() as (role, training, archives, static, unused):
            receipt = self._preflight()
            validated = fold_processing_aws.validate_fold_inventory_preflight_receipt(
                receipt,
                completed_fold_evidence=self.completed,
                archive_copy_receipt=self.archive,
                static_staging_receipt=self.static,
                overlay_publication_receipt=self.publication,
            )

        self.assertEqual(receipt, validated)
        request = receipt["request"]
        self.assertEqual(request["AppSpecification"]["ImageUri"], self.publication["remote_digest_uri"])
        self.assertEqual(
            request["AppSpecification"]["ContainerArguments"],
            [
                "/opt/program/modernbert/processing_fold_eval/inventory_sm.py",
                "--archive-manifest",
                "/opt/ml/processing/input/fold-archives/fold_archive_input_manifest.json",
                "--dataset-dir",
                "/opt/ml/processing/input/dataset",
                "--fold-manifest",
                "/opt/ml/processing/input/control/folds.json",
                "--scratch-parent",
                "/opt/ml/processing/work",
                "--output-dir",
                "/opt/ml/processing/output/evidence",
            ],
        )
        self.assertEqual(
            [item["InputName"] for item in request["ProcessingInputs"]],
            ["fold-archives", "dataset", "control"],
        )
        self.assertEqual(
            [item["S3Input"]["LocalPath"] for item in request["ProcessingInputs"]],
            [
                "/opt/ml/processing/input/fold-archives",
                "/opt/ml/processing/input/dataset",
                "/opt/ml/processing/input/control",
            ],
        )
        for item in request["ProcessingInputs"]:
            self.assertEqual(item["S3Input"]["S3InputMode"], "File")
            self.assertEqual(item["S3Input"]["S3DataType"], "S3Prefix")
            self.assertEqual(
                item["S3Input"]["S3DataDistributionType"],
                "FullyReplicated",
            )
            self.assertEqual(item["S3Input"]["S3CompressionType"], "None")
        self.assertIs(request["NetworkConfig"]["EnableNetworkIsolation"], True)
        self.assertEqual(
            request["ProcessingResources"]["ClusterConfig"],
            {
                "InstanceCount": 1,
                "InstanceType": "ml.g5.12xlarge",
                "VolumeSizeInGB": 100,
            },
        )
        self.assertEqual(request["StoppingCondition"], {"MaxRuntimeInSeconds": 3600})
        self.assertEqual(request["ProcessingOutputConfig"]["KmsKeyId"], _KMS_KEY)
        role.assert_called_once()
        training.assert_called_once()
        archives.assert_called_once()
        static.assert_called_once()
        unused.assert_called_once()

    def test_existing_job_and_used_output_fail_before_receipt(self) -> None:
        with self._runtime(existing_job=True):
            with self.assertRaisesRegex(FileExistsError, "job name already exists"):
                self._preflight()

        self.clients.sagemaker.list_processing_jobs.return_value = {
            "ProcessingJobSummaries": []
        }
        with self._runtime() as (*_, unused):
            unused.side_effect = FileExistsError("synthetic used output prefix")
            with self.assertRaisesRegex(FileExistsError, "used output prefix"):
                self._preflight()

    def test_output_prefix_rejects_input_descendants_and_ancestors(self) -> None:
        cases = (
            (
                self.archive["destination_prefix"] + "nested-output/",
                "fold archives",
            ),
            ("arr-retrieval-cv/inputs/", "corrected dataset"),
        )
        for output_prefix, input_name in cases:
            with self.subTest(output_prefix=output_prefix):
                with self._runtime():
                    with self.assertRaisesRegex(ValueError, input_name):
                        self._preflight(output_prefix=output_prefix)
                self.clients.sts.get_caller_identity.assert_not_called()

    def test_sdk_drift_stops_before_remote_verification(self) -> None:
        with self._runtime(sdk_error=True) as (_, training, archives, static, unused):
            with self.assertRaisesRegex(RuntimeError, "SDK drift"):
                self._preflight()
        training.assert_not_called()
        archives.assert_not_called()
        static.assert_not_called()
        unused.assert_not_called()

    def test_processing_quota_must_be_one_exact_positive_integer(self) -> None:
        for value in (True, 1.5, float("nan"), float("inf"), "1", None):
            with self.subTest(value=value):
                self.clients.service_quotas.get_service_quota.return_value = {
                    "Quota": {"Value": value}
                }
                with self._runtime():
                    with self.assertRaisesRegex(ValueError, "exact integer"):
                        self._preflight()
        self.clients.service_quotas.get_service_quota.return_value = {
            "Quota": {"Value": 0.0}
        }
        with self._runtime():
            with self.assertRaisesRegex(RuntimeError, "below one"):
                self._preflight()
        for response in (None, {}, {"Quota": []}):
            with self.subTest(response=response):
                self.clients.service_quotas.get_service_quota.return_value = response
                with self._runtime():
                    with self.assertRaisesRegex(RuntimeError, "malformed"):
                        self._preflight()

    def test_resealed_request_mutation_is_rejected(self) -> None:
        with self._runtime():
            receipt = self._preflight()
            changed = copy.deepcopy(receipt)
            changed["request"]["NetworkConfig"]["EnableNetworkIsolation"] = False
            changed["request_sha256"] = fold_processing_aws._document_sha256(
                changed["request"]
            )
            changed = _reseal(changed)
            with self.assertRaisesRegex(ValueError, "exact re-rendering"):
                fold_processing_aws.validate_fold_inventory_preflight_receipt(
                    changed,
                    completed_fold_evidence=self.completed,
                    archive_copy_receipt=self.archive,
                    static_staging_receipt=self.static,
                    overlay_publication_receipt=self.publication,
                )

    def test_resealed_cross_receipt_verification_splices_are_rejected(self) -> None:
        with self._runtime():
            receipt = self._preflight()
            mutations: list[tuple[str, dict[str, object]]] = []
            for field, value in (
                ("copy_receipt_sha256", "f" * 64),
                ("verified_source_versions", 11),
                ("verified_destination_versions", 12),
            ):
                changed = copy.deepcopy(receipt)
                changed["archive_verification"][field] = value
                changed["archive_verification"] = _reseal(
                    changed["archive_verification"]
                )
                mutations.append(("archive verification", _reseal(changed)))
            for field, value in (
                ("staging_receipt_sha256", "f" * 64),
                ("verified_object_versions", 10),
            ):
                changed = copy.deepcopy(receipt)
                changed["static_verification"][field] = value
                changed["static_verification"] = _reseal(
                    changed["static_verification"]
                )
                mutations.append(("static verification", _reseal(changed)))

            for message, changed in mutations:
                with self.subTest(message=message, changed=changed):
                    with self.assertRaisesRegex(ValueError, message):
                        fold_processing_aws.validate_fold_inventory_preflight_receipt(
                            changed,
                            completed_fold_evidence=self.completed,
                            archive_copy_receipt=self.archive,
                            static_staging_receipt=self.static,
                            overlay_publication_receipt=self.publication,
                        )

    def _submit(
        self,
        preflight: Mapping[str, object],
        *,
        state_dir: Path,
    ) -> dict[str, object]:
        return fold_processing_aws.submit_fold_inventory_once(
            self.clients,
            preflight_receipt=preflight,
            completed_fold_evidence=self.completed,
            archive_copy_receipt=self.archive,
            static_staging_receipt=self.static,
            overlay_publication_receipt=self.publication,
            state_dir=state_dir,
        )

    def test_submission_persists_intent_before_exactly_one_create(self) -> None:
        state = self.root / "submission-state"
        expected_arn = (
            "arn:aws:sagemaker:us-east-1:371087393859:"
            f"processing-job/{self.job_name}"
        )
        self.clients.sagemaker.create_processing_job.return_value = {
            "ProcessingJobArn": expected_arn
        }
        with self._runtime():
            preflight = self._preflight()
            submission = self._submit(preflight, state_dir=state)
            validated = fold_processing_aws.validate_fold_inventory_submission_receipt(
                submission,
                preflight_receipt=preflight,
                completed_fold_evidence=self.completed,
                archive_copy_receipt=self.archive,
                static_staging_receipt=self.static,
                overlay_publication_receipt=self.publication,
            )

        self.assertEqual(submission, validated)
        self.clients.sagemaker.create_processing_job.assert_called_once_with(
            **preflight["request"]
        )
        self.assertEqual(
            sorted(path.name for path in state.iterdir()),
            ["create-intent.json", "state.json", "submission.json"],
        )
        intent, _ = controlled_supervisor.strict_config.load_canonical_json_object(
            state / "create-intent.json"
        )
        self.assertEqual(intent["request"], preflight["request"])
        self.assertEqual(intent["request_sha256"], preflight["request_sha256"])

    def test_submission_preflight_drift_creates_no_state_or_remote_job(self) -> None:
        with self._runtime():
            preflight = self._preflight()

        sdk_state = self.root / "sdk-drift-submission"
        with self._runtime(sdk_error=True):
            with self.assertRaisesRegex(RuntimeError, "SDK drift"):
                self._submit(preflight, state_dir=sdk_state)
        self.assertFalse(sdk_state.exists())
        self.clients.sagemaker.create_processing_job.assert_not_called()

        remote_state = self.root / "remote-drift-submission"
        with self._runtime() as (_, _, archives, _, _):
            archives.side_effect = RuntimeError("synthetic remote archive drift")
            with self.assertRaisesRegex(RuntimeError, "remote archive drift"):
                self._submit(preflight, state_dir=remote_state)
        self.assertFalse(remote_state.exists())
        self.clients.sagemaker.create_processing_job.assert_not_called()

    def test_lost_create_response_is_permanently_ambiguous_without_retry(self) -> None:
        state = self.root / "ambiguous-state"
        self.clients.sagemaker.create_processing_job.side_effect = TimeoutError(
            "synthetic lost CreateProcessingJob response"
        )
        with self._runtime():
            preflight = self._preflight()
            with self.assertRaisesRegex(TimeoutError, "lost CreateProcessingJob"):
                self._submit(preflight, state_dir=state)
            with self.assertRaisesRegex(FileExistsError, "state"):
                self._submit(preflight, state_dir=state)

        self.clients.sagemaker.create_processing_job.assert_called_once()
        self.assertTrue((state / "create-intent.json").is_file())
        self.assertFalse((state / "submission.json").exists())

    def _completed_description(
        self,
        preflight: Mapping[str, object],
        submission: Mapping[str, object],
    ) -> dict[str, object]:
        start = datetime(2026, 7, 13, 18, 0, 0, 123_456, tzinfo=timezone.utc)
        end = datetime(2026, 7, 13, 18, 0, 1, 358_023, tzinfo=timezone.utc)
        request = preflight["request"]
        processing_inputs = copy.deepcopy(request["ProcessingInputs"])
        for record in processing_inputs:
            record["AppManaged"] = False
        processing_output = copy.deepcopy(request["ProcessingOutputConfig"])
        for record in processing_output["Outputs"]:
            record["AppManaged"] = False
        return {
            "ProcessingJobName": preflight["job_name"],
            "ProcessingJobArn": submission["job_arn"],
            "ProcessingJobStatus": "Completed",
            "FailureReason": None,
            "ExitMessage": "Phase-1 evidence uploaded",
            "ProcessingStartTime": start,
            "ProcessingEndTime": end,
            "ProcessingInputs": processing_inputs,
            "ProcessingOutputConfig": processing_output,
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

    def test_terminal_rechecks_request_and_preserves_exact_microseconds(self) -> None:
        state = self.root / "terminal-state"
        expected_arn = (
            "arn:aws:sagemaker:us-east-1:371087393859:"
            f"processing-job/{self.job_name}"
        )
        self.clients.sagemaker.create_processing_job.return_value = {
            "ProcessingJobArn": expected_arn
        }
        with self._runtime():
            preflight = self._preflight()
            submission = self._submit(preflight, state_dir=state)
            description = self._completed_description(preflight, submission)
            self.clients.sagemaker.describe_processing_job.return_value = description
            terminal = fold_processing_aws.verify_completed_fold_inventory(
                self.clients,
                preflight_receipt=preflight,
                submission_receipt=submission,
                completed_fold_evidence=self.completed,
                archive_copy_receipt=self.archive,
                static_staging_receipt=self.static,
                overlay_publication_receipt=self.publication,
            )
            validated = fold_processing_aws.validate_fold_inventory_terminal_receipt(
                terminal,
                preflight_receipt=preflight,
                submission_receipt=submission,
                completed_fold_evidence=self.completed,
                archive_copy_receipt=self.archive,
                static_staging_receipt=self.static,
                overlay_publication_receipt=self.publication,
            )

        self.assertEqual(terminal, validated)
        self.assertEqual(terminal["processing_time_microseconds"], 1_234_567)
        self.assertEqual(
            terminal["processing_start_time"],
            "2026-07-13T18:00:00.123456Z",
        )
        self.assertEqual(
            terminal["processing_end_time"],
            "2026-07-13T18:00:01.358023Z",
        )
        self.clients.sagemaker.describe_processing_job.assert_called_once_with(
            ProcessingJobName=self.job_name
        )

    def test_terminal_rejects_request_drift_and_noncompletion(self) -> None:
        state = self.root / "terminal-drift-state"
        expected_arn = (
            "arn:aws:sagemaker:us-east-1:371087393859:"
            f"processing-job/{self.job_name}"
        )
        self.clients.sagemaker.create_processing_job.return_value = {
            "ProcessingJobArn": expected_arn
        }
        with self._runtime():
            preflight = self._preflight()
            submission = self._submit(preflight, state_dir=state)
            changed = self._completed_description(preflight, submission)
            changed["NetworkConfig"]["EnableNetworkIsolation"] = False
            self.clients.sagemaker.describe_processing_job.return_value = changed
            with self.assertRaisesRegex(RuntimeError, "NetworkConfig differs"):
                fold_processing_aws.verify_completed_fold_inventory(
                    self.clients,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    completed_fold_evidence=self.completed,
                    archive_copy_receipt=self.archive,
                    static_staging_receipt=self.static,
                    overlay_publication_receipt=self.publication,
                )

            app_managed = self._completed_description(preflight, submission)
            app_managed["ProcessingInputs"][0]["AppManaged"] = True
            self.clients.sagemaker.describe_processing_job.return_value = app_managed
            with self.assertRaisesRegex(RuntimeError, "unexpected service default"):
                fold_processing_aws.verify_completed_fold_inventory(
                    self.clients,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    completed_fold_evidence=self.completed,
                    archive_copy_receipt=self.archive,
                    static_staging_receipt=self.static,
                    overlay_publication_receipt=self.publication,
                )

            omitted_default = self._completed_description(preflight, submission)
            omitted_default["ProcessingOutputConfig"]["Outputs"][0].pop("AppManaged")
            self.clients.sagemaker.describe_processing_job.return_value = omitted_default
            with self.assertRaisesRegex(RuntimeError, "unexpected service default"):
                fold_processing_aws.verify_completed_fold_inventory(
                    self.clients,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    completed_fold_evidence=self.completed,
                    archive_copy_receipt=self.archive,
                    static_staging_receipt=self.static,
                    overlay_publication_receipt=self.publication,
                )

            pending = self._completed_description(preflight, submission)
            pending["ProcessingJobStatus"] = "InProgress"
            pending["ProcessingStartTime"] = None
            pending["ProcessingEndTime"] = None
            self.clients.sagemaker.describe_processing_job.return_value = pending
            with self.assertRaisesRegex(RuntimeError, "not cleanly complete"):
                fold_processing_aws.verify_completed_fold_inventory(
                    self.clients,
                    preflight_receipt=preflight,
                    submission_receipt=submission,
                    completed_fold_evidence=self.completed,
                    archive_copy_receipt=self.archive,
                    static_staging_receipt=self.static,
                    overlay_publication_receipt=self.publication,
                )


class _Phase1OutputS3:
    def __init__(
        self,
        *,
        prefix: str,
        payloads: Mapping[str, bytes],
        kms_key: str = _KMS_KEY,
    ) -> None:
        self.prefix = prefix
        self.payloads = dict(payloads)
        self.kms_key = kms_key
        self.extra_versions: list[dict[str, object]] = []
        self.delete_markers: list[dict[str, object]] = []
        self.head_mutation: str | None = None
        self.post_get_mutation: str | None = None
        self.get_calls = 0

    def _version(self, relative: str, payload: bytes, ordinal: int) -> dict[str, object]:
        return {
            "ETag": f'"{ordinal + 500:032x}"',
            "IsLatest": True,
            "Key": self.prefix + "evidence/" + relative,
            "Size": len(payload),
            "StorageClass": "STANDARD",
            "VersionId": f"phase1-version-{ordinal}",
        }

    def _versions(self) -> list[dict[str, object]]:
        return [
            self._version(relative, payload, ordinal)
            for ordinal, (relative, payload) in enumerate(sorted(self.payloads.items()))
        ] + copy.deepcopy(self.extra_versions)

    def list_object_versions(self, **request: object) -> dict[str, object]:
        return {
            "DeleteMarkers": copy.deepcopy(self.delete_markers),
            "IsTruncated": False,
            "MaxKeys": request["MaxKeys"],
            "Name": request["Bucket"],
            "Prefix": request["Prefix"],
            "Versions": self._versions(),
        }

    def _selected(self, request: Mapping[str, object]) -> tuple[dict[str, object], bytes]:
        for version in self._versions():
            if (
                version["Key"] == request["Key"]
                and version["VersionId"] == request["VersionId"]
            ):
                relative = version["Key"].removeprefix(self.prefix + "evidence/")
                return version, self.payloads[relative]
        raise AssertionError("Synthetic Phase-1 object was not found")

    def head_object(self, **request: object) -> dict[str, object]:
        version, _ = self._selected(request)
        head = {
            "BucketKeyEnabled": True,
            "ContentLength": version["Size"],
            "ETag": version["ETag"],
            "SSEKMSKeyId": self.kms_key,
            "ServerSideEncryption": "aws:kms",
            "VersionId": version["VersionId"],
        }
        if self.head_mutation == "kms":
            head["SSEKMSKeyId"] = _KMS_KEY + "-changed"
        elif self.head_mutation == "size":
            head["ContentLength"] += 1
        return head

    def get_object(self, **request: object) -> dict[str, object]:
        _, payload = self._selected(request)
        self.get_calls += 1
        if self.get_calls == 3 and self.post_get_mutation == "extra-version":
            self.extra_versions = [
                {
                    "ETag": '"ffffffffffffffffffffffffffffffff"',
                    "IsLatest": True,
                    "Key": self.prefix + "evidence/extra.json",
                    "Size": 2,
                    "StorageClass": "STANDARD",
                    "VersionId": "concurrent-extra-version",
                }
            ]
        elif self.get_calls == 3 and self.post_get_mutation == "delete-marker":
            self.delete_markers = [
                {
                    "Key": self.prefix + "evidence/deleted.json",
                    "VersionId": "concurrent-delete-version",
                }
            ]
        return {"Body": io.BytesIO(payload)}


class FoldInventoryAcquisitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.prefix = "arr-retrieval-cv/fold-eval/f0-inventory-output/"
        self.payloads = {
            name: fold_processing_aws._canonical_bytes({"fixture": name})
            for name in (
                "archive_inventory.json",
                "artifact_manifest.json",
                "bm25_storage.json",
            )
        }
        self.s3 = _Phase1OutputS3(prefix=self.prefix, payloads=self.payloads)
        self.clients = FoldArchiveCopyTest._clients(self.s3)
        self.completed = FoldArchiveCopyTest._completed_evidence()
        self.archive = {"fold_archive_input_manifest": {"fixture": "manifest"}}
        self.preflight = {
            "account_id": "371087393859",
            "outer_fold": 0,
            "output_prefix": self.prefix,
            "request": {
                "ProcessingOutputConfig": {
                    "KmsKeyId": _KMS_KEY,
                    "Outputs": [
                        {
                            "S3Output": {
                                "S3Uri": f"s3://ir-sagemaker/{self.prefix}"
                            }
                        }
                    ],
                }
            },
        }
        self.submission = {"fixture": "submission"}
        self.terminal = {"fixture": "terminal"}
        self.static = {"fixture": "static"}
        self.publication = {"fixture": "publication"}
        self.inventory = {"receipt_sha256": "1" * 64}
        self.storage = {"receipt_sha256": "2" * 64}
        self.artifact = {
            "artifact_manifest_sha256": "3" * 64,
            "files": [
                {
                    "path": name,
                    "size": len(self.payloads[name]),
                    "sha256": hashlib.sha256(self.payloads[name]).hexdigest(),
                }
                for name in ("archive_inventory.json", "bm25_storage.json")
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @contextmanager
    def _runtime(self):
        with (
            patch.object(
                fold_processing_aws,
                "validate_fold_inventory_terminal_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "validate_fold_inventory_preflight_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws.controlled_supervisor,
                "validate_completed_fold_evidence",
                side_effect=lambda value: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "validate_fold_archive_copy_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "_validate_phase1_documents",
                return_value=(
                    copy.deepcopy(self.inventory),
                    copy.deepcopy(self.storage),
                    copy.deepcopy(self.artifact),
                ),
            ) as documents,
        ):
            yield documents

    def _acquire(self, *, output_dir: Path | None = None) -> dict[str, object]:
        return fold_processing_aws.acquire_fold_inventory_once(
            self.clients,
            terminal_receipt=self.terminal,
            preflight_receipt=self.preflight,
            submission_receipt=self.submission,
            completed_fold_evidence=self.completed,
            archive_copy_receipt=self.archive,
            static_staging_receipt=self.static,
            overlay_publication_receipt=self.publication,
            output_dir=output_dir or self.root / "acquired",
        )

    def test_acquires_only_three_compact_files_with_exact_version_evidence(self) -> None:
        output = self.root / "acquired"
        with self._runtime() as documents:
            receipt = self._acquire(output_dir=output)

        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            [
                "acquisition_receipt.json",
                "archive_inventory.json",
                "artifact_manifest.json",
                "bm25_storage.json",
            ],
        )
        for name, payload in self.payloads.items():
            self.assertEqual((output / name).read_bytes(), payload)
        self.assertEqual(len(receipt["remote_objects"]), 3)
        self.assertEqual(
            {record["version_id"] for record in receipt["remote_objects"]},
            {"phase1-version-0", "phase1-version-1", "phase1-version-2"},
        )
        self.assertTrue(
            all(
                record["encryption"]
                == {
                    "algorithm": "aws:kms",
                    "kms_key_id": _KMS_KEY,
                    "bucket_key_enabled": True,
                }
                for record in receipt["remote_objects"]
            )
        )
        documents.assert_called_once()
        self.assertEqual(self.s3.get_calls, 3)

    def test_output_history_delete_extra_and_kms_drift_fail_loudly(self) -> None:
        self.s3.delete_markers = [
            {"Key": self.prefix + "deleted", "VersionId": "delete-version"}
        ]
        with self._runtime():
            with self.assertRaisesRegex(RuntimeError, "exactly three versions"):
                self._acquire()
        self.s3.delete_markers = []
        self.s3.extra_versions = [
            {
                "ETag": '"ffffffffffffffffffffffffffffffff"',
                "IsLatest": True,
                "Key": self.prefix + "evidence/extra.json",
                "Size": 2,
                "StorageClass": "STANDARD",
                "VersionId": "extra-version",
            }
        ]
        with self._runtime():
            with self.assertRaisesRegex(RuntimeError, "exactly three versions"):
                self._acquire()
        self.s3.extra_versions = []
        self.s3.head_mutation = "kms"
        with self._runtime():
            with self.assertRaisesRegex(RuntimeError, "metadata changed"):
                self._acquire()

    def test_partial_local_acquisition_is_preserved_and_never_resumed(self) -> None:
        output = self.root / "partial"
        real_write = fold_processing_aws._write_bytes_at
        writes = 0

        def fail_second(directory: Path, name: str, payload: bytes) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("synthetic local write failure")
            real_write(directory, name, payload)

        with self._runtime(), patch.object(
            fold_processing_aws,
            "_write_bytes_at",
            side_effect=fail_second,
        ):
            with self.assertRaisesRegex(OSError, "local write failure"):
                self._acquire(output_dir=output)
        incomplete = output.with_name(".partial.incomplete")
        self.assertTrue(incomplete.is_dir())
        self.assertEqual(len(list(incomplete.iterdir())), 1)
        with self._runtime():
            with self.assertRaisesRegex(FileExistsError, "initially absent"):
                self._acquire(output_dir=output)
        self.assertFalse(output.exists())

    def test_concurrent_final_history_drift_makes_no_local_mutation(self) -> None:
        for mutation in ("extra-version", "delete-marker"):
            with self.subTest(mutation=mutation):
                self.s3 = _Phase1OutputS3(
                    prefix=self.prefix,
                    payloads=self.payloads,
                )
                self.s3.post_get_mutation = mutation
                self.clients = FoldArchiveCopyTest._clients(self.s3)
                output = self.root / f"acquired-{mutation}"
                with self._runtime():
                    with self.assertRaisesRegex(
                        RuntimeError, "changed during acquisition"
                    ):
                        self._acquire(output_dir=output)
                self.assertFalse(output.exists())
                self.assertFalse(
                    output.with_name(f".{output.name}.incomplete").exists()
                )

    def test_resealed_acquisition_remote_splice_is_rejected(self) -> None:
        with self._runtime():
            receipt = self._acquire()
            changed = copy.deepcopy(receipt)
            changed["remote_objects"][0]["key"] = (
                self.prefix + "evidence/bm25_storage.json"
            )
            changed = _reseal(changed)
            with self.assertRaisesRegex(
                ValueError, "(?:coverage|identity) changed"
            ):
                fold_processing_aws.validate_fold_inventory_acquisition_receipt(
                    changed,
                    terminal_receipt=self.terminal,
                    preflight_receipt=self.preflight,
                    submission_receipt=self.submission,
                    completed_fold_evidence=self.completed,
                    archive_copy_receipt=self.archive,
                    static_staging_receipt=self.static,
                    overlay_publication_receipt=self.publication,
                )


class Phase1DocumentValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.completed = {
            "outer_fold": 0,
            "training_plan": {
                "study": {"dataset_manifest_sha256": "d" * 64}
            },
        }
        systems: list[dict[str, object]] = []
        observed: list[dict[str, object]] = []
        for ordinal in range(12):
            raw_digest = bytes([ordinal + 1]) * 32
            checksum = base64.b64encode(raw_digest).decode("ascii")
            expected = {
                "ordinal": ordinal,
                "system_id": f"system-{ordinal:02d}",
                "destination_object": {
                    "size": 100 + ordinal,
                    "checksum": {
                        "algorithm": "SHA256",
                        "type": "FULL_OBJECT",
                        "value": checksum,
                    },
                },
            }
            systems.append(expected)
            observed.append(
                {
                    **copy.deepcopy(expected),
                    "archive_evidence": {
                        "archive": {
                            "size": 100 + ordinal,
                            "sha256": raw_digest.hex(),
                        },
                        "members": [{"kind": "file", "size": ordinal + 1}],
                    },
                }
            )
        self.archive = {
            "fold_archive_input_manifest": {
                "schema_version": 1,
                "systems": systems,
            }
        }
        self.inventory = _reseal(
            {
                "schema_version": 1,
                "protocol": fold_processing_aws.ARCHIVE_INVENTORY_PROTOCOL,
                "input_manifest_sha256": fold_processing_aws._document_sha256(
                    self.archive["fold_archive_input_manifest"]
                ),
                "experiment_id": "arr_retrieval_cv_v1",
                "outer_fold": 0,
                "systems": observed,
                "aggregate": {"system_count": 12},
            }
        )
        tree_hash = "a" * 64
        self.storage = _reseal(
            {
                "schema_version": 1,
                "protocol": fold_processing_aws.PHASE1_STORAGE_PROTOCOL,
                "experiment_id": "arr_retrieval_cv_v1",
                "outer_fold": 0,
                "role": "test",
                "regime": "fold_global",
                "archive_input_manifest_sha256": self.inventory[
                    "input_manifest_sha256"
                ],
                "archive_inventory_receipt_sha256": self.inventory[
                    "receipt_sha256"
                ],
                "dataset_manifest_sha256": "d" * 64,
                "fold_manifest_sha256": (
                    fold_processing_aws.EXPECTED_FOLD_MANIFEST_SHA256
                ),
                "passage_index_sha256": "b" * 64,
                "case_ids": ["case-1"],
                "case_ids_sha256": "c" * 64,
                "query_count": 1,
                "query_ids_sha256": "e" * 64,
                "passage_count": 2,
                "passage_ids_sha256": "f" * 64,
                "candidate_pools_sha256": "1" * 64,
                "evaluation_contract_sha256": "2" * 64,
                "bm25_index_arguments": {"k1": 1.2, "b": 0.75},
                "bm25_runtime": {"fixture": "runtime"},
                "bm25_replicas": [
                    {"ordinal": 1, "allocation_tree_sha256": tree_hash},
                    {"ordinal": 2, "allocation_tree_sha256": tree_hash},
                ],
                "bm25_allocation_tree": {
                    "allocation_tree_sha256": tree_hash,
                    "allocated_bytes": 4_096,
                },
                "filesystem_before": {"available_bytes": 100_000},
                "filesystem_after": {"available_bytes": 90_000},
                "image_runtime": {"fixture": "image"},
            }
        )
        artifact_payload = {
            "schema_version": 1,
            "protocol": fold_processing_aws.PHASE1_OUTPUT_PROTOCOL,
            "experiment_id": "arr_retrieval_cv_v1",
            "outer_fold": 0,
            "archive_input_manifest_sha256": self.inventory[
                "input_manifest_sha256"
            ],
            "archive_inventory_receipt_sha256": self.inventory[
                "receipt_sha256"
            ],
            "bm25_storage_receipt_sha256": self.storage["receipt_sha256"],
            "files": [
                {"path": "archive_inventory.json", "size": 1, "sha256": "3" * 64},
                {"path": "bm25_storage.json", "size": 1, "sha256": "4" * 64},
            ],
        }
        self.artifact = {
            **artifact_payload,
            "artifact_manifest_sha256": fold_processing_aws._document_sha256(
                artifact_payload
            ),
        }

    def _validate(
        self,
        *,
        inventory: Mapping[str, object] | None = None,
        storage: Mapping[str, object] | None = None,
        artifact: Mapping[str, object] | None = None,
    ):
        return fold_processing_aws._validate_phase1_documents(
            archive_inventory=inventory or self.inventory,
            bm25_storage=storage or self.storage,
            artifact_manifest=artifact or self.artifact,
            archive_copy=self.archive,
            completed=self.completed,
        )

    def test_exact_documents_are_accepted(self) -> None:
        inventory, storage, artifact = self._validate()
        self.assertEqual(inventory, self.inventory)
        self.assertEqual(storage, self.storage)
        self.assertEqual(artifact, self.artifact)

    def test_resealed_archive_system_and_bytes_splices_are_rejected(self) -> None:
        changed = copy.deepcopy(self.inventory)
        changed["systems"][0]["system_id"] = changed["systems"][1]["system_id"]
        with self.assertRaisesRegex(ValueError, "system 0 was spliced"):
            self._validate(inventory=_reseal(changed))
        changed = copy.deepcopy(self.inventory)
        changed["systems"][0]["archive_evidence"]["archive"]["size"] += 1
        with self.assertRaisesRegex(ValueError, "system 0 bytes changed"):
            self._validate(inventory=_reseal(changed))

    def test_resealed_storage_and_replica_binding_splices_are_rejected(self) -> None:
        changed = copy.deepcopy(self.storage)
        changed["dataset_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "storage binding changed"):
            self._validate(storage=_reseal(changed))
        changed = copy.deepcopy(self.storage)
        changed["bm25_replicas"][1]["allocation_tree_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "replica allocation changed"):
            self._validate(storage=_reseal(changed))

    def test_rehashed_artifact_cross_document_splice_is_rejected(self) -> None:
        changed = copy.deepcopy(self.artifact)
        changed["archive_inventory_receipt_sha256"] = "0" * 64
        payload = {
            key: copy.deepcopy(value)
            for key, value in changed.items()
            if key != "artifact_manifest_sha256"
        }
        changed["artifact_manifest_sha256"] = fold_processing_aws._document_sha256(
            payload
        )
        with self.assertRaisesRegex(ValueError, "manifest binding changed"):
            self._validate(artifact=changed)


class FoldStorageProofTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.acquisition_dir = self.root / "phase1"
        self.acquisition_dir.mkdir()
        self.payloads = {
            name: fold_processing_aws._canonical_bytes({"fixture": name})
            for name in (
                "archive_inventory.json",
                "artifact_manifest.json",
                "bm25_storage.json",
            )
        }
        self.acquisition = {
            "archive_inventory_receipt_sha256": "1" * 64,
            "bm25_storage_receipt_sha256": "2" * 64,
            "files": [
                {
                    "path": name,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(self.payloads.items())
            ],
            "fixture": "acquisition",
        }
        self.completed = {
            "outer_fold": 0,
            "training_plan": {
                "infrastructure": {"processing_volume_size_gb": 100}
            },
        }
        self.archive = {"fixture": "archive-copy"}
        self.static = {
            "assets": [
                {
                    "name": "e5-snapshot",
                    "files": [{"size": 1}, {"size": 4_096}],
                },
                {"name": "e5-pack", "files": [{"size": 4_097}]},
                {"name": "fixed-base", "files": [{"size": 1}]},
                # Control is already represented by the explicit Phase-2
                # control-bundle sizes, so its staged source bytes are not
                # counted again.
                {"name": "control", "files": [{"size": 999_999}]},
            ]
        }
        self.inventory = {
            "receipt_sha256": "1" * 64,
            "systems": [
                {
                    "archive_evidence": {
                        "members": [
                            {"kind": "directory", "size": 0},
                            {"kind": "file", "size": 1},
                            {"kind": "file", "size": 4_097},
                        ]
                    }
                }
            ],
        }
        fragment = 4_096
        capacity = 3_740_124_893_184
        self.storage = {
            "receipt_sha256": "2" * 64,
            "filesystem_before": {
                "block_size": fragment,
                "fragment_size": fragment,
                "blocks": capacity // fragment,
                "blocks_free": 30,
                "blocks_available": 24,
                "capacity_bytes": capacity,
                "free_bytes": 30 * fragment,
                "available_bytes": 24 * fragment,
            },
            "bm25_allocation_tree": {"allocated_bytes": 5 * fragment},
        }
        self._save_acquisition(self.acquisition)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _save_acquisition(self, acquisition: Mapping[str, object]) -> None:
        for name, payload in self.payloads.items():
            (self.acquisition_dir / name).write_bytes(payload)
        (self.acquisition_dir / "acquisition_receipt.json").write_bytes(
            controlled_supervisor.strict_config.canonical_json_bytes(acquisition)
        )

    @contextmanager
    def _runtime(self, *, storage: Mapping[str, object] | None = None):
        selected_storage = copy.deepcopy(storage or self.storage)
        with (
            patch.object(
                fold_processing_aws,
                "validate_fold_inventory_acquisition_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws.controlled_supervisor,
                "validate_completed_fold_evidence",
                side_effect=lambda value: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "validate_fold_archive_copy_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "validate_static_evaluation_staging_receipt",
                side_effect=lambda value, **_: copy.deepcopy(value),
            ),
            patch.object(
                fold_processing_aws,
                "_validate_phase1_documents",
                return_value=(
                    copy.deepcopy(self.inventory),
                    selected_storage,
                    {"fixture": "artifact"},
                ),
            ),
        ):
            yield

    def _build(
        self,
        *,
        acquisition: Mapping[str, object] | None = None,
        control_sizes: tuple[int, ...] = (1, 4_097),
        output_reserve: int = 4_096,
        safety_reserve: int = 4_095,
    ) -> dict[str, object]:
        return fold_processing_aws.build_fold_storage_proof(
            acquisition_receipt=acquisition or self.acquisition,
            acquisition_dir=self.acquisition_dir,
            terminal_receipt={"fixture": "terminal"},
            preflight_receipt={"fixture": "preflight"},
            submission_receipt={"fixture": "submission"},
            completed_fold_evidence=self.completed,
            archive_copy_receipt=self.archive,
            static_staging_receipt=self.static,
            overlay_publication_receipt={"fixture": "publication"},
            phase2_control_file_sizes=control_sizes,
            phase2_output_reserve_bytes=output_reserve,
            safety_reserve_bytes=safety_reserve,
        )

    def test_exact_allocated_components_and_receipt_bindings(self) -> None:
        with self._runtime():
            proof = self._build()

        self.assertEqual(
            proof["components"],
            {
                "static_phase2_inputs_allocated_upper_bound": 20_480,
                "phase1_evidence_input_allocated_upper_bound": 12_288,
                "phase2_control_bundle_allocated_upper_bound": 12_288,
                "controlled_artifact_extraction_allocated_upper_bound": 16_384,
                "bm25_index_allocated_bytes": 20_480,
                "phase2_output_reserve_bytes": 4_096,
                "safety_reserve_bytes": 4_095,
            },
        )
        self.assertEqual(proof["required_additional_bytes"], 90_111)
        self.assertEqual(proof["remaining_bytes"], 8_193)
        self.assertEqual(proof["volume_size_gb"], 100)
        self.assertEqual(
            proof["archive_copy_receipt_sha256"],
            fold_processing_aws._document_sha256(self.archive),
        )
        self.assertEqual(
            proof["static_staging_receipt_sha256"],
            fold_processing_aws._document_sha256(self.static),
        )
        self.assertEqual(
            proof["inventory_acquisition_receipt_sha256"],
            fold_processing_aws._document_sha256(self.acquisition),
        )

    def test_one_byte_headroom_passes_and_exact_fit_fails(self) -> None:
        # The allocated components before the safety reserve total 86,016;
        # setting safety to 12,287 leaves exactly one byte of headroom in the
        # measured 98,304 available bytes.
        with self._runtime():
            proof = self._build(safety_reserve=12_287)
            self.assertEqual(proof["remaining_bytes"], 1)
            with self.assertRaisesRegex(RuntimeError, "does not fit"):
                self._build(safety_reserve=12_288)

    def test_local_bytes_and_phase1_receipt_bindings_fail_loudly(self) -> None:
        changed_file = self.acquisition_dir / "archive_inventory.json"
        original = changed_file.read_bytes()
        changed_file.write_bytes(original + b" ")
        with self._runtime():
            with self.assertRaisesRegex(ValueError, "acquired file changed"):
                self._build()
        changed_file.write_bytes(original)

        changed = copy.deepcopy(self.acquisition)
        changed["archive_inventory_receipt_sha256"] = "3" * 64
        self._save_acquisition(changed)
        with self._runtime():
            with self.assertRaisesRegex(ValueError, "documents changed"):
                self._build(acquisition=changed)

    def test_filesystem_capacity_free_and_available_arithmetic_is_exact(self) -> None:
        mutations: list[dict[str, object]] = []
        changed = copy.deepcopy(self.storage)
        changed["filesystem_before"]["capacity_bytes"] = 101 * 1_024**3
        changed["filesystem_before"]["blocks"] = (
            changed["filesystem_before"]["capacity_bytes"] // 4_096
        )
        mutations.append(changed)
        changed = copy.deepcopy(self.storage)
        changed["filesystem_before"]["free_bytes"] += 4_096
        mutations.append(changed)
        changed = copy.deepcopy(self.storage)
        changed["filesystem_before"]["available_bytes"] += 4_096
        mutations.append(changed)
        changed = copy.deepcopy(self.storage)
        changed["filesystem_before"]["blocks_available"] = 31
        changed["filesystem_before"]["available_bytes"] = 31 * 4_096
        mutations.append(changed)

        for changed in mutations:
            with self.subTest(filesystem=changed["filesystem_before"]):
                with self._runtime(storage=changed):
                    with self.assertRaisesRegex(ValueError, "filesystem"):
                        self._build()

    def test_resealed_proof_rejects_arithmetic_types_and_capacity_drift(self) -> None:
        with self._runtime():
            proof = self._build()
        mutations: list[dict[str, object]] = []
        changed = copy.deepcopy(proof)
        changed["required_additional_bytes"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(proof)
        changed["components"]["bm25_index_allocated_bytes"] = -1
        mutations.append(changed)
        changed = copy.deepcopy(proof)
        changed["components"]["bm25_index_allocated_bytes"] = True
        mutations.append(changed)
        changed = copy.deepcopy(proof)
        changed["filesystem_available_bytes"] = True
        mutations.append(changed)
        changed = copy.deepcopy(proof)
        changed["fits"] = False
        mutations.append(changed)
        changed = copy.deepcopy(proof)
        changed["volume_size_gb"] = 101
        mutations.append(changed)
        changed = copy.deepcopy(proof)
        changed["filesystem_capacity_bytes"] = 101 * 1_024**3
        mutations.append(changed)

        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    fold_processing_aws.validate_fold_storage_proof(
                        _reseal(changed)
                    )

    def test_control_bundle_sizes_and_reserves_are_explicit_positive_integers(self) -> None:
        with self._runtime():
            for sizes in ((), (0,), (True,), (1, -1)):
                with self.subTest(sizes=sizes):
                    with self.assertRaisesRegex(ValueError, "control-bundle sizes"):
                        self._build(control_sizes=sizes)
            for name, value in (("output", 0), ("output", True), ("safety", -1)):
                with self.subTest(name=name, value=value):
                    arguments = (
                        {"output_reserve": value}
                        if name == "output"
                        else {"safety_reserve": value}
                    )
                    with self.assertRaisesRegex(ValueError, "reserves"):
                        self._build(**arguments)


if __name__ == "__main__":
    unittest.main()
