from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import retriever.determinism_artifacts as determinism_artifacts  # noqa: E402
from corporate_reorganization.modernbert.retriever import (  # noqa: E402
    determinism_artifacts as package_determinism_artifacts,
)
from corporate_reorganization.modernbert.experiments.retrieval_cv import (  # noqa: E402
    aws,
    config as strict_config,
    manifest,
    training_artifacts,
    training_launch,
)
from corporate_reorganization.modernbert.tests import (  # noqa: E402
    test_retrieval_cv_training_launch as training_launch_tests,
)
from corporate_reorganization.modernbert.tests.test_retrieval_determinism_artifacts import (  # noqa: E402
    SYNTHETIC_MODEL_INVENTORY,
    _build,
    _canonical_sha,
    _rebuild_smoke_evidence,
    _record,
    _write_json,
)


_LAST_MODIFIED = datetime(2026, 7, 13, 16, 0, tzinfo=timezone.utc)
_VERSION_ID = "exact-version-id"
_OWNER_ID = "1" * 64
_ETAG = '"0123456789abcdef0123456789abcdef-2"'
_CHECKSUM = "AAAAAA==-2"


def _document_sha256(value: object) -> str:
    return aws.sha256_bytes(aws.canonical_json_bytes(value))


def _completed_smoke_launch(
    helper: training_launch_tests.TrainingLaunchTest,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    helper.run_id = next(
        run["run_id"]
        for run in (*helper.plan["controlled_runs"], *helper.plan["auxiliary_runs"])
        if run["kind"] == manifest.SMOKE_KIND
    )
    clients = helper._clients()
    with helper._remote_dependencies():
        preflight, submission = helper._submit_in_progress(clients)
        clients.sagemaker.reset_mock()
        clients.sagemaker.describe_training_job.side_effect = None
        clients.sagemaker.describe_training_job.return_value = (
            helper._describe_response(preflight, status="Completed")
        )
        clients.sagemaker.list_tags.side_effect = helper._tag_pages(preflight)
        terminal = training_launch.verify_terminal_training_job(
            clients,
            training_plan=helper.plan,
            staging_receipt=helper.staging,
            preflight_receipt=preflight,
            submission_receipt=submission,
        )
    return preflight, submission, terminal


def _bind_artifact_to_plan(
    artifact_root: Path,
    *,
    plan: dict[str, object],
    staging: dict[str, object],
) -> None:
    _build(artifact_root)
    run_path = artifact_root / "determinism_smoke_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    sources = plan["sources"]
    source_bundle = {
        "commit_epoch": sources["commit_epoch"],
        "inventory_sha256": sources["source_inventory_sha256"],
        "name": sources["source_bundle_path"],
        "sha256": sources["source_bundle_sha256"],
        "size": sources["source_bundle_size"],
    }
    run["training_plan_sha256"] = _document_sha256(plan)
    run["training_staging_receipt_sha256"] = _document_sha256(staging)
    run["source_bundle"] = source_bundle
    launch_ledger = {
        "training_image": run["training_image"],
        "training_base_image": run["training_base_image"],
        "runtime_inventory_sha256": run[
            "training_image_runtime_inventory_sha256"
        ],
        "training_image_contract_sha256": run[
            "training_image_contract_sha256"
        ],
        "bootstrap_protocol": run["training_bootstrap_protocol"],
        "training_plan_sha256": run["training_plan_sha256"],
        "training_staging_receipt_sha256": run[
            "training_staging_receipt_sha256"
        ],
        "source_bundle": source_bundle,
    }
    run["determinism_scientific_evidence"]["launch_ledger"] = {
        "sha256": _canonical_sha(launch_ledger)
    }
    run["determinism_scientific_evidence"] = _rebuild_smoke_evidence(
        artifact_root, run
    )
    _write_json(run_path, run)
    manifest_path = artifact_root / "artifact_manifest.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_manifest["determinism_smoke_run"] = _record(
        artifact_root, "determinism_smoke_run.json"
    )
    _write_json(manifest_path, artifact_manifest)


def _pax_record(key: str, value: str) -> bytes:
    body = f"{key}={value}\n".encode("utf-8")
    length = len(body) + 2
    while True:
        record = f"{length} ".encode("ascii") + body
        if len(record) == length:
            return record
        length = len(record)


def _pax_payload(*, extra_key: bool = False) -> bytes:
    value = "1700000000.0000000"
    keys = ("atime", "ctime", "mtime", "LIBARCHIVE.creationtime")
    payload = b"".join(_pax_record(key, value) for key in keys)
    if extra_key:
        payload += _pax_record("path", "forbidden")
    return payload


def _tar_header(
    name: str,
    *,
    member_type: bytes,
    size: int,
) -> bytes:
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.size = size
    info.mode = 0o755 if member_type == tarfile.DIRTYPE else 0o600
    info.uid = 0
    info.gid = 0
    info.mtime = 1_700_000_000
    return info.tobuf(format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict")


def _physical_pax_name(relative: str, *, is_directory: bool) -> str:
    flattened = relative.replace("/", "_") + ("_" if is_directory else "")
    return f"./PaxHeaders.X/{flattened}"


def _archive_tree(
    source_root: Path,
    *,
    numeric_pax_namespace: bool = False,
    extra_pax_key: bool = False,
) -> bytes:
    paths = sorted(
        source_root.rglob("*"),
        key=lambda path: (
            len(path.relative_to(source_root).parts),
            path.relative_to(source_root).as_posix(),
        ),
    )
    tar_bytes = bytearray()
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        is_directory = path.is_dir()
        payload = _pax_payload(extra_key=extra_pax_key)
        pax_name = _physical_pax_name(relative, is_directory=is_directory)
        if numeric_pax_namespace:
            pax_name = pax_name.replace("PaxHeaders.X", "PaxHeaders.0")
        tar_bytes.extend(
            _tar_header(pax_name, member_type=tarfile.XHDTYPE, size=len(payload))
        )
        tar_bytes.extend(payload)
        tar_bytes.extend(b"\0" * ((-len(payload)) % 512))
        if is_directory:
            logical_name = relative + "/"
            content = b""
            member_type = tarfile.DIRTYPE
        else:
            logical_name = relative
            content = path.read_bytes()
            member_type = tarfile.REGTYPE
        tar_bytes.extend(
            _tar_header(logical_name, member_type=member_type, size=len(content))
        )
        tar_bytes.extend(content)
        tar_bytes.extend(b"\0" * ((-len(content)) % 512))
    tar_bytes.extend(b"\0" * 1024)
    tar_bytes.extend(b"\0" * ((-len(tar_bytes)) % 10240))
    return gzip.compress(bytes(tar_bytes), mtime=0)


class _FakeS3:
    def __init__(
        self,
        archive: bytes,
        *,
        preflight: dict[str, object],
        terminal: dict[str, object],
        truncate_body: bool = False,
        extra_version: bool = False,
        drift_on_list_call: int | None = None,
    ) -> None:
        self.archive = archive
        self.preflight = preflight
        self.terminal = terminal
        self.truncate_body = truncate_body
        self.extra_version = extra_version
        self.drift_on_list_call = drift_on_list_call
        self.list_calls = 0
        self.head_calls = 0
        self.get_calls = 0

    @property
    def _coordinates(self) -> tuple[str, str]:
        return strict_config._s3_uri_coordinates(
            self.terminal["model_artifact_s3_uri"]
        )

    def _version(self) -> dict[str, object]:
        _, key = self._coordinates
        return {
            "ChecksumAlgorithm": ["CRC32"],
            "ChecksumType": "COMPOSITE",
            "ETag": _ETAG,
            "IsLatest": True,
            "Key": key,
            "LastModified": _LAST_MODIFIED,
            "Owner": {"ID": _OWNER_ID},
            "Size": len(self.archive),
            "StorageClass": "STANDARD",
            "VersionId": _VERSION_ID,
        }

    def list_object_versions(self, **request: object) -> dict[str, object]:
        self.list_calls += 1
        bucket, _ = self._coordinates
        versions = [self._version()]
        if self.extra_version or self.list_calls == self.drift_on_list_call:
            extra = copy.deepcopy(versions[0])
            extra["VersionId"] = "unexpected-second-version"
            extra["IsLatest"] = False
            versions.append(extra)
        return {
            "DeleteMarkers": [],
            "IsTruncated": False,
            "MaxKeys": request["MaxKeys"],
            "Name": bucket,
            "Prefix": request["Prefix"],
            "Versions": versions,
        }

    def _metadata(self) -> dict[str, object]:
        return {
            "AcceptRanges": "bytes",
            "BucketKeyEnabled": True,
            "ChecksumCRC32": _CHECKSUM,
            "ChecksumType": "COMPOSITE",
            "ContentLength": len(self.archive),
            "ContentType": "application/gzip",
            "ETag": _ETAG,
            "LastModified": _LAST_MODIFIED,
            "Metadata": {},
            "SSEKMSKeyId": (
                f"arn:aws:kms:{self.preflight['region']}:"
                f"{self.preflight['account_id']}:key/exact-key-id"
            ),
            "ServerSideEncryption": "aws:kms",
            "VersionId": _VERSION_ID,
        }

    def head_object(self, **request: object) -> dict[str, object]:
        self.head_calls += 1
        return self._metadata()

    def get_object(self, **request: object) -> dict[str, object]:
        self.get_calls += 1
        body = self.archive[:-1] if self.truncate_body else self.archive
        return {**self._metadata(), "Body": io.BytesIO(body)}


class TrainingArtifactAcquisitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.launch = training_launch_tests.TrainingLaunchTest()
        self.launch.setUp()
        self.addCleanup(self.launch.tearDown)
        self.plan = self.launch.plan
        self.staging = self.launch.staging
        (
            self.preflight,
            self.submission,
            self.terminal,
        ) = _completed_smoke_launch(self.launch)
        self.model_inventory_patch = mock.patch.object(
            determinism_artifacts,
            "_EXPECTED_MODEL_TENSOR_SHAPES",
            SYNTHETIC_MODEL_INVENTORY,
        )
        self.model_inventory_patch.start()
        self.addCleanup(self.model_inventory_patch.stop)
        self.package_model_inventory_patch = mock.patch.object(
            package_determinism_artifacts,
            "_EXPECTED_MODEL_TENSOR_SHAPES",
            SYNTHETIC_MODEL_INVENTORY,
        )
        self.package_model_inventory_patch.start()
        self.addCleanup(self.package_model_inventory_patch.stop)

    def _artifact_archive(self) -> bytes:
        artifact_root = self.launch.root / "remote-artifact"
        _bind_artifact_to_plan(
            artifact_root,
            plan=self.plan,
            staging=self.staging,
        )
        return _archive_tree(artifact_root)

    def _acquire(
        self,
        archive: bytes,
        *,
        s3: _FakeS3 | None = None,
        name: str = "acquired",
    ) -> tuple[Path, _FakeS3, dict[str, object]]:
        output = self.launch.root / name
        client = s3 or _FakeS3(
            archive, preflight=self.preflight, terminal=self.terminal
        )
        with mock.patch.object(training_artifacts, "_require_disk_space"):
            receipt = training_artifacts.acquire_completed_determinism_smoke_artifact(
                client,
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=self.preflight,
                submission_receipt=self.submission,
                terminal_receipt=self.terminal,
                output_bundle=output,
            )
        return output, client, receipt

    def test_acquires_validates_and_binds_exact_completed_artifact(self) -> None:
        archive = self._artifact_archive()
        output, client, receipt = self._acquire(archive)
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"model.tar.gz", "artifact", "acquisition_receipt.json"},
        )
        loaded = training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
            output / "acquisition_receipt.json",
            training_plan=self.plan,
            staging_receipt=self.staging,
        )
        self.assertEqual(loaded.receipt, receipt)
        self.assertEqual(loaded.archive_sha256, hashlib.sha256(archive).hexdigest())
        self.assertEqual(loaded.remote_object["sha256"], loaded.archive_sha256)
        self.assertEqual(loaded.receipt_path, output / "acquisition_receipt.json")
        self.assertEqual(loaded.bundle_root, output)
        self.assertEqual(loaded.artifact_root, output / "artifact")
        self.assertEqual(client.list_calls, 4)
        self.assertEqual(client.head_calls, 4)
        self.assertEqual(client.get_calls, 1)

        detached = self.launch.root / "detached.json"
        shutil.copyfile(output / "acquisition_receipt.json", detached)
        with self.assertRaisesRegex(ValueError, "detached or aliased"):
            training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
                detached,
                training_plan=self.plan,
                staging_receipt=self.staging,
            )

    def test_resealed_equal_size_artifact_cross_splice_is_rejected(self) -> None:
        archive = self._artifact_archive()
        output, _, receipt = self._acquire(archive)
        original = (
            training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
                output / "acquisition_receipt.json",
                training_plan=self.plan,
                staging_receipt=self.staging,
            )
        )
        tampered = copy.deepcopy(receipt)
        files = tampered["artifact"]["files"]
        replacement = "f" * 64
        if files[-1]["sha256"] == replacement:
            replacement = "e" * 64
        files[-1]["sha256"] = replacement
        tampered["artifact"]["inventory_sha256"] = _document_sha256(files)
        payload = {
            key: value for key, value in tampered.items() if key != "receipt_sha256"
        }
        tampered["receipt_sha256"] = _document_sha256(payload)
        (output / "acquisition_receipt.json").write_bytes(
            strict_config.canonical_json_bytes(tampered)
        )

        with (
            mock.patch.object(
                training_artifacts,
                "_artifact_payload",
                return_value=(tampered["artifact"], original.validated_artifact),
            ) as artifact_payload,
            self.assertRaisesRegex(ValueError, "TAR file inventory differs"),
        ):
            training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
                output / "acquisition_receipt.json",
                training_plan=self.plan,
                staging_receipt=self.staging,
            )
        artifact_payload.assert_not_called()

    def test_replay_rejects_late_archive_and_artifact_directory_swaps(self) -> None:
        output, _, _ = self._acquire(self._artifact_archive(), name="late-swaps")
        receipt_path = output / "acquisition_receipt.json"
        original_artifact_payload = training_artifacts._artifact_payload
        displaced_archive = self.launch.root / "late-swap-original-model.tar.gz"

        def validate_then_swap_archive(**arguments: object) -> object:
            result = original_artifact_payload(**arguments)
            archive_path = output / "model.tar.gz"
            replacement = self.launch.root / "late-swap-replacement-model.tar.gz"
            shutil.copyfile(archive_path, replacement)
            archive_path.rename(displaced_archive)
            replacement.rename(archive_path)
            return result

        with (
            mock.patch.object(
                training_artifacts,
                "_artifact_payload",
                side_effect=validate_then_swap_archive,
            ),
            self.assertRaisesRegex(RuntimeError, "snapshot path identity changed"),
        ):
            training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
                receipt_path,
                training_plan=self.plan,
                staging_receipt=self.staging,
            )

        displaced_artifact = self.launch.root / "late-swap-original-artifact"

        def validate_then_swap_artifact(**arguments: object) -> object:
            result = original_artifact_payload(**arguments)
            artifact_root = output / "artifact"
            artifact_root.rename(displaced_artifact)
            shutil.copytree(displaced_artifact, artifact_root)
            return result

        with (
            mock.patch.object(
                training_artifacts,
                "_artifact_payload",
                side_effect=validate_then_swap_artifact,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "bundle or artifact directory was replaced",
            ),
        ):
            training_artifacts.load_and_validate_determinism_smoke_acquisition_receipt(
                receipt_path,
                training_plan=self.plan,
                staging_receipt=self.staging,
            )

    def test_truncated_version_stream_fails_and_removes_owned_incomplete_tree(self) -> None:
        archive = b"not-a-gzip-stream"
        client = _FakeS3(
            archive,
            preflight=self.preflight,
            terminal=self.terminal,
            truncate_body=True,
        )
        output = self.launch.root / "truncated"
        with (
            mock.patch.object(training_artifacts, "_require_disk_space"),
            self.assertRaisesRegex(RuntimeError, "ended before listed size"),
        ):
            training_artifacts.acquire_completed_determinism_smoke_artifact(
                client,
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=self.preflight,
                submission_receipt=self.submission,
                terminal_receipt=self.terminal,
                output_bundle=output,
            )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name("truncated.incomplete").exists())

    def test_extra_remote_version_fails_before_local_mutation(self) -> None:
        archive = b"not-read"
        client = _FakeS3(
            archive,
            preflight=self.preflight,
            terminal=self.terminal,
            extra_version=True,
        )
        output = self.launch.root / "versioned"
        with self.assertRaisesRegex(RuntimeError, "exactly one version"):
            training_artifacts.acquire_completed_determinism_smoke_artifact(
                client,
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=self.preflight,
                submission_receipt=self.submission,
                terminal_receipt=self.terminal,
                output_bundle=output,
            )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name("versioned.incomplete").exists())
        self.assertEqual(client.get_calls, 0)

    def test_postpublication_remote_drift_quarantines_without_commit_marker(self) -> None:
        archive = self._artifact_archive()
        client = _FakeS3(
            archive,
            preflight=self.preflight,
            terminal=self.terminal,
            drift_on_list_call=4,
        )
        output = self.launch.root / "drifted"
        with (
            mock.patch.object(training_artifacts, "_require_disk_space"),
            self.assertRaisesRegex(RuntimeError, "exactly one version"),
        ):
            training_artifacts.acquire_completed_determinism_smoke_artifact(
                client,
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=self.preflight,
                submission_receipt=self.submission,
                terminal_receipt=self.terminal,
                output_bundle=output,
            )
        quarantine = output.with_name("drifted.incomplete")
        self.assertFalse(output.exists())
        self.assertTrue(quarantine.is_dir())
        self.assertFalse((quarantine / "acquisition_receipt.json").exists())

    def test_archive_path_swap_between_hash_and_gzip_is_rejected(self) -> None:
        archive = self._artifact_archive()
        output = self.launch.root / "phase-swap"
        original_hash = training_artifacts._archive_size_sha256

        def hash_then_swap(
            snapshot: training_artifacts._ArchiveSnapshot,
        ) -> tuple[int, str]:
            result = original_hash(snapshot)
            displaced = snapshot.path.with_name("displaced-model.tar.gz")
            replacement = snapshot.path.with_name("replacement-model.tar.gz")
            shutil.copyfile(snapshot.path, replacement)
            snapshot.path.rename(displaced)
            replacement.rename(snapshot.path)
            return result

        with (
            mock.patch.object(
                training_artifacts,
                "_archive_size_sha256",
                side_effect=hash_then_swap,
            ),
            mock.patch.object(training_artifacts, "_require_disk_space"),
            self.assertRaisesRegex(RuntimeError, "snapshot .* identity changed"),
        ):
            training_artifacts.acquire_completed_determinism_smoke_artifact(
                _FakeS3(
                    archive,
                    preflight=self.preflight,
                    terminal=self.terminal,
                ),
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=self.preflight,
                submission_receipt=self.submission,
                terminal_receipt=self.terminal,
                output_bundle=output,
            )
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name("phase-swap.incomplete").exists())

    def test_replaced_published_directory_is_rejected_without_rollback(self) -> None:
        archive = self._artifact_archive()
        output = self.launch.root / "published-swap"
        displaced = self.launch.root / "published-swap.displaced"
        original_rename = training_artifacts._rename_no_replace

        def publish_then_replace(source: Path, target: Path) -> None:
            original_rename(source, target)
            if target == output:
                target.rename(displaced)
                target.mkdir(mode=0o700)

        with (
            mock.patch.object(
                training_artifacts,
                "_rename_no_replace",
                side_effect=publish_then_replace,
            ),
            mock.patch.object(training_artifacts, "_require_disk_space"),
            self.assertRaisesRegex(
                RuntimeError,
                "Published acquisition directory was replaced; refusing rollback",
            ),
        ):
            training_artifacts.acquire_completed_determinism_smoke_artifact(
                _FakeS3(
                    archive,
                    preflight=self.preflight,
                    terminal=self.terminal,
                ),
                training_plan=self.plan,
                staging_receipt=self.staging,
                preflight_receipt=self.preflight,
                submission_receipt=self.submission,
                terminal_receipt=self.terminal,
                output_bundle=output,
            )
        self.assertTrue(output.is_dir())
        self.assertEqual(list(output.iterdir()), [])
        self.assertTrue(displaced.is_dir())
        self.assertTrue((displaced / "acquisition_receipt.json").is_file())


class ExactArchiveEnvelopeTest(unittest.TestCase):
    def _scan(self, archive: bytes) -> tuple[dict[str, object], list[dict[str, object]]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "model.tar.gz"
        path.write_bytes(archive)
        gzip_evidence = training_artifacts._audit_single_gzip(path)
        return training_artifacts._scan_tar(
            path, gzip_evidence=gzip_evidence, extraction_root=None
        )

    def test_accepts_exact_live_libarchive_pax_name_transformations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loss_traces = root / "loss_traces"
            loss_traces.mkdir()
            (loss_traces / "manifest.json").write_text("{}\n", encoding="utf-8")
            archive = _archive_tree(root)
        self.assertIn(b"./PaxHeaders.X/loss_traces_", gzip.decompress(archive))
        self.assertIn(
            b"./PaxHeaders.X/loss_traces_manifest.json",
            gzip.decompress(archive),
        )
        summary, records = self._scan(archive)
        self.assertEqual(summary["logical_member_count"], 2)
        self.assertEqual(
            [(record["kind"], record["path"]) for record in records],
            [
                ("directory", "loss_traces"),
                ("file", "loss_traces/manifest.json"),
            ],
        )

    def test_rejects_numeric_pax_namespace_and_non_time_pax_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
            numeric = _archive_tree(root, numeric_pax_namespace=True)
            extra = _archive_tree(root, extra_pax_key=True)
        with self.assertRaisesRegex(ValueError, "exact local PAX"):
            self._scan(numeric)
        with self.assertRaisesRegex(ValueError, "time-only contract"):
            self._scan(extra)

    def test_rejects_concatenated_gzip_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.tar.gz"
            path.write_bytes(gzip.compress(b"first") + gzip.compress(b"second"))
            with self.assertRaisesRegex(ValueError, "another gzip member"):
                training_artifacts._audit_single_gzip(path)

    def test_archive_snapshot_rejects_symlink_and_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "model.tar.gz"
            archive.write_bytes(gzip.compress(b"one member"))
            symlink = root / "symlink.tar.gz"
            symlink.symlink_to(archive)
            with self.assertRaises(OSError):
                training_artifacts._audit_single_gzip(symlink)

            hardlink = root / "hardlink.tar.gz"
            os.link(archive, hardlink)
            with self.assertRaisesRegex(ValueError, "singly-linked"):
                training_artifacts._audit_single_gzip(archive)


if __name__ == "__main__":
    unittest.main()
