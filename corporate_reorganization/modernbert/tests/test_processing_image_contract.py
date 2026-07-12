from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from processing_eval.build_context import (  # noqa: E402
    BASE_IMAGE_URI,
    DOCKERFILE_FRONTEND,
    MANIFEST_RELATIVE_PATH,
    _buildx_command,
    _local_toolchain_identity,
    _validate_local_image,
    freeze_build_context,
    load_build_context_manifest,
    validate_frozen_build_context,
)
from processing_eval.image_smoke import (  # noqa: E402
    _validate_runtime_sources,
    validate_image_runtime,
)


PROCESSING_DIR = MODERNBERT_DIR / "processing_eval"
CONTRACT_SHA256 = "c0dba1f1a2387bce425b6c33f83e5035d3904ccb62de0e4f1422602ead0cbca8"
FIXED_EPOCH = 1_700_000_000
FIXED_TOOLCHAIN = {
    "builder_driver": "docker",
    "buildkit_version": "v0.16.0",
    "buildx_version": "v0.17.1",
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _run_git(root: Path, *arguments: str, environment: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
    )
    return completed.stdout.strip()


class _CommittedFixture:
    def __init__(self, *, leave_dockerfile_untracked: bool = False) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.modernbert = self.repository / "modernbert"
        self.processing = self.modernbert / "processing_eval"
        self.processing.mkdir(parents=True)
        contract = json.loads((PROCESSING_DIR / "image_contract.json").read_bytes())
        contract["source_inventory"] = [
            "processing_eval/build_context.py",
            "processing_eval/image_contract.json",
        ]
        files = {
            "processing_eval/Dockerfile": (PROCESSING_DIR / "Dockerfile").read_bytes(),
            "processing_eval/Dockerfile.dockerignore": (
                PROCESSING_DIR / "Dockerfile.dockerignore"
            ).read_bytes(),
            "processing_eval/build_context.py": (
                PROCESSING_DIR / "build_context.py"
            ).read_bytes(),
            "processing_eval/build_requirements.lock": (
                PROCESSING_DIR / "build_requirements.lock"
            ).read_bytes(),
            "processing_eval/image_contract.json": _canonical_bytes(contract),
            "processing_eval/requirements.lock": (
                PROCESSING_DIR / "requirements.lock"
            ).read_bytes(),
        }
        for relative, payload in files.items():
            destination = self.modernbert / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            destination.chmod(0o644)
        _run_git(self.repository, "init", "--initial-branch=main")
        _run_git(self.repository, "config", "user.name", "ARR Test")
        _run_git(self.repository, "config", "user.email", "arr-test@example.invalid")
        add_paths = [
            str(path.relative_to(self.repository))
            for path in sorted(self.modernbert.rglob("*"))
            if path.is_file()
            and not (
                leave_dockerfile_untracked
                and path == self.processing / "Dockerfile"
            )
        ]
        _run_git(self.repository, "add", "--", *add_paths)
        environment = dict(os.environ)
        date = f"{FIXED_EPOCH} +0000"
        environment.update({"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})
        _run_git(
            self.repository,
            "commit",
            "--no-gpg-sign",
            "-m",
            "frozen fixture",
            environment=environment,
        )
        self.source_parent_commit = _run_git(self.repository, "rev-parse", "HEAD")
        self.source_parent_epoch = int(
            _run_git(
                self.repository,
                "show",
                "-s",
                "--format=%ct",
                self.source_parent_commit,
            )
        )

    def freeze(self, name: str = "frozen") -> tuple[Path, dict[str, object]]:
        output = self.root / name
        with mock.patch(
            "processing_eval.build_context._local_toolchain_identity",
            return_value=FIXED_TOOLCHAIN,
        ):
            manifest = freeze_build_context(
                self.modernbert,
                output,
                source_parent_commit=self.source_parent_commit,
                source_parent_epoch=self.source_parent_epoch,
            )
        return output, manifest

    def close(self) -> None:
        self.temporary.cleanup()


class ProcessingImageContractTest(unittest.TestCase):
    def test_contract_lock_and_dockerfile_are_immutable_and_strict(self) -> None:
        contract_path = PROCESSING_DIR / "image_contract.json"
        raw_contract = contract_path.read_bytes()
        contract = json.loads(raw_contract)
        self.assertEqual(raw_contract, _canonical_bytes(contract))
        self.assertEqual(hashlib.sha256(raw_contract).hexdigest(), CONTRACT_SHA256)
        self.assertEqual(contract["platform"], "linux/amd64")
        self.assertEqual(contract["dockerfile_frontend"], DOCKERFILE_FRONTEND)
        self.assertEqual(contract["base_image"]["uri"], BASE_IMAGE_URI)
        self.assertEqual(
            contract["build_exporter"],
            {
                "compression": "gzip",
                "compression_level": 6,
                "force_compression": False,
                "oci_mediatypes": False,
                "provenance": False,
                "push": False,
                "rewrite_timestamp": True,
                "sbom": False,
                "type": "image",
                "unpack": False,
            },
        )
        self.assertEqual(contract["build_manifest"]["path"], MANIFEST_RELATIVE_PATH)
        self.assertEqual(contract["build_manifest"]["file_mode"], "0644")
        self.assertEqual(contract["build_manifest"]["directory_mode"], "0755")
        self.assertEqual(
            contract["java"]["version_output_sha256"],
            "64dbcaf74f7772c14d5614c83acefd0aba65da9f90694b8815af908ff6bcf7f1",
        )
        self.assertEqual(contract["sparse_runtime"]["pyserini_installed_file_count"], 161)
        self.assertEqual(
            contract["sparse_runtime"]["pyserini_installed_tree_sha256"],
            "c8a6c1ae730c19a91bd091f4a282b29008f3360396b5cda2431f0f712e3e4f56",
        )
        self.assertEqual(contract["sparse_runtime"]["pyjnius_installed_file_count"], 16)
        self.assertEqual(
            contract["sparse_runtime"]["pyjnius_installed_tree_sha256"],
            "7f2411e7c3f6baf8eb75fc466e1f8be1720b9736bbd72d7700b21515ffab23c0",
        )
        self.assertIn("processing_eval/build_base_modernbert.py", contract["source_inventory"])
        self.assertIn("processing_eval/build_context.py", contract["source_inventory"])
        self.assertNotIn("latest", json.dumps(contract))

        dockerfile = (PROCESSING_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(dockerfile.splitlines()[0], f"# syntax={DOCKERFILE_FRONTEND}")
        from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        self.assertEqual(len(from_lines), 3)
        self.assertTrue(all(line.split()[1] == BASE_IMAGE_URI for line in from_lines))
        self.assertNotIn("ARG BASE_IMAGE", dockerfile)
        self.assertNotIn("${BASE_IMAGE}", dockerfile)
        self.assertNotIn("BUILD_CONTEXT_SHA256", dockerfile)
        self.assertIn("ARG BUILD_IDENTITY_SHA256", dockerfile)
        self.assertIn("ARG SOURCE_PARENT_COMMIT", dockerfile)
        self.assertIn("ARG SOURCE_PARENT_EPOCH", dockerfile)
        self.assertIn("ARG SOURCE_PARENT_RFC3339", dockerfile)
        self.assertNotIn("org.opencontainers.image.revision", dockerfile)
        self.assertIn("processing_eval/build_context_manifest.json", dockerfile)
        self.assertIn("processing_eval/build_base_modernbert.py", dockerfile)
        self.assertIn(
            "ADD --checksum=sha256:" + contract["java"]["archive_sha256"],
            dockerfile,
        )
        self.assertIn("--no-deps", dockerfile)
        self.assertIn("--require-hashes", dockerfile)
        self.assertIn("--no-build-isolation", dockerfile)
        self.assertNotIn("apt ", dockerfile)
        self.assertNotIn("latest", dockerfile)
        self.assertNotIn("COPY /opt/ml", dockerfile)
        self.assertIn(
            'ENTRYPOINT ["/opt/conda/bin/python", '
            '"/opt/program/modernbert/processing_eval/evaluate_sm.py"]',
            dockerfile,
        )
        self.assertIn(
            'LABEL org.opencontainers.image.base.digest="'
            + contract["base_image"]["digest"]
            + '" \\\n'
            '      io.arr-retrieval.build-identity-sha256="${BUILD_IDENTITY_SHA256}" \\\n'
            '      io.arr-retrieval.source-parent-commit="${SOURCE_PARENT_COMMIT}" \\\n'
            '      io.arr-retrieval.source-parent-epoch="${SOURCE_PARENT_EPOCH}" \\\n'
            '      io.arr-retrieval.source-parent-rfc3339="${SOURCE_PARENT_RFC3339}"',
            dockerfile,
        )
        dockerignore = (PROCESSING_DIR / "Dockerfile.dockerignore").read_text(
            encoding="utf-8"
        )
        self.assertIn("!processing_eval/build_context_manifest.json", dockerignore)
        self.assertIn("!processing_eval/build_base_modernbert.py", dockerignore)
        lock = (PROCESSING_DIR / "requirements.lock").read_text(encoding="utf-8")
        self.assertIn(contract["sparse_runtime"]["pyserini_sdist_sha256"], lock)
        self.assertIn(contract["sparse_runtime"]["pyjnius_wheel_sha256"], lock)
        self.assertNotIn(">=", lock)

    def test_freeze_is_stable_complete_and_detached_from_later_live_mutation(self) -> None:
        fixture = _CommittedFixture()
        try:
            first_root, first = fixture.freeze("first")
            second_root, second = fixture.freeze("second")
            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], 1)
            self.assertEqual(
                first["content_tag"],
                f"build-sha256-{first['build_identity_sha256']}",
            )
            paths = [record["path"] for record in first["files"]]
            self.assertEqual(paths, sorted(set(paths)))
            self.assertIn("processing_eval/Dockerfile", paths)
            self.assertIn("processing_eval/build_context.py", paths)
            self.assertEqual(validate_frozen_build_context(first_root), first)
            self.assertEqual(validate_frozen_build_context(second_root), second)
            unrelated = fixture.repository / "later-unrelated.txt"
            unrelated.write_text("later commit\n", encoding="utf-8")
            _run_git(fixture.repository, "add", "--", "later-unrelated.txt")
            environment = dict(os.environ)
            later_date = f"{FIXED_EPOCH + 60} +0000"
            environment.update(
                {"GIT_AUTHOR_DATE": later_date, "GIT_COMMITTER_DATE": later_date}
            )
            _run_git(
                fixture.repository,
                "commit",
                "--no-gpg-sign",
                "-m",
                "later unrelated commit",
                environment=environment,
            )
            third_root, third = fixture.freeze("after-parent-commit")
            self.assertEqual(third, first)
            self.assertEqual(validate_frozen_build_context(third_root), first)
            frozen_payload = (first_root / "processing_eval/build_context.py").read_bytes()
            (fixture.processing / "build_context.py").write_text(
                "post-freeze mutation\n", encoding="utf-8"
            )
            self.assertEqual(
                (first_root / "processing_eval/build_context.py").read_bytes(),
                frozen_payload,
            )
            self.assertEqual(validate_frozen_build_context(first_root), first)
        finally:
            fixture.close()

    def test_symlink_modernbert_root_is_rejected(self) -> None:
        fixture = _CommittedFixture()
        try:
            linked = fixture.root / "linked-modernbert"
            linked.symlink_to(fixture.modernbert, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                freeze_build_context(
                    linked,
                    fixture.root / "output",
                    source_parent_commit=fixture.source_parent_commit,
                    source_parent_epoch=fixture.source_parent_epoch,
                )
        finally:
            fixture.close()

    def test_uncommitted_sources_are_exactly_bound_by_the_frozen_manifest(self) -> None:
        dirty = _CommittedFixture()
        try:
            payload = b"uncommitted but exactly frozen\n"
            (dirty.processing / "build_context.py").write_bytes(payload)
            root, manifest = dirty.freeze()
            records = {record["path"]: record for record in manifest["files"]}
            self.assertEqual(
                records["processing_eval/build_context.py"]["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual((root / "processing_eval/build_context.py").read_bytes(), payload)
        finally:
            dirty.close()
        untracked = _CommittedFixture(leave_dockerfile_untracked=True)
        try:
            root, manifest = untracked.freeze()
            records = {record["path"]: record for record in manifest["files"]}
            self.assertIn("processing_eval/Dockerfile", records)
            self.assertEqual(
                (root / "processing_eval/Dockerfile").read_bytes(),
                (untracked.processing / "Dockerfile").read_bytes(),
            )
        finally:
            untracked.close()

    def test_malformed_build_identifiers_are_rejected(self) -> None:
        fixture = _CommittedFixture()
        try:
            cases = (
                {"source_parent_commit": "HEAD"},
                {"source_parent_epoch": fixture.source_parent_epoch + 1},
            )
            for index, override in enumerate(cases):
                arguments = {
                    "source_parent_commit": fixture.source_parent_commit,
                    "source_parent_epoch": fixture.source_parent_epoch,
                    **override,
                }
                with self.subTest(override=override), self.assertRaises(ValueError):
                    freeze_build_context(
                        fixture.modernbert,
                        fixture.root / f"invalid-{index}",
                        **arguments,
                    )
        finally:
            fixture.close()

    def test_actual_toolchain_output_is_strictly_parsed_and_driver_bound(self) -> None:
        malformed = subprocess.CompletedProcess(
            args=["docker", "buildx", "version"],
            returncode=0,
            stdout="buildx 0.33\n",
            stderr="",
        )
        with mock.patch(
            "processing_eval.build_context.subprocess.run", return_value=malformed
        ), self.assertRaisesRegex(RuntimeError, "Unexpected docker buildx version"):
            _local_toolchain_identity()
        buildx = subprocess.CompletedProcess(
            args=["docker", "buildx", "version"],
            returncode=0,
            stdout=(
                "github.com/docker/buildx v0.33.0 "
                "f7897eba028583e0071642db3c011e860444f8cf\n"
            ),
            stderr="",
        )
        wrong_driver = subprocess.CompletedProcess(
            args=["docker", "buildx", "inspect", "--bootstrap"],
            returncode=0,
            stdout=(
                "Name: test\nDriver: docker-container\n"
                "BuildKit version: v0.29.0\n"
            ),
            stderr="",
        )
        with mock.patch(
            "processing_eval.build_context.subprocess.run",
            side_effect=(buildx, wrong_driver),
        ), self.assertRaisesRegex(RuntimeError, "exact docker driver"):
            _local_toolchain_identity()

    def test_build_wrapper_constructs_the_only_allowed_exporter_command(self) -> None:
        fixture = _CommittedFixture()
        try:
            frozen, manifest = fixture.freeze()
            metadata = fixture.root / "build-metadata.json"
            command, image_name = _buildx_command(
                frozen,
                metadata,
                manifest=manifest,
                build_replica=1,
            )
            self.assertEqual(
                image_name,
                f"arr-retrieval-eval:{manifest['content_tag']}-build1",
            )
            output = command[command.index("--output") + 1]
            self.assertEqual(
                output,
                "type=image,"
                f"name={image_name},"
                "push=false,rewrite-timestamp=true,unpack=false,"
                "compression=gzip,compression-level=6,force-compression=false,"
                "oci-mediatypes=false",
            )
            self.assertEqual(command[-1], str(frozen))
            self.assertEqual(
                command[command.index("--metadata-file") + 1], str(metadata)
            )
            for name, value in (
                ("SOURCE_PARENT_COMMIT", manifest["source_parent_commit"]),
                ("SOURCE_PARENT_EPOCH", manifest["source_parent_epoch"]),
                ("SOURCE_PARENT_RFC3339", manifest["source_parent_rfc3339"]),
                ("BUILD_IDENTITY_SHA256", manifest["build_identity_sha256"]),
            ):
                self.assertIn(f"{name}={value}", command)
        finally:
            fixture.close()

    def test_local_image_inspection_binds_config_environment_and_labels(self) -> None:
        fixture = _CommittedFixture()
        try:
            frozen, manifest = fixture.freeze()
            contract = json.loads(
                (frozen / "processing_eval/image_contract.json").read_bytes()
            )
            config_digest = "sha256:" + "1" * 64
            image_digest = "sha256:" + "2" * 64
            labels = {
                "io.arr-retrieval.build-identity-sha256": manifest[
                    "build_identity_sha256"
                ],
                "io.arr-retrieval.source-parent-commit": manifest[
                    "source_parent_commit"
                ],
                "io.arr-retrieval.source-parent-epoch": str(
                    manifest["source_parent_epoch"]
                ),
                "io.arr-retrieval.source-parent-rfc3339": manifest[
                    "source_parent_rfc3339"
                ],
                "org.opencontainers.image.base.digest": contract["base_image"][
                    "digest"
                ],
            }
            environment = [
                f"{name}={value}"
                for name, value in {
                    **contract["environment"],
                    "SOURCE_DATE_EPOCH": str(manifest["source_parent_epoch"]),
                }.items()
            ]
            inspected = [
                {
                    "Architecture": "amd64",
                    "Config": {
                        "Entrypoint": contract["entrypoint"],
                        "Env": environment,
                        "Labels": labels,
                        "WorkingDir": contract["workdir"],
                    },
                    "Id": image_digest,
                    "Os": "linux",
                    "RepoDigests": [
                        f"arr-retrieval-eval@{image_digest}"
                    ],
                }
            ]
            completed = subprocess.CompletedProcess(
                args=["docker", "image", "inspect"],
                returncode=0,
                stdout=json.dumps(inspected),
                stderr="",
            )
            with mock.patch(
                "processing_eval.build_context.subprocess.run", return_value=completed
            ):
                identity = _validate_local_image(
                    "arr-retrieval-eval:test",
                    manifest=manifest,
                    contract=contract,
                    build_metadata={
                        "config_digest": config_digest,
                        "image_digest": image_digest,
                    },
                )
            self.assertEqual(identity["labels"], labels)
            inspected[0]["Id"] = config_digest
            completed.stdout = json.dumps(inspected)
            with mock.patch(
                "processing_eval.build_context.subprocess.run", return_value=completed
            ), self.assertRaisesRegex(RuntimeError, "identity/platform"):
                _validate_local_image(
                    "arr-retrieval-eval:test",
                    manifest=manifest,
                    contract=contract,
                    build_metadata={
                        "config_digest": config_digest,
                        "image_digest": image_digest,
                    },
                )
            inspected[0]["Id"] = image_digest
            inspected[0]["Config"]["Labels"][
                "io.arr-retrieval.build-identity-sha256"
            ] = "wrong"
            completed.stdout = json.dumps(inspected)
            with mock.patch(
                "processing_eval.build_context.subprocess.run", return_value=completed
            ), self.assertRaisesRegex(RuntimeError, "provenance labels"):
                _validate_local_image(
                    "arr-retrieval-eval:test",
                    manifest=manifest,
                    contract=contract,
                    build_metadata={
                        "config_digest": config_digest,
                        "image_digest": image_digest,
                    },
                )
        finally:
            fixture.close()

    def test_executable_mode_change_is_rejected(self) -> None:
        fixture = _CommittedFixture()
        try:
            source = fixture.processing / "build_context.py"
            source.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "became executable"):
                fixture.freeze()
        finally:
            fixture.close()

    def test_frozen_wrong_bytes_and_extra_arbitrary_file_are_rejected(self) -> None:
        fixture = _CommittedFixture()
        try:
            wrong_root, _ = fixture.freeze("wrong-bytes")
            target = wrong_root / "processing_eval/build_context.py"
            target.write_bytes(b"wrong bytes\n")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                validate_frozen_build_context(wrong_root)

            wrong_mode_root, _ = fixture.freeze("wrong-mode")
            (wrong_mode_root / "processing_eval/build_context.py").chmod(0o600)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                validate_frozen_build_context(wrong_mode_root)

            extra_root, _ = fixture.freeze("extra-file")
            extra = extra_root / "arbitrary.bin"
            extra.write_bytes(b"not allowlisted")
            extra.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                validate_frozen_build_context(extra_root)
        finally:
            fixture.close()

    def test_failed_post_publication_readback_retracts_output_and_allows_retry(self) -> None:
        fixture = _CommittedFixture()
        try:
            output = fixture.root / "frozen"

            def injected_readback(path: Path) -> dict[str, object]:
                if Path(path) == output:
                    raise RuntimeError("injected final readback failure")
                return validate_frozen_build_context(path)

            with mock.patch(
                "processing_eval.build_context.validate_frozen_build_context",
                side_effect=injected_readback,
            ), self.assertRaisesRegex(RuntimeError, "injected final readback failure"):
                fixture.freeze()
            self.assertFalse(output.exists())
            self.assertFalse((fixture.root / ".frozen.incomplete").exists())
            diagnostics = list(fixture.root.glob(".frozen.invalid.*"))
            self.assertEqual(len(diagnostics), 1)
            self.assertTrue((diagnostics[0] / "context").is_dir())
            self.assertFalse(
                (diagnostics[0] / "context" / MANIFEST_RELATIVE_PATH).exists()
            )
            retry_root, retry_manifest = fixture.freeze()
            self.assertEqual(validate_frozen_build_context(retry_root), retry_manifest)
        finally:
            fixture.close()

    def test_lost_incomplete_creation_race_never_deletes_competing_tree(self) -> None:
        fixture = _CommittedFixture()
        try:
            output = fixture.root / "frozen"
            incomplete = fixture.root / ".frozen.incomplete"
            incomplete.mkdir()
            sentinel = incomplete / "sentinel"
            sentinel.write_bytes(b"other freezer")
            with mock.patch(
                "processing_eval.build_context._require_absent_output",
                return_value=(output, incomplete),
            ), self.assertRaises(FileExistsError):
                fixture.freeze()
            self.assertEqual(sentinel.read_bytes(), b"other freezer")
            self.assertFalse(output.exists())
        finally:
            fixture.close()

    def test_runtime_source_validation_rejects_bytes_mode_symlink_and_any_extra_file(self) -> None:
        fixture = _CommittedFixture()
        try:
            frozen_root, manifest = fixture.freeze()
            contract = json.loads(
                (frozen_root / "processing_eval/image_contract.json").read_bytes()
            )

            def runtime_root(name: str) -> Path:
                root = fixture.root / name
                root.mkdir(mode=0o755)
                for relative in [*contract["source_inventory"], MANIFEST_RELATIVE_PATH]:
                    source = frozen_root / relative
                    destination = root / relative
                    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                    destination.parent.chmod(0o755)
                    shutil.copyfile(source, destination)
                    destination.chmod(0o644)
                return root

            valid = runtime_root("runtime-valid")
            _validate_runtime_sources(contract, manifest, valid)

            wrong_bytes = runtime_root("runtime-wrong-bytes")
            (wrong_bytes / "processing_eval/build_context.py").write_bytes(b"wrong\n")
            with self.assertRaisesRegex(RuntimeError, "source identity changed"):
                _validate_runtime_sources(contract, manifest, wrong_bytes)

            wrong_mode = runtime_root("runtime-wrong-mode")
            (wrong_mode / "processing_eval/build_context.py").chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "source identity changed"):
                _validate_runtime_sources(contract, manifest, wrong_mode)

            symlinked = runtime_root("runtime-symlink")
            target = symlinked / "processing_eval/build_context.py"
            target.unlink()
            target.symlink_to("image_contract.json")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                _validate_runtime_sources(contract, manifest, symlinked)

            extra = runtime_root("runtime-extra")
            (extra / "arbitrary.data").write_bytes(b"extra")
            (extra / "arbitrary.data").chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                _validate_runtime_sources(contract, manifest, extra)
        finally:
            fixture.close()

    @unittest.skipUnless(
        os.environ.get("ARR_RUN_PROCESSING_IMAGE_SMOKE") == "1",
        "ARR_RUN_PROCESSING_IMAGE_SMOKE=1 is required",
    )
    def test_runtime_matches_complete_image_contract(self) -> None:
        runtime_processing_dir = Path("/opt/program/modernbert/processing_eval")
        manifest_path = runtime_processing_dir / "build_context_manifest.json"
        manifest = load_build_context_manifest(manifest_path)
        payload = validate_image_runtime(
            runtime_processing_dir / "image_contract.json",
            build_manifest_path=manifest_path,
            expected_build_identity_sha256=manifest["build_identity_sha256"],
            expected_source_parent_commit=manifest["source_parent_commit"],
            expected_source_parent_epoch=str(manifest["source_parent_epoch"]),
            expected_source_parent_rfc3339=manifest["source_parent_rfc3339"],
        )
        self.assertEqual(payload["image_contract_sha256"], CONTRACT_SHA256)
        self.assertEqual(payload["sparse_runtime"]["pyserini"], "1.5.0")
        self.assertIn("build_identity_sha256", payload["build_context"])


if __name__ == "__main__":
    unittest.main()
