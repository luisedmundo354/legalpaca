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
from types import SimpleNamespace
from unittest import mock


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from processing_fold_eval import build_context, image_smoke  # noqa: E402
from retriever.evaluator import FOLD_PROCESSING_IMAGE_CONTRACT_SHA256  # noqa: E402


PROCESSING_DIR = MODERNBERT_DIR / "processing_fold_eval"
FIXED_EPOCH = 1_700_000_000
FIXED_TOOLCHAIN = {
    "builder_driver": "docker",
    "buildkit_version": "v0.16.0",
    "buildx_version": "v0.17.1",
}
EXPECTED_BASE_IMAGE = (
    "371087393859.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval@"
    "sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2"
)
EXPECTED_BASE_CONFIG_DIGEST = (
    "sha256:76c29a7f5ca0a1a36d0f8b53fe1e49f40ab199f8ff1bc594ddbb09107c7749e8"
)
EXPECTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "17",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
EXPECTED_SOURCE_INVENTORY = [
    "processing_fold_eval/__init__.py",
    "processing_fold_eval/archive_bridge.py",
    "processing_fold_eval/build_context.py",
    "processing_fold_eval/evaluate_sm.py",
    "processing_fold_eval/image_contract.json",
    "processing_fold_eval/image_smoke.py",
    "processing_fold_eval/inventory_sm.py",
    "retriever/artifacts.py",
    "retriever/evaluator.py",
    "retriever/provenance.py",
    "retriever/staged_data.py",
]
EXPECTED_PROCESSING_LAYOUT = {
    "archive_manifest_path": (
        "/opt/ml/processing/input/fold-archives/fold_archive_input_manifest.json"
    ),
    "archive_receipt_path": (
        "/opt/ml/processing/input/fold-inventory/archive_inventory.json"
    ),
    "baseline_config_path": (
        "/opt/ml/processing/input/control/evaluation_baselines.json"
    ),
    "bm25_scratch_dir": "/opt/ml/processing/work/bm25-evaluation",
    "dataset_dir": "/opt/ml/processing/input/dataset",
    "e5_pack_artifact_dir": "/opt/ml/processing/input/e5-pack",
    "e5_snapshot_dir": "/opt/ml/processing/input/e5-snapshot",
    "e5_snapshot_manifest_path": (
        "/opt/ml/processing/input/control/e5_snapshot.json"
    ),
    "evaluation_output_dir": "/opt/ml/processing/output/evaluation",
    "evaluation_plan_path": (
        "/opt/ml/processing/input/control/evaluation_plan.json"
    ),
    "evidence_output_dir": "/opt/ml/processing/output/evidence",
    "experiment_config_path": "/opt/ml/processing/input/control/experiment.json",
    "fixed_base_artifact_dir": "/opt/ml/processing/input/fixed-base",
    "fold_manifest_path": "/opt/ml/processing/input/control/folds.json",
    "image_contract_path": (
        "/opt/program/modernbert/processing_fold_eval/image_contract.json"
    ),
    "local_bindings_path": (
        "/opt/ml/processing/input/control/local_bindings.json"
    ),
    "materialization_root": "/opt/ml/processing/work/materialized",
    "output_parent": "/opt/ml/processing/output",
    "work_parent": "/opt/ml/processing/work",
}
EXPECTED_INHERITED_RUNTIME = {
    "build_identity_sha256": (
        "249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8"
    ),
    "build_manifest_path": "processing_eval/build_context_manifest.json",
    "files_sha256": (
        "96f8b4e5569404ed916cd69c4d765b3eb34cbd3f40e3eff8394e9de72f415dc4"
    ),
    "image_contract_path": "processing_eval/image_contract.json",
    "image_contract_sha256": (
        "c0dba1f1a2387bce425b6c33f83e5035d3904ccb62de0e4f1422602ead0cbca8"
    ),
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


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
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


def _record(path: str, payload: bytes) -> dict[str, object]:
    return {
        "mode": "0644",
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "type": "regular_file",
    }


def _write_runtime_file(root: Path, relative: str, payload: bytes) -> None:
    destination = root / relative
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    current = destination.parent
    while current != root:
        current.chmod(0o755)
        current = current.parent
    destination.write_bytes(payload)
    destination.chmod(0o644)


class _CommittedOverlayFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.modernbert = self.repository / "modernbert"
        self.modernbert.mkdir(parents=True)
        relative_paths = sorted(
            {
                "processing_fold_eval/Dockerfile",
                "processing_fold_eval/Dockerfile.dockerignore",
                *EXPECTED_SOURCE_INVENTORY,
            }
        )
        for relative in relative_paths:
            source = MODERNBERT_DIR / relative
            destination = self.modernbert / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            destination.chmod(0o644)
        _run_git(self.repository, "init", "--initial-branch=main")
        _run_git(self.repository, "config", "user.name", "ARR Test")
        _run_git(self.repository, "config", "user.email", "arr-test@example.invalid")
        _run_git(self.repository, "add", "--", "modernbert")
        environment = dict(os.environ)
        date = f"{FIXED_EPOCH} +0000"
        environment.update({"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})
        _run_git(
            self.repository,
            "commit",
            "--no-gpg-sign",
            "-m",
            "frozen overlay fixture",
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

    def freeze(self, name: str) -> tuple[Path, dict[str, object]]:
        destination = self.root / name
        with mock.patch.object(
            build_context,
            "_local_toolchain_identity",
            return_value=FIXED_TOOLCHAIN,
        ):
            manifest = build_context.freeze_build_context(
                self.modernbert,
                destination,
                source_parent_commit=self.source_parent_commit,
                source_parent_epoch=self.source_parent_epoch,
            )
        return destination, manifest

    def close(self) -> None:
        self.temporary.cleanup()


class _MergedRuntimeFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root_parent = Path(self.temporary.name)
        self.contract = json.loads((PROCESSING_DIR / "image_contract.json").read_bytes())
        self.contract_bytes = _canonical_bytes(self.contract)
        self.overlay_payloads = {
            relative: (
                self.contract_bytes
                if relative == "processing_fold_eval/image_contract.json"
                else f"overlay:{relative}\n".encode("utf-8")
            )
            for relative in EXPECTED_SOURCE_INVENTORY
        }
        self.manifest = {
            "files": [
                _record(relative, self.overlay_payloads[relative])
                for relative in EXPECTED_SOURCE_INVENTORY
            ]
        }
        self.manifest_bytes = _pretty_bytes({"overlay_manifest": 1})
        self.inherited_inventory = sorted(
            {
                "processing_eval/image_contract.json",
                "processing_eval/legacy_source.py",
                "retriever/artifacts.py",
                "retriever/evaluator.py",
                "retriever/provenance.py",
                "retriever/staged_data.py",
            }
        )
        self.inherited_contract = {
            "build_manifest": {
                "path": "processing_eval/build_context_manifest.json"
            },
            "source_inventory": self.inherited_inventory,
        }
        inherited_contract_bytes = _canonical_bytes(self.inherited_contract)
        self.inherited_payloads = {
            relative: (
                inherited_contract_bytes
                if relative == "processing_eval/image_contract.json"
                else f"inherited:{relative}\n".encode("utf-8")
            )
            for relative in self.inherited_inventory
        }
        self.inherited_manifest = {
            "files": [
                _record(relative, self.inherited_payloads[relative])
                for relative in self.inherited_inventory
            ]
        }
        self.inherited_manifest_bytes = _pretty_bytes({"inherited_manifest": 1})
        self.valid_root = self._create_root("valid")

    def _create_root(self, name: str) -> Path:
        root = self.root_parent / name
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        for relative, payload in self.overlay_payloads.items():
            _write_runtime_file(root, relative, payload)
        overridden = {
            "retriever/artifacts.py",
            "retriever/evaluator.py",
            "retriever/provenance.py",
            "retriever/staged_data.py",
        }
        for relative in sorted(set(self.inherited_inventory) - overridden):
            _write_runtime_file(root, relative, self.inherited_payloads[relative])
        _write_runtime_file(root, build_context.MANIFEST_RELATIVE_PATH, self.manifest_bytes)
        _write_runtime_file(
            root,
            self.inherited_contract["build_manifest"]["path"],
            self.inherited_manifest_bytes,
        )
        return root

    def clone(self, name: str) -> Path:
        destination = self.root_parent / name
        shutil.copytree(self.valid_root, destination)
        destination.chmod(0o755)
        return destination

    def validate(self, root: Path) -> None:
        image_smoke._validate_runtime_sources(
            self.contract,
            self.manifest,
            root,
            contract_bytes=self.contract_bytes,
            manifest_bytes=self.manifest_bytes,
            inherited_contract=self.inherited_contract,
            inherited_manifest=self.inherited_manifest,
            inherited_manifest_bytes=self.inherited_manifest_bytes,
        )

    def close(self) -> None:
        self.temporary.cleanup()


class ProcessingFoldImageContractTest(unittest.TestCase):
    def test_contract_exactly_binds_base_runtime_environment_sources_and_layout(self) -> None:
        contract_path = PROCESSING_DIR / "image_contract.json"
        raw = contract_path.read_bytes()
        contract = json.loads(raw)
        self.assertEqual(raw, _canonical_bytes(contract))
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            FOLD_PROCESSING_IMAGE_CONTRACT_SHA256,
        )
        self.assertEqual(
            set(contract),
            {
                "base_image",
                "build_exporter",
                "build_manifest",
                "dockerfile_frontend",
                "entrypoint",
                "environment",
                "inherited_runtime",
                "inventory_entrypoint",
                "platform",
                "processing_layout",
                "program_root",
                "schema_version",
                "source_inventory",
                "workdir",
            },
        )
        self.assertEqual(
            contract["base_image"],
            {
                "config_digest": EXPECTED_BASE_CONFIG_DIGEST,
                "digest": EXPECTED_BASE_IMAGE.rsplit("@", 1)[1],
                "uri": EXPECTED_BASE_IMAGE,
            },
        )
        self.assertEqual(contract["platform"], "linux/amd64")
        self.assertEqual(contract["environment"], EXPECTED_ENVIRONMENT)
        self.assertEqual(contract["source_inventory"], EXPECTED_SOURCE_INVENTORY)
        self.assertEqual(contract["processing_layout"], EXPECTED_PROCESSING_LAYOUT)
        self.assertEqual(contract["inherited_runtime"], EXPECTED_INHERITED_RUNTIME)
        self.assertEqual(
            contract["entrypoint"],
            [
                "/opt/conda/bin/python",
                "/opt/program/modernbert/processing_fold_eval/evaluate_sm.py",
            ],
        )
        self.assertEqual(
            contract["inventory_entrypoint"],
            [
                "/opt/conda/bin/python",
                "/opt/program/modernbert/processing_fold_eval/inventory_sm.py",
            ],
        )
        self.assertEqual(contract["program_root"], "/opt/program/modernbert")
        self.assertEqual(contract["workdir"], "/opt/program/modernbert")
        self.assertNotIn("latest", raw.decode("utf-8"))
        image_smoke._validate_contract(contract)
        loaded, paths = build_context._load_source_contract(MODERNBERT_DIR)
        self.assertEqual(loaded, contract)
        self.assertEqual(
            paths,
            sorted(
                {
                    "processing_fold_eval/Dockerfile",
                    "processing_fold_eval/Dockerfile.dockerignore",
                    *EXPECTED_SOURCE_INVENTORY,
                }
            ),
        )

    def test_dockerfile_has_two_offline_smokes_and_an_exact_allowlist(self) -> None:
        dockerfile = (PROCESSING_DIR / "Dockerfile").read_text(encoding="utf-8")
        lines = dockerfile.splitlines()
        self.assertEqual(lines[0], f"# syntax={build_context.DOCKERFILE_FRONTEND}")
        self.assertEqual(
            [line for line in lines if line.startswith("FROM ")],
            [f"FROM {EXPECTED_BASE_IMAGE}"],
        )
        logical_instructions: list[str] = []
        pending = ""
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pending += (" " if pending else "") + stripped.rstrip("\\").rstrip()
            if not stripped.endswith("\\"):
                logical_instructions.append(pending)
                pending = ""
        self.assertFalse(pending)
        run_instructions = [
            instruction
            for instruction in logical_instructions
            if instruction.startswith("RUN ")
        ]
        self.assertEqual(len(run_instructions), 2)
        self.assertTrue(
            all(instruction.startswith("RUN --network=none ") for instruction in run_instructions)
        )
        self.assertIn(
            f'io.arr-retrieval-fold-eval.base-config-digest="{EXPECTED_BASE_CONFIG_DIGEST}"',
            dockerfile,
        )
        self.assertIn(
            'ENTRYPOINT ["/opt/conda/bin/python", '
            '"/opt/program/modernbert/processing_fold_eval/evaluate_sm.py"]',
            dockerfile,
        )
        for forbidden in ("ARG BASE_IMAGE", "${BASE_IMAGE}", " apt ", " pip ", "curl "):
            self.assertNotIn(forbidden, dockerfile)
        expected_allowlist = [
            "**",
            *[f"!{relative}" for relative in EXPECTED_SOURCE_INVENTORY],
        ]
        expected_allowlist.insert(
            expected_allowlist.index("!processing_fold_eval/evaluate_sm.py"),
            "!processing_fold_eval/build_context_manifest.json",
        )
        expected_allowlist.insert(
            expected_allowlist.index("!processing_fold_eval/evaluate_sm.py"),
            "!processing_fold_eval/Dockerfile",
        )
        dockerignore = (PROCESSING_DIR / "Dockerfile.dockerignore").read_text(
            encoding="utf-8"
        )
        self.assertEqual(dockerignore.splitlines(), expected_allowlist)

    def test_freeze_is_reproducible_and_validation_rejects_tree_mutation(self) -> None:
        fixture = _CommittedOverlayFixture()
        try:
            first_root, first = fixture.freeze("first")
            second_root, second = fixture.freeze("second")
            self.assertEqual(first, second)
            self.assertEqual(build_context.validate_frozen_build_context(first_root), first)
            self.assertEqual(build_context.validate_frozen_build_context(second_root), second)
            expected_paths = sorted(
                {
                    "processing_fold_eval/Dockerfile",
                    "processing_fold_eval/Dockerfile.dockerignore",
                    *EXPECTED_SOURCE_INVENTORY,
                }
            )
            self.assertEqual([record["path"] for record in first["files"]], expected_paths)

            wrong_root, _ = fixture.freeze("wrong-bytes")
            (wrong_root / "processing_fold_eval/archive_bridge.py").write_bytes(b"mutated\n")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                build_context.validate_frozen_build_context(wrong_root)

            extra_root, _ = fixture.freeze("extra")
            (extra_root / "unexpected.bin").write_bytes(b"unexpected")
            (extra_root / "unexpected.bin").chmod(0o644)
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                build_context.validate_frozen_build_context(extra_root)

            resealed_root, resealed = fixture.freeze("resealed-extra")
            extra_payload = b"resealed unexpected bytes"
            extra_path = resealed_root / "unexpected.bin"
            extra_path.write_bytes(extra_payload)
            extra_path.chmod(0o644)
            resealed["files"] = sorted(
                [
                    *resealed["files"],
                    {
                        "mode": build_context.FILE_MODE,
                        "path": "unexpected.bin",
                        "sha256": hashlib.sha256(extra_payload).hexdigest(),
                        "size": len(extra_payload),
                        "type": "regular_file",
                    },
                ],
                key=lambda record: record["path"],
            )
            resealed["files_sha256"] = build_context._sha256_bytes(
                build_context._canonical_json(resealed["files"]).encode("utf-8")
            )
            resealed_identity = build_context._sha256_bytes(
                build_context._canonical_json(
                    build_context._identity_payload(resealed)
                ).encode("utf-8")
            )
            resealed["build_identity_sha256"] = resealed_identity
            resealed["content_tag"] = f"build-sha256-{resealed_identity}"
            (resealed_root / build_context.MANIFEST_RELATIVE_PATH).write_bytes(
                build_context._canonical_pretty_bytes(resealed)
            )
            with self.assertRaisesRegex(ValueError, "exact source allowlist"):
                build_context.validate_frozen_build_context(resealed_root)

            hardlinked_root, _ = fixture.freeze("hardlinked")
            source = hardlinked_root / "retriever/artifacts.py"
            os.link(source, fixture.root / "outside-hardlink")
            with self.assertRaisesRegex(ValueError, "unsafe file"):
                build_context.validate_frozen_build_context(hardlinked_root)
        finally:
            fixture.close()

    def test_build_and_offline_smoke_commands_are_closed_and_non_networked(self) -> None:
        fixture = _CommittedOverlayFixture()
        try:
            frozen, manifest = fixture.freeze("frozen")
            command, image_name = build_context._buildx_command(
                frozen,
                fixture.root / "metadata.json",
                manifest=manifest,
                build_replica=1,
            )
            self.assertEqual(command[:3], ["docker", "buildx", "build"])
            self.assertIn("--pull", command)
            self.assertIn("--no-cache", command)
            self.assertIn("--provenance=false", command)
            self.assertIn("--sbom=false", command)
            self.assertEqual(command[-1], str(frozen))
            self.assertEqual(
                image_name,
                f"arr-retrieval-fold-eval:{manifest['content_tag']}-build1",
            )

            smoke_payload = {
                "build_context": {
                    "build_identity_sha256": manifest["build_identity_sha256"]
                }
            }
            completed = subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=0,
                stdout=_canonical_bytes(smoke_payload).decode("utf-8"),
                stderr="",
            )
            with mock.patch.object(
                build_context.subprocess,
                "run",
                return_value=completed,
            ) as run:
                self.assertEqual(
                    build_context._run_offline_image_smoke(
                        image_name,
                        manifest=manifest,
                    ),
                    smoke_payload,
                )
            smoke_command = run.call_args.args[0]
            self.assertEqual(
                smoke_command[:8],
                [
                    "docker",
                    "run",
                    "--pull=never",
                    "--rm",
                    "--network",
                    "none",
                    "--read-only",
                    "--entrypoint",
                ],
            )
            self.assertNotIn("--volume", smoke_command)
            self.assertNotIn("-v", smoke_command)
        finally:
            fixture.close()

    def test_merged_runtime_rejects_extra_hardlink_special_and_mutated_sources(self) -> None:
        fixture = _MergedRuntimeFixture()
        try:
            fixture.validate(fixture.valid_root)

            extra = fixture.clone("extra")
            _write_runtime_file(extra, "unexpected.bin", b"extra")
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                fixture.validate(extra)

            hardlinked = fixture.clone("hardlinked")
            os.link(hardlinked / "retriever/artifacts.py", fixture.root_parent / "anchor")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                fixture.validate(hardlinked)

            special = fixture.clone("special")
            os.mkfifo(special / "unexpected.fifo", mode=0o644)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                fixture.validate(special)

            inherited_mutation = fixture.clone("inherited-mutation")
            inherited_source = inherited_mutation / "processing_eval/legacy_source.py"
            inherited_source.write_bytes(b"x" * inherited_source.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, "Inherited runtime source identity"):
                fixture.validate(inherited_mutation)

            overlay_mutation = fixture.clone("overlay-mutation")
            overlay_source = overlay_mutation / "processing_fold_eval/archive_bridge.py"
            overlay_source.write_bytes(b"x" * overlay_source.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, "Overlay runtime source identity"):
                fixture.validate(overlay_mutation)
        finally:
            fixture.close()

    def test_module_origins_must_resolve_to_the_overlay_files(self) -> None:
        origins = image_smoke._validate_module_origins(MODERNBERT_DIR)
        self.assertEqual(
            origins["retriever.evaluator"],
            str(MODERNBERT_DIR / "retriever/evaluator.py"),
        )
        wrong = SimpleNamespace(
            __file__=str(MODERNBERT_DIR / "elsewhere.py"),
            __spec__=SimpleNamespace(origin=str(MODERNBERT_DIR / "elsewhere.py")),
        )
        with mock.patch.object(
            image_smoke.importlib,
            "import_module",
            return_value=wrong,
        ), self.assertRaisesRegex(RuntimeError, "module origin changed"):
            image_smoke._validate_module_origins(MODERNBERT_DIR)

    def test_reproducible_build_comparator_rejects_spliced_receipts(self) -> None:
        context_identity = "1" * 64
        content_tag = f"build-sha256-{context_identity}"
        common = {
            "build_context_files_sha256": "2" * 64,
            "build_context_identity_sha256": context_identity,
            "config_digest": "sha256:" + "3" * 64,
            "content_tag": content_tag,
            "image_digest": "sha256:" + "4" * 64,
            "local_image_identity_sha256": "5" * 64,
            "manifest_media_type": build_context.DOCKER_MANIFEST_MEDIA_TYPE,
            "offline_smoke_sha256": "6" * 64,
        }
        first = {
            **common,
            "build_ref": "default/default/replica1",
            "build_replica": 1,
            "image_name": (
                f"docker.io/library/{build_context.LOCAL_IMAGE_REPOSITORY}:"
                f"{content_tag}-build1"
            ),
        }
        second = {
            **common,
            "build_ref": "default/default/replica2",
            "build_replica": 2,
            "image_name": (
                f"docker.io/library/{build_context.LOCAL_IMAGE_REPOSITORY}:"
                f"{content_tag}-build2"
            ),
        }
        identity = build_context.validate_reproducible_builds(first, second)
        self.assertEqual(identity["config_digest"], common["config_digest"])
        self.assertEqual(identity["image_digest"], common["image_digest"])

        with self.assertRaisesRegex(ValueError, "schema changed"):
            build_context.validate_reproducible_builds(
                {**first, "unexpected": "value"},
                {**second, "unexpected": "value"},
            )

        for key, malformed in (
            ("build_context_files_sha256", "bad"),
            ("build_context_identity_sha256", "bad"),
            ("config_digest", "bad"),
            ("image_digest", "sha256:bad"),
            ("local_image_identity_sha256", "bad"),
            ("manifest_media_type", "application/unknown"),
            ("offline_smoke_sha256", "bad"),
        ):
            with self.subTest(malformed_key=key), self.assertRaises(ValueError):
                build_context.validate_reproducible_builds(
                    {**first, key: malformed},
                    {**second, key: malformed},
                )

        wrong_common_identity = "7" * 64
        with self.assertRaisesRegex(ValueError, "content tag"):
            build_context.validate_reproducible_builds(
                {
                    **first,
                    "build_context_identity_sha256": wrong_common_identity,
                },
                {
                    **second,
                    "build_context_identity_sha256": wrong_common_identity,
                },
            )

        reused_ref = {**second, "build_ref": first["build_ref"]}
        with self.assertRaisesRegex(RuntimeError, "reused one BuildKit reference"):
            build_context.validate_reproducible_builds(first, reused_ref)

        with self.assertRaisesRegex(ValueError, "replica 1"):
            build_context.validate_reproducible_builds(second, first)

        wrong_name = {**second, "image_name": second["image_name"] + "-wrong"}
        with self.assertRaisesRegex(RuntimeError, "names/ordinals"):
            build_context.validate_reproducible_builds(first, wrong_name)

        malformed_first_ref = {**first, "build_ref": "not-a-build-reference"}
        malformed_second_ref = {**second, "build_ref": "also-invalid"}
        with self.assertRaisesRegex((ValueError, RuntimeError), "reference"):
            build_context.validate_reproducible_builds(
                malformed_first_ref,
                malformed_second_ref,
            )

        for key, value in (
            ("build_context_files_sha256", "7" * 64),
            ("build_context_identity_sha256", "8" * 64),
            ("local_image_identity_sha256", "9" * 64),
            ("config_digest", "sha256:" + "a" * 64),
            ("image_digest", "sha256:" + "b" * 64),
            ("offline_smoke_sha256", "c" * 64),
        ):
            with self.subTest(spliced_key=key):
                spliced = {**second, key: value}
                with self.assertRaises((ValueError, RuntimeError)):
                    build_context.validate_reproducible_builds(first, spliced)


if __name__ == "__main__":
    unittest.main()
