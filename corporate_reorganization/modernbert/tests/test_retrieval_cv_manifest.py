from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from corporate_reorganization.modernbert.experiments.retrieval_cv import config, manifest
from corporate_reorganization.modernbert.tests.test_retrieval_cv_config import (
    valid_aws_config,
    valid_scientific_config,
)


class SourceBundleTest(unittest.TestCase):
    def test_scientific_source_claims_match_exact_tracked_files(self) -> None:
        source = Path(__file__).resolve().parents[1]
        scientific, _ = config.load_scientific_config(
            source / "experiments/retrieval_cv/configs/orchestration.json"
        )
        identities = manifest.validate_scientific_source_claims(source, scientific)
        self.assertEqual(
            identities["deepspeed_config_sha256"],
            scientific["study"]["deepspeed_config_sha256"],
        )
        changed = copy.deepcopy(scientific)
        changed["study"]["deepspeed_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "deepspeed_config_sha256"):
            manifest.validate_scientific_source_claims(source, changed)

    def test_clean_source_checkout_is_bound_to_commit_tree_epoch_and_lfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "ARR Test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "arr@example.invalid"],
                check=True,
            )
            source = root / "modernbert"
            source.mkdir()
            (source / "train_sm.py").write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            environment = dict(os.environ)
            environment["GIT_AUTHOR_DATE"] = "1700000000 +0000"
            environment["GIT_COMMITTER_DATE"] = "1700000000 +0000"
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"],
                check=True,
                env=environment,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
            ).strip()
            self.assertEqual(
                manifest.validate_clean_source_checkout(
                    source,
                    expected_git_commit=commit,
                    expected_git_tree=tree,
                    expected_commit_epoch=1_700_000_000,
                ),
                root.resolve(),
            )
            (source / "train_sm.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completely clean"):
                manifest.validate_clean_source_checkout(
                    source,
                    expected_git_commit=commit,
                    expected_git_tree=tree,
                    expected_commit_epoch=1_700_000_000,
                )

    def _source_tree(self, root: Path) -> Path:
        source = root / "source"
        (source / "pkg" / "nested").mkdir(parents=True)
        (source / "pkg" / "__init__.py").write_bytes(b"")
        (source / "pkg" / "entry.py").write_text("print('ok')\n", encoding="utf-8")
        (source / "pkg" / "nested" / "value.txt").write_text("value\n", encoding="utf-8")
        return source

    def test_bundle_bytes_are_deterministic_and_metadata_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_tree(root)
            first = manifest.build_source_bundle(
                source_root=source,
                include_paths=["pkg"],
                output_path=root / "first.tar.gz",
                commit_epoch=1_700_000_000,
            )
            os.chmod(source / "pkg" / "entry.py", 0o777)
            os.utime(source / "pkg" / "entry.py", (1, 1))
            second = manifest.build_source_bundle(
                source_root=source,
                include_paths=["pkg"],
                output_path=root / "second.tar.gz",
                commit_epoch=1_700_000_000,
            )
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(first.path.read_bytes(), second.path.read_bytes())
            self.assertEqual(first.inventory, second.inventory)
            self.assertEqual(
                [record["path"] for record in first.inventory],
                sorted(record["path"] for record in first.inventory),
            )
            self.assertIn(
                {"path": "pkg", "type": "directory", "mode": "0755"},
                first.inventory,
            )
            deep = manifest.read_source_bundle(
                first.path,
                expected_inventory=first.inventory,
                expected_commit_epoch=1_700_000_000,
                expected_sha256=first.sha256,
            )
            self.assertEqual(deep, first)

    def test_bundle_refuses_links_overlap_specials_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source_tree(root)
            (source / "pkg" / "link").symlink_to("entry.py")
            with self.assertRaisesRegex(ValueError, "forbids symlink"):
                manifest.build_source_inventory(source, ["pkg"])
            (source / "pkg" / "link").unlink()

            with self.assertRaisesRegex(ValueError, "overlapping"):
                manifest.build_source_inventory(source, ["pkg", "pkg/entry.py"])

            fifo = source / "fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "special filesystem entry"):
                manifest.build_source_inventory(source, ["fifo"])

            output = root / "source.tar.gz"
            output.write_bytes(b"occupied")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                manifest.build_source_bundle(
                    source_root=source,
                    include_paths=["pkg"],
                    output_path=output,
                    commit_epoch=1_700_000_000,
                )

    def test_commit_exact_bundle_rejects_ignored_files_under_includes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "ARR Test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "arr@example.invalid"],
                check=True,
            )
            source = root / "modernbert"
            (source / "tests").mkdir(parents=True)
            (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
            (source / "tests" / "test_entry.py").write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            ignored = source / "tests" / "__pycache__"
            ignored.mkdir()
            (ignored / "test_entry.pyc").write_bytes(b"ignored runtime bytes")
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"]
                ),
                b"",
            )
            with self.assertRaisesRegex(ValueError, "exact commit-tracked"):
                manifest.build_source_bundle(
                    source_root=source,
                    include_paths=["tests"],
                    output_path=Path(temporary) / "source.tar.gz",
                    commit_epoch=1_700_000_000,
                    expected_git_commit=commit,
                )


class DryManifestTest(unittest.TestCase):
    def _bundle(self, root: Path) -> manifest.SourceBundle:
        source = root / "source"
        source.mkdir()
        (source / "train_sm.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        return manifest.build_source_bundle(
            source_root=source,
            include_paths=["train_sm.py"],
            output_path=root / "source.tar.gz",
            commit_epoch=1_700_000_000,
        )

    def _dry_manifest(self, root: Path) -> dict[str, object]:
        scientific = valid_scientific_config()
        scientific["sources"]["include_paths"] = ["train_sm.py"]
        return manifest.build_dry_manifest(
            scientific_config=scientific,
            aws_local_config=valid_aws_config(),
            source_bundle=self._bundle(root),
        )

    def test_exact_matrix_order_counts_and_unique_launch_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dry = self._dry_manifest(Path(temporary))
            controlled = dry["controlled_runs"]
            auxiliary = dry["auxiliary_runs"]
            self.assertEqual(
                dry["execution"],
                {
                    "blockers": list(manifest.EXECUTION_BLOCKERS),
                    "status": "blocked",
                    "submittable": False,
                },
            )
            self.assertEqual(len(controlled), 60)
            self.assertEqual(len(auxiliary), 4)
            self.assertEqual(
                controlled[0]["cell"],
                {
                    "outer_fold": 0,
                    "query_view": "flat_masked",
                    "sampler": "local_unique",
                    "experiment_seed": 17,
                },
            )
            self.assertEqual(
                controlled[0]["job_name"], "arr-ret-cv1-f0-flat-local-s17-a1"
            )
            self.assertEqual(
                controlled[-1]["job_name"], "arr-ret-cv1-f4-struct-global-s43-a1"
            )
            self.assertEqual(
                [run["kind"] for run in auxiliary],
                [
                    manifest.LEGACY_KIND,
                    manifest.LEGACY_KIND,
                    manifest.SMOKE_KIND,
                    manifest.SMOKE_KIND,
                ],
            )
            self.assertEqual(
                [run["run_id"] for run in auxiliary[:2]],
                ["corrected-legacy-flat", "corrected-legacy-structured"],
            )
            self.assertEqual(
                [run["job_name"] for run in auxiliary[:2]],
                [
                    "arr-ret-cv1-corrected-legacy-flat-a1",
                    "arr-ret-cv1-corrected-legacy-structured-a1",
                ],
            )
            self.assertEqual(
                auxiliary[0]["hyperparameters"],
                {
                    "base_seed": 17,
                    "epochs": 20,
                    "query_view": "flat_masked",
                    "run_kind": "corrected_legacy_diagnostic",
                    "total_optimizer_updates": 80,
                },
            )
            self.assertEqual(
                [run["entry_point"] for run in auxiliary[:2]],
                ["train_sm.py", "train_sm.py"],
            )
            expected_corrected_artifact = {
                "artifact_type": "corrected_legacy_diagnostic_retriever",
                "schema_version": 1,
                "validator_version": "corrected_legacy_diagnostic_artifact_v1",
            }
            self.assertEqual(
                [run["expected_artifact_identity"] for run in auxiliary[:2]],
                [expected_corrected_artifact, expected_corrected_artifact],
            )
            for field in ("run_id", "job_name", "output_prefix"):
                values = [run[field] for run in [*controlled, *auxiliary]]
                self.assertEqual(len(values), len(set(values)))
            for fold in range(5):
                self.assertEqual(
                    sum(run["cell"]["outer_fold"] == fold for run in controlled), 12
                )

    def test_smoke_replica_is_only_launch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dry = self._dry_manifest(Path(temporary))
            first, second = dry["auxiliary_runs"][2:]
            scientific_container_fields = (
                "kind",
                "cell",
                "entry_point",
                "source_bundle_sha256",
                "hyperparameters",
                "environment",
                "input_channels",
                "expected_artifact_identity",
            )
            for field in scientific_container_fields:
                self.assertEqual(first[field], second[field])
            self.assertEqual(first["launch_metadata"], {"replica_id": "a"})
            self.assertEqual(second["launch_metadata"], {"replica_id": "b"})
            self.assertNotEqual(first["job_name"], second["job_name"])
            self.assertNotEqual(first["output_prefix"], second["output_prefix"])

    def test_validation_rejects_unknown_boolean_duplicate_and_smoke_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dry = self._dry_manifest(Path(temporary))

            unknown = copy.deepcopy(dry)
            unknown["controlled_runs"][0]["retry"] = True
            with self.assertRaisesRegex(ValueError, r"unknown=\['retry'\]"):
                manifest.validate_dry_manifest(unknown)

            boolean_fold = copy.deepcopy(dry)
            boolean_fold["controlled_runs"][0]["cell"]["outer_fold"] = False
            with self.assertRaisesRegex(ValueError, "outer_fold mismatch"):
                manifest.validate_dry_manifest(boolean_fold)

            duplicate = copy.deepcopy(dry)
            duplicate["controlled_runs"][1]["output_prefix"] = duplicate[
                "controlled_runs"
            ][0]["output_prefix"]
            with self.assertRaisesRegex(ValueError, "output_prefix is not canonical"):
                manifest.validate_dry_manifest(duplicate)

            drift = copy.deepcopy(dry)
            drift["auxiliary_runs"][3]["environment"]["EXTRA"] = "drift"
            with self.assertRaisesRegex(ValueError, "environment contract"):
                manifest.validate_dry_manifest(drift)

            invalid_schedule = copy.deepcopy(dry)
            for run in invalid_schedule["auxiliary_runs"][2:]:
                run["hyperparameters"]["epochs"] = 20
            with self.assertRaisesRegex(ValueError, "exactly two smoke epochs"):
                manifest.validate_dry_manifest(invalid_schedule)

            invalid_corrected_legacy = copy.deepcopy(dry)
            invalid_corrected_legacy["auxiliary_runs"][0]["hyperparameters"][
                "total_optimizer_updates"
            ] = 60
            with self.assertRaisesRegex(
                ValueError, "corrected legacy diagnostic hyperparameters changed"
            ):
                manifest.validate_dry_manifest(invalid_corrected_legacy)

            falsely_submittable = copy.deepcopy(dry)
            falsely_submittable["execution"] = {
                "blockers": [],
                "status": "ready",
                "submittable": True,
            }
            with self.assertRaisesRegex(ValueError, "non-submittable"):
                manifest.validate_dry_manifest(falsely_submittable)

            manifest_mutations = []
            changed_channel = copy.deepcopy(dry)
            changed_channel["controlled_runs"][0]["input_channels"]["base_model"][
                "identity_sha256"
            ] = "0" * 64
            manifest_mutations.append(changed_channel)
            changed_entry = copy.deepcopy(dry)
            changed_entry["controlled_runs"][0]["entry_point"] = "other.py"
            manifest_mutations.append(changed_entry)
            changed_artifact = copy.deepcopy(dry)
            changed_artifact["controlled_runs"][0]["expected_artifact_identity"][
                "validator_version"
            ] = "other_v1"
            manifest_mutations.append(changed_artifact)
            changed_environment = copy.deepcopy(dry)
            changed_environment["controlled_runs"][0]["environment"]["EXTRA"] = "1"
            manifest_mutations.append(changed_environment)
            changed_repository = copy.deepcopy(dry)
            changed_repository["infrastructure"]["ecr_repository"] = "other-repo"
            manifest_mutations.append(changed_repository)
            for changed in manifest_mutations:
                with self.subTest(changed=changed):
                    with self.assertRaises(ValueError):
                        manifest.validate_dry_manifest(changed)

    def test_manifest_publication_is_canonical_absent_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dry = self._dry_manifest(root)
            path = root / "launch-manifest.json"
            published, digest = manifest.publish_manifest_absent(path, dry)
            self.assertEqual(published, dry)
            self.assertEqual(path.read_bytes(), config.canonical_json_bytes(dry))
            readback, readback_digest = manifest.read_manifest(
                path, expected_sha256=digest
            )
            self.assertEqual(readback, dry)
            self.assertEqual(readback_digest, digest)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                manifest.publish_manifest_absent(path, dry)


if __name__ == "__main__":
    unittest.main()
