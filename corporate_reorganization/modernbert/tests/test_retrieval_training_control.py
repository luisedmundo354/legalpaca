from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
EXPERIMENT_CONFIG = MODERNBERT_DIR / "experiments/retrieval_cv/configs/experiment.json"
SNAPSHOT_MANIFEST = MODERNBERT_DIR / "experiments/retrieval_cv/configs/modernbert_snapshot.json"
DEEPSPEED_CONFIG = MODERNBERT_DIR / "ds_zero3.json"
DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)

if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import train_sm as controlled_train  # noqa: E402
from retriever.batching import (  # noqa: E402
    DUMMY_QUERY_INDEX,
    GlobalQueryBatchSampler,
    SentinelQueryDataset,
)
from retriever.provenance import (  # noqa: E402
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    load_snapshot_manifest,
    validate_preimport_environment,
    validate_runtime_versions,
    validate_snapshot_directory,
)


class GlobalQueryBatchSamplerTest(unittest.TestCase):
    def make_sampler(self, *, seed: int = 17) -> GlobalQueryBatchSampler:
        return GlobalQueryBatchSampler(
            [f"query-{index:03d}" for index in range(294)],
            experiment_seed=seed,
            world_size=4,
            per_device_batch_size=4,
        )

    def test_exact_controlled_coverage_padding_and_windows(self) -> None:
        sampler = self.make_sampler()
        raw_batches = list(sampler)
        self.assertEqual(len(raw_batches), 76)
        self.assertEqual(sampler.prepared_batches_per_rank, 19)
        self.assertEqual(sampler.num_sentinel_rows, 10)
        self.assertTrue(all(len(batch) == 4 for batch in raw_batches))

        rank_batches = [raw_batches[rank::4] for rank in range(4)]
        self.assertEqual([len(batches) for batches in rank_batches], [19] * 4)
        self.assertEqual(
            [sum(index != DUMMY_QUERY_INDEX for index in batches[-1]) for batches in rank_batches],
            [2, 2, 1, 1],
        )

        real_indices = [
            index
            for batches in rank_batches
            for batch in batches
            for index in batch
            if index != DUMMY_QUERY_INDEX
        ]
        self.assertEqual(len(real_indices), 294)
        self.assertEqual(sorted(real_indices), list(range(294)))
        self.assertEqual(len(set(real_indices)), 294)

        window_valid_counts = []
        for start, stop in ((0, 8), (8, 16), (16, 19)):
            window_valid_counts.append(
                sum(
                    index != DUMMY_QUERY_INDEX
                    for batches in rank_batches
                    for batch in batches[start:stop]
                    for index in batch
                )
            )
        self.assertEqual(window_valid_counts, [128, 128, 38])

    def test_replay_seed_epoch_and_input_order_semantics(self) -> None:
        sampler = self.make_sampler()
        epoch_zero = sampler.batches()
        epoch_zero_checksum = hashlib.sha256(
            json.dumps(epoch_zero, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            epoch_zero_checksum,
            "4a90d382b862f51c9f8f86acaf34c9d1bf06394582c4cacd8161fce5ae66d6bf",
        )
        self.assertEqual(epoch_zero, self.make_sampler().batches())
        self.assertNotEqual(epoch_zero, self.make_sampler(seed=29).batches())

        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, sampler.batches())
        sampler.set_epoch(0)
        self.assertEqual(epoch_zero, sampler.batches())

        query_ids = [f"query-{index:03d}" for index in range(294)]
        reversed_sampler = GlobalQueryBatchSampler(
            list(reversed(query_ids)),
            experiment_seed=17,
            world_size=4,
            per_device_batch_size=4,
        )
        ordered_ids = [query_ids[index] for index in sampler.ordered_real_indices()]
        reversed_ids = [
            list(reversed(query_ids))[index]
            for index in reversed_sampler.ordered_real_indices()
        ]
        self.assertEqual(ordered_ids, reversed_ids)

    def test_invalid_plans_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            GlobalQueryBatchSampler(
                ["q", "q"],
                experiment_seed=17,
                world_size=1,
                per_device_batch_size=1,
            )
        with self.assertRaises(TypeError):
            GlobalQueryBatchSampler(
                ["q"],
                experiment_seed=True,
                world_size=1,
                per_device_batch_size=1,
            )
        with self.assertRaisesRegex(ValueError, "all-dummy"):
            GlobalQueryBatchSampler(
                [f"q{i}" for i in range(17)],
                experiment_seed=17,
                world_size=4,
                per_device_batch_size=4,
            )


class SentinelQueryDatasetTest(unittest.TestCase):
    class Dataset:
        def __init__(self) -> None:
            self.rows = [{"query_id": "q0"}, {"query_id": "q1"}]
            self.epochs: list[int] = []
            self.requested: list[int] = []

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            self.requested.append(index)
            return self.rows[index]

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    def test_sentinel_never_reads_source_and_epoch_updates_both_targets(self) -> None:
        dataset = self.Dataset()
        sampler = GlobalQueryBatchSampler(
            ["q0", "q1"],
            experiment_seed=17,
            world_size=1,
            per_device_batch_size=2,
        )
        wrapped = SentinelQueryDataset(dataset, epoch_target=sampler)

        self.assertEqual(wrapped[DUMMY_QUERY_INDEX], {"is_dummy": True})
        self.assertEqual(dataset.requested, [])
        self.assertEqual(wrapped[1], {"query_id": "q1", "is_dummy": False})
        self.assertEqual(dataset.requested, [1])

        wrapped.set_epoch(3)
        self.assertEqual(dataset.epochs, [3])
        self.assertEqual(sampler.epoch, 3)

    def test_reserved_and_invalid_indices_fail_loudly(self) -> None:
        wrapped = SentinelQueryDataset(self.Dataset())
        with self.assertRaises(TypeError):
            _ = wrapped[True]
        with self.assertRaises(IndexError):
            _ = wrapped[-2]
        with self.assertRaises(IndexError):
            _ = wrapped[2]

        class ReservedDataset(self.Dataset):
            def __getitem__(self, index: int):
                return {"query_id": "q", "is_dummy": False}

        with self.assertRaisesRegex(ValueError, "reserved"):
            _ = SentinelQueryDataset(ReservedDataset())[0]


class ProvenanceValidationTest(unittest.TestCase):
    def test_exact_runtime_inventory_and_mismatches(self) -> None:
        expected = dict(EXPECTED_RUNTIME_VERSIONS)
        self.assertEqual(validate_runtime_versions(expected), expected)
        for changed in (
            {key: value for key, value in expected.items() if key != "numpy"},
            {**expected, "unexpected": "1"},
            {**expected, "transformers": "4.50.0"},
        ):
            with self.assertRaisesRegex(RuntimeError, "frozen inventory"):
                validate_runtime_versions(changed)

    def test_preimport_environment_is_exact(self) -> None:
        exact = {
            "PYTHONHASHSEED": "17",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
        }
        with mock.patch.dict(os.environ, exact, clear=True):
            validate_preimport_environment(17)
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            with self.assertRaisesRegex(RuntimeError, "not exact"):
                validate_preimport_environment(17)

    def test_frozen_snapshot_manifest_is_canonical_and_exact(self) -> None:
        manifest = load_snapshot_manifest(SNAPSHOT_MANIFEST)
        self.assertEqual(manifest["tree_sha256"], EXPECTED_SNAPSHOT_TREE_SHA256)
        self.assertEqual(len(manifest["files"]), 5)

        changed = copy.deepcopy(manifest)
        changed["files"][0]["sha256"] = changed["files"][0]["sha256"].upper()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.json"
            path.write_text(
                json.dumps(changed, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lowercase 64-hex"):
                load_snapshot_manifest(path)

            alternate_files = [
                {
                    "path": "config.json",
                    "size": 2,
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            ]
            alternate = {
                "files": alternate_files,
                "manifest_type": "huggingface_model_snapshot",
                "model_id": "answerdotai/ModernBERT-base",
                "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
                "schema_version": 1,
                "tree_sha256": hashlib.sha256(
                    json.dumps(
                        alternate_files,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
            }
            path.write_text(
                json.dumps(alternate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not the frozen ModernBERT tree"):
                load_snapshot_manifest(path)

    def test_snapshot_directory_rejects_missing_extra_mutated_and_symlink_files(self) -> None:
        content = b"immutable snapshot fixture"
        record = {
            "path": "model.bin",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        manifest = {"files": [record]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot = Path(tmp_dir) / "snapshot"
            snapshot.mkdir()
            model = snapshot / "model.bin"
            model.write_bytes(content)
            validate_snapshot_directory(snapshot, manifest)

            (snapshot / "extra.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                validate_snapshot_directory(snapshot, manifest)
            (snapshot / "extra.json").unlink()

            model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                validate_snapshot_directory(snapshot, manifest)

            model.write_bytes(b"x" * len(content))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_snapshot_directory(snapshot, manifest)
            model.unlink()

            target = Path(tmp_dir) / "target.bin"
            target.write_bytes(content)
            model.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                validate_snapshot_directory(snapshot, manifest)


class StrictEntrypointTest(unittest.TestCase):
    @staticmethod
    def valid_cli() -> list[str]:
        return [
            "--data-dir",
            "/data",
            "--base-model-dir",
            "/model",
            "--output-dir",
            "/output",
            "--outer-fold",
            "0",
            "--query-view",
            "structured",
            "--sampler",
            "local_unique",
            "--experiment-seed",
            "17",
        ]

    def test_cli_rejects_unknown_abbreviated_and_missing_inputs(self) -> None:
        args = controlled_train.parse_args(self.valid_cli())
        self.assertEqual(args.outer_fold, 0)
        self.assertEqual(args.experiment_seed, 17)

        for extra in (["--unknown"], ["--outer", "1"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                controlled_train.parse_args(self.valid_cli() + extra)

        matrix_only = self.valid_cli()[6:]
        with mock.patch.dict(os.environ, {}, clear=True):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                controlled_train.parse_args(matrix_only)

    def test_frozen_experiment_and_deepspeed_configs_validate(self) -> None:
        controlled_train._validate_frozen_control_file_hashes(
            experiment_config_path=EXPERIMENT_CONFIG,
            deepspeed_config_path=DEEPSPEED_CONFIG,
        )
        experiment = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
        controlled_train._validate_experiment_config(
            experiment,
            outer_fold=0,
            query_view="structured",
            sampler="local_unique",
            experiment_seed=17,
        )
        controlled_train._validate_deepspeed_config(DEEPSPEED_CONFIG)

        changed = copy.deepcopy(experiment)
        changed["models"]["modernbert_base"]["snapshot_tree_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ModernBERT"):
            controlled_train._validate_experiment_config(
                changed,
                outer_fold=0,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
            )

        for section, key in (
            (None, "schema_version"),
            ("aws_training", "instance_count"),
            ("training", "max_grad_norm"),
            ("runtime_control", "validation_forward_steps"),
        ):
            changed = copy.deepcopy(experiment)
            if section is None:
                changed[key] = True
            else:
                changed[section][key] = True
            with self.assertRaises(ValueError):
                controlled_train._validate_experiment_config(
                    changed,
                    outer_fold=0,
                    query_view="structured",
                    sampler="local_unique",
                    experiment_seed=17,
                )

        changed = copy.deepcopy(experiment)
        changed["training"]["model_selection"]["primary"] = (
            "validation_case_macro_hit_at_20"
        )
        with self.assertRaisesRegex(ValueError, "model-selection"):
            controlled_train._validate_experiment_config(
                changed,
                outer_fold=0,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
            )

        changed = copy.deepcopy(experiment)
        changed["evaluation"]["primary_candidate_regime"] = "all_42_cases"
        with self.assertRaisesRegex(ValueError, "evaluation"):
            controlled_train._validate_experiment_config(
                changed,
                outer_fold=0,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
            )

        with self.assertRaisesRegex(ValueError, "exact integer"):
            controlled_train._validate_experiment_config(
                experiment,
                outer_fold=True,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
            )

        deepspeed = json.loads(DEEPSPEED_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            reformatted_experiment = Path(tmp_dir) / "experiment.json"
            reformatted_experiment.write_text(
                json.dumps(experiment),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                controlled_train._validate_frozen_control_file_hashes(
                    experiment_config_path=reformatted_experiment,
                    deepspeed_config_path=DEEPSPEED_CONFIG,
                )
            reformatted_deepspeed = Path(tmp_dir) / "ds_zero3.json"
            reformatted_deepspeed.write_text(
                json.dumps(deepspeed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                controlled_train._validate_frozen_control_file_hashes(
                    experiment_config_path=EXPERIMENT_CONFIG,
                    deepspeed_config_path=reformatted_deepspeed,
                )
            for mutation in (
                lambda value: value["bf16"].update(enabled=1),
                lambda value: value["zero_optimization"].update(overlap_comm=False),
                lambda value: value["zero_optimization"].update(unexpected=True),
            ):
                changed_ds = copy.deepcopy(deepspeed)
                mutation(changed_ds)
                path = Path(tmp_dir) / "changed-ds.json"
                path.write_text(json.dumps(changed_ds), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "frozen exact"):
                    controlled_train._validate_deepspeed_config(path)

    def test_artifact_submanifest_records_include_exact_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixtures = (
                ("candidate_traces/manifest.json", b'{"trace":true}\n'),
                ("validation/manifest.json", b'{"validation":true}\n'),
            )
            for logical_path, payload in fixtures:
                with self.subTest(logical_path=logical_path):
                    path = root / Path(logical_path).name
                    path.write_bytes(payload)
                    self.assertEqual(
                        controlled_train._artifact_file_record(
                            path,
                            logical_path=logical_path,
                        ),
                        {
                            "path": logical_path,
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    )

    def test_staged_dataset_and_fold_are_relocation_safe_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            staged = Path(tmp_dir) / "sagemaker-channel-data"
            staged_folds = Path(tmp_dir) / "mounted-folds.json"
            shutil.copytree(DATASET_DIR, staged, symlinks=False)
            shutil.copy2(controlled_train.DEFAULT_FOLDS_CONFIG, staged_folds)
            manifest = controlled_train._validate_staged_fold_manifest(
                dataset_dir=staged,
                fold_manifest_path=staged_folds,
            )
            (staged / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                controlled_train._validate_staged_fold_manifest(
                    dataset_dir=staged,
                    fold_manifest_path=staged_folds,
                )
            (staged / "unexpected.txt").unlink()
            with (staged / "corpus.jsonl").open("ab") as destination:
                destination.write(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                controlled_train._validate_staged_fold_manifest(
                    dataset_dir=staged,
                    fold_manifest_path=staged_folds,
                )

            shutil.copy2(DATASET_DIR / "corpus.jsonl", staged / "corpus.jsonl")
            with staged_folds.open("ab") as destination:
                destination.write(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "fold-manifest SHA-256 changed"):
                controlled_train._validate_staged_fold_manifest(
                    dataset_dir=staged,
                    fold_manifest_path=staged_folds,
                )
        self.assertEqual(manifest["totals"], {"cases": 42, "queries": 490, "passages": 5286})
        self.assertEqual(
            manifest["dataset"]["dataset_manifest_path"],
            controlled_train.EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
        )

    def test_distributed_environment_rejects_conflicts_and_wrong_mapping(self) -> None:
        class FakeCuda:
            def __init__(self) -> None:
                self.devices: list[int] = []

            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 4

            def set_device(self, index: int) -> None:
                self.devices.append(index)

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        exact = {
            "OMPI_COMM_WORLD_LOCAL_RANK": "2",
            "OMPI_COMM_WORLD_RANK": "2",
            "OMPI_COMM_WORLD_SIZE": "4",
        }
        with mock.patch.dict(os.environ, exact, clear=True):
            self.assertEqual(
                controlled_train._configure_distributed_environment(fake_torch),
                (2, 2, 4),
            )
            self.assertEqual(fake_torch.cuda.devices, [2])

        conflicting = {**exact, "RANK": "1"}
        with mock.patch.dict(os.environ, conflicting, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                controlled_train._configure_distributed_environment(fake_torch)

        wrong_mapping = {
            "LOCAL_RANK": "1",
            "RANK": "2",
            "WORLD_SIZE": "4",
        }
        with mock.patch.dict(os.environ, wrong_mapping, clear=True):
            with self.assertRaisesRegex(RuntimeError, "four-GPU host"):
                controlled_train._configure_distributed_environment(fake_torch)

    def test_nonempty_or_symlink_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output = root / "new-output"
            controlled_train._prepare_output_directory(output)
            (output / "existing.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                controlled_train._prepare_output_directory(output)

            target = root / "real-output"
            target.mkdir()
            link = root / "linked-output"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                controlled_train._prepare_output_directory(link)

    def test_loaded_attention_contract_requires_backend_config_and_every_module(self) -> None:
        valid = SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="flash_attention_2",
                deterministic_flash_attn=True,
                reference_compile=False,
            ),
            modules=lambda: [SimpleNamespace(deterministic_flash_attn=True)],
        )
        controlled_train._validate_loaded_modernbert_attention(valid)

        for changed in (
            SimpleNamespace(
                config=SimpleNamespace(
                    _attn_implementation="sdpa",
                    deterministic_flash_attn=True,
                    reference_compile=False,
                ),
                modules=valid.modules,
            ),
            SimpleNamespace(
                config=valid.config,
                modules=lambda: [SimpleNamespace(deterministic_flash_attn=False)],
            ),
            SimpleNamespace(
                config=SimpleNamespace(
                    _attn_implementation="flash_attention_2",
                    deterministic_flash_attn=True,
                    reference_compile=True,
                ),
                modules=valid.modules,
            ),
        ):
            with self.assertRaises(RuntimeError):
                controlled_train._validate_loaded_modernbert_attention(changed)


if __name__ == "__main__":
    unittest.main()
