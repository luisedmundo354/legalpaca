from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.artifacts import (  # noqa: E402
    CONTROLLED_ARTIFACT_PROTOCOL,
    CONTROLLED_ATTENTION_MODULE_COUNT,
    CONTROLLED_EXPERIMENT_ID,
    CONTROLLED_MODEL_STATE_COUNT,
    CONTROLLED_SLOT_TOKEN,
    CONTROLLED_TOKENIZER_SIZE,
    ControlledArtifactExpectation,
    ControlledArtifactRuntime,
    import_pinned_artifact_runtime,
    load_controlled_retriever,
    validate_controlled_artifact,
)
from retriever.provenance import (  # noqa: E402
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_DATASET_OUTPUT_SHA256,
    EXPECTED_DEEPSPEED_CONFIG_SHA256,
    EXPECTED_EXPERIMENT_CONFIG_SHA256,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_IMAGE,
    EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD,
    EXPECTED_VALIDATION_IDENTITY_BY_CELL,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}


def _directory_record(root: Path, relative: str) -> dict[str, Any]:
    directory = root / relative
    files = [
        _record(directory, path.relative_to(directory).as_posix())
        for path in sorted(
            (item for item in directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(directory).as_posix(),
        )
    ]
    return {"path": relative, "files": files}


def _selection(epoch: int) -> dict[str, Any]:
    step = epoch * 3
    return {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": step,
        "checkpoint_dir": f"checkpoint-{step}",
        "deepspeed_tag": f"global_step{step}",
        "primary_metric": float(epoch) / 100.0,
        "secondary_metric": float(epoch) / 1_000.0,
        "ranking_sha256": hashlib.sha256(f"ranking:{epoch}".encode()).hexdigest(),
    }


def _create_checkpoint(root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    checkpoint_name = selection["checkpoint_dir"]
    tag = selection["deepspeed_tag"]
    checkpoint_root = root / checkpoint_name
    checkpoint_root.mkdir()
    for filename in (
        "zero_to_fp32.py",
        "scheduler.pt",
        "training_args.bin",
        "trainer_state.json",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ):
        (checkpoint_root / filename).write_bytes(f"fixture:{filename}\n".encode())
    tag_root = checkpoint_root / tag
    tag_root.mkdir()
    for rank in range(4):
        for filename in (
            f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt",
            f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        ):
            (tag_root / filename).write_bytes(f"fixture:{filename}\n".encode())
    files = [
        _record(checkpoint_root, path.relative_to(checkpoint_root).as_posix())
        for path in sorted(
            (item for item in checkpoint_root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(checkpoint_root).as_posix(),
        )
    ]
    checkpoint_manifest = {
        "schema_version": 1,
        "selection": selection,
        "world_size": 4,
        "client_state_sha256": SHA_A,
        "scheduler_state_sha256": SHA_B,
        "rng_files": [f"rng_state_{rank}.pth" for rank in range(4)],
        "files": files,
    }
    _write_json(checkpoint_root / "checkpoint_manifest.json", checkpoint_manifest)
    return _directory_record(root, checkpoint_name)


def _create_candidate_traces(root: Path, *, passage_index_sha256: str) -> dict[str, Any]:
    trace_root = root / "candidate_traces"
    trace_root.mkdir()
    record_count = 20 * 294
    counts = [record_count // 4] * 4
    merged_lines: list[bytes] = []
    shard_records: list[dict[str, Any]] = []
    for rank, count in enumerate(counts):
        shard_path = trace_root / f"rank-{rank:05d}.jsonl"
        content = b"{}\n" * count
        shard_path.write_bytes(content)
        merged_lines.append(content)
        shard_records.append(
            {
                "rank": rank,
                **_record(trace_root, shard_path.name),
                "record_count": count,
            }
        )
    merged_path = trace_root / "sampling_traces.jsonl"
    merged_path.write_bytes(b"".join(merged_lines))
    manifest = {
        "schema_version": 1,
        "merge_order": ["epoch", "query_id"],
        "epochs": 20,
        "queries_per_epoch": 294,
        "record_count": record_count,
        "query_ids_sha256": EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD[0],
        "passage_index_sha256": passage_index_sha256,
        "merged": {**_record(trace_root, merged_path.name), "record_count": record_count},
        "shards": shard_records,
    }
    _write_json(trace_root / "manifest.json", manifest)
    return manifest


def _create_validation_history(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validation_root = root / "validation"
    validation_root.mkdir()
    history_entries: list[dict[str, Any]] = []
    validation_results: dict[int, dict[str, Any]] = {}
    for epoch in range(1, 21):
        candidate = _selection(epoch)
        validation_result = {
            "schema_version": 1,
            "metrics": {"fixture": float(epoch)},
            "ranking_sha256": candidate["ranking_sha256"],
        }
        validation_results[epoch] = validation_result
        checkpoint_metadata = {
            "checkpoint_dir": candidate["checkpoint_dir"],
            "deepspeed_tag": candidate["deepspeed_tag"],
            "manifest_sha256": SHA_A,
            "scheduler_state_sha256": SHA_B,
            "client_state_sha256": SHA_C,
        }
        epoch_payload = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": epoch * 3,
            "checkpoint": checkpoint_metadata,
            "candidate": candidate,
            "is_new_best": True,
            "best_after_epoch": candidate,
            "validation_result": validation_result,
        }
        epoch_path = validation_root / f"epoch-{epoch:03d}.json"
        _write_json(epoch_path, epoch_payload)
        history_entries.append(
            {
                "epoch": epoch,
                "global_step": epoch * 3,
                "path": epoch_path.name,
                "sha256": _sha256(epoch_path),
                "is_new_best": True,
                "candidate": candidate,
                "best_after_epoch": candidate,
            }
        )
    history = {"schema_version": 1, "records": history_entries}
    _write_json(validation_root / "history.json", history)
    _write_json(validation_root / "latest.json", history_entries[-1])
    _write_json(validation_root / "best.json", history_entries[-1])
    manifest = {
        "schema_version": 1,
        "epochs": 20,
        "selection_order": [
            "maximize validation case-macro set recall@20",
            "maximize validation case-macro full-ranking first-gold reciprocal rank",
            "minimize epoch number",
        ],
        "best": _selection(20),
        "last": _selection(20),
        "retained_checkpoint_dirs": ["checkpoint-60"],
        "records": history_entries,
        "history_sha256": _sha256(validation_root / "history.json"),
        "best_sha256": _sha256(validation_root / "best.json"),
        "latest_sha256": _sha256(validation_root / "latest.json"),
    }
    _write_json(validation_root / "manifest.json", manifest)
    return manifest, validation_results[20]


def _default_asset_writer(root: Path) -> int:
    (root / "model.safetensors").write_bytes(b"fixture-safetensors\n")
    tokenizer_root = root / "tokenizer"
    tokenizer_root.mkdir()
    _write_json(
        tokenizer_root / "special_tokens_map.json",
        {"additional_special_tokens": [CONTROLLED_SLOT_TOKEN]},
    )
    _write_json(tokenizer_root / "tokenizer.json", {"fixture": True})
    _write_json(tokenizer_root / "tokenizer_config.json", {"model_max_length": 8192})
    encoder_root = root / "encoder_config"
    encoder_root.mkdir()
    _write_json(
        encoder_root / "config.json",
        {
            "model_type": "modernbert",
            "vocab_size": CONTROLLED_TOKENIZER_SIZE,
            "deterministic_flash_attn": True,
            "reference_compile": False,
            "torch_dtype": "float32",
        },
    )
    return 7


def _rotation(outer_fold: int) -> dict[str, Any]:
    fold_manifest_path = (
        MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json"
    )
    fold_manifest = json.loads(fold_manifest_path.read_bytes())
    return fold_manifest["rotations"][outer_fold]


def _build_artifact(
    root: Path,
    *,
    asset_writer: Callable[[Path], int] = _default_asset_writer,
) -> ControlledArtifactExpectation:
    root.mkdir()
    slot_token_id = asset_writer(root)
    wrapper = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "slot_token": CONTROLLED_SLOT_TOKEN,
        "slot_token_id": slot_token_id,
        "temperature": 0.07,
        "tokenizer_size": CONTROLLED_TOKENIZER_SIZE,
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": CONTROLLED_ARTIFACT_PROTOCOL,
    }
    _write_json(root / "wrapper_config.json", wrapper)
    trace_manifest = _create_candidate_traces(
        root,
        passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
    )
    validation_manifest, best_result = _create_validation_history(root)
    checkpoint_record = _create_checkpoint(root, _selection(20))
    retained = {"schema_version": 1, "checkpoints": [checkpoint_record]}
    tokenizer_record = _directory_record(root, "tokenizer")
    encoder_record = _directory_record(root, "encoder_config")
    model_record = _record(root, "model.safetensors")
    wrapper_record = _record(root, "wrapper_config.json")
    trace_manifest_record = _record(root, "candidate_traces/manifest.json")
    validation_manifest_record = _record(root, "validation/manifest.json")
    rotation = _rotation(0)
    validation_identity = EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")]
    run = {
        "schema_version": 1,
        "experiment_id": CONTROLLED_EXPERIMENT_ID,
        "outer_fold": 0,
        "query_view": "structured",
        "sampler": "local_unique",
        "experiment_seed": 17,
        "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
        "training_image": EXPECTED_TRAINING_IMAGE,
        "experiment_config": {
            "path": "experiment.json",
            "sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
        },
        "deepspeed_config": {
            "path": "ds_zero3.json",
            "sha256": EXPECTED_DEEPSPEED_CONFIG_SHA256,
        },
        "dataset": {
            "manifest_path": "dataset_manifest.json",
            "manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "output_sha256": EXPECTED_DATASET_OUTPUT_SHA256,
        },
        "folds": {
            "manifest_path": (
                "corporate_reorganization/modernbert/experiments/"
                "retrieval_cv/configs/folds.json"
            ),
            "manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
            "rotation": rotation,
        },
        "snapshot": {
            "manifest_path": "modernbert_snapshot.json",
            "manifest_sha256": EXPECTED_SNAPSHOT_MANIFEST_SHA256,
            "tree_sha256": EXPECTED_SNAPSHOT_TREE_SHA256,
        },
        "passage_index": {
            "schema_version": 1,
            "size": 5_286,
            "sha256": EXPECTED_PASSAGE_INDEX_SHA256,
        },
        "validation_data": {
            "role": "validation",
            "query_view": "structured",
            "case_count": 9,
            "query_count": 98,
            "passage_count": 1_060,
            **validation_identity,
        },
        "candidate_traces": {
            "manifest_path": "candidate_traces/manifest.json",
            "manifest_sha256": trace_manifest_record["sha256"],
            "record_count": 5_880,
            "merged_sha256": trace_manifest["merged"]["sha256"],
        },
        "validation_history": {
            "manifest_path": "validation/manifest.json",
            "manifest_sha256": validation_manifest_record["sha256"],
            "best": validation_manifest["best"],
            "last": validation_manifest["last"],
            "retained_checkpoint_dirs": ["checkpoint-60"],
        },
        "best_checkpoint_reload": {
            "selection": validation_manifest["best"],
            "validation_result": best_result,
            "per_rank": [
                {
                    "rank": rank,
                    "load_path_parent": "/training/output/checkpoint-60/global_step60",
                    "client_state_sha256": SHA_A,
                    "scheduler_state_sha256": SHA_B,
                    "global_step": 60,
                    "rng_sha256": SHA_C,
                    "manifest_sha256": SHA_D,
                }
                for rank in range(4)
            ],
        },
        "final_model": {
            **model_record,
            "weight_dtype": "bfloat16",
            "gathered_tensor_count": CONTROLLED_MODEL_STATE_COUNT,
            "strict_round_trip_tensor_count": CONTROLLED_MODEL_STATE_COUNT,
        },
        "tokenizer": tokenizer_record,
        "encoder_config": encoder_record,
        "wrapper_config": wrapper_record,
        "retained_checkpoints": retained,
    }
    _write_json(root / "controlled_run.json", run)
    artifact_manifest = {
        "schema_version": 1,
        "commit_marker": True,
        "controlled_run": _record(root, "controlled_run.json"),
        "model": model_record,
        "tokenizer": tokenizer_record,
        "encoder_config": encoder_record,
        "wrapper_config": wrapper_record,
        "candidate_trace_manifest": trace_manifest_record,
        "validation_manifest": validation_manifest_record,
        "retained_checkpoints": retained,
    }
    _write_json(root / "artifact_manifest.json", artifact_manifest)
    return ControlledArtifactExpectation(
        artifact_manifest_sha256=_sha256(root / "artifact_manifest.json"),
        experiment_id=CONTROLLED_EXPERIMENT_ID,
        outer_fold=0,
        query_view="structured",
        sampler="local_unique",
        experiment_seed=17,
        dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
        passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
        model_artifact_protocol=CONTROLLED_ARTIFACT_PROTOCOL,
    )


def _refresh_run_and_manifest(root: Path) -> ControlledArtifactExpectation:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((root / "controlled_run.json").read_text(encoding="utf-8"))
    manifest["model"] = _record(root, "model.safetensors")
    manifest["tokenizer"] = _directory_record(root, "tokenizer")
    manifest["encoder_config"] = _directory_record(root, "encoder_config")
    manifest["wrapper_config"] = _record(root, "wrapper_config.json")
    manifest["candidate_trace_manifest"] = _record(root, "candidate_traces/manifest.json")
    manifest["validation_manifest"] = _record(root, "validation/manifest.json")
    manifest["retained_checkpoints"] = {
        "schema_version": 1,
        "checkpoints": [_directory_record(root, "checkpoint-60")],
    }
    run["final_model"].update(manifest["model"])
    run["tokenizer"] = manifest["tokenizer"]
    run["encoder_config"] = manifest["encoder_config"]
    run["wrapper_config"] = manifest["wrapper_config"]
    run["candidate_traces"]["manifest_sha256"] = manifest["candidate_trace_manifest"]["sha256"]
    run["validation_history"]["manifest_sha256"] = manifest["validation_manifest"]["sha256"]
    run["retained_checkpoints"] = manifest["retained_checkpoints"]
    _write_json(root / "controlled_run.json", run)
    manifest["controlled_run"] = _record(root, "controlled_run.json")
    _write_json(root / "artifact_manifest.json", manifest)
    return ControlledArtifactExpectation(
        artifact_manifest_sha256=_sha256(root / "artifact_manifest.json"),
        experiment_id=CONTROLLED_EXPERIMENT_ID,
        outer_fold=0,
        query_view="structured",
        sampler="local_unique",
        experiment_seed=17,
        dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
        passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
        model_artifact_protocol=CONTROLLED_ARTIFACT_PROTOCOL,
    )


def _edit_json(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    _write_json(path, value)


class ControlledArtifactManifestTests(unittest.TestCase):
    def _fixture(self, tmp: str) -> tuple[Path, ControlledArtifactExpectation]:
        root = Path(tmp) / "artifact"
        return root, _build_artifact(root)

    def test_complete_fixture_validates_without_ml_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, expectation = self._fixture(tmp)
            artifact = validate_controlled_artifact(root, expectation=expectation)
            self.assertEqual(artifact.identity.outer_fold, 0)
            self.assertEqual(artifact.identity.model_artifact_protocol, CONTROLLED_ARTIFACT_PROTOCOL)
            self.assertEqual(
                artifact.identity.experiment_config_sha256,
                EXPECTED_EXPERIMENT_CONFIG_SHA256,
            )
            self.assertEqual(
                artifact.identity.snapshot_tree_sha256,
                EXPECTED_SNAPSHOT_TREE_SHA256,
            )
            self.assertEqual(artifact.model_path, root.resolve() / "model.safetensors")
            self.assertEqual(artifact.tokenizer_dir.name, "tokenizer")
            self.assertEqual(artifact.slot_token_id, 7)
            self.assertEqual(len(artifact.files), 55)

    def test_expectation_is_exact_and_rejects_invalid_domain_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "outer_fold"):
            ControlledArtifactExpectation(
                artifact_manifest_sha256=SHA_A,
                experiment_id=CONTROLLED_EXPERIMENT_ID,
                outer_fold=True,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
                dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
                fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
                passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
                model_artifact_protocol=CONTROLLED_ARTIFACT_PROTOCOL,
            )
        with self.assertRaisesRegex(ValueError, "protocol"):
            ControlledArtifactExpectation(
                artifact_manifest_sha256=SHA_A,
                experiment_id=CONTROLLED_EXPERIMENT_ID,
                outer_fold=0,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
                dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
                fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
                passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
                model_artifact_protocol="legacy",
            )

    def test_rejects_missing_commit_marker_and_wrong_external_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, expectation = self._fixture(tmp)
            (root / "artifact_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "artifact_manifest.json"):
                validate_controlled_artifact(root, expectation=expectation)
        with tempfile.TemporaryDirectory() as tmp:
            root, expectation = self._fixture(tmp)
            wrong = replace(expectation, artifact_manifest_sha256=SHA_D)
            with self.assertRaisesRegex(ValueError, "commit-marker SHA-256"):
                validate_controlled_artifact(root, expectation=wrong)

    def test_rejects_tampered_content_at_every_manifest_layer(self) -> None:
        mutations = {
            "model": lambda root: (root / "model.safetensors").write_bytes(b"tampered\n"),
            "tokenizer": lambda root: (root / "tokenizer/tokenizer.json").write_text("{}\nextra", encoding="utf-8"),
            "encoder": lambda root: (root / "encoder_config/config.json").write_text("{}\n", encoding="utf-8"),
            "trace": lambda root: (root / "candidate_traces/rank-00000.jsonl").write_bytes(b"{}\n"),
            "validation": lambda root: (root / "validation/epoch-001.json").write_text("{}\n", encoding="utf-8"),
            "checkpoint": lambda root: (root / "checkpoint-60/scheduler.pt").write_bytes(b"changed\n"),
            "run": lambda root: (root / "controlled_run.json").write_text("{}\n", encoding="utf-8"),
            "wrapper": lambda root: (root / "wrapper_config.json").write_text("{}\n", encoding="utf-8"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, expectation = self._fixture(tmp)
                mutate(root)
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_extra_empty_stale_and_symlink_entries(self) -> None:
        for name, add_entry in (
            ("extra", lambda root: (root / "extra.txt").write_text("x", encoding="utf-8")),
            ("empty", lambda root: (root / "empty.txt").touch()),
            ("stale", lambda root: (root / ".checkpoint-9.incomplete").mkdir()),
            (
                "symlink",
                lambda root: (root / "tokenizer-link").symlink_to(root / "tokenizer", target_is_directory=True),
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, expectation = self._fixture(tmp)
                add_entry(root)
                with self.assertRaises((ValueError, RuntimeError)):
                    validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_a_symlink_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, expectation = self._fixture(tmp)
            link = Path(tmp) / "artifact-link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                validate_controlled_artifact(link, expectation=expectation)

    def test_rejects_wrong_expected_run_identity(self) -> None:
        changes = {
            "fold": {"outer_fold": 1},
            "view": {"query_view": "flat_masked"},
            "sampler": {"sampler": "global_uniform"},
            "seed": {"experiment_seed": 29},
            "data": {"dataset_manifest_sha256": SHA_D},
            "fold_manifest": {"fold_manifest_sha256": SHA_D},
            "passage_index": {"passage_index_sha256": SHA_D},
        }
        for name, values in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, expectation = self._fixture(tmp)
                with self.assertRaises((ValueError, RuntimeError)):
                    validate_controlled_artifact(root, expectation=replace(expectation, **values))

    def test_rejects_semantically_changed_wrapper_even_when_all_outer_hashes_are_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            wrapper_path = root / "wrapper_config.json"
            wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
            wrapper["temperature"] = 0.05
            _write_json(wrapper_path, wrapper)
            expectation = _refresh_run_and_manifest(root)
            with self.assertRaisesRegex(ValueError, "temperature"):
                validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_refreshed_provenance_outside_the_frozen_study(self) -> None:
        def change_rotation(value: dict[str, Any]) -> None:
            train_cases = value["folds"]["rotation"]["train"]["case_ids"]
            test_cases = value["folds"]["rotation"]["test"]["case_ids"]
            train_cases[0], test_cases[0] = test_cases[0], train_cases[0]
            train_cases.sort(key=int)
            test_cases.sort(key=int)

        mutations = {
            "experiment_config": lambda value: value["experiment_config"].__setitem__(
                "sha256", SHA_D
            ),
            "experiment_config_path": lambda value: value[
                "experiment_config"
            ].__setitem__("path", "renamed-experiment.json"),
            "deepspeed_config": lambda value: value["deepspeed_config"].__setitem__(
                "sha256", SHA_D
            ),
            "dataset_outputs": lambda value: value["dataset"]["output_sha256"].__setitem__(
                "corpus.jsonl", SHA_D
            ),
            "rotation": change_rotation,
            "snapshot_manifest": lambda value: value["snapshot"].__setitem__(
                "manifest_sha256", SHA_D
            ),
            "snapshot_manifest_path": lambda value: value["snapshot"].__setitem__(
                "manifest_path", "renamed-snapshot.json"
            ),
            "snapshot_tree": lambda value: value["snapshot"].__setitem__(
                "tree_sha256", SHA_D
            ),
            "validation_contract": lambda value: value["validation_data"].__setitem__(
                "contract_sha256", SHA_D
            ),
            "trace_merged": lambda value: value["candidate_traces"].__setitem__(
                "merged_sha256", SHA_D
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, _ = self._fixture(tmp)
                _edit_json(root / "controlled_run.json", mutate)
                expectation = _refresh_run_and_manifest(root)
                with self.assertRaisesRegex(ValueError, "frozen|changed"):
                    validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_wrong_nested_inventory_even_when_outer_hashes_are_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            _write_json(root / "tokenizer/unexpected.json", {"unexpected": True})
            expectation = _refresh_run_and_manifest(root)
            with self.assertRaisesRegex(ValueError, "tokenizer.*inventory"):
                validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_semantic_submanifest_changes_after_complete_rehash(self) -> None:
        def change_encoder(root: Path) -> None:
            _edit_json(
                root / "encoder_config/config.json",
                lambda value: value.__setitem__("vocab_size", 50_385),
            )

        def change_trace(root: Path) -> None:
            _edit_json(
                root / "candidate_traces/manifest.json",
                lambda value: value.__setitem__("passage_index_sha256", SHA_D),
            )

        def change_validation(root: Path) -> None:
            _edit_json(
                root / "validation/manifest.json",
                lambda value: value.__setitem__("selection_order", list(reversed(value["selection_order"]))),
            )

        def change_trace_query_identity(root: Path) -> None:
            _edit_json(
                root / "candidate_traces/manifest.json",
                lambda value: value.__setitem__("query_ids_sha256", SHA_D),
            )

        def change_checkpoint(root: Path) -> None:
            _edit_json(
                root / "checkpoint-60/checkpoint_manifest.json",
                lambda value: value.__setitem__("world_size", 3),
            )

        mutations = {
            "encoder": change_encoder,
            "trace": change_trace,
            "trace_query_identity": change_trace_query_identity,
            "validation": change_validation,
            "checkpoint": change_checkpoint,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, _ = self._fixture(tmp)
                mutate(root)
                expectation = _refresh_run_and_manifest(root)
                with self.assertRaises((ValueError, RuntimeError)):
                    validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_changed_run_state_count_after_complete_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            _edit_json(
                root / "controlled_run.json",
                lambda value: (
                    value["final_model"].__setitem__("gathered_tensor_count", 133),
                    value["final_model"].__setitem__("strict_round_trip_tensor_count", 133),
                ),
            )
            expectation = _refresh_run_and_manifest(root)
            with self.assertRaisesRegex(ValueError, "model-state inventory"):
                validate_controlled_artifact(root, expectation=expectation)

    def test_rejects_noncanonical_commit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            manifest_path = root / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            expectation = ControlledArtifactExpectation(
                artifact_manifest_sha256=_sha256(manifest_path),
                experiment_id=CONTROLLED_EXPERIMENT_ID,
                outer_fold=0,
                query_view="structured",
                sampler="local_unique",
                experiment_seed=17,
                dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
                fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
                passage_index_sha256=EXPECTED_PASSAGE_INDEX_SHA256,
                model_artifact_protocol=CONTROLLED_ARTIFACT_PROTOCOL,
            )
            with self.assertRaisesRegex(ValueError, "canonical"):
                validate_controlled_artifact(root, expectation=expectation)


class _FakeBoolTensor:
    def all(self) -> bool:
        return True


class _FakeTensor:
    def __init__(self, dtype: object, *, rows: int | None = None) -> None:
        self.dtype = dtype
        self._rows = rows

    def is_floating_point(self) -> bool:
        return True

    def __len__(self) -> int:
        if self._rows is None:
            raise TypeError("not a row tensor")
        return self._rows


class _FakeDevice:
    def __init__(self, value: str) -> None:
        if value not in ("cpu", "cuda", "cuda:0"):
            raise ValueError(value)
        self.value = value
        self.type = value.split(":", 1)[0]
        self.index = int(value.split(":", 1)[1]) if ":" in value else None

    def __str__(self) -> str:
        return self.value


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


class _FakeTorch:
    bfloat16 = object()
    cuda = _FakeCuda()

    @staticmethod
    def device(value: str) -> _FakeDevice:
        return _FakeDevice(value)

    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensor)

    @staticmethod
    def isfinite(value: _FakeTensor) -> _FakeBoolTensor:
        return _FakeBoolTensor()


class _FakeTokenizer:
    unk_token_id = 0

    def __len__(self) -> int:
        return CONTROLLED_TOKENIZER_SIZE

    def convert_tokens_to_ids(self, token: str) -> int:
        return 7 if token == CONTROLLED_SLOT_TOKEN else self.unk_token_id


class _FakeAutoTokenizer:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> _FakeTokenizer:
        cls.calls.append((path, kwargs))
        return _FakeTokenizer()


class _FakeConfig:
    model_type = "modernbert"
    vocab_size = CONTROLLED_TOKENIZER_SIZE
    deterministic_flash_attn = True
    reference_compile = False
    _attn_implementation = "flash_attention_2"


class _FakeAutoConfig:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> _FakeConfig:
        cls.calls.append((path, kwargs))
        return _FakeConfig()


class _FakeAttention:
    deterministic_flash_attn = True


class _FakeEncoder:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self._embedding = SimpleNamespace(
            weight=_FakeTensor(_FakeTorch.bfloat16, rows=CONTROLLED_TOKENIZER_SIZE)
        )

    def modules(self):
        return iter((self, *(_FakeAttention() for _ in range(CONTROLLED_ATTENTION_MODULE_COUNT))))

    def get_input_embeddings(self):
        return self._embedding


class _FakeAutoModel:
    calls: list[tuple[_FakeConfig, dict[str, Any]]] = []

    @classmethod
    def from_config(cls, config: _FakeConfig, **kwargs: Any) -> _FakeEncoder:
        cls.calls.append((config, kwargs))
        return _FakeEncoder()


class _FakeRetriever:
    def __init__(self, *, encoder: _FakeEncoder, slot_token_id: int, temperature: float) -> None:
        self.encoder = encoder
        self.slot_token_id = slot_token_id
        self.temperature = temperature
        self._state = {
            f"encoder.weight_{index}": _FakeTensor(_FakeTorch.bfloat16)
            for index in range(CONTROLLED_MODEL_STATE_COUNT)
        }
        self.training = True
        self.device = "cpu"

    def state_dict(self):
        return self._state

    def named_parameters(self):
        return iter((("encoder.weight", object()),))

    def to(self, device: _FakeDevice):
        self.device = str(device)
        return self

    def eval(self):
        self.training = False
        return self


class ControlledArtifactRuntimeTests(unittest.TestCase):
    def _runtime(self, loader: Callable[..., Any] | None = None) -> ControlledArtifactRuntime:
        def strict_loader(model: Any, path: Path, **kwargs: Any):
            self.assertIsInstance(model, _FakeRetriever)
            self.assertEqual(path.name, "model.safetensors")
            self.assertEqual(kwargs, {"strict": True, "device": "cpu"})
            return ([], [])

        return ControlledArtifactRuntime(
            torch_module=_FakeTorch,
            auto_tokenizer_class=_FakeAutoTokenizer,
            auto_config_class=_FakeAutoConfig,
            auto_model_class=_FakeAutoModel,
            load_safetensors_model=loader or strict_loader,
            retriever_class=_FakeRetriever,
        )

    def test_dependency_injected_loader_is_local_strict_bf16_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            expectation = _build_artifact(root)
            artifact = validate_controlled_artifact(root, expectation=expectation)
            loaded = load_controlled_retriever(
                artifact,
                device="cpu",
                runtime=self._runtime(),
            )
            self.assertEqual(loaded.device, "cpu")
            self.assertFalse(loaded.model.training)
            self.assertEqual(
                _FakeAutoTokenizer.calls[-1][1],
                {"use_fast": True, "local_files_only": True, "trust_remote_code": False},
            )
            self.assertEqual(
                _FakeAutoConfig.calls[-1][1],
                {"local_files_only": True, "trust_remote_code": False},
            )
            self.assertEqual(
                _FakeAutoModel.calls[-1][1],
                {
                    "trust_remote_code": False,
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": _FakeTorch.bfloat16,
                },
            )

    def test_loader_rejects_auto_or_unavailable_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            artifact = validate_controlled_artifact(root, expectation=_build_artifact(root))
            with self.assertRaisesRegex(ValueError, "explicit"):
                load_controlled_retriever(artifact, device="auto", runtime=self._runtime())
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                load_controlled_retriever(artifact, device="cuda", runtime=self._runtime())

    def test_loader_rejects_any_strict_safetensors_incompatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            artifact = validate_controlled_artifact(root, expectation=_build_artifact(root))
            runtime = self._runtime(loader=lambda *args, **kwargs: (["missing"], []))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                load_controlled_retriever(artifact, device="cpu", runtime=runtime)

    def test_loader_revalidates_bytes_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            artifact = validate_controlled_artifact(root, expectation=_build_artifact(root))
            (root / "model.safetensors").write_bytes(b"changed-after-validation\n")
            with self.assertRaisesRegex(ValueError, "model.*(?:size|SHA-256)"):
                load_controlled_retriever(artifact, device="cpu", runtime=self._runtime())


@unittest.skipUnless(
    os.environ.get("ARR_TOKENIZER_DIR"),
    "ARR_TOKENIZER_DIR is required for the pinned ModernBERT artifact runtime gate",
)
class ControlledArtifactPinnedSnapshotTests(unittest.TestCase):
    def test_real_modernbert_tied_bf16_artifact_loads_strictly(self) -> None:
        import torch
        from safetensors.torch import save_model
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        from retriever.markup import SLOT_TOKEN, all_markup_tokens
        from retriever.models import DualEncoderRetriever

        snapshot_dir = Path(os.environ["ARR_TOKENIZER_DIR"]).resolve(strict=True)

        def real_assets(root: Path) -> int:
            tokenizer = AutoTokenizer.from_pretrained(
                str(snapshot_dir),
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": all_markup_tokens()}
            )
            self.assertEqual(added, 19)
            self.assertEqual(len(tokenizer), CONTROLLED_TOKENIZER_SIZE)
            slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
            config = AutoConfig.from_pretrained(
                str(snapshot_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
            config.deterministic_flash_attn = True
            config.reference_compile = False
            # Match the existing pinned-snapshot factory gate: enable only the
            # Transformers availability predicate while construction remains
            # on CPU and no FlashAttention kernel is executed.
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                encoder = AutoModel.from_pretrained(
                    str(snapshot_dir),
                    config=config,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            encoder.resize_token_embeddings(len(tokenizer))
            model = DualEncoderRetriever(
                encoder=encoder,
                slot_token_id=slot_token_id,
                temperature=0.07,
            )
            save_model(model, str(root / "model.safetensors"))
            tokenizer.save_pretrained(str(root / "tokenizer"))
            encoder.config.save_pretrained(str(root / "encoder_config"))
            return slot_token_id

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifact"
            expectation = _build_artifact(root, asset_writer=real_assets)
            artifact = validate_controlled_artifact(root, expectation=expectation)
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                loaded = load_controlled_retriever(
                    artifact,
                    device="cpu",
                    runtime=import_pinned_artifact_runtime(),
                )
            self.assertEqual(loaded.identity.model_sha256, _sha256(root / "model.safetensors"))
            self.assertTrue(
                all(
                    tensor.dtype == torch.bfloat16
                    for tensor in loaded.model.state_dict().values()
                    if tensor.is_floating_point()
                )
            )


if __name__ == "__main__":
    unittest.main()
