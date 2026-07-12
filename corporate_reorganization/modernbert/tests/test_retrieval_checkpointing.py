from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import retriever.checkpointing as checkpointing  # noqa: E402


WORLD_SIZE = 2


def _selection(
    *,
    epoch: int,
    global_step: int,
    primary_metric: float = 0.5,
    secondary_metric: float = 0.25,
) -> checkpointing.CheckpointSelection:
    return checkpointing.CheckpointSelection(
        schema_version=checkpointing.SELECTION_METADATA_SCHEMA_VERSION,
        epoch=epoch,
        global_step=global_step,
        checkpoint_dir=f"checkpoint-{global_step}",
        deepspeed_tag=f"global_step{global_step}",
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        ranking_sha256=hashlib.sha256(
            f"ranking-{epoch}-{global_step}".encode("utf-8")
        ).hexdigest(),
    )


def _client_state(selection: checkpointing.CheckpointSelection) -> dict[str, object]:
    return {
        "controlled_state": {
            "schema_version": 1,
            "epoch": selection.epoch,
            "global_step": selection.global_step,
        }
    }


def _client_state_sha256(client_state: object) -> str:
    normalized = checkpointing._jsonable_state(client_state)
    return hashlib.sha256(
        checkpointing.canonical_json(normalized).encode("utf-8")
    ).hexdigest()


def _scheduler_state(global_step: int) -> dict[str, object]:
    return {
        "last_epoch": global_step,
        "base_lrs": [1e-5],
        "_last_lr": [1e-5],
        "marker": f"scheduler-{global_step}",
    }


def _write_exact_checkpoint_fixture(
    output_dir: Path,
    selection: checkpointing.CheckpointSelection,
) -> Path:
    checkpoint_root = output_dir / selection.checkpoint_dir
    checkpoint_root.mkdir()
    tag_dir = checkpoint_root / selection.deepspeed_tag
    tag_dir.mkdir()
    for rank in range(WORLD_SIZE):
        (tag_dir / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").write_bytes(
            f"model-state-rank-{rank}".encode("utf-8")
        )
        (
            tag_dir / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(f"optimizer-state-rank-{rank}".encode("utf-8"))
    (checkpoint_root / "zero_to_fp32.py").write_text(
        "# exact DeepSpeed recovery fixture\n",
        encoding="utf-8",
    )
    scheduler_state = _scheduler_state(selection.global_step)
    torch.save(scheduler_state, checkpoint_root / "scheduler.pt")
    torch.save({"fixture": True}, checkpoint_root / "training_args.bin")
    (checkpoint_root / "trainer_state.json").write_text(
        json.dumps(
            {"epoch": selection.epoch, "global_step": selection.global_step},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    for rank in range(WORLD_SIZE):
        torch.save(rng_state, checkpoint_root / f"rng_state_{rank}.pth")

    client_state = _client_state(selection)
    manifest = {
        "schema_version": checkpointing.CHECKPOINT_PROTOCOL_SCHEMA_VERSION,
        "selection": selection.to_payload(),
        "world_size": WORLD_SIZE,
        "client_state_sha256": _client_state_sha256(client_state),
        "scheduler_state_sha256": checkpointing.canonical_state_sha256(
            scheduler_state
        ),
        "rng_files": [f"rng_state_{rank}.pth" for rank in range(WORLD_SIZE)],
        "files": checkpointing._tree_inventory(
            checkpoint_root,
            include_hashes=True,
        ),
    }
    (checkpoint_root / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_root


class _FakeScheduler:
    def __init__(self, global_step: int) -> None:
        self._state = _scheduler_state(global_step)
        self.last_epoch = global_step

    def state_dict(self) -> dict[str, object]:
        return copy.deepcopy(self._state)

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._state = copy.deepcopy(state)
        self.last_epoch = state["last_epoch"]


class _FakeTrainerState:
    def __init__(self, selection: checkpointing.CheckpointSelection) -> None:
        self.global_step = selection.global_step
        self.epoch = float(selection.epoch)

    def save_to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(
                {"epoch": self.epoch, "global_step": self.global_step},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class _FakeDeepSpeedEngine:
    def __init__(
        self,
        *,
        rank: int,
        global_step: int,
        loaded_client_state: dict[str, object] | None = None,
        save_return: object = True,
        return_wrong_load_path: bool = False,
    ) -> None:
        self.rank = rank
        self.global_steps = global_step
        self.optimizer = object()
        self.lr_scheduler = None
        self.dp_world_size = WORLD_SIZE
        self.loaded_client_state = loaded_client_state
        self.save_return = save_return
        self.return_wrong_load_path = return_wrong_load_path

    @staticmethod
    def zero_optimization_stage() -> int:
        return 3

    @staticmethod
    def bfloat16_enabled() -> bool:
        return True

    def save_checkpoint(
        self,
        save_dir: str,
        *,
        tag: str,
        client_state: dict[str, object],
        save_latest: bool,
        exclude_frozen_parameters: bool,
    ) -> object:
        if save_latest is not False:
            raise AssertionError("Controlled checkpoints must disable DeepSpeed latest")
        if exclude_frozen_parameters is not False:
            raise AssertionError("Controlled checkpoints must save every parameter")
        if client_state != _client_state(
            _selection(
                epoch=client_state["controlled_state"]["epoch"],
                global_step=client_state["controlled_state"]["global_step"],
            )
        ):
            raise AssertionError("DeepSpeed received a non-exact client-state namespace")
        expected_tag = f"global_step{self.global_steps}"
        if tag != expected_tag:
            raise AssertionError(f"DeepSpeed tag={tag}; expected {expected_tag}")

        checkpoint_root = Path(save_dir)
        tag_dir = checkpoint_root / tag
        tag_dir.mkdir(exist_ok=True)
        (tag_dir / f"zero_pp_rank_{self.rank}_mp_rank_00_model_states.pt").write_bytes(
            f"model-state-rank-{self.rank}".encode("utf-8")
        )
        (
            tag_dir
            / f"bf16_zero_pp_rank_{self.rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(f"optimizer-state-rank-{self.rank}".encode("utf-8"))
        if self.rank == 0:
            (checkpoint_root / "zero_to_fp32.py").write_text(
                "# exact DeepSpeed recovery fixture\n",
                encoding="utf-8",
            )
        dist.barrier()
        return self.save_return

    def load_checkpoint(
        self,
        load_dir: str,
        *,
        tag: str,
        load_module_strict: bool,
        load_optimizer_states: bool,
        load_lr_scheduler_states: bool,
        load_module_only: bool,
    ) -> tuple[str, dict[str, object]]:
        expected_options = (
            load_module_strict,
            load_optimizer_states,
            load_lr_scheduler_states,
            load_module_only,
        )
        if expected_options != (True, True, True, False):
            raise AssertionError(
                f"DeepSpeed full-load options changed: {expected_options}"
            )
        if self.loaded_client_state is None:
            raise AssertionError("Fake load requires the checkpoint client state")
        self.global_steps = int(tag.removeprefix("global_step"))
        tag_dir = Path(load_dir) / tag
        load_path = (
            tag_dir
            if self.return_wrong_load_path
            else tag_dir / f"zero_pp_rank_{self.rank}_mp_rank_00_model_states.pt"
        )
        deep_speed_client_state = {
            **self.loaded_client_state,
            "ds_config": "internal DeepSpeed field",
            "global_steps": self.global_steps,
        }
        return str(load_path), deep_speed_client_state


def _distributed_checkpoint_worker(rank: int, init_file: str, root_path: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        output_dir = Path(root_path) / "checkpoints"
        selections = (
            _selection(epoch=1, global_step=3, primary_metric=0.4),
            _selection(epoch=2, global_step=6, primary_metric=0.6),
            _selection(epoch=3, global_step=9, primary_metric=0.55),
        )
        metadata_store = checkpointing.ValidationMetadataStore(
            output_dir,
            expected_epochs=len(selections),
        )
        for selection in selections:
            engine = _FakeDeepSpeedEngine(
                rank=rank,
                global_step=selection.global_step,
            )
            scheduler = _FakeScheduler(selection.global_step)
            metadata = checkpointing.save_controlled_checkpoint(
                output_dir=output_dir,
                engine=engine,
                scheduler=scheduler,
                trainer_state=_FakeTrainerState(selection),
                training_args={"global_step": selection.global_step},
                selection=selection,
                client_state=_client_state(selection),
                expected_world_size=WORLD_SIZE,
            )
            if metadata["checkpoint_dir"] != selection.checkpoint_dir:
                raise AssertionError("Collective checkpoint publication returned the wrong path")
            best, is_new_best, history_entry = metadata_store.register_checkpoint(
                candidate=selection,
                validation_result={
                    "schema_version": 1,
                    "metrics": {
                        checkpointing.VALIDATION_PRIMARY_METRIC: selection.primary_metric,
                        checkpointing.VALIDATION_SECONDARY_METRIC: selection.secondary_metric,
                    },
                    "ranking_sha256": selection.ranking_sha256,
                },
                checkpoint_metadata=metadata,
            )
            expected_best = selections[1] if selection.epoch >= 2 else selections[0]
            if best != expected_best or is_new_best != (selection.epoch in {1, 2}):
                raise AssertionError("Validation metadata selection trajectory changed")
            if history_entry["epoch"] != selection.epoch:
                raise AssertionError("Validation history entry changed epoch")

        loaded_scheduler = _FakeScheduler(0)
        loaded_engine = _FakeDeepSpeedEngine(
            rank=rank,
            global_step=0,
            loaded_client_state=_client_state(selections[0]),
        )
        loaded = checkpointing.load_controlled_checkpoint(
            checkpoint_root=output_dir / selections[0].checkpoint_dir,
            engine=loaded_engine,
            scheduler=loaded_scheduler,
            selection=selections[0],
            expected_world_size=WORLD_SIZE,
            restore_rng=False,
        )
        if loaded["global_step"] != selections[0].global_step:
            raise AssertionError("Full checkpoint load did not restore the global step")
        if loaded_scheduler.last_epoch != selections[0].global_step:
            raise AssertionError("Full checkpoint load did not restore the scheduler")

        wrong_path_engine = _FakeDeepSpeedEngine(
            rank=rank,
            global_step=0,
            loaded_client_state=_client_state(selections[0]),
            return_wrong_load_path=True,
        )
        try:
            checkpointing.load_controlled_checkpoint(
                checkpoint_root=output_dir / selections[0].checkpoint_dir,
                engine=wrong_path_engine,
                scheduler=_FakeScheduler(0),
                selection=selections[0],
                expected_world_size=WORLD_SIZE,
                restore_rng=False,
            )
        except RuntimeError as error:
            if "expected" not in str(error):
                raise AssertionError(f"Unexpected wrong-load-path failure: {error}") from error
        else:
            raise AssertionError("A non-rank-local DeepSpeed load path was accepted")

        kept = checkpointing.retain_best_and_last_checkpoints(
            output_dir,
            best_checkpoint_dir=selections[1].checkpoint_dir,
            last_checkpoint_dir=selections[2].checkpoint_dir,
        )
        if kept != (selections[1].checkpoint_dir, selections[2].checkpoint_dir):
            raise AssertionError(f"Checkpoint retention returned {kept}")
        inventory = checkpointing.retained_checkpoint_inventory(output_dir, kept)
        if [item["path"] for item in inventory["checkpoints"]] != list(kept):
            raise AssertionError("Retained checkpoint inventory changed order or identity")
        validation_manifest = metadata_store.finalize(
            retained_checkpoint_dirs=kept,
        )
        if (
            validation_manifest["best"] != selections[1].to_payload()
            or validation_manifest["last"] != selections[2].to_payload()
            or validation_manifest["retained_checkpoint_dirs"] != list(kept)
            or len(validation_manifest["records"]) != len(selections)
        ):
            raise AssertionError("Final validation metadata manifest changed")
        divergent_retained = kept if rank == 0 else (selections[2].checkpoint_dir,)
        try:
            metadata_store.finalize(retained_checkpoint_dirs=divergent_retained)
        except RuntimeError as error:
            if "failed collectively" not in str(error):
                raise AssertionError(
                    f"Unexpected metadata-finalization failure: {error}"
                ) from error
        else:
            raise AssertionError("Rank-divergent metadata finalization was accepted")

        rank_specific_best = (
            selections[1].checkpoint_dir
            if rank == 0
            else selections[2].checkpoint_dir
        )
        try:
            checkpointing.retain_best_and_last_checkpoints(
                output_dir,
                best_checkpoint_dir=rank_specific_best,
                last_checkpoint_dir=selections[2].checkpoint_dir,
            )
        except RuntimeError as error:
            if "differs across ranks" not in str(error):
                raise AssertionError(f"Unexpected retention-preflight failure: {error}") from error
        else:
            raise AssertionError("Rank-divergent retention targets were accepted")

        failed_output = Path(root_path) / "failed-save"
        failed_selection = _selection(epoch=4, global_step=12)
        try:
            checkpointing.save_controlled_checkpoint(
                output_dir=failed_output,
                engine=_FakeDeepSpeedEngine(
                    rank=rank,
                    global_step=failed_selection.global_step,
                    save_return=1,
                ),
                scheduler=_FakeScheduler(failed_selection.global_step),
                trainer_state=_FakeTrainerState(failed_selection),
                training_args={"global_step": failed_selection.global_step},
                selection=failed_selection,
                client_state=_client_state(failed_selection),
                expected_world_size=WORLD_SIZE,
            )
        except RuntimeError as error:
            if "exact True" not in str(error):
                raise AssertionError(f"Unexpected exact-return failure: {error}") from error
        else:
            raise AssertionError("DeepSpeed truthy non-True save result was accepted")
        if (failed_output / failed_selection.checkpoint_dir).exists():
            raise AssertionError("A failed checkpoint was published")
        if not (
            failed_output / f".{failed_selection.checkpoint_dir}.incomplete"
        ).is_dir():
            raise AssertionError("A failed checkpoint lost its diagnostic incomplete tree")

        divergent_output = Path(root_path) / "divergent-metadata"
        divergent_store = checkpointing.ValidationMetadataStore(
            divergent_output,
            expected_epochs=1,
        )
        divergent_selection = _selection(epoch=1, global_step=3)
        divergent_primary = (
            divergent_selection.primary_metric if rank == 0 else 0.75
        )
        fake_digest = "0" * 64
        try:
            divergent_store.register_checkpoint(
                candidate=divergent_selection,
                validation_result={
                    "metrics": {
                        checkpointing.VALIDATION_PRIMARY_METRIC: divergent_primary,
                        checkpointing.VALIDATION_SECONDARY_METRIC: (
                            divergent_selection.secondary_metric
                        ),
                    },
                    "ranking_sha256": divergent_selection.ranking_sha256,
                },
                checkpoint_metadata={
                    "checkpoint_dir": divergent_selection.checkpoint_dir,
                    "deepspeed_tag": divergent_selection.deepspeed_tag,
                    "manifest_sha256": fake_digest,
                    "scheduler_state_sha256": fake_digest,
                    "client_state_sha256": fake_digest,
                },
            )
        except RuntimeError as error:
            if "failed collectively" not in str(error):
                raise AssertionError(
                    f"Unexpected metadata-registration failure: {error}"
                ) from error
        else:
            raise AssertionError("Rank-divergent validation registration was accepted")
        if list((divergent_output / "validation").glob("epoch-*.json")):
            raise AssertionError("Failed metadata registration published an epoch record")
    finally:
        dist.destroy_process_group()


class CheckpointSelectionTest(unittest.TestCase):
    def test_exact_schema_and_lexicographic_selection(self) -> None:
        first = _selection(epoch=1, global_step=3, primary_metric=0.5, secondary_metric=0.2)
        checkpointing.validate_selection(first)
        self.assertEqual(checkpointing.choose_better_checkpoint(None, first), (first, True))

        primary_better = _selection(
            epoch=2,
            global_step=6,
            primary_metric=0.6,
            secondary_metric=0.1,
        )
        self.assertEqual(
            checkpointing.choose_better_checkpoint(first, primary_better),
            (primary_better, True),
        )
        secondary_better = _selection(
            epoch=2,
            global_step=6,
            primary_metric=0.5,
            secondary_metric=0.3,
        )
        self.assertEqual(
            checkpointing.choose_better_checkpoint(first, secondary_better),
            (secondary_better, True),
        )
        exact_tie = _selection(
            epoch=2,
            global_step=6,
            primary_metric=0.5,
            secondary_metric=0.2,
        )
        self.assertEqual(
            checkpointing.choose_better_checkpoint(first, exact_tie),
            (first, False),
        )
        primary_worse = _selection(
            epoch=2,
            global_step=6,
            primary_metric=0.4,
            secondary_metric=0.9,
        )
        self.assertEqual(
            checkpointing.choose_better_checkpoint(first, primary_worse),
            (first, False),
        )
        with self.assertRaisesRegex(ValueError, "increasing epoch order"):
            checkpointing.choose_better_checkpoint(first, first)

    def test_nonexact_selection_fields_fail_loudly(self) -> None:
        exact = _selection(epoch=1, global_step=3)
        invalid = (
            replace(exact, schema_version=True),
            replace(exact, epoch=True),
            replace(exact, global_step=True),
            replace(exact, checkpoint_dir=Path("checkpoint-3")),
            replace(exact, deepspeed_tag=Path("global_step3")),
            replace(exact, primary_metric=1),
            replace(exact, secondary_metric=float("nan")),
            replace(exact, ranking_sha256=exact.ranking_sha256.upper()),
        )
        for selection in invalid:
            with self.subTest(selection=selection):
                with self.assertRaises((TypeError, ValueError)):
                    checkpointing.validate_selection(selection)


class CheckpointManifestTest(unittest.TestCase):
    def test_exact_manifest_and_retained_inventory_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            selection = _selection(epoch=1, global_step=3)
            checkpoint_root = _write_exact_checkpoint_fixture(output_dir, selection)
            manifest = checkpointing._load_checkpoint_manifest(checkpoint_root)
            self.assertEqual(manifest["selection"], selection.to_payload())
            inventory = checkpointing.retained_checkpoint_inventory(
                output_dir,
                [selection.checkpoint_dir],
            )
            self.assertEqual(
                [item["path"] for item in inventory["checkpoints"]],
                [selection.checkpoint_dir],
            )
            inventory_paths = {
                record["path"] for record in inventory["checkpoints"][0]["files"]
            }
            self.assertIn("checkpoint_manifest.json", inventory_paths)

    def test_nonexact_deepspeed_layouts_fail_loudly(self) -> None:
        mutations = (
            "latest-file",
            "renamed-model-shard",
            "extra-tag-file",
            "extra-root-directory",
            "root-symlink",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                selection = _selection(epoch=1, global_step=3)
                checkpoint_root = _write_exact_checkpoint_fixture(Path(temp_dir), selection)
                tag_dir = checkpoint_root / selection.deepspeed_tag
                if mutation == "latest-file":
                    (checkpoint_root / "latest").write_text(
                        selection.deepspeed_tag,
                        encoding="utf-8",
                    )
                elif mutation == "renamed-model-shard":
                    source = tag_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt"
                    source.rename(tag_dir / "mp_rank_00_model_states.pt")
                elif mutation == "extra-tag-file":
                    (tag_dir / "unexpected.pt").write_bytes(b"unexpected")
                elif mutation == "extra-root-directory":
                    (checkpoint_root / "unexpected-directory").mkdir()
                elif mutation == "root-symlink":
                    (checkpoint_root / "unsafe-link").symlink_to("zero_to_fp32.py")
                with self.assertRaises(RuntimeError):
                    checkpointing._validate_deepspeed_layout(
                        checkpoint_root,
                        tag=selection.deepspeed_tag,
                        world_size=WORLD_SIZE,
                    )

    def test_manifest_corruption_and_schema_drift_fail_loudly(self) -> None:
        mutations = (
            "corrupt-shard",
            "boolean-schema-version",
            "boolean-world-size",
            "uppercase-digest",
            "noncanonical-file-order",
            "extra-recorded-file",
            "reordered-rng-files",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                selection = _selection(epoch=1, global_step=3)
                checkpoint_root = _write_exact_checkpoint_fixture(Path(temp_dir), selection)
                manifest_path = checkpoint_root / "checkpoint_manifest.json"
                if mutation == "corrupt-shard":
                    shard_path = (
                        checkpoint_root
                        / selection.deepspeed_tag
                        / "zero_pp_rank_0_mp_rank_00_model_states.pt"
                    )
                    with shard_path.open("ab") as target:
                        target.write(b"corruption")
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if mutation == "boolean-schema-version":
                        manifest["schema_version"] = True
                    elif mutation == "boolean-world-size":
                        manifest["world_size"] = True
                    elif mutation == "uppercase-digest":
                        manifest["client_state_sha256"] = manifest[
                            "client_state_sha256"
                        ].upper()
                    elif mutation == "noncanonical-file-order":
                        manifest["files"] = list(reversed(manifest["files"]))
                    elif mutation == "extra-recorded-file":
                        intruder = checkpoint_root / "intruder.bin"
                        intruder.write_bytes(b"intruder")
                        manifest["files"].append(
                            {
                                "path": intruder.name,
                                "size": intruder.stat().st_size,
                                "sha256": checkpointing.sha256_file(intruder),
                            }
                        )
                        manifest["files"].sort(key=lambda record: record["path"])
                    elif mutation == "reordered-rng-files":
                        manifest["rng_files"] = list(reversed(manifest["rng_files"]))
                    manifest_path.write_text(
                        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaises((RuntimeError, ValueError)):
                    checkpointing._load_checkpoint_manifest(checkpoint_root)


class DistributedCheckpointLifecycleTest(unittest.TestCase):
    def test_exact_collective_save_full_load_and_best_last_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_output = root / "checkpoints"
            failed_output = root / "failed-save"
            divergent_metadata_output = root / "divergent-metadata"
            checkpoint_output.mkdir()
            failed_output.mkdir()
            divergent_metadata_output.mkdir()
            mp.spawn(
                _distributed_checkpoint_worker,
                args=(str(root / "process-group"), str(root)),
                nprocs=WORLD_SIZE,
                join=True,
            )
            self.assertEqual(
                sorted(path.name for path in checkpoint_output.glob("checkpoint-*")),
                ["checkpoint-6", "checkpoint-9"],
            )
            for checkpoint_name in ("checkpoint-6", "checkpoint-9"):
                checkpointing._load_checkpoint_manifest(
                    checkpoint_output / checkpoint_name
                )
            validation_manifest = json.loads(
                (checkpoint_output / "validation/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validation_manifest["best"]["checkpoint_dir"], "checkpoint-6")
            self.assertEqual(validation_manifest["last"]["checkpoint_dir"], "checkpoint-9")
            self.assertEqual(len(validation_manifest["records"]), 3)
            self.assertFalse((failed_output / "checkpoint-12").exists())
            self.assertTrue((failed_output / ".checkpoint-12.incomplete").is_dir())


if __name__ == "__main__":
    unittest.main()
