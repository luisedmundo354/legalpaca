from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from retriever.provenance import (
    EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_DEEPSPEED_CONFIG_SHA256,
    EXPECTED_EXPERIMENT_CONFIG_SHA256,
    EXPECTED_FOLD_MANIFEST_LOGICAL_PATH,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_IMAGE,
    load_snapshot_manifest,
    validate_preimport_environment,
    validate_runtime_versions,
    validate_snapshot_directory,
)
from retriever.sampling import (
    SAMPLER_GLOBAL_UNIFORM,
    SAMPLER_LOCAL_UNIQUE,
)
from retriever.staged_data import validate_staged_dataset_and_fold


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_CONFIG = SOURCE_DIR / "experiments/retrieval_cv/configs/experiment.json"
DEFAULT_FOLDS_CONFIG = SOURCE_DIR / "experiments/retrieval_cv/configs/folds.json"
DEFAULT_SNAPSHOT_MANIFEST = SOURCE_DIR / "experiments/retrieval_cv/configs/modernbert_snapshot.json"
DEFAULT_DEEPSPEED_CONFIG = SOURCE_DIR / "ds_zero3.json"

CONTROLLED_QUERY_VIEWS = ("structured", "flat_masked")
CONTROLLED_SAMPLERS = (SAMPLER_LOCAL_UNIQUE, SAMPLER_GLOBAL_UNIFORM)
CONTROLLED_SEEDS = (17, 29, 43)
EXPECTED_MODEL_SELECTION_PRIMARY = "validation_case_macro_set_recall_at_20"
EXPECTED_MODEL_SELECTION_SECONDARY = (
    "validation_case_macro_first_gold_reciprocal_rank_full_ranking"
)


def _exact_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict ARR case-disjoint controlled retrieval training",
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("SM_CHANNEL_DATA"))
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=os.environ.get("SM_CHANNEL_BASE_MODEL"),
    )
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--folds-config", type=Path, default=DEFAULT_FOLDS_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT_MANIFEST)
    parser.add_argument("--deepspeed-config", type=Path, default=DEFAULT_DEEPSPEED_CONFIG)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--query-view", choices=CONTROLLED_QUERY_VIEWS, required=True)
    parser.add_argument("--sampler", choices=CONTROLLED_SAMPLERS, required=True)
    parser.add_argument("--experiment-seed", type=int, choices=CONTROLLED_SEEDS, required=True)
    args = parser.parse_args(argv)

    for field in ("data_dir", "base_model_dir", "output_dir"):
        if getattr(args, field) is None:
            parser.error(
                f"--{field.replace('_', '-')} is required, either explicitly or through its exact "
                "SageMaker channel/output environment variable"
            )
    return args


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_file_record(path: Path, *, logical_path: str) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Artifact record source must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size < 1:
        raise ValueError(f"Artifact record source must be non-empty: {path}")
    return {"path": logical_path, "size": size, "sha256": _sha256(path)}


def _validate_frozen_control_file_hashes(
    *,
    experiment_config_path: Path,
    deepspeed_config_path: Path,
) -> None:
    expected = {
        "Experiment config": (
            Path(experiment_config_path),
            EXPECTED_EXPERIMENT_CONFIG_SHA256,
        ),
        "DeepSpeed config": (
            Path(deepspeed_config_path),
            EXPECTED_DEEPSPEED_CONFIG_SHA256,
        ),
    }
    for name, (path, expected_sha256) in expected.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} must be a regular non-symlink file: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{name} bytes changed from the frozen study: "
                f"actual={actual_sha256}, expected={expected_sha256}"
            )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Artifact directory must be a real directory: {path}")
    records: list[dict[str, Any]] = []
    for entry in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = entry.relative_to(path).as_posix()
        if entry.is_symlink():
            raise ValueError(f"Artifact directory forbids symlink: {relative}")
        if entry.is_dir():
            continue
        if not entry.is_file() or entry.stat().st_size < 1:
            raise ValueError(f"Artifact file must be non-empty and regular: {relative}")
        records.append(
            {
                "path": relative,
                "size": entry.stat().st_size,
                "sha256": _sha256(entry),
            }
        )
    if not records:
        raise ValueError(f"Artifact directory is empty: {path}")
    return records


def _publish_new_binary(path: Path, writer: Callable[[Path], None]) -> dict[str, Any]:
    temporary_path = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite binary artifact: {path}")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"Refusing stale binary temporary artifact: {temporary_path}")
    published = False
    try:
        writer(temporary_path)
        if (
            not temporary_path.is_file()
            or temporary_path.is_symlink()
            or temporary_path.stat().st_size < 1
        ):
            raise RuntimeError(f"Binary writer did not create one non-empty file: {temporary_path}")
        with temporary_path.open("rb") as source:
            os.fsync(source.fileno())
        os.link(temporary_path, path)
        published = True
        temporary_path.unlink()
        _fsync_directory(path.parent)
        return {"path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
    except BaseException:
        if published and (path.exists() or path.is_symlink()):
            path.unlink()
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()
        _fsync_directory(path.parent)
        raise


def _publish_pretrained_directory(
    path: Path,
    writer: Callable[[Path], object],
) -> dict[str, Any]:
    temporary_path = path.with_name(f".{path.name}.incomplete")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite artifact directory: {path}")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"Refusing stale artifact directory: {temporary_path}")
    temporary_path.mkdir()
    renamed = False
    try:
        writer(temporary_path)
        inventory = _directory_inventory(temporary_path)
        for entry in temporary_path.rglob("*"):
            if entry.is_file():
                with entry.open("rb") as source:
                    os.fsync(source.fileno())
        directories = sorted(
            (entry for entry in temporary_path.rglob("*") if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        )
        for directory in directories:
            _fsync_directory(directory)
        _fsync_directory(temporary_path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Artifact target appeared before publication: {path}")
        os.rename(temporary_path, path)
        renamed = True
        _fsync_directory(path.parent)
        return {"path": path.name, "files": inventory}
    except BaseException:
        if renamed and path.exists() and not temporary_path.exists():
            os.rename(path, temporary_path)
            _fsync_directory(path.parent)
        if temporary_path.exists() and temporary_path.is_dir():
            import shutil

            shutil.rmtree(temporary_path)
            _fsync_directory(path.parent)
        raise


def _validate_gathered_bf16_state_dict(state_dict: object, torch_module) -> Mapping[str, Any]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("Rank zero did not receive a non-empty gathered model state")
    keys = list(state_dict)
    if any(type(key) is not str or not key for key in keys) or len(keys) != len(set(keys)):
        raise RuntimeError("Gathered model state has invalid parameter names")
    floating_count = 0
    for key in keys:
        tensor = state_dict[key]
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"Gathered model state {key!r} is not a tensor")
        if tensor.device.type != "cpu":
            raise RuntimeError(f"Gathered model state {key!r} was not offloaded to CPU")
        if tensor.is_floating_point():
            floating_count += 1
            if tensor.dtype != torch_module.bfloat16:
                raise RuntimeError(
                    f"Gathered model state {key!r} has dtype={tensor.dtype}; expected BF16"
                )
            if not torch_module.isfinite(tensor).all():
                raise FloatingPointError(f"Gathered model state {key!r} is non-finite")
    if floating_count < 1:
        raise RuntimeError("Gathered model state contains no floating-point tensors")
    return state_dict


def _require_models_bitwise_equal(first, second, torch_module) -> int:
    first_state = first.state_dict()
    second_state = second.state_dict()
    if list(first_state) != list(second_state):
        raise RuntimeError("Safetensors round trip changed model state key order")
    compared = 0
    for key in first_state:
        first_tensor = first_state[key].detach().cpu()
        second_tensor = second_state[key].detach().cpu()
        if (
            first_tensor.dtype != second_tensor.dtype
            or first_tensor.shape != second_tensor.shape
            or not torch_module.equal(first_tensor, second_tensor)
        ):
            raise RuntimeError(f"Safetensors round trip changed tensor {key!r}")
        compared += 1
    if compared < 1:
        raise RuntimeError("Safetensors round trip compared no tensors")
    return compared


def _require_model_matches_state_dict(model, state_dict: Mapping[str, Any], torch_module) -> int:
    model_state = model.state_dict()
    if list(model_state) != list(state_dict):
        raise RuntimeError("Strict model load changed gathered state key order")
    compared = 0
    for key in model_state:
        model_tensor = model_state[key].detach().cpu()
        source_tensor = state_dict[key].detach().cpu()
        if (
            model_tensor.dtype != source_tensor.dtype
            or model_tensor.shape != source_tensor.shape
            or not torch_module.equal(model_tensor, source_tensor)
        ):
            raise RuntimeError(f"Strict model load changed gathered tensor {key!r}")
        compared += 1
    if compared < 1:
        raise RuntimeError("Strict model load compared no tensors")
    return compared


def _collective_local_call(dist_module, context: str, operation: Callable[[], Any]) -> Any:
    if not dist_module.is_available() or not dist_module.is_initialized():
        raise RuntimeError(f"{context} requires an initialized process group")
    rank = dist_module.get_rank()
    try:
        value = operation()
        status: dict[str, Any] = {"ok": True, "rank": rank}
    except BaseException as error:
        value = None
        status = {
            "ok": False,
            "rank": rank,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    gathered: list[object] = [None for _ in range(dist_module.get_world_size())]
    dist_module.all_gather_object(gathered, status)
    failures = [
        item
        for item in gathered
        if type(item) is not dict or item.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"{context} failed collectively: {failures}")
    return value


def _validate_staged_fold_manifest(
    *,
    dataset_dir: Path,
    fold_manifest_path: Path,
) -> dict[str, Any]:
    """Bind the frozen fold manifest to identical SageMaker-mounted bytes."""
    return validate_staged_dataset_and_fold(
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
        expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        expected_fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
        expected_dataset_manifest_logical_path=(
            EXPECTED_DATASET_MANIFEST_LOGICAL_PATH
        ),
    )


def _validate_experiment_config(
    config: dict[str, Any],
    *,
    outer_fold: int,
    query_view: str,
    sampler: str,
    experiment_seed: int,
) -> None:
    if (
        config.get("experiment_id") != "arr_retrieval_cv_v1"
        or type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Unexpected controlled experiment identity or schema")
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError("outer_fold must be an exact integer 0 through 4")
    if type(query_view) is not str or type(sampler) is not str:
        raise TypeError("query_view and sampler must be exact strings")
    if type(experiment_seed) is not int:
        raise TypeError("experiment_seed must be an exact integer")

    matrix = config.get("run_matrix")
    expected_matrix = {
        "controlled_full_runs": 60,
        "distributed_determinism_smokes": 2,
        "legacy_configuration_replication_attempts": 2,
        "query_views": ["flat_masked", "structured"],
        "samplers": list(CONTROLLED_SAMPLERS),
        "seeds": list(CONTROLLED_SEEDS),
        "total_training_submissions": 64,
    }
    if not _exact_json_equal(matrix, expected_matrix):
        raise ValueError("Experiment run matrix changed")
    if experiment_seed not in matrix["seeds"] or query_view not in matrix["query_views"] or sampler not in matrix["samplers"]:
        raise ValueError("Requested run is outside the frozen matrix")
    expected_model_selection = {
        "candidate_regime": "fold_global",
        "evaluation_role": "validation",
        "lexicographic_order": [
            "maximize validation case-macro set recall@20",
            "maximize validation case-macro full-ranking first-gold reciprocal rank",
            "minimize epoch number",
        ],
        "primary": EXPECTED_MODEL_SELECTION_PRIMARY,
        "secondary": EXPECTED_MODEL_SELECTION_SECONDARY,
        "tertiary": "earlier_epoch",
        "train_all_epochs": True,
    }
    expected_training = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-08,
        "batch_size_queries_per_gpu": 4,
        "effective_global_query_batch": 128,
        "epochs": 20,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-05,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "max_passage_tokens": 500,
        "max_query_tokens": 4096,
        "model_selection": expected_model_selection,
        "outer_training_case_range": [24, 26],
        "outer_training_passage_range": [3169, 3176],
        "outer_training_queries": 294,
        "optimizer": "adamw_torch",
        "temperature": 0.07,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    }
    training = config.get("training")
    if type(training) is not dict:
        raise TypeError("Experiment training section must be an object")
    if not _exact_json_equal(training, expected_training):
        raise ValueError("Frozen controlled training and model-selection settings changed")

    expected_evaluation = {
        "candidate_regimes": {
            "fold_global": {
                "definition": (
                    "all and only corpus passages whose doc_id belongs to the single fold "
                    "currently serving as the evaluated role"
                ),
                "never_all_42_cases": True,
                "test_use": "test-fold passages only; exclude train and validation folds",
                "validation_use": "validation-fold passages only; exclude train and test folds",
            }
        },
        "other_metrics": [
            "case_macro_hit_at_1",
            "case_macro_hit_at_5",
            "case_macro_hit_at_10",
            "case_macro_mrr_full_ranking",
            "case_macro_set_recall_at_1",
            "case_macro_set_recall_at_5",
            "case_macro_set_recall_at_10",
            "case_macro_set_recall_at_20",
            "case_macro_exact_target_recovery_at_1",
            "case_macro_exact_target_recovery_at_5",
            "case_macro_exact_target_recovery_at_10",
            "case_macro_exact_target_recovery_at_20",
            "query_micro_versions",
            "candidate_count",
        ],
        "primary_candidate_regime": "fold_global",
        "primary_evaluation_role": "test",
        "primary_endpoint": "case_macro_hit_at_20",
        "robustness_regime": {
            "candidate_regime": "fold_global_context_excluded",
            "definition": (
                "filter visible_passage_ids from the complete fold_global ranking without "
                "rescoring"
            ),
            "exclude": "visible_passage_ids",
            "gold_precedence": "never exclude a positive passage",
        },
    }
    if not _exact_json_equal(config.get("evaluation"), expected_evaluation):
        raise ValueError("Frozen controlled evaluation settings changed")

    aws_training = config.get("aws_training")
    expected_aws = {
        "accelerate_version": "1.4.0",
        "bf16": True,
        "deepspeed_stage": 3,
        "deepspeed_version": "0.17.1",
        "flash_attn_version": "2.7.3",
        "hjson_version": "3.1.0",
        "instance_count": 1,
        "instance_type": "ml.g5.12xlarge",
        "max_concurrent_full_jobs": 4,
        "mpi_workers": 4,
        "numpy_version": "1.26.4",
        "nvidia_ml_py_version": "13.590.48",
        "python_version": "3.11.10",
        "py_cpuinfo_version": "9.0.0",
        "region": "us-east-1",
        "sagemaker_base_model_channel": "base_model",
        "sagemaker_sdk_version": "2.248.2",
        "training_image": EXPECTED_TRAINING_IMAGE,
    }
    if type(aws_training) is not dict:
        raise TypeError("Experiment aws_training section must be an object")
    changed_aws = {
        key: {"expected": value, "actual": aws_training.get(key)}
        for key, value in expected_aws.items()
        if not _exact_json_equal(aws_training.get(key), value)
    }
    if changed_aws:
        raise ValueError(f"Frozen AWS training settings changed: {changed_aws}")

    expected_runtime_control = {
        "add_special_tokens_return": 19,
        "attention_implementation": "flash_attention_2",
        "base_tokenizer_size": 50_368,
        "batch_order_algorithm": "sha256_query_order_v1",
        "candidate_trace_artifact_schema_version": 1,
        "candidate_trace_merge_order": ["epoch", "query_id"],
        "candidate_trace_publication": "rank_shards_rank0_atomic_manifest_commit_v1",
        "cublas_workspace_config": ":4096:8",
        "dataloader_num_workers": 0,
        "deepspeed_gradient_clipping": 1.0,
        "deterministic_algorithms_warn_only": False,
        "deterministic_flash_attention": True,
        "flash_attention_deterministic_environment": "1",
        "full_determinism_argument": False,
        "global_candidate_deduplication": "sorted_unique_integer_index_v1",
        "markup_tokens_supplied": 19,
        "net_new_vocabulary_rows": 18,
        "optimizer_window_microbatches": [8, 8, 3],
        "optimizer_window_valid_queries": [128, 128, 38],
        "passage_embedding_gather": "torch_distributed_nn_autograd_all_gather_padded_v1",
        "passage_index_order": "lexicographic_passage_id_v1",
        "passage_index_schema_version": 1,
        "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
        "passage_owner_assignment": "sorted_position_mod_world_size_v1",
        "passage_padding_index": -1,
        "prepared_batches_per_rank": 19,
        "reference_compile": False,
        "resized_tokenizer_size": 50_386,
        "retriever_forward": "single_top_level_engine_call_query_and_owner_passages_v1",
        "sentinel_index": -1,
        "tf32": False,
        "total_optimizer_updates": 60,
        "updates_per_epoch": 3,
        "validation_forward_steps": 7,
        "validation_query_sharding": "sorted_global_position_mod_world_size_v1",
        "validation_passage_sharding": "sorted_global_position_mod_world_size_v1",
        "validation_query_batch_max_per_rank": 4,
        "validation_passage_batch_max_per_rank": 38,
        "validation_embedding_gather": "padded_position_all_gather_v1",
        "validation_scoring": "cpu_float32_v1",
        "validation_ranking": "score_desc_passage_id_asc_v1",
        "validation_result_broadcast": "rank0_canonical_payload_v1",
        "checkpoint_save": "all_rank_zero3_explicit_tag_atomic_directory_v1",
        "checkpoint_retention": "best_and_last_v1",
        "checkpoint_reload": "engine_a_free_fresh_engine_b_full_state_v1",
        "final_model_artifact": (
            "fresh_best_engine_zero3_gathered_bf16_safetensors_v1"
        ),
    }
    if not _exact_json_equal(config.get("runtime_control"), expected_runtime_control):
        raise ValueError("Frozen controlled runtime settings changed")

    expected_dataset = {
        "manifest_path": EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
        "manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "queries": 490,
        "cases": 42,
        "passages": 5286,
    }
    if not _exact_json_equal(config.get("dataset"), expected_dataset):
        raise ValueError("Frozen controlled dataset settings changed")

    expected_folds = {
        "capacities": [9, 9, 8, 8, 8],
        "generator_source_sha256": (
            "d6461e8ef638e62631c2bef72ae8afe944ab1cf98583aa705cae2e965cf57ef0"
        ),
        "manifest_path": EXPECTED_FOLD_MANIFEST_LOGICAL_PATH,
        "manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        "refinement": "deterministic strict best-improvement pair swaps after greedy initialization",
        "rotation": {
            "test_fold": "i",
            "validation_fold": "(i+1) mod 5",
            "training_folds": "remaining three",
        },
    }
    if not _exact_json_equal(config.get("folds"), expected_folds):
        raise ValueError("Frozen controlled fold settings changed")

    models = config.get("models")
    expected_model = {
        "model_id": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "snapshot_manifest_path": (
            "corporate_reorganization/modernbert/experiments/retrieval_cv/"
            "configs/modernbert_snapshot.json"
        ),
        "snapshot_tree_sha256": EXPECTED_SNAPSHOT_TREE_SHA256,
    }
    if (
        type(models) is not dict
        or not _exact_json_equal(models.get("modernbert_base"), expected_model)
    ):
        raise ValueError("Frozen ModernBERT model/snapshot settings changed")


def _configure_distributed_environment(torch_module) -> tuple[int, int, int]:
    for source, destination in (
        ("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"),
        ("OMPI_COMM_WORLD_RANK", "RANK"),
        ("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"),
    ):
        source_value = os.environ.get(source)
        destination_value = os.environ.get(destination)
        if source_value is None and destination_value is None:
            raise RuntimeError(
                f"Missing distributed environment variables {source} and {destination}"
            )
        if (
            source_value is not None
            and destination_value is not None
            and source_value != destination_value
        ):
            raise RuntimeError(
                f"Conflicting distributed environment: {source}={source_value}, "
                f"{destination}={destination_value}"
            )
        if destination_value is None:
            os.environ[destination] = source_value

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if (
        world_size != 4
        or rank not in range(4)
        or local_rank not in range(4)
        or rank != local_rank
    ):
        raise RuntimeError(
            f"Controlled topology must be one four-GPU host; "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 4:
        raise RuntimeError(
            f"Controlled runtime requires exactly four visible CUDA devices; "
            f"available={torch_module.cuda.is_available()}, count={torch_module.cuda.device_count()}"
        )
    torch_module.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def _configure_determinism(*, experiment_seed: int, torch_module, numpy_module, transformers_module) -> None:
    import random

    random.seed(experiment_seed)
    numpy_module.random.seed(experiment_seed)
    torch_module.manual_seed(experiment_seed)
    torch_module.cuda.manual_seed_all(experiment_seed)
    transformers_module.set_seed(experiment_seed)
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    _validate_determinism_state(torch_module)


def _validate_determinism_state(torch_module) -> None:
    changed = []
    if not torch_module.are_deterministic_algorithms_enabled():
        changed.append("deterministic_algorithms_disabled")
    if torch_module.is_deterministic_algorithms_warn_only_enabled():
        changed.append("deterministic_algorithms_warn_only")
    if torch_module.backends.cudnn.benchmark:
        changed.append("cudnn_benchmark_enabled")
    if not torch_module.backends.cudnn.deterministic:
        changed.append("cudnn_deterministic_disabled")
    if torch_module.backends.cuda.matmul.allow_tf32:
        changed.append("cuda_matmul_tf32_enabled")
    if torch_module.backends.cudnn.allow_tf32:
        changed.append("cudnn_tf32_enabled")
    if changed:
        raise RuntimeError(f"Controlled deterministic PyTorch state changed: {changed}")


def _validate_deepspeed_config(path: Path) -> None:
    config = _load_json_object(path, name="DeepSpeed config")
    expected = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "stage3_param_persistence_threshold": 1_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
            "stage3_max_live_parameters": 3e7,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,
        "train_batch_size": 128,
    }
    if not _exact_json_equal(config, expected):
        raise ValueError(
            "DeepSpeed configuration changed from the frozen exact ZeRO-3/BF16/batch contract"
        )


def _prepare_output_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(entry.name for entry in path.iterdir())
    if existing:
        raise FileExistsError(f"Controlled output directory must start empty: {path}; entries={existing}")


def _add_and_validate_markup_tokens(tokenizer, *, markup_tokens: list[str], slot_token: str) -> int:
    if len(tokenizer) != 50_368:
        raise RuntimeError(f"Frozen base tokenizer size changed: {len(tokenizer)}")
    if len(markup_tokens) != 19 or len(markup_tokens) != len(set(markup_tokens)):
        raise RuntimeError("Frozen markup token list must contain exactly 19 unique tokens")
    added = tokenizer.add_special_tokens({"additional_special_tokens": markup_tokens})
    if added != 19 or len(tokenizer) != 50_386 or len(tokenizer) - 50_368 != 18:
        raise RuntimeError(
            "Frozen tokenizer extension changed: "
            f"tokens_supplied={len(markup_tokens)}, add_special_tokens_return={added}, "
            f"net_new_rows={len(tokenizer) - 50_368}, final_size={len(tokenizer)}"
        )
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in markup_tokens]
    if len(token_ids) != len(set(token_ids)) or any(token_id == tokenizer.unk_token_id for token_id in token_ids):
        raise RuntimeError("Frozen markup tokens do not map to 19 unique non-unknown IDs")
    slot_token_id = int(tokenizer.convert_tokens_to_ids(slot_token))
    if slot_token_id == tokenizer.unk_token_id:
        raise ValueError(f"{slot_token} was not added to the tokenizer")
    return slot_token_id


def _enable_deterministic_modernbert_flash_attention(config):
    if getattr(config, "model_type", None) != "modernbert":
        raise TypeError(f"Expected ModernBERT config, got {getattr(config, 'model_type', None)!r}")
    if type(getattr(config, "deterministic_flash_attn", None)) is not bool:
        raise TypeError("ModernBERT config must expose exact boolean deterministic_flash_attn")
    if config.deterministic_flash_attn:
        raise ValueError("Frozen base snapshot unexpectedly already enables deterministic FlashAttention")
    if getattr(config, "reference_compile", None) is not None:
        raise ValueError("Frozen base snapshot unexpectedly defines reference_compile")
    config.deterministic_flash_attn = True
    config.reference_compile = False
    return config


def _validate_loaded_modernbert_attention(encoder) -> None:
    config = encoder.config
    if getattr(config, "_attn_implementation", None) != "flash_attention_2":
        raise RuntimeError(
            "ModernBERT did not resolve the frozen flash_attention_2 backend: "
            f"{getattr(config, '_attn_implementation', None)!r}"
        )
    if getattr(config, "deterministic_flash_attn", None) is not True:
        raise RuntimeError("Loaded ModernBERT config did not retain deterministic FlashAttention")
    if getattr(config, "reference_compile", None) is not False:
        raise RuntimeError("Loaded ModernBERT must keep reference_compile disabled")
    module_flags = [
        module.deterministic_flash_attn
        for module in encoder.modules()
        if hasattr(module, "deterministic_flash_attn")
    ]
    if not module_flags or any(flag is not True for flag in module_flags):
        raise RuntimeError(
            "Every loaded ModernBERT attention module must enable deterministic FlashAttention"
        )


def _build_controlled_retriever(
    *,
    base_model_dir: Path,
    tokenizer_size: int,
    slot_token_id: int,
    temperature: float,
    auto_config_class,
    auto_model_class,
    retriever_class,
    torch_dtype=None,
):
    if type(tokenizer_size) is not int or tokenizer_size != 50_386:
        raise RuntimeError("Controlled model factory requires tokenizer size 50,386")
    if type(slot_token_id) is not int or slot_token_id < 0:
        raise ValueError("Controlled model factory requires a non-negative exact slot token ID")
    if type(temperature) is not float or temperature != 0.07:
        raise RuntimeError("Controlled model factory requires exact temperature 0.07")
    encoder_config = _enable_deterministic_modernbert_flash_attention(
        auto_config_class.from_pretrained(
            str(base_model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    model_kwargs = {
        "config": encoder_config,
        "attn_implementation": "flash_attention_2",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    encoder = auto_model_class.from_pretrained(
        str(base_model_dir),
        **model_kwargs,
    )
    _validate_loaded_modernbert_attention(encoder)
    encoder.resize_token_embeddings(tokenizer_size)
    model = retriever_class(
        encoder=encoder,
        slot_token_id=slot_token_id,
        temperature=temperature,
    )
    if len(model.encoder.get_input_embeddings().weight) != tokenizer_size:
        raise RuntimeError("Controlled model factory produced the wrong embedding row count")
    partitioned_parameters = [
        name
        for name, parameter in model.named_parameters()
        if hasattr(parameter, "ds_id")
    ]
    if partitioned_parameters:
        raise RuntimeError(
            "Controlled model factory must produce an unpartitioned model; found ZeRO IDs on "
            f"{partitioned_parameters[:5]}"
        )
    return model


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_frozen_control_file_hashes(
        experiment_config_path=args.experiment_config,
        deepspeed_config_path=args.deepspeed_config,
    )
    validate_preimport_environment(args.experiment_seed)
    validate_runtime_versions()

    import numpy
    import torch
    import transformers
    from safetensors.torch import load_model, save_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config
    from transformers.utils import is_flash_attn_2_available

    from retriever.collator import ControlledRetrievalBatchCollator
    from retriever.checkpointing import (
        publish_new_text,
        rank_zero_call,
        retained_checkpoint_inventory,
    )
    from retriever.data import PassageIndexTable, load_candidates_by_case, load_corpus, load_queries
    from retriever.evaluation import (
        VALIDATION_FORWARD_STEPS,
        VALIDATION_MAX_LEN_PASSAGE,
        VALIDATION_MAX_LEN_QUERY,
        VALIDATION_PASSAGE_BATCH_CAP,
        VALIDATION_PRIMARY_METRIC,
        VALIDATION_QUERY_BATCH_CAP,
        VALIDATION_SECONDARY_METRIC,
        build_fold_global_validation_data,
    )
    from retriever.markup import SLOT_TOKEN, all_markup_tokens
    from retriever.models import DualEncoderRetriever
    from retriever.sampling import ControlledRetrievalTrainDataset
    from trainer import ControlledRetrievalTrainer

    local_rank, rank, world_size = _configure_distributed_environment(torch)
    _configure_determinism(
        experiment_seed=args.experiment_seed,
        torch_module=torch,
        numpy_module=numpy,
        transformers_module=transformers,
    )
    if not is_flash_attn_2_available():
        raise RuntimeError(
            "Frozen controlled runtime requires FlashAttention 2.7.3 on the visible CUDA device"
        )

    experiment = _load_json_object(args.experiment_config, name="Experiment config")
    _validate_experiment_config(
        experiment,
        outer_fold=args.outer_fold,
        query_view=args.query_view,
        sampler=args.sampler,
        experiment_seed=args.experiment_seed,
    )
    runtime_control = experiment["runtime_control"]
    code_runtime_contract = {
        "validation_forward_steps": VALIDATION_FORWARD_STEPS,
        "validation_query_batch_max_per_rank": VALIDATION_QUERY_BATCH_CAP,
        "validation_passage_batch_max_per_rank": VALIDATION_PASSAGE_BATCH_CAP,
    }
    if any(
        runtime_control[name] != expected
        for name, expected in code_runtime_contract.items()
    ):
        raise RuntimeError(
            "Evaluator constants and the frozen runtime-control contract disagree"
        )
    if (
        VALIDATION_MAX_LEN_QUERY != experiment["training"]["max_query_tokens"]
        or VALIDATION_MAX_LEN_PASSAGE != experiment["training"]["max_passage_tokens"]
    ):
        raise RuntimeError("Evaluator token limits and frozen training settings disagree")
    model_selection = experiment["training"]["model_selection"]
    if (
        f"eval_{model_selection['primary']}" != VALIDATION_PRIMARY_METRIC
        or f"eval_{model_selection['secondary']}" != VALIDATION_SECONDARY_METRIC
    ):
        raise RuntimeError("Evaluator metric keys and frozen model selection disagree")
    _validate_deepspeed_config(args.deepspeed_config)
    fold_manifest = _validate_staged_fold_manifest(
        dataset_dir=args.data_dir,
        fold_manifest_path=args.folds_config,
    )
    snapshot_manifest = load_snapshot_manifest(args.snapshot_manifest)
    validate_snapshot_directory(args.base_model_dir, snapshot_manifest)
    _prepare_output_directory(args.output_dir)

    rotations = {
        rotation["outer_fold"]: rotation
        for rotation in fold_manifest["rotations"]
    }
    rotation = rotations[args.outer_fold]
    training_doc_ids = list(rotation["train"]["case_ids"])
    if rotation["train"]["queries"] != 294:
        raise RuntimeError("Frozen rotation no longer has exactly 294 training queries")
    training_doc_id_set = set(training_doc_ids)
    validation_role = rotation["validation"]
    validation_doc_ids = list(validation_role["case_ids"])
    if (
        validation_role["queries"] != 98
        or validation_role["passages"] not in {1_054, 1_055, 1_060, 1_062}
        or validation_role["num_cases"] != len(validation_doc_ids)
        or len(validation_doc_ids) not in {8, 9}
    ):
        raise RuntimeError("Frozen validation rotation inventory changed")

    corpus_by_passage_id = load_corpus(args.data_dir)
    passage_index_table = PassageIndexTable(corpus_by_passage_id)
    if len(passage_index_table) != 5_286:
        raise RuntimeError(
            f"Controlled passage index contains {len(passage_index_table)} rows; expected 5,286"
        )
    if passage_index_table.sha256 != EXPECTED_PASSAGE_INDEX_SHA256:
        raise RuntimeError(
            "Controlled corpus-wide passage index digest changed: "
            f"actual={passage_index_table.sha256}, expected={EXPECTED_PASSAGE_INDEX_SHA256}"
        )
    candidates_by_case = load_candidates_by_case(args.data_dir)
    all_queries = load_queries(args.data_dir, "all")
    validation_data = build_fold_global_validation_data(
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        passage_index_table=passage_index_table,
        validation_case_ids=validation_doc_ids,
        expected_query_count=validation_role["queries"],
        expected_passage_count=validation_role["passages"],
        query_view=args.query_view,
    )
    training_queries = [
        query for query in all_queries if query.doc_id in training_doc_id_set
    ]
    if len(training_queries) != 294:
        raise RuntimeError(f"Loaded {len(training_queries)} training queries; expected exactly 294")
    train_dataset = ControlledRetrievalTrainDataset(
        training_queries,
        corpus_by_passage_id,
        candidates_by_case,
        training_doc_ids,
        passage_index_table=passage_index_table,
        sampler=args.sampler,
        experiment_seed=args.experiment_seed,
        query_view=args.query_view,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_dir),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    slot_token_id = _add_and_validate_markup_tokens(
        tokenizer,
        markup_tokens=all_markup_tokens(),
        slot_token=SLOT_TOKEN,
    )

    model = _build_controlled_retriever(
        base_model_dir=args.base_model_dir,
        tokenizer_size=len(tokenizer),
        slot_token_id=slot_token_id,
        temperature=experiment["training"]["temperature"],
        auto_config_class=AutoConfig,
        auto_model_class=AutoModel,
        retriever_class=DualEncoderRetriever,
    )

    collator = ControlledRetrievalBatchCollator(
        tokenizer,
        corpus_size=len(passage_index_table),
        max_len_query=experiment["training"]["max_query_tokens"],
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=experiment["training"]["epochs"],
        per_device_train_batch_size=experiment["training"]["batch_size_queries_per_gpu"],
        learning_rate=experiment["training"]["learning_rate"],
        warmup_ratio=experiment["training"]["warmup_ratio"],
        weight_decay=experiment["training"]["weight_decay"],
        adam_beta1=experiment["training"]["adam_beta1"],
        adam_beta2=experiment["training"]["adam_beta2"],
        adam_epsilon=experiment["training"]["adam_epsilon"],
        optim=experiment["training"]["optimizer"],
        lr_scheduler_type=experiment["training"]["lr_scheduler_type"],
        max_grad_norm=experiment["training"]["max_grad_norm"],
        gradient_accumulation_steps=experiment["training"]["gradient_accumulation_steps"],
        eval_strategy="epoch",
        eval_on_start=False,
        eval_delay=0,
        save_strategy="epoch",
        logging_steps=1,
        bf16=True,
        tf32=False,
        deepspeed=str(args.deepspeed_config),
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        report_to=[],
        save_only_model=False,
        save_total_limit=None,
        load_best_model_at_end=False,
        metric_for_best_model=model_selection["primary"],
        greater_is_better=True,
        local_rank=local_rank,
        seed=args.experiment_seed,
        data_seed=args.experiment_seed,
        # Transformers 4.49's full_determinism helper overwrites the frozen
        # :4096:8 CUBLAS workspace setting with :16:8. Determinism is installed
        # explicitly before model loading and revalidated after Trainer init.
        full_determinism=False,
        accelerator_config={
            "split_batches": False,
            "dispatch_batches": False,
            "even_batches": False,
            "use_seedable_sampler": False,
        },
    )
    trainer = ControlledRetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_data.queries,
        data_collator=collator,
        processing_class=tokenizer,
        retrieval_eval_config=None,
        experiment_seed=args.experiment_seed,
        passage_index_table=passage_index_table,
        validation_data=validation_data,
        max_len_passage=experiment["training"]["max_passage_tokens"],
    )
    # The Trainer now owns the only Engine-A model reference outside Accelerate.
    model = None
    validate_preimport_environment(args.experiment_seed)
    _validate_determinism_state(torch)

    if rank == 0:
        print(
            "CONTROLLED_RUN "
            + json.dumps(
                {
                    "outer_fold": args.outer_fold,
                    "query_view": args.query_view,
                    "sampler": args.sampler,
                    "experiment_seed": args.experiment_seed,
                    "world_size": world_size,
                    "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
                    "snapshot_tree_sha256": snapshot_manifest["tree_sha256"],
                    "passage_index_sha256": passage_index_table.sha256,
                },
                sort_keys=True,
            )
        )
    trainer.train()
    if trainer.state.global_step != 60 or float(trainer.state.epoch) != 20.0:
        raise RuntimeError(
            "Controlled training did not complete the exact 20-epoch/60-update schedule"
        )
    trace_manifest = trainer.finalize_sampling_traces()
    checkpoint_history = trainer.finalize_checkpoint_selection()

    accelerator = trainer.accelerator
    if trainer.deepspeed is None or trainer.model_wrapped is not trainer.deepspeed:
        raise RuntimeError("Controlled training completed without active Engine A")
    trainer.release_current_deepspeed_engine()

    # Reproduce Engine A's original unpartitioned construction path. The same
    # deterministic seed is installed on every rank before any new parameters
    # are materialized.
    unset_hf_deepspeed_config()
    _configure_determinism(
        experiment_seed=args.experiment_seed,
        torch_module=torch,
        numpy_module=numpy,
        transformers_module=transformers,
    )
    fresh_model = _collective_local_call(
        torch.distributed,
        "Fresh controlled retriever construction",
        lambda: _build_controlled_retriever(
            base_model_dir=args.base_model_dir,
            tokenizer_size=len(tokenizer),
            slot_token_id=slot_token_id,
            temperature=experiment["training"]["temperature"],
            auto_config_class=AutoConfig,
            auto_model_class=AutoModel,
            retriever_class=DualEncoderRetriever,
        ),
    )
    trainer.prepare_fresh_deepspeed_engine(fresh_model)
    fresh_model = None
    best_reload = trainer.load_and_verify_best_checkpoint()
    local_reload_record = {
        "rank": rank,
        **{
            key: best_reload[key]
            for key in (
                "load_path_parent",
                "client_state_sha256",
                "scheduler_state_sha256",
                "global_step",
                "rng_sha256",
                "manifest_sha256",
            )
        },
    }
    reload_records: list[object] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(reload_records, local_reload_record)
    expected_reload_keys = {
        "rank",
        "load_path_parent",
        "client_state_sha256",
        "scheduler_state_sha256",
        "global_step",
        "rng_sha256",
        "manifest_sha256",
    }
    expected_load_parent = str(
        (
            args.output_dir
            / best_reload["selection"]["checkpoint_dir"]
            / best_reload["selection"]["deepspeed_tag"]
        ).resolve()
    )
    for expected_rank, record in enumerate(reload_records):
        if type(record) is not dict or set(record) != expected_reload_keys:
            raise RuntimeError(f"Fresh-engine reload record is malformed: {record!r}")
        if (
            record["rank"] != expected_rank
            or record["global_step"] != best_reload["selection"]["global_step"]
            or record["load_path_parent"] != expected_load_parent
            or any(
                not _is_sha256(record[name])
                for name in (
                    "client_state_sha256",
                    "scheduler_state_sha256",
                    "rng_sha256",
                    "manifest_sha256",
                )
            )
        ):
            raise RuntimeError(f"Fresh-engine reload record changed for rank {expected_rank}")

    # DeepSpeed 0.17.1 requires all ranks to enter the ZeRO-3 gather and returns
    # the complete CPU state only on global rank zero.
    gathered_state = accelerator.get_state_dict(trainer.deepspeed)

    def validate_local_gather_result() -> dict[str, Any]:
        if rank == 0:
            exact_state = _validate_gathered_bf16_state_dict(gathered_state, torch)
            return {"rank": rank, "parameter_and_buffer_count": len(exact_state)}
        if gathered_state is not None:
            raise RuntimeError("Nonzero rank unexpectedly received the consolidated ZeRO-3 state")
        return {"rank": rank, "parameter_and_buffer_count": 0}

    _collective_local_call(
        torch.distributed,
        "Gathered BF16 model-state validation",
        validate_local_gather_result,
    )
    trainer.release_current_deepspeed_engine()
    unset_hf_deepspeed_config()

    def publish_final_artifacts() -> dict[str, Any]:
        exact_state = _validate_gathered_bf16_state_dict(gathered_state, torch)
        cpu_tokenizer = AutoTokenizer.from_pretrained(
            str(args.base_model_dir),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        cpu_slot_token_id = _add_and_validate_markup_tokens(
            cpu_tokenizer,
            markup_tokens=all_markup_tokens(),
            slot_token=SLOT_TOKEN,
        )
        if cpu_slot_token_id != slot_token_id or len(cpu_tokenizer) != len(tokenizer):
            raise RuntimeError("Fresh CPU tokenizer changed the controlled token contract")
        cpu_model = _build_controlled_retriever(
            base_model_dir=args.base_model_dir,
            tokenizer_size=len(cpu_tokenizer),
            slot_token_id=cpu_slot_token_id,
            temperature=experiment["training"]["temperature"],
            auto_config_class=AutoConfig,
            auto_model_class=AutoModel,
            retriever_class=DualEncoderRetriever,
            torch_dtype=torch.bfloat16,
        )
        incompatible = cpu_model.load_state_dict(exact_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict gathered-state load was incomplete: {incompatible}")
        gathered_tensor_count = _require_model_matches_state_dict(
            cpu_model,
            exact_state,
            torch,
        )
        exact_state.clear()

        model_path = args.output_dir / "model.safetensors"
        model_record = _publish_new_binary(
            model_path,
            lambda temporary_path: save_model(
                cpu_model,
                str(temporary_path),
                metadata={
                    "format": "pt",
                    "weight_dtype": "bfloat16",
                    "source": "fresh_best_engine_zero3_gathered_16bit_state",
                },
            ),
        )
        reloaded_cpu_model = _build_controlled_retriever(
            base_model_dir=args.base_model_dir,
            tokenizer_size=len(cpu_tokenizer),
            slot_token_id=cpu_slot_token_id,
            temperature=experiment["training"]["temperature"],
            auto_config_class=AutoConfig,
            auto_model_class=AutoModel,
            retriever_class=DualEncoderRetriever,
            torch_dtype=torch.bfloat16,
        )
        missing, unexpected = load_model(
            reloaded_cpu_model,
            model_path,
            strict=True,
            device="cpu",
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Strict safetensors reload was incomplete: missing={missing}, "
                f"unexpected={unexpected}"
            )
        round_trip_tensor_count = _require_models_bitwise_equal(
            cpu_model,
            reloaded_cpu_model,
            torch,
        )
        if round_trip_tensor_count != gathered_tensor_count:
            raise RuntimeError("Safetensors round trip changed the model state inventory")

        tokenizer_record = _publish_pretrained_directory(
            args.output_dir / "tokenizer",
            lambda path: cpu_tokenizer.save_pretrained(str(path)),
        )
        if [record["path"] for record in tokenizer_record["files"]] != [
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]:
            raise RuntimeError("Final tokenizer artifact inventory changed")
        encoder_config_record = _publish_pretrained_directory(
            args.output_dir / "encoder_config",
            lambda path: cpu_model.encoder.config.save_pretrained(str(path)),
        )
        if [record["path"] for record in encoder_config_record["files"]] != [
            "config.json"
        ]:
            raise RuntimeError("Final encoder-config artifact inventory changed")
        wrapper_payload = {
            "schema_version": 1,
            "architecture": "DualEncoderRetriever",
            "slot_token": SLOT_TOKEN,
            "slot_token_id": cpu_slot_token_id,
            "temperature": experiment["training"]["temperature"],
            "tokenizer_size": len(cpu_tokenizer),
            "weight_dtype": "bfloat16",
            "model_artifact_protocol": runtime_control["final_model_artifact"],
        }
        wrapper_path = args.output_dir / "wrapper_config.json"
        publish_new_text(
            wrapper_path,
            json.dumps(
                wrapper_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        wrapper_record = {
            "path": wrapper_path.name,
            "size": wrapper_path.stat().st_size,
            "sha256": _sha256(wrapper_path),
        }

        retained_inventory = retained_checkpoint_inventory(
            args.output_dir,
            checkpoint_history["retained_checkpoint_dirs"],
        )
        trace_manifest_path = args.output_dir / "candidate_traces/manifest.json"
        validation_manifest_path = args.output_dir / "validation/manifest.json"
        run_record = {
            "schema_version": 1,
            "experiment_id": experiment["experiment_id"],
            "outer_fold": args.outer_fold,
            "query_view": args.query_view,
            "sampler": args.sampler,
            "experiment_seed": args.experiment_seed,
            "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
            "training_image": EXPECTED_TRAINING_IMAGE,
            "experiment_config": {
                "path": "experiment.json",
                "sha256": _sha256(args.experiment_config),
            },
            "deepspeed_config": {
                "path": "ds_zero3.json",
                "sha256": _sha256(args.deepspeed_config),
            },
            "dataset": {
                "manifest_path": "dataset_manifest.json",
                "manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
                "output_sha256": fold_manifest["dataset"]["output_sha256"],
            },
            "folds": {
                "manifest_path": EXPECTED_FOLD_MANIFEST_LOGICAL_PATH,
                "manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
                "rotation": rotation,
            },
            "snapshot": {
                "manifest_path": "modernbert_snapshot.json",
                "manifest_sha256": _sha256(args.snapshot_manifest),
                "tree_sha256": snapshot_manifest["tree_sha256"],
            },
            "passage_index": {
                "schema_version": 1,
                "size": len(passage_index_table),
                "sha256": passage_index_table.sha256,
            },
            "validation_data": {
                "role": validation_data.role,
                "query_view": validation_data.query_view,
                "case_count": validation_data.case_count,
                "query_count": validation_data.query_count,
                "passage_count": validation_data.passage_count,
                "case_ids_sha256": validation_data.case_ids_sha256,
                "query_ids_sha256": validation_data.query_ids_sha256,
                "passage_ids_sha256": validation_data.passage_ids_sha256,
                "contract_sha256": validation_data.contract_sha256,
            },
            "candidate_traces": {
                "manifest_path": "candidate_traces/manifest.json",
                "manifest_sha256": _sha256(trace_manifest_path),
                "record_count": trace_manifest["record_count"],
                "merged_sha256": trace_manifest["merged"]["sha256"],
            },
            "validation_history": {
                "manifest_path": "validation/manifest.json",
                "manifest_sha256": _sha256(validation_manifest_path),
                "best": checkpoint_history["best"],
                "last": checkpoint_history["last"],
                "retained_checkpoint_dirs": checkpoint_history[
                    "retained_checkpoint_dirs"
                ],
            },
            "best_checkpoint_reload": {
                "selection": best_reload["selection"],
                "validation_result": best_reload["validation_result"],
                "per_rank": reload_records,
            },
            "final_model": {
                **model_record,
                "weight_dtype": "bfloat16",
                "gathered_tensor_count": gathered_tensor_count,
                "strict_round_trip_tensor_count": round_trip_tensor_count,
            },
            "tokenizer": tokenizer_record,
            "encoder_config": encoder_config_record,
            "wrapper_config": wrapper_record,
            "retained_checkpoints": retained_inventory,
        }
        run_path = args.output_dir / "controlled_run.json"
        publish_new_text(
            run_path,
            json.dumps(
                run_record,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        run_file_record = {
            "path": run_path.name,
            "size": run_path.stat().st_size,
            "sha256": _sha256(run_path),
        }

        expected_top_level = {
            "candidate_traces",
            "validation",
            *checkpoint_history["retained_checkpoint_dirs"],
            "model.safetensors",
            "tokenizer",
            "encoder_config",
            "wrapper_config.json",
            "controlled_run.json",
        }
        actual_top_level = {entry.name for entry in args.output_dir.iterdir()}
        if actual_top_level != expected_top_level or any(
            entry.is_symlink() for entry in args.output_dir.iterdir()
        ):
            raise RuntimeError(
                "Final artifact top-level inventory changed before commit marker: "
                f"actual={sorted(actual_top_level)}, expected={sorted(expected_top_level)}"
            )
        artifact_manifest = {
            "schema_version": 1,
            "commit_marker": True,
            "controlled_run": run_file_record,
            "model": model_record,
            "tokenizer": tokenizer_record,
            "encoder_config": encoder_config_record,
            "wrapper_config": wrapper_record,
            "candidate_trace_manifest": _artifact_file_record(
                trace_manifest_path,
                logical_path="candidate_traces/manifest.json",
            ),
            "validation_manifest": _artifact_file_record(
                validation_manifest_path,
                logical_path="validation/manifest.json",
            ),
            "retained_checkpoints": retained_inventory,
        }
        artifact_manifest_path = args.output_dir / "artifact_manifest.json"
        publish_new_text(
            artifact_manifest_path,
            json.dumps(
                artifact_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        try:
            final_top_level = {entry.name for entry in args.output_dir.iterdir()}
            if final_top_level != {*expected_top_level, "artifact_manifest.json"}:
                raise RuntimeError(
                    "Artifact commit marker publication changed output inventory"
                )
            artifact_manifest_sha256 = _sha256(artifact_manifest_path)
            if json.loads(artifact_manifest_path.read_text(encoding="utf-8")) != artifact_manifest:
                raise RuntimeError("Artifact commit marker readback changed its canonical payload")
            return {
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "controlled_run_sha256": run_file_record["sha256"],
                "model_sha256": model_record["sha256"],
            }
        except BaseException:
            if artifact_manifest_path.exists() or artifact_manifest_path.is_symlink():
                artifact_manifest_path.unlink()
                _fsync_directory(args.output_dir)
            raise

    final_artifacts = rank_zero_call(
        "Final controlled artifact publication",
        publish_final_artifacts,
    )
    if (
        type(final_artifacts) is not dict
        or set(final_artifacts)
        != {
            "artifact_manifest_sha256",
            "controlled_run_sha256",
            "model_sha256",
        }
    ):
        raise RuntimeError("Final artifact publication returned malformed metadata")
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
