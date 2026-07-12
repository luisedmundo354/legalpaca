from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from retriever.provenance import (
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


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_CONFIG = SOURCE_DIR / "experiments/retrieval_cv/configs/experiment.json"
DEFAULT_FOLDS_CONFIG = SOURCE_DIR / "experiments/retrieval_cv/configs/folds.json"
DEFAULT_SNAPSHOT_MANIFEST = SOURCE_DIR / "experiments/retrieval_cv/configs/modernbert_snapshot.json"
DEFAULT_DEEPSPEED_CONFIG = SOURCE_DIR / "ds_zero3.json"

CONTROLLED_QUERY_VIEWS = ("structured", "flat_masked")
CONTROLLED_SAMPLERS = (SAMPLER_LOCAL_UNIQUE, SAMPLER_GLOBAL_UNIFORM)
CONTROLLED_SEEDS = (17, 29, 43)
EXPECTED_DATASET_MANIFEST_LOGICAL_PATH = (
    "corporate_reorganization/data/final_annotations_gold/"
    "processed_retrieval_v2/dataset_manifest.json"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be"
)
EXPECTED_FOLD_MANIFEST_LOGICAL_PATH = (
    "corporate_reorganization/modernbert/experiments/retrieval_cv/configs/folds.json"
)
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "469858f2f8e42d0b19e53ee71af690f722482120348a2fe9719b99104758e00d"
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


def _validate_staged_fold_manifest(
    *,
    dataset_dir: Path,
    fold_manifest_path: Path,
) -> dict[str, Any]:
    """Bind the frozen fold manifest to identical SageMaker-mounted bytes."""

    if not dataset_dir.is_dir() or dataset_dir.is_symlink():
        raise ValueError(f"Staged dataset must be a real directory: {dataset_dir}")
    if not fold_manifest_path.is_file() or fold_manifest_path.is_symlink():
        raise ValueError(f"Fold manifest must be a regular file: {fold_manifest_path}")
    actual_fold_hash = _sha256(fold_manifest_path)
    if actual_fold_hash != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError(
            "Frozen fold-manifest SHA-256 changed: "
            f"actual={actual_fold_hash}, expected={EXPECTED_FOLD_MANIFEST_SHA256}"
        )
    stored = _load_json_object(fold_manifest_path, name="Fold manifest")
    expected_dataset_record = {
        "dataset_manifest_path": EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "dataset_schema_version": 2,
        "output_sha256": stored.get("dataset", {}).get("output_sha256"),
    }
    if not _exact_json_equal(stored.get("dataset"), expected_dataset_record):
        raise ValueError("Frozen fold manifest has an unexpected dataset identity")
    output_hashes = expected_dataset_record["output_sha256"]
    if type(output_hashes) is not dict or set(output_hashes) != {
        "cases.jsonl",
        "corpus.jsonl",
        "pools/candidates_by_case.json",
        "pools/candidates_global.json",
        "queries/all.jsonl",
    }:
        raise ValueError("Frozen fold manifest has an unexpected dataset output inventory")

    expected_files = {"dataset_manifest.json", *output_hashes}
    expected_directories = {"pools", "queries"}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in dataset_dir.rglob("*"):
        relative = path.relative_to(dataset_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"Staged dataset entry must not be a symlink: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"Unexpected staged dataset entry type: {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "Staged dataset inventory changed: "
            f"files={sorted(actual_files)}, directories={sorted(actual_directories)}"
        )

    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    actual_dataset_hash = _sha256(dataset_manifest_path)
    if actual_dataset_hash != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError(
            "Staged dataset-manifest SHA-256 changed: "
            f"actual={actual_dataset_hash}, expected={EXPECTED_DATASET_MANIFEST_SHA256}"
        )
    for relative_path, expected_hash in sorted(output_hashes.items()):
        actual_hash = _sha256(dataset_dir / relative_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Staged dataset output hash changed for {relative_path}: "
                f"actual={actual_hash}, expected={expected_hash}"
            )
    return stored


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
        "optimizer": "adamw_torch",
        "temperature": 0.07,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    }
    training = config.get("training")
    if type(training) is not dict:
        raise TypeError("Experiment training section must be an object")
    changed_training = {
        key: {"expected": value, "actual": training.get(key)}
        for key, value in expected_training.items()
        if not _exact_json_equal(training.get(key), value)
    }
    if changed_training:
        raise ValueError(f"Frozen controlled training settings changed: {changed_training}")

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
        "cublas_workspace_config": ":4096:8",
        "dataloader_num_workers": 0,
        "deepspeed_gradient_clipping": 1.0,
        "deterministic_algorithms_warn_only": False,
        "deterministic_flash_attention": True,
        "flash_attention_deterministic_environment": "1",
        "full_determinism_argument": False,
        "markup_tokens_supplied": 19,
        "net_new_vocabulary_rows": 18,
        "optimizer_window_microbatches": [8, 8, 3],
        "optimizer_window_valid_queries": [128, 128, 38],
        "prepared_batches_per_rank": 19,
        "reference_compile": False,
        "resized_tokenizer_size": 50_386,
        "sentinel_index": -1,
        "tf32": False,
        "total_optimizer_updates": 60,
        "updates_per_epoch": 3,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_preimport_environment(args.experiment_seed)
    validate_runtime_versions()

    import numpy
    import torch
    import transformers
    from safetensors.torch import save_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config
    from transformers.utils import is_flash_attn_2_available

    from retriever.collator import RetrievalBatchCollator
    from retriever.data import load_candidates_by_case, load_corpus, load_queries
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

    corpus_by_passage_id = load_corpus(args.data_dir)
    candidates_by_case = load_candidates_by_case(args.data_dir)
    all_queries = load_queries(args.data_dir, "all")
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

    encoder_config = _enable_deterministic_modernbert_flash_attention(
        AutoConfig.from_pretrained(
            str(args.base_model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    encoder = AutoModel.from_pretrained(
        str(args.base_model_dir),
        config=encoder_config,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        trust_remote_code=False,
    )
    _validate_loaded_modernbert_attention(encoder)
    encoder.resize_token_embeddings(len(tokenizer))
    model = DualEncoderRetriever(
        encoder=encoder,
        slot_token_id=slot_token_id,
        temperature=experiment["training"]["temperature"],
    )

    passage_text_by_passage_id = {
        passage_id: passage.text for passage_id, passage in corpus_by_passage_id.items()
    }
    collator = RetrievalBatchCollator(
        tokenizer,
        passage_text_by_passage_id,
        max_len_query=experiment["training"]["max_query_tokens"],
        max_len_passage=experiment["training"]["max_passage_tokens"],
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
        eval_strategy="no",
        save_strategy="no",
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
        eval_dataset=None,
        data_collator=collator,
        processing_class=tokenizer,
        retrieval_eval_config=None,
        experiment_seed=args.experiment_seed,
    )
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
                },
                sort_keys=True,
            )
        )
    trainer.train()

    accelerator = trainer.accelerator
    deepspeed_engine = getattr(trainer, "deepspeed", None)
    if deepspeed_engine is None:
        raise RuntimeError("Controlled training completed without a DeepSpeed engine")
    state_dict = accelerator.get_state_dict(deepspeed_engine)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unset_hf_deepspeed_config()
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
        cpu_encoder_config = _enable_deterministic_modernbert_flash_attention(
            AutoConfig.from_pretrained(
                str(args.base_model_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
        )
        cpu_encoder = AutoModel.from_pretrained(
            str(args.base_model_dir),
            config=cpu_encoder_config,
            attn_implementation="flash_attention_2",
            local_files_only=True,
            trust_remote_code=False,
        )
        _validate_loaded_modernbert_attention(cpu_encoder)
        cpu_encoder.resize_token_embeddings(len(cpu_tokenizer))
        cpu_model = DualEncoderRetriever(
            encoder=cpu_encoder,
            slot_token_id=cpu_slot_token_id,
            temperature=experiment["training"]["temperature"],
        )
        cpu_model.load_state_dict(state_dict, strict=True)
        save_model(cpu_model, str(args.output_dir / "model.safetensors"))
        cpu_tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))
        cpu_encoder.config.save_pretrained(str(args.output_dir / "encoder_config"))
        run_record = {
            "experiment_id": experiment["experiment_id"],
            "outer_fold": args.outer_fold,
            "query_view": args.query_view,
            "sampler": args.sampler,
            "experiment_seed": args.experiment_seed,
            "attention_implementation": cpu_encoder.config._attn_implementation,
            "deterministic_flash_attn": cpu_encoder.config.deterministic_flash_attn,
            "reference_compile": cpu_encoder.config.reference_compile,
            "snapshot_tree_sha256": snapshot_manifest["tree_sha256"],
            "resized_tokenizer_size": len(cpu_tokenizer),
            "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
        }
        (args.output_dir / "controlled_run.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
