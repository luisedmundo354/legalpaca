from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import save_model
from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from retriever.collator import RetrievalBatchCollator
from retriever.data import (
    MultiPositiveRetrievalTrainDataset,
    load_candidates_by_case,
    load_corpus,
    load_queries,
    select_distractor_passage_ids,
)
from retriever.markup import SLOT_TOKEN, all_markup_tokens
from retriever.models import DualEncoderRetriever
from trainer import MultiPositiveContrastiveTrainer, RetrievalEvalConfig, SetEpochCallback


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _setup_distributed_environment() -> Tuple[int, int, int]:
    env = os.environ

    for src_key, dst_key in (
        ("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"),
        ("OMPI_COMM_WORLD_RANK", "RANK"),
        ("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"),
        ("MPI_LOCALRANKID", "LOCAL_RANK"),
        ("MPI_RANK", "RANK"),
        ("PMI_RANK", "RANK"),
        ("PMI_SIZE", "WORLD_SIZE"),
    ):
        if dst_key not in env and src_key in env:
            env[dst_key] = env[src_key]

    local_rank = int(env.get("LOCAL_RANK", "0"))
    rank = int(env.get("RANK", "0"))
    world_size = int(env.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="answerdotai/ModernBERT-base")

    parser.add_argument(
        "--processed_dir",
        "--data-dir",
        dest="processed_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_DATA"),
    )
    parser.add_argument(
        "--model_dir",
        "--output-dir",
        dest="model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "outputs"),
    )

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    parser.add_argument("--epochs", "--num_train_epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--batch_size_queries", type=int, default=8)
    parser.add_argument("--max_len_query", type=int, default=4096)
    parser.add_argument("--max_len_passage", type=int, default=600)

    parser.add_argument("--max_pos_per_query", type=int, default=2)
    parser.add_argument("--num_same_case_negatives", type=int, default=4)
    parser.add_argument("--num_distractor_negatives", type=int, default=4)
    parser.add_argument(
        "--distractor_labels",
        type=str,
        default="Background Facts,Procedural History",
        help="Comma-separated corpus labels to sample as safe distractors.",
    )
    parser.add_argument("--base_seed", type=int, default=17)

    parser.add_argument("--eval_query_batch_size", type=int, default=64)
    parser.add_argument("--eval_passage_batch_size", type=int, default=256)
    parser.add_argument("--eval_ks", type=str, default="1,5,10,20,50")

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--deepspeed", type=str, default=None)

    return parser.parse_known_args()


def main() -> None:
    args, _unknown = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")

    local_rank, rank, world_size = _setup_distributed_environment()
    print(
        "SM_DIAG"
        f" rank={rank}"
        f" local_rank={local_rank}"
        f" world_size={world_size}"
        f" cuda_available={torch.cuda.is_available()}"
        f" cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        f" cuda_current_device={torch.cuda.current_device() if torch.cuda.is_available() else -1}"
    )
    if torch.cuda.is_available():
        print(f"SM_DIAG cuda_device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    if not args.processed_dir:
        raise ValueError("--processed_dir is required (or set SM_CHANNEL_DATA)")

    processed_dir = Path(args.processed_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": all_markup_tokens()})
    slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
    if slot_token_id == tokenizer.unk_token_id:
        raise ValueError(f"{SLOT_TOKEN} was not added to the tokenizer")

    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    encoder.resize_token_embeddings(len(tokenizer))
    model = DualEncoderRetriever(encoder=encoder, slot_token_id=slot_token_id, temperature=args.temperature)

    corpus_by_passage_id = load_corpus(processed_dir)
    passage_text_by_passage_id = {pid: passage.text for pid, passage in corpus_by_passage_id.items()}

    candidates_by_case = load_candidates_by_case(processed_dir)
    distractor_passage_ids = select_distractor_passage_ids(
        corpus_by_passage_id,
        distractor_labels=_parse_csv(args.distractor_labels),
    )

    train_queries = load_queries(processed_dir, args.train_split)
    val_queries = load_queries(processed_dir, args.val_split)

    train_dataset = MultiPositiveRetrievalTrainDataset(
        train_queries,
        candidates_by_case,
        distractor_passage_ids,
        base_seed=args.base_seed,
        max_pos_per_query=args.max_pos_per_query,
        num_same_case_negatives=args.num_same_case_negatives,
        num_distractor_negatives=args.num_distractor_negatives,
    )
    val_dataset = MultiPositiveRetrievalTrainDataset(
        val_queries,
        candidates_by_case,
        distractor_passage_ids,
        base_seed=args.base_seed + 100,
        max_pos_per_query=args.max_pos_per_query,
        num_same_case_negatives=args.num_same_case_negatives,
        num_distractor_negatives=args.num_distractor_negatives,
    )

    collator = RetrievalBatchCollator(
        tokenizer,
        passage_text_by_passage_id,
        max_len_query=args.max_len_query,
        max_len_passage=args.max_len_passage,
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.batch_size_queries),
        per_device_eval_batch_size=int(args.batch_size_queries),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=int(args.logging_steps),
        bf16=True,
        deepspeed=args.deepspeed,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=[],
        save_only_model=True,
        local_rank=int(os.environ.get("LOCAL_RANK", local_rank)),
    )
    print(
        "SM_DIAG"
        f" training_args.device={getattr(training_args, 'device', None)}"
        f" training_args.local_rank={getattr(training_args, 'local_rank', None)}"
    )

    eval_config = RetrievalEvalConfig(
        processed_dir=processed_dir,
        split=args.val_split,
        max_len_query=int(args.max_len_query),
        max_len_passage=int(args.max_len_passage),
        query_batch_size=int(args.eval_query_batch_size),
        passage_batch_size=int(args.eval_passage_batch_size),
        ks=tuple(int(x) for x in _parse_csv(args.eval_ks)),
    )

    trainer = MultiPositiveContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[SetEpochCallback(train_dataset)],
        retrieval_eval_config=eval_config,
    )

    trainer.train()

    accelerator = trainer.accelerator
    deepspeed_engine = getattr(trainer, "deepspeed", None)
    state_dict = accelerator.get_state_dict(deepspeed_engine if deepspeed_engine is not None else trainer.model)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unset_hf_deepspeed_config()
        cpu_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        cpu_tokenizer.add_special_tokens({"additional_special_tokens": all_markup_tokens()})
        cpu_slot_token_id = int(cpu_tokenizer.convert_tokens_to_ids(SLOT_TOKEN))

        cpu_encoder = AutoModel.from_pretrained(args.model_name_or_path)
        cpu_encoder.resize_token_embeddings(len(cpu_tokenizer))
        cpu_model = DualEncoderRetriever(
            encoder=cpu_encoder,
            slot_token_id=cpu_slot_token_id,
            temperature=float(args.temperature),
        )

        missing, unexpected = cpu_model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading consolidated weights: {unexpected}")

        save_model(cpu_model, str(model_dir / "model.safetensors"))
        cpu_tokenizer.save_pretrained(str(model_dir))
        AutoConfig.from_pretrained(args.model_name_or_path).save_pretrained(str(model_dir / "encoder_config"))

        wrapper_config = {
            "model_type": "slot_dual_encoder_retriever",
            "base_model_name_or_path": args.model_name_or_path,
            "slot_token": SLOT_TOKEN,
            "temperature": float(args.temperature),
            "weight_file": "model.safetensors",
            "markup_tokens": all_markup_tokens(),
        }
        (model_dir / "wrapper_config.json").write_text(json.dumps(wrapper_config, indent=2), encoding="utf-8")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
