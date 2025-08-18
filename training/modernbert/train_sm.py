import argparse
import os, json
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, AutoConfig
from dataset_json_pairs import JsonPairDataset
from model_prefix_suffix import PrefixSuffixModel
from trainer_contrastive import ContrastiveTrainer
import numpy as np
import evaluate
from transformers import EvalPrediction
from transformers.utils import logging
import deepspeed
import torch
from safetensors.torch import save_model
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

# Compute custom classification metrics
accuracy = evaluate.load("accuracy")
precisionM = evaluate.load("precision")
recallM = evaluate.load("recall")
f1M = evaluate.load("f1")

def compute_metrics(eval_pred: EvalPrediction):
    logits       = eval_pred.predictions
    labels       = eval_pred.label_ids
    preds        = np.argmax(logits, axis=-1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    prec = precisionM.compute(predictions=preds, references=labels, average="weighted")["precision"]
    rec = recallM.compute(predictions=preds, references=labels, average="weighted")["recall"]
    f1 = f1M.compute(predictions=preds, references=labels, average="weighted")["f1"]

    metrics = {
        "eval_accuracy": acc,
        "eval_precision": prec,
        "eval_recall": rec,
        "eval_f1": f1,
    }

    trainer.log(metrics)
    return metrics

def parse_arguments():
    parser = argparse.ArgumentParser()

    # hyperparameters as CLI flags
    parser.add_argument('--epochs', '--num_train_epochs', dest='epochs', type=int,
                        default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='Contrastive temperature')

    # model output and data directories (support both SM channel names and clearer aliases)
    parser.add_argument(
        '--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'),
        help='Directory to save model checkpoints and tokenizer'
    )
    parser.add_argument('--train-dir', '--train', dest='train_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'),
                        help='Directory with training JSONL files')
    parser.add_argument('--val-dir', '--test', dest='val_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST'),
                        help='Directory with validation JSONL files')
    parser.add_argument('--local_rank', dest='local_rank', type=int,
                        default=int(os.getenv('LOCAL_RANK', 0)),
                        help='Local rank for distributed training')

    parser.add_argument(
        "--deepspeed",
        type=str,
        help="Path to Deepseed configuration file JSON",
    )

    return parser.parse_known_args()

if __name__ == '__main__':
    args, _ = parse_arguments()

    # Initialize tokenizer and models
    tok = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    m1 = AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-base')
    m2 = AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-base')
    m1.to('cuda')
    m2.to('cuda')
    model = PrefixSuffixModel(args, m1, m2)

    # Prepare datasets
    train_ds = JsonPairDataset(args.train_dir, tok)
    val_ds = JsonPairDataset(args.val_dir, tok)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=True,
        deepspeed=args.deepspeed,
        dataloader_drop_last=True,
        save_only_model=True,
    )

    # Initialize Trainer with contrastive loss
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    accel = trainer.accelerator
    out_dir = os.environ.get("SM_MODEL_DIR", training_args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Consolidate ZeRO-3
    state_dict = accel.get_state_dict(trainer.deepspeed)
    accel.wait_for_everyone()  # sync all ranks after consolidation

    # 2) Rank-0 only: rebuild a clean CPU copy, load weights, and save with model-aware safetensors.
    if accel.is_main_process:
        # Prevent Deepspeed’s global zero.Init auto-wrap from affecting any model construction here.
        unset_hf_deepspeed_config()

        # Recreate your architecture on CPU (no DS hooks active now).
        m1_cpu = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
        m2_cpu = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
        cpu_model = PrefixSuffixModel(args, m1_cpu, m2_cpu)

        # Load the consolidated weights into the CPU model.
        missing, unexpected = cpu_model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in state_dict: {unexpected}")

        # Model-aware safetensors write
        save_model(cpu_model, os.path.join(out_dir, "model.safetensors"))

    # Save tokenizer for later use.
        tok.save_pretrained(out_dir)

    # Save each base encoder’s config (tiny JSONs; no weights).
        AutoConfig.from_pretrained("answerdotai/ModernBERT-base").save_pretrained(
            os.path.join(out_dir, "prefix_encoder")
        )
        AutoConfig.from_pretrained("answerdotai/ModernBERT-base").save_pretrained(
            os.path.join(out_dir, "suffix_encoder")
        )

    # Save a small wrapper metadata file so your inference code knows how to rebuild the module.
        meta = {
        "model_type": "prefix_suffix_dual_encoder",
        "prefix_model_name_or_path": "answerdotai/ModernBERT-base",
        "suffix_model_name_or_path": "answerdotai/ModernBERT-base",
        "temperature": getattr(args, "temperature", 0.01),
        "weight_file": "model.safetensors"
    }
        with open(os.path.join(out_dir, "wrapper_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

# 3) Final barrier so all ranks exit together cleanly.
    accel.wait_for_everyone()
# --------------------------------------------------------------------------



