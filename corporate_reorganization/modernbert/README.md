# ModernBERT retriever (corporate_reorganization)

This folder is the home for the ModernBERT dual-encoder retriever code for the corporate reorganization dataset.

## Dataset build

- `corporate_reorganization/modernbert/data_prep/build_final_annotations_gold_dataset.py`

This writes the processed dataset under:

- `corporate_reorganization/data/final_annotations_gold/processed/`

## Training (SageMaker / Deepspeed)

Entry point:

- `corporate_reorganization/modernbert/train_sm.py`

Key behavior:

- **Multi-positive** contrastive loss (a query can have 1+ positives).
- Query embedding = hidden state at the `[MASK]` token.
- Passage embedding = mean pooling over non-padding tokens (excluding position 0).
- Negatives = all same-case candidates (padded to max case size) + cross-case negatives (default 32, label-filtered).
- Validation logs both **contrastive loss** (`eval_loss`) and **retrieval metrics** (e.g. `eval_recall_at_20`).

SageMaker notebook template:

- `corporate_reorganization/notebooks/sagemaker_retriever_training.ipynb`

## Evaluation

CLI entry point:

- `corporate_reorganization/modernbert/eval_retriever.py`

Example (download model artifact from S3, evaluate on test split):

```bash
python corporate_reorganization/modernbert/eval_retriever.py \
  --processed_dir corporate_reorganization/data/final_annotations_gold/processed \
  --split test \
  --model_s3_uri s3://.../output/model.tar.gz \
  --output_dir corporate_reorganization/data/final_annotations_gold/eval_runs/latest
```

Outputs:

- `results.json` (global + breakdown metrics, includes random baseline)
- `config.json`
- `report.md` (quick summary table: model vs random)
- `runs/topk_examples.jsonl`

Notes:

- `--model_s3_uri` requires AWS credentials and either `boto3` or the `aws` CLI available in your environment.

## Gradient accumulation

`train_sm.py` supports gradient accumulation via:

- `--effective_batch_size_queries` (default: `64`) to auto-compute `gradient_accumulation_steps`, or
- `--gradient_accumulation_steps` to explicitly set it.

The effective global queries per optimizer step is:

`batch_size_queries * world_size * gradient_accumulation_steps`.
