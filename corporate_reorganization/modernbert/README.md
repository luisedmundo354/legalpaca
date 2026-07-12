# ModernBERT retriever (corporate_reorganization)

This folder is the home for the ModernBERT dual-encoder retriever code for the corporate reorganization dataset.

## Dataset build

- corporate_reorganization/modernbert/data_prep/build_final_annotations_gold_dataset.py
- corporate_reorganization/modernbert/data_prep/relations.py

The corrected builder reads every adjudicated case, normalizes Label Studio
relation direction strictly, and writes a new immutable dataset under:

- corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2/

The historical processed/ directory is the immutable March reconstruction and
is never rewritten by the corrected builder. Experiment folds are deliberately
outside both dataset directories.

Run the focused builder suite from the repository root with the exact pinned
tokenizer snapshot:

    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
      ARR_TOKENIZER_DIR=/path/to/pinned/ModernBERT/snapshot \
      python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_relations \
      corporate_reorganization.modernbert.tests.test_final_annotations_builder

## Training paths

`train_sm.py` is the strict controlled ARR entry point. It accepts one frozen
outer fold, query view, sampler, and experiment seed; rejects unknown options;
validates exact data, fold, model, runtime, and DeepSpeed inputs; and refuses a
nonempty output directory. It loads the pinned ModernBERT snapshot only from
the SageMaker `base_model` channel with Hub access disabled.
The GPU backend is explicitly FlashAttention 2.7.3 with ModernBERT's
`deterministic_flash_attn` flag set and checked on every attention module; the
snapshot's original false value is never used for controlled training.

`legacy_train_sm.py` preserves the reconstructed permissive March entry point
and sampler behavior. It is isolated for the separately labeled legacy
replication attempt and must not be used for the controlled comparison.

The controlled encoder remains multi-positive: a query embedding is the hidden
state at its single slot token and passage embeddings are mean pooled over
non-padding tokens after position zero. The controlled samplers and complete
scientific matrix are documented in `experiments/retrieval_cv/README.md`.

Step 4 freezes process determinism, global query ordering, lossless final
batches, and optimizer-window normalization. The real cross-rank passage path
is intentionally not certified yet: Step 5 must add unique integer passage
tables, padded autograd-aware gathering, and remote-passage gradients before a
controlled training job can be launched.

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
- The SageMaker processing entrypoint `processing_eval/run_eval_sm.py` now supports multi-system, multi-regime ablations including:
  - `bm25_flat`
  - `dense_open_flat`
  - `base_modernbert_flat`
  - `fine_tuned_flat`
  - `fine_tuned_structured`

## Step 4 verification

The dependency-free controls run in the local environment:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_training_control

The Torch/Transformers/Accelerate contracts run in the exact frozen AWS base
image (DeepSpeed 0.17.1 is installed later from `requirements.txt`):

    docker run --rm --entrypoint python \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -v "$PWD:/workspace:ro" -w /workspace \
      763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training@sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9 \
      -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_training_runtime

The derived runtime must also install the exact requirements and pass the
DeepSpeed/Hugging Face reconciliation suite:

    python -m pip install --no-cache-dir \
      -r corporate_reorganization/modernbert/requirements.txt
    python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_training_deepspeed_runtime

The exact snapshot tokenizer contract is a separate required suite. It freezes
the Transformers 4.49 behavior where 19 markup tokens are supplied and
`add_special_tokens` returns 19, while `[MASK]` already exists so the net
vocabulary growth is 18 rows (50,368 to 50,386):

    docker run --rm --entrypoint python \
      -e ARR_TOKENIZER_DIR=/hf/models--answerdotai--ModernBERT-base/snapshots/8949b909ec900327062f0ebf497f51aef5e6f0c8 \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -v "$PWD:/workspace:ro" -v "$HOME/.cache/huggingface/hub:/hf:ro" \
      -w /workspace \
      763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training@sha256:e6ad17f88da21a7dc1347e68a2009a23827ca24fffdc03226095f46d0e9e53c9 \
      -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_training_snapshot_runtime
