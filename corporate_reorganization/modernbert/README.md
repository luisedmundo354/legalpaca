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

`train_sm.py` is the strict 20-epoch controlled ARR entry point. It accepts one frozen
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
batches, and optimizer-window normalization. Step 5 implements the exact
cross-rank passage path: corpus-wide integer indices, deterministic global
deduplication and balanced ownership, padded autograd-aware gathering, and
all-gold multi-positive loss masks.

Step 6 evaluates the complete held-out validation fold after every epoch. Each
rank owns sorted global positions modulo four and makes exactly seven paired
top-level DeepSpeed forwards; rank zero scores all validation queries against
all and only validation-fold passages in CPU float32 and broadcasts one
canonical result. Checkpoint selection is lexicographic: maximize case-macro
set recall@20, then case-macro full-ranking first-gold reciprocal rank, then
retain the earlier epoch.

Every epoch checkpoint is an atomically published, explicit-tag ZeRO-3
checkpoint containing model and optimizer shards, the external Transformers
scheduler, per-rank RNG state, Trainer state, validation selection metrics and
digests, and hashes of every file. The complete per-query/per-case result is in
the corresponding validation epoch metadata. Collective retention keeps
exactly the selected best and chronological last checkpoints. After epoch 20,
the code destroys Engine A,
constructs pristine Engine B, strictly loads the selected explicit tag on all
ranks, restores optimizer/scheduler/RNG state, and requires exact validation
reproduction. This is a same-run verification/export path, not a general
Trainer-resume interface. The final `model.safetensors` is the BF16 state
gathered from the verified Engine B and is strict-loaded into two fresh CPU
retrievers to prove a bitwise round trip. `artifact_manifest.json` is published
last as the successful-run commit marker.

## Evaluation

`eval_retriever.py` retains the Step 7 controlled-artifact-only local-plan
interface. `processing_eval/evaluate_sm.py` is the Step 8 production entry
point: it requires the exact twelve controlled systems plus BM25 flat-plain,
E5-base-v2 flat-plain, and the one fixed-seed ModernBERT-base flat-masked
artifact. Both accept only canonical immutable plan bytes, explicit local
bindings, an absent output directory, and an explicit device. Step 9 freezes
the production plans in version control:

```bash
python corporate_reorganization/modernbert/eval_retriever.py \
  --evaluation-plan /path/to/immutable-plan.json \
  --local-bindings /path/to/local-bindings.json \
  --output-dir /path/to/absent-result-directory \
  --device cuda:0

python corporate_reorganization/modernbert/processing_eval/evaluate_sm.py \
  --evaluation-plan /path/to/immutable-complete-plan.json \
  --local-bindings /path/to/complete-local-bindings.json \
  --output-dir /path/to/absent-complete-result-directory \
  --device cuda:0
```

The evaluator never downloads or discovers a model. AWS orchestration must
mount already-extracted local artifacts and bind their expected hashes. Each
controlled artifact must have the Step 6 `artifact_manifest.json` commit
marker, exact full-tree hashes, matching fold/view/sampler/seed identities, the
nested tokenizer and encoder config, and a BF16 tied-weight safetensors model.
Loading is strict and immutable; no tokenizer patch, `strict=False`, retry, or
fallback path exists.

The Step 7 local runner still accepts only trained controlled dual-encoder
artifacts. The Step 8 Processing runner requires all fifteen systems and
prevalidates every controlled artifact, model snapshot, E5 pack, fixed-base
artifact, data/config input, and image identity before scoring. It evaluates
BM25, E5, the fixed ModernBERT control, and each trained model serially; there
is no partial baseline or missing-artifact fallback.

Every system first produces one finite CPU-float32 score for every query and
role-fold passage. The canonical kernel ranks by score descending and passage
ID ascending, then derives exactly four regimes without rescoring:

- `same_case_legacy`
- `same_case_full`
- `fold_global`
- `fold_global_context_excluded`

The last regime filters visible nongold passages from the complete fold-global
ranking; a visible gold is retained. `global_split` is only the historical
March label for role-local fold-global evaluation and is rejected by the
canonical interface.

Successful output is an atomically published directory containing:

- `evaluation_config.json`
- `rankings.jsonl`, with every candidate score and per-query metric
- `results.json`, with per-case, query-micro, and case-macro metrics
- `artifact_manifest.json`, written last as the commit marker

Readback reconstructs every result from the complete rankings and rejects
missing queries/golds, duplicate or truncated candidates, unstable ties,
non-finite scores, schema changes, or hash changes. The controlled final
lengths are 4,096 query tokens and 500 passage tokens.

`legacy_eval/march.py` is a separate hash-gated, read-only replay of the
reconstructed March archive. It verifies 600 complete ranking rows and all
3,300 stored numeric values using the historical names, tie order, and
sequential summation. It does not claim model-to-ranking reproduction. The old
S3 downloader, sampled random baseline, Cohere path, top-K-only output, and
duplicate evaluator packages have been removed.
`../notebooks/sagemaker_retriever_evaluation_processing.ipynb` is retained only
as an explicitly non-runnable March provenance record.

## Controlled verification

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

The Step 6 CPU/Gloo contracts run in the same pinned base image:

    python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_validation \
      corporate_reorganization.modernbert.tests.test_retrieval_checkpointing \
      corporate_reorganization.modernbert.tests.test_retrieval_trainer_lifecycle_runtime

The two-epoch determinism-smoke runtime, artifact, request, and exact comparison
contracts run in the derived training image with networking disabled:

    python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_determinism \
      corporate_reorganization.modernbert.tests.test_retrieval_determinism_trainer_runtime \
      corporate_reorganization.modernbert.tests.test_retrieval_determinism_artifacts \
      corporate_reorganization.modernbert.tests.test_retrieval_cv_training_aws \
      corporate_reorganization.modernbert.tests.test_retrieval_cv_determinism_gate

The Step 7 scientific, archive, artifact, ranker, bundle, and CLI contracts run
in the pinned image as well:

    python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_canonical_evaluation \
      corporate_reorganization.modernbert.tests.test_retrieval_legacy_march \
      corporate_reorganization.modernbert.tests.test_retrieval_artifacts \
      corporate_reorganization.modernbert.tests.test_retrieval_rankers \
      corporate_reorganization.modernbert.tests.test_retrieval_evaluator_outputs \
      corporate_reorganization.modernbert.tests.test_retrieval_evaluation_plan \
      corporate_reorganization.modernbert.tests.test_retrieval_legacy_trainer_eval \
      corporate_reorganization.modernbert.tests.test_retrieval_eval_cli

The artifact suite's real tied-safetensors/ModernBERT test requires
`ARR_TOKENIZER_DIR` to point at the frozen local snapshot. Without that
variable it reports a skip, which is not evidence for the real-artifact gate.

The Step 8 complete evaluator runs only in the derived Processing image. Build
that image from the two frozen contexts documented in
`experiments/retrieval_cv/README.md`, require identical config and manifest
digests, then run its in-image contract check by local manifest digest:

    VERIFIED_IMAGE_URI='arr-retrieval-eval@sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2'
    docker run --rm --network none \
      --entrypoint /opt/conda/bin/python \
      "${VERIFIED_IMAGE_URI}" \
      /opt/program/modernbert/processing_eval/image_smoke.py \
      --contract /opt/program/modernbert/processing_eval/image_contract.json

The check binds the exact Corretto/Anserini sparse runtime, neural package
versions, frozen source bytes and modes, build identity, parent provenance,
and offline environment. A tag is not launch evidence. Step 9 must compare the
remote ECR manifest with the verified local manifest and launch only
`repository@sha256:...`.

The derived evaluation runtime intentionally does not embed training code or
the GPU lifecycle tests. A local skip is expected on a CPU-only host and is not
a launch pass. Step 9 builds a canonical source bundle from a clean committed
worktree; mount that separately to run the two-GPU lifecycle gate and the
four-GPU NCCL/BF16 tests on a production-shaped `ml.g5.12xlarge` pilot before
submitting the 60 controlled jobs.

    python -m unittest -v \
      corporate_reorganization.modernbert.tests.test_retrieval_deepspeed_lifecycle_cuda

Do not let the SageMaker training toolkit install `requirements.txt` at job
startup. Archived March runs built different DeepSpeed wheel bytes for the same
declared version. `training_image/` therefore derives a separate image from the
SDK-selected DLC, installs four direct hash-locked artifacts with dependency
resolution disabled, preserves the original SageMaker training entrypoint, and
validates the complete runtime contract. Two no-cache builds must have the same
Docker config and manifest digests before publication; jobs use only the ECR
digest URI.

The checked builder rejects a different Docker driver, Buildx/BuildKit version,
source inventory, exporter option, existing metadata path, config digest, or
manifest digest. Rebuild the two accepted replicas only with:

    python -m corporate_reorganization.modernbert.training_image.build \
      --modernbert-dir "$PWD/corporate_reorganization/modernbert" \
      --metadata-file /tmp/arr-training-replica-1.json \
      --build-replica 1
    python -m corporate_reorganization.modernbert.training_image.build \
      --modernbert-dir "$PWD/corporate_reorganization/modernbert" \
      --metadata-file /tmp/arr-training-replica-2.json \
      --build-replica 2

Both metadata paths must be absent. Each invocation uses `--pull --no-cache`,
the Docker image exporter with timestamp rewriting and no unpack, and the
recorded source epoch. It accepts only manifest `b44c9b18...` and config
`24784672...`; a different rebuild is a hard failure, not a replacement image.

The derived image also bakes a read-only trusted bootstrap. With network
isolation enabled, SageMaker mounts the normalized source archive as a File-mode
`source` channel rather than asking the training toolkit to download code. Each
MPI rank independently verifies the requested archive name, size, SHA-256,
normalized tar inventory, commit epoch, runtime contract, and active bootstrap
bytes before extracting into a new rank-local directory and executing the
verified `train_sm.py`. Caller-provided environment values are treated only as
requests; the controlled entry point accepts the corresponding provenance only
after the baked bootstrap has re-emitted independently verified identities.

The Step-9 `retrieval_cv_training_plan` is intentionally non-submittable. The
current source records the derived image digest, runtime inventory, image
contract, base DLC identity, verified source bundle, and launch-receipt hashes
in controlled artifacts. The planned two-epoch determinism and corrected-data
legacy records still have no executable strict entry point. Those remaining
gates and the final remote submission/verification boundary must be implemented
and re-frozen before any training job is submitted; the evaluation-image
Processing runtime smoke does not waive them.
