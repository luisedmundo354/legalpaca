# Retrieval cross-validation specification

This directory freezes the case-disjoint five-fold design for the ARR retrieval
extension. The corrected dataset remains immutable; folds are a separate,
independently hashed experiment input.

## Scientific rule

All 42 cases are indivisible. Fold capacities are 9, 9, 8, 8, and 8.
Cases are ordered by descending:

    max(case_queries / 490, case_passages / 5286)

with numeric case ID as the tie-break. Greedy placement minimizes, in exact
rational arithmetic:

    sum over folds [
      (fold_queries / 490)^2
      + (fold_passages / 5286)^2
    ]

Ties use current fold size and fold ID. The author approved one amendment after
the planned balance stop gate: starting from that auditable greedy assignment,
repeatedly choose the strict best-improvement one-case-for-one-case cross-fold
swap under the same objective. Swap ties use lower fold ID, higher fold ID,
numeric case ID from the lower fold, then numeric case ID from the higher fold.
Neutral swaps are rejected. The manifest retains all 42 greedy placements and
all nine swaps.

## Frozen folds

| Fold | Cases | Queries | Passages |
|---|---|---:|---:|
| 0 | 42, 49, 57, 58, 63, 71, 72, 73, 80 | 98 | 1,054 |
| 1 | 38, 40, 60, 62, 68, 87, 91, 92, 96 | 98 | 1,060 |
| 2 | 41, 66, 67, 69, 74, 76, 85, 97 | 98 | 1,055 |
| 3 | 36, 45, 46, 65, 78, 79, 83, 94 | 98 | 1,055 |
| 4 | 37, 47, 48, 59, 70, 75, 77, 86 | 98 | 1,062 |

For outer run i, fold i is test, fold (i+1) mod 5 is validation, and
the remaining three folds are training. Every case is test once, validation
once, and training three times. Each controlled run trains on 294 queries from
24--26 cases.

## Controlled sampling

The controlled experiment uses two strict samplers. `local_unique` draws 40
unique same-case negatives and 20 unique other-case negatives. Its other-case
draw is passage-uniform over the union of eligible passages in every other
training case; it is not a case-uniform or two-stage draw. `global_uniform`
draws 60 unique negatives passage-uniformly from all training-case passages,
including the query's case.

Both samplers exclude only every gold passage of the current query. Golds of
other queries and visible context remain eligible. A query contributes all of
its golds when it has at most four, or a uniform subset of four when it has
more. Positive selection is matched across sampler and query-view conditions.
There is no replacement, padding, or compensation for fewer than four golds:
every example has exactly 60 negatives and therefore 61--64 candidates.

Selection ranks passages by a versioned SHA-256 digest keyed by experiment
seed, epoch, query ID, sampling component, and passage ID. Each sampled example
emits its selected strata and a checksum over the canonical trace payload.
Pool insufficiency, duplicate IDs, incomplete case pools, or invalid golds are
fatal errors.

The reconstructed March sampler remains isolated in
`retriever/legacy_sampling.py`. It intentionally retains the historical
case-wide gold exclusion, 56+4 Background-Facts configuration, replacement
paths, negative compensation, and query-index-based Python RNG behavior. It is
not used by the controlled comparison.

## Controlled training infrastructure

The strict controlled entry point is `modernbert/train_sm.py`; the reconstructed
March path is `modernbert/legacy_train_sm.py`. The controlled process requires
the exact AWS training image recorded in `configs/experiment.json`, the pinned
Python/package inventory, four MPI ranks on one `ml.g5.12xlarge`, and the
five-file ModernBERT snapshot described by `configs/modernbert_snapshot.json`.
It validates `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline Hub
flags, every snapshot file, and the complete ZeRO-3 configuration before use.
The exact tokenizer extension supplies 19 markup tokens; Transformers 4.49
returns 19 from `add_special_tokens`, while the existing `[MASK]` token makes
the net vocabulary growth 18 rows, from 50,368 to 50,386.

The pinned image includes FlashAttention 2.7.3. ModernBERT's base config stores
`deterministic_flash_attn=false`, so the controlled entry point explicitly
changes that runtime config to true before model construction, requires the
resolved `flash_attention_2` backend, and checks the flag on every attention
module. `FLASH_ATTENTION_DETERMINISTIC=1` is also a required pre-import
environment invariant. PyTorch deterministic mode alone does not control this
custom kernel. The snapshot's unspecified `reference_compile` setting is also
forced and asserted to false, preventing runtime-dependent Triton compilation.

Each epoch orders all 294 training query IDs by a versioned SHA-256 digest. A
global batch plan contains 76 rank-ordered four-row batches, which Accelerate
1.4 shards exactly once into 19 batches per rank. Ten sentinel rows pad only
the final global microbatch; the real-query counts there are 2, 2, 1, and 1.
Every real query appears exactly once. The three optimizer windows therefore
contain 128, 128, and 38 real queries and produce three updates per epoch, or
60 updates over 20 epochs.

For local summed query loss S, global valid-query count N for the complete
optimizer window, and data-parallel size D=4, training backpropagates D*S/N.
DeepSpeed averages gradients across ranks, so this equals the mean over every
real query without another gradient-accumulation division. The DeepSpeed
boundary is set explicitly on every microbatch, including the final
three-microbatch window, and gradient clipping is frozen at 1.0 in
`ds_zero3.json`.

The controlled loss uses a corpus-wide immutable passage-index table. Each
microbatch constructs one sorted global union of proposed candidate indices;
position modulo four assigns every real passage to exactly one rank. Padded
autograd-aware all-gather gives every query the same denominator while routing
remote-query gradients to the sole passage owner. Complete all-gold index sets,
not only the at-most-four sampled training positives, define the numerator.
Candidate traces bind every sampled string ID to the transported integer rows.

## Validation and checkpoint selection

Validation is reconstructed from `queries/all.jsonl` and the validation cases
in the selected rotation. Its candidate pool is all and only passages from that
fold: 98 queries and, by fold, 1,054, 1,060, 1,055, 1,055, or 1,062 passages.
Every complete gold set must be contained in the same validation case.

Sorted query and passage positions are sharded modulo four without sampler
padding. Every rank makes exactly seven paired top-level DeepSpeed forwards,
with query chunks of at most four and passage chunks of at most 38. Rank zero
reconstructs the full embeddings, scores in CPU float32, and ranks by score
descending then passage ID ascending. It broadcasts the strict result schema,
including query-micro and case-macro hit, set recall, exact target recovery,
and full-ranking first-gold reciprocal rank.

The selected checkpoint maximizes validation case-macro set recall@20, then
validation case-macro full-ranking first-gold reciprocal rank, with the earlier
epoch winning an exact tie. Training always completes all 20 epochs. Checkpoint
directories use explicit `global_stepN` DeepSpeed tags and include complete
ZeRO-3 model/optimizer shards, the external linear scheduler, per-rank RNG,
Trainer state, validation metadata, and content hashes. Publication and
best/last retention are collective and fail loudly.

Stock `load_best_model_at_end` is disabled because DeepSpeed 0.17.1 requires a
pristine engine for a ZeRO-3 reload. After training, Engine A is collectively
destroyed; an unpartitioned model is rebuilt with the common seed and prepared
as Engine B; every rank strictly loads the selected explicit tag and restores
optimizer, external scheduler, and RNG state. The selected validation result
must reproduce exactly. The final BF16 safetensors artifact comes from this
verified Engine B. A final artifact manifest is written last; its absence means
the run is incomplete.

## Canonical final evaluation

Final evaluation uses the same stable ranking and multi-positive metric kernel
as validation, projected through a complete result schema. Each system must
provide exactly one finite CPU-float32 score for every query and every passage
in the evaluated role fold. The source ranking is always
`(score descending, passage_id ascending)`; all narrower pools are membership
filters of that ranking and are never rescored.

The four canonical regimes are:

1. `same_case_legacy`: the query case with other-query-only golds excluded and
   current-query gold precedence;
2. `same_case_full`: every passage in the query case;
3. `fold_global`: every passage in the one evaluated validation/test fold;
4. `fold_global_context_excluded`: remove visible nongolds from the complete
   fold-global ranking while retaining visible golds.

`global_split` is not a fifth controlled regime. It is retained only as the
reconstructed March label for the same role-local global construction on the
historical four-case test split.

At K=1/5/10/20, every query stores Hit, gold-set recall, exact-target recovery,
and reciprocal rank of the first gold in the full ranking. Aggregation first
averages queries within each case, then averages cases; query-micro values are
also retained. Complete rankings and scores are the source of truth, and every
aggregate is recomputed during strict readback.

Controlled final lengths are 4,096/500, matching training and model selection.
The March passage length of 600 is legacy-only. E5 is architecturally limited
to 512 tokens; its controlled query truncation side is intentionally a required
future evaluation-plan field and must be author-approved before the Step 8
image is frozen.

The local evaluator accepts no S3 URI. Later AWS orchestration mounts exact
local artifacts and passes a canonical plan plus local bindings. Publication
uses a sibling `.incomplete` directory and writes `artifact_manifest.json`
last. A missing commit marker means failure.

The frozen experiment configuration keeps `fold_global` as the primary
endpoint and context exclusion as the robustness endpoint. The two same-case
regimes are mandatory diagnostic reporting views, not additional co-primary
endpoints; Step 9 plans bind all four output regimes without rewriting the
Step-2-frozen experiment specification.

## Files

- configs/folds.json is the immutable generated fold manifest. It records
  dataset/source hashes, exact formulas and tie-breaks, greedy trace, swap
  trace, final folds, and all five role rotations.
- configs/experiment.json is the locked scientific design: seeds, query views,
  samplers, training settings, metrics, analysis, and artifact hashes.
- configs/modernbert_snapshot.json is the canonical five-file local snapshot
  inventory, revision, sizes, SHA-256 hashes, and tree hash.
- folds.py is the only fold generator and validator.

Regenerate into a fresh path and compare bytes:

    PYTHONDONTWRITEBYTECODE=1 python -m corporate_reorganization.modernbert.experiments.retrieval_cv.folds freeze --dataset-dir corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2 --output /tmp/legalpaca-retrieval-folds-verification.json

Validate the frozen manifest directly:

    PYTHONDONTWRITEBYTECODE=1 python -m corporate_reorganization.modernbert.experiments.retrieval_cv.folds validate --dataset-dir corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2 --folds corporate_reorganization/modernbert/experiments/retrieval_cv/configs/folds.json

Run the focused tests:

    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_cv_folds
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_sampling
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_training_control
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_validation
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_checkpointing
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_trainer_lifecycle_runtime
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_deepspeed_lifecycle_cuda
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_canonical_evaluation
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_legacy_march
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_artifacts
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_rankers
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_evaluator_outputs
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_evaluation_plan
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_legacy_trainer_eval
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_eval_cli

The pinned-container runtime command is documented in
`corporate_reorganization/modernbert/README.md`.
