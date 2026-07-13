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

Controlled and fixed-base ModernBERT lengths are 4,096/500, matching training
and model selection. The March passage length of 600 is legacy-only. E5 has a
hard 512-position architecture limit and uses one frozen semantic
`focus_preserving_semantic_pack_v1`, not tokenizer-side truncation. After the
two special positions and exact two-WordPiece `query: ` prefix, 508 positions
remain. The pack keeps every focus role and target, reserves the root role and
one complete source word, trims the tail of the longest visible focus content
only when necessary, extends the root, then adds complete non-focus context
steps in frozen order while skipping steps that do not fit. It never reads
positive IDs or labels. The derived 490-query token artifact is separate from
the immutable dataset and is recomputed exactly during validation. Three
queries truncate one visible focus premise each, eleven have partial roots,
and 409 retain at least one optional context step. A future paired
flat/structured E5 diagnostic requires a new jointly fitting allocation; the
reported off-the-shelf baseline remains flat-plain only.

Each fold evaluation contains exactly 15 systems: the twelve controlled
view/sampler/seed cells plus BM25 flat-plain, E5-base-v2 flat-plain, and one
fold-independent fixed-seed ModernBERT-base flat-masked artifact. Every model
and index is handled serially. BM25 uses Java 21, Pyserini 1.5.0, k1=0.9 and
b=0.4, completes unreturned scores with float32 zero, and then enters the same
canonical ranking kernel. The fixed ModernBERT artifact uses seed 17, the exact
training tokenizer extension, BF16 weights, mask-slot query pooling, and
passage mean pooling; two fresh builds must be byte-identical.

The Processing image derives from the digest-pinned training DLC, adds
Corretto 21.0.11.10.1 and the exact Pyserini/PyJNIus artifacts, and preserves
the neural package inventory. Pyserini 1.5.0 declares newer dense-stack
dependencies that conflict with this frozen evaluator, so its build occurs in
an isolated stage with no dependency resolution; only its sparse JNI surface
is copied into the final image. The final process rejects the high-level dense
Pyserini path, the inherited Java 11 binary, mutable image tags, and any
dependency or Anserini-JAR drift. Builds must use the BuildKit image exporter
with `rewrite-timestamp=true` and `unpack=false`; ordinary `--load` preserves
time-varying wheel-install metadata and is not a valid reproducibility gate.
Both fresh builds use the same `SOURCE_DATE_EPOCH`, content hash, and VCS
revision and must produce identical config and manifest digests before either
digest is accepted.

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
- configs/e5_snapshot.json freezes the six-file E5-base-v2 revision.
- configs/evaluation_baselines.json freezes all three baseline protocols and
  their system-to-query-view mapping.
- configs/e5_focus_pack contains the canonical 490-query E5 input IDs and
  strict readback manifest.
- configs/fixed_base_artifact.json records the deterministic untrained
  ModernBERT artifact/model/new-row hashes; the 288 MB artifact itself remains
  an independently mounted evaluation input.
- processing_eval/Dockerfile and image_contract.json define the allowlisted,
  offline SageMaker Processing runtime. Launches use its ECR digest, never its
  content-derived tag.
- folds.py is the only fold generator and validator.
- configs/orchestration.json binds the verified evaluation image, the derived
  training image and its distinct base DLC, the exact Step-10C scientific
  source commit/tree/epoch,
  immutable inputs, and planned templates for the 60 controlled runs, two
  corrected legacy-style diagnostics, and two determinism smokes.
- configs/evaluation_runtime_identity.json is the compact canonical identity
  emitted by the verified evaluation image under the exact Processing request
  environment. CloudWatch verification requires its SHA-256
  `75c1d8fd...`; the Docker stdout formatting hash is not the identity hash.
- config.py and manifest.py validate canonical configuration bytes, build a
  commit-exact normalized source archive without symlinks, ignored files, or
  runtime requirements, and expand the exact ready 60+2+2 training plan. The
  archive is exactly one gzip member with no optional header fields, the source
  commit epoch, level-6 `XFL=0`, and `OS=255`; its inventory identity uses the
  compact canonical JSON encoding enforced by the published training
  bootstrap. Host readback verifies that header, member, and inventory
  contract before the archive can enter a plan.
- aws.py uses low-level, one-attempt Botocore clients for immutable ECR
  publication, checked versioned-S3 primitives, Processing preflight, and one
  explicitly submitted runtime smoke. It never retries or selects a fallback.
- training_aws.py validates and stages exactly one source bundle, six
  corrected-v2 data files, and five ModernBERT snapshot files under previously
  unused versioned prefixes; records every VersionId, ETag, size, SHA-256, and
  SSE setting; and rejects delete markers, extra versions, or non-current
  objects. Its v2 receipt binds an attempt-independent input contract, so a
  later explicit attempt may reuse exactly the same immutable versions but
  cannot change account, bucket, channel, prefix, source, data, model, or
  object identity. It renders a
  controlled CreateTrainingJob request, either corrected legacy-style
  diagnostic request, or one of the two sealed determinism-smoke requests only
  from a validated plan cell and the matching staging receipt. The smoke
  renderer proves replicas A and B differ only in job name, output prefix, and
  the toolkit's duplicated job-name field. The pinned
  training-toolkit 5.0.0 mapping is explicit: scientific snake_case plan keys
  become the hyphenated strict CLI flags consumed by the image-baked bootstrap
  and then by train_sm.py. The
  request uses three slash-bounded File-mode channels (`base_model`, `data`,
  and `source`) under network isolation. Every MPI rank verifies the mounted
  source archive and its normalized inventory before safe rank-local extraction;
  no container-side S3 download is permitted. Every request caps active
  capacity waiting at 7,200 seconds and running time at 86,400 seconds, while
  omitting SageMaker's nonzero-only RetryStrategy.
- training_launch.py is the only training-job mutation boundary. For one named
  plan run, it deeply revalidates all staged versions, the exact SDK/caller/
  role/bucket/ECR/quota/offering state, unused job name and output history, and
  a freshly rendered request. It requires the applied `ml.g5.12xlarge`
  training quota to be at least four, paginates both active SageMaker states,
  and refuses to launch when four planned jobs are already active. Submission
  repeats that complete preflight, requires byte-identical evidence, calls
  CreateTrainingJob exactly once, and immediately verifies DescribeTrainingJob
  plus all tags. Status and terminal verification are read-only; only Completed
  is success, while Failed and Stopped are sealed as explicit failure evidence
  before the CLI fails loudly. It has no waiter, mutation retry, resource
  fallback, or automatic reconciliation path.
- training_artifacts.py is the read-only post-training acquisition boundary.
  It accepts only a recursively validated successful terminal chain, discovers
  exactly one current version at the request-derived model key, downloads that
  exact VersionId, computes the complete archive SHA-256, and publishes one
  absent local bundle containing the archive, safely extracted artifact, and
  acquisition receipt. Publication is atomic and no-replace; multipart ETags
  and composite service checksums are recorded but never treated as a
  whole-object content hash. Its physical gzip/PAX/TAR parser rejects multiple
  members, trailing data, path aliases or traversal, links, special files,
  sparse records, duplicate paths, and nonzero padding before the existing
  strict smoke-artifact validator can accept the result.
- Controlled artifact expectations and exported model identities carry the
  exact plan SHA-256, staging-receipt SHA-256, and five-field source-bundle
  identity in addition to the derived/base image, runtime, contract, and
  bootstrap identities. Evaluation plans must obtain those dynamic values from
  the independently validated plan, staging receipt, and re-rendered per-run
  request receipt, never from the artifact's own `controlled_run.json`. The
  local and complete evaluation-plan schemas are versions 2 and 3 respectively,
  and each plan requires one common plan hash, staging-receipt hash, and source
  bundle across all of its controlled systems.
- The separate `determinism_smoke` path is sealed to structured,
  global-uniform, fold 0, seed 17, two epochs, and six optimizer updates. It
  records canonical initial/last/selected model-state identities, all 588
  candidate links, all 152 rank/microbatch loss records with exact float32 bit
  patterns, both validation decisions, the verified reload, and final artifact
  identities. `retriever/determinism_artifacts.py` independently parses the
  safetensors bytes and recomputes the selected state. `determinism_gate.py`
  validates both externally identified artifacts and rejects any scientific
  mismatch without a tolerance. The v2 gate accepts only the two complete
  acquisition receipts, recursively revalidates each launch/terminal/S3/local
  chain, derives both request receipts itself, and records the two remote
  VersionIds, archive hashes, tree inventories, and artifact identities. It
  has no loose-root or user-supplied manifest-hash interface. Neither module
  submits a job.
- aggregate.py performs only strict five-fold completeness and artifact
  readback. Statistical aggregation, intervals, contrasts, and figures remain
  Step 12.

The ready manifest remains `retrieval_cv_training_plan`: readiness permits only
the sealed coordinator above and does not itself stage or submit anything. The
scientific source claim stays frozen to the exact Step-10C commit containing
the controlled, corrected-diagnostic, and strict 2-epoch/6-update determinism
paths. The first plan defaults to `a1` with no parent. Every later freeze must
name its canonical attempt ID and provide the immediately preceding validated
manifest; the CLI derives and binds that parent file's SHA-256 and rejects a
skipped or inferred ancestor. Before training, the staging command validates the bucket, proves all
three complete versioned prefixes have never contained an object or delete
marker, stages all twelve objects, and binds every VersionId and readback
identity into one receipt. Immediately before each submission, the launcher
re-lists complete version history and deeply re-reads the named versions.
`If-None-Match` alone is not treated as historical-prefix immutability. A
partial staging failure permanently taints that prefix and fails; it is never
cleaned up or retried in place. An ambiguous CreateTrainingJob response also
fails without an automatic retry because that API has no idempotency token.
The collision check treats the documented `ResourceNotFound` response and the
exact live SageMaker response `ValidationException: Requested resource not
found.` as absence; every other validation, authorization, transport, or
malformed error fails unchanged.
For immediate and terminal request readback, SageMaker may echo an omitted
output `KmsKeyId` as the empty string that denotes its default S3 key. The
verifier removes only that exact empty service-default field before comparison;
a nonempty KMS key or any other added, omitted, or changed field still fails.
Before an artifact is available, SageMaker may omit the outer `ModelArtifacts`
field. The verifier normalizes only that omission, or a present exact
`S3ModelArtifacts` empty string, to a null snapshot URI. An explicit null,
malformed object, extra field, non-string value, whitespace, or wrong URI still
fails. A Completed job must report the exact request-derived nonempty model URI
in addition to its complete timing evidence.
After a successful terminal receipt, `acquire determinism-smoke` writes a fixed
`model.tar.gz`, `artifact/`, and `acquisition_receipt.json` layout beneath one
new absolute output directory. `verify determinism-smoke` accepts only the A
and B acquisition-receipt paths plus their common plan and staging receipt and
publishes one canonical, absent-only v2 gate receipt. Acquisition never writes
S3, and gate verification constructs no AWS client.

The immediate Processing smoke is
`evaluation_image_runtime_smoke_v1`. It validates account-local digest pull,
SageMaker startup, and the embedded image contract through CloudWatch. It has
no model inputs and produces no rankings, so it is not a fold-0 evaluation and
cannot satisfy any scientific execution gate. A complete 15-system fold job is
invalid until all twelve controlled artifacts for that fold exist.

AWS-local values live only in ignored `aws.local.json`; credentials and profile
names are forbidden. Run the CLI with the pinned `legalpacaenv`, because
source bundling requires Python 3.11.13 with compile/runtime zlib 1.2.13 and
preflight requires the exact SageMaker 2.248.2/Boto3 1.39.12 stack. Inputs and
outputs use unused prefixes in the versioned, SSE-S3 `ir-sagemaker` bucket.
Container entrypoint/arguments and `/opt/ml/processing/` semantics follow the
[AWS custom Processing container contract](https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html)
and [Processing input/output contract](https://docs.aws.amazon.com/sagemaker/latest/dg/byoc-input-and-output.html).

Build the fixed untrained ModernBERT artifact only inside a locally verified
Processing image addressed by digest. The two output paths and their sibling
`.incomplete` paths must not exist. Mount the same validated snapshot at two
different container paths; this is an intentional path-independence gate, not
two aliases for one completed output:

    VERIFIED_IMAGE_URI='arr-retrieval-eval@sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2'
    MODERNBERT_SNAPSHOT='/tmp/arr-step8-modernbert-snapshot-final'
    CONFIGS="$PWD/corporate_reorganization/modernbert/experiments/retrieval_cv/configs"
    docker run --rm --network none --entrypoint /opt/conda/bin/python \
      -v "${MODERNBERT_SNAPSHOT}:/inputs/modernbert-a:ro" \
      -v "${CONFIGS}:/inputs/configs:ro" \
      -v /tmp:/outputs \
      "${VERIFIED_IMAGE_URI}" \
      /opt/program/modernbert/processing_eval/build_base_modernbert.py \
      --snapshot-dir /inputs/modernbert-a \
      --snapshot-manifest /inputs/configs/modernbert_snapshot.json \
      --baseline-config /inputs/configs/evaluation_baselines.json \
      --artifact-contract /inputs/configs/fixed_base_artifact.json \
      --output-dir /outputs/arr-step8-final-base1
    docker run --rm --network none --entrypoint /opt/conda/bin/python \
      -v "${MODERNBERT_SNAPSHOT}:/different/mount/modernbert-b:ro" \
      -v "${CONFIGS}:/inputs/configs:ro" \
      -v /tmp:/outputs \
      "${VERIFIED_IMAGE_URI}" \
      /opt/program/modernbert/processing_eval/build_base_modernbert.py \
      --snapshot-dir /different/mount/modernbert-b \
      --snapshot-manifest /inputs/configs/modernbert_snapshot.json \
      --baseline-config /inputs/configs/evaluation_baselines.json \
      --artifact-contract /inputs/configs/fixed_base_artifact.json \
      --output-dir /outputs/arr-step8-final-base2
    diff -qr /tmp/arr-step8-final-base1 /tmp/arr-step8-final-base2
    sha256sum /tmp/arr-step8-final-base{1,2}/artifact_manifest.json

Both manifest hashes must equal
`ccff3fa4c141290ef9383992a4d3de2b8cfa5e50d02c4cd06e3fe52e92d0202b`.
The builder rejects a changed baseline or artifact contract before model
construction, revalidates the snapshot and both contracts immediately before
publication, and publishes the commit marker with no-replace semantics.

Freeze the build input twice before invoking BuildKit. The selected source
parent is provenance for the uncommitted Step-8 snapshot; it must exist and be
an ancestor of the checked-out revision. The exact frozen bytes and modes—not
that parent commit—are the image source identity. The freezer derives and
records the active Docker driver, Buildx version, and BuildKit version; callers
cannot supply those identities:

    SOURCE_PARENT_COMMIT='4b4f26852c59f809591edfced61bfc1d13650021'
    SOURCE_PARENT_EPOCH="$(git show -s --format=%ct "${SOURCE_PARENT_COMMIT}")"
    python corporate_reorganization/modernbert/processing_eval/build_context.py \
      --modernbert-dir corporate_reorganization/modernbert \
      --output-dir /tmp/arr-step8-final-context1 \
      --source-parent-commit "${SOURCE_PARENT_COMMIT}" \
      --source-parent-epoch "${SOURCE_PARENT_EPOCH}"
    python corporate_reorganization/modernbert/processing_eval/build_context.py \
      --modernbert-dir corporate_reorganization/modernbert \
      --output-dir /tmp/arr-step8-final-context2 \
      --source-parent-commit "${SOURCE_PARENT_COMMIT}" \
      --source-parent-epoch "${SOURCE_PARENT_EPOCH}"
    diff -qr /tmp/arr-step8-final-context1 /tmp/arr-step8-final-context2
    BUILD_IDENTITY_SHA256="$(python corporate_reorganization/modernbert/processing_eval/build_context.py --validate-frozen-context /tmp/arr-step8-final-context1 --print-build-identity-sha256)"
    CONTENT_TAG="build-sha256-${BUILD_IDENTITY_SHA256}"

Build once from each independently frozen directory through the checked
script. It re-derives the active toolchain, constructs the only accepted
Buildx/exporter command, requires an absent metadata file, revalidates the
context after BuildKit consumes it, and verifies metadata plus the locally
stored image config, entrypoint, environment, labels, and manifest digest:

    python corporate_reorganization/modernbert/processing_eval/build_context.py \
      --build-frozen-context /tmp/arr-step8-final-context1 \
      --metadata-file /tmp/arr-step8-final-image1.json \
      --build-replica 1
    python corporate_reorganization/modernbert/processing_eval/build_context.py \
      --build-frozen-context /tmp/arr-step8-final-context2 \
      --metadata-file /tmp/arr-step8-final-image2.json \
      --build-replica 2

The two metadata files must contain identical `containerimage.config.digest`
and `containerimage.digest` values and Docker manifest media type
`application/vnd.docker.distribution.manifest.v2+json`. Do not publish by
changing one exporter flag and rebuilding. Step 9 must tag this already
verified local manifest with the full ECR repository name, verify immutable-tag
configuration, push once, read back the ECR digest and raw manifest, require
exact local/remote digest equality, then pull and launch only by digest URI.

The accepted local build identity is
`249a373465c33d2af5f807eecf6016b08dc086ca04b588e3a2a6a5a640aa2fc8`;
its frozen file-inventory SHA-256 is
`96f8b4e5569404ed916cd69c4d765b3eb34cbd3f40e3eff8394e9de72f415dc4`.
Both no-cache builds produced config digest
`sha256:76c29a7f5ca0a1a36d0f8b53fe1e49f40ab199f8ff1bc594ddbb09107c7749e8`
and manifest digest
`sha256:00feb4550b52712901933a546a561c18896304e7d72109f0a5ce49220dd12cf2`.

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
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_e5_packing
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_baseline_artifacts
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_retrieval_complete_evaluation_plan
    PYTHONDONTWRITEBYTECODE=1 python -m unittest -v corporate_reorganization.modernbert.tests.test_processing_image_contract

The pinned-container runtime command is documented in
`corporate_reorganization/modernbert/README.md`.
