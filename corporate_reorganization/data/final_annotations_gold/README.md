# final_annotations_gold retrieval dataset

This directory contains both the immutable March ARR inputs and the corrected
case-disjoint retrieval dataset used by the new controlled experiments.

## Directory boundary

- raw/ contains the 42 adjudicated Label Studio exports.
- processed/ is the immutable reconstructed March dataset: 5,286 passages and
  471 queries under the historical direction-blind builder and fixed split.
- processed_retrieval_v2/ is the immutable direction-corrected dataset:
  5,286 passages and 490 queries across all 42 cases.

Never overwrite either processed directory. The corrected builder refuses an
existing file, directory, or symlink at its output path.

## Corrected build

Run from the legalpaca repository root with the exact tokenizer environment and
a fresh verification path:

    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      python -m corporate_reorganization.modernbert.data_prep.build_final_annotations_gold_dataset \
      --raw_dir corporate_reorganization/data/final_annotations_gold/raw \
      --processed_dir /tmp/legalpaca-processed-retrieval-v2-verification \
      --tokenizer_dir /path/to/answerdotai--ModernBERT-base/8949b909ec900327062f0ebf497f51aef5e6f0c8

Compare that fresh tree byte-for-byte with committed processed_retrieval_v2/.
Passing the committed directory itself is expected to fail because overwrite
protection is part of the scientific contract.

The builder requires transformers 4.49.0 and tokenizers 0.21.4. It loads only
the local pinned snapshot, hard-gates byte size and SHA-256 for config.json,
special_tokens_map.json, tokenizer.json, and tokenizer_config.json, and never
performs a network lookup. The tokenizer.json SHA-256 is
9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1.

## Direction and graph invariants

Label Studio relations are normalized as follows:

- right: premise=from_id and conclusion=to_id
- left: premise=to_id and conclusion=from_id

The build fails on an absent or unknown direction, invalid endpoint, self-edge,
duplicate or contradictory edge, cycle, rootless case, duplicate identifier,
bad mask rendering, or case with zero retrieval queries. Multiple roots and
isolated annotations are valid; every relation-bearing component is a DAG.

## Corrected output

processed_retrieval_v2 contains exactly:

- cases.jsonl: case counts, ref_id, roots, labels, and direction counts
- corpus.jsonl: every sentence in every case, including Background Facts,
  Procedural History, and Unlabeled sentences
- queries/all.jsonl: all queries before any experiment fold is assigned
- pools/candidates_by_case.json: complete sentence IDs by case
- pools/candidates_global.json: all 5,286 sentence IDs
- dataset_manifest.json: raw/source/tokenizer/output hashes, counts, and
  diagnostics; the manifest excludes its own hash

No train/validation/test split or qrels file is embedded in this immutable
dataset. The independently hashed fold manifest is created by the experiment
pipeline.

Each query contains:

- one structured query with exactly one [MASK]
- flat plain and flat masked views with the same source-node content
- one or more positive sentence passage IDs
- visible source-node and sentence passage IDs
- any visible/gold overlap and exact pinned-tokenizer counts for both trained
  query views

Visibility is source-aware: a sentence is visible only when its byte-exact
text occurs in a rendered, non-masked source node. Serialized markup is never
searched for passage text because bracket boundaries create false matches.
Two query/passage pairs are both visible and gold (cases 78 and 86); gold wins
and both passages remain in every candidate pool.

The canonical manifest verifies 42 cases, 800 nodes, 644 relations (636 right
and 8 left), 44 roots, 5,286 passages, and 490 queries. Case 42 has final
holding root ENq9-QCWLD and 12 queries. Under the pinned tokenizer, the
maximum structured and flat-masked lengths are 3,027 and 3,062 tokens, so no
visible passage is lost at the 4,096-token limit.

## Markup

- Structure: [ARG], [/ARG], [ROOT], [TREE], [/TREE], [FOCUS], [/FOCUS],
  [STEP], [/STEP], [CONCL], and [PREMISE]
- Slot: [MASK], exactly once per structured or flat-masked query
- Hidden non-slot nodes: [MISSING]
- Labels: [RULE], [ANALYSIS], [CONCLUSION], [BACKGROUND], and [PROCEDURE]
- Implicit nodes: [IMPLICIT] [LABEL]; they have no corpus passage

The historical processed/ layout and its split-specific queries/qrels remain
documented by the reconstructed March provenance commit and must not be
regenerated with the corrected builder.
