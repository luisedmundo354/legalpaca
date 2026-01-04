# final_annotations_gold dataset (corporate_reorganization)

## Folder layout

```
corporate_reorganization/data/final_annotations_gold/
  raw/                 # Label Studio export JSONs (one per case)
  processed/           # Generated corpus/queries/qrels/splits/pools
```

## How to (re)build `processed/`

From repo root:

```bash
python corporate_reorganization/modernbert/data_prep/build_final_annotations_gold_dataset.py
```

To include Background Facts / Procedural History as *candidate distractors* (never positives):

```bash
python corporate_reorganization/modernbert/data_prep/build_final_annotations_gold_dataset.py --include_background_procedure_candidates
```

## Processed files

### `processed/corpus.jsonl`

One line per retrievable candidate span (implicit nodes are excluded):

```json
{
  "passage_id": "36::msSlm_n1BN",
  "doc_id": "36",
  "label": "Rule",
  "text": "[RULE] ...",
  "start": 7582,
  "end": 8240,
  "is_implicit": false,
  "order": 17
}
```

### `processed/queries/{train,val,test}.jsonl`

One line per query; each query has **1+ positives** and contains exactly one `[SLOT]`:

```json
{
  "query_id": "36::ROOT=...::TARGET=...::MISSING=PREMISE_GROUP_1",
  "doc_id": "36",
  "motion_root_id": "...",
  "mask_parent_id": "...",
  "query_text": "[ARG] ... [PREMISE] [SLOT] ... [/ARG]",
  "positive_passage_ids": ["36::Wl3tM3NLW3", "36::7qmRQq-4Nt"],
  "positive_labels": ["Rule", "Analysis"]
}
```

### `processed/qrels/{train,val,test}.tsv`

Tab-separated relevance judgments:

```
query_id<TAB>passage_id<TAB>1
```

### `processed/splits/{train,val,test}_cases.txt`

Doc-id splits (split-by-case to prevent leakage).

### `processed/pools/`

- `candidates_by_case.json`: `{doc_id: [passage_id, ...]}`
- `candidates_global.json`: `[passage_id, ...]`

## Markup tokens used

- Structure: `[ARG]`, `[/ARG]`, `[TREE]`, `[/TREE]`, `[FOCUS]`, `[/FOCUS]`, `[STEP]`, `[/STEP]`, `[CONCL]`, `[PREMISE]`
- Slot marker: `[SLOT]` (exactly one per query)
- Hidden/masked non-slot nodes: `[MISSING]` (may appear multiple times in `[TREE]` to preserve structure without leaking labels/text)
- Labels: `[RULE]`, `[ANALYSIS]`, `[CONCLUSION]` (and optionally `[BACKGROUND]`, `[PROCEDURE]`)
- Implicit nodes: represented in queries as `[IMPLICIT] [LABEL]` and excluded from `corpus.jsonl`
