# Retrieval experiment report

- Processed dir: `/home/lbrenap/Documents/projects/legalpaca/corporate_reorganization/data/final_annotations_gold/processed`
- Split: `test`
- K values: `[1, 5, 10, 20]`
- Split passage count: `581`

## Systems

- `bm25_flat`: type=`bm25_pyserini` query_view=`flat_plain`
- `dense_open_flat`: type=`open_dense` query_view=`flat_plain`
  source=`intfloat/e5-base-v2`
- `base_modernbert_flat`: type=`modernbert_base` query_view=`flat_masked`
  source=`answerdotai/ModernBERT-base`
- `fine_tuned_flat`: type=`modernbert_artifact` query_view=`flat_masked`
  source=`s3://sagemaker-us-east-1-371087393859/huggingface-pytorch-training-2026-03-15-23-12-42-656/output/model.tar.gz`
- `fine_tuned_structured`: type=`modernbert_artifact` query_view=`structured`
  source=`s3://sagemaker-us-east-1-371087393859/huggingface-pytorch-training-2026-03-16-00-09-29-909/output/model.tar.gz`

## same_case_legacy

| system | `recall_at_1` | `recall_at_5` | `recall_at_10` | `recall_at_20` | `mrr_at_20` | `set_recall_at_20` | `exact_set_match_at_20` | `candidate_pool_size` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_flat | 0.0250 | 0.2250 | 0.4500 | 0.5750 | 0.1313 | 0.3679 | 0.2000 | 138.8000 |
| dense_open_flat | 0.0750 | 0.3500 | 0.4250 | 0.5750 | 0.1842 | 0.3696 | 0.2250 | 138.8000 |
| base_modernbert_flat | 0.0000 | 0.0500 | 0.1750 | 0.3750 | 0.0406 | 0.2225 | 0.1250 | 138.8000 |
| fine_tuned_flat | 0.0500 | 0.2250 | 0.3750 | 0.5000 | 0.1224 | 0.3017 | 0.1500 | 138.8000 |
| fine_tuned_structured | 0.0500 | 0.2000 | 0.3500 | 0.5500 | 0.1303 | 0.3529 | 0.2000 | 138.8000 |

## same_case_full

| system | `recall_at_1` | `recall_at_5` | `recall_at_10` | `recall_at_20` | `mrr_at_20` | `set_recall_at_20` | `exact_set_match_at_20` | `candidate_pool_size` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_flat | 0.0000 | 0.0500 | 0.1000 | 0.4750 | 0.0517 | 0.2300 | 0.0750 | 156.9750 |
| dense_open_flat | 0.0250 | 0.1750 | 0.3250 | 0.4750 | 0.1076 | 0.2567 | 0.1250 | 156.9750 |
| base_modernbert_flat | 0.0000 | 0.0500 | 0.1500 | 0.3500 | 0.0355 | 0.2112 | 0.1250 | 156.9750 |
| fine_tuned_flat | 0.0250 | 0.1250 | 0.3000 | 0.4500 | 0.0896 | 0.2642 | 0.1250 | 156.9750 |
| fine_tuned_structured | 0.0250 | 0.1750 | 0.2750 | 0.5000 | 0.0962 | 0.3133 | 0.2000 | 156.9750 |

## global_split

| system | `recall_at_1` | `recall_at_5` | `recall_at_10` | `recall_at_20` | `mrr_at_20` | `set_recall_at_20` | `exact_set_match_at_20` | `candidate_pool_size` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_flat | 0.0000 | 0.0500 | 0.1000 | 0.4500 | 0.0504 | 0.2238 | 0.0750 | 581.0000 |
| dense_open_flat | 0.0250 | 0.1750 | 0.3250 | 0.3750 | 0.0994 | 0.2254 | 0.1250 | 581.0000 |
| base_modernbert_flat | 0.0000 | 0.0000 | 0.0250 | 0.0750 | 0.0059 | 0.0425 | 0.0250 | 581.0000 |
| fine_tuned_flat | 0.0000 | 0.0250 | 0.0750 | 0.1500 | 0.0200 | 0.0708 | 0.0000 | 581.0000 |
| fine_tuned_structured | 0.0000 | 0.0000 | 0.0750 | 0.1250 | 0.0165 | 0.0771 | 0.0500 | 581.0000 |
