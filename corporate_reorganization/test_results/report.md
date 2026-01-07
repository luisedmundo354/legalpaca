# Retriever evaluation report (SageMaker Processing)

- Processed dir: `/opt/ml/processing/input/data`
- Split: `test`
- K values: `[1, 5, 10, 20]`

| system | `eval_recall_at_1` | `eval_recall_at_5` | `eval_recall_at_10` | `eval_recall_at_20` | `eval_mrr` | `eval_avg_candidates` | `eval_retrieval_loss` |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine_tuned | 0.0000 | 0.2000 | 0.3000 | 0.6000 | 0.1178 | 138.8000 | 4.1751 |
| base_modernbert | 0.0000 | 0.1000 | 0.2500 | 0.3500 | 0.0581 | 138.8000 | 4.2016 |
| random_baseline | 0.0000 | 0.1000 | 0.1250 | 0.2000 | 0.0512 | 138.8000 | 4.3338 |

Rankings: see `runs/rankings.jsonl`.