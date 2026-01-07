from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _first_hit_rank(retrieved_passage_ids: Sequence[str], gold_set: Set[str]) -> int:
    for rank_idx, passage_id in enumerate(retrieved_passage_ids, start=1):
        if passage_id in gold_set:
            return rank_idx
    return 0


def mrr_full(
    *,
    retrieved_passage_ids: Sequence[str],
    gold_passage_ids: Sequence[str],
) -> float:
    gold_set = set(str(x) for x in gold_passage_ids if str(x))
    hit_rank = _first_hit_rank(retrieved_passage_ids, gold_set)
    return (1.0 / float(hit_rank)) if hit_rank > 0 else 0.0


def compute_query_metrics(
    *,
    retrieved_passage_ids: Sequence[str],
    gold_passage_ids: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    gold_set = set(str(x) for x in gold_passage_ids if str(x))
    hit_rank = _first_hit_rank(retrieved_passage_ids, gold_set)
    num_gold = float(len(gold_set))

    metrics: Dict[str, float] = {}
    for k in ks:
        k = int(k)
        topk = list(retrieved_passage_ids[:k])
        topk_set = set(topk)
        any_hit = hit_rank != 0 and hit_rank <= k

        metrics[f"recall_at_{k}"] = 1.0 if any_hit else 0.0
        metrics[f"mrr_at_{k}"] = (1.0 / float(hit_rank)) if any_hit else 0.0

        if num_gold > 0:
            metrics[f"set_recall_at_{k}"] = float(len(topk_set & gold_set)) / num_gold
            metrics[f"exact_set_match_at_{k}"] = 1.0 if gold_set.issubset(topk_set) else 0.0
        else:
            metrics[f"set_recall_at_{k}"] = 0.0
            metrics[f"exact_set_match_at_{k}"] = 0.0

    metrics["num_gold"] = num_gold
    metrics["first_hit_rank"] = float(hit_rank)
    return metrics


def aggregate_metrics(per_query_metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {}
    sums: Dict[str, float] = {}
    for row in per_query_metrics:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
    return {key: value / float(len(per_query_metrics)) for key, value in sums.items()}


def candidate_stats_from_reranked_jsonl(
    *,
    reranked_jsonl: Path,
    candidates_field: str = "reranked_candidates",
    query_id_field: str = "query_id",
    candidate_pool_size_field: str = "candidate_pool_size",
) -> Dict[str, Any]:
    candidate_counts_by_query_id: Dict[str, int] = {}

    for row in _load_jsonl(Path(reranked_jsonl)):
        query_id = str(row.get(query_id_field) or "")
        candidates = row.get(candidates_field) or []
        candidate_pool_size = row.get(candidate_pool_size_field, None)

        try:
            candidate_count = int(candidate_pool_size) if candidate_pool_size is not None else int(len(candidates))
        except (TypeError, ValueError):
            candidate_count = int(len(candidates))

        candidate_counts_by_query_id[query_id] = candidate_count

    num_queries = int(len(candidate_counts_by_query_id))
    avg_candidates = (float(sum(candidate_counts_by_query_id.values())) / float(num_queries)) if num_queries else 0.0

    return {
        "num_queries": num_queries,
        "avg_candidates": avg_candidates,
        "candidate_counts_by_query_id": candidate_counts_by_query_id,
    }


def mrr_full_from_reranked_jsonl(
    *,
    reranked_jsonl: Path,
    candidates_field: str = "reranked_candidates",
    gold_field: str = "gold_passage_ids",
) -> Dict[str, float]:
    per_query_mrr: List[float] = []

    for row in _load_jsonl(Path(reranked_jsonl)):
        candidates = row.get(candidates_field) or []
        ranked_ids = [str(c.get("passage_id")) for c in candidates if c.get("passage_id") is not None]
        gold_ids = [str(x) for x in (row.get(gold_field) or [])]
        per_query_mrr.append(mrr_full(retrieved_passage_ids=ranked_ids, gold_passage_ids=gold_ids))

    num_queries = float(len(per_query_mrr))
    avg_mrr = float(sum(per_query_mrr) / num_queries) if num_queries else 0.0
    return {"mrr_full": avg_mrr, "num_queries": num_queries}


@dataclass(frozen=True)
class EvaluationSummary:
    global_metrics: Dict[str, float]
    metrics_by_doc_id: Dict[str, Dict[str, float]]
    num_queries: int


def evaluate_ranked_candidates(
    *,
    ranked_passage_ids_by_query: Sequence[Sequence[str]],
    gold_passage_ids_by_query: Sequence[Sequence[str]],
    doc_ids: Sequence[str],
    ks: Sequence[int],
) -> EvaluationSummary:
    if not (len(ranked_passage_ids_by_query) == len(gold_passage_ids_by_query) == len(doc_ids)):
        raise ValueError("ranked_passage_ids_by_query, gold_passage_ids_by_query, and doc_ids must align")

    per_query_rows: List[Dict[str, float]] = []
    per_doc_rows: Dict[str, List[Dict[str, float]]] = {}

    for ranked_ids, gold_ids, doc_id in zip(ranked_passage_ids_by_query, gold_passage_ids_by_query, doc_ids):
        metrics = compute_query_metrics(
            retrieved_passage_ids=[str(x) for x in ranked_ids],
            gold_passage_ids=[str(x) for x in gold_ids],
            ks=ks,
        )
        per_query_rows.append(metrics)
        per_doc_rows.setdefault(str(doc_id), []).append(metrics)

    global_metrics = aggregate_metrics(per_query_rows)
    global_metrics["num_queries"] = float(len(per_query_rows))

    metrics_by_doc_id: Dict[str, Dict[str, float]] = {}
    for doc_id, rows in per_doc_rows.items():
        agg = aggregate_metrics(rows)
        agg["num_queries"] = float(len(rows))
        metrics_by_doc_id[str(doc_id)] = agg

    return EvaluationSummary(
        global_metrics=global_metrics,
        metrics_by_doc_id=metrics_by_doc_id,
        num_queries=len(per_query_rows),
    )


def evaluate_reranked_jsonl(
    *,
    reranked_jsonl: Path,
    ks: Sequence[int] = (1, 5, 10, 20),
    candidates_field: str = "reranked_candidates",
    gold_field: str = "gold_passage_ids",
) -> EvaluationSummary:
    ranked_by_query: List[List[str]] = []
    gold_by_query: List[List[str]] = []
    doc_ids: List[str] = []

    for row in _load_jsonl(Path(reranked_jsonl)):
        candidates = row.get(candidates_field) or []
        ranked_ids = [str(c.get("passage_id")) for c in candidates if c.get("passage_id") is not None]
        ranked_by_query.append(ranked_ids)
        gold_by_query.append([str(x) for x in (row.get(gold_field) or [])])
        doc_ids.append(str(row.get("doc_id") or ""))

    return evaluate_ranked_candidates(
        ranked_passage_ids_by_query=ranked_by_query,
        gold_passage_ids_by_query=gold_by_query,
        doc_ids=doc_ids,
        ks=ks,
    )


def format_summary_table(
    summary: EvaluationSummary,
    *,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> str:
    ks = [int(k) for k in ks]
    keys: List[str] = []
    for k in ks:
        keys.extend(
            [
                f"recall_at_{k}",
                f"mrr_at_{k}",
                f"set_recall_at_{k}",
                f"exact_set_match_at_{k}",
            ]
        )

    def fmt(metrics: Mapping[str, float], key: str) -> str:
        value = metrics.get(key)
        return f"{float(value):.4f}" if value is not None else ""

    lines: List[str] = []
    lines.append("GLOBAL")
    for key in keys:
        lines.append(f"- {key}: {fmt(summary.global_metrics, key)}")

    lines.append("")
    lines.append("BY DOC_ID")
    for doc_id in sorted(summary.metrics_by_doc_id.keys(), key=lambda d: (not d.isdigit(), d)):
        row = summary.metrics_by_doc_id[doc_id]
        parts = [f"doc_id={doc_id}", f"n={int(row.get('num_queries', 0.0))}"]
        for k in ks:
            parts.append(f"R@{k}={fmt(row, f'recall_at_{k}')}")
            parts.append(f"MRR@{k}={fmt(row, f'mrr_at_{k}')}")
        lines.append("  " + "  ".join(parts))

    return "\n".join(lines)
