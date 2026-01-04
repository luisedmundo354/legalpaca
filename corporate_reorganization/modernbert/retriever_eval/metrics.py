from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class QueryInfo:
    query_id: str
    doc_id: str
    query_text: str
    gold_passage_ids: List[str]
    gold_labels: List[str]


def _first_hit_rank(retrieved_passage_ids: Sequence[str], gold_set: Set[str]) -> int:
    for rank_idx, passage_id in enumerate(retrieved_passage_ids, start=1):
        if passage_id in gold_set:
            return rank_idx
    return 0


def compute_query_metrics(
    *,
    retrieved_passage_ids: Sequence[str],
    gold_passage_ids: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    gold_set = set(gold_passage_ids)
    if not gold_set:
        return {f"recall_at_{k}": 0.0 for k in ks}

    hit_rank = _first_hit_rank(retrieved_passage_ids, gold_set)
    num_gold = float(len(gold_set))

    metrics: Dict[str, float] = {}
    for k in ks:
        topk = retrieved_passage_ids[: int(k)]
        topk_set = set(topk)
        any_hit = hit_rank != 0 and hit_rank <= int(k)
        metrics[f"recall_at_{int(k)}"] = 1.0 if any_hit else 0.0
        metrics[f"mrr_at_{int(k)}"] = (1.0 / float(hit_rank)) if any_hit else 0.0
        metrics[f"set_recall_at_{int(k)}"] = float(len(topk_set & gold_set)) / num_gold
        metrics[f"exact_set_match_at_{int(k)}"] = 1.0 if gold_set.issubset(topk_set) else 0.0

    metrics["num_gold"] = num_gold
    metrics["first_hit_rank"] = float(hit_rank)
    return metrics


def aggregate_metrics(per_query: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    sums: Dict[str, float] = {}
    for row in per_query:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
    return {k: v / float(len(per_query)) for k, v in sums.items()}


def bucket_missing_type(query_id: str) -> str:
    if "MISSING=CONCLUSION" in query_id:
        return "missing_conclusion"
    return "missing_premise_group"


def bucket_gold_label_composition(gold_labels: Sequence[str]) -> str:
    label_set = {str(x).strip() for x in gold_labels if str(x).strip()}
    if not label_set:
        return "unknown"
    if len(label_set) == 1:
        only = next(iter(label_set)).lower()
        if only == "rule":
            return "only_rule"
        if only == "analysis":
            return "only_analysis"
        if only == "conclusion":
            return "only_conclusion"
        return f"only_{only}"
    return "mixed"


def bucket_num_positives(num_gold: int) -> str:
    if num_gold <= 1:
        return "1"
    if num_gold <= 3:
        return "2_3"
    return "4_plus"


def bucket_has_implicit(query_text: str) -> str:
    return "has_implicit" if "[IMPLICIT]" in str(query_text) else "no_implicit"


def compute_bucketed_metrics(
    *,
    queries: Sequence[QueryInfo],
    per_query_metrics: Sequence[Dict[str, float]],
    ks: Sequence[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if len(queries) != len(per_query_metrics):
        raise ValueError("queries and per_query_metrics must be aligned")

    def init_group() -> Dict[str, List[Dict[str, float]]]:
        return {}

    groups: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "by_missing_type": init_group(),
        "by_gold_label_composition": init_group(),
        "by_num_positives": init_group(),
        "by_has_implicit": init_group(),
    }

    for query, q_metrics in zip(queries, per_query_metrics):
        groups["by_missing_type"].setdefault(bucket_missing_type(query.query_id), []).append(q_metrics)
        groups["by_gold_label_composition"].setdefault(
            bucket_gold_label_composition(query.gold_labels), []
        ).append(q_metrics)
        groups["by_num_positives"].setdefault(
            bucket_num_positives(int(len(set(query.gold_passage_ids)))), []
        ).append(q_metrics)
        groups["by_has_implicit"].setdefault(bucket_has_implicit(query.query_text), []).append(q_metrics)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_name, buckets in groups.items():
        out[group_name] = {}
        for bucket_name, rows in buckets.items():
            agg = aggregate_metrics(rows)
            agg["num_queries"] = float(len(rows))
            out[group_name][bucket_name] = agg
    return out

