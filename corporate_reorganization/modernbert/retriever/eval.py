from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from .data import load_candidates_by_case, load_corpus, load_queries


@dataclass(frozen=True)
class RetrievalEvalResult:
    metrics: Dict[str, float]
    query_count: int
    corpus_count: int


def _batch_iter(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def _encode_passages(
    retriever,
    tokenizer: PreTrainedTokenizerBase,
    passage_texts: Sequence[str],
    *,
    batch_size: int,
    max_len_passage: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer.truncation_side = "right"
    all_vecs: List[torch.Tensor] = []
    for batch_texts in _batch_iter(list(passage_texts), batch_size):
        toks = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len_passage,
            padding=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        vecs = retriever.encode_passages(toks["input_ids"], toks["attention_mask"])
        all_vecs.append(vecs.detach().cpu())
    return torch.cat(all_vecs, dim=0)


def _encode_queries(
    retriever,
    tokenizer: PreTrainedTokenizerBase,
    query_texts: Sequence[str],
    *,
    batch_size: int,
    max_len_query: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer.truncation_side = "left"
    all_vecs: List[torch.Tensor] = []
    for batch_texts in _batch_iter(list(query_texts), batch_size):
        toks = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len_query,
            padding=True,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        vecs = retriever.encode_queries(toks["input_ids"], toks["attention_mask"])
        all_vecs.append(vecs.detach().cpu())
    return torch.cat(all_vecs, dim=0)


def _recall_at_k(hit_ranks: Sequence[int], k: int) -> float:
    if not hit_ranks:
        return 0.0
    return float(sum(1 for r in hit_ranks if 1 <= r <= k)) / float(len(hit_ranks))


def _mrr(hit_ranks: Sequence[int]) -> float:
    if not hit_ranks:
        return 0.0
    return float(sum(1.0 / r if r > 0 else 0.0 for r in hit_ranks)) / float(len(hit_ranks))


def evaluate_retrieval(
    retriever,
    tokenizer: PreTrainedTokenizerBase,
    *,
    processed_dir: Path,
    split: str,
    max_len_query: int,
    max_len_passage: int,
    query_batch_size: int,
    passage_batch_size: int,
    ks: Sequence[int] = (1, 5, 10, 20, 50),
) -> RetrievalEvalResult:
    corpus_by_passage_id = load_corpus(processed_dir)
    candidates_by_case = load_candidates_by_case(processed_dir)
    queries = load_queries(processed_dir, split)

    passage_ids = list(corpus_by_passage_id.keys())
    passage_texts = [corpus_by_passage_id[pid].text for pid in passage_ids]
    passage_index_by_id = {pid: i for i, pid in enumerate(passage_ids)}

    query_ids = [q.query_id for q in queries]
    query_texts = [q.query_text for q in queries]

    positive_ids_by_doc_id: Dict[str, set[str]] = {}
    for q in queries:
        positive_ids_by_doc_id.setdefault(q.doc_id, set()).update(q.positive_passage_ids)

    device = next(retriever.parameters()).device
    retriever.eval()
    with torch.no_grad():
        passage_vecs = _encode_passages(
            retriever,
            tokenizer,
            passage_texts,
            batch_size=passage_batch_size,
            max_len_passage=max_len_passage,
            device=device,
        )
        query_vecs = _encode_queries(
            retriever,
            tokenizer,
            query_texts,
            batch_size=query_batch_size,
            max_len_query=max_len_query,
            device=device,
        )

    scores = query_vecs @ passage_vecs.T

    hit_ranks: List[int] = []
    candidate_sizes: List[int] = []
    retrieval_losses: List[float] = []
    for qi, query in enumerate(queries):
        doc_candidate_ids = candidates_by_case.get(query.doc_id, [])
        excluded_ids = positive_ids_by_doc_id.get(query.doc_id, set()) - set(query.positive_passage_ids)
        candidate_ids = [
            pid for pid in doc_candidate_ids if (pid in passage_index_by_id) and (pid not in excluded_ids)
        ]
        candidate_indices = [passage_index_by_id[pid] for pid in candidate_ids]
        if not candidate_indices:
            continue
        candidate_sizes.append(len(candidate_indices))

        cand_scores = scores[qi, torch.tensor(candidate_indices, dtype=torch.long)]
        positive_set = set(query.positive_passage_ids)
        pos_mask_list = [pid in positive_set for pid in candidate_ids]
        if any(pos_mask_list):
            pos_mask = torch.tensor(pos_mask_list, dtype=torch.bool)
            numerator = torch.logsumexp(cand_scores[pos_mask], dim=0)
            denominator = torch.logsumexp(cand_scores, dim=0)
            retrieval_losses.append(float((-(numerator - denominator)).item()))
        sorted_local = torch.argsort(cand_scores, descending=True)
        ranked_passage_ids = [candidate_ids[int(j)] for j in sorted_local.tolist()]

        best_rank = 0
        for rank_idx, pid in enumerate(ranked_passage_ids, start=1):
            if pid in positive_set:
                best_rank = rank_idx
                break
        hit_ranks.append(best_rank)

    metrics: Dict[str, float] = {"eval_num_queries": float(len(hit_ranks))}
    for k in ks:
        metrics[f"eval_recall_at_{int(k)}"] = _recall_at_k(hit_ranks, int(k))
    metrics["eval_mrr"] = _mrr(hit_ranks)
    metrics["eval_avg_candidates"] = float(sum(candidate_sizes) / max(1, len(candidate_sizes)))
    metrics["eval_retrieval_loss"] = float(sum(retrieval_losses) / max(1, len(retrieval_losses)))

    return RetrievalEvalResult(metrics=metrics, query_count=len(query_ids), corpus_count=len(passage_ids))
