"""Reconstructed March Trainer-time retrieval evaluation.

This adapter exists only for ``legacy_train_sm.py`` replication attempts.  New
controlled training and final evaluation use ``retriever.evaluation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from retriever.data import (
    load_candidates_by_case,
    load_corpus,
    load_queries,
    load_split_doc_ids,
)
from retriever.query_views import QUERY_VIEW_STRUCTURED, select_query_text
from retriever.regimes import (
    REGIME_SAME_CASE_LEGACY,
    build_candidate_ids_by_query,
    build_split_passage_ids,
)


@dataclass(frozen=True)
class RetrievalEvalResult:
    metrics: Dict[str, float]
    query_count: int
    corpus_count: int


def _batch_iter(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


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
    vectors: List[torch.Tensor] = []
    for batch_texts in _batch_iter(list(passage_texts), batch_size):
        tokens = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len_passage,
            padding=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        encoded = retriever.encode_passages(
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        vectors.append(encoded.detach().cpu())
    return torch.cat(vectors, dim=0)


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
    vectors: List[torch.Tensor] = []
    for batch_texts in _batch_iter(list(query_texts), batch_size):
        tokens = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len_query,
            padding=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        encoded = retriever.encode_queries(
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        vectors.append(encoded.detach().cpu())
    return torch.cat(vectors, dim=0)


def _recall_at_k(hit_ranks: Sequence[int], k: int) -> float:
    if not hit_ranks:
        return 0.0
    return float(sum(1 for rank in hit_ranks if 1 <= rank <= k)) / float(
        len(hit_ranks)
    )


def _mrr(hit_ranks: Sequence[int]) -> float:
    if not hit_ranks:
        return 0.0
    return float(sum(1.0 / rank if rank > 0 else 0.0 for rank in hit_ranks)) / float(
        len(hit_ranks)
    )


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
    query_view: str = QUERY_VIEW_STRUCTURED,
    regime_name: str = REGIME_SAME_CASE_LEGACY,
) -> RetrievalEvalResult:
    """Run the historical Trainer metric path without canonical reinterpretation."""

    corpus_by_passage_id = load_corpus(processed_dir)
    candidates_by_case = load_candidates_by_case(processed_dir)
    queries = load_queries(processed_dir, split)
    split_doc_ids = load_split_doc_ids(processed_dir, split)
    passage_ids = build_split_passage_ids(
        corpus_by_passage_id=corpus_by_passage_id,
        split_doc_ids=split_doc_ids,
    )
    passage_texts = [corpus_by_passage_id[passage_id].text for passage_id in passage_ids]
    passage_index_by_id = {
        passage_id: index for index, passage_id in enumerate(passage_ids)
    }
    query_texts = [select_query_text(query, query_view=query_view) for query in queries]
    candidate_ids_by_query = build_candidate_ids_by_query(
        queries=queries,
        corpus_by_passage_id=corpus_by_passage_id,
        candidates_by_case=candidates_by_case,
        split_doc_ids=split_doc_ids,
        regime_name=regime_name,
    )

    device = next(retriever.parameters()).device
    retriever.eval()
    with torch.no_grad():
        passage_vectors = _encode_passages(
            retriever,
            tokenizer,
            passage_texts,
            batch_size=passage_batch_size,
            max_len_passage=max_len_passage,
            device=device,
        )
        query_vectors = _encode_queries(
            retriever,
            tokenizer,
            query_texts,
            batch_size=query_batch_size,
            max_len_query=max_len_query,
            device=device,
        )
    scores = query_vectors @ passage_vectors.T

    hit_ranks: List[int] = []
    candidate_sizes: List[int] = []
    retrieval_losses: List[float] = []
    for query_index, query in enumerate(queries):
        candidate_ids = [
            passage_id
            for passage_id in candidate_ids_by_query[query_index]
            if passage_id in passage_index_by_id
        ]
        candidate_indices = [passage_index_by_id[passage_id] for passage_id in candidate_ids]
        if not candidate_indices:
            continue
        candidate_sizes.append(len(candidate_indices))
        candidate_scores = scores[
            query_index,
            torch.tensor(candidate_indices, dtype=torch.long),
        ]
        positive_set = set(query.positive_passage_ids)
        positive_mask_values = [passage_id in positive_set for passage_id in candidate_ids]
        if any(positive_mask_values):
            positive_mask = torch.tensor(positive_mask_values, dtype=torch.bool)
            numerator = torch.logsumexp(candidate_scores[positive_mask], dim=0)
            denominator = torch.logsumexp(candidate_scores, dim=0)
            retrieval_losses.append(float((-(numerator - denominator)).item()))
        order = torch.argsort(candidate_scores, descending=True)
        ranked_passage_ids = [candidate_ids[int(position)] for position in order.tolist()]
        first_hit = next(
            (
                rank
                for rank, passage_id in enumerate(ranked_passage_ids, start=1)
                if passage_id in positive_set
            ),
            0,
        )
        hit_ranks.append(first_hit)

    metrics: Dict[str, float] = {"eval_num_queries": float(len(hit_ranks))}
    for k in ks:
        metrics[f"eval_recall_at_{int(k)}"] = _recall_at_k(hit_ranks, int(k))
    metrics["eval_mrr"] = _mrr(hit_ranks)
    metrics["eval_avg_candidates"] = float(
        sum(candidate_sizes) / max(1, len(candidate_sizes))
    )
    metrics["eval_retrieval_loss"] = float(
        sum(retrieval_losses) / max(1, len(retrieval_losses))
    )
    return RetrievalEvalResult(
        metrics=metrics,
        query_count=len(queries),
        corpus_count=len(passage_ids),
    )
