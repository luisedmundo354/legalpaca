from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .cohere_reranker import CohereReranker


@dataclass(frozen=True)
class RankedCandidate:
    passage_id: str
    text: str
    label: str


@dataclass(frozen=True)
class QueryRankingRecord:
    query_id: str
    doc_id: str
    gold_passage_ids: List[str]
    candidates: List[RankedCandidate]


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_query_texts_by_id(processed_dir: Path, *, split: str) -> Dict[str, str]:
    query_path = processed_dir / "queries" / f"{split}.jsonl"
    query_text_by_id: Dict[str, str] = {}
    for row in _load_jsonl(query_path):
        query_text_by_id[str(row["query_id"])] = str(row["query_text"])
    return query_text_by_id


def _load_candidate_pool_from_rankings(
    rankings_jsonl: Path,
    *,
    system_name: str,
) -> List[QueryRankingRecord]:
    records: List[QueryRankingRecord] = []
    for row in _load_jsonl(rankings_jsonl):
        if str(row.get("system")) != str(system_name):
            continue
        candidates = [
            RankedCandidate(
                passage_id=str(c["passage_id"]),
                text=str(c.get("text") or ""),
                label=str(c.get("label") or ""),
            )
            for c in (row.get("ranked_candidates") or [])
        ]
        records.append(
            QueryRankingRecord(
                query_id=str(row["query_id"]),
                doc_id=str(row.get("doc_id") or ""),
                gold_passage_ids=[str(x) for x in (row.get("gold_passage_ids") or [])],
                candidates=candidates,
            )
        )
    return records


def rerank_rankings_file(
    *,
    processed_dir: Path,
    rankings_jsonl: Path,
    split: str,
    input_system: str = "fine_tuned",
    output_path: Path,
    reranker: CohereReranker,
    max_docs_per_request: int = 100,
) -> None:
    query_text_by_id = _load_query_texts_by_id(processed_dir, split=split)
    query_records = _load_candidate_pool_from_rankings(rankings_jsonl, system_name=input_system)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in query_records:
            query_text = query_text_by_id.get(record.query_id)
            if query_text is None:
                raise KeyError(f"Missing query_text for query_id={record.query_id} in {processed_dir}")

            documents = [c.text for c in record.candidates]
            if not documents:
                continue

            scores_by_index: Dict[int, float] = {}
            chunk_size = int(max_docs_per_request)
            for start in range(0, len(documents), chunk_size):
                chunk = documents[start : start + chunk_size]
                results = reranker.rerank(query=query_text, documents=chunk, top_n=len(chunk))
                for item in results:
                    scores_by_index[start + int(item.index)] = float(item.relevance_score)

            ranked_indices = sorted(scores_by_index.keys(), key=lambda i: scores_by_index[i], reverse=True)
            reranked_candidates = []
            for rank_idx, cand_idx in enumerate(ranked_indices, start=1):
                cand = record.candidates[int(cand_idx)]
                reranked_candidates.append(
                    {
                        "rank": int(rank_idx),
                        "passage_id": cand.passage_id,
                        "score": float(scores_by_index[int(cand_idx)]),
                        "label": cand.label,
                        "text": cand.text,
                    }
                )

            out_f.write(
                json.dumps(
                    {
                        "system": "cohere_rerank",
                        "model": reranker.model,
                        "query_id": record.query_id,
                        "doc_id": record.doc_id,
                        "gold_passage_ids": list(record.gold_passage_ids),
                        "candidate_pool_size": int(len(record.candidates)),
                        "reranked_candidates": reranked_candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_topk_for_query(
    *,
    reranked_jsonl: Path,
    query_id: str,
    ks: Sequence[int] = (1, 5, 10, 20),
    max_chars: int = 240,
) -> None:
    target = str(query_id)
    row: Optional[Dict] = None
    for rec in _load_jsonl(reranked_jsonl):
        if str(rec.get("query_id")) == target:
            row = rec
            break
    if row is None:
        raise KeyError(f"query_id not found: {query_id}")

    candidates = list(row.get("reranked_candidates") or [])

    def trunc(text: str) -> str:
        t = str(text).replace("\\n", " ").strip()
        return t if len(t) <= max_chars else t[: max_chars - 3] + "..."

    for k in ks:
        topk = candidates[: int(k)]
        print(f"\nTop {int(k)} for query_id={query_id} ({row.get('model', '')})")
        for item in topk:
            print(
                f"  #{int(item['rank']):>3} score={float(item['score']):.4f} "
                f"passage_id={item['passage_id']} | {trunc(item.get('text', ''))}"
            )
