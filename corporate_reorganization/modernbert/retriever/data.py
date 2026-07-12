from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class CorpusPassage:
    passage_id: str
    doc_id: str
    label: str
    text: str


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    doc_id: str
    motion_root_id: str
    mask_parent_id: str
    query_text: str
    positive_passage_ids: List[str]
    positive_labels: List[str]
    visible_passage_ids: List[str] = field(default_factory=list)
    flat_query_text_plain: str = ""
    flat_query_text_masked: str = ""


def load_corpus(processed_dir: Path) -> Dict[str, CorpusPassage]:
    corpus_path = processed_dir / "corpus.jsonl"
    corpus_by_passage_id: Dict[str, CorpusPassage] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            passage_id = str(rec["passage_id"])
            corpus_by_passage_id[passage_id] = CorpusPassage(
                passage_id=passage_id,
                doc_id=str(rec["doc_id"]),
                label=str(rec["label"]),
                text=str(rec["text"]),
            )
    return corpus_by_passage_id


def load_queries(processed_dir: Path, split: str) -> List[QueryExample]:
    queries_path = processed_dir / "queries" / f"{split}.jsonl"
    queries: List[QueryExample] = []
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            queries.append(
                QueryExample(
                    query_id=str(rec["query_id"]),
                    doc_id=str(rec["doc_id"]),
                    motion_root_id=str(rec.get("motion_root_id") or ""),
                    mask_parent_id=str(rec.get("mask_parent_id") or ""),
                    query_text=str(rec["query_text"]),
                    positive_passage_ids=[str(x) for x in rec.get("positive_passage_ids") or []],
                    positive_labels=[str(x) for x in rec.get("positive_labels") or []],
                    visible_passage_ids=[str(x) for x in rec.get("visible_passage_ids") or []],
                    flat_query_text_plain=str(rec.get("flat_query_text_plain") or ""),
                    flat_query_text_masked=str(rec.get("flat_query_text_masked") or ""),
                )
            )
    return queries


def load_split_doc_ids(processed_dir: Path, split: str) -> List[str]:
    split_path = processed_dir / "splits" / f"{split}_cases.txt"
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_candidates_by_case(processed_dir: Path) -> Dict[str, List[str]]:
    pools_path = processed_dir / "pools" / "candidates_by_case.json"
    raw = json.loads(pools_path.read_text(encoding="utf-8"))
    return {str(doc_id): [str(x) for x in passage_ids] for doc_id, passage_ids in raw.items()}
