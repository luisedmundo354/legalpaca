from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset


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
                )
            )
    return queries


def load_candidates_by_case(processed_dir: Path) -> Dict[str, List[str]]:
    pools_path = processed_dir / "pools" / "candidates_by_case.json"
    raw = json.loads(pools_path.read_text(encoding="utf-8"))
    return {str(doc_id): [str(x) for x in passage_ids] for doc_id, passage_ids in raw.items()}


def select_distractor_passage_ids(
    corpus_by_passage_id: Dict[str, CorpusPassage],
    *,
    distractor_labels: Sequence[str],
) -> List[str]:
    distractor_labels = set(distractor_labels)
    return [
        passage_id
        for passage_id, passage in corpus_by_passage_id.items()
        if passage.label in distractor_labels
    ]


class MultiPositiveRetrievalTrainDataset(Dataset):
    def __init__(
        self,
        queries: Sequence[QueryExample],
        candidates_by_case: Dict[str, List[str]],
        distractor_passage_ids: Sequence[str],
        *,
        base_seed: int,
        max_pos_per_query: int,
        num_same_case_negatives: int,
        num_distractor_negatives: int,
    ):
        self.queries = list(queries)
        self.candidates_by_case = candidates_by_case
        self.distractor_passage_ids = list(distractor_passage_ids)
        self.base_seed = int(base_seed)
        self.epoch = 0

        self.max_pos_per_query = int(max_pos_per_query)
        self.num_same_case_negatives = int(num_same_case_negatives)
        self.num_distractor_negatives = int(num_distractor_negatives)

        if self.max_pos_per_query < 1:
            raise ValueError("max_pos_per_query must be >= 1")
        if self.num_same_case_negatives < 0 or self.num_distractor_negatives < 0:
            raise ValueError("negative counts must be >= 0")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.queries)

    def _rng_for_index(self, idx: int) -> random.Random:
        seed = (self.base_seed * 1000003) + (self.epoch * 1009) + int(idx)
        return random.Random(seed)

    @staticmethod
    def _sample_from_pool(rng: random.Random, pool: Sequence[str], k: int) -> List[str]:
        if k <= 0:
            return []
        if not pool:
            return []
        if len(pool) >= k:
            return rng.sample(list(pool), k=k)
        return [rng.choice(list(pool)) for _ in range(k)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        query = self.queries[int(idx)]
        rng = self._rng_for_index(int(idx))

        positive_passage_ids = list(query.positive_passage_ids)
        if not positive_passage_ids:
            raise ValueError(f"Query has no positives: {query.query_id}")

        num_pos_in_candidates = min(len(positive_passage_ids), self.max_pos_per_query)
        pos_for_candidates = rng.sample(positive_passage_ids, k=num_pos_in_candidates)

        same_case_pool = [
            pid
            for pid in self.candidates_by_case.get(query.doc_id, [])
            if pid not in set(positive_passage_ids)
        ]
        same_case_neg_count = self.num_same_case_negatives + (self.max_pos_per_query - num_pos_in_candidates)
        same_case_negs = self._sample_from_pool(rng, same_case_pool, same_case_neg_count)
        distractor_negs = self._sample_from_pool(rng, self.distractor_passage_ids, self.num_distractor_negatives)

        candidates_per_query = self.max_pos_per_query + self.num_same_case_negatives + self.num_distractor_negatives
        candidate_passage_ids: List[str] = [*pos_for_candidates, *same_case_negs, *distractor_negs]

        if len(candidate_passage_ids) < candidates_per_query:
            pad_pool = same_case_pool or self.distractor_passage_ids or positive_passage_ids
            candidate_passage_ids.extend(
                self._sample_from_pool(rng, pad_pool, candidates_per_query - len(candidate_passage_ids))
            )
        candidate_passage_ids = candidate_passage_ids[:candidates_per_query]

        return {
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "query_text": query.query_text,
            "positive_passage_ids": positive_passage_ids,
            "candidate_passage_ids": candidate_passage_ids,
        }

