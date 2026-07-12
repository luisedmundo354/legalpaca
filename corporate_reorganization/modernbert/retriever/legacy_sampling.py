from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

from .data import CorpusPassage, QueryExample
from .query_views import QUERY_VIEW_STRUCTURED, select_query_text


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


class MultiPositiveRetrievalTrainDataset:
    """Reconstructed March sampler, retained only for legacy replication attempts."""

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
        query_view: str = QUERY_VIEW_STRUCTURED,
    ):
        self.queries = list(queries)
        self.candidates_by_case = candidates_by_case
        self.distractor_passage_ids = list(distractor_passage_ids)
        self.base_seed = int(base_seed)
        self.epoch = 0

        positive_ids_by_doc_id: Dict[str, set[str]] = {}
        for query in self.queries:
            positive_ids_by_doc_id.setdefault(query.doc_id, set()).update(query.positive_passage_ids)

        self.positive_passage_ids_by_doc_id = positive_ids_by_doc_id
        self.base_negative_pool_by_doc_id: Dict[str, List[str]] = {
            doc_id: [pid for pid in passage_ids if pid not in positive_ids_by_doc_id.get(doc_id, set())]
            for doc_id, passage_ids in self.candidates_by_case.items()
        }

        self.max_pos_per_query = int(max_pos_per_query)
        self.num_same_case_negatives = int(num_same_case_negatives)
        self.num_distractor_negatives = int(num_distractor_negatives)
        self.query_view = str(query_view)
        self.use_all_same_case_candidates = self.num_same_case_negatives < 0
        self.max_candidates_per_case = max((len(v) for v in candidates_by_case.values()), default=0)

        if self.max_pos_per_query < 1:
            raise ValueError("max_pos_per_query must be >= 1")
        if self.num_distractor_negatives < 0:
            raise ValueError("num_distractor_negatives must be >= 0")
        if self.max_candidates_per_case < 1:
            raise ValueError("candidates_by_case must contain at least 1 candidate")

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

        if self.use_all_same_case_candidates:
            doc_candidate_ids = list(self.candidates_by_case.get(query.doc_id, []))
            excluded_ids = self.positive_passage_ids_by_doc_id.get(query.doc_id, set()) - set(positive_passage_ids)
            same_case_candidate_ids = [pid for pid in doc_candidate_ids if pid not in excluded_ids]
            if not same_case_candidate_ids:
                raise ValueError(f"No candidates for case doc_id={query.doc_id}")

            if len(same_case_candidate_ids) > self.max_candidates_per_case:
                raise ValueError(
                    f"doc_id={query.doc_id} has {len(same_case_candidate_ids)} candidates which exceeds"
                    f" max_candidates_per_case={self.max_candidates_per_case}"
                )

            num_pad = self.max_candidates_per_case - len(same_case_candidate_ids)
            if num_pad > 0:
                same_case_candidate_ids = [
                    *same_case_candidate_ids,
                    *self._sample_from_pool(rng, same_case_candidate_ids, num_pad),
                ]

            other_case_pool = [
                pid
                for pid in self.distractor_passage_ids
                if (not str(pid).startswith(f"{query.doc_id}::"))
            ]
            other_case_negs = self._sample_from_pool(rng, other_case_pool, self.num_distractor_negatives)

            candidate_passage_ids: List[str] = [*same_case_candidate_ids, *other_case_negs]
        else:
            num_pos_in_candidates = min(len(positive_passage_ids), self.max_pos_per_query)
            pos_for_candidates = rng.sample(positive_passage_ids, k=num_pos_in_candidates)

            same_case_pool = list(self.base_negative_pool_by_doc_id.get(query.doc_id, []))
            same_case_neg_count = self.num_same_case_negatives + (self.max_pos_per_query - num_pos_in_candidates)
            same_case_negs = self._sample_from_pool(rng, same_case_pool, same_case_neg_count)
            distractor_negs = self._sample_from_pool(
                rng,
                [pid for pid in self.distractor_passage_ids if not str(pid).startswith(f"{query.doc_id}::")],
                self.num_distractor_negatives,
            )

            candidates_per_query = self.max_pos_per_query + self.num_same_case_negatives + self.num_distractor_negatives
            candidate_passage_ids = [*pos_for_candidates, *same_case_negs, *distractor_negs]

            if len(candidate_passage_ids) < candidates_per_query:
                pad_pool = same_case_pool or self.distractor_passage_ids or positive_passage_ids
                candidate_passage_ids.extend(
                    self._sample_from_pool(rng, pad_pool, candidates_per_query - len(candidate_passage_ids))
                )
            candidate_passage_ids = candidate_passage_ids[:candidates_per_query]

        return {
            "query_id": query.query_id,
            "doc_id": query.doc_id,
            "query_text": select_query_text(query, query_view=self.query_view),
            "positive_passage_ids": positive_passage_ids,
            "candidate_passage_ids": candidate_passage_ids,
        }
