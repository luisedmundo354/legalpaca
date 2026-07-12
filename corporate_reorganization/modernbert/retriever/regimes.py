from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .data import CorpusPassage, QueryExample


REGIME_SAME_CASE_LEGACY: str = "same_case_legacy"
REGIME_SAME_CASE_FULL: str = "same_case_full"
REGIME_GLOBAL_SPLIT: str = "global_split"

SUPPORTED_CANDIDATE_REGIMES: Tuple[str, ...] = (
    REGIME_SAME_CASE_LEGACY,
    REGIME_SAME_CASE_FULL,
    REGIME_GLOBAL_SPLIT,
)


@dataclass(frozen=True)
class CandidateRegime:
    name: str
    description: str


def available_candidate_regimes() -> List[CandidateRegime]:
    return [
        CandidateRegime(
            name=REGIME_SAME_CASE_LEGACY,
            description="Same-case candidates with other-query positives removed.",
        ),
        CandidateRegime(
            name=REGIME_SAME_CASE_FULL,
            description="All candidates from the same case.",
        ),
        CandidateRegime(
            name=REGIME_GLOBAL_SPLIT,
            description="All passages from docs in the current evaluation split.",
        ),
    ]


def normalize_candidate_regime(regime_name: str) -> str:
    value = str(regime_name).strip().lower()
    if value not in SUPPORTED_CANDIDATE_REGIMES:
        raise ValueError(
            f"Unsupported candidate regime={regime_name!r}; expected one of {SUPPORTED_CANDIDATE_REGIMES}"
        )
    return value


def build_split_passage_ids(
    *,
    corpus_by_passage_id: Dict[str, "CorpusPassage"],
    split_doc_ids: Iterable[str],
) -> List[str]:
    allowed_doc_ids: Set[str] = {str(doc_id) for doc_id in split_doc_ids}
    return [
        passage_id
        for passage_id, passage in corpus_by_passage_id.items()
        if str(passage.doc_id) in allowed_doc_ids
    ]


def build_candidate_ids_by_query(
    *,
    queries: Sequence["QueryExample"],
    corpus_by_passage_id: Dict[str, "CorpusPassage"],
    candidates_by_case: Dict[str, List[str]],
    split_doc_ids: Sequence[str],
    regime_name: str,
) -> List[List[str]]:
    regime = normalize_candidate_regime(regime_name)
    split_passage_ids = build_split_passage_ids(
        corpus_by_passage_id=corpus_by_passage_id,
        split_doc_ids=split_doc_ids,
    )
    allowed_passage_ids = set(split_passage_ids)

    if regime == REGIME_GLOBAL_SPLIT:
        return [list(split_passage_ids) for _ in queries]

    if regime == REGIME_SAME_CASE_FULL:
        return [
            [pid for pid in candidates_by_case.get(query.doc_id, []) if pid in allowed_passage_ids]
            for query in queries
        ]

    positive_ids_by_doc_id: Dict[str, Set[str]] = {}
    for query in queries:
        positive_ids_by_doc_id.setdefault(query.doc_id, set()).update(query.positive_passage_ids)

    candidate_ids_by_query: List[List[str]] = []
    for query in queries:
        excluded_ids = positive_ids_by_doc_id.get(query.doc_id, set()) - set(query.positive_passage_ids)
        doc_candidates = candidates_by_case.get(query.doc_id, [])
        candidate_ids_by_query.append(
            [
                passage_id
                for passage_id in doc_candidates
                if (passage_id in allowed_passage_ids) and (passage_id not in excluded_ids)
            ]
        )
    return candidate_ids_by_query
