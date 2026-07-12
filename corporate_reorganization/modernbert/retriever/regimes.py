from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .data import CorpusPassage, QueryExample


REGIME_SAME_CASE_LEGACY: str = "same_case_legacy"
REGIME_SAME_CASE_FULL: str = "same_case_full"
# Canonical controlled-study names.  ``global_split`` is retained below only as
# a March compatibility alias; it is intentionally not a canonical regime.
REGIME_FOLD_GLOBAL: str = "fold_global"
REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED: str = "fold_global_context_excluded"
REGIME_GLOBAL_SPLIT: str = "global_split"

CANONICAL_CANDIDATE_REGIMES: Tuple[str, ...] = (
    REGIME_SAME_CASE_LEGACY,
    REGIME_SAME_CASE_FULL,
    REGIME_FOLD_GLOBAL,
    REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
)
SUPPORTED_CANDIDATE_REGIMES = CANONICAL_CANDIDATE_REGIMES


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
            name=REGIME_FOLD_GLOBAL,
            description="All passages from cases in the single evaluated role fold.",
        ),
        CandidateRegime(
            name=REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED,
            description=(
                "The fold-global ranking with visible non-gold passages filtered "
                "without rescoring."
            ),
        ),
    ]


def normalize_candidate_regime(regime_name: str) -> str:
    value = str(regime_name).strip().lower()
    if value not in SUPPORTED_CANDIDATE_REGIMES:
        raise ValueError(
            f"Unsupported candidate regime={regime_name!r}; expected one of {SUPPORTED_CANDIDATE_REGIMES}"
        )
    return value


def normalize_legacy_candidate_regime(regime_name: str) -> str:
    """Normalize the sole historical alias before entering the canonical core."""

    value = str(regime_name).strip().lower()
    if value == REGIME_GLOBAL_SPLIT:
        return REGIME_FOLD_GLOBAL
    return normalize_candidate_regime(value)


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
    """Build March-compatible split pools.

    New controlled evaluation must use :func:`build_role_candidate_ids_by_query`
    with explicit role case IDs.  This wrapper alone accepts ``global_split``.
    """

    regime = normalize_legacy_candidate_regime(regime_name)
    split_passage_ids = build_split_passage_ids(
        corpus_by_passage_id=corpus_by_passage_id,
        split_doc_ids=split_doc_ids,
    )
    allowed_passage_ids = set(split_passage_ids)

    if regime == REGIME_FOLD_GLOBAL:
        return [list(split_passage_ids) for _ in queries]

    if regime == REGIME_FOLD_GLOBAL_CONTEXT_EXCLUDED:
        candidate_ids_by_query: List[List[str]] = []
        for query in queries:
            gold_ids = set(query.positive_passage_ids)
            excluded_ids = set(query.visible_passage_ids) - gold_ids
            candidate_ids_by_query.append(
                [passage_id for passage_id in split_passage_ids if passage_id not in excluded_ids]
            )
        return candidate_ids_by_query

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


def build_role_candidate_ids_by_query(
    *,
    queries: Sequence["QueryExample"],
    corpus_by_passage_id: Mapping[str, "CorpusPassage"],
    evaluated_case_ids: Sequence[str],
    regime_name: str,
) -> List[List[str]]:
    """Build one of the four exact query-specific controlled candidate pools.

    The returned passage IDs and the supplied queries are both expected to be
    in canonical order by the caller.  This function validates the complete
    evaluated-role inventory and never accepts the historical ``global_split``
    alias.
    """

    regime = normalize_candidate_regime(regime_name)
    if not isinstance(evaluated_case_ids, (list, tuple)):
        raise TypeError("evaluated_case_ids must be a list or tuple")
    case_ids = tuple(evaluated_case_ids)
    if not case_ids or any(
        type(case_id) is not str or not case_id or case_id.strip() != case_id
        for case_id in case_ids
    ):
        raise ValueError("evaluated_case_ids must contain exact non-empty strings")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluated_case_ids contains duplicates")
    case_id_set = set(case_ids)

    passage_ids_by_case: Dict[str, List[str]] = {case_id: [] for case_id in case_ids}
    for passage_id in sorted(corpus_by_passage_id):
        passage = corpus_by_passage_id[passage_id]
        if passage.passage_id != passage_id:
            raise ValueError(
                f"Corpus key {passage_id!r} disagrees with passage identity "
                f"{passage.passage_id!r}"
            )
        if passage.doc_id in case_id_set:
            passage_ids_by_case[passage.doc_id].append(passage_id)
    empty_cases = sorted(case_id for case_id, passage_ids in passage_ids_by_case.items() if not passage_ids)
    if empty_cases:
        raise ValueError(f"Evaluated cases have no corpus passages: {empty_cases}")

    role_passage_ids = [
        passage_id
        for passage_id in sorted(corpus_by_passage_id)
        if corpus_by_passage_id[passage_id].doc_id in case_id_set
    ]
    if not role_passage_ids:
        raise ValueError("The evaluated role has no corpus passages")

    query_ids: Set[str] = set()
    ordered_query_ids: List[str] = []
    query_case_ids: Set[str] = set()
    for query in queries:
        if query.query_id in query_ids:
            raise ValueError(f"Duplicate evaluated query_id={query.query_id!r}")
        query_ids.add(query.query_id)
        ordered_query_ids.append(query.query_id)
        if query.doc_id not in case_id_set:
            raise ValueError(f"Query {query.query_id!r} is outside the evaluated role")
        query_case_ids.add(query.doc_id)
    if ordered_query_ids != sorted(ordered_query_ids):
        raise ValueError("Evaluated queries must be lexicographically sorted by query_id")
    if query_case_ids != case_id_set:
        raise ValueError(
            "Evaluated queries do not cover exactly the evaluated cases: "
            f"actual={sorted(query_case_ids)}, expected={sorted(case_id_set)}"
        )

    positive_ids_by_doc_id: Dict[str, Set[str]] = {}
    for query in queries:
        positive_ids_by_doc_id.setdefault(query.doc_id, set()).update(
            query.positive_passage_ids
        )

    result: List[List[str]] = []
    for query in queries:
        gold_ids = set(query.positive_passage_ids)
        if not gold_ids or len(gold_ids) != len(query.positive_passage_ids):
            raise ValueError(
                f"Query {query.query_id!r} must have unique non-empty gold passage IDs"
            )
        case_passage_ids = passage_ids_by_case[query.doc_id]
        if not gold_ids.issubset(case_passage_ids):
            raise ValueError(f"Query {query.query_id!r} has a gold outside its case corpus")

        visible_ids = set(query.visible_passage_ids)
        if len(visible_ids) != len(query.visible_passage_ids):
            raise ValueError(f"Query {query.query_id!r} has duplicate visible passage IDs")
        if not visible_ids.issubset(case_passage_ids):
            raise ValueError(f"Query {query.query_id!r} has visible context outside its case")

        if regime == REGIME_SAME_CASE_LEGACY:
            excluded_ids = positive_ids_by_doc_id[query.doc_id] - gold_ids
            candidates = [pid for pid in case_passage_ids if pid not in excluded_ids]
        elif regime == REGIME_SAME_CASE_FULL:
            candidates = list(case_passage_ids)
        elif regime == REGIME_FOLD_GLOBAL:
            candidates = list(role_passage_ids)
        else:
            excluded_ids = visible_ids - gold_ids
            candidates = [pid for pid in role_passage_ids if pid not in excluded_ids]

        if len(candidates) != len(set(candidates)):
            raise RuntimeError(f"Query {query.query_id!r} candidate pool contains duplicates")
        if not candidates:
            raise RuntimeError(f"Query {query.query_id!r} candidate pool is empty")
        if not gold_ids.issubset(candidates):
            raise RuntimeError(f"Query {query.query_id!r} candidate pool dropped a gold")
        result.append(candidates)
    return result
