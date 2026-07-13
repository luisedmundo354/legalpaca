from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from .data import CorpusPassage


TRAIN_CASE_IDS: Final[tuple[str, ...]] = (
    "36",
    "38",
    "40",
    "41",
    "42",
    "48",
    "49",
    "57",
    "58",
    "59",
    "63",
    "66",
    "67",
    "68",
    "69",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "80",
    "83",
    "85",
    "86",
    "87",
    "91",
    "92",
    "94",
    "97",
)
VALIDATION_CASE_IDS: Final[tuple[str, ...]] = ("45", "47", "60", "62")
TEST_CASE_IDS: Final[tuple[str, ...]] = ("37", "46", "65", "96")
CASE_IDS_BY_SPLIT: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "train": TRAIN_CASE_IDS,
        "validation": VALIDATION_CASE_IDS,
        "test": TEST_CASE_IDS,
    }
)

EXPECTED_QUERY_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {"train": 418, "validation": 32, "test": 40}
)
EXPECTED_PASSAGE_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {"train": 4307, "validation": 398, "test": 581}
)
EXPECTED_TOTAL_QUERY_COUNT: Final[int] = 490
EXPECTED_TOTAL_PASSAGE_COUNT: Final[int] = 5286

MEMBERSHIP_SCHEMA_VERSION: Final[int] = 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


MEMBERSHIP_SHA256: Final[str] = hashlib.sha256(
    _canonical_json_bytes(
        {
            "case_ids_by_split": {
                split: list(case_ids) for split, case_ids in CASE_IDS_BY_SPLIT.items()
            },
            "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        }
    )
).hexdigest()


@dataclass(frozen=True)
class CorrectedLegacyQuery:
    query_id: str
    doc_id: str
    motion_root_id: str
    mask_parent_id: str
    query_text: str
    positive_passage_ids: tuple[str, ...]
    visible_passage_ids: tuple[str, ...]
    flat_query_text_plain: str
    flat_query_text_masked: str


@dataclass(frozen=True)
class CorrectedLegacyData:
    corpus_by_passage_id: Mapping[str, CorpusPassage]
    queries_by_split: Mapping[str, tuple[CorrectedLegacyQuery, ...]]
    passage_ids_by_split: Mapping[str, tuple[str, ...]]
    candidate_passage_ids_by_case: Mapping[str, tuple[str, ...]]
    gold_passage_ids_by_case: Mapping[str, frozenset[str]]
    training_background_passage_ids: tuple[str, ...]
    membership_sha256: str


def _require_non_empty_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string, not {type(value).__name__}")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and whitespace-trimmed")
    return value


def _require_string_list(name: str, value: object, *, allow_empty: bool) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON list")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    result = tuple(
        _require_non_empty_string(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate strings")
    return result


def _read_jsonl(path: Path) -> list[object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corrected-v2 input: {path}")
    records: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Corrected-v2 input is empty: {path}")
    return records


def _load_corpus(path: Path) -> Mapping[str, CorpusPassage]:
    corpus: dict[str, CorpusPassage] = {}
    for row_number, raw in enumerate(_read_jsonl(path), start=1):
        if type(raw) is not dict:
            raise TypeError(f"corpus row {row_number} must be a JSON object")
        passage_id = _require_non_empty_string(
            f"corpus row {row_number}.passage_id", raw.get("passage_id")
        )
        if passage_id in corpus:
            raise ValueError(f"Duplicate corrected-v2 passage_id={passage_id!r}")
        doc_id = _require_non_empty_string(f"corpus row {row_number}.doc_id", raw.get("doc_id"))
        label = _require_non_empty_string(f"corpus row {row_number}.label", raw.get("label"))
        text = _require_non_empty_string(f"corpus row {row_number}.text", raw.get("text"))
        corpus[passage_id] = CorpusPassage(passage_id, doc_id, label, text)
    return MappingProxyType(dict(sorted(corpus.items())))


def _load_queries(path: Path) -> tuple[CorrectedLegacyQuery, ...]:
    queries: list[CorrectedLegacyQuery] = []
    seen_query_ids: set[str] = set()
    for row_number, raw in enumerate(_read_jsonl(path), start=1):
        if type(raw) is not dict:
            raise TypeError(f"query row {row_number} must be a JSON object")
        query_id = _require_non_empty_string(
            f"query row {row_number}.query_id", raw.get("query_id")
        )
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate corrected-v2 query_id={query_id!r}")
        seen_query_ids.add(query_id)
        positives = _require_string_list(
            f"query {query_id}.positive_passage_ids",
            raw.get("positive_passage_ids"),
            allow_empty=False,
        )
        visible = _require_string_list(
            f"query {query_id}.visible_passage_ids",
            raw.get("visible_passage_ids", []),
            allow_empty=True,
        )
        queries.append(
            CorrectedLegacyQuery(
                query_id=query_id,
                doc_id=_require_non_empty_string(f"query {query_id}.doc_id", raw.get("doc_id")),
                motion_root_id=str(raw.get("motion_root_id") or ""),
                mask_parent_id=str(raw.get("mask_parent_id") or ""),
                query_text=_require_non_empty_string(
                    f"query {query_id}.query_text", raw.get("query_text")
                ),
                positive_passage_ids=tuple(sorted(positives)),
                visible_passage_ids=tuple(sorted(visible)),
                flat_query_text_plain=str(raw.get("flat_query_text_plain") or ""),
                flat_query_text_masked=str(raw.get("flat_query_text_masked") or ""),
            )
        )
    return tuple(sorted(queries, key=lambda query: query.query_id))


def _load_candidate_pools(
    path: Path,
    *,
    corpus_by_passage_id: Mapping[str, CorpusPassage],
) -> Mapping[str, tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corrected-v2 input: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if type(raw) is not dict:
        raise TypeError("corrected-v2 candidates_by_case.json must be a JSON object")
    result: dict[str, tuple[str, ...]] = {}
    for raw_doc_id, raw_passage_ids in raw.items():
        doc_id = _require_non_empty_string("candidate pool doc_id", raw_doc_id)
        passage_ids = _require_string_list(
            f"candidate pool {doc_id}", raw_passage_ids, allow_empty=False
        )
        for passage_id in passage_ids:
            passage = corpus_by_passage_id.get(passage_id)
            if passage is None:
                raise ValueError(f"Candidate pool {doc_id} contains unknown passage {passage_id!r}")
            if passage.doc_id != doc_id:
                raise ValueError(
                    f"Candidate pool {doc_id} contains passage {passage_id!r} from {passage.doc_id}"
                )
        result[doc_id] = tuple(sorted(passage_ids))
    return MappingProxyType(dict(sorted(result.items(), key=lambda item: int(item[0]))))


def load_corrected_legacy_data(corrected_v2_dir: Path) -> CorrectedLegacyData:
    if not isinstance(corrected_v2_dir, Path):
        raise TypeError("corrected_v2_dir must be pathlib.Path")
    if not corrected_v2_dir.is_dir():
        raise FileNotFoundError(f"Corrected-v2 directory does not exist: {corrected_v2_dir}")

    corpus = _load_corpus(corrected_v2_dir / "corpus.jsonl")
    queries = _load_queries(corrected_v2_dir / "queries" / "all.jsonl")
    candidates_by_case = _load_candidate_pools(
        corrected_v2_dir / "pools" / "candidates_by_case.json",
        corpus_by_passage_id=corpus,
    )

    expected_cases = frozenset(
        case_id for case_ids in CASE_IDS_BY_SPLIT.values() for case_id in case_ids
    )
    if sum(len(case_ids) for case_ids in CASE_IDS_BY_SPLIT.values()) != len(expected_cases):
        raise RuntimeError("Embedded corrected-legacy case membership overlaps")
    corpus_cases = frozenset(passage.doc_id for passage in corpus.values())
    query_cases = frozenset(query.doc_id for query in queries)
    candidate_cases = frozenset(candidates_by_case)
    for name, actual in (
        ("corpus", corpus_cases),
        ("queries", query_cases),
        ("candidate pools", candidate_cases),
    ):
        if actual != expected_cases:
            raise ValueError(
                f"Corrected-v2 {name} case membership differs from frozen March membership: "
                f"missing={sorted(expected_cases - actual, key=int)}, "
                f"extra={sorted(actual - expected_cases, key=int)}"
            )

    corpus_ids_by_case: dict[str, tuple[str, ...]] = {
        case_id: tuple(
            passage_id
            for passage_id, passage in corpus.items()
            if passage.doc_id == case_id
        )
        for case_id in sorted(expected_cases, key=int)
    }
    for case_id, corpus_ids in corpus_ids_by_case.items():
        if candidates_by_case[case_id] != corpus_ids:
            raise ValueError(
                f"Corrected-v2 candidate pool for case {case_id} is not the complete corpus case"
            )

    queries_by_split: dict[str, tuple[CorrectedLegacyQuery, ...]] = {}
    passage_ids_by_split: dict[str, tuple[str, ...]] = {}
    for split, case_ids in CASE_IDS_BY_SPLIT.items():
        case_set = frozenset(case_ids)
        split_queries = tuple(query for query in queries if query.doc_id in case_set)
        split_passages = tuple(
            passage_id for passage_id, passage in corpus.items() if passage.doc_id in case_set
        )
        if len(split_queries) != EXPECTED_QUERY_COUNTS[split]:
            raise ValueError(
                f"Corrected-v2 {split} query count={len(split_queries)}; "
                f"expected exactly {EXPECTED_QUERY_COUNTS[split]}"
            )
        if len(split_passages) != EXPECTED_PASSAGE_COUNTS[split]:
            raise ValueError(
                f"Corrected-v2 {split} passage count={len(split_passages)}; "
                f"expected exactly {EXPECTED_PASSAGE_COUNTS[split]}"
            )
        queries_by_split[split] = split_queries
        passage_ids_by_split[split] = split_passages

    if len(queries) != EXPECTED_TOTAL_QUERY_COUNT or len(corpus) != EXPECTED_TOTAL_PASSAGE_COUNT:
        raise ValueError(
            "Corrected-v2 total counts changed: "
            f"queries={len(queries)}, passages={len(corpus)}"
        )

    gold_by_case: dict[str, set[str]] = {case_id: set() for case_id in expected_cases}
    for query in queries:
        for passage_id in query.positive_passage_ids:
            passage = corpus.get(passage_id)
            if passage is None:
                raise ValueError(f"Query {query.query_id!r} has unknown gold passage {passage_id!r}")
            if passage.doc_id != query.doc_id:
                raise ValueError(
                    f"Query {query.query_id!r} has cross-case gold passage {passage_id!r}"
                )
            gold_by_case[query.doc_id].add(passage_id)

    train_cases = frozenset(TRAIN_CASE_IDS)
    training_background = tuple(
        passage_id
        for passage_id, passage in corpus.items()
        if passage.doc_id in train_cases and passage.label == "Background Facts"
    )
    if not training_background:
        raise ValueError("Corrected-v2 training membership has no Background Facts passages")

    return CorrectedLegacyData(
        corpus_by_passage_id=corpus,
        queries_by_split=MappingProxyType(queries_by_split),
        passage_ids_by_split=MappingProxyType(passage_ids_by_split),
        candidate_passage_ids_by_case=candidates_by_case,
        gold_passage_ids_by_case=MappingProxyType(
            {case_id: frozenset(values) for case_id, values in gold_by_case.items()}
        ),
        training_background_passage_ids=training_background,
        membership_sha256=MEMBERSHIP_SHA256,
    )
