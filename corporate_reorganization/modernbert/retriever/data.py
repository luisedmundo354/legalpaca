from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Tuple


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


PASSAGE_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True, eq=False)
class PassageIndexTable:
    """Immutable corpus-wide bijection between passage IDs and contiguous integers."""

    _passage_ids: Tuple[str, ...]
    _passage_texts: Tuple[str, ...]
    _index_by_passage_id: Mapping[str, int]
    _sha256: str

    def __init__(self, corpus_by_passage_id: Mapping[str, CorpusPassage]) -> None:
        if not isinstance(corpus_by_passage_id, Mapping) or not corpus_by_passage_id:
            raise ValueError("PassageIndexTable requires a non-empty corpus mapping")

        passage_ids = tuple(sorted(corpus_by_passage_id))
        for passage_id in passage_ids:
            if type(passage_id) is not str or not passage_id or passage_id.strip() != passage_id:
                raise ValueError(f"Invalid corpus passage ID: {passage_id!r}")
            passage = corpus_by_passage_id[passage_id]
            if not isinstance(passage, CorpusPassage):
                raise TypeError(
                    f"Corpus record for {passage_id!r} must be CorpusPassage, "
                    f"not {type(passage).__name__}"
                )
            if passage.passage_id != passage_id:
                raise ValueError(
                    f"Corpus key {passage_id!r} does not match record "
                    f"passage_id={passage.passage_id!r}"
                )
            if type(passage.text) is not str or not passage.text:
                raise ValueError(f"Corpus passage {passage_id!r} has empty or non-string text")

        object.__setattr__(self, "_passage_ids", passage_ids)
        object.__setattr__(
            self,
            "_passage_texts",
            tuple(corpus_by_passage_id[passage_id].text for passage_id in passage_ids),
        )
        object.__setattr__(
            self,
            "_index_by_passage_id",
            MappingProxyType(
                {passage_id: index for index, passage_id in enumerate(passage_ids)}
            ),
        )
        canonical_payload = {
            "schema_version": PASSAGE_INDEX_SCHEMA_VERSION,
            "passage_ids": list(passage_ids),
        }
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "_sha256", hashlib.sha256(canonical_bytes).hexdigest())

    def __len__(self) -> int:
        return len(self._passage_ids)

    @property
    def passage_ids(self) -> Tuple[str, ...]:
        return self._passage_ids

    @property
    def sha256(self) -> str:
        return self._sha256

    def index_for_id(self, passage_id: str) -> int:
        if type(passage_id) is not str:
            raise TypeError(f"passage_id must be an exact string, not {type(passage_id).__name__}")
        try:
            return self._index_by_passage_id[passage_id]
        except KeyError as exc:
            raise KeyError(f"Unknown corpus passage_id={passage_id!r}") from exc

    def id_for_index(self, passage_index: int) -> str:
        index = self._validate_index(passage_index)
        return self._passage_ids[index]

    def text_for_index(self, passage_index: int) -> str:
        index = self._validate_index(passage_index)
        return self._passage_texts[index]

    def indices_for_ids(self, passage_ids: List[str]) -> List[int]:
        if type(passage_ids) is not list:
            raise TypeError("passage_ids must be an exact list")
        return [self.index_for_id(passage_id) for passage_id in passage_ids]

    def _validate_index(self, passage_index: int) -> int:
        if type(passage_index) is not int:
            raise TypeError(
                f"passage_index must be an exact int, not {type(passage_index).__name__}"
            )
        if passage_index < 0 or passage_index >= len(self):
            raise IndexError(
                f"Passage index out of range: {passage_index}; corpus_size={len(self)}"
            )
        return passage_index


def load_corpus(processed_dir: Path) -> Dict[str, CorpusPassage]:
    corpus_path = processed_dir / "corpus.jsonl"
    corpus_by_passage_id: Dict[str, CorpusPassage] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            passage_id = str(rec["passage_id"])
            if passage_id in corpus_by_passage_id:
                raise ValueError(f"Duplicate corpus passage_id={passage_id!r} in {corpus_path}")
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
