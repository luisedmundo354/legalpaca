import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

__all__ = [
    "load_reranker_records",
    "iter_reranker_records",
    "select_reranker_record",
    "parse_rank_expression",
    "borda_from_rank_expression",
    "_load_jsonl",
    "_select_record",
    "_parse_rank_notation",
    "_borda_from_groups",
    "_build_batches",
    "_iter_jsonl",
]

_rank_pat = re.compile(r"\[(\d+)\]")


def load_reranker_records(path: Path) -> List[dict]:
    return _load_jsonl(path)


def iter_reranker_records(path: Path) -> Iterable[dict]:
    yield from _iter_jsonl(path)


def select_reranker_record(
    records: List[dict], query_id: str = None, query_text: str = None
) -> dict:
    return _select_record(records, query_id=query_id, query_text=query_text)


def parse_rank_expression(s: str, n_local: int) -> List[List[int]]:
    return _parse_rank_notation(s, n_local)


def borda_from_rank_expression(s: str, n_local: int) -> Dict[int, float]:
    groups = _parse_rank_notation(s, n_local)
    return _borda_from_groups(groups, n_local)


def _load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _select_record(records: List[dict], query_id: str = None, query_text: str = None) -> dict:
    if query_id:
        for r in records:
            if str(r.get("query_id")) == str(query_id):
                return r
    if query_text:
        for r in records:
            if r.get("query_text", "").strip() == query_text.strip():
                return r
    return records[0]


def _parse_rank_notation(s: str, n_local: int) -> List[List[int]]:
    groups: List[List[int]] = []
    print("This is the s:", s)
    cleaned = _clean_rank_string(s)
    for seg in _split_rank_segments(cleaned):
        ids = _extract_rank_ids(seg, n_local)
        if ids:
            groups.append(ids)
        print("This is the segment from the reranker output string bfr parsing:", seg)
    if not groups:
        print("The segment from the reranker output string bfr parsing is empty. Parse number in order.")
        groups = _fallback_rank_groups(s, n_local)
    return groups


def _borda_from_groups(groups: List[List[int]], n_local: int) -> Dict[int, float]:
    scores = defaultdict(float)
    pos = 1
    for g in groups:
        m = len(g)
        avg = sum(n_local - r for r in range(pos, pos + m)) / float(m)
        for item in g:
            scores[item] += avg
        pos += m
    for i in range(1, n_local + 1):
        scores.setdefault(i, 0.0)
    return scores


def _build_batches(n: int, k: int = 8, stride: int = 6) -> List[List[int]]:
    batches: List[List[int]] = []
    if n <= k:
        return [list(range(n))]
    for start in range(0, n, stride):
        end = min(n, start + k)
        if end - start < k:
            start = max(0, n - k)
            end = n
        batches.append(list(range(start, end)))
        if end == n:
            break
    return batches


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path} invalid JSON on line {i}: {line[:120]!r}") from e


def _clean_rank_string(s: str) -> str:
    return re.sub(r"[^\[\]\d=>]", "", s)


def _split_rank_segments(cleaned: str) -> List[str]:
    return cleaned.split(">")


def _extract_rank_ids(seg: str, n_local: int) -> List[int]:
    ids = [int(x) for x in _rank_pat.findall(seg)]
    return [i for i in ids if 1 <= i <= n_local]


def _fallback_rank_groups(s: str, n_local: int) -> List[List[int]]:
    ids = [int(x) for x in re.findall(r"\d+", s)]
    ids = [i for i in ids if 1 <= i <= n_local]
    return [[i] for i in ids] if ids else []
