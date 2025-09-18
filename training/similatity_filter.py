from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
import re
from utils import parse_docid

# Progress bar: prefer tqdm if available
try:
    from tqdm.auto import tqdm
except Exception:  # fallback no-op
    def tqdm(x, **kwargs): return x

def prepare_near_duplicate_filter(
        prefix: Dict[str, dict],
        corpus_keys: List[str],
        corpus_docids: List[str],
        corpus_texts: List[str],
        positive_prefix: str = "p_",
        docid_from_key=parse_docid,
        *,
        # --- NEW: make docid extraction pluggable ---

        prefix_docids: Optional[Dict[str, str]] = None,
        # --- thresholds (tunable) ---
        dup_min_char_len: int = 40,    # only run char-ratio when both strings >= this
        dup_char_ratio: float = 0.93,  # SequenceMatcher ratio threshold
        dup_jaccard3: float = 0.80,    # 3-gram Jaccard threshold
        dup_overlap3: float = 0.90,    # 3-gram Overlap (Szymkiewicz–Simpson)
        dup_jaccard2: float = 0.85,    # 2-gram fallback Jaccard
        dup_overlap2: float = 0.92,    # 2-gram fallback Overlap
        show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Precompute a boolean mask over the *corpus* marking which items should be
    removed as near-duplicates of {positives ∪ prefix} texts from the same document.

    Returns:
      near_dup_remove_mask: np.ndarray[bool] of shape [N]; True => remove candidate
      stats: dict with global counters (docs, candidates_scanned, removed, refs)
    """
    N = len(corpus_keys)
    near_dup_remove_mask = np.zeros(N, dtype=bool)

    keys_arr   = np.array(corpus_keys, dtype=object)
    docids_arr = np.array(corpus_docids, dtype=object)
    texts_arr  = np.array(corpus_texts, dtype=object)
    is_pos_arr = np.array([str(k).startswith(positive_prefix) for k in corpus_keys], dtype=bool)

    # --- helper for prefix docid extraction (no hard dependency on parse_docid) ---
    def _docid_for_prefix_key(k: str) -> str:
        if prefix_docids is not None and k in prefix_docids:
            return prefix_docids[k]
        if docid_from_key is not None:
            return docid_from_key(k)
        # safe default: assume keys look like "docid:sentid"
        return k.split(":", 1)[0]

    # ---------- Build per-doc reference set: POSITIVES ∪ PREFIX ----------
    doc_to_ref_texts: Dict[str, List[str]] = defaultdict(list)

    # positives in corpus (from the positives you added to the corpus)
    for i in range(N):
        if is_pos_arr[i]:
            did = docids_arr[i]
            doc_to_ref_texts[did].append(str(texts_arr[i]))

    # prefix/query texts (same doc)
    for qk, qval in prefix.items():
        did = _docid_for_prefix_key(qk)
        doc_to_ref_texts[did].append(str(qval.get("textual", "")))

    # ---------- Normalization & shingling helpers ----------
    _ws_re    = re.compile(r"\s+")
    _tag_re   = re.compile(r"<[^>]+>")
    _punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

    def _normalize(s: str) -> str:
        s = s.lower()
        s = _tag_re.sub(" ", s)     # strip XML/HTML-like tags
        s = _punct_re.sub(" ", s)   # drop punctuation
        s = _ws_re.sub(" ", s).strip()
        return s

    def _tokens(s: str) -> List[str]:
        return re.findall(r"\w+", s)

    def _shingles(tokens: List[str], n: int) -> set:
        if not tokens:
            return set()
        if len(tokens) < n:
            n = max(1, len(tokens))
        return {" ".join(tokens[i:i+n]) for i in range(0, len(tokens) - n + 1)}

    def _jaccard(a: set, b: set) -> float:
        if not a and not b: return 1.0
        u = len(a | b)
        return (len(a & b) / u) if u else 1.0

    def _overlap(a: set, b: set) -> float:
        if not a or not b: return 0.0
        return len(a & b) / min(len(a), len(b))

    # ---------- Precompute reference features per doc ----------
    docs = sorted(set(docids_arr.tolist()))
    if show_progress:
        print("Building reference features per document…")

    doc_ref_feats: Dict[str, List[dict]] = {}
    for did in tqdm(docs, disable=not show_progress, desc="Refs per doc"):
        feats = []
        for t in doc_to_ref_texts.get(did, []):
            nrm = _normalize(t)
            toks = _tokens(nrm)
            sh3  = _shingles(toks, 3)
            sh2  = _shingles(toks, 2)
            feats.append({"norm": nrm, "sh3": sh3, "sh2": sh2})
        doc_ref_feats[did] = feats

    # ---------- Scan suffix candidates per doc with progress ----------
    total_candidates = 0
    total_removed = 0
    total_refs = sum(len(v) for v in doc_ref_feats.values())

    if show_progress:
        print("Scanning suffix candidates against doc references…")

    for did in tqdm(docs, disable=not show_progress, desc="Scan per doc"):
        ref_feats = doc_ref_feats.get(did, [])
        if not ref_feats:
            continue

        # candidates: same doc & not positive (i.e., suffix items)
        cand_idx = np.flatnonzero((docids_arr == did) & (~is_pos_arr))
        total_candidates += cand_idx.size

        # precompute candidate features once
        cand_feats = {}
        for i in cand_idx:
            nrm = _normalize(str(texts_arr[i]))
            toks = _tokens(nrm)
            cand_feats[i] = {
                "norm": nrm,
                "sh3": _shingles(toks, 3),
                "sh2": _shingles(toks, 2),
            }

        # compare each candidate to the doc refs
        for i in cand_idx:
            ci = cand_feats[i]
            cn = ci["norm"]
            c3 = ci["sh3"]
            c2 = ci["sh2"]

            is_dup = False
            for rf in ref_feats:
                # 1) character-based similarity (longer strings)
                if len(cn) >= dup_min_char_len and len(rf["norm"]) >= dup_min_char_len:
                    if SequenceMatcher(None, cn, rf["norm"]).ratio() >= dup_char_ratio:
                        is_dup = True
                        break
                # 2) 3-gram overlap / jaccard
                if _overlap(c3, rf["sh3"]) >= dup_overlap3 or _jaccard(c3, rf["sh3"]) >= dup_jaccard3:
                    is_dup = True
                    break
                # 3) fallback for short: 2-grams
                if (len(c3) < 2 or len(rf["sh3"]) < 2) and (
                        _overlap(c2, rf["sh2"]) >= dup_overlap2 or _jaccard(c2, rf["sh2"]) >= dup_jaccard2
                ):
                    is_dup = True
                    break

            if is_dup:
                near_dup_remove_mask[i] = True
                total_removed += 1

    stats = {
        "docs": len(docs),
        "ref_texts_total": int(total_refs),
        "candidates_scanned": int(total_candidates),
        "candidates_removed": int(total_removed),
    }
    return near_dup_remove_mask, stats
