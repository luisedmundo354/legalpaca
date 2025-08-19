import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np
from collections import defaultdict

# -------------- basic helpers --------------

def load_json(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_docid(key: str) -> str:
    # keys shaped like "docid:sentid"
    return key.split(":", 1)[0]

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32)

def topk_indices_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores in DESC order."""
    k = min(k, scores.size)
    if k <= 0:
        return np.empty(0, dtype=int)
    # fast top-k pattern: argpartition to get k largest (unsorted), then argsort them
    idx_part = np.argpartition(scores, -k)[-k:]
    return idx_part[np.argsort(scores[idx_part])][::-1]  # highest first


# -------------- corpus build: suffix + positives --------------

def build_corpus(
        suffix: Dict[str, dict],
        positives: Dict[str, dict],
        positive_prefix: str = "p_"
):
    """
    Returns:
      corpus_keys:   List[str]         # unique ids; positives are prefixed with "p_"
      corpus_docids: List[str]         # docid for each corpus item
      corpus_texts:  List[str]         # human-readable text
      X:             np.ndarray [N, d] # L2-normalized embeddings (float32)
      key_to_row:    Dict[str, int]    # key -> row index in X
    """
    keys, docids, texts, vecs = [], [], [], []

    # 1) suffix items
    for key, val in suffix.items():
        emb = np.asarray(val["embedding"], dtype=np.float32)
        txt = str(val.get("textual", ""))
        keys.append(key)                       # keep original key for suffix
        docids.append(parse_docid(key))
        texts.append(txt)
        vecs.append(emb)

    # 2) positives (re-keyed to avoid collision)
    for pos_key, val in positives.items():
        emb = np.asarray(val["embedding"], dtype=np.float32)
        txt = str(val.get("textual", ""))     # positive text
        new_key = f"{positive_prefix}{pos_key}"
        keys.append(new_key)
        docids.append(parse_docid(pos_key))   # docid from the original key
        texts.append(txt)
        vecs.append(emb)

    X = np.vstack(vecs).astype(np.float32)
    X = l2_normalize_rows(X)  # cosine via normalized dot product
    key_to_row = {k: i for i, k in enumerate(keys)}
    return keys, docids, texts, X, key_to_row


# -------------- evaluation --------------

def evaluate(
        prefix: Dict[str, dict],
        corpus_keys: List[str],
        corpus_docids: List[str],
        corpus_texts: List[str],
        X: np.ndarray,
        key_to_row: Dict[str, int],
        ks: Iterable[int] = (1, 5, 10, 20, 50),
        print_examples: int = 5,
        positive_prefix: str = "p_"
):
    """
    For each query in prefix:
      - compute cosine scores against all corpus items
      - top-k retrieval
      - define gold set as all corpus items with the same docid as the query
      - compute Precision@k, Recall@k, F1@k
    Also prints 5 query reports (doc name, query, top-10 with markers).
    Returns:
      macro_metrics: {k: {precision, recall, f1}}
      per_query:     {qkey: {...}}
      score_collections: {k: {qkey: [{"key","score","text","is_gold","is_positive"}...]}}
    """
    ks = sorted(set(int(k) for k in ks))
    N = X.shape[0]
    d = X.shape[1]

    # Precompute normalized query embeddings
    q_keys = list(prefix.keys())
    Q = []
    q_texts = []
    q_docids = []
    pos_keys = []  # per-query positive key in the combined corpus ("p_"+qkey)

    for qk in q_keys:
        qvec = np.asarray(prefix[qk]["embedding"], dtype=np.float32)
        qvec = qvec / max(1e-12, float(np.linalg.norm(qvec)))
        Q.append(qvec.astype(np.float32))
        q_texts.append(str(prefix[qk].get("textual", "")))
        q_docids.append(parse_docid(qk))
        pos_keys.append(f"{positive_prefix}{qk}")

    Q = np.vstack(Q)  # [num_queries, d]

    # Build an index of gold membership by docid (same-doc == gold)
    gold_indices_by_doc = defaultdict(list)
    for i, did in enumerate(corpus_docids):
        gold_indices_by_doc[did].append(i)

    per_query = {}
    score_collections = {k: {} for k in ks}
    agg_prec = {k: [] for k in ks}
    agg_rec  = {k: [] for k in ks}
    agg_f1   = {k: [] for k in ks}

    # Compute all scores query-by-query to keep it simple and memory-safe
    for qi, qk in enumerate(q_keys):
        qvec = Q[qi]                       # [d]
        scores = X @ qvec                  # cosine similarities
        max_k = min(max(ks), N)
        top_idx = topk_indices_desc(scores, max_k)  # indices of top max_k

        # predicted order for all k
        pred_indices = list(top_idx.tolist())
        pred_keys = [corpus_keys[i] for i in pred_indices]
        pred_scores = [float(scores[i]) for i in pred_indices]

        # gold set = same document as the query
        did = q_docids[qi]
        gold_indices = set(gold_indices_by_doc.get(did, []))
        G = len(gold_indices)

        # metrics per k
        metrics_at_k = {}
        for k in ks:
            kk = min(k, len(pred_indices))
            topk_idx = pred_indices[:kk]
            hit = sum((i in gold_indices) for i in topk_idx)
            precision = hit / k if k > 0 else 0.0
            recall = (hit / G) if G > 0 else np.nan
            f1 = 0.0
            if not np.isnan(recall) and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            metrics_at_k[k] = {"precision": precision, "recall": recall, "f1": f1, "hits": hit}

            # for macro, ignore queries with G==0 (no golds)
            if G > 0:
                agg_prec[k].append(precision)
                agg_rec[k].append(recall)
                agg_f1[k].append(f1)

            # store top-k containers for convenience
            entries = []
            for i in pred_indices[:kk]:
                entries.append({
                    "key": corpus_keys[i],
                    "score": float(scores[i]),
                    "text": corpus_texts[i],
                    "is_gold": (i in gold_indices),
                    "is_positive": (corpus_keys[i] == pos_keys[qi]),
                })
            score_collections[k][qk] = {
                "query_text": q_texts[qi],
                "docid": did,
                "results": entries
            }

        per_query[qk] = {
            "docid": did,
            "query_text": q_texts[qi],
            "positive_key": pos_keys[qi],
            "pred_keys": pred_keys,
            "pred_scores": pred_scores,
            "metrics": metrics_at_k
        }

    # macro averages
    def _mean(xs):
        arr = np.asarray(xs, dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    macro_metrics = {
        k: {
            "precision": _mean(agg_prec[k]),
            "recall":    _mean(agg_rec[k]),
            "f1":        _mean(agg_f1[k]),
        } for k in ks
    }

    # ---- Print a compact report for 5 queries ----
    print("\n=== Macro metrics ===")
    for k in ks:
        m = macro_metrics[k]
        print(f"@{k:>2}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")

    print("\n=== Examples (5 queries) ===")
    for qk in q_keys[:min(5, len(q_keys))]:
        entry_k10 = score_collections[10].get(qk, {"results": []})
        docid = score_collections[10].get(qk, {}).get("docid", parse_docid(qk))
        qtxt = score_collections[10].get(qk, {}).get("query_text", "")
        pos_key = per_query[qk]["positive_key"]

        print(f"\n--- Query: {qk}")
        print(f"Document: {docid}")
        print(f"Query: {qtxt}")
        print("Top-10 retrieved (cosine):")
        for rank, item in enumerate(entry_k10["results"], 1):
            tags = []
            if item["is_positive"]:
                tags.append("POSITIVE")
            if item["is_gold"]:
                tags.append("GOLD")
            tag_str = f" [{' & '.join(tags)}]" if tags else ""
            preview = item["text"].replace("\n", " ")[:160]
            print(f"{rank:>2}. {item['score']:.4f}  {item['key']}{tag_str}")
            print(f"    {preview}")

        # Show the designated positive target text explicitly
        if pos_key in (r["key"] for r in entry_k10["results"]):
            pos_text = next(r["text"] for r in entry_k10["results"] if r["key"] == pos_key)
        else:
            # If not in top-10, fetch from corpus (by key_to_row)
            idx = key_to_row.get(pos_key)
            pos_text = corpus_texts[idx] if idx is not None else "(positive text missing)"
        print(f"Right one [POSITIVE]: {pos_key}")
        print(f"    {pos_text[:200]}")

    return macro_metrics, per_query, score_collections


# -------------- main runner --------------

if __name__ == "__main__":
    # ---- Update these paths to your files ----
    prefix_path   = Path("embedded_data/embedded-output-query/f84d458b9c8b4cff8b8db170fca64a6c.json")
    suffix_path   = Path("embedded_data/embedded-output/f84d458b9c8b4cff8b8db170fca64a6c.json")
    positives_path = Path("embedded_data/embedded_output_positives/f84d458b9c8b4cff8b8db170fca64a6c.json")  # placeholder path

    # JSON expectations:
    # prefix[key]    = { "embedding": [...], "textual": "..." }
    # suffix[key]    = { "embedding": [...], "textual": "..." }
    # positives[key] = { "embedding": [...], "textual": "..." }  # SAME KEYS AS prefix

    prefix    = load_json(prefix_path)
    suffix    = load_json(suffix_path)
    positives = load_json(positives_path)

    corpus_keys, corpus_docids, corpus_texts, X, key_to_row = build_corpus(suffix, positives, positive_prefix="p_")

    # Evaluate and print 5 query examples
    metrics, per_query, score_cols = evaluate(
        prefix=prefix,
        corpus_keys=corpus_keys,
        corpus_docids=corpus_docids,
        corpus_texts=corpus_texts,
        X=X,
        key_to_row=key_to_row,
        ks=(1,5,10,20,50),
        print_examples=5,
        positive_prefix="p_"
    )
