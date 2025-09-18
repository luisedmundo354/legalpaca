import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from collections import defaultdict
from datetime import datetime

# --------------  helpers --------------

def load_json(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_docid(key: str) -> str:

    return key.split(":", 1)[0]

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32)

def save_retrieval_results(
    out_dir: Path,
    per_query: Dict[str, dict],
    score_collections: Dict[int, Dict[str, dict]],
    corpus_keys: List[str],
    corpus_docids: List[str],
    corpus_texts: List[str],
    key_to_row: Dict[str, int],
    metrics: Dict,
    run_name: str = "baseline",
    include_text: bool = True,
    text_char_limit: int = 512,
) -> Path:

    out_dir.mkdir(parents=True, exist_ok=True)

    key_to_docid = {k: d for k, d in zip(corpus_keys, corpus_docids)}

    # 1) Full ranking per query (JSONL)
    run_jsonl = out_dir / "run.jsonl"
    with open(run_jsonl, "w", encoding="utf-8") as f:
        for qid, payload in per_query.items():
            pred_keys = payload.get("pred_keys", [])
            pred_scores = payload.get("pred_scores", [])
            pos_key = payload.get("positive_key")
            items = []
            for r, (k, s) in enumerate(zip(pred_keys, pred_scores), start=1):
                entry = {
                    "rank": r,
                    "key": k,
                    "docid": key_to_docid.get(k, parse_docid(k)),
                    "score": float(s),
                    "is_positive": (k == pos_key),
                }
                if include_text:
                    idx = key_to_row.get(k)
                    if idx is not None:
                        txt = str(corpus_texts[idx]).replace("\n", " ")
                        entry["text"] = txt[:text_char_limit]
                items.append(entry)

            record = {
                "query_id": qid,
                "query_text": payload.get("query_text", ""),
                "query_docid": payload.get("docid"),
                "positive_key": pos_key,
                "positive_rank": payload.get("rank"),
                "positive_score": payload.get("pos_score"),
                "candidates_same_doc": payload.get("candidates_same_doc"),
                "candidates_after_filter": payload.get("candidates_after_filter"),
                "candidates_after_dedup": payload.get("candidates_after_dedup"),
                "results": items,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2) Convenience: top-k JSONL dumps from your score_collections
    for k, perq in score_collections.items():
        path_k = out_dir / f"topk_{k}.jsonl"
        with open(path_k, "w", encoding="utf-8") as f:
            for qid, bundle in perq.items():
                # bundle["results"] already has key/score/text/is_positive
                rows = []
                for r, it in enumerate(bundle.get("results", []), start=1):
                    rows.append({
                        "rank": r,
                        "key": it["key"],
                        "docid": key_to_docid.get(it["key"], parse_docid(it["key"])),
                        "score": float(it["score"]),
                        "is_positive": bool(it.get("is_positive", False)),
                        "text": str(it.get("text", ""))[:text_char_limit] if include_text else None,
                    })
                rec = {
                    "query_id": qid,
                    "query_text": bundle.get("query_text", ""),
                    "query_docid": bundle.get("docid", None),
                    "k": k,
                    "results": rows,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) Standard TREC run (qid Q0 docid rank score run_name)
    trec_path = out_dir / "run.trec"
    with open(trec_path, "w", encoding="utf-8") as f:
        for qid, payload in per_query.items():
            for r, (k, s) in enumerate(zip(payload.get("pred_keys", []),
                                           payload.get("pred_scores", [])), start=1):
                docid = key_to_docid.get(k, parse_docid(k))
                f.write(f"{qid} Q0 {docid} {r} {float(s):.8f} {run_name}\n")

    # 4) Summary/context
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_name": run_name,
        "num_queries": len(per_query),
        "files": {
            "run_jsonl": str(run_jsonl),
            "trec_run": str(trec_path),
            "topk_jsonl": [f"topk_{k}.jsonl" for k in score_collections.keys()],
        },
        "metrics": metrics,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {run_jsonl}")
    print(f"[saved] {trec_path}")
    for k in score_collections.keys():
        print(f"[saved] {out_dir / f'topk_{k}.jsonl'}")
    print(f"[saved] {out_dir / 'summary.json'}")

    return out_dir


# -------------- corpus build: target_set + positives --------------

def build_corpus(
        target_set: Dict[str, dict],
        positives: Dict[str, dict],
        positive_prefix: str = "p_",
) -> Tuple[List[str], List[str], List[str], np.ndarray, Dict[str, int]]:
    """
    Build the full candidate corpus by combining:
      - target_set: your candidate passages (the 'suffix' in your original code)
      - positives: per-query positives, re-keyed with a prefix so they can't collide

    Returns:
      corpus_keys:   List[str]         # unique ids; positives are prefixed with "p_"
      corpus_docids: List[str]         # docid for each corpus item
      corpus_texts:  List[str]         # human-readable text
      X:             np.ndarray [N, d] # L2-normalized embeddings (float32)
      key_to_row:    Dict[str, int]    # key -> row index in X
    """
    keys, docids, texts, vecs = [], [], [], []

    # 1) target_set items
    for key, val in target_set.items():
        emb = np.asarray(val["embedding"], dtype=np.float32)
        txt = str(val.get("textual", ""))  # plain text field
        keys.append(key)
        docids.append(parse_docid(key))
        texts.append(txt)
        vecs.append(emb)

    # 2) positives (re-keyed to avoid collision)
    h = 0
    for pos_key, val in positives.items():
        emb = np.asarray(val["embedding"], dtype=np.float32)
        # Prefer "textual"; fall back to older "positive" field if present
        txt = str(val.get("positive", ""))
        k = 0
        if k <2 and h<2:
            print("The positive text:")
            print(txt)
            k += 1
            h += 1
        new_key = f"{positive_prefix}{pos_key}"
        keys.append(new_key)
        docids.append(parse_docid(pos_key))
        texts.append(txt)
        vecs.append(emb)
    print("It passed positive assignation")
    X = np.vstack(vecs).astype(np.float32)
    X = l2_normalize_rows(X)  # cosine via normalized dot product
    key_to_row = {k: i for i, k in enumerate(keys)}
    return keys, docids, texts, X, key_to_row


# -------------- evaluation --------------

def evaluate(
        prefix: Dict[str, dict],              # queries
        corpus_keys: List[str],
        corpus_docids: List[str],
        corpus_texts: List[str],
        X: np.ndarray,
        key_to_row: Dict[str, int],
        ks: Iterable[int] = (1, 5, 10, 20, 50),
        print_examples: int = 5,
        positive_prefix: str = "p_",
        # default behaviors
        restrict_to_same_doc: bool = True,
        remove_other_same_doc_positives: bool = True,
        dedup_text_prefer_positive: bool = False,  # near-duplicate removal no longer needed
):
    ks = sorted(set(int(k) for k in ks))
    N = X.shape[0]

    # ---- Precompute normalized query embeddings + metadata ----
    q_keys   = list(prefix.keys())
    Q        = []
    q_texts  = []
    q_docids = []
    pos_keys = []
    for qk in q_keys:
        qvec = np.asarray(prefix[qk]["embedding"], dtype=np.float32)
        qvec = qvec / max(1e-12, float(np.linalg.norm(qvec)))
        Q.append(qvec.astype(np.float32))
        q_texts.append(str(prefix[qk].get("textual", "")))
        q_docids.append(parse_docid(qk))
        pos_keys.append(f"{positive_prefix}{qk}")
    Q = np.vstack(Q)

    # ---- Vectorized helpers over corpus ----
    keys_arr   = np.array(corpus_keys, dtype=object)
    docids_arr = np.array(corpus_docids, dtype=object)
    is_pos_arr = np.array([str(k).startswith(positive_prefix) for k in corpus_keys], dtype=bool)

    # ---- Aggregation containers ----
    per_query = {}
    score_collections = {k: {} for k in ks}
    hits_at_k   = {k: [] for k in ks}
    recip_ranks = []
    ranks_found = []
    eligible    = 0  # queries whose designated positive appears in the filtered set

    # per-document aggregation
    doc_buckets = defaultdict(lambda: {
        "num_queries": 0,
        "same_doc_sizes": [],
        "after_filter_sizes": [],
        "ranks": [],
        "rrs": [],
    })

    for qi, qk in enumerate(q_keys):
        qvec = Q[qi]
        did  = q_docids[qi]
        pos_key = pos_keys[qi]
        pos_idx = key_to_row.get(pos_key, None)

        # Cosine similarity
        scores = X @ qvec

        #  Restrict to same-doc items
        if restrict_to_same_doc:
            allowed_mask = (docids_arr == did)
            same_doc_count = int(np.sum(allowed_mask))  # == target_set + positives
        else:
            allowed_mask = np.ones(N, dtype=bool)
            same_doc_count = N

        #  Remove same-doc positives
        if remove_other_same_doc_positives:
            other_pos_mask = is_pos_arr & (docids_arr == did) & (keys_arr != pos_key)
            allowed_mask[other_pos_mask] = False


        if pos_idx is not None:
            allowed_mask[pos_idx] = True

        allowed_indices = np.flatnonzero(allowed_mask)
        after_filter = allowed_indices.size  # == target_set + 1 positive


        if after_filter == 0:
            per_query[qk] = {
                "docid": did,
                "query_text": q_texts[qi],
                "positive_key": pos_key,
                "pred_keys": [],
                "pred_scores": [],
                "rank": None,
                "pos_score": None,
                "found_at_k": {k: False for k in ks},
                "candidates_same_doc": same_doc_count,
                "candidates_after_filter": 0,
                "candidates_after_dedup": 0,
            }

            doc_buckets[did]["num_queries"] += 1
            doc_buckets[did]["same_doc_sizes"].append(same_doc_count)
            doc_buckets[did]["after_filter_sizes"].append(0)
            continue

        if dedup_text_prefer_positive:
            groups = defaultdict(list)
            for i in allowed_indices:
                norm = " ".join(str(corpus_texts[i]).split()).lower()
                groups[norm].append(i)
            selected = []
            for idxs in groups.values():
                if pos_idx is not None and pos_idx in idxs:
                    chosen = pos_idx
                else:
                    chosen = max(idxs, key=lambda j: scores[j])
                selected.append(chosen)
            selected = sorted(set(selected), key=lambda j: scores[j], reverse=True)
        else:
            selected = sorted(allowed_indices, key=lambda j: scores[j], reverse=True)

        after_dedup = len(selected)

        # Positive rank & metrics
        if pos_idx is not None and (pos_idx in selected):
            eligible += 1
            pos_score = float(scores[pos_idx])
            # 1-based rank; stable under ties (minimum rank)
            rank = 1 + int(np.sum(scores[selected] > pos_score))
            recip_ranks.append(1.0 / rank)
            ranks_found.append(rank)
            found_at_k = {k: (rank <= k) for k in ks}
            for k in ks:
                hits_at_k[k].append(1 if rank <= k else 0)
        else:
            pos_score = None
            rank = None
            found_at_k = {k: False for k in ks}

        #  (top-k)
        for k in ks:
            kk = min(k, after_dedup)
            entries = []
            for i in selected[:kk]:
                entries.append({
                    "key": corpus_keys[i],
                    "score": float(scores[i]),
                    "text": corpus_texts[i],
                    "is_positive": (corpus_keys[i] == pos_key),
                })
            score_collections[k][qk] = {
                "query_text": q_texts[qi],
                "docid": did,
                "results": entries
            }

        #  payload
        per_query[qk] = {
            "docid": did,
            "query_text": q_texts[qi],
            "positive_key": pos_key,
            "pred_keys": [corpus_keys[i] for i in selected],
            "pred_scores": [float(scores[i]) for i in selected],
            "rank": rank,
            "pos_score": pos_score,
            "found_at_k": found_at_k,
            "candidates_same_doc": same_doc_count,
            "candidates_after_filter": after_filter,
            "candidates_after_dedup": after_dedup,
        }


        doc_buckets[did]["num_queries"] += 1
        doc_buckets[did]["same_doc_sizes"].append(same_doc_count)
        doc_buckets[did]["after_filter_sizes"].append(after_filter)
        if rank is not None:
            doc_buckets[did]["ranks"].append(rank)
            doc_buckets[did]["rrs"].append(1.0 / rank)

    # ---- Macro averages ----
    def _mean(xs):
        arr = np.asarray(xs, dtype=float)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    macro_metrics = {
        "hit_rate": {k: _mean(hits_at_k[k]) if eligible > 0 else float("nan") for k in ks},
        "mrr": _mean(recip_ranks) if eligible > 0 else float("nan"),
        "mean_rank_found": _mean(ranks_found) if ranks_found else float("nan"),
        "queries_with_positive": int(eligible),
    }

    # ---- Per-document stats and macro over documents ----
    per_doc_stats = {}
    for did, b in doc_buckets.items():
        per_doc_stats[did] = {
            "num_queries": b["num_queries"],
            # these two implement your requested labels
            "candidates_target_plus_positives": _mean(b["same_doc_sizes"]),
            "candidates_target_plus_one_positive": _mean(b["after_filter_sizes"]),
            "avg_rank": _mean(b["ranks"]) if b["ranks"] else float("nan"),
            "mrr": _mean(b["rrs"]) if b["rrs"] else float("nan"),
        }

    per_doc_macro = {
        "avg_candidates_target_plus_positives": _mean([v["candidates_target_plus_positives"] for v in per_doc_stats.values()]) if per_doc_stats else float("nan"),
        "avg_candidates_target_plus_one_positive": _mean([v["candidates_target_plus_one_positive"] for v in per_doc_stats.values()]) if per_doc_stats else float("nan"),
        "avg_rank": _mean([v["avg_rank"] for v in per_doc_stats.values()]) if per_doc_stats else float("nan"),
        "mrr": _mean([v["mrr"] for v in per_doc_stats.values()]) if per_doc_stats else float("nan"),
        "num_documents": len(per_doc_stats),
    }

    # ---- Console report ----
    print("\n=== Macro metrics (same-document ranking) ===")
    for k in ks:
        hr = macro_metrics["hit_rate"][k]
        print(f"Hit@{k:>2} = {hr:.4f}" if not np.isnan(hr) else f"Hit@{k:>2} = NaN")
    print(f"MRR     = {macro_metrics['mrr']:.4f}" if not np.isnan(macro_metrics['mrr']) else "MRR     = NaN")
    print(f"Mean rank (found) = {macro_metrics['mean_rank_found']:.2f}" if not np.isnan(macro_metrics['mean_rank_found']) else "Mean rank (found) = NaN")
    print(f"Queries with positive in filtered corpus = {eligible}")

    print("\n=== Per-document aggregates ===")
    for did in sorted(per_doc_stats.keys()):
        s = per_doc_stats[did]
        print(
            f"{did} | #queries={s['num_queries']} | "
            f"target_set + positives: {s['candidates_target_plus_positives']:.1f} | "
            f"target_set + 1 positive: {s['candidates_target_plus_one_positive']:.1f} | "
            f"avg rank={s['avg_rank']:.2f} | MRR={s['mrr']:.4f}"
        )

    print("\n=== Macro over documents ===")
    print(
        f"Avg candidates (target_set + positives): {per_doc_macro['avg_candidates_target_plus_positives']:.2f}\n"
        f"Avg candidates (target_set + 1 positive): {per_doc_macro['avg_candidates_target_plus_one_positive']:.2f}\n"
        f"Avg rank across documents: {per_doc_macro['avg_rank']:.2f}\n"
        f"MRR across documents: {per_doc_macro['mrr']:.4f}\n"
        f"Documents counted: {per_doc_macro['num_documents']}"
    )

    print("\n=== Examples ===")
    show_n = min(int(print_examples), len(q_keys))
    k_display = 10 if 10 in ks else max(ks)
    for qk in q_keys[:show_n]:
        entry_k   = score_collections.get(k_display, {}).get(qk, {"results": []})
        did       = per_query[qk]["docid"]
        qtxt      = per_query[qk]["query_text"]
        pos_key   = per_query[qk]["positive_key"]
        rank      = per_query[qk]["rank"]
        pos_score = per_query[qk]["pos_score"]
        found_at_k = per_query[qk]["found_at_k"]
        c_same    = per_query[qk]["candidates_same_doc"]
        c_filt    = per_query[qk]["candidates_after_filter"]
        c_dedp    = per_query[qk]["candidates_after_dedup"]
        pred_keys = per_query[qk]["pred_keys"]
        pred_scores = per_query[qk]["pred_scores"]

        print(f"\n--- Query: {qk}")
        print(f"Document: {did}")
        print(f"[Filter] target_set + positives: {c_same} | target_set + 1 positive: {c_filt} | after dedup (if enabled): {c_dedp}")
        print(f"Query: {qtxt}")

        if rank is not None:
            found_str = ", ".join(f"@{k}:{'Y' if found_at_k[k] else 'N'}" for k in ks)
            print(f"Positive: {pos_key}  |  rank={rank}  |  score={pos_score:.4f}  |  found {found_str}")
        else:
            print(f"Positive: {pos_key}  |  rank=NA (positive missing in filtered set)")

        # Top-K
        print(f"Top-{k_display} retrieved (cosine) within same document:")
        for r, item in enumerate(entry_k["results"], 1):
            tag = " [POSITIVE]" if item["is_positive"] else ""
            preview = str(item["text"]).replace("\n", " ")[:160]
            print(f"{r:>2}. {item['score']:.4f}  {item['key']}{tag}")
            print(f"    {preview}")


        if pred_keys:
            # Top-2
            print("Top-2 passages:")
            for i in range(min(2, len(pred_keys))):
                k_i, s_i = pred_keys[i], pred_scores[i]
                tag = " [POSITIVE]" if k_i == pos_key else ""
                print(f"  {i+1}. {s_i:.4f}  {k_i}{tag}")
            # Worst-2
            if len(pred_keys) >= 2:
                print("Worst-2 passages:")
                for j, idx in enumerate(range(max(0, len(pred_keys)-2), len(pred_keys)), 1):
                    k_j, s_j = pred_keys[idx], pred_scores[idx]
                    tag = " [POSITIVE]" if k_j == pos_key else ""
                    print(f"  -{2-j+1}. {s_j:.4f}  {k_j}{tag}")


        idx = key_to_row.get(pos_key)
        pos_text = corpus_texts[idx] if idx is not None else "(positive text missing)"
        print(f"Right one [POSITIVE]: {pos_key}")
        print(f"    {str(pos_text)[:200]}")


    metrics = {
        **macro_metrics,
        "per_doc": per_doc_stats,
        "per_doc_macro": per_doc_macro,
    }
    return metrics, per_query, score_collections


# -------------- main runner --------------
if __name__ == "__main__":

    print("file paths assigned")
    prefix_path    = Path("embedded_data/embedded_output_query/ed53ff8aeaf44f02825061eb1c2ef08b.json")
    target_path    = Path("embedded_data/embedded-output/b4185d7e944043eb858c4ab87154f397.json")  # formerly 'suffix'
    positives_path = Path("embedded_data/embedded_output_positives/ed53ff8aeaf44f02825061eb1c2ef08b.json")
    print("\n=== paths assigned ===")
    prefix    = load_json(prefix_path)     # queries
    target_set = load_json(target_path)    # candidate passages
    positives = load_json(positives_path)  # positives
    print("Documents loaded")
    corpus_keys, corpus_docids, corpus_texts, X, key_to_row = build_corpus(
        target_set, positives, positive_prefix="p_"
    )

    # Evaluate
    metrics, per_query, score_cols = evaluate(
        prefix=prefix,
        corpus_keys=corpus_keys,
        corpus_docids=corpus_docids,
        corpus_texts=corpus_texts,
        X=X,
        key_to_row=key_to_row,
        ks=(1, 5, 10, 20, 50),
        print_examples=5,
        positive_prefix="p_",
        restrict_to_same_doc=True,
        remove_other_same_doc_positives=True,
        dedup_text_prefer_positive=False,  # keep off per your request
    )

    out_dir = Path("retrieval_results/retrieved")
    save_retrieval_results(
        out_dir=out_dir,
        per_query=per_query,
        score_collections=score_cols,
        corpus_keys=corpus_keys,
        corpus_docids=corpus_docids,
        corpus_texts=corpus_texts,
        key_to_row=key_to_row,
        metrics=metrics,
        run_name="baseline",
        include_text=True,
        text_char_limit=5600,
    )
