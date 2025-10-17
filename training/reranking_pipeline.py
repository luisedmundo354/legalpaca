from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from reranking_helpers import (
    _borda_from_groups,
    _build_batches,
    _iter_jsonl,
    _parse_rank_notation,
)

PIPELINE_VERSION = "2025-09-30"

__all__ = [
    "_read_streaming_body",
    "_call_llm_rank_order",
    "_call_sagemaker_rank_order",
    "_call_openai_rank_order",
    "_fill_rankk_template",
    "rerank_topk50_rankk_all",
    "evaluate_doc_full_recall_jsonl",
    "evaluate_reranked_jsonl",
    "PIPELINE_VERSION",
]


def _read_streaming_body(resp) -> str:
    """Collect streamed response text from SageMaker runtime events."""
    chunks: List[str] = []
    body = resp["Body"]
    for event in body:
        if "PayloadPart" in event:
            raw = event["PayloadPart"]["Bytes"].decode("utf-8", errors="ignore")
            for line in raw.splitlines():
                data = line.strip()
                if not data:
                    continue
                if data.startswith("data: "):
                    data = data[6:]
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    chunks.append(content)
        elif "ModelStreamError" in event:
            msg = event["ModelStreamError"].get("Message", "ModelStreamError")
            raise RuntimeError(f"Model stream error: {msg}")
        elif "InternalServerException" in event:
            raise RuntimeError("InternalServerException while streaming from endpoint.")
    return "".join(chunks)


def _call_llm_rank_order(
    smr_client,
    endpoint_name: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    return _call_sagemaker_rank_order(
        smr_client,
        endpoint_name,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _call_sagemaker_rank_order(
    smr_client,
    endpoint_name: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    resp = smr_client.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        Body=json.dumps(body),
        ContentType="application/json",
    )
    return _read_streaming_body(resp).strip()


def _call_openai_rank_order(
        client=None,
        prompt: str = "",
        *,
        model: str = "gpt-5",
        temperature: float = 1,
        max_tokens: int = 128,
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("OpenAI reranker received an empty prompt.")
    else:
        print(f"OpenAI reranker received the prompt")

    if client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set and no OpenAI client provided")
        client = OpenAI(api_key=api_key)
    else:
        print("------OpenAI reranker received API and client------")
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            reasoning={"effort": "low"},
            max_output_tokens=int(max_tokens),
        )
        print(f"OpenAI reranker response: {resp}")
        print(getattr(resp, "output_text", "no output text from openai model"))
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


def _fill_rankk_template(template: str, query_text: str, block: str) -> str:
    if "{passages}" in template:
        return template.format(query=query_text, passages=block)
    if "{docs}" in template:
        return template.format(query=query_text, docs=block)
    return (template.rstrip() + "\n\n" + block).format(query=query_text)


def rerank_topk50_rankk_all(
    smr_client,
    endpoint_name: str,
    rank_k_prompt: str,
    *,
    provider: str = "sagemaker",
    openai_client=None,
    openai_model: str = "gpt5",
    topk_path: Path = Path("retrieval_results/retrieved/topk_50.jsonl"),
    out_dir: Path = Path("retrieval_results/reranked"),
    window_k: int = 8,
    stride: int = 6,
    snippet_limit: int = 5000,
    temperature: float = 1,
    max_tokens: int = 128,
    run_name: str = "rankk_top50",
    reference_whole_run: str = "run.jsonl",
    verbose: bool = False,
    resume: bool = True,
    processing_batch_size: int = 5,
) -> None:
    if verbose:
        print(
            f"[rerank] provider={provider} window_k={window_k} stride={stride} "
            f"temperature={temperature} max_tokens={max_tokens}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "rankk_top50.jsonl"
    out_trec = out_dir / "rankk_top50.trec"
    out_judg = out_dir / "rankk_top50_judgments.jsonl"
    ref_path = Path("retrieval_results/retrieved") / reference_whole_run
    pos_lookup = _load_positive_lookup(ref_path)

    rank_fn = _resolve_rank_function(
        provider=provider,
        smr_client=smr_client,
        endpoint_name=endpoint_name,
        openai_client=openai_client,
        openai_model=openai_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    processed: set[str] = set()
    if resume and out_jsonl.exists():
        if verbose:
            print(f"[rerank] resume enabled; reading processed IDs from {out_jsonl}")
        with out_jsonl.open("r", encoding="utf-8") as f_prev:
            for line in f_prev:
                if not line.strip():
                    continue
                try:
                    rec_prev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec_prev.get("query_id")
                if pid:
                    processed.add(str(pid))

    mode = "a" if resume else "w"
    if verbose:
        print(f"[rerank] writing outputs in mode='{mode}'")
    print("Resume is:", resume, "mode is:", mode)
    with (
        out_jsonl.open(mode, encoding="utf-8") as fj,
        out_trec.open(mode, encoding="utf-8") as ft,
        out_judg.open(mode, encoding="utf-8") as fg,
    ):
        j = 0
        for i, rec in tqdm(enumerate(_iter_jsonl(topk_path))):
            if j == 0:
                starting_index = i
                j += 1
            if i > starting_index + processing_batch_size:
                break
            qid = str(rec.get("query_id", "q0"))
            if resume and qid in processed:
                if verbose:
                    print(f"[rerank] skipping already processed query {qid}")
                continue
            qtext = rec.get("query_text", "")
            items: List[dict] = rec.get("results", []) or []
            pos_id = pos_lookup.get(qid, "")
            if not items:
                continue

            _prepare_prompt_text(items, snippet_limit)
            batches = _build_batches(len(items), k=window_k, stride=stride)
            agg_scores, judgments = _score_batches(
                rank_fn,
                rank_k_prompt,
                qid,
                qtext,
                items,
                batches,
                verbose=verbose,
            )
            order = _rank_items(items, agg_scores)

            _write_query_outputs(
                fj,
                ft,
                fg,
                run_name,
                qid,
                qtext,
                pos_id,
                items,
                order,
                agg_scores,
                judgments,
                window_k,
                stride,
            )

        print(f"[saved] {out_jsonl}")
        print(f"[saved] {out_trec}")
        print(f"[saved] {out_judg}")


def _load_positive_lookup(path: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not path.exists():
        return lookup
    with path.open("r", encoding="utf-8") as info:
        for line in info:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("query_id", ""))
            pkey = obj.get("positive_key")
            if qid and isinstance(pkey, str):
                lookup[qid] = pkey
    return lookup


def _prepare_prompt_text(items: Sequence[dict], snippet_limit: int) -> None:
    for it in items:
        text = str(it.get("text", ""))
        it["_prompt_text"] = text[:snippet_limit].replace("\n", " ")


def _score_batches(
    rank_fn: Callable[[str], str],
    rank_k_prompt: str,
    qid: str,
    qtext: str,
    items: Sequence[dict],
    batches: Sequence[Sequence[int]],
    *,
    verbose: bool = False,
) -> Tuple[Dict[int, float], List[Dict]]:
    agg_scores: Dict[int, float] = defaultdict(float)
    judgments: List[Dict] = []

    for bidx, batch in enumerate(batches):
        block, local_to_global = _compose_prompt_block(items, batch)
        prompt = _fill_rankk_template(rank_k_prompt, qtext, block)
        if verbose:
            print(f"[rerank] prompt batch={bidx} size={len(batch)}\n{prompt}")
        llm_out = rank_fn(prompt)
        groups = _parse_rank_notation(llm_out, n_local=len(batch))
        borda = _borda_from_groups(groups, n_local=len(batch))

        for local_idx, score in borda.items():
            gi = local_to_global[local_idx]
            agg_scores[gi] += float(score)

        judgments.append({
            "query_id": qid,
            "batch_index": bidx,
            "global_indices": list(batch),
            "llm_output": llm_out,
            "parsed_groups": groups,
            "local_borda": {int(k): float(v) for k, v in borda.items()},
        })

    return agg_scores, judgments


def _resolve_rank_function(
    *,
    provider: str,
    smr_client,
    endpoint_name: str,
    openai_client,
    openai_model: str,
    temperature: float,
    max_tokens: int,
) -> Callable[[str], str]:
    provider = (provider or "sagemaker").lower()

    if provider == "sagemaker":
        if smr_client is None:
            raise RuntimeError("smr_client is required when provider='sagemaker'")

        def _rank(prompt: str) -> str:
            return _call_sagemaker_rank_order(
                smr_client,
                endpoint_name,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return _rank

    if provider == "openai":

        def _rank(prompt: str) -> str:
            return _call_openai_rank_order(
                openai_client,
                prompt,
                model=openai_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return _rank

    raise ValueError("Unsupported provider. Use 'sagemaker' or 'openai'.")


def _compose_prompt_block(
    items: Sequence[dict],
    batch: Sequence[int],
) -> Tuple[str, Dict[int, int]]:
    lines: List[str] = []
    local_to_global: Dict[int, int] = {}
    for local_idx, global_idx in enumerate(batch, start=1):
        local_to_global[local_idx] = global_idx
        lines.append(f"[{local_idx}] {items[global_idx]['_prompt_text']}")
    return "\n".join(lines), local_to_global


def _rank_items(items: Sequence[dict], agg_scores: Dict[int, float]) -> List[int]:
    return sorted(
        range(len(items)),
        key=lambda gi: (
            -agg_scores.get(gi, 0.0),
            int(items[gi].get("rank", 10**9)),
            -float(items[gi].get("score", 0.0)),
        ),
    )


def _write_query_outputs(
    fj,
    ft,
    fg,
    run_name: str,
    qid: str,
    qtext: str,
    pos_id: str,
    items: Sequence[dict],
    order: Sequence[int],
    agg_scores: Dict[int, float],
    judgments: List[Dict],
    window_k: int,
    stride: int,
) -> None:
    fj.write(
        json.dumps(
            {
                "method": "rankk_listwise_borda",
                "run_name": run_name,
                "query_id": qid,
                "positive_id": pos_id,
                "query_text": qtext,
                "k": 50,
                "window_k": window_k,
                "stride": stride,
                "final": [
                    {
                        "rerank_rank": r,
                        "agg_score": float(agg_scores[gi]),
                        "key": items[gi]["key"],
                        "docid": items[gi]["docid"],
                        "text": items[gi].get("text", ""),
                        "orig_rank": int(items[gi].get("rank", r)),
                        "orig_score": float(items[gi].get("score", 0.0)),
                    }
                    for r, gi in enumerate(order, start=1)
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    for r, gi in enumerate(order, start=1):
        ft.write(f"{qid} Q0 {items[gi]['docid']} {r} {float(agg_scores[gi]):.6f} {run_name}\n")

    fg.write(
        json.dumps(
            {
                "query_id": qid,
                "query_text": qtext,
                "window_k": window_k,
                "stride": stride,
                "batches": judgments,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def evaluate_reranked_jsonl(
    path: Path,
    ks: Iterable[int] = (1, 5, 10, 20, 50),
    *,
    verbose_examples: int = 5,
) -> Dict:
    ks = sorted(set(int(k) for k in ks))
    total = 0
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    ranks_found: List[int] = []
    per_doc: Dict[Optional[str], Dict[str, List[float] | int]] = {}
    examples: List[Dict] = []

    for rec in _iter_reranked_json(path):
        total += 1
        pos_key: Optional[str] = rec.get("positive_id")
        final = rec.get("final", []) or []
        final_sorted = sorted(final, key=lambda it: int(it.get("rerank_rank", 10**9)))
        key_to_rank = {it.get("key"): idx + 1 for idx, it in enumerate(final_sorted)}
        pos_rank: Optional[int] = key_to_rank.get(pos_key)

        for k in ks:
            if pos_rank is not None and pos_rank <= k:
                hits[k] += 1

        if pos_rank is not None:
            rr_sum += 1.0 / pos_rank
            ranks_found.append(pos_rank)
        else:
            rr_sum += 0.0

        qdoc: Optional[str] = None
        if pos_rank is not None:
            for it in final_sorted:
                if it.get("key") == pos_key:
                    qdoc = it.get("docid")
                    break

        if qdoc not in per_doc:
            per_doc[qdoc] = {"num_queries": 0, "ranks": [], "rrs": []}
        per_doc[qdoc]["num_queries"] += 1
        if pos_rank is not None:
            per_doc[qdoc]["ranks"].append(float(pos_rank))
            per_doc[qdoc]["rrs"].append(1.0 / pos_rank)

        if len(examples) < verbose_examples:
            examples.append(
                {
                    "query_id": rec.get("query_id"),
                    "query_docid": qdoc,
                    "pos_rank": pos_rank,
                    "found_at": {k: (pos_rank is not None and pos_rank <= k) for k in ks},
                    "top_keys": [it.get("key") for it in final_sorted[:3]],
                }
            )

    metrics = {
        "num_queries": total,
        "hit_rate": {k: (hits[k] / total if total else float("nan")) for k in ks},
        "mrr": (rr_sum / total) if total else float("nan"),
        "mean_rank_found": _safe_mean(ranks_found),
        "queries_with_positive_in_topk": len(ranks_found),
    }

    metrics["per_doc"] = {
        did: {
            "num_queries": bucket["num_queries"],
            "avg_rank": _safe_mean(bucket["ranks"]),
            "mrr": _safe_mean(bucket["rrs"]),
        }
        for did, bucket in per_doc.items()
    }

    _print_rerank_report(metrics, ranks_found, ks, examples)
    return metrics


def evaluate_doc_full_recall_jsonl(
    path: Path,
    ks: Iterable[int] = (1, 5, 10, 20, 50),
    *,
    print_report: bool = True,
) -> Dict:
    ks = sorted(set(int(k) for k in ks))
    doc_hits: Dict[str, Dict[int, bool]] = {}
    doc_query_counts: Dict[str, int] = defaultdict(int)

    for rec in _iter_reranked_json(path):
        pos_key: Optional[str] = rec.get("positive_id")
        doc_id = _positive_id_to_docid(pos_key)
        if doc_id is None:
            continue

        final = rec.get("final", []) or []
        final_sorted = sorted(final, key=lambda it: int(it.get("rerank_rank", 10**9)))
        key_to_rank = {it.get("key"): idx + 1 for idx, it in enumerate(final_sorted)}
        pos_rank: Optional[int] = key_to_rank.get(pos_key)

        doc_query_counts[doc_id] += 1
        state = doc_hits.setdefault(doc_id, {k: True for k in ks})
        for k in ks:
            if not (pos_rank is not None and pos_rank <= k):
                state[k] = False

    num_docs = len(doc_query_counts)
    docs_with_all_queries = {
        k: sum(1 for state in doc_hits.values() if state.get(k, False))
        for k in ks
    }
    docs_with_all_queries_pct = {
        k: (docs_with_all_queries[k] / num_docs) if num_docs else float("nan")
        for k in ks
    }

    metrics = {
        "num_documents": num_docs,
        "docs_with_all_queries_in_topk": docs_with_all_queries,
        "docs_with_all_queries_in_topk_pct": docs_with_all_queries_pct,
        "queries_per_doc": dict(doc_query_counts),
    }

    if print_report:
        print("\n=== Document coverage (all queries within top-k) ===")
        for k in ks:
            count = docs_with_all_queries[k]
            if num_docs:
                pct = docs_with_all_queries_pct[k] * 100
                print(f"Docs Hit@{k:>2} (all queries): {count}/{num_docs} ({pct:.2f}%)")
            else:
                print(f"Docs Hit@{k:>2} (all queries): 0/0 (NaN)")

    return metrics


def _iter_reranked_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else float("nan")


def _print_rerank_report(
    metrics: Dict,
    ranks_found: Sequence[int],
    ks: Sequence[int],
    examples: Sequence[Dict],
) -> None:
    print("\n=== Macro metrics (from RERANKED top-k) ===")
    for k in ks:
        hit = metrics["hit_rate"][k]
        print(f"Hit@{k:>2} = {hit:.4f}" if metrics["num_queries"] else f"Hit@{k:>2} = NaN")
    mrr = metrics["mrr"]
    print(f"MRR     = {mrr:.4f}" if metrics["num_queries"] else "MRR     = NaN")
    mean_rank = metrics["mean_rank_found"]
    print(
        f"Mean rank (found) = {mean_rank:.2f}" if ranks_found else "Mean rank (found) = NaN"
    )
    print(
        "Queries with positive in top-k = "
        f"{metrics['queries_with_positive_in_topk']} of {metrics['num_queries']}"
    )

    print("\n=== Examples ===")
    for ex in examples:
        print(
            f"- {ex['query_id']} | doc={ex['query_docid']} | pos_rank={ex['pos_rank']} | "
            f"found={ex['found_at']}"
        )
        print(f"  top3: {ex['top_keys']}")


def _positive_id_to_docid(pos_id: Optional[str]) -> Optional[str]:
    if not pos_id:
        return None
    doc_id = pos_id[2:] if pos_id.startswith("p_") else pos_id
    if ":" in doc_id:
        doc_id = doc_id.rsplit(":", 1)[0]
    return doc_id or None
