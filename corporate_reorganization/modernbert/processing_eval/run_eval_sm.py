from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retriever.data import CorpusPassage, QueryExample, load_candidates_by_case, load_corpus, load_queries
from retriever.markup import SLOT_TOKEN, all_markup_tokens
from retriever.models import DualEncoderRetriever
from retriever_eval.artifact import ModelArtifactRef, cleanup_model_artifact, resolve_model_artifact
from retriever_eval.run import encode_passages, encode_queries, load_retriever_from_artifact


def _parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _stable_seed(text: str) -> int:
    acc = 1469598103934665603
    for ch in str(text):
        acc ^= ord(ch)
        acc *= 1099511628211
        acc &= 0xFFFFFFFFFFFFFFFF
    if acc >= 2**63:
        acc -= 2**64
    return int(acc)


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _candidate_ids_by_query(
    *,
    queries: Sequence[QueryExample],
    candidates_by_case: Dict[str, List[str]],
) -> List[List[str]]:
    positive_ids_by_doc_id: Dict[str, set[str]] = {}
    for query in queries:
        positive_ids_by_doc_id.setdefault(query.doc_id, set()).update(query.positive_passage_ids)

    candidate_ids_by_query: List[List[str]] = []
    for query in queries:
        excluded_ids = positive_ids_by_doc_id.get(query.doc_id, set()) - set(query.positive_passage_ids)
        doc_candidates = candidates_by_case.get(query.doc_id, [])
        candidate_ids_by_query.append([pid for pid in doc_candidates if pid not in excluded_ids])
    return candidate_ids_by_query


def _best_hit_rank(ranked_passage_ids: Sequence[str], gold_set: set[str]) -> int:
    for rank_idx, pid in enumerate(ranked_passage_ids, start=1):
        if pid in gold_set:
            return rank_idx
    return 0


def _recall_at_k(hit_ranks: Sequence[int], k: int) -> float:
    if not hit_ranks:
        return 0.0
    return float(sum(1 for r in hit_ranks if 1 <= r <= k)) / float(len(hit_ranks))


def _mrr(hit_ranks: Sequence[int]) -> float:
    if not hit_ranks:
        return 0.0
    return float(sum(1.0 / r if r > 0 else 0.0 for r in hit_ranks)) / float(len(hit_ranks))


def _random_ranking(candidate_ids: Sequence[str], *, seed: int) -> List[str]:
    rng = random.Random(int(seed))
    ids = list(candidate_ids)
    rng.shuffle(ids)
    return ids


def _evaluate_with_embeddings(
    *,
    system_name: str,
    queries: Sequence[QueryExample],
    candidate_ids_by_query: Sequence[Sequence[str]],
    corpus_by_passage_id: Dict[str, CorpusPassage],
    passage_ids: Sequence[str],
    passage_idx_by_id: Dict[str, int],
    scores: torch.Tensor,
    ks: Sequence[int],
    rankings_path: Path,
) -> Dict[str, Any]:
    hit_ranks: List[int] = []
    candidate_sizes: List[int] = []
    retrieval_losses: List[float] = []

    with rankings_path.open("a", encoding="utf-8") as f:
        for qi, query in enumerate(queries):
            candidate_ids = [pid for pid in candidate_ids_by_query[qi] if pid in passage_idx_by_id]
            candidate_indices = [passage_idx_by_id[pid] for pid in candidate_ids]
            if not candidate_indices:
                continue
            candidate_sizes.append(len(candidate_indices))

            candidate_scores = scores[qi, torch.tensor(candidate_indices, dtype=torch.long)]
            ranked_idx = torch.argsort(candidate_scores, descending=True)
            ranked_passage_ids = [candidate_ids[int(i)] for i in ranked_idx.tolist()]
            ranked_scores = [float(candidate_scores[int(i)]) for i in ranked_idx.tolist()]

            gold_set = set(query.positive_passage_ids)
            best_rank = _best_hit_rank(ranked_passage_ids, gold_set)
            hit_ranks.append(best_rank)

            pos_mask_list = [pid in gold_set for pid in candidate_ids]
            if any(pos_mask_list):
                pos_mask = torch.tensor(pos_mask_list, dtype=torch.bool)
                numerator = torch.logsumexp(candidate_scores[pos_mask], dim=0)
                denominator = torch.logsumexp(candidate_scores, dim=0)
                retrieval_losses.append(float((-(numerator - denominator)).item()))

            f.write(
                json.dumps(
                    {
                        "system": system_name,
                        "query_id": query.query_id,
                        "doc_id": query.doc_id,
                        "candidate_pool_size": int(len(candidate_ids)),
                        "gold_passage_ids": list(query.positive_passage_ids),
                        "ranked_candidates": [
                            {
                                "rank": int(rank_idx),
                                "passage_id": passage_id,
                                "score": float(score),
                                "label": corpus_by_passage_id[passage_id].label,
                                "text": corpus_by_passage_id[passage_id].text,
                            }
                            for rank_idx, (passage_id, score) in enumerate(
                                zip(ranked_passage_ids, ranked_scores), start=1
                            )
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics: Dict[str, float] = {"eval_num_queries": float(len(hit_ranks))}
    for k in ks:
        metrics[f"eval_recall_at_{int(k)}"] = _recall_at_k(hit_ranks, int(k))
    metrics["eval_mrr"] = _mrr(hit_ranks)
    metrics["eval_avg_candidates"] = float(sum(candidate_sizes) / max(1, len(candidate_sizes)))
    metrics["eval_retrieval_loss"] = float(sum(retrieval_losses) / max(1, len(retrieval_losses)))

    return {"metrics": metrics, "query_count": int(len(queries)), "corpus_count": int(len(passage_ids))}


def _evaluate_random(
    *,
    system_name: str,
    queries: Sequence[QueryExample],
    candidate_ids_by_query: Sequence[Sequence[str]],
    corpus_by_passage_id: Dict[str, CorpusPassage],
    ks: Sequence[int],
    random_seed: int,
    rankings_path: Path,
) -> Dict[str, Any]:
    hit_ranks: List[int] = []
    candidate_sizes: List[int] = []
    retrieval_losses: List[float] = []

    with rankings_path.open("a", encoding="utf-8") as f:
        for query, candidate_ids in zip(queries, candidate_ids_by_query):
            candidate_ids = list(candidate_ids)
            candidate_sizes.append(len(candidate_ids))

            rng = random.Random(_stable_seed(f"{random_seed}:{system_name}:{query.query_id}"))
            random_scores = [float(rng.random()) for _ in candidate_ids]
            ranked_idx = sorted(range(len(candidate_ids)), key=lambda i: random_scores[i], reverse=True)
            ranked_ids = [candidate_ids[i] for i in ranked_idx]
            ranked_scores = [random_scores[i] for i in ranked_idx]
            gold_set = set(query.positive_passage_ids)
            best_rank = _best_hit_rank(ranked_ids, gold_set)
            hit_ranks.append(best_rank)

            pos_mask = torch.tensor([pid in gold_set for pid in candidate_ids], dtype=torch.bool)
            if bool(pos_mask.any()):
                cand_scores = torch.tensor(random_scores, dtype=torch.float)
                numerator = torch.logsumexp(cand_scores[pos_mask], dim=0)
                denominator = torch.logsumexp(cand_scores, dim=0)
                retrieval_losses.append(float((-(numerator - denominator)).item()))

            f.write(
                json.dumps(
                    {
                        "system": system_name,
                        "query_id": query.query_id,
                        "doc_id": query.doc_id,
                        "candidate_pool_size": int(len(candidate_ids)),
                        "gold_passage_ids": list(query.positive_passage_ids),
                        "ranked_candidates": [
                            {
                                "rank": int(rank_idx),
                                "passage_id": passage_id,
                                "score": float(ranked_scores[rank_idx - 1]),
                                "label": corpus_by_passage_id[passage_id].label,
                                "text": corpus_by_passage_id[passage_id].text,
                            }
                            for rank_idx, passage_id in enumerate(ranked_ids, start=1)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics: Dict[str, float] = {"eval_num_queries": float(len(hit_ranks))}
    for k in ks:
        metrics[f"eval_recall_at_{int(k)}"] = _recall_at_k(hit_ranks, int(k))
    metrics["eval_mrr"] = _mrr(hit_ranks)
    metrics["eval_avg_candidates"] = float(sum(candidate_sizes) / max(1, len(candidate_sizes)))
    metrics["eval_retrieval_loss"] = float(sum(retrieval_losses) / max(1, len(retrieval_losses)))

    return {"metrics": metrics, "query_count": int(len(queries))}


def _load_base_modernbert(
    *,
    base_model_name_or_path: str,
    temperature: float,
    device: torch.device,
) -> Tuple[DualEncoderRetriever, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": all_markup_tokens()})
    slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
    if slot_token_id == tokenizer.unk_token_id:
        raise ValueError(f"{SLOT_TOKEN} is not present in base tokenizer vocab")

    encoder = AutoModel.from_pretrained(base_model_name_or_path)
    encoder.resize_token_embeddings(len(tokenizer))
    retriever = DualEncoderRetriever(encoder=encoder, slot_token_id=slot_token_id, temperature=float(temperature))
    retriever.to(device)
    retriever.eval()
    return retriever, tokenizer


def _render_report_md(results: Dict[str, Any]) -> str:
    config = results.get("config") or {}
    systems = results.get("systems") or {}

    def fmt(metrics: Dict[str, Any], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return f"{float(value):.4f}"
        return str(value)

    lines: List[str] = []
    lines.append("# Retriever evaluation report (SageMaker Processing)")
    lines.append("")
    lines.append(f"- Processed dir: `{config.get('processed_dir', '')}`")
    lines.append(f"- Split: `{config.get('split', '')}`")
    lines.append(f"- K values: `{config.get('k_values', [])}`")
    lines.append("")

    report_keys = [
        "eval_recall_at_1",
        "eval_recall_at_5",
        "eval_recall_at_10",
        "eval_recall_at_20",
        "eval_mrr",
        "eval_avg_candidates",
        "eval_retrieval_loss",
    ]

    lines.append("| system | " + " | ".join(f"`{k}`" for k in report_keys) + " |")
    lines.append("|---" + "|---:" * len(report_keys) + "|")
    for system_name, payload in systems.items():
        metrics = (payload or {}).get("metrics") or {}
        lines.append("| " + str(system_name) + " | " + " | ".join(fmt(metrics, k) for k in report_keys) + " |")
    lines.append("")
    lines.append("Rankings: see `runs/rankings.jsonl`.")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--fine_tuned_model_s3_uri", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--base_temperature", type=float, default=0.05)
    parser.add_argument("--work_dir", type=str, default=None)

    parser.add_argument("--max_len_query", type=int, default=4096)
    parser.add_argument("--max_len_passage", type=int, default=600)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--passage_batch_size", type=int, default=256)
    parser.add_argument("--k_values", type=str, default="1,5,10,20")
    parser.add_argument("--random_seed", type=int, default=17)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runs_dir = output_dir / "runs"
    _ensure_dir(runs_dir)

    corpus_by_passage_id = load_corpus(processed_dir)
    candidates_by_case = load_candidates_by_case(processed_dir)
    queries = load_queries(processed_dir, args.split)

    candidate_ids_by_query = _candidate_ids_by_query(queries=queries, candidates_by_case=candidates_by_case)

    passage_ids = list(corpus_by_passage_id.keys())
    passage_texts = [corpus_by_passage_id[pid].text for pid in passage_ids]
    passage_idx_by_id = {pid: i for i, pid in enumerate(passage_ids)}

    rankings_path = runs_dir / "rankings.jsonl"
    rankings_path.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    systems: Dict[str, Any] = {}
    model_artifact: Optional[ModelArtifactRef] = None
    try:
        model_artifact = resolve_model_artifact(
            model_dir=None,
            model_s3_uri=str(args.fine_tuned_model_s3_uri),
            work_dir=args.work_dir,
        )
        fine_tuned_retriever, fine_tuned_tokenizer = load_retriever_from_artifact(
            model_artifact.local_dir, device=device
        )

        fine_passage_vecs = encode_passages(
            fine_tuned_retriever,
            fine_tuned_tokenizer,
            passage_texts,
            batch_size=int(args.passage_batch_size),
            max_len_passage=int(args.max_len_passage),
            device=device,
        )
        fine_query_vecs = encode_queries(
            fine_tuned_retriever,
            fine_tuned_tokenizer,
            [q.query_text for q in queries],
            batch_size=int(args.query_batch_size),
            max_len_query=int(args.max_len_query),
            device=device,
        )
        fine_scores = fine_query_vecs @ fine_passage_vecs.T

        systems["fine_tuned"] = _evaluate_with_embeddings(
            system_name="fine_tuned",
            queries=queries,
            candidate_ids_by_query=candidate_ids_by_query,
            corpus_by_passage_id=corpus_by_passage_id,
            passage_ids=passage_ids,
            passage_idx_by_id=passage_idx_by_id,
            scores=fine_scores,
            ks=_parse_csv_ints(args.k_values),
            rankings_path=rankings_path,
        )

        base_retriever, base_tokenizer = _load_base_modernbert(
            base_model_name_or_path=str(args.base_model_name_or_path),
            temperature=float(args.base_temperature),
            device=device,
        )
        base_passage_vecs = encode_passages(
            base_retriever,
            base_tokenizer,
            passage_texts,
            batch_size=int(args.passage_batch_size),
            max_len_passage=int(args.max_len_passage),
            device=device,
        )
        base_query_vecs = encode_queries(
            base_retriever,
            base_tokenizer,
            [q.query_text for q in queries],
            batch_size=int(args.query_batch_size),
            max_len_query=int(args.max_len_query),
            device=device,
        )
        base_scores = base_query_vecs @ base_passage_vecs.T

        systems["base_modernbert"] = _evaluate_with_embeddings(
            system_name="base_modernbert",
            queries=queries,
            candidate_ids_by_query=candidate_ids_by_query,
            corpus_by_passage_id=corpus_by_passage_id,
            passage_ids=passage_ids,
            passage_idx_by_id=passage_idx_by_id,
            scores=base_scores,
            ks=_parse_csv_ints(args.k_values),
            rankings_path=rankings_path,
        )

        systems["random_baseline"] = _evaluate_random(
            system_name="random_baseline",
            queries=queries,
            candidate_ids_by_query=candidate_ids_by_query,
            corpus_by_passage_id=corpus_by_passage_id,
            ks=_parse_csv_ints(args.k_values),
            random_seed=int(args.random_seed),
            rankings_path=rankings_path,
        )
    finally:
        if model_artifact is not None:
            cleanup_model_artifact(model_artifact)

    config = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processed_dir": str(processed_dir),
        "split": str(args.split),
        "k_values": _parse_csv_ints(args.k_values),
        "max_len_query": int(args.max_len_query),
        "max_len_passage": int(args.max_len_passage),
        "query_batch_size": int(args.query_batch_size),
        "passage_batch_size": int(args.passage_batch_size),
        "device": str(device),
        "models": {
            "fine_tuned_model_s3_uri": str(args.fine_tuned_model_s3_uri),
            "base_model_name_or_path": str(args.base_model_name_or_path),
            "base_temperature": float(args.base_temperature),
        },
        "data_sha256": {
            "corpus.jsonl": _sha256_file(processed_dir / "corpus.jsonl"),
            f"queries/{args.split}.jsonl": _sha256_file(processed_dir / "queries" / f"{args.split}.jsonl"),
            "pools/candidates_by_case.json": _sha256_file(processed_dir / "pools" / "candidates_by_case.json"),
        },
    }

    results = {"config": config, "systems": systems}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report_md(results), encoding="utf-8")

    for system_name, payload in systems.items():
        metrics = (payload or {}).get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"SM_METRIC {system_name}.{key}={float(value)}")


if __name__ == "__main__":
    main()
