from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from retriever.data import load_candidates_by_case, load_corpus, load_queries
from retriever.markup import SLOT_TOKEN
from retriever.models import DualEncoderRetriever

from .artifact import ModelArtifactRef, cleanup_model_artifact, resolve_model_artifact
from .metrics import QueryInfo, aggregate_metrics, compute_bucketed_metrics, compute_query_metrics


@dataclass(frozen=True)
class EvalRegime:
    name: str
    candidate_ids_by_query_idx: List[List[str]]


def _batch_iter(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def load_retriever_from_artifact(model_dir: Path, *, device: torch.device) -> Tuple[DualEncoderRetriever, Any]:
    wrapper_config_path = model_dir / "wrapper_config.json"
    wrapper = json.loads(wrapper_config_path.read_text(encoding="utf-8")) if wrapper_config_path.exists() else {}
    temperature = float(wrapper.get("temperature", 0.05))

    tokenizer = _load_tokenizer_with_compat(model_dir)
    slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
    if slot_token_id == tokenizer.unk_token_id:
        raise ValueError(f"{SLOT_TOKEN} is not present in the model tokenizer")

    encoder_config_dir = model_dir / "encoder_config"
    try:
        config = AutoConfig.from_pretrained(str(encoder_config_dir))
    except KeyError as e:
        raise RuntimeError(
            "Failed to load ModernBERT config; install transformers==4.49.0 (or newer) to run evaluation."
        ) from e
    encoder = AutoModel.from_config(config)
    encoder.resize_token_embeddings(len(tokenizer))

    retriever = DualEncoderRetriever(encoder=encoder, slot_token_id=slot_token_id, temperature=temperature)
    state_dict = load_file(str(model_dir / "model.safetensors"))
    retriever.load_state_dict(state_dict, strict=False)
    retriever.to(device)
    retriever.eval()
    return retriever, tokenizer


def _load_tokenizer_with_compat(model_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    except Exception:
        patched = _patch_tokenizer_json_for_legacy_tokenizers(model_dir)
        if not patched:
            raise
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)


def _patch_tokenizer_json_for_legacy_tokenizers(model_dir: Path) -> bool:
    tokenizer_json_path = model_dir / "tokenizer.json"
    if not tokenizer_json_path.exists():
        return False

    try:
        data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    merges = (data.get("model") or {}).get("merges")
    if not merges or not isinstance(merges, list) or not isinstance(merges[0], list):
        return False

    data["model"]["merges"] = [" ".join(pair) for pair in merges]
    backup_path = model_dir / "tokenizer.json.orig"
    if not backup_path.exists():
        backup_path.write_bytes(tokenizer_json_path.read_bytes())
    tokenizer_json_path.write_text(json.dumps(data), encoding="utf-8")
    return True


def encode_passages(
    retriever: DualEncoderRetriever,
    tokenizer,
    passage_texts: Sequence[str],
    *,
    batch_size: int,
    max_len_passage: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer.truncation_side = "right"
    all_vecs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_texts in _batch_iter(list(passage_texts), batch_size):
            toks = tokenizer(
                batch_texts,
                truncation=True,
                max_length=int(max_len_passage),
                padding=True,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            vecs = retriever.encode_passages(toks["input_ids"], toks["attention_mask"])
            all_vecs.append(vecs.detach().cpu())
    return torch.cat(all_vecs, dim=0)


def encode_queries(
    retriever: DualEncoderRetriever,
    tokenizer,
    query_texts: Sequence[str],
    *,
    batch_size: int,
    max_len_query: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer.truncation_side = "left"
    all_vecs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_texts in _batch_iter(list(query_texts), batch_size):
            toks = tokenizer(
                batch_texts,
                truncation=True,
                max_length=int(max_len_query),
                padding=True,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            vecs = retriever.encode_queries(toks["input_ids"], toks["attention_mask"])
            all_vecs.append(vecs.detach().cpu())
    return torch.cat(all_vecs, dim=0)


def _build_all_cases_candidate_ids(
    corpus_by_passage_id: Dict[str, Any],
    *,
    processed_dir: Path,
    include_train: bool,
) -> List[str]:
    if include_train:
        return sorted(list(corpus_by_passage_id.keys()))

    val_cases = set((processed_dir / "splits" / "val_cases.txt").read_text(encoding="utf-8").split())
    test_cases = set((processed_dir / "splits" / "test_cases.txt").read_text(encoding="utf-8").split())
    allowed = val_cases | test_cases
    return sorted([pid for pid, passage in corpus_by_passage_id.items() if passage.doc_id in allowed])


def _random_ranking(
    candidate_ids: Sequence[str],
    *,
    seed: int,
    k: int,
) -> List[str]:
    rng = random.Random(int(seed))
    if len(candidate_ids) <= k:
        ids = list(candidate_ids)
        rng.shuffle(ids)
        return ids
    sampled = list(candidate_ids)
    rng.shuffle(sampled)
    return sampled[:k]


def _stable_seed(text: str) -> int:
    acc = 1469598103934665603
    for ch in str(text):
        acc ^= ord(ch)
        acc *= 1099511628211
        acc &= 0xFFFFFFFFFFFFFFFF
    if acc >= 2**63:
        acc -= 2**64
    return int(acc)


def run_eval(
    *,
    processed_dir: Path,
    model_artifact: ModelArtifactRef,
    output_dir: Path,
    split: str,
    max_len_query: int,
    max_len_passage: int,
    query_batch_size: int,
    passage_batch_size: int,
    ks: Sequence[int],
    random_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    corpus_by_passage_id = load_corpus(processed_dir)
    candidates_by_case = load_candidates_by_case(processed_dir)
    queries = load_queries(processed_dir, split)

    passage_ids = list(corpus_by_passage_id.keys())
    passage_texts = [corpus_by_passage_id[pid].text for pid in passage_ids]
    passage_labels = {pid: corpus_by_passage_id[pid].label for pid in passage_ids}
    passage_text_by_id = {pid: corpus_by_passage_id[pid].text for pid in passage_ids}
    passage_idx_by_id = {pid: i for i, pid in enumerate(passage_ids)}

    query_infos: List[QueryInfo] = [
        QueryInfo(
            query_id=q.query_id,
            doc_id=q.doc_id,
            query_text=q.query_text,
            gold_passage_ids=list(q.positive_passage_ids),
            gold_labels=list(q.positive_labels),
        )
        for q in queries
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever, tokenizer = load_retriever_from_artifact(model_artifact.local_dir, device=device)

    passage_vecs = encode_passages(
        retriever,
        tokenizer,
        passage_texts,
        batch_size=passage_batch_size,
        max_len_passage=max_len_passage,
        device=device,
    )
    query_vecs = encode_queries(
        retriever,
        tokenizer,
        [q.query_text for q in queries],
        batch_size=query_batch_size,
        max_len_query=max_len_query,
        device=device,
    )

    scores = query_vecs @ passage_vecs.T

    max_k = int(max(ks))
    regimes: List[EvalRegime] = []

    same_case_candidates = [candidates_by_case.get(q.doc_id, []) for q in queries]
    regimes.append(EvalRegime(name="same_case", candidate_ids_by_query_idx=same_case_candidates))

    all_cases_train_val_test = [sorted(list(corpus_by_passage_id.keys())) for _ in queries]
    regimes.append(EvalRegime(name="all_cases_train_val_test", candidate_ids_by_query_idx=all_cases_train_val_test))

    all_cases_val_test_ids = _build_all_cases_candidate_ids(corpus_by_passage_id, processed_dir=processed_dir, include_train=False)
    all_cases_val_test = [all_cases_val_test_ids for _ in queries]
    regimes.append(EvalRegime(name="all_cases_val_test", candidate_ids_by_query_idx=all_cases_val_test))

    results: Dict[str, Any] = {
        "config": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_source": model_artifact.source,
            "model_dir": str(model_artifact.local_dir),
            "processed_dir": str(processed_dir),
            "split": split,
            "data_sha256": {
                "corpus.jsonl": _sha256_file(processed_dir / "corpus.jsonl"),
                f"queries/{split}.jsonl": _sha256_file(processed_dir / "queries" / f"{split}.jsonl"),
                "pools/candidates_by_case.json": _sha256_file(processed_dir / "pools" / "candidates_by_case.json"),
            },
            "max_len_query": int(max_len_query),
            "max_len_passage": int(max_len_passage),
            "query_batch_size": int(query_batch_size),
            "passage_batch_size": int(passage_batch_size),
            "k_values": [int(k) for k in ks],
            "random_seed": int(random_seed),
            "device": str(device),
        },
        "regimes": {},
    }

    examples_path = runs_dir / "topk_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as examples_f:
        for regime in regimes:
            candidate_ids_by_query = regime.candidate_ids_by_query_idx

            model_per_query_rows: List[Dict[str, float]] = []
            baseline_per_query_rows: List[Dict[str, float]] = []

            for qi, query_info in enumerate(query_infos):
                candidate_ids = [pid for pid in candidate_ids_by_query[qi] if pid in passage_idx_by_id]
                retrieved_ids: List[str] = []
                top_scores_list: List[float] = []
                if candidate_ids:
                    candidate_indices = torch.tensor(
                        [passage_idx_by_id[pid] for pid in candidate_ids], dtype=torch.long
                    )
                    candidate_scores = scores[qi, candidate_indices]
                    top_scores, top_pos = torch.topk(candidate_scores, k=min(max_k, candidate_scores.numel()))
                    retrieved_ids = [candidate_ids[int(i)] for i in top_pos.tolist()]
                    top_scores_list = [float(x) for x in top_scores.tolist()]

                model_metrics = compute_query_metrics(
                    retrieved_passage_ids=retrieved_ids,
                    gold_passage_ids=query_info.gold_passage_ids,
                    ks=ks,
                )
                model_per_query_rows.append(model_metrics)

                baseline_ids = _random_ranking(
                    candidate_ids,
                    seed=_stable_seed(f"{random_seed}:{regime.name}:{query_info.query_id}"),
                    k=min(max_k, len(candidate_ids)),
                )
                baseline_metrics = compute_query_metrics(
                    retrieved_passage_ids=baseline_ids,
                    gold_passage_ids=query_info.gold_passage_ids,
                    ks=ks,
                )
                baseline_per_query_rows.append(baseline_metrics)

                for system_name, system_ids, system_scores in (
                    ("model", retrieved_ids, top_scores_list),
                    ("random", baseline_ids, None),
                ):
                    topk_payload = []
                    for rank_idx, passage_id in enumerate(system_ids[:max_k], start=1):
                        score = (
                            float(system_scores[rank_idx - 1])
                            if system_scores is not None and rank_idx - 1 < len(system_scores)
                            else None
                        )
                        topk_payload.append(
                            {
                                "rank": int(rank_idx),
                                "passage_id": passage_id,
                                "score": score,
                                "label": passage_labels.get(passage_id, ""),
                                "is_gold": passage_id in set(query_info.gold_passage_ids),
                                "text": _truncate_text(passage_text_by_id.get(passage_id, ""), 240),
                            }
                        )

                    examples_f.write(
                        json.dumps(
                            {
                                "regime": regime.name,
                                "system": system_name,
                                "query_id": query_info.query_id,
                                "doc_id": query_info.doc_id,
                                "query_text": query_info.query_text,
                                "gold_passage_ids": list(query_info.gold_passage_ids),
                                "gold_labels": list(query_info.gold_labels),
                                "candidate_pool_size": int(len(candidate_ids)),
                                "topk": topk_payload,
                            }
                        )
                        + "\n"
                    )

            regime_out: Dict[str, Any] = {}
            regime_out["model"] = {
                "global": aggregate_metrics(model_per_query_rows),
                "breakdowns": compute_bucketed_metrics(queries=query_infos, per_query_metrics=model_per_query_rows, ks=ks),
            }
            regime_out["random_baseline"] = {
                "global": aggregate_metrics(baseline_per_query_rows),
                "breakdowns": compute_bucketed_metrics(queries=query_infos, per_query_metrics=baseline_per_query_rows, ks=ks),
            }
            results["regimes"][regime.name] = regime_out

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(results["config"], indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report_md(results), encoding="utf-8")


def _render_report_md(results: Dict[str, Any]) -> str:
    config = results.get("config") or {}
    lines: List[str] = []
    lines.append("# Retriever evaluation report")
    lines.append("")
    lines.append(f"- Model source: `{config.get('model_source', '')}`")
    lines.append(f"- Processed dir: `{config.get('processed_dir', '')}`")
    lines.append(f"- Split: `{config.get('split', '')}`")
    lines.append(f"- K values: `{config.get('k_values', [])}`")
    lines.append("")

    def pick(metrics: Dict[str, Any], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return f"{float(value):.4f}"
        return str(value)

    report_keys = [
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "mrr_at_20",
        "set_recall_at_20",
        "exact_set_match_at_20",
    ]

    for regime_name, regime in (results.get("regimes") or {}).items():
        lines.append(f"## {regime_name}")
        lines.append("")
        model_global = (regime.get("model") or {}).get("global") or {}
        rand_global = (regime.get("random_baseline") or {}).get("global") or {}
        lines.append("| metric | model | random |")
        lines.append("|---|---:|---:|")
        for key in report_keys:
            lines.append(f"| `{key}` | {pick(model_global, key)} | {pick(rand_global, key)} |")
        lines.append("")
        lines.append("Breakdowns: see `results.json`.")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_s3_uri", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--max_len_query", type=int, default=4096)
    parser.add_argument("--max_len_passage", type=int, default=600)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--passage_batch_size", type=int, default=256)
    parser.add_argument("--k_values", type=str, default="1,5,10,20,50")
    parser.add_argument("--random_seed", type=int, default=17)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is None:
        output_dir = processed_dir.parent / "eval_runs" / time.strftime("%Y%m%d_%H%M%S")

    ks = tuple(int(x.strip()) for x in str(args.k_values).split(",") if x.strip())

    model_artifact = resolve_model_artifact(
        model_dir=args.model_dir,
        model_s3_uri=args.model_s3_uri,
        work_dir=args.work_dir,
    )
    try:
        run_eval(
            processed_dir=processed_dir,
            model_artifact=model_artifact,
            output_dir=output_dir,
            split=args.split,
            max_len_query=args.max_len_query,
            max_len_passage=args.max_len_passage,
            query_batch_size=args.query_batch_size,
            passage_batch_size=args.passage_batch_size,
            ks=ks,
            random_seed=args.random_seed,
        )
    finally:
        cleanup_model_artifact(model_artifact)


if __name__ == "__main__":
    main()
