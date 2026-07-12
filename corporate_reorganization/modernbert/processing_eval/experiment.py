from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import torch
import torch.nn.functional as torch_nn_func
from transformers import AutoConfig, AutoModel, AutoTokenizer

from retriever.data import CorpusPassage, QueryExample, load_candidates_by_case, load_corpus, load_queries, load_split_doc_ids
from retriever.markup import SLOT_TOKEN, all_markup_tokens
from retriever.query_views import (
    QUERY_VIEW_FLAT_MASKED,
    QUERY_VIEW_FLAT_PLAIN,
    QUERY_VIEW_STRUCTURED,
    normalize_query_view,
    select_query_text,
)
from retriever.regimes import (
    REGIME_GLOBAL_SPLIT,
    REGIME_SAME_CASE_FULL,
    REGIME_SAME_CASE_LEGACY,
    build_candidate_ids_by_query,
    build_split_passage_ids,
    normalize_candidate_regime,
)
from retriever.models import DualEncoderRetriever
from retriever_eval.artifact import ModelArtifactRef, cleanup_model_artifact, resolve_model_artifact
from retriever_eval.metrics import QueryInfo, aggregate_metrics, compute_bucketed_metrics, compute_query_metrics
from retriever_eval.run import encode_passages, encode_queries, load_retriever_from_artifact

SYSTEM_TYPE_BM25 = "bm25_pyserini"
SYSTEM_TYPE_MODERNBERT_BASE = "modernbert_base"
SYSTEM_TYPE_MODERNBERT_ARTIFACT = "modernbert_artifact"
SYSTEM_TYPE_OPEN_DENSE = "open_dense"
SYSTEM_TYPE_COHERE = "cohere_embed"

DEFAULT_OPEN_DENSE_MODEL = "intfloat/e5-base-v2"
DEFAULT_COHERE_MODEL = "embed-v4.0"
DEFAULT_BM25_K1 = 0.9
DEFAULT_BM25_B = 0.4


@dataclass(frozen=True)
class SystemSpec:
    name: str
    system_type: str
    query_view: str
    model_name_or_path: Optional[str] = None
    model_dir: Optional[str] = None
    model_s3_uri: Optional[str] = None
    work_dir: Optional[str] = None
    temperature: float = 0.05
    query_prefix: str = ""
    passage_prefix: str = ""
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    cohere_model_name: str = DEFAULT_COHERE_MODEL
    cohere_api_key_env: str = "COHERE_API_KEY"
    cohere_output_dimension: Optional[int] = None

    def normalized(self) -> "SystemSpec":
        return SystemSpec(
            name=str(self.name).strip(),
            system_type=str(self.system_type).strip(),
            query_view=normalize_query_view(self.query_view),
            model_name_or_path=self.model_name_or_path,
            model_dir=self.model_dir,
            model_s3_uri=self.model_s3_uri,
            work_dir=self.work_dir,
            temperature=float(self.temperature),
            query_prefix=str(self.query_prefix),
            passage_prefix=str(self.passage_prefix),
            bm25_k1=float(self.bm25_k1),
            bm25_b=float(self.bm25_b),
            cohere_model_name=str(self.cohere_model_name),
            cohere_api_key_env=str(self.cohere_api_key_env),
            cohere_output_dimension=(
                int(self.cohere_output_dimension) if self.cohere_output_dimension is not None else None
            ),
        )


@dataclass(frozen=True)
class FullRankingResult:
    ranked_passage_ids_by_query: List[List[str]]
    ranked_scores_by_query: List[List[float]]
    metadata: Dict[str, Any]


def build_default_system_specs(
    *,
    structured_model_s3_uri: Optional[str] = None,
    flat_model_s3_uri: Optional[str] = None,
    base_model_name_or_path: str = "answerdotai/ModernBERT-base",
    open_dense_model_name_or_path: str = DEFAULT_OPEN_DENSE_MODEL,
    include_base_modernbert_flat: bool = True,
    include_cohere_flat: bool = False,
    cohere_model_name: str = DEFAULT_COHERE_MODEL,
) -> List[SystemSpec]:
    systems: List[SystemSpec] = [
        SystemSpec(
            name="bm25_flat",
            system_type=SYSTEM_TYPE_BM25,
            query_view=QUERY_VIEW_FLAT_PLAIN,
        ),
        SystemSpec(
            name="dense_open_flat",
            system_type=SYSTEM_TYPE_OPEN_DENSE,
            query_view=QUERY_VIEW_FLAT_PLAIN,
            model_name_or_path=open_dense_model_name_or_path,
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
    ]
    if include_base_modernbert_flat:
        systems.append(
            SystemSpec(
                name="base_modernbert_flat",
                system_type=SYSTEM_TYPE_MODERNBERT_BASE,
                query_view=QUERY_VIEW_FLAT_MASKED,
                model_name_or_path=base_model_name_or_path,
            )
        )
    systems.append(
        SystemSpec(
            name="fine_tuned_flat",
            system_type=SYSTEM_TYPE_MODERNBERT_ARTIFACT,
            query_view=QUERY_VIEW_FLAT_MASKED,
            model_s3_uri=flat_model_s3_uri,
        )
    )
    systems.append(
        SystemSpec(
            name="fine_tuned_structured",
            system_type=SYSTEM_TYPE_MODERNBERT_ARTIFACT,
            query_view=QUERY_VIEW_STRUCTURED,
            model_s3_uri=structured_model_s3_uri,
        )
    )
    if include_cohere_flat:
        systems.append(
            SystemSpec(
                name="cohere_embed4_flat",
                system_type=SYSTEM_TYPE_COHERE,
                query_view=QUERY_VIEW_FLAT_PLAIN,
                cohere_model_name=cohere_model_name,
            )
        )
    return [system.normalized() for system in systems]


def run_retrieval_experiment(
    *,
    processed_dir: Path,
    output_dir: Path,
    split: str,
    systems: Sequence[SystemSpec],
    regimes: Sequence[str] = (
        REGIME_SAME_CASE_LEGACY,
        REGIME_SAME_CASE_FULL,
        REGIME_GLOBAL_SPLIT,
    ),
    max_len_query: int = 4096,
    max_len_passage: int = 600,
    query_batch_size: int = 64,
    passage_batch_size: int = 256,
    ks: Sequence[int] = (1, 5, 10, 20),
    random_seed: int = 17,
    write_rankings_top_n: Optional[int] = None,
) -> Dict[str, Any]:
    processed_dir = Path(processed_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    systems = [system.normalized() for system in systems]
    if not systems:
        raise ValueError("systems must contain at least one SystemSpec")

    normalized_regimes = [normalize_candidate_regime(regime) for regime in regimes]

    corpus_by_passage_id = load_corpus(processed_dir)
    candidates_by_case = load_candidates_by_case(processed_dir)
    queries = load_queries(processed_dir, split)
    split_doc_ids = load_split_doc_ids(processed_dir, split)
    if not queries:
        raise ValueError(f"No queries found for split={split} under {processed_dir}")
    if not split_doc_ids:
        raise ValueError(f"No split doc ids found for split={split} under {processed_dir}")

    split_passage_ids = build_split_passage_ids(
        corpus_by_passage_id=corpus_by_passage_id,
        split_doc_ids=split_doc_ids,
    )
    if not split_passage_ids:
        raise ValueError(f"No passages found for split={split} using doc ids {split_doc_ids}")

    split_passage_idx_by_id = {passage_id: idx for idx, passage_id in enumerate(split_passage_ids)}
    split_passage_texts = [corpus_by_passage_id[pid].text for pid in split_passage_ids]

    candidate_ids_by_query_by_regime = {
        regime: build_candidate_ids_by_query(
            queries=queries,
            corpus_by_passage_id=corpus_by_passage_id,
            candidates_by_case=candidates_by_case,
            split_doc_ids=split_doc_ids,
            regime_name=regime,
        )
        for regime in normalized_regimes
    }

    query_infos: List[QueryInfo] = [
        QueryInfo(
            query_id=query.query_id,
            doc_id=query.doc_id,
            query_text=query.query_text,
            gold_passage_ids=list(query.positive_passage_ids),
            gold_labels=list(query.positive_labels),
        )
        for query in queries
    ]

    config = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "split": str(split),
        "k_values": [int(k) for k in ks],
        "max_len_query": int(max_len_query),
        "max_len_passage": int(max_len_passage),
        "query_batch_size": int(query_batch_size),
        "passage_batch_size": int(passage_batch_size),
        "random_seed": int(random_seed),
        "split_doc_ids": list(split_doc_ids),
        "split_passage_count": int(len(split_passage_ids)),
        "systems": [asdict(system) for system in systems],
        "regimes": list(normalized_regimes),
        "data_sha256": {
            "corpus.jsonl": _sha256_file(processed_dir / "corpus.jsonl"),
            f"queries/{split}.jsonl": _sha256_file(processed_dir / "queries" / f"{split}.jsonl"),
            "pools/candidates_by_case.json": _sha256_file(processed_dir / "pools" / "candidates_by_case.json"),
            f"splits/{split}_cases.txt": _sha256_file(processed_dir / "splits" / f"{split}_cases.txt"),
        },
    }

    results: Dict[str, Any] = {
        "config": config,
        "systems": {},
        "regimes": {regime: {"systems": {}} for regime in normalized_regimes},
    }

    rankings_path = runs_dir / "rankings.jsonl"
    rankings_path.write_text("", encoding="utf-8")

    for system in systems:
        ranking = _run_system(
            system=system,
            queries=queries,
            split_passage_ids=split_passage_ids,
            split_passage_texts=split_passage_texts,
            split_passage_idx_by_id=split_passage_idx_by_id,
            processed_dir=processed_dir,
            max_len_query=max_len_query,
            max_len_passage=max_len_passage,
            query_batch_size=query_batch_size,
            passage_batch_size=passage_batch_size,
        )
        results["systems"][system.name] = {
            "system_type": system.system_type,
            "query_view": system.query_view,
            "metadata": ranking.metadata,
        }

        for regime_name in normalized_regimes:
            regime_metrics = _evaluate_full_ranking_under_regime(
                system=system,
                regime_name=regime_name,
                ranking=ranking,
                queries=queries,
                query_infos=query_infos,
                candidate_ids_by_query=candidate_ids_by_query_by_regime[regime_name],
                ks=ks,
                rankings_path=rankings_path,
                write_rankings_top_n=write_rankings_top_n,
            )
            results["regimes"][regime_name]["systems"][system.name] = regime_metrics

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report_md(results, ks=ks), encoding="utf-8")
    return results


def _run_system(
    *,
    system: SystemSpec,
    queries: Sequence[QueryExample],
    split_passage_ids: Sequence[str],
    split_passage_texts: Sequence[str],
    split_passage_idx_by_id: Dict[str, int],
    processed_dir: Path,
    max_len_query: int,
    max_len_passage: int,
    query_batch_size: int,
    passage_batch_size: int,
) -> FullRankingResult:
    if system.system_type == SYSTEM_TYPE_BM25:
        return _run_bm25_system(
            system=system,
            queries=queries,
            split_passage_ids=split_passage_ids,
            split_passage_texts=split_passage_texts,
            work_dir=_resolve_work_dir(system),
        )

    if system.system_type == SYSTEM_TYPE_COHERE:
        return _run_cohere_system(
            system=system,
            queries=queries,
            split_passage_ids=split_passage_ids,
            split_passage_texts=split_passage_texts,
            max_len_query=max_len_query,
            max_len_passage=max_len_passage,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if system.system_type == SYSTEM_TYPE_OPEN_DENSE:
        return _run_open_dense_system(
            system=system,
            queries=queries,
            split_passage_ids=split_passage_ids,
            split_passage_texts=split_passage_texts,
            model_name_or_path=_require_model_name_or_path(system),
            device=device,
            max_len_query=max_len_query,
            max_len_passage=max_len_passage,
            query_batch_size=query_batch_size,
            passage_batch_size=passage_batch_size,
        )

    if system.system_type == SYSTEM_TYPE_MODERNBERT_BASE:
        retriever, tokenizer = _load_base_modernbert(
            base_model_name_or_path=_require_model_name_or_path(system),
            temperature=float(system.temperature),
            device=device,
        )
        return _run_modernbert_ranker(
            system=system,
            queries=queries,
            split_passage_ids=split_passage_ids,
            split_passage_texts=split_passage_texts,
            retriever=retriever,
            tokenizer=tokenizer,
            device=device,
            max_len_query=max_len_query,
            max_len_passage=max_len_passage,
            query_batch_size=query_batch_size,
            passage_batch_size=passage_batch_size,
            model_metadata={
                "model_source": _require_model_name_or_path(system),
                "device": str(device),
            },
        )

    if system.system_type == SYSTEM_TYPE_MODERNBERT_ARTIFACT:
        model_artifact = resolve_model_artifact(
            model_dir=system.model_dir,
            model_s3_uri=system.model_s3_uri,
            work_dir=system.work_dir,
        )
        try:
            retriever, tokenizer = load_retriever_from_artifact(model_artifact.local_dir, device=device)
            return _run_modernbert_ranker(
                system=system,
                queries=queries,
                split_passage_ids=split_passage_ids,
                split_passage_texts=split_passage_texts,
                retriever=retriever,
                tokenizer=tokenizer,
                device=device,
                max_len_query=max_len_query,
                max_len_passage=max_len_passage,
                query_batch_size=query_batch_size,
                passage_batch_size=passage_batch_size,
                model_metadata={
                    "model_source": model_artifact.source,
                    "model_dir": str(model_artifact.local_dir),
                    "device": str(device),
                },
            )
        finally:
            cleanup_model_artifact(model_artifact)

    raise ValueError(f"Unsupported system_type={system.system_type!r} for system {system.name}")


def _run_modernbert_ranker(
    *,
    system: SystemSpec,
    queries: Sequence[QueryExample],
    split_passage_ids: Sequence[str],
    split_passage_texts: Sequence[str],
    retriever: DualEncoderRetriever,
    tokenizer: Any,
    device: torch.device,
    max_len_query: int,
    max_len_passage: int,
    query_batch_size: int,
    passage_batch_size: int,
    model_metadata: Dict[str, Any],
) -> FullRankingResult:
    passage_vecs = encode_passages(
        retriever,
        tokenizer,
        split_passage_texts,
        batch_size=int(passage_batch_size),
        max_len_passage=int(max_len_passage),
        device=device,
    )
    query_texts = [select_query_text(query, query_view=system.query_view) for query in queries]
    query_vecs = encode_queries(
        retriever,
        tokenizer,
        query_texts,
        batch_size=int(query_batch_size),
        max_len_query=int(max_len_query),
        device=device,
    )
    scores = query_vecs @ passage_vecs.T
    return _ranking_from_score_matrix(
        scores=scores,
        split_passage_ids=split_passage_ids,
        metadata={
            **model_metadata,
            "query_view": system.query_view,
            "query_count": len(query_texts),
            "split_passage_count": len(split_passage_ids),
        },
    )


def _run_open_dense_system(
    *,
    system: SystemSpec,
    queries: Sequence[QueryExample],
    split_passage_ids: Sequence[str],
    split_passage_texts: Sequence[str],
    model_name_or_path: str,
    device: torch.device,
    max_len_query: int,
    max_len_passage: int,
    query_batch_size: int,
    passage_batch_size: int,
) -> FullRankingResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    query_texts = [select_query_text(query, query_view=system.query_view) for query in queries]
    prefixed_query_texts = [_apply_prefix(text, system.query_prefix) for text in query_texts]
    prefixed_passage_texts = [_apply_prefix(text, system.passage_prefix) for text in split_passage_texts]

    max_len_query_effective = _resolve_model_max_length(
        tokenizer=tokenizer,
        config=config,
        requested_max_length=max_len_query,
        model_name_or_path=model_name_or_path,
        field_name="query",
    )
    max_len_passage_effective = _resolve_model_max_length(
        tokenizer=tokenizer,
        config=config,
        requested_max_length=max_len_passage,
        model_name_or_path=model_name_or_path,
        field_name="passage",
    )

    query_stats = _compute_token_stats(
        tokenizer=tokenizer,
        texts=prefixed_query_texts,
        max_length=max_len_query_effective,
    )
    passage_stats = _compute_token_stats(
        tokenizer=tokenizer,
        texts=prefixed_passage_texts,
        max_length=max_len_passage_effective,
    )

    query_vecs = _encode_texts_mean_pooled(
        model=model,
        tokenizer=tokenizer,
        texts=prefixed_query_texts,
        batch_size=query_batch_size,
        max_length=max_len_query_effective,
        truncation_side="left",
        device=device,
    )
    passage_vecs = _encode_texts_mean_pooled(
        model=model,
        tokenizer=tokenizer,
        texts=prefixed_passage_texts,
        batch_size=passage_batch_size,
        max_length=max_len_passage_effective,
        truncation_side="right",
        device=device,
    )
    scores = query_vecs @ passage_vecs.T

    return _ranking_from_score_matrix(
        scores=scores,
        split_passage_ids=split_passage_ids,
        metadata={
            "model_source": model_name_or_path,
            "device": str(device),
            "query_view": system.query_view,
            "query_prefix": system.query_prefix,
            "passage_prefix": system.passage_prefix,
            "max_len_query_effective": int(max_len_query_effective),
            "max_len_passage_effective": int(max_len_passage_effective),
            "query_token_stats": query_stats,
            "passage_token_stats": passage_stats,
        },
    )


def _run_cohere_system(
    *,
    system: SystemSpec,
    queries: Sequence[QueryExample],
    split_passage_ids: Sequence[str],
    split_passage_texts: Sequence[str],
    max_len_query: int,
    max_len_passage: int,
) -> FullRankingResult:
    api_key = os.environ.get(system.cohere_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{system.name} requires environment variable {system.cohere_api_key_env} to be set"
        )

    query_texts = [select_query_text(query, query_view=system.query_view) for query in queries]
    query_embeddings = _embed_texts_with_cohere(
        texts=query_texts,
        model_name=system.cohere_model_name,
        input_type="search_query",
        api_key=api_key,
        output_dimension=system.cohere_output_dimension,
        truncate_mode="START",
        max_tokens=max_len_query,
    )
    passage_embeddings = _embed_texts_with_cohere(
        texts=list(split_passage_texts),
        model_name=system.cohere_model_name,
        input_type="search_document",
        api_key=api_key,
        output_dimension=system.cohere_output_dimension,
        truncate_mode="END",
        max_tokens=max_len_passage,
    )

    scores = query_embeddings @ passage_embeddings.T
    return _ranking_from_score_matrix(
        scores=scores,
        split_passage_ids=split_passage_ids,
        metadata={
            "model_source": system.cohere_model_name,
            "query_view": system.query_view,
            "cohere_api_key_env": system.cohere_api_key_env,
            "output_dimension": system.cohere_output_dimension,
            "query_truncate_mode": "START",
            "passage_truncate_mode": "END",
        },
    )


def _run_bm25_system(
    *,
    system: SystemSpec,
    queries: Sequence[QueryExample],
    split_passage_ids: Sequence[str],
    split_passage_texts: Sequence[str],
    work_dir: Path,
) -> FullRankingResult:
    _ensure_pyserini_available()
    work_dir.mkdir(parents=True, exist_ok=True)
    collection_dir = work_dir / "bm25_collection"
    index_dir = work_dir / "bm25_index"
    if collection_dir.exists():
        _remove_tree(collection_dir)
    if index_dir.exists():
        _remove_tree(index_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    collection_path = collection_dir / "corpus.jsonl"
    with collection_path.open("w", encoding="utf-8") as f:
        for passage_id, text in zip(split_passage_ids, split_passage_texts):
            f.write(json.dumps({"id": passage_id, "contents": text}, ensure_ascii=False) + "\n")

    stdout_path = work_dir / "bm25_index.stdout.txt"
    stderr_path = work_dir / "bm25_index.stderr.txt"
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    _run_logged_subprocess(cmd=cmd, stdout_path=stdout_path, stderr_path=stderr_path)

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(float(system.bm25_k1), float(system.bm25_b))

    ranked_passage_ids_by_query: List[List[str]] = []
    ranked_scores_by_query: List[List[float]] = []
    for query in queries:
        query_text = select_query_text(query, query_view=system.query_view)
        hits = searcher.search(query_text, k=len(split_passage_ids))
        seen_ids = set()
        ranked_ids = []
        ranked_scores = []
        for hit in hits:
            docid = str(hit.docid)
            if docid in seen_ids:
                continue
            seen_ids.add(docid)
            ranked_ids.append(docid)
            ranked_scores.append(float(hit.score))

        missing_ids = [passage_id for passage_id in split_passage_ids if passage_id not in seen_ids]
        ranked_ids.extend(missing_ids)
        ranked_scores.extend([0.0] * len(missing_ids))
        if len(ranked_ids) != len(split_passage_ids):
            raise RuntimeError(
                f"BM25 ranking for query {query.query_id} returned {len(ranked_ids)} ids "
                f"but split has {len(split_passage_ids)} passages"
            )
        ranked_passage_ids_by_query.append(ranked_ids)
        ranked_scores_by_query.append(ranked_scores)

    return FullRankingResult(
        ranked_passage_ids_by_query=ranked_passage_ids_by_query,
        ranked_scores_by_query=ranked_scores_by_query,
        metadata={
            "bm25_k1": float(system.bm25_k1),
            "bm25_b": float(system.bm25_b),
            "index_dir": str(index_dir),
            "query_view": system.query_view,
            "index_stdout_log": str(stdout_path),
            "index_stderr_log": str(stderr_path),
        },
    )


def _evaluate_full_ranking_under_regime(
    *,
    system: SystemSpec,
    regime_name: str,
    ranking: FullRankingResult,
    queries: Sequence[QueryExample],
    query_infos: Sequence[QueryInfo],
    candidate_ids_by_query: Sequence[Sequence[str]],
    ks: Sequence[int],
    rankings_path: Path,
    write_rankings_top_n: Optional[int],
) -> Dict[str, Any]:
    if len(queries) != len(query_infos):
        raise ValueError("queries and query_infos must be aligned")
    if len(query_infos) != len(candidate_ids_by_query):
        raise ValueError("query_infos and candidate_ids_by_query must be aligned")
    if len(query_infos) != len(ranking.ranked_passage_ids_by_query):
        raise ValueError("query_infos and full rankings must be aligned")

    per_query_rows: List[Dict[str, float]] = []
    with rankings_path.open("a", encoding="utf-8") as f:
        for idx, query_info in enumerate(query_infos):
            query_text_used = select_query_text(queries[idx], query_view=system.query_view)
            candidate_ids = [str(pid) for pid in candidate_ids_by_query[idx]]
            candidate_set = set(candidate_ids)
            filtered_ranked_ids: List[str] = []
            filtered_ranked_scores: List[float] = []
            for passage_id, score in zip(
                ranking.ranked_passage_ids_by_query[idx],
                ranking.ranked_scores_by_query[idx],
            ):
                if passage_id in candidate_set:
                    filtered_ranked_ids.append(passage_id)
                    filtered_ranked_scores.append(float(score))

            if len(filtered_ranked_ids) != len(candidate_ids):
                raise RuntimeError(
                    f"{system.name}/{regime_name} query {query_info.query_id} produced "
                    f"{len(filtered_ranked_ids)} filtered ids for candidate pool size {len(candidate_ids)}"
                )

            metrics = compute_query_metrics(
                retrieved_passage_ids=filtered_ranked_ids,
                gold_passage_ids=query_info.gold_passage_ids,
                ks=ks,
            )
            metrics["candidate_pool_size"] = float(len(candidate_ids))
            per_query_rows.append(metrics)

            record_limit = len(filtered_ranked_ids) if write_rankings_top_n is None else int(write_rankings_top_n)
            f.write(
                json.dumps(
                    {
                        "system": system.name,
                        "system_type": system.system_type,
                        "query_view": system.query_view,
                        "regime": regime_name,
                        "query_id": query_info.query_id,
                        "doc_id": query_info.doc_id,
                        "query_text": query_text_used,
                        "candidate_pool_size": int(len(candidate_ids)),
                        "gold_passage_ids": list(query_info.gold_passage_ids),
                        "ranked_candidates": [
                            {
                                "rank": int(rank_idx),
                                "passage_id": passage_id,
                                "score": float(score),
                            }
                            for rank_idx, (passage_id, score) in enumerate(
                                zip(filtered_ranked_ids[:record_limit], filtered_ranked_scores[:record_limit]),
                                start=1,
                            )
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    global_metrics = aggregate_metrics(per_query_rows)
    global_metrics["num_queries"] = float(len(per_query_rows))
    return {
        "global": global_metrics,
        "breakdowns": compute_bucketed_metrics(
            queries=query_infos,
            per_query_metrics=per_query_rows,
            ks=ks,
        ),
    }


def _ranking_from_score_matrix(
    *,
    scores: torch.Tensor,
    split_passage_ids: Sequence[str],
    metadata: Dict[str, Any],
) -> FullRankingResult:
    ranked_passage_ids_by_query: List[List[str]] = []
    ranked_scores_by_query: List[List[float]] = []
    for row_idx in range(scores.size(0)):
        order = torch.argsort(scores[row_idx], descending=True)
        ranked_passage_ids_by_query.append([split_passage_ids[int(i)] for i in order.tolist()])
        ranked_scores_by_query.append([float(scores[row_idx, int(i)]) for i in order.tolist()])
    return FullRankingResult(
        ranked_passage_ids_by_query=ranked_passage_ids_by_query,
        ranked_scores_by_query=ranked_scores_by_query,
        metadata=metadata,
    )


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


def _resolve_model_max_length(
    *,
    tokenizer: Any,
    config: Any,
    requested_max_length: int,
    model_name_or_path: str,
    field_name: str,
) -> int:
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if tokenizer_limit <= 0 or tokenizer_limit >= 10**9:
        tokenizer_limit = int(getattr(config, "max_position_embeddings", 0) or 0)
    if tokenizer_limit <= 0:
        raise ValueError(
            f"Could not determine max input length for {model_name_or_path} ({field_name}). "
            "Pass a tokenizer/config with a finite model_max_length or max_position_embeddings."
        )
    return min(int(requested_max_length), tokenizer_limit)


def _compute_token_stats(
    *,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    batch_size: int = 64,
) -> Dict[str, Any]:
    lengths: List[int] = []
    truncated = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        toks = tokenizer(
            batch_texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for input_ids in toks["input_ids"]:
            length = int(len(input_ids))
            lengths.append(length)
            if length > int(max_length):
                truncated += 1
    if not lengths:
        return {
            "num_texts": 0,
            "max_length": int(max_length),
            "num_truncated": 0,
            "fraction_truncated": 0.0,
        }
    return {
        "num_texts": int(len(lengths)),
        "max_length": int(max_length),
        "num_truncated": int(truncated),
        "fraction_truncated": float(truncated) / float(len(lengths)),
        "mean_tokens_before_truncation": float(sum(lengths) / len(lengths)),
        "max_tokens_before_truncation": int(max(lengths)),
    }


def _encode_texts_mean_pooled(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    truncation_side: str,
    device: torch.device,
) -> torch.Tensor:
    tokenizer.truncation_side = str(truncation_side)
    all_vecs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            toks = tokenizer(
                batch_texts,
                truncation=True,
                max_length=int(max_length),
                padding=True,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            outputs = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"], return_dict=True)
            last_hidden = outputs.last_hidden_state
            attention_mask = toks["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-6)
            all_vecs.append(torch_nn_func.normalize(pooled, p=2, dim=-1).detach().cpu())
    return torch.cat(all_vecs, dim=0)


def _embed_texts_with_cohere(
    *,
    texts: Sequence[str],
    model_name: str,
    input_type: str,
    api_key: str,
    output_dimension: Optional[int],
    truncate_mode: str,
    max_tokens: int,
    batch_size: int = 96,
) -> torch.Tensor:
    if not texts:
        raise ValueError("texts must contain at least one item for Cohere embedding")

    url = "https://api.cohere.com/v2/embed"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    all_embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        payload: Dict[str, Any] = {
            "model": model_name,
            "input_type": input_type,
            "texts": batch_texts,
            "embedding_types": ["float"],
            "truncate": str(truncate_mode),
            "max_tokens": int(max_tokens),
        }
        if output_dimension is not None:
            payload["output_dimension"] = int(output_dimension)

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Cohere embed request failed with status={response.status_code}: {response.text}"
            )
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, dict):
            raise RuntimeError(f"Unexpected Cohere response shape: {data}")
        float_embeddings = embeddings.get("float")
        if not isinstance(float_embeddings, list):
            raise RuntimeError(f"Cohere response does not contain embeddings.float: {data}")
        all_embeddings.extend(float_embeddings)

    tensor = torch.tensor(all_embeddings, dtype=torch.float32)
    return torch_nn_func.normalize(tensor, p=2, dim=-1)


def _render_report_md(results: Dict[str, Any], *, ks: Sequence[int]) -> str:
    config = results.get("config") or {}
    systems = results.get("systems") or {}
    regimes = results.get("regimes") or {}

    def fmt(metrics: Dict[str, Any], key: str) -> str:
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
        "mrr_at_20",
        "set_recall_at_20",
        "exact_set_match_at_20",
        "candidate_pool_size",
    ]

    lines: List[str] = []
    lines.append("# Retrieval experiment report")
    lines.append("")
    lines.append(f"- Processed dir: `{config.get('processed_dir', '')}`")
    lines.append(f"- Split: `{config.get('split', '')}`")
    lines.append(f"- K values: `{list(ks)}`")
    lines.append(f"- Split passage count: `{config.get('split_passage_count', '')}`")
    lines.append("")

    lines.append("## Systems")
    lines.append("")
    for system_name, payload in systems.items():
        metadata = (payload or {}).get("metadata") or {}
        lines.append(
            f"- `{system_name}`: type=`{payload.get('system_type', '')}` "
            f"query_view=`{payload.get('query_view', '')}`"
        )
        if metadata.get("model_source"):
            lines.append(f"  source=`{metadata['model_source']}`")
    lines.append("")

    for regime_name, regime_payload in regimes.items():
        lines.append(f"## {regime_name}")
        lines.append("")
        lines.append("| system | " + " | ".join(f"`{key}`" for key in report_keys) + " |")
        lines.append("|---" + "|---:" * len(report_keys) + "|")
        for system_name, system_payload in (regime_payload.get("systems") or {}).items():
            global_metrics = (system_payload or {}).get("global") or {}
            lines.append(
                "| "
                + system_name
                + " | "
                + " | ".join(fmt(global_metrics, key) for key in report_keys)
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def _apply_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return str(text)
    return f"{prefix}{text}"


def _ensure_pyserini_available() -> None:
    try:
        import pyserini  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyserini is required for the BM25 baseline. Install pyserini and ensure Java is available, "
            "or run BM25 from a custom/extended SageMaker container."
        ) from exc


def _run_logged_subprocess(
    *,
    cmd: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}. "
            f"See stdout={stdout_path} stderr={stderr_path}"
        )


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def _resolve_work_dir(system: SystemSpec) -> Path:
    if system.work_dir:
        return Path(system.work_dir).expanduser().resolve()
    return Path.cwd() / ".retrieval_experiment" / system.name


def _require_model_name_or_path(system: SystemSpec) -> str:
    if not system.model_name_or_path:
        raise ValueError(f"{system.name} requires model_name_or_path")
    return str(system.model_name_or_path)


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
