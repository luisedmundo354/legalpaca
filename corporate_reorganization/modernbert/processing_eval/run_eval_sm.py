from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing_eval.experiment import (
    DEFAULT_COHERE_MODEL,
    DEFAULT_OPEN_DENSE_MODEL,
    REGIME_GLOBAL_SPLIT,
    REGIME_SAME_CASE_FULL,
    REGIME_SAME_CASE_LEGACY,
    SYSTEM_TYPE_BM25,
    SYSTEM_TYPE_COHERE,
    SYSTEM_TYPE_MODERNBERT_ARTIFACT,
    SYSTEM_TYPE_MODERNBERT_BASE,
    SYSTEM_TYPE_OPEN_DENSE,
    QUERY_VIEW_FLAT_MASKED,
    QUERY_VIEW_FLAT_PLAIN,
    QUERY_VIEW_STRUCTURED,
    SystemSpec,
    run_retrieval_experiment,
)


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _coalesce_model_ref(*, s3_uri: Optional[str], local_dir: Optional[str], alias_s3_uri: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    return (s3_uri or alias_s3_uri, local_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument(
        "--systems",
        type=str,
        default="bm25_flat,dense_open_flat,base_modernbert_flat,fine_tuned_flat,fine_tuned_structured",
        help="Comma-separated system names.",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=f"{REGIME_SAME_CASE_LEGACY},{REGIME_SAME_CASE_FULL},{REGIME_GLOBAL_SPLIT}",
        help="Comma-separated candidate regime names.",
    )

    parser.add_argument("--fine_tuned_model_s3_uri", type=str, default=None)
    parser.add_argument("--structured_model_s3_uri", type=str, default=None)
    parser.add_argument("--structured_model_dir", type=str, default=None)
    parser.add_argument("--flat_model_s3_uri", type=str, default=None)
    parser.add_argument("--flat_model_dir", type=str, default=None)
    parser.add_argument("--base_model_name_or_path", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--open_dense_model_name_or_path", type=str, default=DEFAULT_OPEN_DENSE_MODEL)
    parser.add_argument("--cohere_model_name", type=str, default=DEFAULT_COHERE_MODEL)
    parser.add_argument("--cohere_api_key_env", type=str, default="COHERE_API_KEY")
    parser.add_argument("--cohere_output_dimension", type=int, default=None)
    parser.add_argument("--work_dir", type=str, default=None)

    parser.add_argument("--max_len_query", type=int, default=4096)
    parser.add_argument("--max_len_passage", type=int, default=600)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--passage_batch_size", type=int, default=256)
    parser.add_argument("--k_values", type=str, default="1,5,10,20")
    parser.add_argument("--random_seed", type=int, default=17)
    parser.add_argument("--write_rankings_top_n", type=int, default=None)
    return parser.parse_args(argv)


def _build_system_specs(args: argparse.Namespace) -> List[SystemSpec]:
    requested_systems = set(_parse_csv(args.systems))
    structured_s3_uri, structured_model_dir = _coalesce_model_ref(
        s3_uri=args.structured_model_s3_uri,
        local_dir=args.structured_model_dir,
        alias_s3_uri=args.fine_tuned_model_s3_uri,
    )
    flat_s3_uri, flat_model_dir = _coalesce_model_ref(
        s3_uri=args.flat_model_s3_uri,
        local_dir=args.flat_model_dir,
        alias_s3_uri=None,
    )

    systems: List[SystemSpec] = []
    if "bm25_flat" in requested_systems:
        systems.append(
            SystemSpec(
                name="bm25_flat",
                system_type=SYSTEM_TYPE_BM25,
                query_view=QUERY_VIEW_FLAT_PLAIN,
                work_dir=args.work_dir,
            )
        )
    if "dense_open_flat" in requested_systems:
        systems.append(
            SystemSpec(
                name="dense_open_flat",
                system_type=SYSTEM_TYPE_OPEN_DENSE,
                query_view=QUERY_VIEW_FLAT_PLAIN,
                model_name_or_path=args.open_dense_model_name_or_path,
                query_prefix="query: ",
                passage_prefix="passage: ",
            )
        )
    if "base_modernbert_flat" in requested_systems:
        systems.append(
            SystemSpec(
                name="base_modernbert_flat",
                system_type=SYSTEM_TYPE_MODERNBERT_BASE,
                query_view=QUERY_VIEW_FLAT_MASKED,
                model_name_or_path=args.base_model_name_or_path,
            )
        )
    if "fine_tuned_flat" in requested_systems:
        if not flat_s3_uri and not flat_model_dir:
            raise ValueError("fine_tuned_flat requires --flat_model_s3_uri or --flat_model_dir")
        systems.append(
            SystemSpec(
                name="fine_tuned_flat",
                system_type=SYSTEM_TYPE_MODERNBERT_ARTIFACT,
                query_view=QUERY_VIEW_FLAT_MASKED,
                model_s3_uri=flat_s3_uri,
                model_dir=flat_model_dir,
                work_dir=args.work_dir,
            )
        )
    if "fine_tuned_structured" in requested_systems:
        if not structured_s3_uri and not structured_model_dir:
            raise ValueError(
                "fine_tuned_structured requires --structured_model_s3_uri/--structured_model_dir "
                "or deprecated --fine_tuned_model_s3_uri"
            )
        systems.append(
            SystemSpec(
                name="fine_tuned_structured",
                system_type=SYSTEM_TYPE_MODERNBERT_ARTIFACT,
                query_view=QUERY_VIEW_STRUCTURED,
                model_s3_uri=structured_s3_uri,
                model_dir=structured_model_dir,
                work_dir=args.work_dir,
            )
        )
    if "cohere_embed4_flat" in requested_systems:
        systems.append(
            SystemSpec(
                name="cohere_embed4_flat",
                system_type=SYSTEM_TYPE_COHERE,
                query_view=QUERY_VIEW_FLAT_PLAIN,
                cohere_model_name=args.cohere_model_name,
                cohere_api_key_env=args.cohere_api_key_env,
                cohere_output_dimension=args.cohere_output_dimension,
            )
        )

    unknown = requested_systems - {system.name for system in systems}
    if unknown:
        raise ValueError(f"Unknown or unsupported systems requested: {sorted(unknown)}")
    return systems


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")

    results = run_retrieval_experiment(
        processed_dir=Path(args.processed_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        split=str(args.split),
        systems=_build_system_specs(args),
        regimes=_parse_csv(args.regimes),
        max_len_query=int(args.max_len_query),
        max_len_passage=int(args.max_len_passage),
        query_batch_size=int(args.query_batch_size),
        passage_batch_size=int(args.passage_batch_size),
        ks=_parse_csv_ints(args.k_values),
        random_seed=int(args.random_seed),
        write_rankings_top_n=args.write_rankings_top_n,
    )

    for regime_name, regime_payload in (results.get("regimes") or {}).items():
        for system_name, system_payload in (regime_payload.get("systems") or {}).items():
            metrics = (system_payload or {}).get("global") or {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"SM_METRIC {regime_name}.{system_name}.{key}={float(value)}")

    print(json.dumps(results["config"], indent=2))


if __name__ == "__main__":
    main()
