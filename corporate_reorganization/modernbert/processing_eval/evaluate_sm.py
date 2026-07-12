"""Strict SageMaker Processing entry point for the complete retrieval evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.evaluator import run_complete_evaluation_plan  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one immutable 15-system ARR retrieval fold evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--local-bindings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_complete_evaluation_plan(
        evaluation_plan_path=args.evaluation_plan,
        local_bindings_path=args.local_bindings,
        output_dir=args.output_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
