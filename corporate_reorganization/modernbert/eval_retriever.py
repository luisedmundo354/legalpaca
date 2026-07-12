from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Sequence

from retriever.evaluator import run_local_controlled_evaluation_plan


def _json_ready(value):
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"Publication record contains a non-JSON value: {type(value).__name__}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one immutable local controlled-retrieval evaluation plan.",
        allow_abbrev=False,
    )
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--local-bindings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    publication = run_local_controlled_evaluation_plan(
        evaluation_plan_path=args.evaluation_plan.resolve(),
        local_bindings_path=args.local_bindings.resolve(),
        output_dir=args.output_dir.resolve(),
        device=args.device,
    )
    print(
        json.dumps(
            _json_ready(publication),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
