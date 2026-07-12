from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.artifacts import import_pinned_artifact_runtime  # noqa: E402
from retriever.baseline_artifacts import build_fixed_base_artifact  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the one deterministic untrained ModernBERT retrieval artifact.",
        allow_abbrev=False,
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--artifact-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_fixed_base_artifact(
        snapshot_dir=args.snapshot_dir,
        snapshot_manifest_path=args.snapshot_manifest,
        baseline_config_path=args.baseline_config,
        artifact_contract_path=args.artifact_contract,
        output_dir=args.output_dir,
        runtime=import_pinned_artifact_runtime(),
        numpy_module=numpy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
