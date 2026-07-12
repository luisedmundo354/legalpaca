"""SageMaker Processing entry point for the strict local-plan evaluator."""

from __future__ import annotations

import sys
from pathlib import Path


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from eval_retriever import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
