from __future__ import annotations

from typing import List

SLOT_TOKEN = "[SLOT]"
MISSING_TOKEN = "[MISSING]"
IMPLICIT_TOKEN = "[IMPLICIT]"

LABEL_TOKENS: List[str] = [
    "[RULE]",
    "[ANALYSIS]",
    "[CONCLUSION]",
    "[BACKGROUND]",
    "[PROCEDURE]",
]

STRUCTURE_TOKENS: List[str] = [
    "[ARG]",
    "[/ARG]",
    "[ROOT]",
    "[TREE]",
    "[/TREE]",
    "[FOCUS]",
    "[/FOCUS]",
    "[STEP]",
    "[/STEP]",
    "[CONCL]",
    "[PREMISE]",
]


def all_markup_tokens() -> List[str]:
    return [SLOT_TOKEN, MISSING_TOKEN, IMPLICIT_TOKEN, *LABEL_TOKENS, *STRUCTURE_TOKENS]

