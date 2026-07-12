from __future__ import annotations

from typing import TYPE_CHECKING, Final, Tuple

from .markup import SLOT_TOKEN

if TYPE_CHECKING:
    from .data import QueryExample


QUERY_VIEW_STRUCTURED: Final[str] = "structured"
QUERY_VIEW_FLAT_PLAIN: Final[str] = "flat_plain"
QUERY_VIEW_FLAT_MASKED: Final[str] = "flat_masked"

SUPPORTED_QUERY_VIEWS: Final[Tuple[str, ...]] = (
    QUERY_VIEW_STRUCTURED,
    QUERY_VIEW_FLAT_PLAIN,
    QUERY_VIEW_FLAT_MASKED,
)


def normalize_query_view(query_view: str) -> str:
    value = str(query_view).strip().lower()
    if value not in SUPPORTED_QUERY_VIEWS:
        raise ValueError(f"Unsupported query_view={query_view!r}; expected one of {SUPPORTED_QUERY_VIEWS}")
    return value


def select_query_text(query: "QueryExample", *, query_view: str) -> str:
    view = normalize_query_view(query_view)
    if view == QUERY_VIEW_STRUCTURED:
        text = str(query.query_text)
    elif view == QUERY_VIEW_FLAT_PLAIN:
        text = str(query.flat_query_text_plain or "")
        if not text:
            raise ValueError(
                f"Query {query.query_id} is missing flat_query_text_plain. "
                "Rebuild the processed dataset with the updated builder."
            )
        if SLOT_TOKEN in text:
            raise ValueError(
                f"flat_query_text_plain for query {query.query_id} unexpectedly contains {SLOT_TOKEN}"
            )
    else:
        text = str(query.flat_query_text_masked or "")
        if not text:
            raise ValueError(
                f"Query {query.query_id} is missing flat_query_text_masked. "
                "Rebuild the processed dataset with the updated builder."
            )
        slot_count = text.count(SLOT_TOKEN)
        if slot_count != 1:
            raise ValueError(
                f"flat_query_text_masked for query {query.query_id} must contain exactly one "
                f"{SLOT_TOKEN}; found {slot_count}"
            )

    if not text.strip():
        raise ValueError(f"Selected query text is empty for query_id={query.query_id} view={view}")
    return text
