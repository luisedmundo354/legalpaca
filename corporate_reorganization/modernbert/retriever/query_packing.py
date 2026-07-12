from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .data import QueryExample
from .markup import SLOT_TOKEN
from .query_views import (
    QUERY_VIEW_FLAT_MASKED,
    QUERY_VIEW_FLAT_PLAIN,
    QUERY_VIEW_STRUCTURED,
    normalize_query_view,
)


FOCUS_PRESERVING_PACK_PROTOCOL = "focus_preserving_semantic_pack_v1"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
E5_MAX_POSITIONS = 512
E5_SINGLE_SEQUENCE_SPECIAL_TOKENS = 2
E5_QUERY_PREFIX_TOKEN_IDS = (23_032, 1_024)

_LABEL_TOKEN_TO_NAME = {
    "[RULE]": "rule",
    "[ANALYSIS]": "analysis",
    "[CONCLUSION]": "conclusion",
    "[BACKGROUND]": "background",
    "[PROCEDURE]": "procedure",
}
_LABEL_NAME_TO_TOKEN = {value: key for key, value in _LABEL_TOKEN_TO_NAME.items()}
_NODE_MARKER = re.compile(r"(?m)^\[(CONCL|PREMISE)\] ")
_STEP_MARKER = re.compile(r"(?m)^\[STEP\]$")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SemanticNode:
    role: str
    state: str
    label: str | None
    text: str | None


@dataclass(frozen=True)
class SemanticStep:
    conclusion: SemanticNode
    premises: tuple[SemanticNode, ...]

    @property
    def nodes(self) -> tuple[SemanticNode, ...]:
        return (self.conclusion, *self.premises)


@dataclass(frozen=True)
class SemanticQuery:
    root: SemanticNode
    context_steps: tuple[SemanticStep, ...]
    focus: SemanticStep


@dataclass(frozen=True)
class PackedQuery:
    query_id: str
    protocol: str
    output_view: str
    fit_views: tuple[str, ...]
    rendered_text: str
    input_ids: tuple[int, ...]
    selected_content_tokens: tuple[tuple[str, int, int], ...]
    root_included: bool
    context_step_positions: tuple[int, ...]
    contract_sha256: str


@dataclass(frozen=True)
class _ContentPrefixes:
    texts: tuple[str, ...]
    wordpiece_counts: tuple[int, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_node_payload(payload: str, *, role: str) -> SemanticNode:
    if role not in {"root", "conclusion", "premise"}:
        raise ValueError(f"Unsupported semantic node role={role!r}")
    if type(payload) is not str or not payload or payload.strip() != payload:
        raise ValueError(f"Malformed {role} payload")
    if payload == SLOT_TOKEN:
        return SemanticNode(role=role, state="slot", label=None, text=None)
    if payload == "[MISSING]":
        return SemanticNode(role=role, state="missing", label=None, text=None)
    if payload.startswith("[IMPLICIT] "):
        label_token = payload.removeprefix("[IMPLICIT] ")
        if label_token not in _LABEL_TOKEN_TO_NAME:
            raise ValueError(f"Malformed implicit {role} label={label_token!r}")
        return SemanticNode(
            role=role,
            state="implicit",
            label=_LABEL_TOKEN_TO_NAME[label_token],
            text=None,
        )
    label_token, separator, text = payload.partition(" ")
    if label_token not in _LABEL_TOKEN_TO_NAME or separator != " " or not text:
        raise ValueError(f"Malformed visible {role} payload")
    if text.strip() != text:
        raise ValueError(f"Visible {role} text has boundary whitespace")
    return SemanticNode(
        role=role,
        state="visible",
        label=_LABEL_TOKEN_TO_NAME[label_token],
        text=text,
    )


def _parse_step(block: str) -> SemanticStep:
    if not block.startswith("[STEP]\n") or not block.endswith("\n[/STEP]"):
        raise ValueError("Malformed structured step boundary")
    body = block[len("[STEP]\n") : -len("\n[/STEP]")]
    matches = list(_NODE_MARKER.finditer(body))
    if not matches or matches[0].group(1) != "CONCL":
        raise ValueError("Structured step must begin with one conclusion")
    nodes: list[SemanticNode] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() - 1 if position + 1 < len(matches) else len(body)
        payload = body[match.end() : end]
        role = "conclusion" if match.group(1) == "CONCL" else "premise"
        if position > 0 and role != "premise":
            raise ValueError("Structured step contains more than one conclusion")
        nodes.append(_parse_node_payload(payload, role=role))
    return SemanticStep(conclusion=nodes[0], premises=tuple(nodes[1:]))


def _parse_steps(section: str) -> tuple[SemanticStep, ...]:
    if not section:
        return ()
    starts = list(_STEP_MARKER.finditer(section))
    if not starts or starts[0].start() != 0:
        raise ValueError("Structured tree contains text outside a step")
    blocks: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(section)
        block = section[start.start() : end]
        if position + 1 < len(starts):
            if not block.endswith("\n\n"):
                raise ValueError("Structured steps must use one blank-line separator")
            block = block[:-2]
        blocks.append(block)
    return tuple(_parse_step(block) for block in blocks)


def parse_semantic_query(query: QueryExample) -> SemanticQuery:
    if not isinstance(query, QueryExample):
        raise TypeError("query must be QueryExample")
    text = query.query_text
    prefix = "[ARG]\n[ROOT] "
    tree_delimiter = "\n[TREE]\n"
    focus_delimiter = "\n[/TREE]\n[FOCUS]\n"
    suffix = "\n[/FOCUS]\n[/ARG]"
    if (
        type(text) is not str
        or not text.startswith(prefix)
        or not text.endswith(suffix)
        or text.count(tree_delimiter) != 1
        or text.count(focus_delimiter) != 1
    ):
        raise ValueError(f"Query {query.query_id!r} has malformed structured boundaries")
    body = text[len(prefix) : -len(suffix)]
    root_payload, tree_and_focus = body.split(tree_delimiter, 1)
    tree_text, focus_text = tree_and_focus.split(focus_delimiter, 1)
    focus_steps = _parse_steps(focus_text)
    if len(focus_steps) != 1:
        raise ValueError(f"Query {query.query_id!r} must contain exactly one focus step")
    semantic = SemanticQuery(
        root=_parse_node_payload(root_payload, role="root"),
        context_steps=_parse_steps(tree_text),
        focus=focus_steps[0],
    )
    if sum(node.state == "slot" for node in semantic.focus.nodes) != 1:
        raise ValueError(f"Query {query.query_id!r} focus must contain exactly one slot")
    if semantic.root.state == "slot" or any(
        node.state == "slot" for step in semantic.context_steps for node in step.nodes
    ):
        raise ValueError(f"Query {query.query_id!r} contains a slot outside the focus")
    return semantic


def _structured_node_payload(node: SemanticNode, *, content_text: str | None) -> str:
    if node.state == "slot":
        return SLOT_TOKEN
    if node.state == "missing":
        return "[MISSING]"
    if node.label not in _LABEL_NAME_TO_TOKEN:
        raise ValueError(f"Semantic node has invalid label={node.label!r}")
    label_token = _LABEL_NAME_TO_TOKEN[node.label]
    if node.state == "implicit":
        return f"[IMPLICIT] {label_token}"
    if node.state != "visible" or type(content_text) is not str or not content_text:
        raise ValueError("Visible structured node has no selected content")
    return f"{label_token} {content_text}"


def _flat_node_payload(
    node: SemanticNode,
    *,
    content_text: str | None,
    query_view: str,
) -> str:
    if node.state == "slot":
        return "missing span" if query_view == QUERY_VIEW_FLAT_PLAIN else SLOT_TOKEN
    if node.state == "missing":
        return "missing"
    if node.label not in _LABEL_NAME_TO_TOKEN:
        raise ValueError(f"Semantic node has invalid label={node.label!r}")
    if node.state == "implicit":
        return f"implicit {node.label}"
    if node.state != "visible" or type(content_text) is not str or not content_text:
        raise ValueError("Visible flat node has no selected content")
    return f"{node.label}: {content_text}"


def _node_id(section: str, step_position: int, node_position: int) -> str:
    return f"{section}:{step_position}:{node_position}"


def _all_nodes(semantic: SemanticQuery) -> tuple[tuple[str, SemanticNode], ...]:
    records: list[tuple[str, SemanticNode]] = [("root:0:0", semantic.root)]
    for step_position, step in enumerate(semantic.context_steps):
        for node_position, node in enumerate(step.nodes):
            records.append((_node_id("context", step_position, node_position), node))
    for node_position, node in enumerate(semantic.focus.nodes):
        records.append((_node_id("focus", 0, node_position), node))
    return tuple(records)


def _focus_context_step_position(semantic: SemanticQuery) -> int:
    def normalized(node: SemanticNode) -> tuple[str, str, str | None, str | None]:
        state = "placeholder" if node.state in {"slot", "missing"} else node.state
        return node.role, state, node.label, node.text

    focus = tuple(normalized(node) for node in semantic.focus.nodes)
    matches = [
        position
        for position, step in enumerate(semantic.context_steps)
        if tuple(normalized(node) for node in step.nodes) == focus
    ]
    if len(matches) != 1:
        raise ValueError(
            "Structured query must contain exactly one context step corresponding to focus"
        )
    return matches[0]


def _selected_text_by_id(
    semantic: SemanticQuery,
    *,
    content_limits: Mapping[str, int],
    content_prefixes: Mapping[str, _ContentPrefixes],
) -> dict[str, str | None]:
    selected: dict[str, str | None] = {}
    for unit_id, node in _all_nodes(semantic):
        if node.state != "visible":
            selected[unit_id] = None
            continue
        prefixes = content_prefixes[unit_id].texts
        limit = content_limits.get(unit_id, 0)
        if type(limit) is not int or limit < 0 or limit > len(prefixes):
            raise ValueError(f"Invalid selected content limit for {unit_id}")
        if limit == 0:
            selected[unit_id] = None
        else:
            selected[unit_id] = prefixes[limit - 1]
    return selected


def _render_step(
    step: SemanticStep,
    *,
    section: str,
    step_position: int,
    selected_text: Mapping[str, str | None],
    query_view: str,
) -> str:
    lines: list[str] = []
    for node_position, node in enumerate(step.nodes):
        unit_id = _node_id(section, step_position, node_position)
        content_text = selected_text[unit_id]
        if node.state == "visible" and content_text is None:
            continue
        if query_view == QUERY_VIEW_STRUCTURED:
            marker = "[CONCL]" if node.role == "conclusion" else "[PREMISE]"
            payload = _structured_node_payload(node, content_text=content_text)
            lines.append(f"{marker} {payload}")
        else:
            marker = "conclusion" if node.role == "conclusion" else "premise"
            payload = _flat_node_payload(
                node,
                content_text=content_text,
                query_view=query_view,
            )
            lines.append(f"{marker}: {payload}")
    if not lines or not lines[0].startswith(("[CONCL] ", "conclusion: ")):
        raise ValueError(f"Selected {section} step has no conclusion")
    if query_view == QUERY_VIEW_STRUCTURED:
        return "\n".join(("[STEP]", *lines, "[/STEP]"))
    return "\n".join(lines)


def render_semantic_query(
    semantic: SemanticQuery,
    *,
    query_view: str,
    content_limits: Mapping[str, int] | None = None,
    content_prefixes: Mapping[str, _ContentPrefixes] | None = None,
    include_root: bool = True,
    context_step_positions: Sequence[int] | None = None,
) -> str:
    view = normalize_query_view(query_view)
    if content_limits is None or content_prefixes is None:
        if content_limits is not None or content_prefixes is not None:
            raise ValueError("content_limits and content_prefixes must be supplied together")
        selected_text = {
            unit_id: node.text if node.state == "visible" else None
            for unit_id, node in _all_nodes(semantic)
        }
    else:
        selected_text = _selected_text_by_id(
            semantic,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
        )
    if type(include_root) is not bool:
        raise TypeError("include_root must be exact bool")
    if context_step_positions is None:
        normalized_context_positions = tuple(range(len(semantic.context_steps)))
    else:
        if not isinstance(context_step_positions, Sequence) or isinstance(
            context_step_positions, (str, bytes)
        ):
            raise TypeError("context_step_positions must be a sequence")
        normalized_context_positions = tuple(context_step_positions)
        if (
            any(type(value) is not int for value in normalized_context_positions)
            or normalized_context_positions != tuple(sorted(set(normalized_context_positions)))
            or any(
                value < 0 or value >= len(semantic.context_steps)
                for value in normalized_context_positions
            )
        ):
            raise ValueError("context_step_positions are not canonical context positions")
    included_context_ids = {
        _node_id("context", step_position, node_position)
        for step_position in normalized_context_positions
        for node_position, _ in enumerate(semantic.context_steps[step_position].nodes)
    }
    for unit_id, node in _all_nodes(semantic):
        if unit_id.startswith("context:") and unit_id not in included_context_ids:
            selected_text[unit_id] = None

    if view == QUERY_VIEW_STRUCTURED:
        parts = ["[ARG]"]
        if include_root:
            root_text = selected_text["root:0:0"]
            if semantic.root.state == "visible" and root_text is None:
                raise ValueError("Included root has no selected content")
            parts.append(
                f"[ROOT] {_structured_node_payload(semantic.root, content_text=root_text)}"
            )
        context_steps: list[str] = []
        for step_position in normalized_context_positions:
            step = semantic.context_steps[step_position]
            context_steps.append(
                _render_step(
                    step,
                    section="context",
                    step_position=step_position,
                    selected_text=selected_text,
                    query_view=view,
                )
            )
        if context_steps:
            parts.extend(("[TREE]", "\n\n".join(context_steps), "[/TREE]"))
        parts.append("[FOCUS]")
        parts.append(
            _render_step(
                semantic.focus,
                section="focus",
                step_position=0,
                selected_text=selected_text,
                query_view=view,
            )
        )
        parts.extend(("[/FOCUS]", "[/ARG]"))
        return "\n".join(parts)

    parts = ["argument"]
    if include_root:
        root_text = selected_text["root:0:0"]
        if semantic.root.state == "visible" and root_text is None:
            raise ValueError("Included root has no selected content")
        parts.append(
            "root: "
            + _flat_node_payload(
                semantic.root,
                content_text=root_text,
                query_view=view,
            )
        )
    context_steps = []
    for step_position in normalized_context_positions:
        step = semantic.context_steps[step_position]
        context_steps.append(
            _render_step(
                step,
                section="context",
                step_position=step_position,
                selected_text=selected_text,
                query_view=view,
            )
        )
    if context_steps:
        parts.extend(("", "context", "\n\n".join(context_steps)))
    parts.extend(("", "focus"))
    parts.append(
        _render_step(
            semantic.focus,
            section="focus",
            step_position=0,
            selected_text=selected_text,
            query_view=view,
        )
    )
    return "\n".join(parts).strip()


def validate_query_renderings(query: QueryExample) -> SemanticQuery:
    semantic = parse_semantic_query(query)
    rendered = {
        QUERY_VIEW_STRUCTURED: render_semantic_query(
            semantic,
            query_view=QUERY_VIEW_STRUCTURED,
        ),
        QUERY_VIEW_FLAT_PLAIN: render_semantic_query(
            semantic,
            query_view=QUERY_VIEW_FLAT_PLAIN,
        ),
        QUERY_VIEW_FLAT_MASKED: render_semantic_query(
            semantic,
            query_view=QUERY_VIEW_FLAT_MASKED,
        ),
    }
    expected = {
        QUERY_VIEW_STRUCTURED: query.query_text,
        QUERY_VIEW_FLAT_PLAIN: query.flat_query_text_plain,
        QUERY_VIEW_FLAT_MASKED: query.flat_query_text_masked,
    }
    for view in rendered:
        if rendered[view] != expected[view]:
            raise ValueError(
                f"Query {query.query_id!r} semantic {view} rendering changed"
            )
    return semantic


def _token_content_prefixes(
    semantic: SemanticQuery,
    *,
    tokenizer: Any,
) -> tuple[dict[str, _ContentPrefixes], dict[str, int]]:
    prefixes: dict[str, _ContentPrefixes] = {}
    limits: dict[str, int] = {}
    for unit_id, node in _all_nodes(semantic):
        if node.state != "visible":
            continue
        if type(node.text) is not str or not node.text:
            raise ValueError(f"Visible semantic unit {unit_id} has no text")
        full_ids = _encode_without_specials(tokenizer, node.text)
        texts: list[str] = []
        wordpiece_counts: list[int] = []
        for match in re.finditer(r"\S+(?:\s+|$)", node.text):
            prefix = node.text[: match.end()].rstrip()
            prefix_ids = _encode_without_specials(tokenizer, prefix)
            if prefix_ids != full_ids[: len(prefix_ids)]:
                raise RuntimeError(
                    f"Complete-word prefix changed E5 token identities for {unit_id}"
                )
            texts.append(prefix)
            wordpiece_counts.append(len(prefix_ids))
        if not texts or texts[-1] != node.text or wordpiece_counts[-1] != len(full_ids):
            raise ValueError(f"Complete-word prefixes do not cover all content for {unit_id}")
        if wordpiece_counts != sorted(set(wordpiece_counts)):
            raise ValueError(f"Complete-word token counts are not strictly increasing for {unit_id}")
        prefixes[unit_id] = _ContentPrefixes(
            texts=tuple(texts),
            wordpiece_counts=tuple(wordpiece_counts),
        )
        limits[unit_id] = len(texts)
    return prefixes, limits


def _encode_without_specials(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tokenizer(text, add_special_tokens=False).get("input_ids")
    if (
        not isinstance(ids, Sequence)
        or isinstance(ids, (str, bytes))
        or not ids
        or any(type(value) is not int or value < 0 for value in ids)
    ):
        raise TypeError("Tokenizer returned malformed input_ids")
    return tuple(ids)


def _rendered_lengths(
    semantic: SemanticQuery,
    *,
    tokenizer: Any,
    fit_views: tuple[str, ...],
    content_limits: Mapping[str, int],
    content_prefixes: Mapping[str, _ContentPrefixes],
    include_root: bool,
    context_step_positions: tuple[int, ...],
) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for view in fit_views:
        rendered = render_semantic_query(
            semantic,
            query_view=view,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
            include_root=include_root,
            context_step_positions=context_step_positions,
        )
        lengths[view] = len(_encode_without_specials(tokenizer, rendered))
    return lengths


def _fits(lengths: Mapping[str, int], *, body_budget: int) -> bool:
    return bool(lengths) and max(lengths.values()) <= body_budget


def _maximize_unit_limit(
    unit_id: str,
    *,
    semantic: SemanticQuery,
    tokenizer: Any,
    fit_views: tuple[str, ...],
    content_limits: dict[str, int],
    content_prefixes: Mapping[str, _ContentPrefixes],
    include_root: bool,
    context_step_positions: tuple[int, ...],
    body_budget: int,
) -> bool:
    full_limit = len(content_prefixes[unit_id].texts)
    original_limit = content_limits[unit_id]
    low = 1
    high = min(original_limit, full_limit)
    content_limits[unit_id] = low
    minimum_lengths = _rendered_lengths(
        semantic,
        tokenizer=tokenizer,
        fit_views=fit_views,
        content_limits=content_limits,
        content_prefixes=content_prefixes,
        include_root=include_root,
        context_step_positions=context_step_positions,
    )
    if not _fits(minimum_lengths, body_budget=body_budget):
        content_limits[unit_id] = original_limit
        return False
    best = low
    while low <= high:
        middle = (low + high) // 2
        content_limits[unit_id] = middle
        lengths = _rendered_lengths(
            semantic,
            tokenizer=tokenizer,
            fit_views=fit_views,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
            include_root=include_root,
            context_step_positions=context_step_positions,
        )
        if _fits(lengths, body_budget=body_budget):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    content_limits[unit_id] = best
    return True


def pack_focus_preserving_query(
    query: QueryExample,
    *,
    tokenizer: Any,
    output_view: str = QUERY_VIEW_FLAT_PLAIN,
    fit_views: Sequence[str] = (QUERY_VIEW_FLAT_PLAIN,),
    max_length: int = E5_MAX_POSITIONS,
) -> PackedQuery:
    """Build one exact E5 input without inspecting gold passage identities."""

    if type(max_length) is not int or max_length != E5_MAX_POSITIONS:
        raise ValueError(f"E5 max_length must be exact integer {E5_MAX_POSITIONS}")
    view = normalize_query_view(output_view)
    normalized_fit_views = tuple(normalize_query_view(item) for item in fit_views)
    if not normalized_fit_views or len(normalized_fit_views) != len(set(normalized_fit_views)):
        raise ValueError("fit_views must contain unique canonical query views")
    if view not in normalized_fit_views:
        raise ValueError("output_view must be included in fit_views")
    if tuple(fit_views) != normalized_fit_views:
        raise ValueError("fit_views must already be canonical")
    if getattr(tokenizer, "is_fast", None) is not True:
        raise TypeError("focus_preserving_pack requires the exact fast E5 tokenizer")
    if getattr(tokenizer, "model_max_length", None) != E5_MAX_POSITIONS:
        raise RuntimeError("E5 tokenizer model_max_length changed")
    if tokenizer.num_special_tokens_to_add(pair=False) != E5_SINGLE_SEQUENCE_SPECIAL_TOKENS:
        raise RuntimeError("E5 single-sequence special-token count changed")
    prefix_ids = _encode_without_specials(tokenizer, E5_QUERY_PREFIX)
    if prefix_ids != E5_QUERY_PREFIX_TOKEN_IDS:
        raise RuntimeError(
            f"E5 query prefix token IDs changed: actual={prefix_ids}, "
            f"expected={E5_QUERY_PREFIX_TOKEN_IDS}"
        )
    body_budget = (
        max_length - E5_SINGLE_SEQUENCE_SPECIAL_TOKENS - len(E5_QUERY_PREFIX_TOKEN_IDS)
    )
    if body_budget != 508:
        raise RuntimeError("E5 packed-query body budget changed")

    semantic = validate_query_renderings(query)
    content_prefixes, full_limits = _token_content_prefixes(
        semantic,
        tokenizer=tokenizer,
    )
    content_limits = {unit_id: 0 for unit_id in content_prefixes}
    focus_visible_ids = [
        _node_id("focus", 0, node_position)
        for node_position, node in enumerate(semantic.focus.nodes)
        if node.state == "visible"
    ]
    for unit_id in focus_visible_ids:
        content_limits[unit_id] = full_limits[unit_id]

    root_id = "root:0:0"
    if semantic.root.state == "visible":
        content_limits[root_id] = 1
    root_included = True

    lengths = _rendered_lengths(
        semantic,
        tokenizer=tokenizer,
        fit_views=normalized_fit_views,
        content_limits=content_limits,
        content_prefixes=content_prefixes,
        include_root=True,
        context_step_positions=(),
    )
    if not _fits(lengths, body_budget=body_budget):
        overflow_order = sorted(
            focus_visible_ids,
            key=lambda unit_id: (
                -content_prefixes[unit_id].wordpiece_counts[-1],
                unit_id,
            ),
        )
        for unit_id in overflow_order:
            if _maximize_unit_limit(
                unit_id,
                semantic=semantic,
                tokenizer=tokenizer,
                fit_views=normalized_fit_views,
                content_limits=content_limits,
                content_prefixes=content_prefixes,
                include_root=True,
                context_step_positions=(),
                body_budget=body_budget,
            ):
                lengths = _rendered_lengths(
                    semantic,
                    tokenizer=tokenizer,
                    fit_views=normalized_fit_views,
                    content_limits=content_limits,
                    content_prefixes=content_prefixes,
                    include_root=True,
                    context_step_positions=(),
                )
                if _fits(lengths, body_budget=body_budget):
                    break
            content_limits[unit_id] = 1
        lengths = _rendered_lengths(
            semantic,
            tokenizer=tokenizer,
            fit_views=normalized_fit_views,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
            include_root=True,
            context_step_positions=(),
        )
        if not _fits(lengths, body_budget=body_budget):
            raise RuntimeError(
                f"Required focus and root units cannot fit E5 for {query.query_id}"
            )

    if semantic.root.state == "visible":
        content_limits[root_id] = full_limits[root_id]
        root_lengths = _rendered_lengths(
            semantic,
            tokenizer=tokenizer,
            fit_views=normalized_fit_views,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
            include_root=True,
            context_step_positions=(),
        )
        if _fits(root_lengths, body_budget=body_budget):
            lengths = root_lengths
        elif not _maximize_unit_limit(
            root_id,
            semantic=semantic,
            tokenizer=tokenizer,
            fit_views=normalized_fit_views,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
            include_root=True,
            context_step_positions=(),
            body_budget=body_budget,
        ):
            raise RuntimeError(f"Root reservation was lost for {query.query_id}")
        else:
            lengths = _rendered_lengths(
                semantic,
                tokenizer=tokenizer,
                fit_views=normalized_fit_views,
                content_limits=content_limits,
                content_prefixes=content_prefixes,
                include_root=True,
                context_step_positions=(),
            )

    source_node_by_id = dict(_all_nodes(semantic))
    focus_context_position = _focus_context_step_position(semantic)
    context_step_positions: list[int] = []
    for step_position, step in enumerate(semantic.context_steps):
        if step_position == focus_context_position:
            continue
        step_ids = [
            _node_id("context", step_position, node_position)
            for node_position, _ in enumerate(step.nodes)
        ]
        for unit_id in step_ids:
            node = source_node_by_id[unit_id]
            if node.state == "visible":
                content_limits[unit_id] = full_limits[unit_id]
        candidate_positions = tuple((*context_step_positions, step_position))
        candidate_lengths = _rendered_lengths(
            semantic,
            tokenizer=tokenizer,
            fit_views=normalized_fit_views,
            content_limits=content_limits,
            content_prefixes=content_prefixes,
            include_root=True,
            context_step_positions=candidate_positions,
        )
        if not _fits(candidate_lengths, body_budget=body_budget):
            for unit_id in step_ids:
                if unit_id in content_limits:
                    content_limits[unit_id] = 0
            continue
        context_step_positions.append(step_position)
        lengths = candidate_lengths

    rendered_text = render_semantic_query(
        semantic,
        query_view=view,
        content_limits=content_limits,
        content_prefixes=content_prefixes,
        include_root=root_included,
        context_step_positions=tuple(context_step_positions),
    )
    body_ids = _encode_without_specials(tokenizer, rendered_text)
    if len(body_ids) != lengths[view] or len(body_ids) > body_budget:
        raise RuntimeError("Final packed body violates its exact token budget")
    sequence = tokenizer.build_inputs_with_special_tokens(
        [*E5_QUERY_PREFIX_TOKEN_IDS, *body_ids]
    )
    if (
        not isinstance(sequence, Sequence)
        or isinstance(sequence, (str, bytes))
        or any(type(value) is not int or value < 0 for value in sequence)
        or len(sequence) > max_length
    ):
        raise RuntimeError("E5 tokenizer built a malformed packed sequence")
    if tuple(sequence[1:3]) != E5_QUERY_PREFIX_TOKEN_IDS:
        raise RuntimeError("E5 special-token construction displaced the query prefix")

    selected_records = tuple(
        (
            unit_id,
            content_prefixes[unit_id].wordpiece_counts[
                content_limits[unit_id] - 1
            ],
            content_prefixes[unit_id].wordpiece_counts[-1],
        )
        for unit_id in sorted(full_limits)
        if content_limits.get(unit_id, 0) > 0
    )
    contract_payload = {
        "query_id": query.query_id,
        "protocol": FOCUS_PRESERVING_PACK_PROTOCOL,
        "output_view": view,
        "fit_views": list(normalized_fit_views),
        "input_ids": list(sequence),
        "selected_content_tokens": [list(record) for record in selected_records],
        "root_included": root_included,
        "context_step_positions": context_step_positions,
    }
    contract_sha256 = _sha256_canonical(contract_payload)
    if _LOWER_SHA256.fullmatch(contract_sha256) is None:
        raise RuntimeError("Packed-query contract did not produce SHA-256")
    return PackedQuery(
        query_id=query.query_id,
        protocol=FOCUS_PRESERVING_PACK_PROTOCOL,
        output_view=view,
        fit_views=normalized_fit_views,
        rendered_text=rendered_text,
        input_ids=tuple(sequence),
        selected_content_tokens=selected_records,
        root_included=root_included,
        context_step_positions=tuple(context_step_positions),
        contract_sha256=contract_sha256,
    )


def packed_query_inventory_sha256(packed_queries: Sequence[PackedQuery]) -> str:
    if not isinstance(packed_queries, Sequence) or isinstance(packed_queries, (str, bytes)):
        raise TypeError("packed_queries must be a sequence")
    records = []
    previous_query_id: str | None = None
    for position, packed in enumerate(packed_queries):
        if not isinstance(packed, PackedQuery):
            raise TypeError(f"packed_queries[{position}] must be PackedQuery")
        if previous_query_id is not None and packed.query_id <= previous_query_id:
            raise ValueError("Packed queries must be unique and sorted by query_id")
        records.append(
            {
                "query_id": packed.query_id,
                "contract_sha256": packed.contract_sha256,
            }
        )
        previous_query_id = packed.query_id
    if not records:
        raise ValueError("packed_queries must not be empty")
    return _sha256_canonical(records)
