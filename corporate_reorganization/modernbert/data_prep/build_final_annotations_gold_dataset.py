"""
Builds a masked-slot, multi-positive retrieval dataset from Label Studio exports in final_annotations_gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..retriever.markup import all_markup_tokens
from .relations import NormalizedRelation, normalize_relations
from .sentence_splitter import SentenceSpan, split_sentences_with_offsets

LABEL_RULE = "Rule"
LABEL_ANALYSIS = "Analysis"
LABEL_CONCLUSION = "Conclusion"
LABEL_BACKGROUND = "Background Facts"
LABEL_PROCEDURE = "Procedural History"
LABEL_UNLABELED = "Unlabeled"

LABEL_TOKEN_BY_LABEL: Dict[str, str] = {
    LABEL_RULE: "[RULE]",
    LABEL_ANALYSIS: "[ANALYSIS]",
    LABEL_CONCLUSION: "[CONCLUSION]",
    LABEL_BACKGROUND: "[BACKGROUND]",
    LABEL_PROCEDURE: "[PROCEDURE]",
}

ALLOWED_POSITIVE_LABELS: Set[str] = {LABEL_RULE, LABEL_ANALYSIS, LABEL_CONCLUSION}
MISSING_MARKER = "[MISSING]"
SLOT_MARKER = "[MASK]"
FLAT_MISSING_TEXT = "missing"
PINNED_MODEL_ID = "answerdotai/ModernBERT-base"
PINNED_MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
PINNED_TRANSFORMERS_VERSION = "4.49.0"
PINNED_TOKENIZERS_VERSION = "0.21.4"
PINNED_TOKENIZER_FILES = {
    "config.json": {
        "bytes": 1_193,
        "sha256": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
    },
    "special_tokens_map.json": {
        "bytes": 694,
        "sha256": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
    },
    "tokenizer.json": {
        "bytes": 2_132_967,
        "sha256": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
    },
    "tokenizer_config.json": {
        "bytes": 20_810,
        "sha256": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
    },
}
MAX_QUERY_TOKENS = 4096

EXPECTED_CASES = 42
EXPECTED_NODES = 800
EXPECTED_PASSAGES = 5_286
EXPECTED_QUERIES = 490
EXPECTED_RELATIONS = 644
EXPECTED_ROOTS = 44
EXPECTED_POSITIVE_ASSIGNMENTS = 1_181
EXPECTED_DISTINCT_POSITIVE_PASSAGES = 1_080
EXPECTED_VISIBLE_ASSIGNMENTS = 8_223
EXPECTED_CASE_42_ROOT = "ENq9-QCWLD"
EXPECTED_CASE_42_QUERIES = 12
EXPECTED_CASE_42_HOLDING_PREFIX = (
    "For these reasons, this court concludes that the transfers in question failed to qualify"
)
EXPECTED_VISIBLE_GOLD_OVERLAPS = {
    (
        "78::ROOT=mHknhScxBS::TARGET=qWL5wevcmI::MISSING=PREMISE_GROUP_1",
        "78::SENT_00103",
    ),
    (
        "86::ROOT=TngQAxOF5Y::TARGET=bBERqsOSbo::MISSING=PREMISE_GROUP_1",
        "86::SENT_00110",
    ),
}


@dataclass(frozen=True)
class SpanNode:
    node_id: str
    label: str
    text: str
    start: Optional[int]
    end: Optional[int]

    @property
    def is_implicit(self) -> bool:
        return self.start is None or self.end is None


@dataclass(frozen=True)
class CaseGraph:
    doc_id: str
    ref_id: Optional[str]
    source_file: str
    case_text: str
    nodes_by_id: Dict[str, SpanNode]
    relations: Tuple[NormalizedRelation, ...]
    premise_ids_by_conclusion_id: Dict[str, List[str]]
    conclusion_ids_by_premise_id: Dict[str, List[str]]
    root_conclusion_ids: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_json_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.json") if p.is_file()])


def _safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a top-level JSON object")
    return value


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_sort_key(node: SpanNode) -> Tuple[int, int, str]:
    start_key = node.start if node.start is not None else 10**18
    end_key = node.end if node.end is not None else 10**18
    return (start_key, end_key, node.node_id)


def _canonical_doc_id(raw_doc_id: Any, *, source_name: str) -> str:
    if isinstance(raw_doc_id, bool) or not isinstance(raw_doc_id, (int, str)):
        raise ValueError(f"{source_name}: top-level id must be a non-empty integer or string")
    doc_id = str(raw_doc_id)
    if not doc_id or doc_id != doc_id.strip():
        raise ValueError(f"{source_name}: top-level id must not be empty")
    return doc_id


def _parse_case_graph(label_studio_path: Path) -> CaseGraph:
    export = _safe_read_json(label_studio_path)

    doc_id = _canonical_doc_id(export.get("id"), source_name=label_studio_path.name)
    if label_studio_path.stem != doc_id:
        raise ValueError(
            f"{label_studio_path.name}: filename stem does not match export id {doc_id!r}"
        )
    task = export.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"{label_studio_path.name}: task must be an object")
    task_data = task.get("data")
    if not isinstance(task_data, dict):
        raise ValueError(f"{label_studio_path.name}: task.data must be an object")
    case_text_raw = task_data.get("case_content")
    if not isinstance(case_text_raw, str) or not case_text_raw:
        raise ValueError(f"{label_studio_path.name}: task.data.case_content is empty")
    case_text = case_text_raw
    ref_id_raw = task_data.get("ref_id")
    ref_id = str(ref_id_raw) if ref_id_raw is not None else None
    result_items = export.get("result")
    if not isinstance(result_items, list):
        raise ValueError(f"{label_studio_path.name}: result must be a list")

    nodes_by_id: Dict[str, SpanNode] = {}
    for result_index, item in enumerate(result_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] must be an object"
            )
        item_type = item.get("type")
        if item_type == "relation":
            continue
        if item_type != "labels":
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] has unsupported type "
                f"{item_type!r}"
            )
        node_id_raw = item.get("id")
        if (
            not isinstance(node_id_raw, str)
            or not node_id_raw
            or node_id_raw != node_id_raw.strip()
        ):
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] has invalid node id "
                f"{node_id_raw!r}"
            )
        node_id = node_id_raw
        if node_id in nodes_by_id:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] duplicates node id "
                f"{node_id!r}"
            )
        value = item.get("value")
        if not isinstance(value, dict):
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}].value must be an object"
            )
        labels = value.get("labels")
        if not isinstance(labels, list) or len(labels) != 1:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] must have exactly one label"
            )
        label = labels[0]
        if label not in {
            LABEL_RULE,
            LABEL_ANALYSIS,
            LABEL_CONCLUSION,
            LABEL_BACKGROUND,
            LABEL_PROCEDURE,
        }:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] has unsupported label "
                f"{label!r}"
            )
        text = value.get("text")
        start = value.get("start")
        end = value.get("end")
        has_start = isinstance(start, int) and not isinstance(start, bool)
        has_end = isinstance(end, int) and not isinstance(end, bool)
        if start is not None and not has_start:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] has invalid start "
                f"{start!r}"
            )
        if end is not None and not has_end:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] has invalid end {end!r}"
            )
        if has_start != has_end:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] must provide both start "
                "and end, or neither for an implicit node"
            )
        if has_start and has_end and not (0 <= int(start) < int(end) <= len(case_text)):
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] has invalid offsets "
                f"{start!r}:{end!r}"
            )
        if has_start and (not isinstance(text, str) or not text.strip()):
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] explicit node has "
                "empty text"
            )
        if has_start and has_end and text != case_text[int(start) : int(end)]:
            raise ValueError(
                f"{label_studio_path.name}: result[{result_index}] text does not exactly "
                f"match case_content[{start}:{end}]"
            )
        nodes_by_id[node_id] = SpanNode(
            node_id=node_id,
            label=label,
            text=str(text or "").strip(),
            start=int(start) if has_start else None,
            end=int(end) if has_end else None,
        )

    relations = normalize_relations(
        result_items,
        node_ids=nodes_by_id,
        source_name=label_studio_path.name,
    )
    premise_ids_by_conclusion_id: Dict[str, List[str]] = {}
    conclusion_ids_by_premise_id: Dict[str, List[str]] = {}
    for relation in relations:
        premise_ids_by_conclusion_id.setdefault(relation.conclusion_id, []).append(
            relation.premise_id
        )
        conclusion_ids_by_premise_id.setdefault(relation.premise_id, []).append(
            relation.conclusion_id
        )

    premise_ids_by_conclusion_id = {
        k: sorted(set(v)) for k, v in premise_ids_by_conclusion_id.items()
    }
    conclusion_ids_by_premise_id = {
        k: sorted(set(v)) for k, v in conclusion_ids_by_premise_id.items()
    }

    out_degree_by_node_id = {
        node_id: len(conclusion_ids_by_premise_id.get(node_id, [])) for node_id in nodes_by_id
    }
    root_conclusion_ids = [
        node_id
        for node_id, node in nodes_by_id.items()
        if node.label == LABEL_CONCLUSION and out_degree_by_node_id.get(node_id, 0) == 0
    ]
    root_conclusion_ids.sort()
    if not root_conclusion_ids:
        raise ValueError(
            f"{label_studio_path.name}: case has no terminal Conclusion root after "
            "direction normalization"
        )

    return CaseGraph(
        doc_id=doc_id,
        ref_id=ref_id,
        source_file=label_studio_path.name,
        case_text=case_text,
        nodes_by_id=nodes_by_id,
        relations=relations,
        premise_ids_by_conclusion_id=premise_ids_by_conclusion_id,
        conclusion_ids_by_premise_id=conclusion_ids_by_premise_id,
        root_conclusion_ids=root_conclusion_ids,
    )


def _format_node_for_query(node: SpanNode) -> str:
    label_token = LABEL_TOKEN_BY_LABEL[node.label]
    if node.is_implicit:
        return f"[IMPLICIT] {label_token}"
    return f"{label_token} {node.text}".strip()


def _flat_label_name(label: str) -> str:
    return str(label).strip().lower()


def _format_node_for_flat_query(node: SpanNode) -> str:
    label_name = _flat_label_name(node.label)
    if node.is_implicit:
        return f"implicit {label_name}"
    return f"{label_name}: {node.text}".strip()


def _format_step_block(
    *,
    conclusion_node: SpanNode,
    premise_nodes_in_order: Sequence[SpanNode],
    excluded_node_ids: Set[str],
    missing_node_ids: Set[str],
    slot_location: Optional[str],
) -> str:
    lines: List[str] = ["[STEP]"]
    conclusion_is_hidden = conclusion_node.node_id in excluded_node_ids
    conclusion_is_slot = slot_location == "conclusion" and conclusion_node.node_id in missing_node_ids
    if conclusion_is_slot:
        lines.append(f"[CONCL] {SLOT_MARKER}")
    elif conclusion_is_hidden:
        lines.append(f"[CONCL] {MISSING_MARKER}")
    else:
        lines.append(f"[CONCL] {_format_node_for_query(conclusion_node)}")

    slot_inserted = False
    missing_placeholder_inserted = False
    for premise_node in premise_nodes_in_order:
        premise_is_missing = premise_node.node_id in missing_node_ids
        if premise_is_missing:
            if slot_location == "premise" and not slot_inserted:
                lines.append(f"[PREMISE] {SLOT_MARKER}")
                slot_inserted = True
                missing_placeholder_inserted = True
            elif not missing_placeholder_inserted:
                lines.append(f"[PREMISE] {MISSING_MARKER}")
                missing_placeholder_inserted = True
            continue
        if premise_node.node_id in excluded_node_ids:
            continue

        lines.append(f"[PREMISE] {_format_node_for_query(premise_node)}")

    if slot_location == "premise" and not slot_inserted:
        lines.append(f"[PREMISE] {SLOT_MARKER}")

    lines.append("[/STEP]")
    return "\n".join(lines)


def _format_flat_step_block(
    *,
    conclusion_node: SpanNode,
    premise_nodes_in_order: Sequence[SpanNode],
    excluded_node_ids: Set[str],
    missing_node_ids: Set[str],
    slot_location: Optional[str],
    slot_text: str,
) -> str:
    lines: List[str] = []
    conclusion_is_hidden = conclusion_node.node_id in excluded_node_ids
    conclusion_is_slot = slot_location == "conclusion" and conclusion_node.node_id in missing_node_ids
    if conclusion_is_slot:
        lines.append(f"conclusion: {slot_text}")
    elif conclusion_is_hidden:
        lines.append(f"conclusion: {FLAT_MISSING_TEXT}")
    else:
        lines.append(f"conclusion: {_format_node_for_flat_query(conclusion_node)}")

    slot_inserted = False
    missing_placeholder_inserted = False
    for premise_node in premise_nodes_in_order:
        premise_is_missing = premise_node.node_id in missing_node_ids
        if premise_is_missing:
            if slot_location == "premise" and not slot_inserted:
                lines.append(f"premise: {slot_text}")
                slot_inserted = True
                missing_placeholder_inserted = True
            elif not missing_placeholder_inserted:
                lines.append(f"premise: {FLAT_MISSING_TEXT}")
                missing_placeholder_inserted = True
            continue
        if premise_node.node_id in excluded_node_ids:
            continue
        lines.append(f"premise: {_format_node_for_flat_query(premise_node)}")

    if slot_location == "premise" and not slot_inserted:
        lines.append(f"premise: {slot_text}")
    return "\n".join(lines)


def _collect_motion_node_ids(
    *,
    root_conclusion_id: str,
    premise_ids_by_conclusion_id: Dict[str, List[str]],
) -> Set[str]:
    motion_node_ids: Set[str] = set()
    stack = [root_conclusion_id]
    while stack:
        node_id = stack.pop()
        if node_id in motion_node_ids:
            continue
        motion_node_ids.add(node_id)
        for premise_id in premise_ids_by_conclusion_id.get(node_id, []):
            stack.append(premise_id)
    return motion_node_ids


def _topological_depths(
    *,
    node_ids: Set[str],
    conclusion_ids_by_premise_id: Dict[str, List[str]],
    premise_ids_by_conclusion_id: Dict[str, List[str]],
) -> Dict[str, int]:
    in_degree_by_node_id: Dict[str, int] = {node_id: 0 for node_id in node_ids}
    for conclusion_id in node_ids:
        for premise_id in premise_ids_by_conclusion_id.get(conclusion_id, []):
            if premise_id in node_ids:
                in_degree_by_node_id[conclusion_id] += 1

    queue = [node_id for node_id, deg in in_degree_by_node_id.items() if deg == 0]
    queue.sort()
    topo_order: List[str] = []
    while queue:
        node_id = queue.pop(0)
        topo_order.append(node_id)
        for conclusion_id in conclusion_ids_by_premise_id.get(node_id, []):
            if conclusion_id not in node_ids:
                continue
            in_degree_by_node_id[conclusion_id] -= 1
            if in_degree_by_node_id[conclusion_id] == 0:
                queue.append(conclusion_id)
                queue.sort()

    if len(topo_order) != len(node_ids):
        blocked = sorted(node_id for node_id, degree in in_degree_by_node_id.items() if degree > 0)
        raise ValueError(f"Relation cycle reached query renderer: nodes={blocked}")

    depth_by_node_id = {node_id: 0 for node_id in node_ids}
    for premise_id in topo_order:
        for conclusion_id in conclusion_ids_by_premise_id.get(premise_id, []):
            if conclusion_id not in node_ids:
                continue
            depth_by_node_id[conclusion_id] = max(
                depth_by_node_id[conclusion_id], depth_by_node_id[premise_id] + 1
            )
    return depth_by_node_id


def _derive_step_conclusion_ids_for_motion(case_graph: CaseGraph, motion_node_ids: Set[str]) -> List[str]:
    derived_conclusion_ids: List[str] = []
    for node_id in motion_node_ids:
        premise_ids = case_graph.premise_ids_by_conclusion_id.get(node_id, [])
        if any(premise_id in motion_node_ids for premise_id in premise_ids):
            derived_conclusion_ids.append(node_id)
    return sorted(set(derived_conclusion_ids))


def _ordered_premise_nodes_for_conclusion(
    case_graph: CaseGraph, *, conclusion_id: str, motion_node_ids: Set[str]
) -> List[SpanNode]:
    premise_ids = [
        premise_id
        for premise_id in case_graph.premise_ids_by_conclusion_id.get(conclusion_id, [])
        if premise_id in motion_node_ids
    ]
    premise_nodes = [case_graph.nodes_by_id[premise_id] for premise_id in premise_ids]
    premise_nodes.sort(key=_node_sort_key)
    return premise_nodes


def _build_query_text(
    *,
    case_graph: CaseGraph,
    motion_root_id: str,
    target_conclusion_id: str,
    excluded_node_ids: Set[str],
    missing_node_ids: Set[str],
    focus_slot_location: str,
) -> str:
    motion_node_ids = _collect_motion_node_ids(
        root_conclusion_id=motion_root_id,
        premise_ids_by_conclusion_id=case_graph.premise_ids_by_conclusion_id,
    )
    derived_conclusion_ids = _derive_step_conclusion_ids_for_motion(case_graph, motion_node_ids)

    depths = _topological_depths(
        node_ids=motion_node_ids,
        conclusion_ids_by_premise_id=case_graph.conclusion_ids_by_premise_id,
        premise_ids_by_conclusion_id=case_graph.premise_ids_by_conclusion_id,
    )

    def _step_order_key(conclusion_id: str) -> Tuple[int, int, str]:
        node = case_graph.nodes_by_id[conclusion_id]
        depth = depths[conclusion_id]
        start_key = node.start if node.start is not None else 10**18
        return (depth, start_key, conclusion_id)

    ordered_step_conclusion_ids = sorted(derived_conclusion_ids, key=_step_order_key)

    tree_step_blocks: List[str] = []
    for conclusion_id in ordered_step_conclusion_ids:
        conclusion_node = case_graph.nodes_by_id[conclusion_id]
        premise_nodes = _ordered_premise_nodes_for_conclusion(
            case_graph, conclusion_id=conclusion_id, motion_node_ids=motion_node_ids
        )
        step_text = _format_step_block(
            conclusion_node=conclusion_node,
            premise_nodes_in_order=premise_nodes,
            excluded_node_ids=excluded_node_ids,
            missing_node_ids=missing_node_ids,
            slot_location=None,
        )
        if step_text:
            tree_step_blocks.append(step_text)

    focus_conclusion_node = case_graph.nodes_by_id[target_conclusion_id]
    focus_premise_nodes = _ordered_premise_nodes_for_conclusion(
        case_graph, conclusion_id=target_conclusion_id, motion_node_ids=motion_node_ids
    )
    focus_step_block = _format_step_block(
        conclusion_node=focus_conclusion_node,
        premise_nodes_in_order=focus_premise_nodes,
        excluded_node_ids=excluded_node_ids,
        missing_node_ids=missing_node_ids,
        slot_location=focus_slot_location,
    )

    root_node = case_graph.nodes_by_id[motion_root_id]
    root_is_hidden = motion_root_id in excluded_node_ids
    parts = [
        "[ARG]",
        f"[ROOT] {MISSING_MARKER}" if root_is_hidden else f"[ROOT] {_format_node_for_query(root_node)}",
        "[TREE]",
        "\n\n".join(tree_step_blocks).strip(),
        "[/TREE]",
        "[FOCUS]",
        focus_step_block.strip(),
        "[/FOCUS]",
        "[/ARG]",
    ]
    return "\n".join([p for p in parts if p])


def _build_flat_query_text(
    *,
    case_graph: CaseGraph,
    motion_root_id: str,
    target_conclusion_id: str,
    excluded_node_ids: Set[str],
    missing_node_ids: Set[str],
    focus_slot_location: str,
    slot_text: str,
) -> str:
    motion_node_ids = _collect_motion_node_ids(
        root_conclusion_id=motion_root_id,
        premise_ids_by_conclusion_id=case_graph.premise_ids_by_conclusion_id,
    )
    derived_conclusion_ids = _derive_step_conclusion_ids_for_motion(case_graph, motion_node_ids)

    depths = _topological_depths(
        node_ids=motion_node_ids,
        conclusion_ids_by_premise_id=case_graph.conclusion_ids_by_premise_id,
        premise_ids_by_conclusion_id=case_graph.premise_ids_by_conclusion_id,
    )

    def _step_order_key(conclusion_id: str) -> Tuple[int, int, str]:
        node = case_graph.nodes_by_id[conclusion_id]
        depth = depths[conclusion_id]
        start_key = node.start if node.start is not None else 10**18
        return (depth, start_key, conclusion_id)

    ordered_step_conclusion_ids = sorted(derived_conclusion_ids, key=_step_order_key)

    context_blocks: List[str] = []
    for conclusion_id in ordered_step_conclusion_ids:
        conclusion_node = case_graph.nodes_by_id[conclusion_id]
        premise_nodes = _ordered_premise_nodes_for_conclusion(
            case_graph, conclusion_id=conclusion_id, motion_node_ids=motion_node_ids
        )
        step_text = _format_flat_step_block(
            conclusion_node=conclusion_node,
            premise_nodes_in_order=premise_nodes,
            excluded_node_ids=excluded_node_ids,
            missing_node_ids=missing_node_ids,
            slot_location=None,
            slot_text=slot_text,
        )
        if step_text:
            context_blocks.append(step_text)

    focus_conclusion_node = case_graph.nodes_by_id[target_conclusion_id]
    focus_premise_nodes = _ordered_premise_nodes_for_conclusion(
        case_graph, conclusion_id=target_conclusion_id, motion_node_ids=motion_node_ids
    )
    focus_block = _format_flat_step_block(
        conclusion_node=focus_conclusion_node,
        premise_nodes_in_order=focus_premise_nodes,
        excluded_node_ids=excluded_node_ids,
        missing_node_ids=missing_node_ids,
        slot_location=focus_slot_location,
        slot_text=slot_text,
    )

    root_node = case_graph.nodes_by_id[motion_root_id]
    root_is_hidden = motion_root_id in excluded_node_ids
    root_line = f"root: {FLAT_MISSING_TEXT}" if root_is_hidden else f"root: {_format_node_for_flat_query(root_node)}"

    parts = [
        "argument",
        root_line,
        "",
        "context",
        "\n\n".join(block.strip() for block in context_blocks if block.strip()).strip(),
        "",
        "focus",
        focus_block.strip(),
    ]
    return "\n".join(part for part in parts if part is not None).strip()


def _contiguous_label_blocks(nodes_in_order: Sequence[SpanNode]) -> List[List[SpanNode]]:
    blocks: List[List[SpanNode]] = []
    for node in nodes_in_order:
        if not blocks or blocks[-1][-1].label != node.label:
            blocks.append([node])
        else:
            blocks[-1].append(node)
    return blocks


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _visibility_metadata(
    *,
    case_graph: CaseGraph,
    motion_root_id: str,
    excluded_node_ids: Set[str],
    visible_passage_ids_by_node_id: Dict[str, List[str]],
    query_text: str,
    flat_query_text_masked: str,
) -> Tuple[List[str], List[str]]:
    """Return source nodes and full sentence passages actually exposed by a query."""

    motion_node_ids = _collect_motion_node_ids(
        root_conclusion_id=motion_root_id,
        premise_ids_by_conclusion_id=case_graph.premise_ids_by_conclusion_id,
    )
    visible_nodes = [
        case_graph.nodes_by_id[node_id]
        for node_id in motion_node_ids
        if node_id not in excluded_node_ids
        and not case_graph.nodes_by_id[node_id].is_implicit
        and bool(case_graph.nodes_by_id[node_id].text)
    ]
    visible_nodes.sort(key=_node_sort_key)
    visible_node_ids = [node.node_id for node in visible_nodes]

    for node in visible_nodes:
        if node.text not in query_text or node.text not in flat_query_text_masked:
            raise ValueError(
                f"{case_graph.source_file}: visible node {node.node_id!r} was not emitted "
                "by both query renderers"
            )

    visible_passage_ids: List[str] = []
    for node_id in visible_node_ids:
        visible_passage_ids.extend(visible_passage_ids_by_node_id.get(node_id, []))
    return visible_node_ids, sorted(set(visible_passage_ids))


def _build_queries_for_case(
    *,
    case_graph: CaseGraph,
    sentence_passage_ids_by_node_id: Dict[str, List[str]],
    visible_passage_ids_by_node_id: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    query_ids_seen: Set[str] = set()

    for motion_root_id in case_graph.root_conclusion_ids:
        motion_node_ids = _collect_motion_node_ids(
            root_conclusion_id=motion_root_id,
            premise_ids_by_conclusion_id=case_graph.premise_ids_by_conclusion_id,
        )
        derived_conclusion_ids = _derive_step_conclusion_ids_for_motion(case_graph, motion_node_ids)

        for target_conclusion_id in derived_conclusion_ids:
            premise_nodes_in_order = _ordered_premise_nodes_for_conclusion(
                case_graph, conclusion_id=target_conclusion_id, motion_node_ids=motion_node_ids
            )
            if not premise_nodes_in_order:
                continue

            premise_blocks = _contiguous_label_blocks(premise_nodes_in_order)
            for block_idx, block_nodes in enumerate(premise_blocks, start=1):
                maskable_nodes = [
                    node
                    for node in block_nodes
                    if (not node.is_implicit)
                    and (node.label in ALLOWED_POSITIVE_LABELS)
                    and bool(sentence_passage_ids_by_node_id.get(node.node_id))
                ]
                if not maskable_nodes:
                    continue

                missing_node_ids = {node.node_id for node in maskable_nodes}
                excluded_node_ids = set(missing_node_ids)

                positive_sentence_ids: List[str] = []
                for node in maskable_nodes:
                    positive_sentence_ids.extend(sentence_passage_ids_by_node_id.get(node.node_id, []))

                positive_passage_ids = positive_sentence_ids
                positive_passage_ids = _dedupe_preserve_order(positive_passage_ids)
                if not positive_passage_ids:
                    raise ValueError(
                        f"{case_graph.source_file}: maskable query target has no positives"
                    )

                positive_labels = [node.label for node in maskable_nodes]
                group_name = f"PREMISE_GROUP_{block_idx}"

                query_id = (
                    f"{case_graph.doc_id}::ROOT={motion_root_id}"
                    f"::TARGET={target_conclusion_id}::MISSING={group_name}"
                )
                if query_id in query_ids_seen:
                    raise ValueError(f"{case_graph.source_file}: duplicate query id {query_id}")
                query_ids_seen.add(query_id)

                query_text = _build_query_text(
                    case_graph=case_graph,
                    motion_root_id=motion_root_id,
                    target_conclusion_id=target_conclusion_id,
                    excluded_node_ids=excluded_node_ids,
                    missing_node_ids=missing_node_ids,
                    focus_slot_location="premise",
                )
                if query_text.count(SLOT_MARKER) != 1:
                    raise ValueError(
                        f"{case_graph.source_file}: structured query must contain exactly one "
                        f"{SLOT_MARKER}: {query_id}"
                    )

                flat_query_text_plain = _build_flat_query_text(
                    case_graph=case_graph,
                    motion_root_id=motion_root_id,
                    target_conclusion_id=target_conclusion_id,
                    excluded_node_ids=excluded_node_ids,
                    missing_node_ids=missing_node_ids,
                    focus_slot_location="premise",
                    slot_text="missing span",
                )
                flat_query_text_masked = _build_flat_query_text(
                    case_graph=case_graph,
                    motion_root_id=motion_root_id,
                    target_conclusion_id=target_conclusion_id,
                    excluded_node_ids=excluded_node_ids,
                    missing_node_ids=missing_node_ids,
                    focus_slot_location="premise",
                    slot_text=SLOT_MARKER,
                )
                if SLOT_MARKER in flat_query_text_plain:
                    raise ValueError(f"flat_query_text_plain unexpectedly contains {SLOT_MARKER}: {query_id}")
                if flat_query_text_masked.count(SLOT_MARKER) != 1:
                    raise ValueError(
                        f"flat_query_text_masked must contain exactly one {SLOT_MARKER}: {query_id}"
                    )
                visible_node_ids, visible_passage_ids = _visibility_metadata(
                    case_graph=case_graph,
                    motion_root_id=motion_root_id,
                    excluded_node_ids=excluded_node_ids,
                    visible_passage_ids_by_node_id=visible_passage_ids_by_node_id,
                    query_text=query_text,
                    flat_query_text_masked=flat_query_text_masked,
                )
                visible_gold_overlap_passage_ids = sorted(
                    set(positive_passage_ids).intersection(visible_passage_ids)
                )

                queries.append(
                    {
                        "query_id": query_id,
                        "doc_id": case_graph.doc_id,
                        "motion_root_id": motion_root_id,
                        "mask_parent_id": target_conclusion_id,
                        "query_text": query_text,
                        "flat_query_text_plain": flat_query_text_plain,
                        "flat_query_text_masked": flat_query_text_masked,
                        "positive_passage_ids": positive_passage_ids,
                        "positive_labels": positive_labels,
                        "visible_node_ids": visible_node_ids,
                        "visible_passage_ids": visible_passage_ids,
                        "visible_gold_overlap_passage_ids": visible_gold_overlap_passage_ids,
                    }
                )

        root_node = case_graph.nodes_by_id[motion_root_id]
        root_premise_nodes_in_order = _ordered_premise_nodes_for_conclusion(
            case_graph, conclusion_id=motion_root_id, motion_node_ids=motion_node_ids
        )
        if root_node.is_implicit or not root_premise_nodes_in_order:
            continue

        positive_passage_ids = _dedupe_preserve_order(
            sentence_passage_ids_by_node_id.get(motion_root_id, [])
        )
        if not positive_passage_ids:
            continue
        excluded_node_ids = {motion_root_id}
        missing_node_ids = {motion_root_id}

        query_id = f"{case_graph.doc_id}::ROOT={motion_root_id}::TARGET={motion_root_id}::MISSING=CONCLUSION"
        if query_id in query_ids_seen:
            raise ValueError(f"{case_graph.source_file}: duplicate query id {query_id}")
        query_ids_seen.add(query_id)

        query_text = _build_query_text(
            case_graph=case_graph,
            motion_root_id=motion_root_id,
            target_conclusion_id=motion_root_id,
            excluded_node_ids=excluded_node_ids,
            missing_node_ids=missing_node_ids,
            focus_slot_location="conclusion",
        )
        if query_text.count(SLOT_MARKER) != 1:
            raise ValueError(
                f"{case_graph.source_file}: structured query must contain exactly one "
                f"{SLOT_MARKER}: {query_id}"
            )

        flat_query_text_plain = _build_flat_query_text(
            case_graph=case_graph,
            motion_root_id=motion_root_id,
            target_conclusion_id=motion_root_id,
            excluded_node_ids=excluded_node_ids,
            missing_node_ids=missing_node_ids,
            focus_slot_location="conclusion",
            slot_text="missing span",
        )
        flat_query_text_masked = _build_flat_query_text(
            case_graph=case_graph,
            motion_root_id=motion_root_id,
            target_conclusion_id=motion_root_id,
            excluded_node_ids=excluded_node_ids,
            missing_node_ids=missing_node_ids,
            focus_slot_location="conclusion",
            slot_text=SLOT_MARKER,
        )
        if SLOT_MARKER in flat_query_text_plain:
            raise ValueError(f"flat_query_text_plain unexpectedly contains {SLOT_MARKER}: {query_id}")
        if flat_query_text_masked.count(SLOT_MARKER) != 1:
            raise ValueError(
                f"flat_query_text_masked must contain exactly one {SLOT_MARKER}: {query_id}"
            )
        visible_node_ids, visible_passage_ids = _visibility_metadata(
            case_graph=case_graph,
            motion_root_id=motion_root_id,
            excluded_node_ids=excluded_node_ids,
            visible_passage_ids_by_node_id=visible_passage_ids_by_node_id,
            query_text=query_text,
            flat_query_text_masked=flat_query_text_masked,
        )
        visible_gold_overlap_passage_ids = sorted(
            set(positive_passage_ids).intersection(visible_passage_ids)
        )

        queries.append(
            {
                "query_id": query_id,
                "doc_id": case_graph.doc_id,
                "motion_root_id": motion_root_id,
                "mask_parent_id": motion_root_id,
                "query_text": query_text,
                "flat_query_text_plain": flat_query_text_plain,
                "flat_query_text_masked": flat_query_text_masked,
                "positive_passage_ids": positive_passage_ids,
                "positive_labels": [LABEL_CONCLUSION],
                "visible_node_ids": visible_node_ids,
                "visible_passage_ids": visible_passage_ids,
                "visible_gold_overlap_passage_ids": visible_gold_overlap_passage_ids,
            }
        )

    if not queries:
        raise ValueError(
            f"{case_graph.source_file}: case {case_graph.doc_id} produced zero retrieval queries"
        )
    return queries


def _overlap_len(*, start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    start = max(int(start_a), int(start_b))
    end = min(int(end_a), int(end_b))
    return max(0, end - start)


def _sentence_spans_for_case(case_graph: CaseGraph) -> List[SentenceSpan]:
    return split_sentences_with_offsets(case_graph.case_text)


def _build_sentence_corpus_records_for_case(
    case_graph: CaseGraph,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    nodes_with_positions = [
        node for node in case_graph.nodes_by_id.values() if (not node.is_implicit) and node.text
    ]
    nodes_with_positions.sort(key=_node_sort_key)

    sentence_spans = _sentence_spans_for_case(case_graph)
    sentence_passage_ids_by_node_id: Dict[str, List[str]] = {node.node_id: [] for node in nodes_with_positions}
    visible_passage_ids_by_node_id: Dict[str, List[str]] = {
        node.node_id: [] for node in nodes_with_positions
    }

    records: List[Dict[str, Any]] = []
    for sentence_idx, sent in enumerate(sentence_spans):
        passage_id = f"{case_graph.doc_id}::SENT_{sentence_idx:05d}"
        best_key: Optional[Tuple[int, int, int, str]] = None
        best_node_id: Optional[str] = None

        for node in nodes_with_positions:
            if node.start is None or node.end is None:
                continue
            if sent.text in node.text:
                visible_passage_ids_by_node_id.setdefault(node.node_id, []).append(passage_id)
            overlap = _overlap_len(start_a=sent.start, end_a=sent.end, start_b=node.start, end_b=node.end)
            if overlap <= 0:
                continue
            sentence_passage_ids_by_node_id.setdefault(node.node_id, []).append(passage_id)

            start_key = int(node.start) if node.start is not None else 10**18
            end_key = int(node.end) if node.end is not None else 10**18
            candidate_key = (-int(overlap), start_key, end_key, str(node.node_id))
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_node_id = node.node_id

        label = (
            case_graph.nodes_by_id[best_node_id].label if best_node_id is not None else LABEL_UNLABELED
        )
        records.append(
            {
                "passage_id": passage_id,
                "doc_id": case_graph.doc_id,
                "label": label,
                "text": sent.text,
                "start": int(sent.start),
                "end": int(sent.end),
                "source_node_id": best_node_id,
                "is_implicit": False,
                "order": int(sentence_idx),
            }
        )

    return records, sentence_passage_ids_by_node_id, visible_passage_ids_by_node_id


def _build_corpus_records(
    *,
    case_graphs: Sequence[CaseGraph],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[str]]],
]:
    records: List[Dict[str, Any]] = []
    sentence_passage_ids_by_node_id_by_doc_id: Dict[str, Dict[str, List[str]]] = {}
    visible_passage_ids_by_node_id_by_doc_id: Dict[str, Dict[str, List[str]]] = {}

    for case_graph in case_graphs:
        (
            case_records,
            sentence_ids_by_node_id,
            visible_ids_by_node_id,
        ) = _build_sentence_corpus_records_for_case(case_graph)
        records.extend(case_records)
        sentence_passage_ids_by_node_id_by_doc_id[case_graph.doc_id] = sentence_ids_by_node_id
        visible_passage_ids_by_node_id_by_doc_id[case_graph.doc_id] = visible_ids_by_node_id

    def _corpus_order_key(rec: Dict[str, Any]) -> Tuple[str, int, str]:
        order_raw = rec.get("order")
        order_key = int(order_raw) if order_raw is not None else 10**18
        return (str(rec["doc_id"]), order_key, str(rec["passage_id"]))

    records.sort(key=_corpus_order_key)
    return (
        records,
        sentence_passage_ids_by_node_id_by_doc_id,
        visible_passage_ids_by_node_id_by_doc_id,
    )


def _build_case_records(case_graphs: Sequence[CaseGraph]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case_graph in case_graphs:
        label_counts: Dict[str, int] = {}
        for node in case_graph.nodes_by_id.values():
            label_counts[node.label] = label_counts.get(node.label, 0) + 1

        num_sentences = len(_sentence_spans_for_case(case_graph))
        records.append(
            {
                "doc_id": case_graph.doc_id,
                "ref_id": case_graph.ref_id,
                "source_file": case_graph.source_file,
                "num_sentences": int(num_sentences),
                "num_nodes": len(case_graph.nodes_by_id),
                "num_relations": len(case_graph.relations),
                "root_conclusion_ids": case_graph.root_conclusion_ids,
                "relation_direction_counts": {
                    direction: sum(
                        relation.direction == direction for relation in case_graph.relations
                    )
                    for direction in ("left", "right")
                },
                "label_counts": label_counts,
            }
        )
    records.sort(key=lambda r: str(r["doc_id"]))
    return records


def _ensure_fresh_output_path(processed_dir: Path) -> None:
    if os.path.lexists(processed_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing output path: {processed_dir}"
        )


def _load_pinned_tokenizer(tokenizer_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Pinned tokenizer directory does not exist: {tokenizer_dir}")

    try:
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The corrected builder requires transformers==4.49.0 and "
            "tokenizers==0.21.4; install the exact pinned environment"
        ) from exc

    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Expected transformers=={PINNED_TRANSFORMERS_VERSION}, found "
            f"{transformers.__version__}"
        )
    if tokenizers.__version__ != PINNED_TOKENIZERS_VERSION:
        raise RuntimeError(
            f"Expected tokenizers=={PINNED_TOKENIZERS_VERSION}, found "
            f"{tokenizers.__version__}"
        )

    tokenizer_files = {}
    for filename, expected in PINNED_TOKENIZER_FILES.items():
        path = tokenizer_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned tokenizer input: {path}")
        actual = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if actual != expected:
            raise ValueError(
                f"Pinned tokenizer input mismatch for {filename}: "
                f"expected={expected}, found={actual}"
            )
        tokenizer_files[filename] = actual

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("Pinned ModernBERT tokenizer must be a fast tokenizer")
    markup_tokens = all_markup_tokens()
    tokenizer.add_special_tokens({"additional_special_tokens": markup_tokens})
    unknown = [
        token
        for token in markup_tokens
        if int(tokenizer.convert_tokens_to_ids(token)) == int(tokenizer.unk_token_id)
    ]
    if unknown:
        raise ValueError(f"Pinned tokenizer did not register markup tokens: {unknown}")
    tokenizer.truncation_side = "left"

    provenance = {
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_files": tokenizer_files,
        "additional_special_tokens": markup_tokens,
        "truncation_side": "left",
        "max_query_tokens": MAX_QUERY_TOKENS,
    }
    return tokenizer, provenance


def _audit_query_lengths(tokenizer: Any, queries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    view_fields = {
        "structured": "query_text",
        "flat_masked": "flat_query_text_masked",
    }
    maximum_by_view = {view: 0 for view in view_fields}
    overlong_by_view: Dict[str, List[str]] = {view: [] for view in view_fields}

    slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_MARKER))
    for query in queries:
        token_counts: Dict[str, int] = {}
        for view, field in view_fields.items():
            text = str(query[field])
            full_ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            token_count = len(full_ids)
            token_counts[view] = token_count
            maximum_by_view[view] = max(maximum_by_view[view], token_count)
            if token_count > MAX_QUERY_TOKENS:
                overlong_by_view[view].append(str(query["query_id"]))

            truncated_ids = tokenizer(
                text,
                truncation=True,
                max_length=MAX_QUERY_TOKENS,
                add_special_tokens=True,
            )["input_ids"]
            if truncated_ids.count(slot_token_id) != 1:
                raise ValueError(
                    f"{query['query_id']}: {view} query loses or duplicates {SLOT_MARKER} "
                    f"under {MAX_QUERY_TOKENS}-token left truncation"
                )
        query["model_input_token_counts"] = token_counts

    if any(overlong_by_view.values()):
        raise ValueError(
            "Canonical corrected queries unexpectedly require truncation; "
            f"overlong_query_ids={overlong_by_view}"
        )

    return {
        "maximum_tokens_by_view": maximum_by_view,
        "queries_over_limit_by_view": {
            view: len(query_ids) for view, query_ids in overlong_by_view.items()
        },
        "all_visible_passages_survive": True,
        "visible_passage_assignments_lost_by_view": {
            view: 0 for view in view_fields
        },
    }


def _validate_canonical_dataset(
    *,
    case_graphs: Sequence[CaseGraph],
    corpus_records: Sequence[Dict[str, Any]],
    queries: Sequence[Dict[str, Any]],
    candidates_by_case: Dict[str, List[str]],
    candidates_global: Sequence[str],
) -> Dict[str, Any]:
    if len(case_graphs) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} cases, found {len(case_graphs)}")
    node_count = sum(len(case_graph.nodes_by_id) for case_graph in case_graphs)
    if node_count != EXPECTED_NODES:
        raise ValueError(f"Expected {EXPECTED_NODES} nodes, found {node_count}")
    if len(corpus_records) != EXPECTED_PASSAGES:
        raise ValueError(
            f"Expected {EXPECTED_PASSAGES} passages, found {len(corpus_records)}"
        )
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"Expected {EXPECTED_QUERIES} queries, found {len(queries)}")

    relation_count = sum(len(case_graph.relations) for case_graph in case_graphs)
    root_count = sum(len(case_graph.root_conclusion_ids) for case_graph in case_graphs)
    if relation_count != EXPECTED_RELATIONS:
        raise ValueError(f"Expected {EXPECTED_RELATIONS} relations, found {relation_count}")
    if root_count != EXPECTED_ROOTS:
        raise ValueError(f"Expected {EXPECTED_ROOTS} roots, found {root_count}")

    doc_ids = [case_graph.doc_id for case_graph in case_graphs]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate document IDs found after graph parsing")
    passage_ids = [str(record["passage_id"]) for record in corpus_records]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Duplicate passage IDs found")
    query_ids = [str(query["query_id"]) for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs found")
    if list(candidates_global) != passage_ids:
        raise ValueError("Global candidate pool must contain every passage exactly once")

    passage_id_set = set(passage_ids)
    query_counts_by_case = {doc_id: 0 for doc_id in doc_ids}
    overlap_pairs: Set[Tuple[str, str]] = set()
    positive_assignments = 0
    visible_assignments = 0
    distinct_positive_ids: Set[str] = set()
    for query in queries:
        query_id = str(query["query_id"])
        doc_id = str(query["doc_id"])
        query_counts_by_case[doc_id] += 1
        same_case_pool = set(candidates_by_case[doc_id])
        positive_ids = [str(value) for value in query["positive_passage_ids"]]
        visible_ids = [str(value) for value in query["visible_passage_ids"]]
        if not positive_ids:
            raise ValueError(f"{query_id}: empty positive_passage_ids")
        if not set(positive_ids).issubset(same_case_pool):
            raise ValueError(f"{query_id}: positive passage outside same-case pool")
        if not set(visible_ids).issubset(same_case_pool):
            raise ValueError(f"{query_id}: visible passage outside same-case pool")
        if not set(positive_ids).issubset(passage_id_set):
            raise ValueError(f"{query_id}: positive passage outside global pool")
        if not set(visible_ids).issubset(passage_id_set):
            raise ValueError(f"{query_id}: visible passage outside global pool")

        expected_overlap = sorted(set(positive_ids).intersection(visible_ids))
        if expected_overlap != query["visible_gold_overlap_passage_ids"]:
            raise ValueError(f"{query_id}: inconsistent visible/gold overlap diagnostic")
        overlap_pairs.update((query_id, passage_id) for passage_id in expected_overlap)
        positive_assignments += len(positive_ids)
        visible_assignments += len(visible_ids)
        distinct_positive_ids.update(positive_ids)

    zero_query_cases = sorted(
        doc_id for doc_id, count in query_counts_by_case.items() if count == 0
    )
    if zero_query_cases:
        raise ValueError(f"Cases with zero retrieval queries: {zero_query_cases}")
    if overlap_pairs != EXPECTED_VISIBLE_GOLD_OVERLAPS:
        raise ValueError(
            "Visible/gold overlap gate changed: "
            f"expected={sorted(EXPECTED_VISIBLE_GOLD_OVERLAPS)}, "
            f"found={sorted(overlap_pairs)}"
        )
    if positive_assignments != EXPECTED_POSITIVE_ASSIGNMENTS:
        raise ValueError(
            f"Expected {EXPECTED_POSITIVE_ASSIGNMENTS} positive assignments, "
            f"found {positive_assignments}"
        )
    if len(distinct_positive_ids) != EXPECTED_DISTINCT_POSITIVE_PASSAGES:
        raise ValueError(
            f"Expected {EXPECTED_DISTINCT_POSITIVE_PASSAGES} distinct positive passages, "
            f"found {len(distinct_positive_ids)}"
        )
    if visible_assignments != EXPECTED_VISIBLE_ASSIGNMENTS:
        raise ValueError(
            f"Expected {EXPECTED_VISIBLE_ASSIGNMENTS} visible assignments, "
            f"found {visible_assignments}"
        )

    case_42 = next(case_graph for case_graph in case_graphs if case_graph.doc_id == "42")
    if case_42.root_conclusion_ids != [EXPECTED_CASE_42_ROOT]:
        raise ValueError(
            f"Case 42 root mismatch: expected {[EXPECTED_CASE_42_ROOT]}, "
            f"found {case_42.root_conclusion_ids}"
        )
    case_42_holding = case_42.nodes_by_id[EXPECTED_CASE_42_ROOT].text
    if not case_42_holding.startswith(EXPECTED_CASE_42_HOLDING_PREFIX):
        raise ValueError(
            "Case 42 root is not the expected final holding: "
            f"{case_42_holding[:120]!r}"
        )
    if query_counts_by_case["42"] != EXPECTED_CASE_42_QUERIES:
        raise ValueError(
            f"Case 42 query mismatch: expected {EXPECTED_CASE_42_QUERIES}, "
            f"found {query_counts_by_case['42']}"
        )

    direction_counts = {
        direction: sum(
            relation.direction == direction
            for case_graph in case_graphs
            for relation in case_graph.relations
        )
        for direction in ("left", "right")
    }
    if direction_counts != {"left": 8, "right": 636}:
        raise ValueError(f"Relation direction counts changed: {direction_counts}")

    passage_counts_by_case = {
        doc_id: len(candidates_by_case[doc_id]) for doc_id in sorted(doc_ids)
    }
    return {
        "relation_direction_counts": direction_counts,
        "query_counts_by_case": {
            doc_id: query_counts_by_case[doc_id] for doc_id in sorted(doc_ids)
        },
        "passage_counts_by_case": passage_counts_by_case,
        "positive_passage_assignments": positive_assignments,
        "distinct_positive_passages": len(distinct_positive_ids),
        "visible_passage_assignments": visible_assignments,
        "visible_gold_overlap_pairs": [
            {"query_id": query_id, "passage_id": passage_id}
            for query_id, passage_id in sorted(overlap_pairs)
        ],
    }


def _file_record(path: Path, *, records: Optional[int] = None) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        record["records"] = int(records)
    return record


def build_dataset(
    *,
    raw_dir: Path,
    processed_dir: Path,
    tokenizer_dir: Path,
) -> Dict[str, Any]:
    _ensure_fresh_output_path(processed_dir)

    raw_paths = _iter_json_files(raw_dir)
    if not raw_paths:
        raise ValueError(f"No JSON annotation files found under {raw_dir}")
    case_graphs = [_parse_case_graph(path) for path in raw_paths]
    if len({case_graph.doc_id for case_graph in case_graphs}) != len(case_graphs):
        raise ValueError("Duplicate document IDs found across raw annotation files")

    (
        corpus_records,
        sentence_ids_by_node_by_doc,
        visible_ids_by_node_by_doc,
    ) = _build_corpus_records(case_graphs=case_graphs)
    case_records = _build_case_records(case_graphs)

    candidates_by_case: Dict[str, List[str]] = {
        case_graph.doc_id: [] for case_graph in case_graphs
    }
    for record in corpus_records:
        candidates_by_case[str(record["doc_id"])].append(str(record["passage_id"]))
    candidates_by_case = {
        doc_id: passage_ids
        for doc_id, passage_ids in sorted(candidates_by_case.items())
    }
    candidates_global = [str(record["passage_id"]) for record in corpus_records]

    queries: List[Dict[str, Any]] = []
    for case_graph in sorted(case_graphs, key=lambda graph: graph.doc_id):
        queries.extend(
            _build_queries_for_case(
                case_graph=case_graph,
                sentence_passage_ids_by_node_id=sentence_ids_by_node_by_doc[
                    case_graph.doc_id
                ],
                visible_passage_ids_by_node_id=visible_ids_by_node_by_doc[
                    case_graph.doc_id
                ],
            )
        )
    queries.sort(key=lambda query: (str(query["doc_id"]), str(query["query_id"])))

    tokenizer, tokenizer_provenance = _load_pinned_tokenizer(tokenizer_dir)
    truncation_diagnostics = _audit_query_lengths(tokenizer, queries)
    dataset_diagnostics = _validate_canonical_dataset(
        case_graphs=case_graphs,
        corpus_records=corpus_records,
        queries=queries,
        candidates_by_case=candidates_by_case,
        candidates_global=candidates_global,
    )

    processed_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "cases.jsonl": processed_dir / "cases.jsonl",
        "corpus.jsonl": processed_dir / "corpus.jsonl",
        "queries/all.jsonl": processed_dir / "queries/all.jsonl",
        "pools/candidates_by_case.json": processed_dir / "pools/candidates_by_case.json",
        "pools/candidates_global.json": processed_dir / "pools/candidates_global.json",
    }
    _write_jsonl(case_records, output_paths["cases.jsonl"])
    _write_jsonl(corpus_records, output_paths["corpus.jsonl"])
    _write_jsonl(queries, output_paths["queries/all.jsonl"])
    _write_json(candidates_by_case, output_paths["pools/candidates_by_case.json"])
    _write_json(candidates_global, output_paths["pools/candidates_global.json"])

    raw_file_records = {
        path.name: _file_record(path) for path in raw_paths
    }
    code_paths = {
        "data_prep/build_final_annotations_gold_dataset.py": Path(__file__).resolve(),
        "data_prep/relations.py": Path(__file__).with_name("relations.py").resolve(),
        "data_prep/sentence_splitter.py": Path(__file__).with_name(
            "sentence_splitter.py"
        ).resolve(),
        "retriever/markup.py": Path(__file__).resolve().parents[1]
        / "retriever"
        / "markup.py",
    }
    source_code_records = {
        name: _file_record(path) for name, path in sorted(code_paths.items())
    }
    output_file_records = {
        "cases.jsonl": _file_record(output_paths["cases.jsonl"], records=len(case_records)),
        "corpus.jsonl": _file_record(
            output_paths["corpus.jsonl"], records=len(corpus_records)
        ),
        "queries/all.jsonl": _file_record(
            output_paths["queries/all.jsonl"], records=len(queries)
        ),
        "pools/candidates_by_case.json": _file_record(
            output_paths["pools/candidates_by_case.json"],
            records=len(candidates_by_case),
        ),
        "pools/candidates_global.json": _file_record(
            output_paths["pools/candidates_global.json"],
            records=len(candidates_global),
        ),
    }

    manifest = {
        "schema_version": 2,
        "document_id_source": "label_studio_export.id",
        "counts": {
            "cases": len(case_graphs),
            "nodes": sum(len(case_graph.nodes_by_id) for case_graph in case_graphs),
            "relations": sum(len(case_graph.relations) for case_graph in case_graphs),
            "roots": sum(
                len(case_graph.root_conclusion_ids) for case_graph in case_graphs
            ),
            "passages": len(corpus_records),
            "queries": len(queries),
        },
        "raw_annotation_files": raw_file_records,
        "source_code": source_code_records,
        "tokenizer": tokenizer_provenance,
        "diagnostics": {
            **dataset_diagnostics,
            "truncation": truncation_diagnostics,
        },
        "output_files": output_file_records,
        "manifest_hash_policy": "dataset_manifest.json excludes its own hash",
    }
    _write_json(manifest, processed_dir / "dataset_manifest.json")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = _repo_root()
    default_raw_dir = repo_root / "corporate_reorganization/data/final_annotations_gold/raw"
    default_processed_dir = (
        repo_root
        / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
    )

    parser = argparse.ArgumentParser(
        description="Build the immutable direction-corrected retrieval dataset."
    )
    parser.add_argument("--raw_dir", type=Path, default=default_raw_dir)
    parser.add_argument("--processed_dir", type=Path, default=default_processed_dir)
    parser.add_argument(
        "--tokenizer_dir",
        type=Path,
        required=True,
        help=(
            "Local snapshot directory for answerdotai/ModernBERT-base revision "
            f"{PINNED_MODEL_REVISION}; network lookup is never attempted."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    build_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        tokenizer_dir=args.tokenizer_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
