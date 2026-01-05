"""
Builds a masked-slot, multi-positive retrieval dataset from Label Studio exports in final_annotations_gold.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

LABEL_RULE = "Rule"
LABEL_ANALYSIS = "Analysis"
LABEL_CONCLUSION = "Conclusion"
LABEL_BACKGROUND = "Background Facts"
LABEL_PROCEDURE = "Procedural History"

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
    source_file: str
    nodes_by_id: Dict[str, SpanNode]
    premise_ids_by_conclusion_id: Dict[str, List[str]]
    conclusion_ids_by_premise_id: Dict[str, List[str]]
    root_conclusion_ids: List[str]
    order_by_node_id: Dict[str, int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _iter_json_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.json") if p.is_file()])


def _safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_lines(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _copy_raw_exports(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in _iter_json_files(src_dir):
        shutil.copy2(src_path, dst_dir / src_path.name)


def _node_sort_key(node: SpanNode) -> Tuple[int, int, str]:
    start_key = node.start if node.start is not None else 10**18
    end_key = node.end if node.end is not None else 10**18
    return (start_key, end_key, node.node_id)


def _canonical_doc_id(raw_doc_id: Any, fallback_path: Path) -> str:
    if raw_doc_id is None:
        return fallback_path.stem.split("_", 1)[0]
    return str(raw_doc_id)


def _parse_case_graph(label_studio_path: Path) -> CaseGraph:
    export = _safe_read_json(label_studio_path)

    doc_id = _canonical_doc_id(export.get("id"), label_studio_path)
    result_items = export.get("result") or []

    nodes_by_id: Dict[str, SpanNode] = {}
    for item in result_items:
        if not isinstance(item, dict) or item.get("type") != "labels":
            continue
        node_id = str(item.get("id"))
        value = item.get("value") or {}
        label = (value.get("labels") or [None])[0]
        if not isinstance(label, str) or not label:
            continue
        text = value.get("text")
        start = value.get("start")
        end = value.get("end")
        nodes_by_id[node_id] = SpanNode(
            node_id=node_id,
            label=label,
            text=str(text or "").strip(),
            start=int(start) if isinstance(start, int) else None,
            end=int(end) if isinstance(end, int) else None,
        )

    premise_ids_by_conclusion_id: Dict[str, List[str]] = {}
    conclusion_ids_by_premise_id: Dict[str, List[str]] = {}
    for item in result_items:
        if not isinstance(item, dict) or item.get("type") != "relation":
            continue
        premise_id = item.get("from_id")
        conclusion_id = item.get("to_id")
        if not isinstance(premise_id, str) or not isinstance(conclusion_id, str):
            continue
        if premise_id not in nodes_by_id or conclusion_id not in nodes_by_id:
            continue
        premise_ids_by_conclusion_id.setdefault(conclusion_id, []).append(premise_id)
        conclusion_ids_by_premise_id.setdefault(premise_id, []).append(conclusion_id)

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

    ordered_nodes = sorted(
        [node for node in nodes_by_id.values() if not node.is_implicit],
        key=_node_sort_key,
    )
    order_by_node_id = {node.node_id: idx for idx, node in enumerate(ordered_nodes)}

    return CaseGraph(
        doc_id=doc_id,
        source_file=label_studio_path.name,
        nodes_by_id=nodes_by_id,
        premise_ids_by_conclusion_id=premise_ids_by_conclusion_id,
        conclusion_ids_by_premise_id=conclusion_ids_by_premise_id,
        root_conclusion_ids=root_conclusion_ids,
        order_by_node_id=order_by_node_id,
    )


def _format_node_for_query(node: SpanNode) -> str:
    label_token = LABEL_TOKEN_BY_LABEL.get(node.label, "[UNKNOWN]")
    if node.is_implicit:
        return f"[IMPLICIT] {label_token}"
    return f"{label_token} {node.text}".strip()


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
) -> Optional[Dict[str, int]]:
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
        return None

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
        depth = depths.get(conclusion_id, 0) if depths is not None else 0
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


def _build_queries_for_case(
    *,
    case_graph: CaseGraph,
    include_background_procedure_candidates: bool,
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    query_ids_seen: Set[str] = set()

    candidate_labels: Set[str] = {LABEL_RULE, LABEL_ANALYSIS, LABEL_CONCLUSION}
    if include_background_procedure_candidates:
        candidate_labels |= {LABEL_BACKGROUND, LABEL_PROCEDURE}

    corpus_node_ids = {
        node_id
        for node_id, node in case_graph.nodes_by_id.items()
        if (node.label in candidate_labels) and (not node.is_implicit)
    }

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
                    and (node.node_id in corpus_node_ids)
                ]
                if not maskable_nodes:
                    continue

                missing_node_ids = {node.node_id for node in maskable_nodes}
                excluded_node_ids = set(missing_node_ids)

                positive_passage_ids = [
                    f"{case_graph.doc_id}::{node.node_id}" for node in maskable_nodes
                ]
                positive_passage_ids = _dedupe_preserve_order(positive_passage_ids)
                if not positive_passage_ids:
                    continue

                positive_labels = [node.label for node in maskable_nodes]
                group_name = f"PREMISE_GROUP_{block_idx}"

                query_id = (
                    f"{case_graph.doc_id}::ROOT={motion_root_id}"
                    f"::TARGET={target_conclusion_id}::MISSING={group_name}"
                )
                if query_id in query_ids_seen:
                    continue
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
                    continue

                queries.append(
                    {
                        "query_id": query_id,
                        "doc_id": case_graph.doc_id,
                        "motion_root_id": motion_root_id,
                        "mask_parent_id": target_conclusion_id,
                        "query_text": query_text,
                        "positive_passage_ids": positive_passage_ids,
                        "positive_labels": positive_labels,
                    }
                )

        root_node = case_graph.nodes_by_id[motion_root_id]
        root_premise_nodes_in_order = _ordered_premise_nodes_for_conclusion(
            case_graph, conclusion_id=motion_root_id, motion_node_ids=motion_node_ids
        )
        if root_node.is_implicit or not root_premise_nodes_in_order:
            continue

        positive_passage_ids = [f"{case_graph.doc_id}::{motion_root_id}"]
        excluded_node_ids = {motion_root_id}
        missing_node_ids = {motion_root_id}

        query_id = f"{case_graph.doc_id}::ROOT={motion_root_id}::TARGET={motion_root_id}::MISSING=CONCLUSION"
        if query_id in query_ids_seen:
            continue
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
            continue

        queries.append(
            {
                "query_id": query_id,
                "doc_id": case_graph.doc_id,
                "motion_root_id": motion_root_id,
                "mask_parent_id": motion_root_id,
                "query_text": query_text,
                "positive_passage_ids": positive_passage_ids,
                "positive_labels": [LABEL_CONCLUSION],
            }
        )

    return queries


def _build_corpus_records(
    *,
    case_graphs: Sequence[CaseGraph],
    include_background_procedure_candidates: bool,
) -> List[Dict[str, Any]]:
    candidate_labels: Set[str] = {LABEL_RULE, LABEL_ANALYSIS, LABEL_CONCLUSION}
    if include_background_procedure_candidates:
        candidate_labels |= {LABEL_BACKGROUND, LABEL_PROCEDURE}

    records: List[Dict[str, Any]] = []
    for case_graph in case_graphs:
        nodes_in_order = sorted(
            [
                node
                for node in case_graph.nodes_by_id.values()
                if (node.label in candidate_labels) and (not node.is_implicit)
            ],
            key=_node_sort_key,
        )
        for node in nodes_in_order:
            label_token = LABEL_TOKEN_BY_LABEL.get(node.label, "[UNKNOWN]")
            passage_text = f"{label_token} {node.text}".strip()
            records.append(
                {
                    "passage_id": f"{case_graph.doc_id}::{node.node_id}",
                    "doc_id": case_graph.doc_id,
                    "label": node.label,
                    "text": passage_text,
                    "start": node.start,
                    "end": node.end,
                    "is_implicit": False,
                    "order": case_graph.order_by_node_id.get(node.node_id),
                }
            )

    def _corpus_order_key(rec: Dict[str, Any]) -> Tuple[str, int, str]:
        return (str(rec["doc_id"]), int(rec.get("order") or 10**18), str(rec["passage_id"]))

    records.sort(key=_corpus_order_key)
    return records


def _build_case_records(case_graphs: Sequence[CaseGraph]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case_graph in case_graphs:
        label_counts: Dict[str, int] = {}
        for node in case_graph.nodes_by_id.values():
            label_counts[node.label] = label_counts.get(node.label, 0) + 1

        records.append(
            {
                "doc_id": case_graph.doc_id,
                "source_file": case_graph.source_file,
                "num_nodes": len(case_graph.nodes_by_id),
                "num_relations": sum(
                    len(v) for v in case_graph.premise_ids_by_conclusion_id.values()
                ),
                "root_conclusion_ids": case_graph.root_conclusion_ids,
                "label_counts": label_counts,
            }
        )
    records.sort(key=lambda r: str(r["doc_id"]))
    return records


def _split_doc_ids(
    doc_ids: Sequence[str],
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    doc_ids = list(sorted(set(doc_ids)))
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    num_docs = len(doc_ids)
    num_val = int(round(num_docs * val_frac))
    num_test = int(round(num_docs * test_frac))
    num_train = max(0, num_docs - num_val - num_test)

    train_doc_ids = doc_ids[:num_train]
    val_doc_ids = doc_ids[num_train : num_train + num_val]
    test_doc_ids = doc_ids[num_train + num_val :]

    return sorted(train_doc_ids), sorted(val_doc_ids), sorted(test_doc_ids)


def _write_qrels(queries: Sequence[Dict[str, Any]], qrels_path: Path) -> None:
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with qrels_path.open("w", encoding="utf-8") as f:
        for q in queries:
            query_id = q["query_id"]
            for passage_id in q["positive_passage_ids"]:
                f.write(f"{query_id}\t{passage_id}\t1\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = _repo_root()
    default_raw_dir = repo_root / "corporate_reorganization/data/final_annotations_gold/raw"
    default_processed_dir = repo_root / "corporate_reorganization/data/final_annotations_gold/processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=default_raw_dir)
    parser.add_argument("--processed_dir", type=Path, default=default_processed_dir)
    parser.add_argument(
        "--copy_raw_from",
        type=Path,
        default=None,
        help="Optional: copy *.json exports from this folder into --raw_dir before processing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument(
        "--include_background_procedure_candidates",
        action="store_true",
        default=False,
        help="Include Background/Procedural spans as candidates (never positives).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    include_background_procedure_candidates = bool(args.include_background_procedure_candidates)

    if args.copy_raw_from is not None:
        _copy_raw_exports(args.copy_raw_from, args.raw_dir)

    raw_paths = _iter_json_files(args.raw_dir)
    if not raw_paths:
        raise SystemExit(f"No *.json files found under {args.raw_dir}")

    case_graphs = [_parse_case_graph(p) for p in raw_paths]
    corpus_records = _build_corpus_records(
        case_graphs=case_graphs,
        include_background_procedure_candidates=include_background_procedure_candidates,
    )
    case_records = _build_case_records(case_graphs)

    processed_dir = args.processed_dir
    _write_jsonl(corpus_records, processed_dir / "corpus.jsonl")
    _write_jsonl(case_records, processed_dir / "cases.jsonl")

    candidates_by_case: Dict[str, List[str]] = {}
    for record in corpus_records:
        candidates_by_case.setdefault(record["doc_id"], []).append(record["passage_id"])

    (processed_dir / "pools").mkdir(parents=True, exist_ok=True)
    with (processed_dir / "pools/candidates_by_case.json").open("w", encoding="utf-8") as f:
        json.dump(candidates_by_case, f, ensure_ascii=False, indent=2)

    candidates_global = [record["passage_id"] for record in corpus_records]
    with (processed_dir / "pools/candidates_global.json").open("w", encoding="utf-8") as f:
        json.dump(candidates_global, f, ensure_ascii=False, indent=2)

    doc_ids = [case_graph.doc_id for case_graph in case_graphs]
    train_doc_ids, val_doc_ids, test_doc_ids = _split_doc_ids(
        doc_ids, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac
    )

    splits_dir = processed_dir / "splits"
    _write_lines(train_doc_ids, splits_dir / "train_cases.txt")
    _write_lines(val_doc_ids, splits_dir / "val_cases.txt")
    _write_lines(test_doc_ids, splits_dir / "test_cases.txt")

    case_graph_by_doc_id = {case_graph.doc_id: case_graph for case_graph in case_graphs}
    queries_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, split_doc_ids in [
        ("train", train_doc_ids),
        ("val", val_doc_ids),
        ("test", test_doc_ids),
    ]:
        for doc_id in split_doc_ids:
            case_graph = case_graph_by_doc_id.get(doc_id)
            if case_graph is None:
                continue
            queries_by_split[split_name].extend(
                _build_queries_for_case(
                    case_graph=case_graph,
                    include_background_procedure_candidates=include_background_procedure_candidates,
                )
            )

    queries_dir = processed_dir / "queries"
    qrels_dir = processed_dir / "qrels"
    for split_name, queries in queries_by_split.items():
        queries_sorted = sorted(queries, key=lambda q: (q["doc_id"], q["query_id"]))
        _write_jsonl(queries_sorted, queries_dir / f"{split_name}.jsonl")
        _write_qrels(queries_sorted, qrels_dir / f"{split_name}.tsv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
