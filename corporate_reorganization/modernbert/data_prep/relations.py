"""Strict Label Studio relation normalization for retrieval data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Mapping, Sequence, Tuple


class RelationValidationError(ValueError):
    """Raised when a relation cannot be interpreted unambiguously."""


@dataclass(frozen=True, order=True)
class NormalizedRelation:
    """A directed argumentative edge from premise to conclusion."""

    premise_id: str
    conclusion_id: str
    direction: str
    result_index: int


def _relation_error(source_name: str, result_index: int, message: str) -> RelationValidationError:
    return RelationValidationError(f"{source_name}: result[{result_index}] {message}")


def normalize_relations(
    result_items: Sequence[object],
    *,
    node_ids: Collection[str],
    source_name: str,
) -> Tuple[NormalizedRelation, ...]:
    """Normalize every Label Studio relation to premise -> conclusion.

    Label Studio stores the visual direction separately from from_id and
    to_id. right means from -> to, while left means to -> from.
    Malformed, duplicate, and contradictory relations are rejected rather than
    skipped or guessed.
    """

    known_node_ids = frozenset(node_ids)
    edge_to_index: dict[tuple[str, str], int] = {}
    normalized: list[NormalizedRelation] = []

    for result_index, raw_item in enumerate(result_items):
        if not isinstance(raw_item, Mapping) or raw_item.get("type") != "relation":
            continue

        direction = raw_item.get("direction")
        if direction not in {"right", "left"}:
            raise _relation_error(
                source_name,
                result_index,
                f"has invalid direction {direction!r}; expected 'right' or 'left'",
            )

        from_id = raw_item.get("from_id")
        to_id = raw_item.get("to_id")
        for endpoint_name, endpoint in (("from_id", from_id), ("to_id", to_id)):
            if not isinstance(endpoint, str) or not endpoint:
                raise _relation_error(
                    source_name,
                    result_index,
                    f"has invalid {endpoint_name} {endpoint!r}",
                )
            if endpoint not in known_node_ids:
                raise _relation_error(
                    source_name,
                    result_index,
                    f"references unknown {endpoint_name} {endpoint!r}",
                )

        assert isinstance(from_id, str)
        assert isinstance(to_id, str)
        premise_id, conclusion_id = (
            (from_id, to_id) if direction == "right" else (to_id, from_id)
        )
        if premise_id == conclusion_id:
            raise _relation_error(
                source_name,
                result_index,
                f"contains self-edge {premise_id!r}",
            )

        edge = (premise_id, conclusion_id)
        reverse_edge = (conclusion_id, premise_id)
        if edge in edge_to_index:
            first_index = edge_to_index[edge]
            raise _relation_error(
                source_name,
                result_index,
                f"duplicates normalized edge {premise_id!r} -> {conclusion_id!r} "
                f"from result[{first_index}]",
            )
        if reverse_edge in edge_to_index:
            first_index = edge_to_index[reverse_edge]
            raise _relation_error(
                source_name,
                result_index,
                f"contradicts normalized edge {conclusion_id!r} -> {premise_id!r} "
                f"from result[{first_index}]",
            )

        edge_to_index[edge] = result_index
        normalized.append(
            NormalizedRelation(
                premise_id=premise_id,
                conclusion_id=conclusion_id,
                direction=direction,
                result_index=result_index,
            )
        )

    relations = tuple(
        sorted(normalized, key=lambda rel: (rel.premise_id, rel.conclusion_id, rel.result_index))
    )
    assert_acyclic(relations, node_ids=known_node_ids, source_name=source_name)
    return relations


def assert_acyclic(
    relations: Iterable[NormalizedRelation],
    *,
    node_ids: Collection[str],
    source_name: str,
) -> None:
    """Reject cycles, including cycles in disconnected graph components."""

    known_node_ids = frozenset(node_ids)
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in known_node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in known_node_ids}

    for relation in relations:
        if relation.premise_id not in known_node_ids or relation.conclusion_id not in known_node_ids:
            raise RelationValidationError(
                f"{source_name}: normalized relation references an unknown node: "
                f"{relation.premise_id!r} -> {relation.conclusion_id!r}"
            )
        outgoing[relation.premise_id].append(relation.conclusion_id)
        indegree[relation.conclusion_id] += 1

    queue = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_id = queue.pop(0)
        visited += 1
        for conclusion_id in sorted(outgoing[node_id]):
            indegree[conclusion_id] -= 1
            if indegree[conclusion_id] == 0:
                queue.append(conclusion_id)
                queue.sort()

    if visited != len(known_node_ids):
        blocked = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise RelationValidationError(
            f"{source_name}: relation graph contains a cycle involving nodes {blocked}"
        )
