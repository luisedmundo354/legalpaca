from __future__ import annotations

import unittest

from corporate_reorganization.modernbert.data_prep.relations import (
    NormalizedRelation,
    RelationValidationError,
    assert_acyclic,
    normalize_relations,
)


def relation(
    from_id: object = "a",
    to_id: object = "b",
    direction: object = "right",
) -> dict[str, object]:
    return {
        "type": "relation",
        "from_id": from_id,
        "to_id": to_id,
        "direction": direction,
    }


class NormalizeRelationsTest(unittest.TestCase):
    def test_right_and_left_are_normalized_to_premise_conclusion(self) -> None:
        normalized = normalize_relations(
            [relation(), relation("c", "b", "left")],
            node_ids={"a", "b", "c"},
            source_name="fixture.json",
        )
        self.assertEqual(
            [(item.premise_id, item.conclusion_id) for item in normalized],
            [("a", "b"), ("b", "c")],
        )

    def test_invalid_direction_fails(self) -> None:
        for direction in (None, "", "Right", "up", 1):
            with self.subTest(direction=direction):
                with self.assertRaisesRegex(RelationValidationError, "invalid direction"):
                    normalize_relations(
                        [relation(direction=direction)],
                        node_ids={"a", "b"},
                        source_name="fixture.json",
                    )

    def test_invalid_or_unknown_endpoint_fails(self) -> None:
        fixtures = [
            relation(None, "b"),
            relation("", "b"),
            relation(7, "b"),
            relation("missing", "b"),
            relation("a", None),
            relation("a", ""),
            relation("a", 7),
            relation("a", "missing"),
        ]
        for raw_relation in fixtures:
            with self.subTest(raw_relation=raw_relation):
                with self.assertRaises(RelationValidationError):
                    normalize_relations(
                        [raw_relation],
                        node_ids={"a", "b"},
                        source_name="fixture.json",
                    )

    def test_self_edge_fails_under_either_visual_direction(self) -> None:
        for direction in ("right", "left"):
            with self.subTest(direction=direction):
                with self.assertRaisesRegex(RelationValidationError, "self-edge"):
                    normalize_relations(
                        [relation("a", "a", direction)],
                        node_ids={"a"},
                        source_name="fixture.json",
                    )

    def test_exact_duplicate_fails(self) -> None:
        with self.assertRaisesRegex(RelationValidationError, "duplicates normalized edge"):
            normalize_relations(
                [relation(), relation()],
                node_ids={"a", "b"},
                source_name="fixture.json",
            )

    def test_reverse_edge_fails_as_contradiction(self) -> None:
        with self.assertRaisesRegex(RelationValidationError, "contradicts normalized edge"):
            normalize_relations(
                [relation(), relation("b", "a")],
                node_ids={"a", "b"},
                source_name="fixture.json",
            )

    def test_two_node_cycle_fails(self) -> None:
        with self.assertRaisesRegex(RelationValidationError, "contradicts normalized edge"):
            normalize_relations(
                [relation("a", "b"), relation("b", "a")],
                node_ids={"a", "b"},
                source_name="fixture.json",
            )

    def test_three_node_cycle_fails(self) -> None:
        with self.assertRaisesRegex(RelationValidationError, "contains a cycle"):
            normalize_relations(
                [
                    relation("a", "b"),
                    relation("b", "c"),
                    relation("c", "a"),
                ],
                node_ids={"a", "b", "c"},
                source_name="fixture.json",
            )

    def test_cycle_in_disconnected_component_fails(self) -> None:
        relations = (
            NormalizedRelation("a", "b", "right", 0),
            NormalizedRelation("x", "y", "right", 1),
            NormalizedRelation("y", "x", "right", 2),
        )
        with self.assertRaisesRegex(RelationValidationError, "contains a cycle"):
            assert_acyclic(
                relations,
                node_ids={"a", "b", "x", "y", "isolated"},
                source_name="fixture.json",
            )

    def test_valid_multi_root_dag_and_isolated_node_pass(self) -> None:
        normalized = normalize_relations(
            [relation("a", "root_1"), relation("b", "root_2")],
            node_ids={"a", "b", "root_1", "root_2", "isolated"},
            source_name="fixture.json",
        )
        self.assertEqual(len(normalized), 2)


if __name__ == "__main__":
    unittest.main()
