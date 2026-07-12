from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.data import load_queries  # noqa: E402
from retriever.e5_pack_artifact import (  # noqa: E402
    validate_e5_pack_artifact,
)
from retriever.query_packing import (  # noqa: E402
    E5_QUERY_PREFIX_TOKEN_IDS,
    FOCUS_PRESERVING_PACK_PROTOCOL,
    pack_focus_preserving_query,
    validate_query_renderings,
)


DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
PACK_DIR = MODERNBERT_DIR / "experiments/retrieval_cv/configs/e5_focus_pack"
PACK_MANIFEST_SHA256 = "9875bd57c23a7e390c85d2a4b1b3aab7415597c0223c2fed621e613d4dfded10"
PACK_INVENTORY_SHA256 = "9cfe6cbd83c60a686751c82d1c811612a27eb5a04d835a1a600335081f5b1edf"


class E5SemanticPackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.queries = tuple(sorted(load_queries(DATASET_DIR, "all"), key=lambda query: query.query_id))
        cls.rows = tuple(
            json.loads(line)
            for line in (PACK_DIR / "packed_queries.jsonl").read_text(encoding="utf-8").splitlines()
        )

    def test_all_frozen_query_renderings_reconstruct_exactly(self) -> None:
        self.assertEqual(len(self.queries), 490)
        for query in self.queries:
            with self.subTest(query_id=query.query_id):
                validate_query_renderings(query)

    def test_committed_pack_has_exact_counts_prefixes_and_overflow_allocations(self) -> None:
        self.assertEqual(len(self.rows), 490)
        self.assertEqual(
            hashlib.sha256((PACK_DIR / "manifest.json").read_bytes()).hexdigest(),
            PACK_MANIFEST_SHA256,
        )
        manifest = json.loads((PACK_DIR / "manifest.json").read_bytes())
        self.assertEqual(manifest["packing_protocol"], FOCUS_PRESERVING_PACK_PROTOCOL)
        self.assertEqual(manifest["packed_query_inventory_sha256"], PACK_INVENTORY_SHA256)

        focus_truncated = []
        root_truncated = []
        context_queries = 0
        context_steps = 0
        for row in self.rows:
            self.assertLessEqual(len(row["input_ids"]), 512)
            self.assertEqual(tuple(row["input_ids"][1:3]), E5_QUERY_PREFIX_TOKEN_IDS)
            self.assertTrue(row["root_included"])
            selection = {
                record["unit_id"]: (record["selected"], record["full"])
                for record in row["selected_content_tokens"]
            }
            truncated_focus = {
                unit_id: values
                for unit_id, values in selection.items()
                if unit_id.startswith("focus:") and values[0] < values[1]
            }
            if truncated_focus:
                focus_truncated.append((row["query_id"], truncated_focus))
            if (
                "root:0:0" in selection
                and selection["root:0:0"][0] < selection["root:0:0"][1]
            ):
                root_truncated.append(row["query_id"])
            context_queries += bool(row["context_step_positions"])
            context_steps += len(row["context_step_positions"])
        self.assertEqual(len(focus_truncated), 3)
        self.assertEqual(len(root_truncated), 11)
        self.assertEqual(context_queries, 409)
        self.assertEqual(context_steps, 1_230)
        self.assertEqual(
            focus_truncated,
            [
                (
                    "66::ROOT=nbHYSFiiqR::TARGET=iMoPeOPevl::MISSING=PREMISE_GROUP_2",
                    {"focus:0:1": (141, 285)},
                ),
                (
                    "66::ROOT=nbHYSFiiqR::TARGET=iMoPeOPevl::MISSING=PREMISE_GROUP_3",
                    {"focus:0:1": (240, 285)},
                ),
                (
                    "71::ROOT=mbEa5XhZe0::TARGET=mbEa5XhZe0::MISSING=PREMISE_GROUP_1",
                    {"focus:0:3": (216, 238)},
                ),
            ],
        )

    def test_packer_source_has_no_positive_field_access(self) -> None:
        source = inspect.getsource(pack_focus_preserving_query)
        self.assertNotIn("positive_passage_ids", source)
        self.assertNotIn("positive_labels", source)

    @unittest.skipUnless(os.environ.get("ARR_E5_TOKENIZER_DIR"), "ARR_E5_TOKENIZER_DIR is absent")
    def test_exact_tokenizer_regeneration_and_positive_independence(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["ARR_E5_TOKENIZER_DIR"],
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        validated = validate_e5_pack_artifact(
            PACK_DIR,
            expected_manifest_sha256=PACK_MANIFEST_SHA256,
            queries=self.queries,
            tokenizer=tokenizer,
        )
        self.assertEqual(validated.packed_query_inventory_sha256, PACK_INVENTORY_SHA256)
        overflow_query = next(
            query
            for query in self.queries
            if query.query_id.endswith("TARGET=iMoPeOPevl::MISSING=PREMISE_GROUP_2")
        )
        mutated = dataclasses.replace(
            overflow_query,
            positive_passage_ids=["foreign::positive"],
            positive_labels=["Procedure", "Background Facts"],
        )
        self.assertEqual(
            pack_focus_preserving_query(overflow_query, tokenizer=tokenizer).input_ids,
            pack_focus_preserving_query(mutated, tokenizer=tokenizer).input_ids,
        )


if __name__ == "__main__":
    unittest.main()
