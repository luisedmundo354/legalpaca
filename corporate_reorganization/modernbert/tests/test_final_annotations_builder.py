from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from corporate_reorganization.modernbert.data_prep import (
    build_final_annotations_gold_dataset as builder,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_export(
    path: Path,
    *,
    doc_id: int,
    result: list[dict[str, object]],
    case_text: str = "Alpha.",
) -> None:
    path.write_text(
        json.dumps(
            {
                "id": doc_id,
                "task": {"data": {"case_content": case_text, "ref_id": 999}},
                "result": result,
            }
        ),
        encoding="utf-8",
    )


def implicit_node(node_id: str, label: str) -> dict[str, object]:
    return {
        "id": node_id,
        "type": "labels",
        "value": {"labels": [label], "text": ""},
    }


def explicit_node(
    node_id: str,
    label: str,
    *,
    text: str,
    start: int,
    end: int,
) -> dict[str, object]:
    return {
        "id": node_id,
        "type": "labels",
        "value": {
            "labels": [label],
            "text": text,
            "start": start,
            "end": end,
        },
    }


class BuilderInvariantTest(unittest.TestCase):
    def test_unknown_label_fails_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            write_export(
                path,
                doc_id=1,
                result=[implicit_node("root", "Unknown Label")],
            )
            with self.assertRaisesRegex(ValueError, "unsupported label"):
                builder._parse_case_graph(path)

    def test_rootless_case_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            write_export(
                path,
                doc_id=1,
                result=[implicit_node("rule", builder.LABEL_RULE)],
            )
            with self.assertRaisesRegex(ValueError, "no terminal Conclusion root"):
                builder._parse_case_graph(path)

    def test_multiple_roots_and_isolated_annotations_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            write_export(
                path,
                doc_id=1,
                result=[
                    implicit_node("root_b", builder.LABEL_CONCLUSION),
                    implicit_node("root_a", builder.LABEL_CONCLUSION),
                    implicit_node("isolated", builder.LABEL_RULE),
                ],
            )
            graph = builder._parse_case_graph(path)
            self.assertEqual(graph.root_conclusion_ids, ["root_a", "root_b"])
            self.assertEqual(graph.ref_id, "999")

    def test_explicit_text_must_exactly_match_annotated_source_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            write_export(
                path,
                doc_id=1,
                case_text="Alpha.",
                result=[
                    explicit_node(
                        "root",
                        builder.LABEL_CONCLUSION,
                        text="Other",
                        start=0,
                        end=5,
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "text does not exactly match"):
                builder._parse_case_graph(path)

    def test_zero_query_case_fails_without_requiring_each_root_to_yield(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            write_export(
                path,
                doc_id=1,
                result=[implicit_node("root", builder.LABEL_CONCLUSION)],
            )
            graph = builder._parse_case_graph(path)
            with self.assertRaisesRegex(ValueError, "produced zero retrieval queries"):
                builder._build_queries_for_case(
                    case_graph=graph,
                    sentence_passage_ids_by_node_id={},
                    visible_passage_ids_by_node_id={},
                )

    def test_existing_file_directory_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_file = root / "file"
            existing_file.write_text("unchanged", encoding="utf-8")
            existing_dir = root / "directory"
            existing_dir.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(existing_file)
            for path in (existing_file, existing_dir, symlink):
                with self.subTest(path=path):
                    with self.assertRaises(FileExistsError):
                        builder._ensure_fresh_output_path(path)
            self.assertEqual(existing_file.read_text(encoding="utf-8"), "unchanged")

    def test_every_pinned_tokenizer_input_is_hard_gated(self) -> None:
        tokenizer_source = Path(os.environ["ARR_TOKENIZER_DIR"])
        for changed_filename in builder.PINNED_TOKENIZER_FILES:
            with self.subTest(filename=changed_filename):
                with tempfile.TemporaryDirectory() as tmp:
                    tokenizer_copy = Path(tmp)
                    for filename in builder.PINNED_TOKENIZER_FILES:
                        shutil.copy2(
                            tokenizer_source / filename,
                            tokenizer_copy / filename,
                        )
                    with (tokenizer_copy / changed_filename).open("ab") as f:
                        f.write(b"\n")
                    with self.assertRaisesRegex(
                        ValueError,
                        f"Pinned tokenizer input mismatch for {changed_filename}",
                    ):
                        builder._load_pinned_tokenizer(tokenizer_copy)


class CanonicalDatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer_dir_raw = os.environ.get("ARR_TOKENIZER_DIR")
        if not tokenizer_dir_raw:
            raise RuntimeError(
                "ARR_TOKENIZER_DIR must point to the pinned ModernBERT snapshot"
            )
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls._temporary_directory.name)
        cls.output_a = cls.temp_root / "build_a"
        cls.output_b = cls.temp_root / "build_b"
        repo_root = builder._repo_root()
        raw_dir = (
            repo_root
            / "corporate_reorganization/data/final_annotations_gold/raw"
        )
        tokenizer_dir = Path(tokenizer_dir_raw)
        builder.build_dataset(
            raw_dir=raw_dir,
            processed_dir=cls.output_a,
            tokenizer_dir=tokenizer_dir,
        )
        builder.build_dataset(
            raw_dir=raw_dir,
            processed_dir=cls.output_b,
            tokenizer_dir=tokenizer_dir,
        )
        cls.manifest = json.loads(
            (cls.output_a / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        cls.cases = read_jsonl(cls.output_a / "cases.jsonl")
        cls.corpus = read_jsonl(cls.output_a / "corpus.jsonl")
        cls.queries = read_jsonl(cls.output_a / "queries/all.jsonl")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_two_clean_builds_are_byte_identical(self) -> None:
        files_a = sorted(
            path.relative_to(self.output_a)
            for path in self.output_a.rglob("*")
            if path.is_file()
        )
        files_b = sorted(
            path.relative_to(self.output_b)
            for path in self.output_b.rglob("*")
            if path.is_file()
        )
        self.assertEqual(files_a, files_b)
        for relative_path in files_a:
            with self.subTest(path=relative_path):
                self.assertEqual(
                    (self.output_a / relative_path).read_bytes(),
                    (self.output_b / relative_path).read_bytes(),
                )

    def test_output_tree_has_no_split_or_qrels_artifacts(self) -> None:
        actual = {
            str(path.relative_to(self.output_a))
            for path in self.output_a.rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            actual,
            {
                "cases.jsonl",
                "corpus.jsonl",
                "dataset_manifest.json",
                "pools/candidates_by_case.json",
                "pools/candidates_global.json",
                "queries/all.jsonl",
            },
        )

    def test_manifest_hashes_and_counts_match_readback(self) -> None:
        self.assertEqual(
            self.manifest["counts"],
            {
                "cases": 42,
                "nodes": 800,
                "passages": 5286,
                "queries": 490,
                "relations": 644,
                "roots": 44,
            },
        )
        for relative_path, record in self.manifest["output_files"].items():
            path = self.output_a / relative_path
            with self.subTest(path=relative_path):
                self.assertEqual(record["sha256"], sha256(path))
                self.assertEqual(record["bytes"], path.stat().st_size)
        self.assertNotIn("dataset_manifest.json", self.manifest["output_files"])

    def test_case_42_uses_final_holding_and_has_twelve_queries(self) -> None:
        case_42 = next(case for case in self.cases if case["doc_id"] == "42")
        self.assertEqual(case_42["root_conclusion_ids"], ["ENq9-QCWLD"])
        case_42_queries = [
            query for query in self.queries if query["doc_id"] == "42"
        ]
        self.assertEqual(len(case_42_queries), 12)

    def test_visible_gold_overlaps_are_exact_and_retained(self) -> None:
        overlaps = {
            (query["query_id"], passage_id)
            for query in self.queries
            for passage_id in query["visible_gold_overlap_passage_ids"]
        }
        self.assertEqual(overlaps, builder.EXPECTED_VISIBLE_GOLD_OVERLAPS)
        candidates_by_case = json.loads(
            (self.output_a / "pools/candidates_by_case.json").read_text(
                encoding="utf-8"
            )
        )
        candidates_global = set(
            json.loads(
                (self.output_a / "pools/candidates_global.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        for query_id, passage_id in overlaps:
            query = next(item for item in self.queries if item["query_id"] == query_id)
            self.assertIn(passage_id, query["positive_passage_ids"])
            self.assertIn(passage_id, query["visible_passage_ids"])
            self.assertIn(passage_id, candidates_by_case[query["doc_id"]])
            self.assertIn(passage_id, candidates_global)

    def test_source_aware_visibility_avoids_markup_boundary_false_matches(self) -> None:
        case_passages = [
            passage for passage in self.corpus if passage["doc_id"] == "87"
        ]
        false_assignments = []
        for query in self.queries:
            if query["doc_id"] != "87":
                continue
            naive_ids = {
                passage["passage_id"]
                for passage in case_passages
                if passage["text"] in query["query_text"]
            }
            false_assignments.extend(
                sorted(naive_ids.difference(query["visible_passage_ids"]))
            )
        self.assertEqual(false_assignments, ["87::SENT_00150"] * 9)

    def test_all_sentences_are_candidates_and_truncation_is_absent(self) -> None:
        labels = {passage["label"] for passage in self.corpus}
        self.assertIn(builder.LABEL_BACKGROUND, labels)
        self.assertIn(builder.LABEL_PROCEDURE, labels)
        self.assertIn(builder.LABEL_UNLABELED, labels)
        diagnostics = self.manifest["diagnostics"]
        self.assertEqual(diagnostics["visible_passage_assignments"], 8223)
        self.assertEqual(
            diagnostics["truncation"],
            {
                "all_visible_passages_survive": True,
                "maximum_tokens_by_view": {
                    "flat_masked": 3062,
                    "structured": 3027,
                },
                "queries_over_limit_by_view": {
                    "flat_masked": 0,
                    "structured": 0,
                },
                "visible_passage_assignments_lost_by_view": {
                    "flat_masked": 0,
                    "structured": 0,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
