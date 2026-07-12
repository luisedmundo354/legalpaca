from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
CORPORATE_REORGANIZATION_DIR = MODERNBERT_DIR.parent
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from legacy_eval import (  # noqa: E402
    LEGACY_REGIME_SEMANTICS,
    MarchReplayError,
    replay_reconstructed_march,
)
from legacy_eval import march  # noqa: E402


ARCHIVE_DIR = CORPORATE_REORGANIZATION_DIR / "test_results" / "retrieval_ablation_local"
PROCESSED_DIR = (
    CORPORATE_REORGANIZATION_DIR
    / "data"
    / "final_annotations_gold"
    / "processed"
)


class ReconstructedMarchReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frozen_bytes = {
            "config.json": (ARCHIVE_DIR / "config.json").read_bytes(),
            "results.json": (ARCHIVE_DIR / "results.json").read_bytes(),
            "runs/rankings.jsonl": (ARCHIVE_DIR / "runs" / "rankings.jsonl").read_bytes(),
            "corpus.jsonl": (PROCESSED_DIR / "corpus.jsonl").read_bytes(),
            "queries/test.jsonl": (PROCESSED_DIR / "queries" / "test.jsonl").read_bytes(),
            "pools/candidates_by_case.json": (
                PROCESSED_DIR / "pools" / "candidates_by_case.json"
            ).read_bytes(),
            "splits/test_cases.txt": (
                PROCESSED_DIR / "splits" / "test_cases.txt"
            ).read_bytes(),
        }

    def test_exact_archive_replays_all_rows_cells_and_numbers(self) -> None:
        before = self._current_hashes()
        replay = replay_reconstructed_march(
            archive_dir=ARCHIVE_DIR,
            processed_dir=PROCESSED_DIR,
        )
        self.assertEqual(replay.namespace, "reconstructed_march_2026")
        self.assertEqual(replay.query_count, 40)
        self.assertEqual(replay.ranking_row_count, 600)
        self.assertEqual(replay.result_cell_count, 15)
        self.assertEqual(replay.numeric_values_verified, 3300)
        self.assertEqual(replay.stable_tie_rows_reordered, 386)
        self.assertEqual(replay.stable_tie_metric_changes, 0)
        self.assertEqual(
            tuple(replay.legacy_regime_labels),
            ("same_case_legacy", "same_case_full", "global_split"),
        )
        self.assertEqual(
            dict(replay.semantic_regime_by_legacy_label),
            {
                "same_case_legacy": "same_case_legacy",
                "same_case_full": "same_case_full",
                "global_split": "fold_global",
            },
        )
        self.assertIn("global_split", replay.replayed_regimes)
        self.assertNotIn("fold_global", replay.replayed_regimes)
        self.assertEqual(before, self._current_hashes())

    def test_public_result_and_hash_contracts_are_immutable(self) -> None:
        replay = replay_reconstructed_march(
            archive_dir=ARCHIVE_DIR,
            processed_dir=PROCESSED_DIR,
        )
        with self.assertRaises(TypeError):
            replay.sha256_by_input["config.json"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            replay.semantic_regime_by_legacy_label["global_split"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            replay.replayed_regimes["global_split"] = {}  # type: ignore[index]
        self.assertEqual(LEGACY_REGIME_SEMANTICS["global_split"], "fold_global")

    def test_hash_mutation_fails_loudly_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_copy = Path(tmp) / "archive"
            shutil.copytree(ARCHIVE_DIR, archive_copy)
            config_path = archive_copy / "config.json"
            config_path.write_bytes(config_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(MarchReplayError, "SHA-256 mismatch.*config.json"):
                replay_reconstructed_march(
                    archive_dir=archive_copy,
                    processed_dir=PROCESSED_DIR,
                )

    def test_missing_frozen_file_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_archive = Path(tmp) / "archive"
            empty_archive.mkdir()
            with self.assertRaisesRegex(MarchReplayError, "Missing frozen March input"):
                replay_reconstructed_march(
                    archive_dir=empty_archive,
                    processed_dir=PROCESSED_DIR,
                )

    def test_missing_ranking_row_fails_exact_600_row_gate(self) -> None:
        mutated = dict(self.frozen_bytes)
        lines = mutated["runs/rankings.jsonl"].splitlines(keepends=True)
        mutated["runs/rankings.jsonl"] = b"".join(lines[:-1])
        with self.assertRaisesRegex(MarchReplayError, "exactly 600 ranking rows"):
            march._replay_hash_verified_bytes(mutated)

    def test_truncated_candidate_pool_fails_complete_set_gate(self) -> None:
        mutated = dict(self.frozen_bytes)
        lines = mutated["runs/rankings.jsonl"].decode("utf-8").splitlines()
        first = json.loads(lines[0])
        first["ranked_candidates"].pop()
        first["candidate_pool_size"] -= 1
        lines[0] = json.dumps(first, ensure_ascii=False)
        mutated["runs/rankings.jsonl"] = ("\n".join(lines) + "\n").encode("utf-8")
        with self.assertRaisesRegex(MarchReplayError, "candidate_pool_size|complete exact"):
            march._replay_hash_verified_bytes(mutated)

    def test_ranking_row_schema_and_sequence_reject_mutations(self) -> None:
        row = json.loads(
            self.frozen_bytes["runs/rankings.jsonl"].splitlines()[0].decode("utf-8")
        )
        extra = copy.deepcopy(row)
        extra["unexpected"] = True
        with self.assertRaisesRegex(MarchReplayError, "schema keys/order changed"):
            march._validate_ranking_record_shape(extra, "fixture")

        duplicate = copy.deepcopy(row)
        duplicate["ranked_candidates"][1]["passage_id"] = duplicate["ranked_candidates"][0][
            "passage_id"
        ]
        with self.assertRaisesRegex(MarchReplayError, "duplicate ranked passage_id"):
            march._validate_ranking_record_shape(duplicate, "fixture")

        nonsequential = copy.deepcopy(row)
        nonsequential["ranked_candidates"][0]["rank"] = 2
        with self.assertRaisesRegex(MarchReplayError, "sequential integer 1"):
            march._validate_ranking_record_shape(nonsequential, "fixture")

    def test_nonfinite_json_number_is_rejected(self) -> None:
        with self.assertRaisesRegex(MarchReplayError, "non-finite JSON number"):
            march._strict_json_loads('{"score": NaN}', "fixture")

    def test_stored_numeric_mutation_does_not_replay(self) -> None:
        replay = replay_reconstructed_march(
            archive_dir=ARCHIVE_DIR,
            processed_dir=PROCESSED_DIR,
        )
        stored = json.loads(self.frozen_bytes["results.json"].decode("utf-8"))
        cell = stored["regimes"]["global_split"]["systems"]["bm25_flat"]
        cell["global"]["recall_at_20"] += 0.001
        actual = replay.replayed_regimes["global_split"]["systems"]["bm25_flat"][
            "global"
        ]["recall_at_20"]
        with self.assertRaisesRegex(MarchReplayError, "does not replay exactly"):
            march._compare_replayed_value(
                actual,
                cell["global"]["recall_at_20"],
                "fixture",
            )

    def _current_hashes(self) -> dict[str, str]:
        paths = {
            "config.json": ARCHIVE_DIR / "config.json",
            "results.json": ARCHIVE_DIR / "results.json",
            "runs/rankings.jsonl": ARCHIVE_DIR / "runs" / "rankings.jsonl",
            "corpus.jsonl": PROCESSED_DIR / "corpus.jsonl",
            "queries/test.jsonl": PROCESSED_DIR / "queries" / "test.jsonl",
            "pools/candidates_by_case.json": (
                PROCESSED_DIR / "pools" / "candidates_by_case.json"
            ),
            "splits/test_cases.txt": PROCESSED_DIR / "splits" / "test_cases.txt",
        }
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        }


if __name__ == "__main__":
    unittest.main()
