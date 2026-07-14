from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from corporate_reorganization.modernbert.experiments.retrieval_cv import analysis


REGIMES = (
    "same_case_legacy",
    "same_case_full",
    analysis.PRIMARY_REGIME,
    analysis.CONTEXT_EXCLUDED_REGIME,
)
FOLD_CASE_COUNTS = (9, 9, 8, 8, 8)
FOLD_PASSAGE_COUNTS = (1054, 1060, 1055, 1055, 1062)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _per_query(query_id: str, case_id: str, passage_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "doc_id": case_id,
        "gold_passage_ids": [passage_id],
        "visible_passage_ids": [],
        "gold_count": 1,
        "first_gold_rank": 1,
        **{f"hit_at_{k}": 1.0 for k in analysis.KS},
        **{f"set_recall_at_{k}": 1.0 for k in analysis.KS},
        **{f"exact_target_recovery_at_{k}": 1.0 for k in analysis.KS},
        "first_gold_reciprocal_rank_full_ranking": 1.0,
        "candidate_count": 1,
    }


def _ranking(query_id: str, case_id: str, passage_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "doc_id": case_id,
        "candidate_count": 1,
        "ranking_sha256": "a" * 64,
        "parent_ranking_sha256": None,
        "ranked_candidates": [{"rank": 1, "passage_id": passage_id, "score": 1.0}],
    }


def _systems() -> list[dict[str, str]]:
    return [
        {
            "system_id": system_id,
            "system_type": system_type,
            "query_view": query_view,
        }
        for system_id, system_type, query_view in analysis.aggregate.EXPECTED_SYSTEM_CONTRACT
    ]


def _seal(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "receipt_sha256": analysis._document_sha256(payload)}


def _make_fold(
    root: Path,
    *,
    fold: int,
    case_ids: list[str],
) -> tuple[Path, Path, dict[str, object]]:
    acquisition = root / f"fold-{fold}-acquisition"
    evaluation = acquisition / "evaluation"
    evidence = acquisition / "evidence"
    query_ids = [f"f{fold}-q{position:03d}" for position in range(98)]
    query_case = {
        query_id: case_ids[position % len(case_ids)]
        for position, query_id in enumerate(query_ids)
    }
    systems = _systems()
    config = {
        "identity": {"outer_fold": fold, "role": "test"},
        "case_ids": sorted(case_ids),
        "query_ids": query_ids,
        "passage_ids": [
            f"f{fold}-p{position:04d}" for position in range(FOLD_PASSAGE_COUNTS[fold])
        ],
        "systems": systems,
        "regimes": [{"regime_name": regime} for regime in REGIMES],
    }
    _write(evaluation / "evaluation_config.json", analysis._canonical_bytes(config))
    ranking_lines: list[bytes] = []
    for system in systems:
        for regime in REGIMES:
            for query_index, query_id in enumerate(query_ids):
                case_id = query_case[query_id]
                passage_id = f"gold-{query_id}"
                ranking_lines.append(
                    analysis._canonical_bytes(
                        {
                            "schema_version": 1,
                            "system_id": system["system_id"],
                            "system_type": system["system_type"],
                            "query_view": system["query_view"],
                            "regime_name": regime,
                            "query_index": query_index,
                            "query_id": query_id,
                            "per_query": _per_query(query_id, case_id, passage_id),
                            "source_ranking": {},
                            "ranking": _ranking(query_id, case_id, passage_id),
                        }
                    )
                )
    _write(evaluation / "rankings.jsonl", b"".join(ranking_lines))
    for path, value in (
        (evaluation / "artifact_manifest.json", {"fixture": "evaluation-manifest"}),
        (evaluation / "results.json", {"fixture": "results"}),
        (evidence / "artifact_manifest.json", {"fixture": "evidence-manifest"}),
        (evidence / "materialization_receipt.json", {"fixture": "materialization"}),
    ):
        _write(path, analysis._canonical_bytes(value))

    terminal = _seal(
        {
            "schema_version": 1,
            "protocol": "retrieval_cv_fold_evaluation_terminal_v1",
            "outer_fold": fold,
            "job_name": f"arr-ret-cv1-f{fold}-evaluate-a3-r1",
            "job_arn": (
                "arn:aws:sagemaker:us-east-1:123456789012:processing-job/"
                f"arr-ret-cv1-f{fold}-evaluate-a3-r1"
            ),
            "status": "Completed",
            "failure_reason": None,
            "exit_message": None,
            "processing_start_time": "2026-07-14T00:00:00.000000Z",
            "processing_end_time": "2026-07-14T00:01:00.000000Z",
            "processing_time_microseconds": 60_000_000,
            "request_sha256": "1" * 64,
            "preflight_receipt_sha256": "2" * 64,
            "submission_receipt_sha256": "3" * 64,
        }
    )
    terminal_path = root / f"fold-{fold}-terminal.json"
    _write(terminal_path, analysis._pretty_canonical_bytes(terminal))

    files = []
    remote = []
    output_prefix = f"arr-retrieval-cv/evaluation-a3/fold-{fold}/fixture/"
    for relative_name in analysis.EXPECTED_ACQUIRED_FILES:
        path = acquisition / relative_name
        record = {
            "path": relative_name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        files.append(record)
        remote.append(
            {
                "bucket": "fixture-bucket",
                "key": f"{output_prefix}{relative_name}",
                "version_id": f"version-{fold}-{len(remote)}",
                "etag": '"00000000000000000000000000000000"',
                "size": record["size"],
                "sha256": record["sha256"],
                "encryption": {"algorithm": "aws:kms"},
            }
        )
    acquisition_receipt = _seal(
        {
            "schema_version": 1,
            "protocol": "retrieval_cv_fold_evaluation_acquisition_v1",
            "outer_fold": fold,
            "output_prefix": output_prefix,
            "terminal_receipt_sha256": analysis._document_sha256(terminal),
            "control_bundle_receipt_sha256": "4" * 64,
            "evaluation_artifact_manifest_sha256": files[0]["sha256"],
            "materialization_artifact_manifest_sha256": files[4]["sha256"],
            "remote_objects": remote,
            "files": files,
        }
    )
    _write(
        acquisition / "acquisition_receipt.json",
        analysis._pretty_canonical_bytes(acquisition_receipt),
    )
    index_record = {
        "outer_fold": fold,
        "case_count": len(case_ids),
        "query_count": 98,
        "passage_count": FOLD_PASSAGE_COUNTS[fold],
        "ranking_row_count": 15 * 4 * 98,
    }
    return acquisition, terminal_path, index_record


class FullAnalysisIntegrationTest(unittest.TestCase):
    def test_five_fold_receipt_to_case_first_bundle(self) -> None:
        cases = [f"c{position:02d}" for position in range(42)]
        case_offset = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquisitions: list[Path] = []
            terminals: list[Path] = []
            index_records: list[dict[str, object]] = []
            for fold, count in enumerate(FOLD_CASE_COUNTS):
                acquisition, terminal, index_record = _make_fold(
                    root,
                    fold=fold,
                    case_ids=cases[case_offset : case_offset + count],
                )
                case_offset += count
                acquisitions.append(acquisition)
                terminals.append(terminal)
                index_records.append(index_record)
            module_root = Path(analysis.__file__).parent
            repo_data = module_root.parents[2] / "data" / "final_annotations_gold" / "processed_retrieval_v2"
            with mock.patch.object(
                analysis.aggregate,
                "build_evaluation_index",
                return_value={"folds": index_records},
            ), mock.patch.object(
                analysis,
                "_paired_case_bootstrap",
                return_value=(0.0, 0.0),
            ), mock.patch.object(
                analysis,
                "_hierarchical_case_seed_bootstrap",
                return_value=(0.0, 0.0),
            ):
                bundle = analysis.build_analysis_bundle(
                    acquisition_dirs=list(reversed(acquisitions)),
                    terminal_receipts=list(reversed(terminals)),
                    dataset_dir=repo_data,
                    fold_manifest_path=module_root / "configs" / "folds.json",
                    experiment_config_path=module_root / "configs" / "experiment.json",
                )
        self.assertEqual(len(bundle.query_metrics), 5 * 15 * 4 * 98)
        self.assertEqual(len(bundle.case_metrics), 42 * 15 * 4)
        self.assertEqual(len(bundle.contrasts), 5)
        self.assertEqual(len(bundle.per_case_primary), 42)
        self.assertEqual(sum(row["passage_count"] for row in bundle.fold_load), 5_286)
        self.assertTrue(all(record["estimate"] == 0.0 for record in bundle.contrasts))
        self.assertEqual(
            tuple(relative_name for relative_name, _ in bundle.dataset_input_files),
            analysis.EXPECTED_DATASET_INPUT_FILES,
        )
        for relative_name, payload in bundle.dataset_input_files:
            self.assertEqual(payload, (repo_data / relative_name).read_bytes())


if __name__ == "__main__":
    unittest.main()
