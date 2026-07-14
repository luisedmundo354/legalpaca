from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from corporate_reorganization.modernbert.experiments.retrieval_cv import (
    analysis,
    cli,
    reporting,
)
from corporate_reorganization.modernbert.retriever.data import PassageIndexTable
from corporate_reorganization.modernbert.tests.test_retrieval_cv_aggregate import (
    synthetic_study,
    write_genuine_bundle,
)


def _metric_values(value: float, *, candidate_count: int = 100) -> dict[str, float | int]:
    return {
        **{f"hit_at_{k}": value for k in analysis.KS},
        **{f"set_recall_at_{k}": value for k in analysis.KS},
        **{f"exact_target_recovery_at_{k}": value for k in analysis.KS},
        "first_gold_reciprocal_rank_full_ranking": value,
        "candidate_count": candidate_count,
    }


class RawRankingMetricTest(unittest.TestCase):
    def _row(self) -> tuple[dict[str, object], dict[str, object]]:
        per_query: dict[str, object] = {
            "query_id": "q1",
            "doc_id": "c1",
            "gold_passage_ids": ["p1", "p2"],
            "visible_passage_ids": [],
            "gold_count": 2,
            "first_gold_rank": 1,
            "first_gold_reciprocal_rank_full_ranking": 1.0,
            "candidate_count": 3,
        }
        for k in analysis.KS:
            per_query[f"hit_at_{k}"] = 1.0
            per_query[f"set_recall_at_{k}"] = 0.5 if k == 1 else 1.0
            per_query[f"exact_target_recovery_at_{k}"] = 0.0 if k == 1 else 1.0
        ranking = {
            "query_id": "q1",
            "doc_id": "c1",
            "candidate_count": 3,
            "ranking_sha256": "a" * 64,
            "parent_ranking_sha256": None,
            "ranked_candidates": [
                {"rank": 1, "passage_id": "p1", "score": 3.0},
                {"rank": 2, "passage_id": "p3", "score": 2.0},
                {"rank": 3, "passage_id": "p2", "score": 1.0},
            ],
        }
        return per_query, ranking

    def test_multi_positive_metrics_are_recomputed_from_complete_ranking(self) -> None:
        per_query, ranking = self._row()
        computed = analysis._recompute_query_metrics(per_query, ranking)
        self.assertEqual(computed["gold_count"], 2)
        self.assertEqual(computed["set_recall_at_1"], 0.5)
        self.assertEqual(computed["exact_target_recovery_at_1"], 0.0)
        self.assertEqual(computed["exact_target_recovery_at_5"], 1.0)

    def test_stored_metric_mutation_fails(self) -> None:
        per_query, ranking = self._row()
        per_query["set_recall_at_1"] = 1.0
        with self.assertRaisesRegex(ValueError, "disagrees with raw ranking"):
            analysis._recompute_query_metrics(per_query, ranking)

    def test_genuine_canonical_bundle_streams_exact_coverage(self) -> None:
        corpus, queries, _ = synthetic_study()
        with tempfile.TemporaryDirectory() as directory:
            output = write_genuine_bundle(
                Path(directory),
                fold=0,
                corpus=corpus,
                queries=queries,
                passage_index_sha256=PassageIndexTable(corpus).sha256,
            )
            rows, ranking_record, fold_load = analysis._stream_fold_query_metrics(
                output,
                outer_fold=0,
            )
        self.assertEqual(len(rows), 15 * 4)
        self.assertEqual(ranking_record["rows"], 15 * 4)
        self.assertEqual(fold_load["case_count"], 1)
        self.assertEqual(fold_load["query_count"], 1)
        self.assertEqual({row["case_id"] for row in rows}, {"c0"})


class ReceiptEncodingTest(unittest.TestCase):
    def test_pretty_receipt_and_compact_document_hash_are_both_exact(self) -> None:
        payload = {"protocol": "fixture", "schema_version": 1}
        sealed = {**payload, "receipt_sha256": analysis._document_sha256(payload)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(analysis._pretty_canonical_bytes(sealed))
            loaded = analysis._load_canonical_receipt(path)
            analysis._validate_self_hash(loaded, name="fixture")
            self.assertEqual(analysis._document_sha256(loaded), analysis._document_sha256(sealed))
            path.write_text(json.dumps(sealed, indent=4) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted canonical encoding"):
                analysis._load_canonical_receipt(path)

    def test_frozen_experiment_config_matches_analysis_contract(self) -> None:
        path = Path(analysis.__file__).parent / "configs" / "experiment.json"
        config, raw = analysis._load_exact_hashed_object(path)
        self.assertEqual(raw, path.read_bytes())
        analysis._validate_locked_analysis_config(config)
        config["analysis"]["bootstrap"]["resamples"] = 9_999
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            analysis._validate_locked_analysis_config(config)


class DatasetEvidenceTest(unittest.TestCase):
    def test_complete_dataset_is_loaded_by_exact_manifest_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records: dict[str, dict[str, object]] = {}
            for relative_name in analysis.EXPECTED_DATASET_INPUT_FILES:
                payload = f"fixture:{relative_name}\n".encode("utf-8")
                path = root / relative_name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                records[relative_name] = {
                    "bytes": len(payload),
                    "records": 1,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            manifest = {"output_files": records}
            (root / "dataset_manifest.json").write_bytes(
                analysis._canonical_bytes(manifest)
            )
            loaded = analysis._load_dataset_input_files(root, manifest)
            self.assertEqual(
                tuple(relative_name for relative_name, _ in loaded),
                analysis.EXPECTED_DATASET_INPUT_FILES,
            )
            (root / analysis.EXPECTED_DATASET_INPUT_FILES[0]).write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                analysis._load_dataset_input_files(root, manifest)


class CaseFirstAggregationTest(unittest.TestCase):
    def test_queries_cases_and_seeds_are_aggregated_in_locked_order(self) -> None:
        case_ids = [f"c{position:02d}" for position in range(42)]
        queries_per_case = {
            case_id: 12 if position < 28 else 11
            for position, case_id in enumerate(case_ids)
        }
        controlled = [
            (
                f"{query_view}_{sampler}_seed{seed}",
                query_view,
                sampler,
                seed,
            )
            for query_view in analysis.QUERY_VIEWS
            for sampler in analysis.SAMPLERS
            for seed in analysis.SEEDS
        ]
        baselines = [
            ("bm25_flat_plain", None, None, None),
            ("e5_base_v2_flat_plain", None, None, None),
            ("modernbert_base_flat_masked", None, None, None),
        ]
        regimes = (
            "same_case_legacy",
            "same_case_full",
            analysis.PRIMARY_REGIME,
            analysis.CONTEXT_EXCLUDED_REGIME,
        )
        rows: list[dict[str, object]] = []
        for system_id, controlled_view, sampler, seed in [*baselines, *controlled]:
            for regime in regimes:
                for case_id in case_ids:
                    value = 1.0 if case_id == "c00" else 0.0
                    for query_position in range(queries_per_case[case_id]):
                        rows.append(
                            {
                                "outer_fold": 0,
                                "case_id": case_id,
                                "query_id": f"{case_id}-q{query_position:02d}",
                                "system_id": system_id,
                                "system_type": "fixture",
                                "query_view": controlled_view or "flat_plain",
                                "regime_name": regime,
                                "controlled_query_view": controlled_view,
                                "sampler": sampler,
                                "seed": seed,
                                **_metric_values(value),
                            }
                        )
        self.assertEqual(len(rows), 5 * 15 * 4 * 98)
        (
            case_rows,
            system_summary,
            controlled_case_rows,
            cell_case_rows,
            cell_summary,
            seed_summary,
        ) = analysis._aggregate_query_rows(rows)
        self.assertEqual(len(case_rows), 42 * 15 * 4)
        self.assertEqual(len(system_summary), 15 * 4)
        self.assertEqual(len(controlled_case_rows), 42 * 12 * 4)
        self.assertEqual(len(cell_case_rows), 42 * 4 * 4)
        self.assertEqual(len(cell_summary), 4 * 4)
        self.assertEqual(len(seed_summary), 12 * 4)
        record = next(
            item
            for item in system_summary
            if item["system_id"] == "bm25_flat_plain"
            and item["regime_name"] == analysis.PRIMARY_REGIME
        )
        self.assertAlmostEqual(record["case_macro_hit_at_20"], 1 / 42)
        self.assertAlmostEqual(record["query_micro_hit_at_20"], 12 / 490)
        self.assertNotEqual(
            record["case_macro_hit_at_20"],
            record["query_micro_hit_at_20"],
        )


class PrespecifiedContrastTest(unittest.TestCase):
    def test_five_matched_contrasts_seed_sd_and_hierarchical_sensitivity(self) -> None:
        controlled_case_rows: list[dict[str, object]] = []
        cell_case_rows: list[dict[str, object]] = []
        seed_global_effect = {17: 0.01, 29: 0.02, 43: 0.03}
        structured_global_effect = {17: 0.03, 29: 0.04, 43: 0.05}
        for case_position in range(42):
            case_id = f"c{case_position:02d}"
            values_by_cell: dict[tuple[str, str], list[float]] = {}
            for query_view in analysis.QUERY_VIEWS:
                for sampler in analysis.SAMPLERS:
                    values_by_cell[(query_view, sampler)] = []
                    for seed in analysis.SEEDS:
                        base = 0.20 + case_position * 0.001
                        if query_view == "structured":
                            base += 0.03
                        if sampler == "global_uniform":
                            base += (
                                seed_global_effect[seed]
                                if query_view == "flat_masked"
                                else structured_global_effect[seed]
                            )
                        values_by_cell[(query_view, sampler)].append(base)
                        controlled_case_rows.append(
                            {
                                "case_id": case_id,
                                "seed": seed,
                                "controlled_query_view": query_view,
                                "sampler": sampler,
                                "regime_name": analysis.PRIMARY_REGIME,
                                analysis.PRIMARY_METRIC: base,
                            }
                        )
                    cell_case_rows.append(
                        {
                            "case_id": case_id,
                            "controlled_query_view": query_view,
                            "sampler": sampler,
                            "regime_name": analysis.PRIMARY_REGIME,
                            analysis.PRIMARY_METRIC: sum(
                                values_by_cell[(query_view, sampler)]
                            )
                            / 3,
                        }
                    )
        with mock.patch.object(analysis, "BOOTSTRAP_RESAMPLES", 200):
            contrasts, per_case = analysis._build_primary_contrasts(
                controlled_case_rows,
                cell_case_rows,
            )
        self.assertEqual([row["contrast_id"] for row in contrasts], [item.contrast_id for item in analysis.CONTRASTS])
        self.assertEqual(len(per_case), 42)
        expected = (0.02, 0.04, 0.03, 0.05, 0.02)
        for record, value in zip(contrasts, expected):
            self.assertAlmostEqual(record["estimate"], value)
            self.assertEqual(record["claim_status"], "positive_supported")
            self.assertLessEqual(record["hierarchical_lower"], record["hierarchical_upper"])
        self.assertAlmostEqual(contrasts[0]["seed_sd"], 0.01)
        self.assertAlmostEqual(contrasts[2]["seed_sd"], 0.0)


def _fake_bundle() -> analysis.AnalysisBundle:
    contrasts = tuple(
        {
            "contrast_id": definition.contrast_id,
            "label": definition.label,
            "regime_name": analysis.PRIMARY_REGIME,
            "metric_name": analysis.PRIMARY_METRIC,
            "case_count": 42,
            "seed_count": 3,
            "estimate": 0.01 * (position + 1),
            "case_bootstrap_lower": -0.01 if position == 0 else 0.001,
            "case_bootstrap_upper": 0.03 + position * 0.01,
            "hierarchical_lower": -0.02 if position == 0 else 0.0005,
            "hierarchical_upper": 0.04 + position * 0.01,
            "seed_17_estimate": 0.01,
            "seed_29_estimate": 0.02,
            "seed_43_estimate": 0.03,
            "seed_sd": 0.01,
            "claim_status": "uncertain_crosses_zero" if position == 0 else "positive_supported",
        }
        for position, definition in enumerate(analysis.CONTRASTS)
    )
    cell_summary = tuple(
        {
            "regime_name": analysis.PRIMARY_REGIME,
            "controlled_query_view": query_view,
            "sampler": sampler,
            "case_count": 42,
            **{f"hit_at_{k}": 0.1 + 0.01 * position for position, k in enumerate(analysis.KS)},
        }
        for query_view in analysis.QUERY_VIEWS
        for sampler in analysis.SAMPLERS
    )
    per_case = tuple(
        {
            "case_id": f"c{position:02d}",
            **{
                definition.contrast_id: (position - 20) / 100
                for definition in analysis.CONTRASTS
            },
        }
        for position in range(42)
    )
    fold_load = tuple(
        {
            "outer_fold": fold,
            "case_count": 9 if fold < 2 else 8,
            "query_count": 98,
            "passage_count": 1054 + fold,
            "ranking_row_count": 5880,
        }
        for fold in range(5)
    )
    summary = {
        "primary_endpoint": {
            "cells": [
                {
                    "query_view": row["controlled_query_view"],
                    "sampler": row["sampler"],
                    "estimate": row["hit_at_20"],
                    "seed_sd": 0.01,
                }
                for row in cell_summary
            ]
        },
        "prespecified_contrasts": contrasts,
        "context_excluded_sensitivity": [
            {
                "query_view": query_view,
                "sampler": sampler,
                "fold_global": 0.2,
                "fold_global_context_excluded": 0.21,
                "difference": 0.01,
            }
            for query_view in analysis.QUERY_VIEWS
            for sampler in analysis.SAMPLERS
        ],
    }
    singleton = ({"fixture": 1},)
    return analysis.AnalysisBundle(
        experiment_config={"fixture": "experiment"},
        fold_manifest={"fixture": "folds"},
        dataset_manifest={"fixture": "dataset"},
        experiment_config_bytes=b'{"fixture":"experiment"}\n',
        fold_manifest_bytes=b'{"fixture":"folds"}\n',
        dataset_manifest_bytes=b'{"fixture":"dataset"}\n',
        dataset_input_files=tuple(
            (name, f"fixture:{name}\n".encode("utf-8"))
            for name in analysis.EXPECTED_DATASET_INPUT_FILES
        ),
        terminal_receipts=tuple({"outer_fold": fold} for fold in range(5)),
        acquisition_receipts=tuple({"outer_fold": fold} for fold in range(5)),
        evaluation_index={"fixture": "index"},
        jobs=singleton,
        rankings=singleton,
        fold_load=fold_load,
        query_metrics=singleton,
        case_metrics=singleton,
        system_summary=singleton,
        cell_case_metrics=singleton,
        cell_summary=cell_summary,
        seed_summary=singleton,
        contrasts=contrasts,
        per_case_primary=per_case,
        summary=summary,
    )


class ReportingTest(unittest.TestCase):
    def test_compact_report_is_atomic_hashed_and_contains_valid_svgs(self) -> None:
        bundle = _fake_bundle()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            receipt = reporting.publish_analysis_bundle(bundle, output_dir=output)
            self.assertEqual(receipt["output_name"], "report")
            manifest = json.loads((output / "analysis_manifest.json").read_bytes())
            self.assertTrue(manifest["commit_marker"])
            self.assertEqual(receipt["files"], len(manifest["files"]))
            for relative in (
                "figures/primary_contrasts_forest.svg",
                "figures/hit_at_k_curves.svg",
                "figures/per_case_sampler_effect.svg",
                "figures/fold_load_table.svg",
            ):
                ET.fromstring((output / relative).read_bytes())
            for relative_name, payload in bundle.dataset_input_files:
                self.assertEqual(
                    (output / "input" / "dataset" / relative_name).read_bytes(),
                    payload,
                )
            with self.assertRaises(FileExistsError):
                reporting.publish_analysis_bundle(bundle, output_dir=output)

    def test_failed_post_publish_readback_retracts_commit_marker(self) -> None:
        bundle = _fake_bundle()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            reporting,
            "_sha256_file",
            return_value="0" * 64,
        ):
            output = Path(directory) / "report"
            with self.assertRaisesRegex(RuntimeError, "readback failed"):
                reporting.publish_analysis_bundle(bundle, output_dir=output)
            self.assertFalse(output.exists())
            incomplete = Path(directory) / ".report.incomplete"
            self.assertTrue(incomplete.is_dir())
            self.assertFalse((incomplete / "analysis_manifest.json").exists())

    def test_cli_requires_explicit_analysis_inputs(self) -> None:
        args = cli.parse_args(
            [
                "analyze",
                "--acquisition-dir",
                "/tmp/a",
                "--terminal-receipt",
                "/tmp/t",
                "--dataset-dir",
                "/tmp/d",
                "--fold-manifest",
                "/tmp/f",
                "--experiment-config",
                "/tmp/e",
                "--output-dir",
                "/tmp/o",
            ]
        )
        self.assertEqual(args.command, "analyze")
        self.assertEqual(args.acquisition_dir, [Path("/tmp/a")])


if __name__ == "__main__":
    unittest.main()
