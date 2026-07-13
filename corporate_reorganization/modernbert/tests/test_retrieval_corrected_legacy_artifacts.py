from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from corporate_reorganization.modernbert import corrected_legacy_train
from corporate_reorganization.modernbert.experiments.retrieval_cv.corrected_legacy_config import (
    load_corrected_legacy_config,
)
from corporate_reorganization.modernbert.retriever import (
    corrected_legacy_artifacts,
    legacy_diagnostic_sampling,
)
from corporate_reorganization.modernbert.retriever.corrected_legacy_artifacts import (
    CorrectedLegacyArtifactExpectation,
)
from corporate_reorganization.modernbert.retriever.corrected_legacy_evaluation import (
    CORRECTED_LEGACY_TEST_REGIMES,
    build_corrected_legacy_test_data,
    build_corrected_legacy_validation_evidence_data,
)
from corporate_reorganization.modernbert.retriever.data import (
    PassageIndexTable,
    load_corpus,
    load_queries,
)
from corporate_reorganization.modernbert.retriever.evaluation import (
    compute_canonical_retrieval_result_from_scores,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_data import (
    load_corrected_legacy_data,
)
from corporate_reorganization.modernbert.retriever.legacy_diagnostic_batching import (
    CorrectedLegacyQueryBatchSampler,
)
from corporate_reorganization.modernbert.retriever.provenance import (
    EXPECTED_BASE_TRAINING_IMAGE,
    EXPECTED_DERIVED_TRAINING_IMAGE,
    EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
    EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
)


MODERNBERT = Path(__file__).resolve().parents[1]
DATASET = MODERNBERT.parent / "data/final_annotations_gold/processed_retrieval_v2"
CONFIG = MODERNBERT / "experiments/retrieval_cv/configs/corrected_legacy.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _canonical(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write(path: Path, value: str | bytes) -> None:
    """Atomically replace a fixture file so hard-linked fixture copies stay isolated."""

    temporary = path.with_name(path.name + ".replacement")
    if isinstance(value, str):
        temporary.write_text(value, encoding="utf-8", newline="\n")
    else:
        temporary.write_bytes(value)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _write(path, _canonical(value, indent=2) + "\n")


def _refresh_top_record(root: Path, field: str) -> None:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = root / manifest[field]["path"]
    manifest[field] = _record(target, root)
    _write_json(manifest_path, manifest)


def _directory_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "files": [
            _record(item, path)
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ],
    }


def _build_trace_bundle(root: Path, corrected_data) -> None:
    directory = root / "candidate_traces"
    directory.mkdir()
    dataset = legacy_diagnostic_sampling.CorrectedLegacyDiagnosticDataset(
        corrected_data,
        experiment_seed=17,
        query_view="structured",
    )
    records_by_rank = {rank: [] for rank in range(4)}
    for epoch in range(20):
        dataset.set_epoch(epoch)
        batcher = CorrectedLegacyQueryBatchSampler(
            [query.query_id for query in dataset.queries],
            experiment_seed=17,
            world_size=4,
            per_device_batch_size=4,
        )
        batcher.set_epoch(epoch)
        raw_batches = batcher.batches()
        for prepared_microbatch_index in range(27):
            for rank in range(4):
                raw_batch = raw_batches[prepared_microbatch_index * 4 + rank]
                local_row = 0
                for query_index in raw_batch:
                    if query_index == -1:
                        continue
                    trace = dataset[query_index]["sampling_trace"]
                    records_by_rank[rank].append(
                        {
                            "rank": rank,
                            "prepared_microbatch_index": prepared_microbatch_index,
                            "local_row": local_row,
                            "trace": trace,
                        }
                    )
                    local_row += 1

    shard_records = []
    merged = []
    for rank in range(4):
        rows = records_by_rank[rank]
        path = directory / f"rank-{rank:05d}.jsonl"
        _write(path, "".join(_canonical(row) + "\n" for row in rows))
        merged.extend(rows)
        shard_records.append(
            {
                "rank": rank,
                "record_count": len(rows),
                "query_counts_by_epoch": [106 if rank == 0 else 104] * 20,
                **_record(path, directory),
            }
        )
    merged.sort(key=lambda row: (row["trace"]["epoch"], row["trace"]["query_id"]))
    merged_path = directory / "sampling_traces.jsonl"
    _write(merged_path, "".join(_canonical(row) + "\n" for row in merged))
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "record_count": 8_360,
            "epochs": 20,
            "queries_per_epoch": 418,
            "merge_order": ["epoch", "query_id"],
            "shards": shard_records,
            "merged": _record(merged_path, directory),
        },
    )


def _build_validation_bundle(root: Path, result) -> None:
    directory = root / "validation"
    directory.mkdir()
    records = []
    files = []
    for epoch in range(1, 21):
        record = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": epoch * 4,
            "validation_result": result.to_payload(),
        }
        record["record_sha256"] = hashlib.sha256(
            _canonical(record).encode("utf-8")
        ).hexdigest()
        path = directory / f"epoch-{epoch:03d}.json"
        _write_json(path, record)
        files.append(_record(path, directory))
        records.append(record)
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "epochs": 20,
            "global_steps": list(range(4, 81, 4)),
            "model_selection": "none_final_epoch_only",
            "files": files,
            "records_sha256": hashlib.sha256(
                _canonical(records).encode("utf-8")
            ).hexdigest(),
        },
    )


def _build_artifact(root: Path) -> None:
    root.mkdir()
    loaded_config = load_corrected_legacy_config(CONFIG, dataset_dir=DATASET)
    corpus = load_corpus(DATASET)
    queries = load_queries(DATASET, "all")
    passage_index = PassageIndexTable(corpus)
    corrected_data = load_corrected_legacy_data(DATASET)

    _build_trace_bundle(root, corrected_data)
    validation_data = build_corrected_legacy_validation_evidence_data(
        all_queries=queries,
        corpus_by_passage_id=corpus,
        passage_index_table=passage_index,
        validation_case_ids=loaded_config.memberships.validation,
        query_view="structured",
    )
    validation_result = compute_canonical_retrieval_result_from_scores(
        scores=torch.zeros((32, 398), dtype=torch.float32),
        evaluation_data=validation_data.evaluation_data,
    )
    _build_validation_bundle(root, validation_result)

    model_path = root / "model.safetensors"
    _write(model_path, b"synthetic-134-tensor-model-fixture")
    model_record = _record(model_path, root)
    test_data = build_corrected_legacy_test_data(
        all_queries=queries,
        corpus_by_passage_id=corpus,
        passage_index_table=passage_index,
        test_case_ids=loaded_config.memberships.test,
        query_view="structured",
    )
    test_results = {
        regime: compute_canonical_retrieval_result_from_scores(
            scores=torch.zeros((40, 581), dtype=torch.float32),
            evaluation_data=test_data.evaluation_data_by_regime[regime],
        )
        for regime in CORRECTED_LEGACY_TEST_REGIMES
    }
    corrected_legacy_train._publish_evaluation(
        test_results,
        output_dir=root,
        query_view="structured",
        model_record=model_record,
        test_contract_sha256=test_data.contract_sha256,
    )
    corrected_legacy_train._publish_input_evidence(
        SimpleNamespace(data_dir=DATASET, corrected_legacy_config=CONFIG),
        loaded_config,
        root,
    )

    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    _write_json(tokenizer / "special_tokens_map.json", {"mask_token": "[MASK]"})
    _write_json(
        tokenizer / "tokenizer.json",
        {
            "added_tokens": [],
            "model": {
                "vocab": {
                    ("[MASK]" if token_id == 7 else f"token-{token_id}"): token_id
                    for token_id in range(50_386)
                }
            },
        },
    )
    _write_json(tokenizer / "tokenizer_config.json", {"mask_token": "[MASK]"})
    encoder = root / "encoder_config"
    encoder.mkdir()
    _write_json(
        encoder / "config.json",
        {
            "deterministic_flash_attn": True,
            "model_type": "modernbert",
            "reference_compile": False,
            "torch_dtype": "float32",
            "vocab_size": 50386,
        },
    )
    wrapper = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "artifact_type": "corrected_legacy_diagnostic_retriever",
        "query_view": "structured",
        "slot_token": "[MASK]",
        "slot_token_id": 7,
        "temperature": 0.07,
        "tokenizer_size": 50386,
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": loaded_config.value["training"]["final_model"],
    }
    wrapper_path = root / "wrapper_config.json"
    _write_json(wrapper_path, wrapper)
    explanation_path = root / "setting_explanation.md"
    _write(
        explanation_path,
        "# Corrected legacy-style diagnostic\n\n"
        + loaded_config.value["setting_explanation"]
        + "\n\nThe job completed 20 full epochs and exported the active epoch-20 model. "
        "No best-epoch selection or checkpoint reload was performed.\n",
    )
    run = {
        "schema_version": 1,
        "diagnostic_id": loaded_config.value["diagnostic_id"],
        "label": loaded_config.value["label"],
        "run_kind": "corrected_legacy_diagnostic",
        "run_id": "corrected-legacy-structured",
        "query_view": "structured",
        "seed": 17,
        "schedule": {"epochs": 20, "updates_per_epoch": 4, "total_updates": 80},
        "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
        "training_image": EXPECTED_DERIVED_TRAINING_IMAGE,
        "training_base_image": EXPECTED_BASE_TRAINING_IMAGE,
        "training_image_runtime_inventory_sha256": (
            EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
        ),
        "training_launch_provenance": {
            "bootstrap_protocol": EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
            "source_bundle": {
                "commit_epoch": 1_700_000_000,
                "inventory_sha256": SHA_B,
                "name": f"source-{SHA_A}.tar.gz",
                "sha256": SHA_A,
                "size": 12_345,
            },
            "training_image_contract_sha256": (
                EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256
            ),
            "training_plan_sha256": SHA_A,
            "training_request_payload_sha256": SHA_C,
            "training_run_id": "corrected-legacy-structured",
            "training_staging_receipt_sha256": SHA_B,
        },
        "snapshot": {
            "manifest_sha256": EXPECTED_SNAPSHOT_MANIFEST_SHA256,
            "tree_sha256": EXPECTED_SNAPSHOT_TREE_SHA256,
        },
        "config_sha256": loaded_config.config_sha256,
        "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
        "validation_records": 20,
        "candidate_traces": {
            "record_count": 8_360,
            "manifest_sha256": _sha(root / "candidate_traces/manifest.json"),
        },
        "final_model": {**model_record, "tensor_count": 134},
        "inputs_manifest_sha256": _sha(root / "inputs/manifest.json"),
        "validation_manifest_sha256": _sha(root / "validation/manifest.json"),
        "evaluation_manifest_sha256": _sha(root / "evaluation/artifact_manifest.json"),
        "reporting_boundary": loaded_config.value["reporting_boundary"],
    }
    run_path = root / "corrected_legacy_run.json"
    _write_json(run_path, run)
    manifest = {
        "schema_version": 1,
        "commit_marker": True,
        "artifact_type": "corrected_legacy_diagnostic_retriever",
        "run": _record(run_path, root),
        "model": model_record,
        "tokenizer": _directory_record(tokenizer, root),
        "encoder_config": _directory_record(encoder, root),
        "wrapper": _record(wrapper_path, root),
        "setting_explanation": _record(explanation_path, root),
        "inputs_manifest": _record(root / "inputs/manifest.json", root),
        "trace_manifest": _record(root / "candidate_traces/manifest.json", root),
        "validation_manifest": _record(root / "validation/manifest.json", root),
        "evaluation_manifest": _record(root / "evaluation/artifact_manifest.json", root),
    }
    _write_json(root / "artifact_manifest.json", manifest)


class CorrectedLegacyArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture_tmp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls._fixture_tmp.name) / "artifact"
        _build_artifact(cls.fixture)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture_tmp.cleanup()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "artifact"
        shutil.copytree(self.fixture, self.root, copy_function=os.link)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _validate(self, *, bypass: frozenset[str] = frozenset()):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    corrected_legacy_artifacts,
                    "_canonical_bf16_safetensors_identity",
                    return_value={"tensor_count": 134},
                )
            )
            if "trace" in bypass:
                stack.enter_context(
                    patch.object(
                        corrected_legacy_artifacts,
                        "_validate_trace_bundle",
                        return_value=(SHA_A, 8_360),
                    )
                )
            if "validation" in bypass:
                stack.enter_context(
                    patch.object(
                        corrected_legacy_artifacts,
                        "_validate_validation_bundle",
                        return_value=SHA_A,
                    )
                )
            if "evaluation" in bypass:
                stack.enter_context(
                    patch.object(
                        corrected_legacy_artifacts,
                        "_validate_evaluation_bundle",
                        return_value=SHA_A,
                    )
                )
            expectation = CorrectedLegacyArtifactExpectation(
                artifact_manifest_sha256=(
                    _sha(self.root / "artifact_manifest.json")
                    if (self.root / "artifact_manifest.json").is_file()
                    else SHA_A
                ),
                run_id="corrected-legacy-structured",
                query_view="structured",
                training_plan_sha256=SHA_A,
                training_staging_receipt_sha256=SHA_B,
                training_request_payload_sha256=SHA_C,
                source_bundle_name=f"source-{SHA_A}.tar.gz",
                source_bundle_size=12_345,
                source_bundle_sha256=SHA_A,
                source_bundle_inventory_sha256=SHA_B,
                source_bundle_commit_epoch=1_700_000_000,
            )
            return corrected_legacy_artifacts.validate_corrected_legacy_artifact(
                self.root,
                expectation=expectation,
            )

    def test_valid_artifact_full_readback(self) -> None:
        artifact = self._validate()
        self.assertEqual(artifact.run_id, "corrected-legacy-structured")
        self.assertEqual(artifact.query_view, "structured")
        self.assertEqual(artifact.model_sha256, _sha(self.root / "model.safetensors"))
        self.assertEqual(artifact.trace_merged_sha256, _sha(self.root / "candidate_traces/sampling_traces.jsonl"))

    def test_external_expectation_is_mandatory_and_binds_commit_marker(self) -> None:
        with self.assertRaisesRegex(TypeError, "external expectation"):
            corrected_legacy_artifacts.validate_corrected_legacy_artifact(
                self.root,
                expectation=None,
            )
        wrong = CorrectedLegacyArtifactExpectation(
            artifact_manifest_sha256="0" * 64,
            run_id="corrected-legacy-structured",
            query_view="structured",
            training_plan_sha256=SHA_A,
            training_staging_receipt_sha256=SHA_B,
            training_request_payload_sha256=SHA_C,
            source_bundle_name=f"source-{SHA_A}.tar.gz",
            source_bundle_size=12_345,
            source_bundle_sha256=SHA_A,
            source_bundle_inventory_sha256=SHA_B,
            source_bundle_commit_epoch=1_700_000_000,
        )
        with self.assertRaisesRegex(ValueError, "commit-marker digest"):
            corrected_legacy_artifacts.validate_corrected_legacy_artifact(
                self.root,
                expectation=wrong,
            )

    def test_missing_commit_marker_is_rejected(self) -> None:
        (self.root / "artifact_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "top-level|artifact_manifest"):
            self._validate()

    def test_extra_and_temporary_files_are_rejected(self) -> None:
        _write(self.root / "unexpected.txt", "not allowed\n")
        with self.assertRaisesRegex(ValueError, "top-level"):
            self._validate()
        (self.root / "unexpected.txt").unlink()
        _write(self.root / "tokenizer/.publication.tmp", "not allowed\n")
        with self.assertRaisesRegex(ValueError, "temporary"):
            self._validate()

    def test_symlink_is_rejected(self) -> None:
        target = self.root / "tokenizer/tokenizer_config.json"
        link = self.root / "tokenizer/alias.json"
        link.symlink_to(target.name)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self._validate()

    def test_mutated_subset_and_membership_are_rejected(self) -> None:
        subset = self.root / "inputs/subsets/train_queries.jsonl"
        _write(subset, subset.read_bytes() + b"{}\n")
        with self.assertRaises(ValueError):
            self._validate()
        source_subset = self.fixture / "inputs/subsets/train_queries.jsonl"
        _write(subset, source_subset.read_bytes())
        membership = self.root / "inputs/corrected_legacy_membership/train_cases.txt"
        _write(membership, membership.read_text(encoding="utf-8") + "tampered-case\n")
        with self.assertRaises(ValueError):
            self._validate()

    def test_resealed_global_pool_substitution_is_rejected_by_dataset_manifest(self) -> None:
        pool_path = self.root / "inputs/data/pools/candidates_global.json"
        _write(pool_path, pool_path.read_bytes() + b"\n")

        inputs_manifest_path = self.root / "inputs/manifest.json"
        inputs_manifest = json.loads(
            inputs_manifest_path.read_text(encoding="utf-8")
        )
        pool_record = next(
            record
            for record in inputs_manifest["data"]
            if record["path"] == "data/pools/candidates_global.json"
        )
        pool_record.update(_record(pool_path, self.root / "inputs"))
        _write_json(inputs_manifest_path, inputs_manifest)

        run_path = self.root / "corrected_legacy_run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["inputs_manifest_sha256"] = _sha(inputs_manifest_path)
        _write_json(run_path, run)

        artifact_manifest_path = self.root / "artifact_manifest.json"
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
        artifact_manifest["inputs_manifest"] = _record(
            inputs_manifest_path,
            self.root,
        )
        artifact_manifest["run"] = _record(run_path, self.root)
        _write_json(artifact_manifest_path, artifact_manifest)

        with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
            self._validate(bypass=frozenset({"trace", "validation", "evaluation"}))

    def test_trace_coverage_and_multiplicity_are_rejected(self) -> None:
        shard = self.root / "candidate_traces/rank-00000.jsonl"
        rows = shard.read_text(encoding="utf-8").splitlines()
        first = json.loads(rows[0])
        first["trace"]["multiplicity_by_unique_candidate"][0] += 1
        payload = {key: value for key, value in first["trace"].items() if key != "trace_sha256"}
        first["trace"]["trace_sha256"] = legacy_diagnostic_sampling.legacy_diagnostic_trace_checksum(payload)
        rows[0] = _canonical(first)
        _write(shard, "\n".join(rows) + "\n")
        manifest_path = self.root / "candidate_traces/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shards"][0].update(_record(shard, shard.parent))
        _write_json(manifest_path, manifest)
        _refresh_top_record(self.root, "trace_manifest")
        with self.assertRaisesRegex(ValueError, "multiplicities"):
            self._validate()

    def test_trace_epoch_coverage_is_rejected_after_resealing_file_record(self) -> None:
        shard = self.root / "candidate_traces/rank-00000.jsonl"
        rows = shard.read_text(encoding="utf-8").splitlines()
        _write(shard, "\n".join(rows[1:]) + "\n")
        manifest_path = self.root / "candidate_traces/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shards"][0].update(_record(shard, shard.parent))
        _write_json(manifest_path, manifest)
        _refresh_top_record(self.root, "trace_manifest")
        with self.assertRaisesRegex(ValueError, "rows disagree|coverage"):
            self._validate()

    def test_self_consistent_trace_forgery_fails_deterministic_replay(self) -> None:
        corrected_data = load_corrected_legacy_data(DATASET)
        trace_directory = self.root / "candidate_traces"
        shard_path = trace_directory / "rank-00000.jsonl"
        shard_rows = [
            json.loads(line)
            for line in shard_path.read_text(encoding="utf-8").splitlines()
        ]
        target_position = None
        replacement_passage_id = None
        occurrence_position = None
        for row_position, row in enumerate(shard_rows):
            trace = row["trace"]
            if trace["epoch"] != 0:
                break
            case_id = trace["doc_id"]
            used = {occurrence["passage_id"] for occurrence in trace["occurrences"]}
            eligible = [
                passage_id
                for passage_id in corrected_data.candidate_passage_ids_by_case[case_id]
                if passage_id not in corrected_data.gold_passage_ids_by_case[case_id]
                and passage_id not in used
            ]
            if not eligible:
                continue
            target_position = row_position
            replacement_passage_id = eligible[0]
            occurrence_position = next(
                position
                for position, occurrence in enumerate(trace["occurrences"])
                if occurrence["role"] == legacy_diagnostic_sampling.ROLE_SAME_CASE
            )
            break
        self.assertIsNotNone(target_position)
        self.assertIsNotNone(replacement_passage_id)
        self.assertIsNotNone(occurrence_position)

        target = shard_rows[target_position]
        original_trace_sha256 = target["trace"]["trace_sha256"]
        occurrence = target["trace"]["occurrences"][occurrence_position]
        occurrence["passage_id"] = replacement_passage_id
        occurrence["source_doc_id"] = target["trace"]["doc_id"]
        occurrence["selection_sha256"] = SHA_D
        candidate_ids = [
            item["passage_id"] for item in target["trace"]["occurrences"]
        ]
        multiplicities = Counter(candidate_ids)
        target["trace"]["unique_candidate_passage_ids"] = sorted(multiplicities)
        target["trace"]["multiplicity_by_unique_candidate"] = [
            multiplicities[passage_id]
            for passage_id in target["trace"]["unique_candidate_passage_ids"]
        ]
        trace_payload = {
            key: value
            for key, value in target["trace"].items()
            if key != "trace_sha256"
        }
        target["trace"]["trace_sha256"] = (
            legacy_diagnostic_sampling.legacy_diagnostic_trace_checksum(trace_payload)
        )
        legacy_diagnostic_sampling.validate_legacy_diagnostic_trace(target["trace"])

        _write(
            shard_path,
            "".join(_canonical(row) + "\n" for row in shard_rows),
        )
        merged_path = trace_directory / "sampling_traces.jsonl"
        merged_rows = [
            json.loads(line)
            for line in merged_path.read_text(encoding="utf-8").splitlines()
        ]
        matches = [
            position
            for position, row in enumerate(merged_rows)
            if row["trace"]["trace_sha256"] == original_trace_sha256
        ]
        self.assertEqual(len(matches), 1)
        merged_rows[matches[0]] = target
        _write(
            merged_path,
            "".join(_canonical(row) + "\n" for row in merged_rows),
        )
        manifest_path = trace_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shards"][0].update(_record(shard_path, trace_directory))
        manifest["merged"] = _record(merged_path, trace_directory)
        _write_json(manifest_path, manifest)
        _refresh_top_record(self.root, "trace_manifest")

        with self.assertRaisesRegex(ValueError, "deterministic sampling replay"):
            self._validate()

    def test_validation_chronology_and_result_are_rejected(self) -> None:
        path = self.root / "validation/epoch-001.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["global_step"] = 8
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        record["record_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        _write_json(path, record)
        manifest_path = self.root / "validation/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0] = _record(path, path.parent)
        _write_json(manifest_path, manifest)
        _refresh_top_record(self.root, "validation_manifest")
        with self.assertRaisesRegex(ValueError, "chronology"):
            self._validate(bypass=frozenset({"trace"}))

    def test_validation_ranking_is_independently_replayed(self) -> None:
        path = self.root / "validation/epoch-001.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        ranking = record["validation_result"]["rankings"][0]["ranked_candidates"]
        ranking[0], ranking[1] = ranking[1], ranking[0]
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        record["record_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        _write_json(path, record)
        manifest_path = self.root / "validation/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0] = _record(path, path.parent)
        _write_json(manifest_path, manifest)
        _refresh_top_record(self.root, "validation_manifest")
        with self.assertRaises(ValueError):
            self._validate(bypass=frozenset({"trace"}))

    def test_evaluation_rankings_and_metrics_are_rejected(self) -> None:
        results_path = self.root / "evaluation/results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        regime = CORRECTED_LEGACY_TEST_REGIMES[0]
        results["results"][regime]["metrics"]["num_queries"] = 41.0
        _write_json(results_path, results)
        evaluation_manifest = self.root / "evaluation/artifact_manifest.json"
        manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
        manifest["results"] = _record(results_path, results_path.parent)
        _write_json(evaluation_manifest, manifest)
        _refresh_top_record(self.root, "evaluation_manifest")
        with self.assertRaises(ValueError):
            self._validate(bypass=frozenset({"trace", "validation"}))

    def test_evaluation_rankings_jsonl_is_bound_to_results(self) -> None:
        rankings_path = self.root / "evaluation/rankings.jsonl"
        rows = rankings_path.read_text(encoding="utf-8").splitlines()
        row = json.loads(rows[0])
        row["candidate_count"] += 1
        rows[0] = _canonical(row)
        _write(rankings_path, "\n".join(rows) + "\n")
        evaluation_manifest = self.root / "evaluation/artifact_manifest.json"
        manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
        manifest["rankings"] = _record(rankings_path, rankings_path.parent)
        _write_json(evaluation_manifest, manifest)
        _refresh_top_record(self.root, "evaluation_manifest")
        with self.assertRaisesRegex(ValueError, "rankings JSONL"):
            self._validate(bypass=frozenset({"trace", "validation"}))

    def test_model_hash_is_bound_through_run_and_evaluation(self) -> None:
        model = self.root / "model.safetensors"
        _write(model, b"different-synthetic-model")
        manifest_path = self.root / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model"] = _record(model, self.root)
        _write_json(manifest_path, manifest)
        with self.assertRaises(ValueError):
            self._validate(bypass=frozenset({"trace", "validation", "evaluation"}))

    def test_wrapper_exact_schema_is_enforced(self) -> None:
        wrapper_path = self.root / "wrapper_config.json"
        wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
        wrapper["architecture"] = "UnrelatedRetriever"
        _write_json(wrapper_path, wrapper)
        _refresh_top_record(self.root, "wrapper")
        with self.assertRaisesRegex(ValueError, "wrapper"):
            self._validate()

    def test_tokenizer_mask_id_is_bound_to_wrapper(self) -> None:
        tokenizer_path = self.root / "tokenizer/tokenizer.json"
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        vocab = tokenizer["model"]["vocab"]
        vocab["[MASK]"], vocab["token-8"] = vocab["token-8"], vocab["[MASK]"]
        _write_json(tokenizer_path, tokenizer)
        manifest_path = self.root / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer"] = _directory_record(self.root / "tokenizer", self.root)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "MASK|tokenizer"):
            self._validate()

    def test_run_provenance_requires_lowercase_sha256(self) -> None:
        run_path = self.root / "corrected_legacy_run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["training_launch_provenance"]["training_plan_sha256"] = "g" * 64
        _write_json(run_path, run)
        _refresh_top_record(self.root, "run")
        with self.assertRaisesRegex(ValueError, "provenance|run record"):
            self._validate(bypass=frozenset({"trace", "validation", "evaluation"}))

    def test_run_request_payload_provenance_is_externally_bound(self) -> None:
        run_path = self.root / "corrected_legacy_run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["training_launch_provenance"]["training_request_payload_sha256"] = SHA_D
        _write_json(run_path, run)
        _refresh_top_record(self.root, "run")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self._validate(bypass=frozenset({"trace", "validation", "evaluation"}))

    def test_run_manifest_bindings_are_enforced(self) -> None:
        run_path = self.root / "corrected_legacy_run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["evaluation_manifest_sha256"] = "0" * 64
        _write_json(run_path, run)
        _refresh_top_record(self.root, "run")
        with self.assertRaisesRegex(ValueError, "run record|manifest"):
            self._validate(bypass=frozenset({"trace", "validation", "evaluation"}))


if __name__ == "__main__":
    unittest.main()
