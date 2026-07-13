from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.artifacts import CONTROLLED_ARTIFACT_PROTOCOL  # noqa: E402
from retriever.data import PassageIndexTable, load_corpus  # noqa: E402
from retriever.evaluator import (  # noqa: E402
    BM25_SYSTEM_TYPE,
    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
    E5_SYSTEM_TYPE,
    FIXED_BASE_SYSTEM_TYPE,
    _validate_complete_evaluation_plan,
    _validate_complete_local_bindings,
    run_complete_evaluation_plan,
)


DATASET_DIR = (
    REPO_ROOT
    / "corporate_reorganization/data/final_annotations_gold/processed_retrieval_v2"
)
FOLDS_PATH = MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json"
EXPERIMENT_PATH = MODERNBERT_DIR / "experiments/retrieval_cv/configs/experiment.json"
BASELINE_PATH = MODERNBERT_DIR / "experiments/retrieval_cv/configs/evaluation_baselines.json"
IMAGE_CONTRACT_PATH = MODERNBERT_DIR / "processing_eval/image_contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _systems() -> list[dict[str, object]]:
    systems: list[dict[str, object]] = [
        {
            "system_id": "bm25_flat_plain",
            "system_type": BM25_SYSTEM_TYPE,
            "query_view": "flat_plain",
            "expectation": {
                "baseline_config_sha256": _sha256(BASELINE_PATH),
                "runtime_protocol": "pyserini_1_5_0_sparse_jni_only_v1",
                "pyserini_version": "1.5.0",
                "pyjnius_version": "1.7.0",
                "anserini_jar_sha256": "bb0761df51ef7db5be361199a40a45722cccf7f0b2271e2b25337e97dd578aea",
                "k1": 0.9,
                "b": 0.4,
            },
        },
        {
            "system_id": "e5_base_v2_flat_plain",
            "system_type": E5_SYSTEM_TYPE,
            "query_view": "flat_plain",
            "expectation": {
                "baseline_config_sha256": _sha256(BASELINE_PATH),
                "model_id": "intfloat/e5-base-v2",
                "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
                "snapshot_manifest_sha256": "7629cf8c8bf60569d72f653d21a4c47a8fa806d8fd907db05c65a3288b24b635",
                "snapshot_tree_sha256": "1181a9758ea858d6679df0e04f6ac67b26dab90e91f63e76238c2eecec1c1a61",
                "pack_artifact_protocol": "frozen_e5_flat_plain_focus_pack_v1",
                "pack_manifest_sha256": "9875bd57c23a7e390c85d2a4b1b3aab7415597c0223c2fed621e613d4dfded10",
                "packed_query_inventory_sha256": "9cfe6cbd83c60a686751c82d1c811612a27eb5a04d835a1a600335081f5b1edf",
                "packing_protocol": "focus_preserving_semantic_pack_v1",
                "weight_dtype": "float32",
                "attention_implementation": "eager",
                "pooling": "attention_masked_mean_then_l2_normalize_v1",
                "max_positions": 512,
                "max_passage_tokens": 500,
                "passage_truncation": "right",
                "token_type_ids": "explicit_all_zero",
            },
        },
        {
            "system_id": "modernbert_base_flat_masked",
            "system_type": FIXED_BASE_SYSTEM_TYPE,
            "query_view": "flat_masked",
            "expectation": {
                "baseline_config_sha256": _sha256(BASELINE_PATH),
                "artifact_manifest_sha256": "ccff3fa4c141290ef9383992a4d3de2b8cfa5e50d02c4cd06e3fe52e92d0202b",
                "model_artifact_protocol": "fixed_seed_untrained_modernbert_dual_encoder_bf16_v1",
                "fixed_initialization_seed": 17,
                "model_sha256": "a2822fd04d0ba9b5df5289d9384e89740d113ddd68810a8d05ba6dbefbc33300",
                "new_embedding_rows_sha256": "6dba50931329f2bea4618616ba222440488b776dd1216a2a61279f83f9e9a26b",
                "state_key_sha256": "d715c23e469ddfad4e731db3c01f30ef8b7fc1a6e7117fc37915d845d20386a9",
                "snapshot_manifest_sha256": "0807d16ba5b49a5e30c8b09b72acef7d8c6326823a850640027cc1363ee446b5",
                "snapshot_tree_sha256": "aca85feea4adb60c4b021eb1a439aff47c844495005f2acdee1baef9d611d63d",
                "weight_dtype": "bfloat16",
            },
        },
    ]
    position = 1
    for query_view in ("flat_masked", "structured"):
        for sampler in ("global_uniform", "local_unique"):
            for seed in (17, 29, 43):
                systems.append(
                    {
                        "system_id": f"{query_view}_{sampler}_seed{seed}",
                        "system_type": CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                        "query_view": query_view,
                        "expectation": {
                            "artifact_manifest_sha256": f"{position:064x}",
                            "training_plan_sha256": "1" * 64,
                            "training_staging_receipt_sha256": "2" * 64,
                            "source_bundle_name": f"source-{'3' * 64}.tar.gz",
                            "source_bundle_size": 12_345,
                            "source_bundle_sha256": "3" * 64,
                            "source_bundle_inventory_sha256": "4" * 64,
                            "source_bundle_commit_epoch": 1_700_000_000,
                            "experiment_id": "arr_retrieval_cv_v1",
                            "outer_fold": 0,
                            "query_view": query_view,
                            "sampler": sampler,
                            "experiment_seed": seed,
                            "dataset_manifest_sha256": _sha256(
                                DATASET_DIR / "dataset_manifest.json"
                            ),
                            "fold_manifest_sha256": _sha256(FOLDS_PATH),
                            "passage_index_sha256": PassageIndexTable(
                                load_corpus(DATASET_DIR)
                            ).sha256,
                            "model_artifact_protocol": CONTROLLED_ARTIFACT_PROTOCOL,
                        },
                    }
                )
                position += 1
    return sorted(systems, key=lambda system: system["system_id"])


def _plan() -> dict[str, object]:
    folds = json.loads(FOLDS_PATH.read_bytes())
    role = folds["rotations"][0]["test"]
    return {
        "schema_version": 3,
        "experiment_id": "arr_retrieval_cv_v1",
        "outer_fold": 0,
        "role": "test",
        "experiment_config_sha256": _sha256(EXPERIMENT_PATH),
        "dataset_manifest_sha256": _sha256(DATASET_DIR / "dataset_manifest.json"),
        "fold_manifest_sha256": _sha256(FOLDS_PATH),
        "passage_index_sha256": PassageIndexTable(load_corpus(DATASET_DIR)).sha256,
        "baseline_config_sha256": _sha256(BASELINE_PATH),
        "image_contract_sha256": _sha256(IMAGE_CONTRACT_PATH),
        "image_uri": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
            "arr-retrieval-eval@sha256:" + "f" * 64
        ),
        "case_ids": role["case_ids"],
        "query_count": role["queries"],
        "passage_count": role["passages"],
        "controlled_max_len_query": 4_096,
        "controlled_max_len_passage": 500,
        "e5_max_len_passage": 500,
        "query_batch_size": 4,
        "passage_batch_size": 38,
        "runtime_identity": {"runtime": "complete-plan-fixture"},
        "systems": _systems(),
    }


def _processing_job_config(
    plan: dict[str, object],
    *,
    plan_path: Path,
    bindings_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "AppSpecification": {
            "ImageUri": plan["image_uri"],
            "ContainerEntrypoint": [
                "/opt/conda/bin/python",
                "/opt/program/modernbert/processing_eval/evaluate_sm.py",
            ],
            "ContainerArguments": [
                "--evaluation-plan",
                str(plan_path),
                "--local-bindings",
                str(bindings_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
        }
    }


def _complete_failure_fixture(
    root: Path,
    plan: dict[str, object],
) -> SimpleNamespace:
    inputs = root / "inputs"
    work = root / "work"
    inputs.mkdir()
    work.mkdir()
    plan_path = root / "plan.json"
    bindings_path = root / "bindings.json"
    output_dir = root / "output"
    processing_config_path = root / "processingjobconfig.json"
    plan_path.write_bytes(_canonical_bytes(plan))

    records = []
    for system in plan["systems"]:
        if system["system_type"] in {
            CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
            FIXED_BASE_SYSTEM_TYPE,
        }:
            record = {
                "system_id": system["system_id"],
                "artifact_dir": str(inputs / system["system_id"]),
            }
        elif system["system_type"] == E5_SYSTEM_TYPE:
            record = {
                "system_id": system["system_id"],
                "snapshot_dir": str(inputs / "e5"),
                "snapshot_manifest_path": str(inputs / "e5.json"),
                "pack_artifact_dir": str(inputs / "pack"),
            }
        else:
            record = {"system_id": system["system_id"]}
        records.append(record)
    bindings_path.write_bytes(
        _canonical_bytes(
            {
                "schema_version": 2,
                "dataset_dir": str(DATASET_DIR.resolve()),
                "fold_manifest_path": str(FOLDS_PATH.resolve()),
                "experiment_config_path": str(EXPERIMENT_PATH.resolve()),
                "baseline_config_path": str(BASELINE_PATH.resolve()),
                "image_contract_path": str(IMAGE_CONTRACT_PATH.resolve()),
                "bm25_scratch_dir": str(work / "bm25-scratch"),
                "systems": records,
            }
        )
    )
    processing_config_path.write_bytes(
        json.dumps(
            _processing_job_config(
                plan,
                plan_path=plan_path,
                bindings_path=bindings_path,
                output_dir=output_dir,
            )
        ).encode("utf-8")
    )

    controlled_artifacts = []
    for system in plan["systems"]:
        if system["system_type"] != CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE:
            continue
        expectation = system["expectation"]
        controlled_artifacts.append(
            SimpleNamespace(
                slot_token_id=50_284,
                identity=SimpleNamespace(
                    artifact_manifest_sha256=expectation[
                        "artifact_manifest_sha256"
                    ],
                    query_view=expectation["query_view"],
                    sampler=expectation["sampler"],
                    experiment_seed=expectation["experiment_seed"],
                ),
            )
        )
    e5_identity = SimpleNamespace(files=(("model.safetensors", 1, "9" * 64),))
    all_query_ids = [
        json.loads(line)["query_id"]
        for line in (DATASET_DIR / "queries/all.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validated_pack = SimpleNamespace(
        packed_query_inventory_sha256=(
            "9cfe6cbd83c60a686751c82d1c811612a27eb5a04d835a1a600335081f5b1edf"
        ),
        packed_queries=tuple(
            SimpleNamespace(
                query_id=query_id,
                input_ids=(101, 23_032, 1_024, 102),
            )
            for query_id in all_query_ids
        ),
    )
    runtime = SimpleNamespace(
        torch_module=__import__("torch"),
        auto_tokenizer_class=SimpleNamespace(
            from_pretrained=mock.Mock(return_value=SimpleNamespace())
        ),
    )
    fixed_artifact = SimpleNamespace(
        slot_token_id=50_284,
        manifest_sha256=(
            "ccff3fa4c141290ef9383992a4d3de2b8cfa5e50d02c4cd06e3fe52e92d0202b"
        ),
    )
    loaded = SimpleNamespace(model=SimpleNamespace(), tokenizer=SimpleNamespace())
    return SimpleNamespace(
        plan_path=plan_path,
        bindings_path=bindings_path,
        output_dir=output_dir,
        processing_config_path=processing_config_path,
        controlled_artifacts=controlled_artifacts,
        e5_identity=e5_identity,
        validated_pack=validated_pack,
        runtime=runtime,
        fixed_artifact=fixed_artifact,
        loaded=loaded,
    )


class CompleteEvaluationPlanTest(unittest.TestCase):
    def test_complete_orchestration_publishes_all_fifteen_systems(self) -> None:
        plan = _plan()
        image_runtime = {"image_contract_sha256": _sha256(IMAGE_CONTRACT_PATH)}
        plan["runtime_identity"] = {
            "device": "cpu",
            "image_uri": plan["image_uri"],
            **image_runtime,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs = root / "inputs"
            work = root / "work"
            inputs.mkdir()
            work.mkdir()
            plan_path = root / "plan.json"
            bindings_path = root / "bindings.json"
            output_dir = root / "output"
            processing_config_path = root / "processingjobconfig.json"
            plan_path.write_bytes(_canonical_bytes(plan))
            records = []
            for system in plan["systems"]:
                if system["system_type"] in {
                    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                    FIXED_BASE_SYSTEM_TYPE,
                }:
                    record = {
                        "system_id": system["system_id"],
                        "artifact_dir": str(inputs / system["system_id"]),
                    }
                elif system["system_type"] == E5_SYSTEM_TYPE:
                    record = {
                        "system_id": system["system_id"],
                        "snapshot_dir": str(inputs / "e5"),
                        "snapshot_manifest_path": str(inputs / "e5.json"),
                        "pack_artifact_dir": str(inputs / "pack"),
                    }
                else:
                    record = {"system_id": system["system_id"]}
                records.append(record)
            bindings_path.write_bytes(
                _canonical_bytes(
                    {
                        "schema_version": 2,
                        "dataset_dir": str(DATASET_DIR.resolve()),
                        "fold_manifest_path": str(FOLDS_PATH.resolve()),
                        "experiment_config_path": str(EXPERIMENT_PATH.resolve()),
                        "baseline_config_path": str(BASELINE_PATH.resolve()),
                        "image_contract_path": str(IMAGE_CONTRACT_PATH.resolve()),
                        "bm25_scratch_dir": str(work / "bm25-scratch"),
                        "systems": records,
                    }
                )
            )
            processing_config_path.write_bytes(
                json.dumps(
                    _processing_job_config(
                        plan,
                        plan_path=plan_path,
                        bindings_path=bindings_path,
                        output_dir=output_dir,
                    )
                ).encode("utf-8")
            )

            controlled_plans = [
                system
                for system in plan["systems"]
                if system["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
            ]
            controlled_artifacts = []
            for system in controlled_plans:
                expectation = system["expectation"]
                controlled_artifacts.append(
                    SimpleNamespace(
                        slot_token_id=50_284,
                        identity=SimpleNamespace(
                            artifact_manifest_sha256=expectation[
                                "artifact_manifest_sha256"
                            ],
                            query_view=expectation["query_view"],
                            sampler=expectation["sampler"],
                            experiment_seed=expectation["experiment_seed"],
                        ),
                    )
                )
            e5_identity = SimpleNamespace(
                files=(("model.safetensors", 1, "9" * 64),)
            )
            all_query_ids = [
                json.loads(line)["query_id"]
                for line in (DATASET_DIR / "queries/all.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            validated_pack = SimpleNamespace(
                packed_query_inventory_sha256=(
                    "9cfe6cbd83c60a686751c82d1c811612a27eb5a04d835a1a600335081f5b1edf"
                ),
                packed_queries=tuple(
                    SimpleNamespace(
                        query_id=query_id,
                        input_ids=(101, 23_032, 1_024, 102),
                    )
                    for query_id in all_query_ids
                ),
            )
            runtime = SimpleNamespace(
                torch_module=__import__("torch"),
                auto_tokenizer_class=SimpleNamespace(
                    from_pretrained=mock.Mock(return_value=SimpleNamespace())
                ),
            )
            score_counter = iter(range(15))

            def score(**kwargs):
                value = float(next(score_counter))
                return runtime.torch_module.full(
                    (len(kwargs["query_ids"]), len(kwargs["passage_ids"])),
                    value,
                    dtype=runtime.torch_module.float32,
                )

            fixed_artifact = SimpleNamespace(
                slot_token_id=50_284,
                manifest_sha256=(
                    "ccff3fa4c141290ef9383992a4d3de2b8cfa5e50d02c4cd06e3fe52e92d0202b"
                ),
            )
            loaded = SimpleNamespace(model=SimpleNamespace(), tokenizer=SimpleNamespace())
            with (
                mock.patch.dict(
                    os.environ,
                    {"ARR_EVALUATION_IMAGE_URI": "spoofed-and-ignored"},
                ),
                mock.patch(
                    "retriever.evaluator.PROCESSING_JOB_CONFIG_PATH",
                    processing_config_path,
                ),
                mock.patch(
                    "processing_eval.image_smoke.validate_image_runtime",
                    return_value=image_runtime,
                ),
                mock.patch(
                    "retriever.artifacts.import_pinned_artifact_runtime",
                    return_value=runtime,
                ),
                mock.patch(
                    "retriever.artifacts.validate_controlled_artifact",
                    side_effect=controlled_artifacts,
                ) as controlled_validator,
                mock.patch(
                    "retriever.artifacts.load_controlled_retriever",
                    return_value=loaded,
                ) as controlled_loader,
                mock.patch(
                    "retriever.baseline_artifacts.validate_fixed_base_artifact",
                    return_value=fixed_artifact,
                ),
                mock.patch(
                    "retriever.baseline_artifacts.load_fixed_base_retriever",
                    return_value=loaded,
                ) as fixed_loader,
                mock.patch(
                    "retriever.baseline_artifacts.validate_snapshot",
                    return_value=e5_identity,
                ),
                mock.patch(
                    "retriever.baseline_artifacts.load_e5_encoder",
                    return_value=SimpleNamespace(
                        model=SimpleNamespace(),
                        tokenizer=SimpleNamespace(),
                        snapshot_identity=e5_identity,
                    ),
                ) as e5_loader,
                mock.patch(
                    "retriever.e5_pack_artifact.validate_e5_pack_artifact",
                    return_value=validated_pack,
                ),
                mock.patch(
                    "retriever.bm25.build_and_score_bm25",
                    side_effect=score,
                ) as bm25_scorer,
                mock.patch(
                    "retriever.rankers.score_loaded_e5_encoder",
                    side_effect=score,
                ) as e5_scorer,
                mock.patch(
                    "retriever.evaluator.score_loaded_dual_encoder",
                    side_effect=score,
                ) as dual_scorer,
                mock.patch(
                    "retriever.evaluator.asdict",
                    side_effect=lambda value: dict(vars(value)),
                ),
            ):
                publication = run_complete_evaluation_plan(
                    evaluation_plan_path=plan_path,
                    local_bindings_path=bindings_path,
                    output_dir=output_dir,
                    device="cpu",
                )

            self.assertTrue((output_dir / "artifact_manifest.json").is_file())
            self.assertEqual(publication["output_name"], "output")
            results = json.loads((output_dir / "results.json").read_bytes())
            self.assertEqual(len(results["result_records"]), 15 * 4)
            ranking_count = len(
                (output_dir / "rankings.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(ranking_count, 15 * 4 * plan["query_count"])
            self.assertEqual(controlled_validator.call_count, 12)
            self.assertEqual(controlled_loader.call_count, 12)
            fixed_loader.assert_called_once()
            e5_loader.assert_called_once()
            bm25_scorer.assert_called_once()
            e5_scorer.assert_called_once()
            self.assertEqual(dual_scorer.call_count, 13)

    def test_scoring_failures_never_publish_a_complete_output(self) -> None:
        failure_stages = (
            "bm25",
            "e5",
            "fixed-base dual encoder",
            "later controlled dual encoder",
        )
        for failure_stage in failure_stages:
            with self.subTest(failure_stage=failure_stage):
                plan = _plan()
                image_runtime = {
                    "image_contract_sha256": _sha256(IMAGE_CONTRACT_PATH)
                }
                plan["runtime_identity"] = {
                    "device": "cpu",
                    "image_uri": plan["image_uri"],
                    **image_runtime,
                }
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    fixture = _complete_failure_fixture(root, plan)

                    def complete_score(**kwargs):
                        return fixture.runtime.torch_module.full(
                            (
                                len(kwargs["query_ids"]),
                                len(kwargs["passage_ids"]),
                            ),
                            0.0,
                            dtype=fixture.runtime.torch_module.float32,
                        )

                    def bm25_score(**kwargs):
                        if failure_stage == "bm25":
                            raise RuntimeError("injected bm25 scoring failure")
                        return complete_score(**kwargs)

                    def e5_score(**kwargs):
                        if failure_stage == "e5":
                            raise RuntimeError("injected e5 scoring failure")
                        return complete_score(**kwargs)

                    dual_call_count = 0

                    def dual_score(**kwargs):
                        nonlocal dual_call_count
                        dual_call_count += 1
                        if (
                            failure_stage == "fixed-base dual encoder"
                            and dual_call_count == 1
                        ):
                            raise RuntimeError(
                                "injected fixed-base dual encoder scoring failure"
                            )
                        if (
                            failure_stage == "later controlled dual encoder"
                            and dual_call_count == 4
                        ):
                            raise RuntimeError(
                                "injected later controlled dual encoder scoring failure"
                            )
                        return complete_score(**kwargs)

                    with (
                        mock.patch.dict(
                            os.environ,
                            {"ARR_EVALUATION_IMAGE_URI": "spoofed-and-ignored"},
                        ),
                        mock.patch(
                            "retriever.evaluator.PROCESSING_JOB_CONFIG_PATH",
                            fixture.processing_config_path,
                        ),
                        mock.patch(
                            "processing_eval.image_smoke.validate_image_runtime",
                            return_value=image_runtime,
                        ),
                        mock.patch(
                            "retriever.artifacts.import_pinned_artifact_runtime",
                            return_value=fixture.runtime,
                        ),
                        mock.patch(
                            "retriever.artifacts.validate_controlled_artifact",
                            side_effect=fixture.controlled_artifacts,
                        ) as controlled_validator,
                        mock.patch(
                            "retriever.artifacts.load_controlled_retriever",
                            return_value=fixture.loaded,
                        ),
                        mock.patch(
                            "retriever.baseline_artifacts.validate_fixed_base_artifact",
                            return_value=fixture.fixed_artifact,
                        ),
                        mock.patch(
                            "retriever.baseline_artifacts.load_fixed_base_retriever",
                            return_value=fixture.loaded,
                        ),
                        mock.patch(
                            "retriever.baseline_artifacts.validate_snapshot",
                            return_value=fixture.e5_identity,
                        ),
                        mock.patch(
                            "retriever.baseline_artifacts.load_e5_encoder",
                            return_value=SimpleNamespace(
                                model=SimpleNamespace(),
                                tokenizer=SimpleNamespace(),
                                snapshot_identity=fixture.e5_identity,
                            ),
                        ),
                        mock.patch(
                            "retriever.e5_pack_artifact.validate_e5_pack_artifact",
                            return_value=fixture.validated_pack,
                        ),
                        mock.patch(
                            "retriever.bm25.build_and_score_bm25",
                            side_effect=bm25_score,
                        ),
                        mock.patch(
                            "retriever.rankers.score_loaded_e5_encoder",
                            side_effect=e5_score,
                        ),
                        mock.patch(
                            "retriever.evaluator.score_loaded_dual_encoder",
                            side_effect=dual_score,
                        ),
                        mock.patch(
                            "retriever.evaluator.asdict",
                            side_effect=lambda value: dict(vars(value)),
                        ),
                        self.assertRaisesRegex(
                            RuntimeError,
                            f"injected {failure_stage} scoring failure",
                        ),
                    ):
                        run_complete_evaluation_plan(
                            evaluation_plan_path=fixture.plan_path,
                            local_bindings_path=fixture.bindings_path,
                            output_dir=fixture.output_dir,
                            device="cpu",
                        )

                    self.assertEqual(controlled_validator.call_count, 12)
                    self.assertFalse(fixture.output_dir.exists())
                    self.assertEqual(
                        list(root.rglob("artifact_manifest.json")),
                        [],
                    )

    def test_exact_fifteen_system_plan_is_accepted(self) -> None:
        identity, case_ids, systems = _validate_complete_evaluation_plan(
            _plan(),
            evaluation_plan_sha256="e" * 64,
        )
        self.assertEqual(identity.outer_fold, 0)
        self.assertEqual(len(case_ids), 9)
        self.assertEqual(len(systems), 15)
        self.assertEqual(
            {system["system_type"] for system in systems},
            {
                BM25_SYSTEM_TYPE,
                E5_SYSTEM_TYPE,
                FIXED_BASE_SYSTEM_TYPE,
                CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
            },
        )

    def test_missing_cell_wrong_view_pack_hash_and_mutable_image_fail(self) -> None:
        mutations = []
        missing = _plan()
        missing["systems"].pop()
        mutations.append((missing, "exactly 15"))
        wrong_view = _plan()
        next(
            system
            for system in wrong_view["systems"]
            if system["system_type"] == E5_SYSTEM_TYPE
        )["query_view"] = "structured"
        mutations.append((wrong_view, "E5 system contract"))
        wrong_pack = _plan()
        next(
            system
            for system in wrong_pack["systems"]
            if system["system_type"] == E5_SYSTEM_TYPE
        )["expectation"]["pack_manifest_sha256"] = "0" * 64
        mutations.append((wrong_pack, "E5 system contract"))
        mutable_image = _plan()
        mutable_image["image_uri"] = (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/arr-retrieval-eval:latest"
        )
        mutations.append((mutable_image, "image_uri"))
        renamed = _plan()
        controlled = next(
            system
            for system in renamed["systems"]
            if system["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
        )
        controlled["system_id"] = "renamed_controlled_cell"
        renamed["systems"].sort(key=lambda system: system["system_id"])
        mutations.append((renamed, "left the plan identity"))
        wrong_batch = _plan()
        wrong_batch["passage_batch_size"] = 1
        mutations.append((wrong_batch, "passage_batch_size"))
        mixed_source = _plan()
        controlled = next(
            system
            for system in mixed_source["systems"]
            if system["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
        )
        controlled["expectation"]["source_bundle_sha256"] = "5" * 64
        controlled["expectation"]["source_bundle_name"] = f"source-{'5' * 64}.tar.gz"
        mutations.append((mixed_source, "mixes controlled launch"))
        mixed_ledger = _plan()
        controlled = next(
            system
            for system in mixed_ledger["systems"]
            if system["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
        )
        controlled["expectation"]["training_plan_sha256"] = "5" * 64
        controlled["expectation"]["training_staging_receipt_sha256"] = "6" * 64
        mutations.append((mixed_ledger, "mixes controlled launch"))
        for plan, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _validate_complete_evaluation_plan(
                    plan,
                    evaluation_plan_sha256="e" * 64,
                )

    def test_discriminated_local_bindings_require_absent_scratch_and_exact_order(self) -> None:
        plan = _plan()
        _, _, systems = _validate_complete_evaluation_plan(
            plan,
            evaluation_plan_sha256="e" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs = root / "inputs"
            work = root / "work"
            inputs.mkdir()
            work.mkdir()
            records = []
            for system in systems:
                if system["system_type"] in {
                    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                    FIXED_BASE_SYSTEM_TYPE,
                }:
                    record = {
                        "system_id": system["system_id"],
                        "artifact_dir": str(inputs / system["system_id"]),
                    }
                elif system["system_type"] == E5_SYSTEM_TYPE:
                    record = {
                        "system_id": system["system_id"],
                        "snapshot_dir": str(inputs / "e5"),
                        "snapshot_manifest_path": str(inputs / "e5.json"),
                        "pack_artifact_dir": str(inputs / "pack"),
                    }
                else:
                    record = {"system_id": system["system_id"]}
                records.append(record)
            bindings = {
                "schema_version": 2,
                "dataset_dir": str(DATASET_DIR.resolve()),
                "fold_manifest_path": str(FOLDS_PATH.resolve()),
                "experiment_config_path": str(EXPERIMENT_PATH.resolve()),
                "baseline_config_path": str(BASELINE_PATH.resolve()),
                "image_contract_path": str(IMAGE_CONTRACT_PATH.resolve()),
                "bm25_scratch_dir": str(work / "bm25-scratch"),
                "systems": records,
            }
            normalized = _validate_complete_local_bindings(
                copy.deepcopy(bindings),
                system_plans=systems,
            )
            self.assertEqual(normalized["bm25_scratch_dir"], work / "bm25-scratch")
            reversed_bindings = copy.deepcopy(bindings)
            reversed_bindings["systems"].reverse()
            with self.assertRaisesRegex(ValueError, "order"):
                _validate_complete_local_bindings(
                    reversed_bindings,
                    system_plans=systems,
                )
            (work / "bm25-scratch").mkdir()
            with self.assertRaisesRegex(FileExistsError, "scratch"):
                _validate_complete_local_bindings(
                    bindings,
                    system_plans=systems,
                )
            nested = copy.deepcopy(bindings)
            nested["bm25_scratch_dir"] = str(DATASET_DIR / "forbidden-bm25-scratch")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                _validate_complete_local_bindings(
                    nested,
                    system_plans=systems,
                )

    def test_every_controlled_artifact_is_prevalidated_before_any_baseline_scoring(self) -> None:
        plan = _plan()
        image_runtime = {"image_contract_sha256": _sha256(IMAGE_CONTRACT_PATH)}
        plan["runtime_identity"] = {
            "device": "cpu",
            "image_uri": plan["image_uri"],
            **image_runtime,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs = root / "inputs"
            work = root / "work"
            inputs.mkdir()
            work.mkdir()
            plan_path = root / "plan.json"
            bindings_path = root / "bindings.json"
            output_dir = root / "output"
            processing_config_path = root / "processingjobconfig.json"
            plan_path.write_bytes(_canonical_bytes(plan))
            records = []
            for system in plan["systems"]:
                if system["system_type"] in {
                    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                    FIXED_BASE_SYSTEM_TYPE,
                }:
                    record = {
                        "system_id": system["system_id"],
                        "artifact_dir": str(inputs / system["system_id"]),
                    }
                elif system["system_type"] == E5_SYSTEM_TYPE:
                    record = {
                        "system_id": system["system_id"],
                        "snapshot_dir": str(inputs / "e5"),
                        "snapshot_manifest_path": str(inputs / "e5.json"),
                        "pack_artifact_dir": str(inputs / "pack"),
                    }
                else:
                    record = {"system_id": system["system_id"]}
                records.append(record)
            bindings_path.write_bytes(
                _canonical_bytes(
                    {
                        "schema_version": 2,
                        "dataset_dir": str(DATASET_DIR.resolve()),
                        "fold_manifest_path": str(FOLDS_PATH.resolve()),
                        "experiment_config_path": str(EXPERIMENT_PATH.resolve()),
                        "baseline_config_path": str(BASELINE_PATH.resolve()),
                        "image_contract_path": str(IMAGE_CONTRACT_PATH.resolve()),
                        "bm25_scratch_dir": str(work / "bm25-scratch"),
                        "systems": records,
                    }
                )
            )
            processing_config_path.write_bytes(
                json.dumps(
                    _processing_job_config(
                        plan,
                        plan_path=plan_path,
                        bindings_path=bindings_path,
                        output_dir=output_dir,
                    )
                ).encode("utf-8")
            )
            controlled_count = sum(
                system["system_type"] == CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE
                for system in plan["systems"]
            )
            valid_artifacts = [
                SimpleNamespace(
                    identity=SimpleNamespace(
                        artifact_manifest_sha256=f"{position:064x}",
                        query_view="flat_masked",
                        sampler="local_unique",
                        experiment_seed=17,
                    )
                )
                for position in range(1, controlled_count)
            ]
            with (
                mock.patch.dict(
                    os.environ,
                    {"ARR_EVALUATION_IMAGE_URI": "spoofed-and-ignored"},
                ),
                mock.patch(
                    "retriever.evaluator.PROCESSING_JOB_CONFIG_PATH",
                    processing_config_path,
                ),
                mock.patch(
                    "processing_eval.image_smoke.validate_image_runtime",
                    return_value=image_runtime,
                ),
                mock.patch(
                    "retriever.artifacts.import_pinned_artifact_runtime",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "retriever.artifacts.validate_controlled_artifact",
                    side_effect=[*valid_artifacts, ValueError("invalid final controlled artifact")],
                ),
                mock.patch("retriever.bm25.build_and_score_bm25") as bm25_scorer,
                self.assertRaisesRegex(ValueError, "invalid final controlled artifact"),
            ):
                run_complete_evaluation_plan(
                    evaluation_plan_path=plan_path,
                    local_bindings_path=bindings_path,
                    output_dir=output_dir,
                    device="cpu",
                )
            bm25_scorer.assert_not_called()

            wrong_service_config = _processing_job_config(
                plan,
                plan_path=plan_path,
                bindings_path=bindings_path,
                output_dir=output_dir,
            )
            wrong_service_config["AppSpecification"]["ImageUri"] = (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                "arr-retrieval-eval@sha256:" + "0" * 64
            )
            processing_config_path.write_bytes(
                json.dumps(wrong_service_config).encode("utf-8")
            )
            with (
                mock.patch(
                    "retriever.evaluator.PROCESSING_JOB_CONFIG_PATH",
                    processing_config_path,
                ),
                mock.patch("retriever.bm25.build_and_score_bm25") as scorer,
                self.assertRaisesRegex(RuntimeError, "image or entrypoint"),
            ):
                run_complete_evaluation_plan(
                    evaluation_plan_path=plan_path,
                    local_bindings_path=bindings_path,
                    output_dir=output_dir,
                    device="cpu",
                )
            scorer.assert_not_called()

            output_dir.mkdir()
            with (
                mock.patch("retriever.bm25.build_and_score_bm25") as scorer,
                self.assertRaisesRegex(FileExistsError, "output must be absent"),
            ):
                run_complete_evaluation_plan(
                    evaluation_plan_path=plan_path,
                    local_bindings_path=bindings_path,
                    output_dir=output_dir,
                    device="cpu",
                )
            scorer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
