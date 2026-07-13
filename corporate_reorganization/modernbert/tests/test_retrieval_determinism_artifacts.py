from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from retriever.determinism import (  # noqa: E402
    SMOKE_GLOBAL_WINDOW_VALID_QUERIES,
    SMOKE_MODEL_STATE_PROTOCOL,
    SMOKE_WINDOW_MICROBATCHES,
    build_smoke_loss_trace_identity,
    build_smoke_scientific_evidence,
)
import retriever.determinism_artifacts as determinism_artifacts  # noqa: E402
from retriever.determinism_artifacts import (  # noqa: E402
    SMOKE_ARTIFACT_TYPE,
    SMOKE_MODEL_ARTIFACT_PROTOCOL,
    DeterminismSmokeArtifactExpectation,
    _canonical_bf16_safetensors_identity,
    validate_determinism_smoke_artifact,
)
from retriever.provenance import (  # noqa: E402
    EXPECTED_BASE_TRAINING_IMAGE,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_DATASET_OUTPUT_SHA256,
    EXPECTED_DEEPSPEED_CONFIG_SHA256,
    EXPECTED_DERIVED_TRAINING_IMAGE,
    EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
    EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
    EXPECTED_EXPERIMENT_CONFIG_SHA256,
    EXPECTED_FOLD_MANIFEST_LOGICAL_PATH,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
    EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD,
    EXPECTED_VALIDATION_IDENTITY_BY_CELL,
)
from retriever.sampling import (  # noqa: E402
    SELECTION_ALGORITHM,
    TRACE_SCHEMA_VERSION,
    sampling_trace_checksum,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SYNTHETIC_MODEL_INVENTORY = {
    f"tensor_{index:03d}": (1,) for index in range(134)
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "size": path.stat().st_size, "sha256": _sha(path)}


def _directory(root: Path, relative: str) -> dict[str, Any]:
    directory = root / relative
    return {
        "path": relative,
        "files": [
            _record(directory, item.relative_to(directory).as_posix())
            for item in sorted(
                (path for path in directory.rglob("*") if path.is_file()),
                key=lambda path: path.relative_to(directory).as_posix(),
            )
        ],
    }


def _query_ids() -> list[str]:
    fold = json.loads(
        (MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json").read_text(
            encoding="utf-8"
        )
    )
    train_cases = set(fold["rotations"][0]["train"]["case_ids"])
    query_path = (
        MODERNBERT_DIR.parent
        / "data/final_annotations_gold/processed_retrieval_v2/queries/all.jsonl"
    )
    result = []
    for line in query_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["doc_id"] in train_cases:
            result.append(record["query_id"])
    if len(result) != 294 or _canonical_sha(sorted(result)) != EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD[0]:
        raise AssertionError("Fixture query inventory left frozen fold zero")
    return result


def _write_safetensors(path: Path) -> dict[str, Any]:
    data = b""
    header: dict[str, Any] = {"__metadata__": {
        "format": "pt",
        "source": "fresh_best_engine_zero3_gathered_16bit_state",
        "weight_dtype": "bfloat16",
    }}
    for index in range(134):
        start = len(data)
        data += b"\x80\x3f"
        header[f"tensor_{index:03d}"] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [start, len(data)],
        }
    raw = _canonical(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + data)
    return _canonical_bf16_safetensors_identity(path)


def _selection(epoch: int) -> dict[str, Any]:
    step = epoch * 3
    return {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": step,
        "checkpoint_dir": f"checkpoint-{step}",
        "deepspeed_tag": f"global_step{step}",
        "primary_metric": float(epoch - 1),
        "secondary_metric": 1.0 / 21.0 if epoch == 1 else 1.0,
        "ranking_sha256": hashlib.sha256(f"rank:{epoch}".encode()).hexdigest(),
    }


def _fold_global_result(epoch: int, *, ranking_sha256: str) -> dict[str, Any]:
    case_ids, query_contract, passage_count = (
        determinism_artifacts._frozen_validation_query_contract()
    )
    per_query = []
    for query_id, doc_id, gold_count in query_contract:
        first_gold_rank = 21 if epoch == 1 else 1
        record: dict[str, Any] = {
            "query_id": query_id,
            "doc_id": doc_id,
            "gold_count": gold_count,
            "first_gold_rank": first_gold_rank,
            "first_gold_reciprocal_rank_full_ranking": 1.0 / first_gold_rank,
            "candidate_count": float(passage_count),
        }
        for k in (1, 5, 10, 20):
            recovered = 0 if epoch == 1 else min(gold_count, k)
            record[f"hit_at_{k}"] = 1.0 if recovered else 0.0
            record[f"set_recall_at_{k}"] = recovered / gold_count
            record[f"exact_target_recovery_at_{k}"] = (
                1.0 if recovered == gold_count else 0.0
            )
        per_query.append(record)
    per_case = []
    for case_id in case_ids:
        rows = [record for record in per_query if record["doc_id"] == case_id]
        metrics = {
            metric: math.fsum(record[metric] for record in rows) / len(rows)
            for metric in determinism_artifacts._QUERY_METRIC_NAMES
        }
        per_case.append(
            {"doc_id": case_id, "query_count": len(rows), "metrics": metrics}
        )
    metrics = {
        "eval_validation_num_queries": float(len(per_query)),
        "eval_validation_num_cases": float(len(per_case)),
        "eval_validation_num_passages": float(passage_count),
    }
    for metric in determinism_artifacts._QUERY_METRIC_NAMES:
        metrics[f"eval_validation_query_micro_{metric}"] = (
            math.fsum(record[metric] for record in per_query) / len(per_query)
        )
        metrics[f"eval_validation_case_macro_{metric}"] = (
            math.fsum(record["metrics"][metric] for record in per_case)
            / len(per_case)
        )
    metrics["eval_validation_query_micro_mrr_full_ranking"] = metrics[
        "eval_validation_query_micro_first_gold_reciprocal_rank_full_ranking"
    ]
    metrics["eval_validation_case_macro_mrr_full_ranking"] = metrics[
        "eval_validation_case_macro_first_gold_reciprocal_rank_full_ranking"
    ]
    identity = EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")]
    return {
        "schema_version": 1,
        "metrics": metrics,
        "per_query": per_query,
        "per_case": per_case,
        "ranking_sha256": ranking_sha256,
        "case_ids_sha256": identity["case_ids_sha256"],
        "query_ids_sha256": identity["query_ids_sha256"],
        "passage_ids_sha256": identity["passage_ids_sha256"],
        "validation_contract_sha256": identity["contract_sha256"],
    }


def _checkpoint(root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    name = selection["checkpoint_dir"]
    tag = selection["deepspeed_tag"]
    checkpoint = root / name
    checkpoint.mkdir()
    for filename in (
        "zero_to_fp32.py",
        "scheduler.pt",
        "training_args.bin",
        "trainer_state.json",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    ):
        (checkpoint / filename).write_bytes(f"fixture:{filename}\n".encode())
    tag_root = checkpoint / tag
    tag_root.mkdir()
    for rank in range(4):
        for filename in (
            f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt",
            f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        ):
            (tag_root / filename).write_bytes(f"fixture:{filename}\n".encode())
    files = [
        _record(checkpoint, item.relative_to(checkpoint).as_posix())
        for item in sorted(
            (path for path in checkpoint.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(checkpoint).as_posix(),
        )
    ]
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "schema_version": 1,
            "selection": selection,
            "world_size": 4,
            "client_state_sha256": SHA_A,
            "scheduler_state_sha256": SHA_B,
            "rng_files": [f"rng_state_{rank}.pth" for rank in range(4)],
            "files": files,
        },
    )
    return _directory(root, name)


def _candidate_trace(query_id: str, epoch: int) -> dict[str, Any]:
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sampler": "global_uniform",
        "experiment_seed": 17,
        "epoch": epoch,
        "query_id": query_id,
        "doc_id": "fixture-case",
        "positive_passage_ids": ["p-positive"],
        "selected_positive_passage_ids": ["p-positive"],
        "negative_passage_ids_by_stratum": {
            "global": [f"p-negative-{index:02d}" for index in range(60)]
        },
        "eligible_pool_sizes_by_stratum": {"global": 100},
        "candidate_passage_ids": [
            "p-positive",
            *(f"p-negative-{index:02d}" for index in range(60)),
        ],
    }
    return {**payload, "trace_sha256": sampling_trace_checksum(payload)}


def _write_candidate_and_loss(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    queries = _query_ids()
    by_rank_queries = [queries[rank::4] for rank in range(4)]
    candidate_root = root / "candidate_traces"
    loss_root = root / "loss_traces"
    candidate_root.mkdir()
    loss_root.mkdir()
    candidate_by_rank: list[list[dict[str, Any]]] = []
    candidate_shards = []
    loss_by_rank: list[list[dict[str, Any]]] = []
    loss_shards = []
    for rank, rank_queries in enumerate(by_rank_queries):
        candidates = [
            _candidate_trace(query_id, epoch)
            for epoch in range(2)
            for query_id in rank_queries
        ]
        candidate_by_rank.append(candidates)
        candidate_path = candidate_root / f"rank-{rank:05d}.jsonl"
        candidate_path.write_text(
            "".join(_canonical(record) + "\n" for record in candidates), encoding="utf-8"
        )
        candidate_shards.append(
            {"rank": rank, **_record(candidate_root, candidate_path.name), "record_count": len(candidates)}
        )

        loss_records = []
        position = 0
        for epoch in range(2):
            for local_index in range(19):
                if local_index < 8:
                    window = 0
                    within = local_index
                elif local_index < 16:
                    window = 1
                    within = local_index - 8
                else:
                    window = 2
                    within = local_index - 16
                local_count = 4 if local_index < 18 else (2 if rank < 2 else 1)
                selected = candidates[position : position + local_count]
                position += local_count
                loss_records.append(
                    {
                        "schema_version": 1,
                        "epoch": epoch,
                        "rank": rank,
                        "local_microbatch_index": local_index,
                        "optimizer_window_index": window,
                        "window_microbatch_index": within,
                        "global_step_before": epoch * 3 + window,
                        "is_window_end": within == SMOKE_WINDOW_MICROBATCHES[window] - 1,
                        "query_ids": [record["query_id"] for record in selected],
                        "candidate_trace_sha256": [record["trace_sha256"] for record in selected],
                        "local_valid_query_count": local_count,
                        "global_window_valid_query_count": SMOKE_GLOBAL_WINDOW_VALID_QUERIES[window],
                        "local_loss_sum_float32_bits": "00000000",
                        "scaled_loss_float32_bits": "00000000",
                        "per_query_loss_float32_bits": ["00000000"] * local_count,
                    }
                )
        if position != len(candidates):
            raise AssertionError("Fixture candidate/loss link coverage changed")
        loss_by_rank.append(loss_records)
        loss_path = loss_root / f"rank-{rank:05d}.jsonl"
        loss_path.write_text(
            "".join(_canonical(record) + "\n" for record in loss_records), encoding="utf-8"
        )
        loss_shards.append(
            {"rank": rank, **_record(loss_root, loss_path.name), "record_count": len(loss_records)}
        )
    merged = [record for records in candidate_by_rank for record in records]
    merged.sort(key=lambda record: (record["epoch"], record["query_id"]))
    merged_path = candidate_root / "sampling_traces.jsonl"
    merged_path.write_text("".join(_canonical(record) + "\n" for record in merged), encoding="utf-8")
    candidate_manifest = {
        "schema_version": 1,
        "merge_order": ["epoch", "query_id"],
        "epochs": 2,
        "queries_per_epoch": 294,
        "record_count": 588,
        "query_ids_sha256": EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD[0],
        "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
        "merged": {**_record(candidate_root, merged_path.name), "record_count": 588},
        "shards": candidate_shards,
    }
    _write_json(candidate_root / "manifest.json", candidate_manifest)
    loss_identity = build_smoke_loss_trace_identity(loss_by_rank)
    loss_manifest = {"schema_version": 1, "identity": loss_identity, "shards": loss_shards}
    _write_json(loss_root / "manifest.json", loss_manifest)
    return candidate_manifest, loss_manifest


def _write_validation(
    root: Path, *, retained_checkpoint_metadata: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = root / "validation"
    validation.mkdir()
    records = []
    results = []
    for epoch in (1, 2):
        selection = _selection(epoch)
        result = _fold_global_result(
            epoch, ranking_sha256=selection["ranking_sha256"]
        )
        results.append(result)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": epoch * 3,
            "checkpoint": (
                retained_checkpoint_metadata
                if epoch == 2
                else {
                    "checkpoint_dir": selection["checkpoint_dir"],
                    "deepspeed_tag": selection["deepspeed_tag"],
                    "manifest_sha256": SHA_D,
                    "scheduler_state_sha256": SHA_B,
                    "client_state_sha256": SHA_A,
                }
            ),
            "candidate": selection,
            "is_new_best": True,
            "best_after_epoch": selection,
            "validation_result": result,
        }
        path = validation / f"epoch-{epoch:03d}.json"
        _write_json(path, payload)
        records.append(
            {
                "epoch": epoch,
                "global_step": epoch * 3,
                "path": path.name,
                "sha256": _sha(path),
                "is_new_best": True,
                "candidate": selection,
                "best_after_epoch": selection,
            }
        )
    _write_json(validation / "history.json", {"schema_version": 1, "records": records})
    _write_json(validation / "best.json", records[-1])
    _write_json(validation / "latest.json", records[-1])
    manifest = {
        "schema_version": 1,
        "epochs": 2,
        "selection_order": [
            "maximize validation case-macro set recall@20",
            "maximize validation case-macro full-ranking first-gold reciprocal rank",
            "minimize epoch number",
        ],
        "best": _selection(2),
        "last": _selection(2),
        "retained_checkpoint_dirs": ["checkpoint-6"],
        "records": records,
        "history_sha256": _sha(validation / "history.json"),
        "best_sha256": _sha(validation / "best.json"),
        "latest_sha256": _sha(validation / "latest.json"),
    }
    _write_json(validation / "manifest.json", manifest)
    normalized = []
    for epoch in (1, 2):
        payload = json.loads((validation / f"epoch-{epoch:03d}.json").read_text())
        normalized.append(
            {
                key: payload[key]
                for key in (
                    "schema_version",
                    "epoch",
                    "global_step",
                    "candidate",
                    "is_new_best",
                    "best_after_epoch",
                    "validation_result",
                )
            }
        )
    return manifest, {
        "result": results[-1],
        "selection_sha256": _canonical_sha(
            {"records": normalized, "best": manifest["best"], "last": manifest["last"]}
        ),
    }


def _launch() -> dict[str, Any]:
    return {
        "training_plan_sha256": SHA_A,
        "training_staging_receipt_sha256": SHA_B,
        "source_bundle_name": f"source-{SHA_D}.tar.gz",
        "source_bundle_size": 12345,
        "source_bundle_sha256": SHA_D,
        "source_bundle_inventory_sha256": SHA_C,
        "source_bundle_commit_epoch": 1700000000,
    }


def _build(root: Path) -> DeterminismSmokeArtifactExpectation:
    root.mkdir()
    model_state = _write_safetensors(root / "model.safetensors")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    _write_json(tokenizer / "special_tokens_map.json", {"additional_special_tokens": ["[MASK]"]})
    _write_json(tokenizer / "tokenizer.json", {"fixture": True})
    _write_json(tokenizer / "tokenizer_config.json", {"model_max_length": 8192})
    encoder = root / "encoder_config"
    encoder.mkdir()
    _write_json(
        encoder / "config.json",
        {
            "model_type": "modernbert",
            "vocab_size": 50386,
            "deterministic_flash_attn": True,
            "reference_compile": False,
            "torch_dtype": "float32",
            "num_hidden_layers": 22,
        },
    )
    wrapper = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "slot_token": "[MASK]",
        "slot_token_id": 7,
        "temperature": 0.07,
        "tokenizer_size": 50386,
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": SMOKE_MODEL_ARTIFACT_PROTOCOL,
    }
    _write_json(root / "wrapper_config.json", wrapper)
    candidate, loss = _write_candidate_and_loss(root)
    retained_checkpoint = _checkpoint(root, _selection(2))
    checkpoint_manifest = json.loads(
        (root / "checkpoint-6/checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    validation, validation_info = _write_validation(
        root,
        retained_checkpoint_metadata={
            "checkpoint_dir": "checkpoint-6",
            "deepspeed_tag": "global_step6",
            "manifest_sha256": _sha(
                root / "checkpoint-6/checkpoint_manifest.json"
            ),
            "scheduler_state_sha256": checkpoint_manifest[
                "scheduler_state_sha256"
            ],
            "client_state_sha256": checkpoint_manifest["client_state_sha256"],
        },
    )
    retained = {"schema_version": 1, "checkpoints": [retained_checkpoint]}
    model_record = _record(root, "model.safetensors")
    tokenizer_record = _directory(root, "tokenizer")
    encoder_record = _directory(root, "encoder_config")
    wrapper_record = _record(root, "wrapper_config.json")
    candidate_record = _record(root, "candidate_traces/manifest.json")
    validation_record = _record(root, "validation/manifest.json")
    loss_record = _record(root, "loss_traces/manifest.json")
    source_bundle = {
        "commit_epoch": 1700000000,
        "inventory_sha256": SHA_C,
        "name": f"source-{SHA_D}.tar.gz",
        "sha256": SHA_D,
        "size": 12345,
    }
    launch_ledger = {
        "training_image": EXPECTED_DERIVED_TRAINING_IMAGE,
        "training_base_image": EXPECTED_BASE_TRAINING_IMAGE,
        "runtime_inventory_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
        "training_image_contract_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
        "bootstrap_protocol": EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
        "training_plan_sha256": SHA_A,
        "training_staging_receipt_sha256": SHA_B,
        "source_bundle": source_bundle,
    }
    reload_records = [
        {
            "rank": rank,
            "load_path_parent": "/opt/ml/model/checkpoint-6/global_step6",
            "client_state_sha256": SHA_A,
            "scheduler_state_sha256": SHA_B,
            "global_step": 6,
            "rng_sha256": _sha(root / f"checkpoint-6/rng_state_{rank}.pth"),
            "manifest_sha256": _sha(
                root / "checkpoint-6/checkpoint_manifest.json"
            ),
        }
        for rank in range(4)
    ]
    evidence = build_smoke_scientific_evidence(
        initial_model_state={"protocol": SMOKE_MODEL_STATE_PROTOCOL, "tensor_count": 134, "sha256": hashlib.sha256(b"initial").hexdigest()},
        last_model_state={"protocol": SMOKE_MODEL_STATE_PROTOCOL, "tensor_count": 134, "sha256": hashlib.sha256(b"last").hexdigest()},
        selected_model_state=model_state,
        roundtrip_model_state=model_state,
        candidate_traces={
            "manifest_sha256": candidate_record["sha256"],
            "merged_sha256": candidate["merged"]["sha256"],
            "record_count": 588,
            "rank_shards": [
                {"rank": item["rank"], "record_count": item["record_count"], "sha256": item["sha256"]}
                for item in candidate["shards"]
            ],
        },
        loss_traces=loss["identity"],
        validation_selection={"epochs": 2, "sha256": validation_info["selection_sha256"]},
        reload={
            "validation_sha256": _canonical_sha(validation_info["result"]),
            "scheduler_state_sha256": SHA_B,
            "client_state_sha256": SHA_A,
            "per_rank_rng_sha256": [item["rng_sha256"] for item in reload_records],
        },
        final_artifacts={
            "model_sha256": model_record["sha256"],
            "tokenizer_inventory_sha256": _canonical_sha(tokenizer_record["files"]),
            "encoder_config_sha256": _canonical_sha(encoder_record["files"]),
            "wrapper_config_sha256": wrapper_record["sha256"],
        },
        launch_ledger={"sha256": _canonical_sha(launch_ledger)},
    )
    rotation = json.loads(
        (MODERNBERT_DIR / "experiments/retrieval_cv/configs/folds.json").read_text()
    )["rotations"][0]
    run = {
        "schema_version": 1,
        "experiment_id": "arr_retrieval_cv_v1",
        "outer_fold": 0,
        "query_view": "structured",
        "sampler": "global_uniform",
        "experiment_seed": 17,
        "run_kind": "determinism_smoke",
        "schedule": {"epochs": 2, "updates_per_epoch": 3, "total_optimizer_updates": 6},
        "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
        "training_image": EXPECTED_DERIVED_TRAINING_IMAGE,
        "training_base_image": EXPECTED_BASE_TRAINING_IMAGE,
        "training_image_runtime_inventory_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
        "training_image_contract_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
        "training_bootstrap_protocol": EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
        "training_plan_sha256": SHA_A,
        "training_staging_receipt_sha256": SHA_B,
        "source_bundle": source_bundle,
        "experiment_config": {"path": "experiment.json", "sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256},
        "deepspeed_config": {"path": "ds_zero3.json", "sha256": EXPECTED_DEEPSPEED_CONFIG_SHA256},
        "dataset": {"manifest_path": "dataset_manifest.json", "manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256, "output_sha256": EXPECTED_DATASET_OUTPUT_SHA256},
        "folds": {"manifest_path": EXPECTED_FOLD_MANIFEST_LOGICAL_PATH, "manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256, "rotation": rotation},
        "snapshot": {"manifest_path": "modernbert_snapshot.json", "manifest_sha256": EXPECTED_SNAPSHOT_MANIFEST_SHA256, "tree_sha256": EXPECTED_SNAPSHOT_TREE_SHA256},
        "passage_index": {"schema_version": 1, "size": 5286, "sha256": EXPECTED_PASSAGE_INDEX_SHA256},
        "validation_data": {
            "role": "validation",
            "query_view": "structured",
            "case_count": rotation["validation"]["num_cases"],
            "query_count": rotation["validation"]["queries"],
            "passage_count": rotation["validation"]["passages"],
            **EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")],
        },
        "candidate_traces": {"manifest_path": "candidate_traces/manifest.json", "manifest_sha256": candidate_record["sha256"], "record_count": 588, "merged_sha256": candidate["merged"]["sha256"]},
        "validation_history": {"manifest_path": "validation/manifest.json", "manifest_sha256": validation_record["sha256"], "best": validation["best"], "last": validation["last"], "retained_checkpoint_dirs": ["checkpoint-6"]},
        "best_checkpoint_reload": {"selection": validation["best"], "validation_result": validation_info["result"], "per_rank": reload_records},
        "final_model": {**model_record, "weight_dtype": "bfloat16", "gathered_tensor_count": 134, "strict_round_trip_tensor_count": 134},
        "tokenizer": tokenizer_record,
        "encoder_config": encoder_record,
        "wrapper_config": wrapper_record,
        "retained_checkpoints": retained,
        "loss_traces": {"manifest_path": "loss_traces/manifest.json", "manifest_sha256": loss_record["sha256"], "identity": loss["identity"]},
        "determinism_scientific_evidence": evidence,
    }
    _write_json(root / "determinism_smoke_run.json", run)
    manifest = {
        "schema_version": 1,
        "commit_marker": True,
        "artifact_type": SMOKE_ARTIFACT_TYPE,
        "determinism_smoke_run": _record(root, "determinism_smoke_run.json"),
        "model": model_record,
        "tokenizer": tokenizer_record,
        "encoder_config": encoder_record,
        "wrapper_config": wrapper_record,
        "candidate_trace_manifest": candidate_record,
        "validation_manifest": validation_record,
        "loss_trace_manifest": loss_record,
        "retained_checkpoints": retained,
    }
    _write_json(root / "artifact_manifest.json", manifest)
    return DeterminismSmokeArtifactExpectation(
        artifact_manifest_sha256=_sha(root / "artifact_manifest.json"), **_launch()
    )


def _refresh_run_commit(
    root: Path, expectation: DeterminismSmokeArtifactExpectation
) -> DeterminismSmokeArtifactExpectation:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["determinism_smoke_run"] = _record(
        root, "determinism_smoke_run.json"
    )
    _write_json(manifest_path, manifest)
    return replace(expectation, artifact_manifest_sha256=_sha(manifest_path))


def _refresh_validation_manifest(root: Path) -> dict[str, Any]:
    validation_root = root / "validation"
    manifest_path = validation_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_json(
        validation_root / "history.json",
        {"schema_version": 1, "records": manifest["records"]},
    )
    _write_json(validation_root / "latest.json", manifest["records"][-1])
    best_records = [
        record
        for record in manifest["records"]
        if record["candidate"] == manifest["best"]
    ]
    if len(best_records) != 1:
        raise AssertionError("Fixture validation best is not unique")
    _write_json(validation_root / "best.json", best_records[0])
    manifest["history_sha256"] = _sha(validation_root / "history.json")
    manifest["best_sha256"] = _sha(validation_root / "best.json")
    manifest["latest_sha256"] = _sha(validation_root / "latest.json")
    _write_json(manifest_path, manifest)
    return manifest


def _replace_validation_epoch(root: Path, epoch: int, payload: dict[str, Any]) -> None:
    path = root / f"validation/epoch-{epoch:03d}.json"
    _write_json(path, payload)
    manifest_path = root / "validation/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["records"][epoch - 1]
    entry["sha256"] = _sha(path)
    for field in (
        "epoch",
        "global_step",
        "candidate",
        "is_new_best",
        "best_after_epoch",
    ):
        entry[field] = payload[field]
    _write_json(manifest_path, manifest)
    _refresh_validation_manifest(root)


def _rebuild_smoke_evidence(root: Path, run: dict[str, Any]) -> dict[str, Any]:
    previous = run["determinism_scientific_evidence"]
    validation_manifest = json.loads(
        (root / "validation/manifest.json").read_text(encoding="utf-8")
    )
    normalized = []
    for epoch in (1, 2):
        payload = json.loads(
            (root / f"validation/epoch-{epoch:03d}.json").read_text(
                encoding="utf-8"
            )
        )
        normalized.append(
            {
                key: payload[key]
                for key in (
                    "schema_version",
                    "epoch",
                    "global_step",
                    "candidate",
                    "is_new_best",
                    "best_after_epoch",
                    "validation_result",
                )
            }
        )
    reload = run["best_checkpoint_reload"]
    return build_smoke_scientific_evidence(
        initial_model_state=previous["model_states"]["initial"],
        last_model_state=previous["model_states"]["last"],
        selected_model_state=previous["model_states"]["selected"],
        roundtrip_model_state=previous["model_states"]["roundtrip"],
        candidate_traces=previous["candidate_traces"],
        loss_traces=previous["loss_traces"],
        validation_selection={
            "epochs": 2,
            "sha256": _canonical_sha(
                {
                    "records": normalized,
                    "best": validation_manifest["best"],
                    "last": validation_manifest["last"],
                }
            ),
        },
        reload={
            "validation_sha256": _canonical_sha(reload["validation_result"]),
            "scheduler_state_sha256": reload["per_rank"][0][
                "scheduler_state_sha256"
            ],
            "client_state_sha256": reload["per_rank"][0]["client_state_sha256"],
            "per_rank_rng_sha256": [
                record["rng_sha256"] for record in reload["per_rank"]
            ],
        },
        final_artifacts=previous["final_artifacts"],
        launch_ledger=previous["launch_ledger"],
    )


def _refresh_all_commit(
    root: Path, expectation: DeterminismSmokeArtifactExpectation
) -> DeterminismSmokeArtifactExpectation:
    manifest_path = root / "artifact_manifest.json"
    run_path = root / "determinism_smoke_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    validation = json.loads(
        (root / "validation/manifest.json").read_text(encoding="utf-8")
    )
    manifest["validation_manifest"] = _record(
        root, "validation/manifest.json"
    )
    run["validation_history"] = {
        "manifest_path": "validation/manifest.json",
        "manifest_sha256": manifest["validation_manifest"]["sha256"],
        "best": validation["best"],
        "last": validation["last"],
        "retained_checkpoint_dirs": validation["retained_checkpoint_dirs"],
    }
    run["retained_checkpoints"] = manifest["retained_checkpoints"]
    run["determinism_scientific_evidence"] = _rebuild_smoke_evidence(root, run)
    _write_json(run_path, run)
    manifest["determinism_smoke_run"] = _record(
        root, "determinism_smoke_run.json"
    )
    _write_json(manifest_path, manifest)
    return replace(expectation, artifact_manifest_sha256=_sha(manifest_path))


class ModelInventoryContractTests(unittest.TestCase):
    def test_production_inventory_is_exact_modernbert_dual_encoder(self) -> None:
        inventory = determinism_artifacts._EXPECTED_MODEL_TENSOR_SHAPES
        self.assertEqual(len(inventory), 134)
        self.assertEqual(
            inventory["encoder.embeddings.tok_embeddings.weight"],
            (50_386, 768),
        )
        self.assertEqual(inventory["encoder.embeddings.norm.weight"], (768,))
        self.assertEqual(inventory["encoder.final_norm.weight"], (768,))
        self.assertNotIn("encoder.layers.0.attn_norm.weight", inventory)
        for layer in range(1, 22):
            self.assertEqual(
                inventory[f"encoder.layers.{layer}.attn_norm.weight"], (768,)
            )
        self.assertTrue(all(name.startswith("encoder.") for name in inventory))

    def test_arbitrary_134_scalar_tensor_inventory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            with mock.patch.object(
                determinism_artifacts,
                "_EXPECTED_MODEL_TENSOR_SHAPES",
                SYNTHETIC_MODEL_INVENTORY,
            ):
                _write_safetensors(path)
            with self.assertRaisesRegex(ValueError, "tensor names changed"):
                _canonical_bf16_safetensors_identity(path)


class FrozenValidationInputContractTests(unittest.TestCase):
    def test_every_contract_read_rehashes_fold_query_and_corpus_inputs(self) -> None:
        original = determinism_artifacts._sha256_file
        with mock.patch.object(
            determinism_artifacts,
            "_sha256_file",
            wraps=original,
        ) as sha256_file:
            first = determinism_artifacts._frozen_validation_query_contract()
            second = determinism_artifacts._frozen_validation_query_contract()
        self.assertEqual(first, second)
        self.assertEqual(sha256_file.call_count, 6)


class DeterminismSmokeArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_inventory_patch = mock.patch.object(
            determinism_artifacts,
            "_EXPECTED_MODEL_TENSOR_SHAPES",
            SYNTHETIC_MODEL_INVENTORY,
        )
        self.model_inventory_patch.start()
        self.addCleanup(self.model_inventory_patch.stop)

    def test_exact_artifact_is_valid_and_path_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_root = Path(temporary) / "first"
            expectation = _build(first_root)
            first = validate_determinism_smoke_artifact(first_root, expectation=expectation)
            second_root = Path(temporary) / "second"
            shutil.copytree(first_root, second_root)
            second = validate_determinism_smoke_artifact(second_root, expectation=expectation)
            self.assertEqual(first.scientific_evidence, second.scientific_evidence)
            self.assertEqual(first.identity, second.identity)
            self.assertEqual(first.identity.model_state_sha256, first.scientific_evidence["model_states"]["selected"]["sha256"])

    def test_external_launch_expectation_is_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            with self.assertRaisesRegex(ValueError, "launch provenance"):
                validate_determinism_smoke_artifact(
                    root,
                    expectation=replace(expectation, training_plan_sha256="e" * 64),
                )

    def test_candidate_or_loss_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            loss_path = root / "loss_traces/rank-00000.jsonl"
            loss_path.write_bytes(loss_path.read_bytes().replace(b"00000000", b"00000001", 1))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_determinism_smoke_artifact(root, expectation=expectation)

    def test_extra_empty_and_symlink_entries_are_rejected(self) -> None:
        for kind in ("extra", "empty", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact"
                expectation = _build(root)
                if kind == "extra":
                    (root / "unexpected.txt").write_text("unexpected\n")
                elif kind == "empty":
                    (root / "empty.txt").touch()
                else:
                    (root / "unsafe-link").symlink_to(root / "model.safetensors")
                with self.assertRaises(ValueError):
                    validate_determinism_smoke_artifact(root, expectation=expectation)

    def test_nonfinite_model_tensor_is_rejected_even_with_updated_commit_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            model = root / "model.safetensors"
            raw = bytearray(model.read_bytes())
            raw[-2:] = b"\x80\x7f"
            model.write_bytes(raw)
            manifest_path = root / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["model"] = _record(root, "model.safetensors")
            _write_json(manifest_path, manifest)
            expectation = replace(expectation, artifact_manifest_sha256=_sha(manifest_path))
            with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                validate_determinism_smoke_artifact(root, expectation=expectation)

    def test_reload_result_must_equal_selected_validation_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            run_path = root / "determinism_smoke_run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["best_checkpoint_reload"]["validation_result"]["metrics"][
                "untrusted"
            ] = 1.0
            _write_json(run_path, run)
            expectation = _refresh_run_commit(root, expectation)
            with self.assertRaisesRegex(ValueError, "reload selection/result"):
                validate_determinism_smoke_artifact(root, expectation=expectation)

    def test_self_consistent_numeric_type_confusion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            run_path = root / "determinism_smoke_run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["schedule"]["epochs"] = 2.0
            _write_json(run_path, run)
            expectation = _refresh_run_commit(root, expectation)
            with self.assertRaisesRegex(ValueError, "schedule"):
                validate_determinism_smoke_artifact(
                    root,
                    expectation=expectation,
                )

    def test_rehashed_validation_data_count_float_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            run_path = root / "determinism_smoke_run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["validation_data"]["query_count"] = 98.0
            _write_json(run_path, run)
            expectation = _refresh_run_commit(root, expectation)
            with self.assertRaisesRegex(ValueError, "validation-data identity"):
                validate_determinism_smoke_artifact(root, expectation=expectation)

    def test_rehashed_epoch_schema_and_metric_types_are_rejected(self) -> None:
        for attack in ("schema", "metric"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact"
                expectation = _build(root)
                epoch = 1 if attack == "schema" else 2
                epoch_path = root / f"validation/epoch-{epoch:03d}.json"
                payload = json.loads(epoch_path.read_text(encoding="utf-8"))
                if attack == "schema":
                    payload["schema_version"] = True
                else:
                    payload["validation_result"]["metrics"][
                        "eval_validation_case_macro_set_recall_at_20"
                    ] = 1
                _replace_validation_epoch(root, epoch, payload)
                if attack == "metric":
                    run_path = root / "determinism_smoke_run.json"
                    run = json.loads(run_path.read_text(encoding="utf-8"))
                    run["best_checkpoint_reload"]["validation_result"] = payload[
                        "validation_result"
                    ]
                    _write_json(run_path, run)
                expectation = _refresh_all_commit(root, expectation)
                expected = "schema version" if attack == "schema" else "exact float"
                with self.assertRaisesRegex(ValueError, expected):
                    validate_determinism_smoke_artifact(
                        root, expectation=expectation
                    )

    def test_rehashed_fold_global_coverage_aggregates_and_identity_are_rejected(self) -> None:
        for attack in ("coverage", "case_aggregate", "global_aggregate", "identity"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact"
                expectation = _build(root)
                epoch_path = root / "validation/epoch-001.json"
                payload = json.loads(epoch_path.read_text(encoding="utf-8"))
                result = payload["validation_result"]
                if attack == "coverage":
                    result["per_query"].pop()
                    expected = "exactly 98"
                elif attack == "case_aggregate":
                    result["per_case"][0]["metrics"]["hit_at_1"] = 0.5
                    expected = "per_case.*hit_at_1 changed"
                elif attack == "global_aggregate":
                    result["metrics"][
                        "eval_validation_query_micro_candidate_count"
                    ] = 1_059.0
                    expected = "query_micro_candidate_count changed"
                else:
                    result["query_ids_sha256"] = SHA_D
                    expected = "left the frozen validation role"
                _replace_validation_epoch(root, 1, payload)
                expectation = _refresh_all_commit(root, expectation)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_determinism_smoke_artifact(
                        root, expectation=expectation
                    )

    def test_rehashed_reload_rng_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            run_path = root / "determinism_smoke_run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            for rank, record in enumerate(run["best_checkpoint_reload"]["per_rank"]):
                record["rng_sha256"] = hashlib.sha256(
                    f"substituted-rng:{rank}".encode()
                ).hexdigest()
            _write_json(run_path, run)
            expectation = _refresh_all_commit(root, expectation)
            with self.assertRaisesRegex(ValueError, "state/RNG files"):
                validate_determinism_smoke_artifact(root, expectation=expectation)

    def test_rehashed_checkpoint_numeric_types_are_rejected(self) -> None:
        for field, replacement in (("schema_version", True), ("world_size", 4.0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact"
                expectation = _build(root)
                checkpoint_path = root / "checkpoint-6/checkpoint_manifest.json"
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                checkpoint[field] = replacement
                _write_json(checkpoint_path, checkpoint)
                checkpoint_sha = _sha(checkpoint_path)

                epoch_path = root / "validation/epoch-002.json"
                epoch_payload = json.loads(epoch_path.read_text(encoding="utf-8"))
                epoch_payload["checkpoint"]["manifest_sha256"] = checkpoint_sha
                _replace_validation_epoch(root, 2, epoch_payload)

                run_path = root / "determinism_smoke_run.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                for record in run["best_checkpoint_reload"]["per_rank"]:
                    record["manifest_sha256"] = checkpoint_sha
                _write_json(run_path, run)

                manifest_path = root / "artifact_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["retained_checkpoints"] = {
                    "schema_version": 1,
                    "checkpoints": [_directory(root, "checkpoint-6")],
                }
                _write_json(manifest_path, manifest)
                expectation = _refresh_all_commit(root, expectation)
                with self.assertRaisesRegex(ValueError, "version/world size"):
                    validate_determinism_smoke_artifact(
                        root, expectation=expectation
                    )

    def test_rehashed_checkpoint_selection_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            expectation = _build(root)
            checkpoint_path = root / "checkpoint-6/checkpoint_manifest.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["selection"]["primary_metric"] = 999.0
            _write_json(checkpoint_path, checkpoint)
            checkpoint_sha = _sha(checkpoint_path)

            epoch_path = root / "validation/epoch-002.json"
            epoch_payload = json.loads(epoch_path.read_text(encoding="utf-8"))
            epoch_payload["checkpoint"]["manifest_sha256"] = checkpoint_sha
            _replace_validation_epoch(root, 2, epoch_payload)

            run_path = root / "determinism_smoke_run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            for record in run["best_checkpoint_reload"]["per_rank"]:
                record["manifest_sha256"] = checkpoint_sha
            _write_json(run_path, run)

            manifest_path = root / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["retained_checkpoints"] = {
                "schema_version": 1,
                "checkpoints": [_directory(root, "checkpoint-6")],
            }
            _write_json(manifest_path, manifest)
            expectation = _refresh_all_commit(root, expectation)
            with self.assertRaisesRegex(ValueError, "selection"):
                validate_determinism_smoke_artifact(
                    root,
                    expectation=expectation,
                )

    def test_rehashed_wrong_best_chronology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            _build(root)
            validation_root = root / "validation"
            epoch_path = validation_root / "epoch-002.json"
            payload = json.loads(epoch_path.read_text(encoding="utf-8"))
            tied_candidate = dict(payload["candidate"])
            tied_candidate["primary_metric"] = 0.0
            tied_candidate["secondary_metric"] = 1.0 / 21.0
            payload["candidate"] = tied_candidate
            payload["best_after_epoch"] = tied_candidate
            payload["validation_result"] = _fold_global_result(
                1, ranking_sha256=tied_candidate["ranking_sha256"]
            )
            _write_json(epoch_path, payload)
            manifest_path = validation_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records"][1].update(
                {
                    "sha256": _sha(epoch_path),
                    "candidate": tied_candidate,
                    "best_after_epoch": tied_candidate,
                    "is_new_best": True,
                }
            )
            manifest["best"] = tied_candidate
            manifest["last"] = tied_candidate
            _write_json(manifest_path, manifest)
            _refresh_validation_manifest(root)
            with self.assertRaisesRegex(ValueError, "chronology"):
                determinism_artifacts._validate_validation(
                    root, _record(root, "validation/manifest.json")
                )

    def test_validation_candidate_cannot_be_assigned_to_another_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            _build(root)
            manifest_path = root / "validation/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records"][0]["candidate"] = _selection(2)
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "assigned to another epoch"):
                determinism_artifacts._validate_validation(
                    root,
                    _record(root, "validation/manifest.json"),
                )


if __name__ == "__main__":
    unittest.main()
