from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .artifacts import (
    ArtifactFileRecord,
    CONTROLLED_ATTENTION_MODULE_COUNT,
    CONTROLLED_EXPERIMENT_ID,
    CONTROLLED_MODEL_STATE_COUNT,
    CONTROLLED_SLOT_TOKEN,
    CONTROLLED_TEMPERATURE,
    CONTROLLED_TOKENIZER_SIZE,
    _canonical_json,
    _load_json_object,
    _record_payload,
    _regular_tree_inventory,
    _require_sha256,
    _sha256_file,
    _validate_checkpoint_inventory,
    _validate_directory_record,
    _validate_file_record,
)
from .determinism import (
    SMOKE_CELL,
    SMOKE_EPOCHS,
    SMOKE_EXPECTED_MODEL_TENSOR_COUNT,
    SMOKE_MODEL_STATE_PROTOCOL,
    SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH,
    SMOKE_RUN_KIND,
    SMOKE_TOTAL_OPTIMIZER_UPDATES,
    SMOKE_TOTAL_QUERY_LINKS,
    SMOKE_UPDATES_PER_EPOCH,
    SMOKE_WORLD_SIZE,
    build_smoke_loss_trace_identity,
    validate_smoke_scientific_evidence,
)
from .provenance import (
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
    EXPECTED_FOLD_ROTATION_SHA256_BY_OUTER_FOLD,
    EXPECTED_PASSAGE_INDEX_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_SNAPSHOT_MANIFEST_SHA256,
    EXPECTED_SNAPSHOT_TREE_SHA256,
    EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
    EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD,
    EXPECTED_VALIDATION_IDENTITY_BY_CELL,
)
from .sampling import validate_sampling_trace


SMOKE_ARTIFACT_TYPE = "determinism_smoke_retriever"
SMOKE_MODEL_ARTIFACT_PROTOCOL = "determinism_smoke_selected_bf16_safetensors_v1"

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "commit_marker",
        "artifact_type",
        "determinism_smoke_run",
        "model",
        "tokenizer",
        "encoder_config",
        "wrapper_config",
        "candidate_trace_manifest",
        "validation_manifest",
        "loss_trace_manifest",
        "retained_checkpoints",
    }
)
_BASE_RUN_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "outer_fold",
        "query_view",
        "sampler",
        "experiment_seed",
        "runtime_versions",
        "training_image",
        "training_base_image",
        "training_image_contract_sha256",
        "training_image_runtime_inventory_sha256",
        "training_bootstrap_protocol",
        "training_plan_sha256",
        "training_staging_receipt_sha256",
        "source_bundle",
        "experiment_config",
        "deepspeed_config",
        "dataset",
        "folds",
        "snapshot",
        "passage_index",
        "validation_data",
        "candidate_traces",
        "validation_history",
        "best_checkpoint_reload",
        "final_model",
        "tokenizer",
        "encoder_config",
        "wrapper_config",
        "retained_checkpoints",
    }
)
_RUN_KEYS = frozenset(
    {
        *_BASE_RUN_KEYS,
        "run_kind",
        "schedule",
        "loss_traces",
        "determinism_scientific_evidence",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "epoch",
        "global_step",
        "checkpoint_dir",
        "deepspeed_tag",
        "primary_metric",
        "secondary_metric",
        "ranking_sha256",
    }
)
_EXPECTED_SELECTION_ORDER = [
    "maximize validation case-macro set recall@20",
    "maximize validation case-macro full-ranking first-gold reciprocal rank",
    "minimize epoch number",
]
_EXPECTED_SAFETENSORS_METADATA = {
    "format": "pt",
    "source": "fresh_best_engine_zero3_gathered_16bit_state",
    "weight_dtype": "bfloat16",
}
_VALIDATION_PRIMARY_METRIC = "eval_validation_case_macro_set_recall_at_20"
_VALIDATION_SECONDARY_METRIC = (
    "eval_validation_case_macro_first_gold_reciprocal_rank_full_ranking"
)
_VALIDATION_KS = (1, 5, 10, 20)
_QUERY_METRIC_NAMES = tuple(
    [f"hit_at_{k}" for k in _VALIDATION_KS]
    + [f"set_recall_at_{k}" for k in _VALIDATION_KS]
    + [f"exact_target_recovery_at_{k}" for k in _VALIDATION_KS]
    + ["first_gold_reciprocal_rank_full_ranking", "candidate_count"]
)
_PER_QUERY_KEYS = frozenset(
    {"query_id", "doc_id", "gold_count", "first_gold_rank", *_QUERY_METRIC_NAMES}
)
_PER_CASE_KEYS = frozenset({"doc_id", "query_count", "metrics"})
_AGGREGATE_METRIC_KEYS = frozenset(
    {
        "eval_validation_num_queries",
        "eval_validation_num_cases",
        "eval_validation_num_passages",
        *(
            f"eval_validation_{aggregation}_{metric_name}"
            for aggregation in ("query_micro", "case_macro")
            for metric_name in _QUERY_METRIC_NAMES
        ),
        "eval_validation_query_micro_mrr_full_ranking",
        "eval_validation_case_macro_mrr_full_ranking",
    }
)


def _build_expected_model_tensor_shapes() -> dict[str, tuple[int, ...]]:
    result = {
        "encoder.embeddings.norm.weight": (768,),
        "encoder.embeddings.tok_embeddings.weight": (50_386, 768),
        "encoder.final_norm.weight": (768,),
    }
    for layer in range(22):
        prefix = f"encoder.layers.{layer}"
        result.update(
            {
                f"{prefix}.attn.Wo.weight": (768, 768),
                f"{prefix}.attn.Wqkv.weight": (2_304, 768),
                f"{prefix}.mlp.Wi.weight": (2_304, 768),
                f"{prefix}.mlp.Wo.weight": (768, 1_152),
                f"{prefix}.mlp_norm.weight": (768,),
            }
        )
        if layer:
            result[f"{prefix}.attn_norm.weight"] = (768,)
    if len(result) != SMOKE_EXPECTED_MODEL_TENSOR_COUNT:
        raise RuntimeError("Frozen ModernBERT tensor inventory does not contain 134 tensors")
    return result


_EXPECTED_MODEL_TENSOR_SHAPES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    _build_expected_model_tensor_shapes()
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], expected[key]) for key in expected
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _frozen_validation_query_contract() -> tuple[
    tuple[str, ...], tuple[tuple[str, str, int], ...], int
]:
    modernbert_root = Path(__file__).resolve().parents[1]
    fold_path = modernbert_root / "experiments/retrieval_cv/configs/folds.json"
    query_path = (
        modernbert_root.parent
        / "data/final_annotations_gold/processed_retrieval_v2/queries/all.jsonl"
    )
    corpus_path = (
        modernbert_root.parent
        / "data/final_annotations_gold/processed_retrieval_v2/corpus.jsonl"
    )
    if _sha256_file(fold_path) != EXPECTED_FOLD_MANIFEST_SHA256:
        raise ValueError("Frozen fold manifest bytes changed during artifact validation")
    if _sha256_file(query_path) != EXPECTED_DATASET_OUTPUT_SHA256["queries/all.jsonl"]:
        raise ValueError("Frozen query bytes changed during artifact validation")
    if _sha256_file(corpus_path) != EXPECTED_DATASET_OUTPUT_SHA256["corpus.jsonl"]:
        raise ValueError("Frozen corpus bytes changed during artifact validation")
    fold_payload = _load_json_object(
        fold_path,
        name="Frozen fold manifest",
        require_canonical=True,
    )
    rotation = fold_payload["rotations"][0]
    validation = rotation["validation"]
    case_ids = tuple(validation["case_ids"])
    if (
        case_ids != tuple(sorted(case_ids))
        or len(case_ids) != validation["num_cases"]
        or len(case_ids) != 9
        or _canonical_sha256(list(case_ids))
        != EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")][
            "case_ids_sha256"
        ]
    ):
        raise ValueError("Frozen validation case inventory changed")
    case_set = set(case_ids)
    queries: list[tuple[str, str, int]] = []
    with query_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"Frozen query file line {line_number} is malformed")
            value = json.loads(line)
            if type(value) is not dict:
                raise TypeError(f"Frozen query line {line_number} is not an object")
            if value.get("doc_id") not in case_set:
                continue
            query_id = value.get("query_id")
            doc_id = value.get("doc_id")
            positives = value.get("positive_passage_ids")
            if (
                type(query_id) is not str
                or not query_id
                or type(doc_id) is not str
                or type(positives) is not list
                or not positives
                or any(type(item) is not str or not item for item in positives)
                or len(positives) != len(set(positives))
            ):
                raise ValueError(f"Frozen validation query line {line_number} changed")
            queries.append((query_id, doc_id, len(positives)))
    queries.sort(key=lambda item: item[0])
    query_ids = [item[0] for item in queries]
    if (
        len(queries) != 98
        or len(query_ids) != len(set(query_ids))
        or {item[1] for item in queries} != case_set
        or _canonical_sha256(query_ids)
        != EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")][
            "query_ids_sha256"
        ]
    ):
        raise ValueError("Frozen validation query inventory changed")
    passage_count = validation["passages"]
    if type(passage_count) is not int or passage_count != 1_060:
        raise ValueError("Frozen validation passage count changed")
    passage_ids: list[str] = []
    passage_case_ids: set[str] = set()
    with corpus_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"Frozen corpus file line {line_number} is malformed")
            value = json.loads(line)
            if type(value) is not dict or value.get("doc_id") not in case_set:
                continue
            passage_id = value.get("passage_id")
            doc_id = value.get("doc_id")
            if type(passage_id) is not str or not passage_id or type(doc_id) is not str:
                raise ValueError(f"Frozen validation corpus line {line_number} changed")
            passage_ids.append(passage_id)
            passage_case_ids.add(doc_id)
    passage_ids.sort()
    if (
        len(passage_ids) != passage_count
        or len(passage_ids) != len(set(passage_ids))
        or passage_case_ids != case_set
        or _canonical_sha256(passage_ids)
        != EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")][
            "passage_ids_sha256"
        ]
    ):
        raise ValueError("Frozen validation passage inventory changed")
    return case_ids, tuple(queries), passage_count


def _exact_finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be one finite exact float")
    return value


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"Cannot average an empty collection for {name}")
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite aggregate for {name}")
    return result


def _validate_fold_global_result(value: object, *, name: str) -> dict[str, Any]:
    result = _require_exact_dict(
        value,
        {
            "schema_version",
            "metrics",
            "per_query",
            "per_case",
            "ranking_sha256",
            "case_ids_sha256",
            "query_ids_sha256",
            "passage_ids_sha256",
            "validation_contract_sha256",
        },
        name=name,
    )
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError(f"{name}.schema_version must be exact integer 1")
    case_ids, query_contract, passage_count = _frozen_validation_query_contract()
    expected_identity = EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")]
    identity = {
        "case_ids_sha256": expected_identity["case_ids_sha256"],
        "query_ids_sha256": expected_identity["query_ids_sha256"],
        "passage_ids_sha256": expected_identity["passage_ids_sha256"],
        "validation_contract_sha256": expected_identity["contract_sha256"],
    }
    for field, expected in identity.items():
        if result[field] != expected or type(result[field]) is not str:
            raise ValueError(f"{name}.{field} left the frozen validation role")
    _require_sha256(result["ranking_sha256"], name=f"{name}.ranking_sha256")

    raw_queries = result["per_query"]
    if type(raw_queries) is not list or len(raw_queries) != len(query_contract):
        raise ValueError(f"{name}.per_query must contain exactly 98 records")
    per_query: list[dict[str, Any]] = []
    for position, (raw, expected) in enumerate(zip(raw_queries, query_contract)):
        record = _require_exact_dict(raw, _PER_QUERY_KEYS, name=f"{name}.per_query[{position}]")
        query_id, doc_id, expected_gold_count = expected
        if record["query_id"] != query_id or record["doc_id"] != doc_id:
            raise ValueError(f"{name}.per_query IDs or order changed at {position}")
        gold_count = record["gold_count"]
        first_gold_rank = record["first_gold_rank"]
        if type(gold_count) is not int or gold_count != expected_gold_count:
            raise ValueError(f"{name}.per_query[{position}].gold_count changed")
        if (
            type(first_gold_rank) is not int
            or first_gold_rank < 1
            or first_gold_rank > passage_count
        ):
            raise ValueError(f"{name}.per_query[{position}].first_gold_rank is invalid")
        normalized: dict[str, Any] = {
            "query_id": query_id,
            "doc_id": doc_id,
            "gold_count": gold_count,
            "first_gold_rank": first_gold_rank,
        }
        reciprocal = _exact_finite_float(
            record["first_gold_reciprocal_rank_full_ranking"],
            name=f"{name}.per_query[{position}].reciprocal_rank",
        )
        if reciprocal != 1.0 / first_gold_rank:
            raise ValueError(f"{name}.per_query[{position}] reciprocal rank changed")
        normalized["first_gold_reciprocal_rank_full_ranking"] = reciprocal
        candidate_count = _exact_finite_float(
            record["candidate_count"],
            name=f"{name}.per_query[{position}].candidate_count",
        )
        if candidate_count != float(passage_count):
            raise ValueError(f"{name}.per_query[{position}] candidate count changed")
        normalized["candidate_count"] = candidate_count
        recovered_by_k: list[int] = []
        for k in _VALIDATION_KS:
            hit = _exact_finite_float(
                record[f"hit_at_{k}"], name=f"{name}.per_query[{position}].hit_at_{k}"
            )
            recall = _exact_finite_float(
                record[f"set_recall_at_{k}"],
                name=f"{name}.per_query[{position}].set_recall_at_{k}",
            )
            exact = _exact_finite_float(
                record[f"exact_target_recovery_at_{k}"],
                name=f"{name}.per_query[{position}].exact_at_{k}",
            )
            matches = [
                recovered
                for recovered in range(gold_count + 1)
                if recall == recovered / gold_count
            ]
            if len(matches) != 1:
                raise ValueError(f"{name}.per_query[{position}] recall@{k} is invalid")
            recovered = matches[0]
            if recovered > min(gold_count, k):
                raise ValueError(
                    f"{name}.per_query[{position}] recovers more than k golds"
                )
            expected_hit = 1.0 if first_gold_rank <= k else 0.0
            if hit != expected_hit or hit != (1.0 if recovered else 0.0):
                raise ValueError(f"{name}.per_query[{position}] hit@{k} is inconsistent")
            if exact != (1.0 if recovered == gold_count else 0.0):
                raise ValueError(f"{name}.per_query[{position}] exact@{k} is inconsistent")
            normalized[f"hit_at_{k}"] = hit
            normalized[f"set_recall_at_{k}"] = recall
            normalized[f"exact_target_recovery_at_{k}"] = exact
            recovered_by_k.append(recovered)
        if recovered_by_k != sorted(recovered_by_k):
            raise ValueError(f"{name}.per_query[{position}] recovery decreases with k")
        per_query.append(normalized)

    raw_cases = result["per_case"]
    if type(raw_cases) is not list or len(raw_cases) != len(case_ids):
        raise ValueError(f"{name}.per_case coverage changed")
    per_case: list[dict[str, Any]] = []
    for position, (raw, case_id) in enumerate(zip(raw_cases, case_ids)):
        record = _require_exact_dict(raw, _PER_CASE_KEYS, name=f"{name}.per_case[{position}]")
        rows = [item for item in per_query if item["doc_id"] == case_id]
        if (
            record["doc_id"] != case_id
            or type(record["query_count"]) is not int
            or record["query_count"] != len(rows)
            or not rows
        ):
            raise ValueError(f"{name}.per_case[{position}] identity changed")
        metrics = _require_exact_dict(
            record["metrics"], set(_QUERY_METRIC_NAMES), name=f"{name}.per_case[{position}].metrics"
        )
        normalized_metrics: dict[str, float] = {}
        for metric in _QUERY_METRIC_NAMES:
            actual = _exact_finite_float(
                metrics[metric], name=f"{name}.per_case[{position}].{metric}"
            )
            expected = _mean(
                [item[metric] for item in rows], name=f"{name}.per_case[{position}].{metric}"
            )
            if actual != expected:
                raise ValueError(f"{name}.per_case[{position}].{metric} changed")
            normalized_metrics[metric] = actual
        per_case.append(
            {"doc_id": case_id, "query_count": len(rows), "metrics": normalized_metrics}
        )

    raw_metrics = _require_exact_dict(
        result["metrics"], _AGGREGATE_METRIC_KEYS, name=f"{name}.metrics"
    )
    expected_metrics: dict[str, float] = {
        "eval_validation_num_queries": float(len(per_query)),
        "eval_validation_num_cases": float(len(per_case)),
        "eval_validation_num_passages": float(passage_count),
    }
    for metric in _QUERY_METRIC_NAMES:
        expected_metrics[f"eval_validation_query_micro_{metric}"] = _mean(
            [item[metric] for item in per_query], name=f"query_micro.{metric}"
        )
        expected_metrics[f"eval_validation_case_macro_{metric}"] = _mean(
            [item["metrics"][metric] for item in per_case], name=f"case_macro.{metric}"
        )
    expected_metrics["eval_validation_query_micro_mrr_full_ranking"] = expected_metrics[
        "eval_validation_query_micro_first_gold_reciprocal_rank_full_ranking"
    ]
    expected_metrics["eval_validation_case_macro_mrr_full_ranking"] = expected_metrics[
        _VALIDATION_SECONDARY_METRIC
    ]
    normalized_metrics: dict[str, float] = {}
    for metric, expected in expected_metrics.items():
        actual = _exact_finite_float(raw_metrics[metric], name=f"{name}.metrics.{metric}")
        if actual != expected:
            raise ValueError(f"{name}.metrics.{metric} changed")
        normalized_metrics[metric] = actual
    return {
        "schema_version": 1,
        "metrics": normalized_metrics,
        "per_query": per_query,
        "per_case": per_case,
        "ranking_sha256": result["ranking_sha256"],
        **identity,
    }


@dataclass(frozen=True)
class DeterminismSmokeArtifactExpectation:
    artifact_manifest_sha256: str
    training_plan_sha256: str
    training_staging_receipt_sha256: str
    source_bundle_name: str
    source_bundle_size: int
    source_bundle_sha256: str
    source_bundle_inventory_sha256: str
    source_bundle_commit_epoch: int

    def __post_init__(self) -> None:
        for field in (
            "artifact_manifest_sha256",
            "training_plan_sha256",
            "training_staging_receipt_sha256",
            "source_bundle_sha256",
            "source_bundle_inventory_sha256",
        ):
            _require_sha256(getattr(self, field), name=f"expectation.{field}")
        if (
            type(self.source_bundle_name) is not str
            or self.source_bundle_name != f"source-{self.source_bundle_sha256}.tar.gz"
            or type(self.source_bundle_size) is not int
            or self.source_bundle_size < 1
            or type(self.source_bundle_commit_epoch) is not int
            or self.source_bundle_commit_epoch < 1
        ):
            raise ValueError("Smoke expected source-bundle identity is invalid")

    def source_bundle_payload(self) -> dict[str, Any]:
        return {
            "commit_epoch": self.source_bundle_commit_epoch,
            "inventory_sha256": self.source_bundle_inventory_sha256,
            "name": self.source_bundle_name,
            "sha256": self.source_bundle_sha256,
            "size": self.source_bundle_size,
        }


@dataclass(frozen=True)
class DeterminismSmokeArtifactIdentity:
    artifact_manifest_sha256: str
    determinism_smoke_run_sha256: str
    scientific_evidence_sha256: str
    launch_ledger_sha256: str
    model_file_sha256: str
    model_state_sha256: str
    candidate_manifest_sha256: str
    candidate_merged_sha256: str
    loss_manifest_sha256: str
    loss_trace_sha256: str
    validation_manifest_sha256: str
    validation_selection_sha256: str


@dataclass(frozen=True)
class ValidatedDeterminismSmokeArtifact:
    root: Path
    expectation: DeterminismSmokeArtifactExpectation
    identity: DeterminismSmokeArtifactIdentity
    files: tuple[ArtifactFileRecord, ...]
    scientific_evidence: Mapping[str, Any]
    run_path: Path
    model_path: Path


def _require_exact_dict(value: object, keys: set[str] | frozenset[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(keys):
        actual = set(value) if type(value) is dict else set()
        raise ValueError(
            f"{name} fields changed: missing={sorted(set(keys) - actual)}, "
            f"extra={sorted(actual - set(keys))}"
        )
    return value


def _load_canonical_jsonl(path: Path, *, name: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
        raise ValueError(f"{name} must be one non-empty regular file")
    records: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.endswith(b"\n") or raw == b"\n":
                raise ValueError(f"{name}:{line_number} is not one newline-terminated JSON record")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{name}:{line_number} is not valid UTF-8 JSON") from error
            if type(value) is not dict:
                raise TypeError(f"{name}:{line_number} must be one JSON object")
            expected = (_canonical_json(value) + "\n").encode("utf-8")
            if raw != expected:
                raise ValueError(f"{name}:{line_number} is not canonical compact JSON")
            records.append(value)
    return records


def _selection(value: object, *, name: str) -> dict[str, Any]:
    item = _require_exact_dict(value, _SELECTION_KEYS, name=name)
    epoch = item["epoch"]
    step = item["global_step"]
    if (
        type(item["schema_version"]) is not int
        or item["schema_version"] != 1
        or type(epoch) is not int
        or epoch not in range(1, SMOKE_EPOCHS + 1)
        or type(step) is not int
        or step != epoch * SMOKE_UPDATES_PER_EPOCH
        or item["checkpoint_dir"] != f"checkpoint-{step}"
        or item["deepspeed_tag"] != f"global_step{step}"
    ):
        raise ValueError(f"{name} is outside the exact smoke schedule")
    for metric in ("primary_metric", "secondary_metric"):
        if type(item[metric]) is not float or not math.isfinite(item[metric]):
            raise ValueError(f"{name}.{metric} must be one finite exact float")
    _require_sha256(item["ranking_sha256"], name=f"{name}.ranking_sha256")
    return dict(item)


def _validate_wrapper(path: Path) -> dict[str, Any]:
    value = _load_json_object(path, name="Smoke wrapper config", require_canonical=True)
    expected_keys = {
        "schema_version",
        "architecture",
        "slot_token",
        "slot_token_id",
        "temperature",
        "tokenizer_size",
        "weight_dtype",
        "model_artifact_protocol",
    }
    _require_exact_dict(value, expected_keys, name="Smoke wrapper config")
    fixed = {
        "schema_version": 1,
        "architecture": "DualEncoderRetriever",
        "slot_token": CONTROLLED_SLOT_TOKEN,
        "temperature": CONTROLLED_TEMPERATURE,
        "tokenizer_size": CONTROLLED_TOKENIZER_SIZE,
        "weight_dtype": "bfloat16",
        "model_artifact_protocol": SMOKE_MODEL_ARTIFACT_PROTOCOL,
    }
    for field, expected in fixed.items():
        if type(value[field]) is not type(expected) or value[field] != expected:
            raise ValueError(f"Smoke wrapper {field} changed")
    if (
        type(value["slot_token_id"]) is not int
        or value["slot_token_id"] not in range(CONTROLLED_TOKENIZER_SIZE)
    ):
        raise ValueError("Smoke wrapper slot_token_id is invalid")
    return value


def _read_exact(source: Any, count: int, *, name: str) -> bytes:
    value = source.read(count)
    if len(value) != count:
        raise ValueError(f"Safetensors {name} is truncated")
    return value


def _frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _canonical_bf16_safetensors_identity(
    path: Path,
    *,
    expected_metadata: Mapping[str, str] = _EXPECTED_SAFETENSORS_METADATA,
) -> dict[str, Any]:
    """Recompute the canonical tensor-state identity without Torch/safetensors imports."""

    if not path.is_file() or path.is_symlink() or path.stat().st_size < 9:
        raise ValueError("Smoke model must be one non-empty regular safetensors file")
    with path.open("rb") as source:
        header_length = int.from_bytes(_read_exact(source, 8, name="header length"), "little")
        if header_length < 2 or header_length > path.stat().st_size - 8:
            raise ValueError("Safetensors header length is invalid")
        header_raw = _read_exact(source, header_length, name="header")
    try:
        header = json.loads(header_raw.rstrip(b" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Safetensors header is not valid JSON") from error
    if type(expected_metadata) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in expected_metadata.items()
    ):
        raise TypeError("Expected safetensors metadata must be one exact string mapping")
    if type(header) is not dict or header.get("__metadata__") != expected_metadata:
        raise ValueError("Safetensors metadata changed from the expected publication contract")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if set(tensors) != set(_EXPECTED_MODEL_TENSOR_SHAPES):
        missing = sorted(set(_EXPECTED_MODEL_TENSOR_SHAPES) - set(tensors))
        extra = sorted(set(tensors) - set(_EXPECTED_MODEL_TENSOR_SHAPES))
        raise ValueError(
            "Safetensors model-state tensor names changed: "
            f"missing={missing}, extra={extra}"
        )
    data_start = 8 + header_length
    data_size = path.stat().st_size - data_start
    ranges: list[tuple[int, int, str]] = []
    normalized: dict[str, tuple[list[int], int, int]] = {}
    for key, raw in tensors.items():
        if type(key) is not str or not key:
            raise ValueError("Safetensors tensor name is invalid")
        item = _require_exact_dict(raw, {"dtype", "shape", "data_offsets"}, name=f"tensor {key!r}")
        shape = item["shape"]
        offsets = item["data_offsets"]
        if (
            item["dtype"] != "BF16"
            or type(shape) is not list
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise ValueError(f"Safetensors tensor {key!r} metadata is invalid")
        if tuple(shape) != _EXPECTED_MODEL_TENSOR_SHAPES[key]:
            raise ValueError(
                f"Safetensors tensor {key!r} shape changed: "
                f"actual={shape}, expected={list(_EXPECTED_MODEL_TENSOR_SHAPES[key])}"
            )
        elements = math.prod(shape)
        if offsets[1] - offsets[0] != elements * 2:
            raise ValueError(f"Safetensors tensor {key!r} byte range disagrees with BF16 shape")
        ranges.append((offsets[0], offsets[1], key))
        normalized[key] = (shape, offsets[0], offsets[1])
    cursor = 0
    for start, end, _ in sorted(ranges):
        if start != cursor:
            raise ValueError("Safetensors tensor byte ranges contain a gap or overlap")
        cursor = end
    if cursor != data_size:
        raise ValueError("Safetensors tensor byte ranges do not cover the complete data buffer")

    digest = hashlib.sha256()
    _frame(digest, SMOKE_MODEL_STATE_PROTOCOL.encode("utf-8"))
    digest.update(len(tensors).to_bytes(8, byteorder="big", signed=False))
    with path.open("rb") as source:
        for key in sorted(tensors):
            shape, start, end = normalized[key]
            _frame(digest, key.encode("utf-8"))
            _frame(digest, b"torch.bfloat16")
            digest.update(len(shape).to_bytes(8, byteorder="big", signed=False))
            for dimension in shape:
                digest.update(dimension.to_bytes(8, byteorder="big", signed=False))
            byte_count = end - start
            digest.update(byte_count.to_bytes(8, byteorder="big", signed=False))
            source.seek(data_start + start)
            remaining = byte_count
            carry = b""
            while remaining:
                chunk = _read_exact(source, min(1024 * 1024, remaining), name=f"tensor {key!r}")
                digest.update(chunk)
                finite_bytes = carry + chunk
                if len(finite_bytes) % 2:
                    carry = finite_bytes[-1:]
                    finite_bytes = finite_bytes[:-1]
                else:
                    carry = b""
                for index in range(0, len(finite_bytes), 2):
                    bits = finite_bytes[index] | (finite_bytes[index + 1] << 8)
                    if bits & 0x7F80 == 0x7F80:
                        raise FloatingPointError(f"Safetensors tensor {key!r} contains non-finite BF16")
                remaining -= len(chunk)
            if carry:
                raise RuntimeError("BF16 safetensors tensor has an odd byte count")
    return {
        "protocol": SMOKE_MODEL_STATE_PROTOCOL,
        "tensor_count": SMOKE_EXPECTED_MODEL_TENSOR_COUNT,
        "sha256": digest.hexdigest(),
    }


def _validate_candidate_traces(
    root: Path,
    record: object,
) -> tuple[tuple[ArtifactFileRecord, ...], dict[str, Any], list[list[dict[str, Any]]]]:
    manifest_record = _validate_file_record(
        root,
        record,
        name="artifact_manifest.candidate_trace_manifest",
        expected_path="candidate_traces/manifest.json",
    )
    trace_root = root / "candidate_traces"
    manifest = _load_json_object(
        trace_root / "manifest.json", name="Smoke candidate manifest", require_canonical=True
    )
    keys = {
        "schema_version",
        "merge_order",
        "epochs",
        "queries_per_epoch",
        "record_count",
        "query_ids_sha256",
        "passage_index_sha256",
        "merged",
        "shards",
    }
    _require_exact_dict(manifest, keys, name="Smoke candidate manifest")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["merge_order"] != ["epoch", "query_id"]
        or type(manifest["epochs"]) is not int
        or manifest["epochs"] != SMOKE_EPOCHS
        or type(manifest["queries_per_epoch"]) is not int
        or manifest["queries_per_epoch"] != 294
        or type(manifest["record_count"]) is not int
        or manifest["record_count"] != SMOKE_TOTAL_QUERY_LINKS
        or manifest["query_ids_sha256"] != EXPECTED_TRAIN_QUERY_IDS_SHA256_BY_OUTER_FOLD[0]
        or manifest["passage_index_sha256"] != EXPECTED_PASSAGE_INDEX_SHA256
    ):
        raise ValueError("Smoke candidate manifest left its fixed coverage/provenance contract")

    merged = _require_exact_dict(
        manifest["merged"], {"path", "record_count", "size", "sha256"}, name="Smoke merged candidate record"
    )
    if (
        type(merged["record_count"]) is not int
        or merged["record_count"] != SMOKE_TOTAL_QUERY_LINKS
    ):
        raise ValueError("Smoke merged candidate count changed")
    merged_file = _validate_file_record(
        trace_root,
        {field: merged[field] for field in ("path", "size", "sha256")},
        name="Smoke merged candidate file",
        expected_path="sampling_traces.jsonl",
    )
    shards = manifest["shards"]
    if type(shards) is not list or len(shards) != SMOKE_WORLD_SIZE:
        raise ValueError("Smoke candidate manifest must contain four rank shards")
    per_rank: list[list[dict[str, Any]]] = []
    shard_files: list[ArtifactFileRecord] = []
    all_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    expected_rank_counts = [
        SMOKE_EPOCHS * count for count in SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH
    ]
    for rank, raw_shard in enumerate(shards):
        shard = _require_exact_dict(
            raw_shard, {"rank", "path", "record_count", "size", "sha256"}, name=f"Smoke candidate shard {rank}"
        )
        if (
            type(shard["rank"]) is not int
            or shard["rank"] != rank
            or type(shard["record_count"]) is not int
            or shard["record_count"] != expected_rank_counts[rank]
        ):
            raise ValueError(f"Smoke candidate shard {rank} dimensions changed")
        file_record = _validate_file_record(
            trace_root,
            {field: shard[field] for field in ("path", "size", "sha256")},
            name=f"Smoke candidate shard {rank}",
            expected_path=f"rank-{rank:05d}.jsonl",
        )
        records = _load_canonical_jsonl(trace_root / file_record.path, name=f"Smoke candidate shard {rank}")
        if len(records) != expected_rank_counts[rank]:
            raise ValueError(f"Smoke candidate shard {rank} line count changed")
        for trace in records:
            validate_sampling_trace(trace)
            if (
                trace["sampler"] != SMOKE_CELL["sampler"]
                or trace["experiment_seed"] != SMOKE_CELL["experiment_seed"]
                or trace["epoch"] not in range(SMOKE_EPOCHS)
            ):
                raise ValueError("Smoke candidate trace left its fixed sampler/seed/epoch cell")
            key = (trace["epoch"], trace["query_id"])
            if key in all_by_key:
                raise ValueError(f"Duplicate smoke candidate trace key: {key!r}")
            all_by_key[key] = trace
        per_rank.append(records)
        shard_files.append(file_record)
    query_ids = sorted({query_id for _, query_id in all_by_key})
    expected_keys = {(epoch, query_id) for epoch in range(SMOKE_EPOCHS) for query_id in query_ids}
    if len(query_ids) != 294 or set(all_by_key) != expected_keys:
        raise ValueError("Smoke candidate traces do not cover exactly 294 queries in both epochs")
    if _canonical_sha256(query_ids) != manifest["query_ids_sha256"]:
        raise ValueError("Smoke candidate query-ID digest is not reproduced by shard contents")
    expected_merged = b"".join(
        (_canonical_json(all_by_key[key]) + "\n").encode("utf-8") for key in sorted(all_by_key)
    )
    if (trace_root / merged_file.path).read_bytes() != expected_merged:
        raise ValueError("Smoke merged candidate trace is not the canonical shard merge")
    actual = _regular_tree_inventory(trace_root)
    expected_paths = {
        "manifest.json",
        "sampling_traces.jsonl",
        *(f"rank-{rank:05d}.jsonl" for rank in range(SMOKE_WORLD_SIZE)),
    }
    if {item.path for item in actual} != expected_paths:
        raise ValueError("Smoke candidate-trace directory inventory changed")
    records = tuple(
        ArtifactFileRecord(f"candidate_traces/{item.path}", item.size, item.sha256)
        for item in actual
    )
    if manifest_record not in records:
        raise ValueError("Smoke candidate manifest record changed during validation")
    return records, manifest, per_rank


def _validate_loss_traces(
    root: Path,
    record: object,
    candidate_by_rank: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[tuple[ArtifactFileRecord, ...], dict[str, Any], dict[str, Any]]:
    manifest_record = _validate_file_record(
        root,
        record,
        name="artifact_manifest.loss_trace_manifest",
        expected_path="loss_traces/manifest.json",
    )
    loss_root = root / "loss_traces"
    manifest = _load_json_object(
        loss_root / "manifest.json", name="Smoke loss manifest", require_canonical=True
    )
    _require_exact_dict(manifest, {"schema_version", "identity", "shards"}, name="Smoke loss manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Smoke loss manifest schema version changed")
    shards = manifest["shards"]
    if type(shards) is not list or len(shards) != SMOKE_WORLD_SIZE:
        raise ValueError("Smoke loss manifest must contain four shards")
    per_rank: list[list[dict[str, Any]]] = []
    for rank, raw in enumerate(shards):
        shard = _require_exact_dict(
            raw, {"rank", "path", "record_count", "size", "sha256"}, name=f"Smoke loss shard {rank}"
        )
        if (
            shard["rank"] != rank
            or type(shard["rank"]) is not int
            or type(shard["record_count"]) is not int
            or shard["record_count"] != 38
        ):
            raise ValueError(f"Smoke loss shard {rank} dimensions changed")
        file_record = _validate_file_record(
            loss_root,
            {field: shard[field] for field in ("path", "size", "sha256")},
            name=f"Smoke loss shard {rank}",
            expected_path=f"rank-{rank:05d}.jsonl",
        )
        records = _load_canonical_jsonl(loss_root / file_record.path, name=f"Smoke loss shard {rank}")
        if len(records) != 38 or shard["record_count"] != len(records):
            raise ValueError(f"Smoke loss shard {rank} line count changed")
        candidate_links = [
            (trace["epoch"], trace["query_id"], trace["trace_sha256"])
            for trace in candidate_by_rank[rank]
        ]
        loss_links = [
            (record["epoch"], query_id, digest)
            for record in records
            for query_id, digest in zip(record.get("query_ids", []), record.get("candidate_trace_sha256", []))
        ]
        if loss_links != candidate_links:
            raise ValueError(f"Smoke loss shard {rank} does not link its candidate shard exactly")
        per_rank.append(records)
    identity = build_smoke_loss_trace_identity(per_rank)
    if manifest["identity"] != identity:
        raise ValueError("Smoke loss identity is not reproduced by its four JSONL shards")
    actual = _regular_tree_inventory(loss_root)
    expected_paths = {"manifest.json", *(f"rank-{rank:05d}.jsonl" for rank in range(SMOKE_WORLD_SIZE))}
    if {item.path for item in actual} != expected_paths:
        raise ValueError("Smoke loss-trace directory inventory changed")
    records = tuple(
        ArtifactFileRecord(f"loss_traces/{item.path}", item.size, item.sha256) for item in actual
    )
    if manifest_record not in records:
        raise ValueError("Smoke loss manifest record changed during validation")
    return records, manifest, identity


def _validate_validation(
    root: Path, record: object
) -> tuple[tuple[ArtifactFileRecord, ...], dict[str, Any], str, dict[str, Any]]:
    manifest_record = _validate_file_record(
        root,
        record,
        name="artifact_manifest.validation_manifest",
        expected_path="validation/manifest.json",
    )
    validation_root = root / "validation"
    manifest = _load_json_object(
        validation_root / "manifest.json", name="Smoke validation manifest", require_canonical=True
    )
    keys = {
        "schema_version",
        "epochs",
        "selection_order",
        "best",
        "last",
        "retained_checkpoint_dirs",
        "records",
        "history_sha256",
        "best_sha256",
        "latest_sha256",
    }
    _require_exact_dict(manifest, keys, name="Smoke validation manifest")
    if (
        manifest["schema_version"] != 1
        or type(manifest["schema_version"]) is not int
        or type(manifest["epochs"]) is not int
        or manifest["epochs"] != 2
    ):
        raise ValueError("Smoke validation manifest dimensions changed")
    if manifest["selection_order"] != _EXPECTED_SELECTION_ORDER:
        raise ValueError("Smoke validation selection order changed")
    declared_best = _selection(manifest["best"], name="Smoke validation best")
    declared_last = _selection(manifest["last"], name="Smoke validation last")
    records = manifest["records"]
    if type(records) is not list or len(records) != 2:
        raise ValueError("Smoke validation history must contain exactly two records")
    epoch_payloads: list[dict[str, Any]] = []
    recomputed_best: dict[str, Any] | None = None
    recomputed_last: dict[str, Any] | None = None
    for epoch, history in enumerate(records, start=1):
        entry = _require_exact_dict(
            history,
            {"epoch", "global_step", "path", "sha256", "is_new_best", "candidate", "best_after_epoch"},
            name=f"Smoke validation history {epoch}",
        )
        expected_path = f"epoch-{epoch:03d}.json"
        if (
            type(entry["epoch"]) is not int
            or entry["epoch"] != epoch
            or type(entry["global_step"]) is not int
            or entry["global_step"] != epoch * 3
            or entry["path"] != expected_path
            or type(entry["is_new_best"]) is not bool
        ):
            raise ValueError("Smoke validation history sequence changed")
        candidate = _selection(
            entry["candidate"], name=f"Smoke validation candidate {epoch}"
        )
        if (
            candidate["epoch"] != epoch
            or candidate["global_step"] != epoch * SMOKE_UPDATES_PER_EPOCH
        ):
            raise ValueError(
                f"Smoke validation candidate {epoch} is assigned to another epoch"
            )
        declared_best_after = _selection(
            entry["best_after_epoch"], name=f"Smoke validation best-after {epoch}"
        )
        epoch_path = validation_root / expected_path
        if _require_sha256(entry["sha256"], name=f"Smoke validation epoch {epoch} sha256") != _sha256_file(epoch_path):
            raise ValueError(f"Smoke validation epoch {epoch} digest changed")
        payload = _load_json_object(epoch_path, name=f"Smoke validation epoch {epoch}", require_canonical=True)
        _require_exact_dict(
            payload,
            {"schema_version", "epoch", "global_step", "checkpoint", "candidate", "is_new_best", "best_after_epoch", "validation_result"},
            name=f"Smoke validation epoch {epoch}",
        )
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError(f"Smoke validation epoch {epoch} schema version changed")
        if not _exact_json_equal({
            key: payload[key]
            for key in ("epoch", "global_step", "candidate", "is_new_best", "best_after_epoch")
        }, {
            key: entry[key]
            for key in ("epoch", "global_step", "candidate", "is_new_best", "best_after_epoch")
        }):
            raise ValueError(f"Smoke validation epoch {epoch} and history entry disagree")
        validation_result = _validate_fold_global_result(
            payload["validation_result"],
            name=f"Smoke validation epoch {epoch} result",
        )
        if validation_result["ranking_sha256"] != candidate["ranking_sha256"]:
            raise ValueError(
                f"Smoke validation epoch {epoch} result and candidate ranking differ"
            )
        checkpoint = _require_exact_dict(
            payload["checkpoint"],
            {
                "checkpoint_dir",
                "deepspeed_tag",
                "manifest_sha256",
                "scheduler_state_sha256",
                "client_state_sha256",
            },
            name=f"Smoke validation epoch {epoch} checkpoint",
        )
        if (
            checkpoint["checkpoint_dir"] != payload["candidate"]["checkpoint_dir"]
            or checkpoint["deepspeed_tag"] != payload["candidate"]["deepspeed_tag"]
        ):
            raise ValueError(
                f"Smoke validation epoch {epoch} checkpoint and candidate disagree"
            )
        for digest_name in (
            "manifest_sha256",
            "scheduler_state_sha256",
            "client_state_sha256",
        ):
            _require_sha256(
                checkpoint[digest_name],
                name=f"Smoke validation epoch {epoch} checkpoint {digest_name}",
            )
        metrics = validation_result["metrics"]
        if not _exact_json_equal(
            {
                "primary": metrics[_VALIDATION_PRIMARY_METRIC],
                "secondary": metrics[_VALIDATION_SECONDARY_METRIC],
            },
            {
                "primary": candidate["primary_metric"],
                "secondary": candidate["secondary_metric"],
            },
        ):
            raise ValueError(
                f"Smoke validation epoch {epoch} result and candidate metrics disagree"
            )
        is_new_best = (
            recomputed_best is None
            or candidate["primary_metric"] > recomputed_best["primary_metric"]
            or (
                candidate["primary_metric"] == recomputed_best["primary_metric"]
                and candidate["secondary_metric"]
                > recomputed_best["secondary_metric"]
            )
        )
        if is_new_best:
            recomputed_best = candidate
        if (
            entry["is_new_best"] is not is_new_best
            or payload["is_new_best"] is not is_new_best
            or not _exact_json_equal(declared_best_after, recomputed_best)
            or not _exact_json_equal(payload["best_after_epoch"], recomputed_best)
        ):
            raise ValueError(
                f"Smoke validation epoch {epoch} best-selection chronology changed"
            )
        recomputed_last = candidate
        if checkpoint["checkpoint_dir"] in manifest["retained_checkpoint_dirs"]:
            checkpoint_root = root / checkpoint["checkpoint_dir"]
            checkpoint_manifest = _load_json_object(
                checkpoint_root / "checkpoint_manifest.json",
                name=f"Smoke retained checkpoint {checkpoint['checkpoint_dir']}",
                require_canonical=True,
            )
            if not _exact_json_equal(
                checkpoint_manifest.get("selection"), candidate
            ):
                raise ValueError(
                    f"Smoke validation epoch {epoch} retained checkpoint selection changed"
                )
            if (
                checkpoint["manifest_sha256"]
                != _sha256_file(checkpoint_root / "checkpoint_manifest.json")
                or checkpoint["scheduler_state_sha256"]
                != checkpoint_manifest.get("scheduler_state_sha256")
                or checkpoint["client_state_sha256"]
                != checkpoint_manifest.get("client_state_sha256")
            ):
                raise ValueError(
                    f"Smoke validation epoch {epoch} retained checkpoint evidence changed"
                )
        epoch_payloads.append(payload)
    if recomputed_best is None or recomputed_last is None:
        raise RuntimeError("Smoke validation chronology produced no selection")
    expected_retained = sorted(
        {recomputed_best["checkpoint_dir"], recomputed_last["checkpoint_dir"]},
        key=lambda value: int(value.removeprefix("checkpoint-")),
    )
    if (
        not _exact_json_equal(declared_best, recomputed_best)
        or not _exact_json_equal(declared_last, recomputed_last)
        or not _exact_json_equal(
            manifest["retained_checkpoint_dirs"], expected_retained
        )
        or recomputed_last["epoch"] != 2
        or recomputed_last["global_step"] != SMOKE_TOTAL_OPTIMIZER_UPDATES
    ):
        raise ValueError("Smoke validation final best/last/retained chronology changed")
    best = recomputed_best
    last = recomputed_last
    history_payload = _load_json_object(validation_root / "history.json", name="Smoke validation history", require_canonical=True)
    if not _exact_json_equal(
        history_payload, {"schema_version": 1, "records": records}
    ):
        raise ValueError("Smoke validation history.json differs from its manifest")
    if not _exact_json_equal(
        _load_json_object(
            validation_root / "latest.json",
            name="Smoke latest selection",
            require_canonical=True,
        ),
        records[-1],
    ):
        raise ValueError("Smoke validation latest.json differs from its manifest")
    best_record = next((entry for entry in records if entry["candidate"] == best), None)
    if best_record is None or not _exact_json_equal(
        _load_json_object(
            validation_root / "best.json",
            name="Smoke best selection",
            require_canonical=True,
        ),
        best_record,
    ):
        raise ValueError("Smoke validation best.json differs from its manifest")
    for digest_name, filename in (
        ("history_sha256", "history.json"),
        ("best_sha256", "best.json"),
        ("latest_sha256", "latest.json"),
    ):
        if _require_sha256(manifest[digest_name], name=f"Smoke validation {digest_name}") != _sha256_file(validation_root / filename):
            raise ValueError(f"Smoke validation {filename} digest changed")
    normalized_records = [
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
        for payload in epoch_payloads
    ]
    selection_sha256 = _canonical_sha256({"records": normalized_records, "best": best, "last": last})
    actual = _regular_tree_inventory(validation_root)
    expected_paths = {
        "manifest.json", "history.json", "best.json", "latest.json", "epoch-001.json", "epoch-002.json"
    }
    if {item.path for item in actual} != expected_paths:
        raise ValueError("Smoke validation directory inventory changed")
    prefixed = tuple(
        ArtifactFileRecord(f"validation/{item.path}", item.size, item.sha256) for item in actual
    )
    if manifest_record not in prefixed:
        raise ValueError("Smoke validation manifest record changed during validation")
    best_payloads = [
        payload for payload in epoch_payloads if payload["candidate"] == best
    ]
    if len(best_payloads) != 1:
        raise ValueError("Smoke validation selected best epoch is ambiguous or absent")
    return prefixed, manifest, selection_sha256, dict(
        best_payloads[0]["validation_result"]
    )


def _validate_frozen_run(
    run: object,
    *,
    root: Path,
    expectation: DeterminismSmokeArtifactExpectation,
    manifest: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    loss_manifest: Mapping[str, Any],
    loss_identity: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    validation_selection_sha256: str,
    selected_validation_result: Mapping[str, Any],
    retained_names: Sequence[str],
    model_state_identity: Mapping[str, Any],
    tokenizer_records: Sequence[ArtifactFileRecord],
    encoder_records: Sequence[ArtifactFileRecord],
) -> dict[str, Any]:
    value = _require_exact_dict(run, _RUN_KEYS, name="determinism_smoke_run.json")
    fixed_identity = {
        "schema_version": 1,
        "experiment_id": CONTROLLED_EXPERIMENT_ID,
        **dict(SMOKE_CELL),
        "run_kind": SMOKE_RUN_KIND,
    }
    for field, expected in fixed_identity.items():
        if type(value[field]) is not type(expected) or value[field] != expected:
            raise ValueError(f"Smoke run identity {field} changed")
    if not _exact_json_equal(value["schedule"], {
        "epochs": SMOKE_EPOCHS,
        "updates_per_epoch": SMOKE_UPDATES_PER_EPOCH,
        "total_optimizer_updates": SMOKE_TOTAL_OPTIMIZER_UPDATES,
    }):
        raise ValueError("Smoke run schedule changed")
    if not _exact_json_equal(value["runtime_versions"], EXPECTED_RUNTIME_VERSIONS):
        raise ValueError("Smoke run runtime versions changed")
    frozen_launch = {
        "training_image": EXPECTED_DERIVED_TRAINING_IMAGE,
        "training_base_image": EXPECTED_BASE_TRAINING_IMAGE,
        "training_image_runtime_inventory_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
        "training_image_contract_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
        "training_bootstrap_protocol": EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
        "training_plan_sha256": expectation.training_plan_sha256,
        "training_staging_receipt_sha256": expectation.training_staging_receipt_sha256,
        "source_bundle": expectation.source_bundle_payload(),
    }
    for field, expected in frozen_launch.items():
        if not _exact_json_equal(value[field], expected):
            raise ValueError(f"Smoke run launch provenance {field} changed")
    if not _exact_json_equal(value["experiment_config"], {"path": "experiment.json", "sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256}):
        raise ValueError("Smoke experiment config changed")
    if not _exact_json_equal(value["deepspeed_config"], {"path": "ds_zero3.json", "sha256": EXPECTED_DEEPSPEED_CONFIG_SHA256}):
        raise ValueError("Smoke DeepSpeed config changed")
    if not _exact_json_equal(value["dataset"], {
        "manifest_path": "dataset_manifest.json",
        "manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "output_sha256": EXPECTED_DATASET_OUTPUT_SHA256,
    }):
        raise ValueError("Smoke dataset identity changed")
    folds = _require_exact_dict(value["folds"], {"manifest_path", "manifest_sha256", "rotation"}, name="Smoke folds")
    if (
        folds["manifest_path"] != EXPECTED_FOLD_MANIFEST_LOGICAL_PATH
        or folds["manifest_sha256"] != EXPECTED_FOLD_MANIFEST_SHA256
        or _canonical_sha256(folds["rotation"]) != EXPECTED_FOLD_ROTATION_SHA256_BY_OUTER_FOLD[0]
    ):
        raise ValueError("Smoke fold rotation left the frozen fold manifest")
    rotation = folds["rotation"]
    if type(rotation) is not dict or rotation.get("outer_fold") != 0 or rotation.get("train", {}).get("queries") != 294:
        raise ValueError("Smoke fold rotation dimensions changed")
    if not _exact_json_equal(value["snapshot"], {
        "manifest_path": "modernbert_snapshot.json",
        "manifest_sha256": EXPECTED_SNAPSHOT_MANIFEST_SHA256,
        "tree_sha256": EXPECTED_SNAPSHOT_TREE_SHA256,
    }):
        raise ValueError("Smoke ModernBERT snapshot identity changed")
    if not _exact_json_equal(value["passage_index"], {"schema_version": 1, "size": 5286, "sha256": EXPECTED_PASSAGE_INDEX_SHA256}):
        raise ValueError("Smoke passage-index identity changed")
    validation_data = _require_exact_dict(
        value["validation_data"],
        {"role", "query_view", "case_count", "query_count", "passage_count", "case_ids_sha256", "query_ids_sha256", "passage_ids_sha256", "contract_sha256"},
        name="Smoke validation data",
    )
    validation_role = rotation["validation"]
    if not _exact_json_equal(
        validation_data,
        {
            "role": "validation",
            "query_view": "structured",
            "case_count": validation_role["num_cases"],
            "query_count": validation_role["queries"],
            "passage_count": validation_role["passages"],
            **EXPECTED_VALIDATION_IDENTITY_BY_CELL[(0, "structured")],
        },
    ):
        raise ValueError("Smoke validation-data identity changed")
    if not _exact_json_equal(value["candidate_traces"], {
        "manifest_path": "candidate_traces/manifest.json",
        "manifest_sha256": manifest["candidate_trace_manifest"]["sha256"],
        "record_count": SMOKE_TOTAL_QUERY_LINKS,
        "merged_sha256": candidate_manifest["merged"]["sha256"],
    }):
        raise ValueError("Smoke run candidate-trace identity changed")
    if not _exact_json_equal(value["loss_traces"], {
        "manifest_path": "loss_traces/manifest.json",
        "manifest_sha256": manifest["loss_trace_manifest"]["sha256"],
        "identity": loss_identity,
    }):
        raise ValueError("Smoke run loss-trace identity changed")
    if not _exact_json_equal(value["validation_history"], {
        "manifest_path": "validation/manifest.json",
        "manifest_sha256": manifest["validation_manifest"]["sha256"],
        "best": validation_manifest["best"],
        "last": validation_manifest["last"],
        "retained_checkpoint_dirs": list(retained_names),
    }):
        raise ValueError("Smoke validation-history identity changed")
    reload = _require_exact_dict(value["best_checkpoint_reload"], {"selection", "validation_result", "per_rank"}, name="Smoke best-checkpoint reload")
    if (
        not _exact_json_equal(reload["selection"], validation_manifest["best"])
        or not _exact_json_equal(
            reload["validation_result"], selected_validation_result
        )
    ):
        raise ValueError("Smoke best-checkpoint reload selection/result changed")
    ranks = reload["per_rank"]
    reload_keys = {"rank", "load_path_parent", "client_state_sha256", "scheduler_state_sha256", "global_step", "rng_sha256", "manifest_sha256"}
    if type(ranks) is not list or len(ranks) != SMOKE_WORLD_SIZE:
        raise ValueError("Smoke reload must contain four ranks")
    for rank, raw in enumerate(ranks):
        item = _require_exact_dict(raw, reload_keys, name=f"Smoke reload rank {rank}")
        if (
            type(item["rank"]) is not int
            or item["rank"] != rank
            or type(item["global_step"]) is not int
            or item["global_step"] != reload["selection"]["global_step"]
            or type(item["load_path_parent"]) is not str
            or not item["load_path_parent"]
        ):
            raise ValueError(f"Smoke reload rank {rank} identity changed")
        load_parent = Path(item["load_path_parent"])
        if (
            load_parent.name != reload["selection"]["deepspeed_tag"]
            or load_parent.parent.name != reload["selection"]["checkpoint_dir"]
        ):
            raise ValueError(f"Smoke reload rank {rank} logical checkpoint path changed")
        for digest_name in ("client_state_sha256", "scheduler_state_sha256", "rng_sha256", "manifest_sha256"):
            _require_sha256(item[digest_name], name=f"Smoke reload rank {rank} {digest_name}")
    if len({item["client_state_sha256"] for item in ranks}) != 1 or len({item["scheduler_state_sha256"] for item in ranks}) != 1:
        raise ValueError("Smoke reload client/scheduler state differs across ranks")
    selected_checkpoint_root = root / reload["selection"]["checkpoint_dir"]
    selected_checkpoint_manifest_path = selected_checkpoint_root / "checkpoint_manifest.json"
    selected_checkpoint_manifest = _load_json_object(
        selected_checkpoint_manifest_path,
        name="Smoke selected checkpoint manifest",
        require_canonical=True,
    )
    selected_checkpoint_manifest_sha256 = _sha256_file(
        selected_checkpoint_manifest_path
    )
    selected_checkpoint_files = selected_checkpoint_manifest.get("files")
    if type(selected_checkpoint_files) is not list:
        raise ValueError("Smoke selected checkpoint file inventory is malformed")
    selected_checkpoint_file_by_path = {
        record.get("path"): record
        for record in selected_checkpoint_files
        if type(record) is dict
    }
    if any(
        not _exact_json_equal(
            selected_checkpoint_manifest.get("selection"), reload["selection"]
        )
        or item["manifest_sha256"] != selected_checkpoint_manifest_sha256
        or item["client_state_sha256"]
        != selected_checkpoint_manifest.get("client_state_sha256")
        or item["scheduler_state_sha256"]
        != selected_checkpoint_manifest.get("scheduler_state_sha256")
        or selected_checkpoint_file_by_path.get(f"rng_state_{rank}.pth", {}).get(
            "sha256"
        )
        != item["rng_sha256"]
        for rank, item in enumerate(ranks)
    ):
        raise ValueError(
            "Smoke reload evidence differs from selected checkpoint state/RNG files"
        )
    final = _require_exact_dict(value["final_model"], {"path", "size", "sha256", "weight_dtype", "gathered_tensor_count", "strict_round_trip_tensor_count"}, name="Smoke final model")
    if (
        not _exact_json_equal(
            {field: final[field] for field in ("path", "size", "sha256")},
            manifest["model"],
        )
        or final["weight_dtype"] != "bfloat16"
        or type(final["gathered_tensor_count"]) is not int
        or final["gathered_tensor_count"] != CONTROLLED_MODEL_STATE_COUNT
        or type(final["strict_round_trip_tensor_count"]) is not int
        or final["strict_round_trip_tensor_count"] != CONTROLLED_MODEL_STATE_COUNT
    ):
        raise ValueError("Smoke final-model identity changed")
    for field in ("tokenizer", "encoder_config", "wrapper_config", "retained_checkpoints"):
        if not _exact_json_equal(value[field], manifest[field]):
            raise ValueError(f"Smoke run {field} differs from artifact manifest")
    if wrapper["model_artifact_protocol"] != SMOKE_MODEL_ARTIFACT_PROTOCOL:
        raise ValueError("Smoke wrapper model protocol changed")

    evidence = validate_smoke_scientific_evidence(value["determinism_scientific_evidence"])
    expected_candidate = {
        "manifest_sha256": manifest["candidate_trace_manifest"]["sha256"],
        "merged_sha256": candidate_manifest["merged"]["sha256"],
        "record_count": SMOKE_TOTAL_QUERY_LINKS,
        "rank_shards": [
            {"rank": shard["rank"], "record_count": shard["record_count"], "sha256": shard["sha256"]}
            for shard in candidate_manifest["shards"]
        ],
    }
    if evidence["candidate_traces"] != expected_candidate or evidence["loss_traces"] != loss_manifest["identity"]:
        raise ValueError("Smoke scientific trace evidence is not reproduced by artifact bytes")
    if evidence["validation_selection"] != {"epochs": 2, "sha256": validation_selection_sha256}:
        raise ValueError("Smoke scientific validation evidence is not reproduced by artifact bytes")
    expected_reload = {
        "validation_sha256": _canonical_sha256(reload["validation_result"]),
        "scheduler_state_sha256": ranks[0]["scheduler_state_sha256"],
        "client_state_sha256": ranks[0]["client_state_sha256"],
        "per_rank_rng_sha256": [item["rng_sha256"] for item in ranks],
    }
    if evidence["reload"] != expected_reload:
        raise ValueError("Smoke scientific reload evidence is not reproduced by run metadata")
    expected_final = {
        "model_sha256": manifest["model"]["sha256"],
        "tokenizer_inventory_sha256": _canonical_sha256(
            [
                {
                    **_record_payload(item),
                    "path": item.path.removeprefix("tokenizer/"),
                }
                for item in tokenizer_records
            ]
        ),
        "encoder_config_sha256": _canonical_sha256(
            [
                {
                    **_record_payload(item),
                    "path": item.path.removeprefix("encoder_config/"),
                }
                for item in encoder_records
            ]
        ),
        "wrapper_config_sha256": manifest["wrapper_config"]["sha256"],
    }
    if evidence["final_artifacts"] != expected_final:
        raise ValueError("Smoke scientific final-artifact evidence is not reproduced by artifact bytes")
    if evidence["model_states"]["selected"] != model_state_identity or evidence["model_states"]["roundtrip"] != model_state_identity:
        raise ValueError("Smoke selected/round-trip model-state evidence is not reproduced by safetensors")
    launch_payload = {
        "training_image": EXPECTED_DERIVED_TRAINING_IMAGE,
        "training_base_image": EXPECTED_BASE_TRAINING_IMAGE,
        "runtime_inventory_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
        "training_image_contract_sha256": EXPECTED_DERIVED_TRAINING_IMAGE_CONTRACT_SHA256,
        "bootstrap_protocol": EXPECTED_TRAINING_BOOTSTRAP_PROTOCOL,
        "training_plan_sha256": expectation.training_plan_sha256,
        "training_staging_receipt_sha256": expectation.training_staging_receipt_sha256,
        "source_bundle": expectation.source_bundle_payload(),
    }
    expected_ledger = _canonical_sha256(launch_payload)
    if evidence["launch_ledger"] != {"sha256": expected_ledger}:
        raise ValueError("Smoke scientific launch ledger is not reproduced by external expectations")
    return evidence


def validate_determinism_smoke_artifact(
    root: Path,
    *,
    expectation: DeterminismSmokeArtifactExpectation,
) -> ValidatedDeterminismSmokeArtifact:
    """Validate a committed two-epoch smoke artifact without ML dependencies."""

    if not isinstance(expectation, DeterminismSmokeArtifactExpectation):
        raise TypeError("expectation must be DeterminismSmokeArtifactExpectation")
    if not isinstance(root, Path):
        raise TypeError("Smoke artifact root must be a Path")
    root = root.expanduser()
    if root.is_symlink():
        raise ValueError("Smoke artifact root must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Smoke artifact root must be a real non-symlink directory")
    initial_inventory = _regular_tree_inventory(root)
    manifest_path = root / "artifact_manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != expectation.artifact_manifest_sha256:
        raise ValueError("Smoke artifact commit-marker SHA-256 changed")
    manifest = _load_json_object(manifest_path, name="Smoke artifact commit marker", require_canonical=True)
    _require_exact_dict(manifest, _MANIFEST_KEYS, name="Smoke artifact manifest")
    if manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int or manifest["commit_marker"] is not True or type(manifest["commit_marker"]) is not bool or manifest["artifact_type"] != SMOKE_ARTIFACT_TYPE:
        raise ValueError("Smoke artifact commit marker is invalid")
    direct_records = [
        _validate_file_record(root, manifest["determinism_smoke_run"], name="artifact_manifest.determinism_smoke_run", expected_path="determinism_smoke_run.json"),
        _validate_file_record(root, manifest["model"], name="artifact_manifest.model", expected_path="model.safetensors"),
        _validate_file_record(root, manifest["wrapper_config"], name="artifact_manifest.wrapper_config", expected_path="wrapper_config.json"),
    ]
    tokenizer_records = _validate_directory_record(
        root, manifest["tokenizer"], name="artifact_manifest.tokenizer", expected_path="tokenizer", expected_files=("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
    )
    encoder_records = _validate_directory_record(
        root, manifest["encoder_config"], name="artifact_manifest.encoder_config", expected_path="encoder_config", expected_files=("config.json",)
    )
    checkpoint_records, retained_names = _validate_checkpoint_inventory(root, manifest["retained_checkpoints"])
    for checkpoint_name in retained_names:
        checkpoint_manifest = _load_json_object(
            root / checkpoint_name / "checkpoint_manifest.json",
            name=f"Smoke retained checkpoint {checkpoint_name}",
            require_canonical=True,
        )
        if (
            type(checkpoint_manifest.get("schema_version")) is not int
            or checkpoint_manifest["schema_version"] != 1
            or type(checkpoint_manifest.get("world_size")) is not int
            or checkpoint_manifest["world_size"] != SMOKE_WORLD_SIZE
        ):
            raise ValueError(
                f"Smoke retained checkpoint {checkpoint_name} version/world size changed"
            )
    candidate_records, candidate_manifest, candidate_by_rank = _validate_candidate_traces(root, manifest["candidate_trace_manifest"])
    loss_records, loss_manifest, loss_identity = _validate_loss_traces(root, manifest["loss_trace_manifest"], candidate_by_rank)
    (
        validation_records,
        validation_manifest,
        validation_selection_sha256,
        selected_validation_result,
    ) = _validate_validation(root, manifest["validation_manifest"])
    if list(retained_names) != validation_manifest["retained_checkpoint_dirs"]:
        raise ValueError("Smoke checkpoint inventory and validation manifest disagree")
    wrapper = _validate_wrapper(root / "wrapper_config.json")
    model_state_identity = _canonical_bf16_safetensors_identity(root / "model.safetensors")
    run_path = root / "determinism_smoke_run.json"
    run = _load_json_object(run_path, name="Smoke run metadata", require_canonical=True)
    evidence = _validate_frozen_run(
        run,
        root=root,
        expectation=expectation,
        manifest=manifest,
        wrapper=wrapper,
        candidate_manifest=candidate_manifest,
        loss_manifest=loss_manifest,
        loss_identity=loss_identity,
        validation_manifest=validation_manifest,
        validation_selection_sha256=validation_selection_sha256,
        selected_validation_result=selected_validation_result,
        retained_names=retained_names,
        model_state_identity=model_state_identity,
        tokenizer_records=tokenizer_records,
        encoder_records=encoder_records,
    )
    encoder = _load_json_object(root / "encoder_config/config.json", name="Smoke encoder config", require_canonical=False)
    required_encoder = {
        "model_type": "modernbert",
        "vocab_size": CONTROLLED_TOKENIZER_SIZE,
        "deterministic_flash_attn": True,
        "reference_compile": False,
        "torch_dtype": "float32",
    }
    for field, expected in required_encoder.items():
        if type(encoder.get(field)) is not type(expected) or encoder.get(field) != expected:
            raise ValueError(f"Smoke encoder config field {field} changed")
    if encoder.get("num_hidden_layers") not in (None, CONTROLLED_ATTENTION_MODULE_COUNT):
        raise ValueError("Smoke encoder layer count changed")
    top_level_expected = {
        "artifact_manifest.json", "determinism_smoke_run.json", "model.safetensors", "wrapper_config.json", "tokenizer", "encoder_config", "candidate_traces", "validation", "loss_traces", *retained_names
    }
    if {entry.name for entry in root.iterdir()} != top_level_expected:
        raise ValueError("Smoke artifact top-level inventory changed")
    final_inventory = _regular_tree_inventory(root)
    if final_inventory != initial_inventory:
        raise RuntimeError("Smoke artifact bytes changed during validation")
    expected_records = {
        "artifact_manifest.json": ArtifactFileRecord("artifact_manifest.json", manifest_path.stat().st_size, manifest_sha256),
        **{item.path: item for item in direct_records},
        **{item.path: item for item in tokenizer_records},
        **{item.path: item for item in encoder_records},
        **{item.path: item for item in checkpoint_records},
        **{item.path: item for item in candidate_records},
        **{item.path: item for item in validation_records},
        **{item.path: item for item in loss_records},
    }
    actual_records = {item.path: item for item in final_inventory}
    if actual_records != expected_records:
        raise ValueError("Smoke artifact complete inventory does not match its manifests")
    identity = DeterminismSmokeArtifactIdentity(
        artifact_manifest_sha256=manifest_sha256,
        determinism_smoke_run_sha256=manifest["determinism_smoke_run"]["sha256"],
        scientific_evidence_sha256=evidence["sha256"],
        launch_ledger_sha256=evidence["launch_ledger"]["sha256"],
        model_file_sha256=manifest["model"]["sha256"],
        model_state_sha256=model_state_identity["sha256"],
        candidate_manifest_sha256=manifest["candidate_trace_manifest"]["sha256"],
        candidate_merged_sha256=candidate_manifest["merged"]["sha256"],
        loss_manifest_sha256=manifest["loss_trace_manifest"]["sha256"],
        loss_trace_sha256=loss_identity["sha256"],
        validation_manifest_sha256=manifest["validation_manifest"]["sha256"],
        validation_selection_sha256=validation_selection_sha256,
    )
    return ValidatedDeterminismSmokeArtifact(
        root=root,
        expectation=expectation,
        identity=identity,
        files=final_inventory,
        scientific_evidence=evidence,
        run_path=run_path,
        model_path=root / "model.safetensors",
    )
