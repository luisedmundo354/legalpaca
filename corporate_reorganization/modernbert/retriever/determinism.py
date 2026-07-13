from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


SMOKE_RUN_KIND = "determinism_smoke"
SMOKE_MODEL_STATE_PROTOCOL = "canonical_tensor_state_sha256_v1"
SMOKE_LOSS_TRACE_PROTOCOL = "rank_microbatch_loss_trace_v1"
SMOKE_EVIDENCE_PROTOCOL = "determinism_smoke_scientific_evidence_v1"
SMOKE_COMPARISON_PROTOCOL = "determinism_smoke_exact_comparison_v1"

SMOKE_EPOCHS = 2
SMOKE_UPDATES_PER_EPOCH = 3
SMOKE_TOTAL_OPTIMIZER_UPDATES = 6
SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH = 19
SMOKE_WINDOW_MICROBATCHES = (8, 8, 3)
SMOKE_GLOBAL_WINDOW_VALID_QUERIES = (128, 128, 38)
SMOKE_WORLD_SIZE = 4
SMOKE_EXPECTED_MODEL_TENSOR_COUNT = 134
SMOKE_TOTAL_MICROBATCH_RECORDS = (
    SMOKE_EPOCHS * SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH * SMOKE_WORLD_SIZE
)
SMOKE_TOTAL_QUERY_LINKS = 588
SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH = (74, 74, 73, 73)

SMOKE_CELL = MappingProxyType(
    {
        "outer_fold": 0,
        "query_view": "structured",
        "sampler": "global_uniform",
        "experiment_seed": 17,
    }
)
SMOKE_SCHEDULE = MappingProxyType(
    {
        "epochs": SMOKE_EPOCHS,
        "updates_per_epoch": SMOKE_UPDATES_PER_EPOCH,
        "total_optimizer_updates": SMOKE_TOTAL_OPTIMIZER_UPDATES,
        "microbatches_per_rank_per_epoch": SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH,
        "window_microbatches": SMOKE_WINDOW_MICROBATCHES,
        "global_window_valid_queries": SMOKE_GLOBAL_WINDOW_VALID_QUERIES,
        "world_size": SMOKE_WORLD_SIZE,
    }
)

_MODEL_STATE_IDENTITY_KEYS = frozenset({"protocol", "tensor_count", "sha256"})
_MICROBATCH_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "epoch",
        "rank",
        "local_microbatch_index",
        "optimizer_window_index",
        "window_microbatch_index",
        "global_step_before",
        "is_window_end",
        "query_ids",
        "candidate_trace_sha256",
        "local_valid_query_count",
        "global_window_valid_query_count",
        "local_loss_sum_float32_bits",
        "scaled_loss_float32_bits",
        "per_query_loss_float32_bits",
    }
)
_LOSS_TRACE_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "record_count",
        "query_link_count",
        "rank_traces",
        "sha256",
    }
)
_RANK_TRACE_KEYS = frozenset(
    {"rank", "record_count", "query_link_count", "sha256"}
)
_CANDIDATE_TRACE_KEYS = frozenset(
    {"manifest_sha256", "merged_sha256", "record_count", "rank_shards"}
)
_CANDIDATE_RANK_KEYS = frozenset(
    {"rank", "record_count", "sha256"}
)
_VALIDATION_SELECTION_KEYS = frozenset({"epochs", "sha256"})
_RELOAD_KEYS = frozenset(
    {
        "validation_sha256",
        "scheduler_state_sha256",
        "client_state_sha256",
        "per_rank_rng_sha256",
    }
)
_FINAL_ARTIFACT_KEYS = frozenset(
    {
        "model_sha256",
        "tokenizer_inventory_sha256",
        "encoder_config_sha256",
        "wrapper_config_sha256",
    }
)
_LAUNCH_LEDGER_KEYS = frozenset({"sha256"})
_EVIDENCE_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "run_kind",
        "cell",
        "schedule",
        "model_states",
        "candidate_traces",
        "loss_traces",
        "validation_selection",
        "reload",
        "final_artifacts",
        "launch_ledger",
    }
)
_EVIDENCE_KEYS = frozenset({*_EVIDENCE_PAYLOAD_KEYS, "sha256"})
_MODEL_STATE_PHASES = ("initial", "last", "selected", "roundtrip")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _smoke_schedule_payload() -> dict[str, Any]:
    return {
        key: list(item) if type(item) is tuple else item
        for key, item in SMOKE_SCHEDULE.items()
    }


def _require_exact_keys(value: object, keys: frozenset[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be one exact dict")
    if set(value) != keys:
        raise ValueError(
            f"{name} fields changed: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _require_exact_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be one exact int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be one non-empty whitespace-trimmed exact string")
    return value


def _frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def validate_gathered_bf16_model_state(
    state_dict: object,
    torch_module: Any,
) -> Mapping[str, Any]:
    """Require the exact rank-zero CPU/BF16 state returned by ZeRO-3."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("Rank zero did not receive a non-empty gathered model state")
    keys = list(state_dict)
    if any(type(key) is not str or not key for key in keys) or len(keys) != len(
        set(keys)
    ):
        raise RuntimeError("Gathered model state has invalid parameter names")
    floating_count = 0
    for key in keys:
        tensor = state_dict[key]
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"Gathered model state {key!r} is not a tensor")
        if tensor.device.type != "cpu":
            raise RuntimeError(
                f"Gathered model state {key!r} was not offloaded to CPU"
            )
        if tensor.is_floating_point():
            floating_count += 1
            if tensor.dtype != torch_module.bfloat16:
                raise RuntimeError(
                    f"Gathered model state {key!r} has dtype={tensor.dtype}; "
                    "expected BF16"
                )
            if not bool(torch_module.isfinite(tensor).all().item()):
                raise FloatingPointError(
                    f"Gathered model state {key!r} is non-finite"
                )
    if floating_count < 1:
        raise RuntimeError("Gathered model state contains no floating-point tensors")
    return state_dict


def canonical_model_state_identity(
    state_dict: object,
    torch_module: Any,
    expected_tensor_count: int = SMOKE_EXPECTED_MODEL_TENSOR_COUNT,
) -> dict[str, Any]:
    """Return a serialization-independent exact identity for one tensor state."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    expected_count = _require_exact_int(
        expected_tensor_count,
        name="expected_tensor_count",
        minimum=1,
    )
    keys = list(state_dict.keys())
    if any(type(key) is not str or not key for key in keys):
        raise TypeError("Every model-state key must be one non-empty exact string")
    if len(keys) != len(set(keys)):
        raise ValueError("Model-state keys contain duplicates")
    if len(keys) != expected_count:
        raise ValueError(
            f"Model state contains {len(keys)} tensors; expected exactly {expected_count}"
        )

    digest = hashlib.sha256()
    _frame(digest, SMOKE_MODEL_STATE_PROTOCOL.encode("utf-8"))
    digest.update(expected_count.to_bytes(8, byteorder="big", signed=False))
    for key in sorted(keys):
        tensor = state_dict[key]
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"Model state {key!r} is not a tensor")
        if tensor.device.type == "meta":
            raise ValueError(f"Model state {key!r} is a meta tensor")
        if tensor.layout != torch_module.strided:
            raise ValueError(f"Model state {key!r} must use strided layout")
        if bool(getattr(tensor, "is_quantized", False)):
            raise ValueError(f"Model state {key!r} must not be quantized")
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch_module.isfinite(tensor).all().item()):
                raise FloatingPointError(f"Model state {key!r} contains a non-finite value")

        exact = tensor.detach().to(device="cpu").contiguous()
        raw = exact.reshape(-1).view(torch_module.uint8).numpy().tobytes(order="C")
        expected_bytes = exact.numel() * exact.element_size()
        if len(raw) != expected_bytes:
            raise RuntimeError(f"Model state {key!r} byte count changed during canonicalization")

        _frame(digest, key.encode("utf-8"))
        _frame(digest, str(exact.dtype).encode("ascii"))
        digest.update(exact.ndim.to_bytes(8, byteorder="big", signed=False))
        for dimension in exact.shape:
            digest.update(int(dimension).to_bytes(8, byteorder="big", signed=False))
        digest.update(expected_bytes.to_bytes(8, byteorder="big", signed=False))
        digest.update(raw)

    return {
        "protocol": SMOKE_MODEL_STATE_PROTOCOL,
        "tensor_count": expected_count,
        "sha256": digest.hexdigest(),
    }


def validate_model_state_identity(value: object, *, name: str) -> dict[str, Any]:
    identity = _require_exact_keys(value, _MODEL_STATE_IDENTITY_KEYS, name=name)
    if identity["protocol"] != SMOKE_MODEL_STATE_PROTOCOL:
        raise ValueError(f"{name}.protocol changed")
    if identity["tensor_count"] != SMOKE_EXPECTED_MODEL_TENSOR_COUNT or type(
        identity["tensor_count"]
    ) is not int:
        raise ValueError(f"{name}.tensor_count must be exactly 134")
    _require_sha256(identity["sha256"], name=f"{name}.sha256")
    return dict(identity)


def encode_float32_scalar_bits(value: object, torch_module: Any) -> str:
    if not torch_module.is_tensor(value):
        raise TypeError("float32 scalar must be a torch tensor")
    if value.dtype != torch_module.float32:
        raise TypeError(f"float32 scalar has dtype={value.dtype}; expected torch.float32")
    if value.ndim != 0:
        raise ValueError(f"float32 scalar must be rank zero; got shape={tuple(value.shape)}")
    scalar = float(value.detach().to(device="cpu").item())
    if not math.isfinite(scalar):
        raise FloatingPointError("float32 scalar must be finite")
    encoded = struct.pack(">f", scalar).hex()
    if len(encoded) != 8:
        raise RuntimeError("Internal float32 encoding did not produce four bytes")
    return encoded


def decode_float32_scalar_bits(value: object) -> float:
    if (
        type(value) is not str
        or len(value) != 8
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("float32 bits must be exactly eight lowercase hexadecimal digits")
    scalar = struct.unpack(">f", bytes.fromhex(value))[0]
    if not math.isfinite(scalar):
        raise FloatingPointError("float32 bits encode a non-finite value")
    return scalar


def _expected_window(local_microbatch_index: int) -> tuple[int, int, bool]:
    offset = 0
    for window_index, count in enumerate(SMOKE_WINDOW_MICROBATCHES):
        if local_microbatch_index < offset + count:
            within = local_microbatch_index - offset
            return window_index, within, within == count - 1
        offset += count
    raise ValueError(f"local_microbatch_index is outside 0..18: {local_microbatch_index}")


def _expected_local_valid_count(rank: int, local_microbatch_index: int) -> int:
    if local_microbatch_index < SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH - 1:
        return 4
    return 2 if rank in (0, 1) else 1


def build_smoke_microbatch_loss_record(
    *,
    epoch: int,
    rank: int,
    local_microbatch_index: int,
    optimizer_window_index: int,
    window_microbatch_index: int,
    global_step_before: int,
    is_window_end: bool,
    query_ids: Sequence[str],
    candidate_trace_sha256: Sequence[str],
    local_valid_query_count: int,
    global_window_valid_query_count: int,
    local_loss_sum: object,
    scaled_loss: object,
    per_query_losses: object,
    torch_module: Any,
) -> dict[str, Any]:
    if not torch_module.is_tensor(per_query_losses):
        raise TypeError("per_query_losses must be one torch tensor")
    if per_query_losses.dtype != torch_module.float32:
        raise TypeError("per_query_losses must have exact torch.float32 dtype")
    if per_query_losses.ndim != 1:
        raise ValueError("per_query_losses must be rank one")
    record = {
        "schema_version": 1,
        "epoch": epoch,
        "rank": rank,
        "local_microbatch_index": local_microbatch_index,
        "optimizer_window_index": optimizer_window_index,
        "window_microbatch_index": window_microbatch_index,
        "global_step_before": global_step_before,
        "is_window_end": is_window_end,
        "query_ids": list(query_ids),
        "candidate_trace_sha256": list(candidate_trace_sha256),
        "local_valid_query_count": local_valid_query_count,
        "global_window_valid_query_count": global_window_valid_query_count,
        "local_loss_sum_float32_bits": encode_float32_scalar_bits(
            local_loss_sum, torch_module
        ),
        "scaled_loss_float32_bits": encode_float32_scalar_bits(scaled_loss, torch_module),
        "per_query_loss_float32_bits": [
            encode_float32_scalar_bits(per_query_losses[index], torch_module)
            for index in range(per_query_losses.shape[0])
        ],
    }
    return validate_smoke_microbatch_loss_record(record)


def validate_smoke_microbatch_loss_record(value: object) -> dict[str, Any]:
    record = _require_exact_keys(value, _MICROBATCH_RECORD_KEYS, name="loss record")
    if record["schema_version"] != 1 or type(record["schema_version"]) is not int:
        raise ValueError("loss record schema_version must be exactly 1")
    epoch = _require_exact_int(record["epoch"], name="loss record epoch", minimum=0)
    if epoch not in range(SMOKE_EPOCHS):
        raise ValueError("loss record epoch must be zero or one")
    rank = _require_exact_int(record["rank"], name="loss record rank", minimum=0)
    if rank not in range(SMOKE_WORLD_SIZE):
        raise ValueError("loss record rank must be zero through three")
    local_index = _require_exact_int(
        record["local_microbatch_index"],
        name="loss record local_microbatch_index",
        minimum=0,
    )
    if local_index >= SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH:
        raise ValueError("loss record local_microbatch_index must be zero through 18")
    expected_window, expected_within, expected_end = _expected_window(local_index)
    if (
        type(record["optimizer_window_index"]) is not int
        or record["optimizer_window_index"] != expected_window
        or type(record["window_microbatch_index"]) is not int
        or record["window_microbatch_index"] != expected_within
        or type(record["is_window_end"]) is not bool
        or record["is_window_end"] is not expected_end
    ):
        raise ValueError("loss record optimizer-window coordinates changed")
    expected_step = epoch * SMOKE_UPDATES_PER_EPOCH + expected_window
    if (
        type(record["global_step_before"]) is not int
        or record["global_step_before"] != expected_step
    ):
        raise ValueError("loss record global_step_before changed")
    expected_global = SMOKE_GLOBAL_WINDOW_VALID_QUERIES[expected_window]
    if (
        type(record["global_window_valid_query_count"]) is not int
        or record["global_window_valid_query_count"] != expected_global
    ):
        raise ValueError("loss record global optimizer-window query count changed")
    expected_local = _expected_local_valid_count(rank, local_index)
    if (
        type(record["local_valid_query_count"]) is not int
        or record["local_valid_query_count"] != expected_local
    ):
        raise ValueError("loss record rank-local valid-query count changed")

    query_ids = record["query_ids"]
    trace_hashes = record["candidate_trace_sha256"]
    per_query_bits = record["per_query_loss_float32_bits"]
    for name, sequence in (
        ("query_ids", query_ids),
        ("candidate_trace_sha256", trace_hashes),
        ("per_query_loss_float32_bits", per_query_bits),
    ):
        if type(sequence) is not list or len(sequence) != expected_local:
            raise ValueError(f"loss record {name} does not match local query count")
    if any(_require_string(query_id, name="loss record query_id") != query_id for query_id in query_ids):
        raise ValueError("loss record query IDs changed")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("loss record contains duplicate query IDs")
    for index, digest in enumerate(trace_hashes):
        _require_sha256(digest, name=f"loss record candidate_trace_sha256[{index}]")
    decode_float32_scalar_bits(record["local_loss_sum_float32_bits"])
    decode_float32_scalar_bits(record["scaled_loss_float32_bits"])
    for encoded in per_query_bits:
        decode_float32_scalar_bits(encoded)
    return {
        key: list(item) if type(item) is list else item
        for key, item in record.items()
    }


def build_smoke_loss_trace_identity(per_rank_records: object) -> dict[str, Any]:
    if type(per_rank_records) not in (list, tuple) or len(per_rank_records) != SMOKE_WORLD_SIZE:
        raise ValueError("per_rank_records must contain exactly four rank sequences")
    normalized_by_rank: list[list[dict[str, Any]]] = []
    rank_traces: list[dict[str, Any]] = []
    seen_query_keys: set[tuple[int, str]] = set()
    seen_candidate_trace_hashes: set[str] = set()

    for rank, supplied_records in enumerate(per_rank_records):
        if type(supplied_records) not in (list, tuple):
            raise TypeError(f"per_rank_records[{rank}] must be one list or tuple")
        if len(supplied_records) != SMOKE_EPOCHS * SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH:
            raise ValueError(f"rank {rank} must contain exactly 38 microbatch records")
        normalized: list[dict[str, Any]] = []
        query_link_count = 0
        for position, supplied in enumerate(supplied_records):
            record = validate_smoke_microbatch_loss_record(supplied)
            expected_epoch, expected_index = divmod(
                position, SMOKE_MICROBATCHES_PER_RANK_PER_EPOCH
            )
            if (
                record["rank"] != rank
                or record["epoch"] != expected_epoch
                or record["local_microbatch_index"] != expected_index
            ):
                raise ValueError(f"rank {rank} loss records are not in canonical epoch/microbatch order")
            for query_id, trace_sha256 in zip(
                record["query_ids"], record["candidate_trace_sha256"]
            ):
                key = (record["epoch"], query_id)
                if key in seen_query_keys:
                    raise ValueError(f"Duplicate smoke query link {key!r}")
                seen_query_keys.add(key)
                if trace_sha256 in seen_candidate_trace_hashes:
                    raise ValueError(
                        f"Duplicate smoke candidate-trace SHA-256 {trace_sha256}"
                    )
                seen_candidate_trace_hashes.add(trace_sha256)
            query_link_count += record["local_valid_query_count"]
            normalized.append(record)
        expected_links = SMOKE_EPOCHS * SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH[rank]
        if query_link_count != expected_links:
            raise RuntimeError(
                f"rank {rank} has {query_link_count} query links; expected {expected_links}"
            )
        normalized_by_rank.append(normalized)
        rank_traces.append(
            {
                "rank": rank,
                "record_count": len(normalized),
                "query_link_count": query_link_count,
                "sha256": _sha256_canonical(normalized),
            }
        )

    if len(seen_query_keys) != SMOKE_TOTAL_QUERY_LINKS:
        raise RuntimeError(
            f"Smoke loss traces link {len(seen_query_keys)} epoch/query pairs; "
            f"expected {SMOKE_TOTAL_QUERY_LINKS}"
        )
    if len(seen_candidate_trace_hashes) != SMOKE_TOTAL_QUERY_LINKS:
        raise RuntimeError("Smoke loss traces do not link 588 unique candidate traces")
    identity_payload = {
        "schema_version": 1,
        "protocol": SMOKE_LOSS_TRACE_PROTOCOL,
        "record_count": sum(record["record_count"] for record in rank_traces),
        "query_link_count": len(seen_query_keys),
        "rank_traces": rank_traces,
    }
    if identity_payload["record_count"] != SMOKE_TOTAL_MICROBATCH_RECORDS:
        raise RuntimeError("Smoke loss trace record total changed")
    return {**identity_payload, "sha256": _sha256_canonical(identity_payload)}


def validate_smoke_loss_trace_identity(value: object) -> dict[str, Any]:
    identity = _require_exact_keys(value, _LOSS_TRACE_IDENTITY_KEYS, name="loss trace identity")
    if (
        identity["schema_version"] != 1
        or type(identity["schema_version"]) is not int
        or identity["protocol"] != SMOKE_LOSS_TRACE_PROTOCOL
        or identity["record_count"] != SMOKE_TOTAL_MICROBATCH_RECORDS
        or type(identity["record_count"]) is not int
        or identity["query_link_count"] != SMOKE_TOTAL_QUERY_LINKS
        or type(identity["query_link_count"]) is not int
    ):
        raise ValueError("Smoke loss trace identity dimensions or protocol changed")
    ranks = identity["rank_traces"]
    if type(ranks) is not list or len(ranks) != SMOKE_WORLD_SIZE:
        raise ValueError("Smoke loss trace identity must contain four ranks")
    normalized_ranks: list[dict[str, Any]] = []
    for rank, value in enumerate(ranks):
        record = _require_exact_keys(value, _RANK_TRACE_KEYS, name=f"loss rank {rank}")
        expected_links = SMOKE_EPOCHS * SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH[rank]
        if (
            record["rank"] != rank
            or type(record["rank"]) is not int
            or record["record_count"] != 38
            or type(record["record_count"]) is not int
            or record["query_link_count"] != expected_links
            or type(record["query_link_count"]) is not int
        ):
            raise ValueError(f"Smoke loss rank {rank} dimensions changed")
        _require_sha256(record["sha256"], name=f"loss rank {rank}.sha256")
        normalized_ranks.append(dict(record))
    payload = {
        "schema_version": 1,
        "protocol": SMOKE_LOSS_TRACE_PROTOCOL,
        "record_count": SMOKE_TOTAL_MICROBATCH_RECORDS,
        "query_link_count": SMOKE_TOTAL_QUERY_LINKS,
        "rank_traces": normalized_ranks,
    }
    if identity["sha256"] != _sha256_canonical(payload):
        raise ValueError("Smoke loss trace identity SHA-256 changed")
    return {**payload, "sha256": identity["sha256"]}


def _validate_candidate_trace_identity(value: object) -> dict[str, Any]:
    candidate = _require_exact_keys(value, _CANDIDATE_TRACE_KEYS, name="candidate traces")
    _require_sha256(candidate["manifest_sha256"], name="candidate manifest_sha256")
    _require_sha256(candidate["merged_sha256"], name="candidate merged_sha256")
    if candidate["record_count"] != SMOKE_TOTAL_QUERY_LINKS or type(candidate["record_count"]) is not int:
        raise ValueError("Candidate traces must contain exactly 588 records")
    shards = candidate["rank_shards"]
    if type(shards) is not list or len(shards) != SMOKE_WORLD_SIZE:
        raise ValueError("Candidate traces must contain exactly four rank shards")
    normalized_shards: list[dict[str, Any]] = []
    for rank, value in enumerate(shards):
        shard = _require_exact_keys(value, _CANDIDATE_RANK_KEYS, name=f"candidate rank {rank}")
        expected_count = SMOKE_EPOCHS * SMOKE_QUERY_LINKS_PER_RANK_PER_EPOCH[rank]
        if (
            shard["rank"] != rank
            or type(shard["rank"]) is not int
            or shard["record_count"] != expected_count
            or type(shard["record_count"]) is not int
        ):
            raise ValueError(f"Candidate rank {rank} dimensions changed")
        _require_sha256(shard["sha256"], name=f"candidate rank {rank}.sha256")
        normalized_shards.append(dict(shard))
    return {
        "manifest_sha256": candidate["manifest_sha256"],
        "merged_sha256": candidate["merged_sha256"],
        "record_count": candidate["record_count"],
        "rank_shards": normalized_shards,
    }


def _validate_validation_selection(value: object) -> dict[str, Any]:
    validation = _require_exact_keys(
        value, _VALIDATION_SELECTION_KEYS, name="validation selection"
    )
    if validation["epochs"] != SMOKE_EPOCHS or type(validation["epochs"]) is not int:
        raise ValueError("Validation selection must contain exactly two epochs")
    _require_sha256(validation["sha256"], name="validation selection sha256")
    return dict(validation)


def _validate_reload(value: object) -> dict[str, Any]:
    reload_identity = _require_exact_keys(value, _RELOAD_KEYS, name="reload identity")
    for field in ("validation_sha256", "scheduler_state_sha256", "client_state_sha256"):
        _require_sha256(reload_identity[field], name=f"reload {field}")
    rng = reload_identity["per_rank_rng_sha256"]
    if type(rng) is not list or len(rng) != SMOKE_WORLD_SIZE:
        raise ValueError("Reload identity must contain four rank RNG hashes")
    for rank, digest in enumerate(rng):
        _require_sha256(digest, name=f"reload per_rank_rng_sha256[{rank}]")
    return {**reload_identity, "per_rank_rng_sha256": list(rng)}


def _validate_final_artifacts(value: object) -> dict[str, Any]:
    artifacts = _require_exact_keys(value, _FINAL_ARTIFACT_KEYS, name="final artifacts")
    for field, digest in artifacts.items():
        _require_sha256(digest, name=f"final artifacts {field}")
    return dict(artifacts)


def _validate_launch_ledger(value: object) -> dict[str, Any]:
    ledger = _require_exact_keys(value, _LAUNCH_LEDGER_KEYS, name="launch ledger")
    _require_sha256(ledger["sha256"], name="launch ledger sha256")
    return dict(ledger)


def _validate_fixed_cell(value: object) -> dict[str, Any]:
    if type(value) is not dict or value != dict(SMOKE_CELL):
        raise ValueError("Smoke evidence cell changed from the fixed fold/view/sampler/seed")
    if any(type(value[key]) is not type(SMOKE_CELL[key]) for key in SMOKE_CELL):
        raise TypeError("Smoke evidence cell contains a type-confused value")
    return dict(value)


def _validate_fixed_schedule(value: object) -> dict[str, Any]:
    expected = _smoke_schedule_payload()
    if type(value) is not dict or value != expected:
        raise ValueError("Smoke evidence schedule changed from the exact 2-epoch/6-update contract")
    if any(type(value[key]) is not type(expected[key]) for key in expected) or any(
        type(actual) is not type(required)
        for key in ("window_microbatches", "global_window_valid_queries")
        for actual, required in zip(value[key], expected[key])
    ):
        raise TypeError("Smoke evidence schedule contains a type-confused value")
    return {key: list(item) if type(item) is list else item for key, item in value.items()}


def build_smoke_scientific_evidence(
    *,
    initial_model_state: object,
    last_model_state: object,
    selected_model_state: object,
    roundtrip_model_state: object,
    candidate_traces: object,
    loss_traces: object,
    validation_selection: object,
    reload: object,
    final_artifacts: object,
    launch_ledger: object,
) -> dict[str, Any]:
    model_states = {
        "initial": validate_model_state_identity(initial_model_state, name="initial model state"),
        "last": validate_model_state_identity(last_model_state, name="last model state"),
        "selected": validate_model_state_identity(
            selected_model_state, name="selected model state"
        ),
        "roundtrip": validate_model_state_identity(
            roundtrip_model_state, name="roundtrip model state"
        ),
    }
    if model_states["selected"] != model_states["roundtrip"]:
        raise ValueError("Selected and round-trip model-state identities differ")
    payload = {
        "schema_version": 1,
        "protocol": SMOKE_EVIDENCE_PROTOCOL,
        "run_kind": SMOKE_RUN_KIND,
        "cell": dict(SMOKE_CELL),
        "schedule": _smoke_schedule_payload(),
        "model_states": model_states,
        "candidate_traces": _validate_candidate_trace_identity(candidate_traces),
        "loss_traces": validate_smoke_loss_trace_identity(loss_traces),
        "validation_selection": _validate_validation_selection(validation_selection),
        "reload": _validate_reload(reload),
        "final_artifacts": _validate_final_artifacts(final_artifacts),
        "launch_ledger": _validate_launch_ledger(launch_ledger),
    }
    return {**payload, "sha256": _sha256_canonical(payload)}


def validate_smoke_scientific_evidence(value: object) -> dict[str, Any]:
    evidence = _require_exact_keys(value, _EVIDENCE_KEYS, name="smoke scientific evidence")
    if (
        evidence["schema_version"] != 1
        or type(evidence["schema_version"]) is not int
        or evidence["protocol"] != SMOKE_EVIDENCE_PROTOCOL
        or evidence["run_kind"] != SMOKE_RUN_KIND
    ):
        raise ValueError("Smoke scientific evidence identity changed")
    model_states = evidence["model_states"]
    if type(model_states) is not dict or set(model_states) != set(_MODEL_STATE_PHASES):
        raise ValueError("Smoke model-state phases changed")
    normalized_states = {
        phase: validate_model_state_identity(model_states[phase], name=f"{phase} model state")
        for phase in _MODEL_STATE_PHASES
    }
    if normalized_states["selected"] != normalized_states["roundtrip"]:
        raise ValueError("Selected and round-trip model-state identities differ")
    payload = {
        "schema_version": 1,
        "protocol": SMOKE_EVIDENCE_PROTOCOL,
        "run_kind": SMOKE_RUN_KIND,
        "cell": _validate_fixed_cell(evidence["cell"]),
        "schedule": _validate_fixed_schedule(evidence["schedule"]),
        "model_states": normalized_states,
        "candidate_traces": _validate_candidate_trace_identity(evidence["candidate_traces"]),
        "loss_traces": validate_smoke_loss_trace_identity(evidence["loss_traces"]),
        "validation_selection": _validate_validation_selection(
            evidence["validation_selection"]
        ),
        "reload": _validate_reload(evidence["reload"]),
        "final_artifacts": _validate_final_artifacts(evidence["final_artifacts"]),
        "launch_ledger": _validate_launch_ledger(evidence["launch_ledger"]),
    }
    expected_sha256 = _sha256_canonical(payload)
    if evidence["sha256"] != expected_sha256:
        raise ValueError("Smoke scientific evidence SHA-256 changed")
    return {**payload, "sha256": expected_sha256}


def compare_smoke_scientific_evidence(first: object, second: object) -> dict[str, Any]:
    first_evidence = validate_smoke_scientific_evidence(first)
    second_evidence = validate_smoke_scientific_evidence(second)
    if first_evidence != second_evidence:
        changed = [
            key
            for key in sorted(_EVIDENCE_PAYLOAD_KEYS)
            if first_evidence[key] != second_evidence[key]
        ]
        raise RuntimeError(f"Determinism smoke scientific evidence differs: {changed}")
    identity_sha256 = first_evidence["sha256"]
    receipt_payload = {
        "schema_version": 1,
        "protocol": SMOKE_COMPARISON_PROTOCOL,
        "run_kind": SMOKE_RUN_KIND,
        "scientific_identity_sha256": identity_sha256,
        "replicas": 2,
        "exact_match": True,
    }
    return {**receipt_payload, "sha256": _sha256_canonical(receipt_payload)}
