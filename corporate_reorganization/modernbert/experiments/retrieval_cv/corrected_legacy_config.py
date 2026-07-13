"""Strict configuration contract for the corrected legacy-style diagnostic.

The corrected diagnostic is a single, approved scientific design rather than a
general-purpose configuration surface.  This module consequently rejects any
schema or value drift, reads every input exactly once without following
symbolic links, and can bind the role memberships to the exact raw JSONL lines
in the corrected-v2 staged dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


CORRECTED_LEGACY_CONFIG_SHA256 = (
    "ab1f5c294a0d26f3949925d54cc85c981ebf682bcfc8de26abc8a19487ad515b"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_ROLES = ("train", "validation", "test")
_EXPECTED_CASE_IDS = (
    "36", "37", "38", "40", "41", "42", "45", "46", "47", "48", "49",
    "57", "58", "59", "60", "62", "63", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80",
    "83", "85", "86", "87", "91", "92", "94", "96", "97",
)


_EXPECTED_CONFIG: dict[str, object] = {
    "artifact": {
        "artifact_type": "corrected_legacy_diagnostic_retriever",
        "schema_version": 1,
        "validator_version": "corrected_legacy_diagnostic_artifact_v1",
    },
    "batching": {
        "batch_order_algorithm": "sha256_corrected_legacy_query_order_v1",
        "global_microbatch_queries": 16,
        "gradient_accumulation_steps": 8,
        "optimizer_window_microbatches": [8, 8, 8, 3],
        "optimizer_window_valid_queries": [128, 128, 128, 34],
        "per_device_query_batch": 4,
        "prepared_batches_per_rank": 27,
        "sentinel_rows": 14,
        "tail_rebalance": "two_global_groups_of_9_round_robin_across_four_ranks_v1",
        "world_size": 4,
    },
    "candidate_sampling": {
        "candidate_occurrences_per_query": 64,
        "case_gold_exclusion": "union_of_all_training_query_golds_in_current_case",
        "cross_case_label": "Background Facts",
        "cross_case_negative_occurrences": 4,
        "cross_case_pool": "passages_from_other_training_role_cases_only",
        "max_selected_positives": 4,
        "multiplicity_loss": (
            "unique_global_passage_logit_plus_log_global_occurrence_multiplicity_"
            "in_numerator_and_denominator_v1"
        ),
        "replacement": (
            "whole_requested_stratum_independent_digest_draws_only_when_pool_is_"
            "smaller_than_requested_quota"
        ),
        "replacement_query_count": 103,
        "same_case_base_negative_occurrences": 56,
        "same_case_negative_occurrences": "56 + (4 - selected_positive_count)",
        "selection_algorithm": "sha256_corrected_legacy_occurrences_v1",
        "selection_key_excludes": ["query_view"],
        "trace_schema_version": 1,
    },
    "dataset": {
        "manifest_path": "dataset_manifest.json",
        "manifest_sha256": (
            "cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be"
        ),
        "schema_version": 2,
    },
    "diagnostic_id": "arr_corrected_legacy_diagnostic_v1",
    "evaluation": {
        "canonical_regimes": [
            "same_case_legacy",
            "same_case_full",
            "fold_global",
            "fold_global_context_excluded",
        ],
        "context_excluded_is_robustness_only": True,
        "paper_comparable_regimes": [
            "same_case_legacy",
            "same_case_full",
            "fold_global",
        ],
        "ranking": "score_desc_passage_id_asc_v1",
        "scoring": "cpu_float32_v1",
    },
    "label": "intended-configuration corrected legacy-style diagnostic",
    "membership": {
        "test": {
            "case_count": 4,
            "membership_path": "corrected_legacy_membership/test_cases.txt",
            "membership_sha256": (
                "ed59e5830650a1947b86e986db0820ca888589dc54f132d6d0620efa5e3f37c7"
            ),
            "passage_count": 581,
            "passage_subset_sha256": (
                "63bda2cc1cae1d922f68c259c2ca77509a7f4d45f108934481f1f81b28b114db"
            ),
            "query_count": 40,
            "query_subset_sha256": (
                "4a8361f39fe96a218da886fe1028e1866063102ac70f4adb7af2e6506b0c4969"
            ),
        },
        "train": {
            "case_count": 34,
            "membership_path": "corrected_legacy_membership/train_cases.txt",
            "membership_sha256": (
                "5ff8e5c7bfb53bf3449ab1fe606e5d890feb2cae44948416902a40e933e438ae"
            ),
            "passage_count": 4307,
            "passage_subset_sha256": (
                "178f02f769cd50410e9d98a8b932ce936b48ac970ae558feecb4277b08574d0e"
            ),
            "query_count": 418,
            "query_subset_sha256": (
                "4ec11453d4683e7a55c40d7651f0d46d7a9ec8f91fc71723d6dc9d9a4df6d361"
            ),
        },
        "validation": {
            "case_count": 4,
            "membership_path": "corrected_legacy_membership/validation_cases.txt",
            "membership_sha256": (
                "f2f9153f85eaae37c11c65674b55f94990a4b1c0987f508b735983784136a8cf"
            ),
            "passage_count": 398,
            "passage_subset_sha256": (
                "2dd5e0acac571fd46866420acd6ee9c552932a44c1168b865cafed1eaee1ea84"
            ),
            "query_count": 32,
            "query_subset_sha256": (
                "a3229950f01f42cdcd1d58e15092b1c2624377290f385b2d1ee50a8e4bdcf6cc"
            ),
        },
    },
    "query_views": ["flat_masked", "structured"],
    "reporting_boundary": {
        "controlled_aggregate_inclusion": False,
        "exact_march_replication": False,
        "five_fold_evidence": False,
        "historical_metrics_are_reference_only": True,
    },
    "schema_version": 1,
    "setting_explanation": (
        "Two ModernBERT jobs compare flat-masked and structured query views using "
        "corrected-v2 records and the original March train/validation/test case "
        "memberships. Training retains the intended local-heavy 64-occurrence "
        "candidate design while correcting case leakage, incomplete batches, "
        "duplicate-weight loss, parsing, provenance, and final-state export. These "
        "results are a corrected legacy-style diagnostic, not an exact March "
        "replication and not part of the controlled five-fold causal comparison."
    ),
    "training": {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "epochs": 20,
        "final_model": "active_engine_epoch_20_bf16_safetensors_strict_round_trip_v1",
        "learning_rate": 1e-5,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "max_passage_tokens": 500,
        "max_query_tokens": 4096,
        "model_selection": "none_final_epoch_only",
        "optimizer": "adamw_torch",
        "seed": 17,
        "temperature": 0.07,
        "total_optimizer_updates": 80,
        "updates_per_epoch": 4,
        "validation_epochs": 20,
        "validation_steps": [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
            44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        ],
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    },
}


@dataclass(frozen=True)
class CorrectedLegacyMemberships:
    """Case memberships read from the three hash-bound source files."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def for_role(self, role: str) -> tuple[str, ...]:
        if role not in _ROLES:
            raise ValueError(f"Unknown corrected-legacy role: {role!r}")
        return getattr(self, role)


@dataclass(frozen=True)
class LoadedCorrectedLegacyConfig:
    """One validated design plus the exact memberships it names."""

    value: dict[str, Any]
    config_sha256: str
    memberships: CorrectedLegacyMemberships


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _reject_symlink_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Symbolic links are forbidden: {current}")


def _read_regular_file_once(path: Path) -> bytes:
    path = Path(path)
    _reject_symlink_chain(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> object:
    raise ValueError(f"Non-finite JSON number is forbidden: {token}")


def _validate_json_value(value: object, *, name: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} contains a non-string object key")
            _validate_json_value(item, name=f"{name}.{key}")
        return
    raise TypeError(f"{name} contains unsupported type {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json_value(value, name="JSON value")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_canonical_json_object(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_regular_file_once(path)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"Canonical JSON is not UTF-8: {path}") from error
    if type(value) is not dict:
        raise TypeError(f"Canonical JSON must contain one exact object: {path}")
    if payload != _canonical_json_bytes(value):
        raise ValueError(f"JSON does not use canonical deterministic bytes: {path}")
    return value, _sha256(payload)


def _assert_exact(actual: object, expected: object, *, name: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(
            f"{name} must be exact {type(expected).__name__}, "
            f"not {type(actual).__name__}"
        )
    if type(expected) is dict:
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise ValueError(
                f"{name} keys mismatch: missing={sorted(expected_keys - actual_keys)}, "
                f"unknown={sorted(actual_keys - expected_keys)}"
            )
        for key in expected:
            _assert_exact(actual[key], expected[key], name=f"{name}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(
                f"{name} length changed: expected={len(expected)}, found={len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_exact(actual_item, expected_item, name=f"{name}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{name} changed: expected={expected!r}, found={actual!r}")


def validate_corrected_legacy_config(value: object) -> dict[str, Any]:
    """Validate the exact, approved corrected-legacy scientific design."""

    _assert_exact(value, _EXPECTED_CONFIG, name="corrected legacy config")
    assert type(value) is dict

    sampling = value["candidate_sampling"]
    occurrence_total = sampling["candidate_occurrences_per_query"]
    for selected_positive_count in range(1, sampling["max_selected_positives"] + 1):
        computed = (
            selected_positive_count
            + sampling["same_case_base_negative_occurrences"]
            + (sampling["max_selected_positives"] - selected_positive_count)
            + sampling["cross_case_negative_occurrences"]
        )
        if computed != occurrence_total:
            raise ValueError("Candidate occurrence formula no longer totals 64")

    batching = value["batching"]
    training = value["training"]
    train_queries = value["membership"]["train"]["query_count"]
    if (
        batching["per_device_query_batch"] * batching["world_size"]
        != batching["global_microbatch_queries"]
    ):
        raise ValueError("Global microbatch size is inconsistent")
    prepared_slots = (
        batching["prepared_batches_per_rank"]
        * batching["world_size"]
        * batching["per_device_query_batch"]
    )
    if prepared_slots != train_queries + batching["sentinel_rows"]:
        raise ValueError("Prepared batch slots do not cover 418 queries plus sentinels")
    if sum(batching["optimizer_window_microbatches"]) != batching["prepared_batches_per_rank"]:
        raise ValueError("Optimizer-window microbatches do not cover one epoch")
    if sum(batching["optimizer_window_valid_queries"]) != train_queries:
        raise ValueError("Optimizer-window query counts do not cover one epoch")
    if training["updates_per_epoch"] != len(batching["optimizer_window_microbatches"]):
        raise ValueError("Updates per epoch disagree with optimizer windows")
    if training["total_optimizer_updates"] != training["epochs"] * training["updates_per_epoch"]:
        raise ValueError("Total optimizer updates disagree with the epoch schedule")
    if training["validation_steps"] != [
        training["updates_per_epoch"] * epoch
        for epoch in range(1, training["validation_epochs"] + 1)
    ]:
        raise ValueError("Validation steps are not the end of every complete epoch")

    membership = value["membership"]
    if sum(membership[role]["case_count"] for role in _ROLES) != 42:
        raise ValueError("Membership case counts do not total 42")
    if sum(membership[role]["query_count"] for role in _ROLES) != 490:
        raise ValueError("Membership query counts do not total 490")
    if sum(membership[role]["passage_count"] for role in _ROLES) != 5_286:
        raise ValueError("Membership passage counts do not total 5,286")

    for role in _ROLES:
        role_record = membership[role]
        for key in (
            "membership_sha256",
            "query_subset_sha256",
            "passage_subset_sha256",
        ):
            _require_sha256(role_record[key], name=f"membership.{role}.{key}")
    _require_sha256(value["dataset"]["manifest_sha256"], name="dataset.manifest_sha256")
    return value


def _load_membership_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int,
    role: str,
) -> tuple[str, ...]:
    payload = _read_regular_file_once(path)
    actual_sha256 = _sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{role} membership hash mismatch: "
            f"expected={expected_sha256}, found={actual_sha256}"
        )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} membership must use ASCII bytes") from error
    case_ids = tuple(text.splitlines())
    canonical = "".join(f"{case_id}\n" for case_id in case_ids).encode("ascii")
    if payload != canonical or not case_ids:
        raise ValueError(f"{role} membership must contain one canonical case ID per line")
    if any(_CASE_ID_RE.fullmatch(case_id) is None for case_id in case_ids):
        raise ValueError(f"{role} membership contains a noncanonical case ID")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{role} membership contains duplicate case IDs")
    if tuple(sorted(case_ids, key=int)) != case_ids:
        raise ValueError(f"{role} membership case IDs must be numerically sorted")
    if len(case_ids) != expected_count:
        raise ValueError(
            f"{role} membership count changed: "
            f"expected={expected_count}, found={len(case_ids)}"
        )
    return case_ids


def load_corrected_legacy_memberships(
    value: object,
    *,
    config_dir: Path,
) -> CorrectedLegacyMemberships:
    """Read and cross-check the three membership files named by the design."""

    config = validate_corrected_legacy_config(value)
    config_dir = Path(config_dir)
    _reject_symlink_chain(config_dir)
    loaded: dict[str, tuple[str, ...]] = {}
    for role in _ROLES:
        record = config["membership"][role]
        relative_path = record["membership_path"]
        path = config_dir / relative_path
        if path.parent == config_dir or config_dir not in path.parents:
            raise ValueError(f"{role} membership path escapes the config directory")
        loaded[role] = _load_membership_file(
            path,
            expected_sha256=record["membership_sha256"],
            expected_count=record["case_count"],
            role=role,
        )

    role_sets = {role: set(case_ids) for role, case_ids in loaded.items()}
    for index, left in enumerate(_ROLES):
        for right in _ROLES[index + 1 :]:
            overlap = role_sets[left] & role_sets[right]
            if overlap:
                raise ValueError(
                    f"Corrected-legacy memberships overlap for {left}/{right}: "
                    f"{sorted(overlap, key=int)}"
                )
    union = set().union(*role_sets.values())
    if union != set(_EXPECTED_CASE_IDS):
        raise ValueError(
            "Corrected-legacy memberships do not form the exact 42-case dataset union"
        )
    return CorrectedLegacyMemberships(
        train=loaded["train"],
        validation=loaded["validation"],
        test=loaded["test"],
    )


def _parse_jsonl_raw_lines(
    payload: bytes,
    *,
    name: str,
) -> list[tuple[dict[str, Any], bytes]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{name} must be a non-empty newline-terminated JSONL file")
    rows: list[tuple[dict[str, Any], bytes]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            raise ValueError(f"{name} has an invalid raw line at {line_number}")
        try:
            record = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} has invalid JSON at line {line_number}") from error
        if type(record) is not dict:
            raise TypeError(f"{name} line {line_number} must be one exact JSON object")
        rows.append((record, line))
    return rows


def validate_corrected_legacy_dataset(
    loaded: LoadedCorrectedLegacyConfig,
    *,
    dataset_dir: Path,
) -> None:
    """Bind role counts and digests to exact raw corrected-v2 JSONL lines.

    Subset digests concatenate the selected source lines byte-for-byte in their
    original file order.  No parsing/reserialization participates in hashing.
    """

    if type(loaded) is not LoadedCorrectedLegacyConfig:
        raise TypeError("loaded must be an exact LoadedCorrectedLegacyConfig")
    dataset_dir = Path(dataset_dir)
    _reject_symlink_chain(dataset_dir)
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    dataset_record = loaded.value["dataset"]
    manifest_path = dataset_dir / dataset_record["manifest_path"]
    manifest, manifest_sha256 = _load_canonical_json_object(manifest_path)
    if manifest_sha256 != dataset_record["manifest_sha256"]:
        raise ValueError(
            "Dataset manifest hash mismatch: "
            f"expected={dataset_record['manifest_sha256']}, found={manifest_sha256}"
        )
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
        raise ValueError("Dataset manifest must have exact schema_version=2")
    if manifest.get("counts") != {
        "cases": 42,
        "nodes": 800,
        "passages": 5_286,
        "queries": 490,
        "relations": 644,
        "roots": 44,
    }:
        raise ValueError("Dataset manifest counts changed")
    output_files = manifest.get("output_files")
    if type(output_files) is not dict:
        raise TypeError("Dataset manifest output_files must be an exact object")

    rows_by_kind: dict[str, list[tuple[dict[str, Any], bytes]]] = {}
    for kind, relative_path in {
        "case": "cases.jsonl",
        "passage": "corpus.jsonl",
        "query": "queries/all.jsonl",
    }.items():
        record = output_files.get(relative_path)
        if type(record) is not dict:
            raise ValueError(f"Dataset manifest lacks output identity for {relative_path}")
        path = dataset_dir / relative_path
        payload = _read_regular_file_once(path)
        if _sha256(payload) != record.get("sha256") or len(payload) != record.get("bytes"):
            raise ValueError(f"Dataset output identity mismatch for {relative_path}")
        rows = _parse_jsonl_raw_lines(payload, name=relative_path)
        if len(rows) != record.get("records"):
            raise ValueError(f"Dataset output record count mismatch for {relative_path}")
        rows_by_kind[kind] = rows

    all_membership_cases = set(_EXPECTED_CASE_IDS)
    case_ids: list[str] = []
    for record, _ in rows_by_kind["case"]:
        case_id = record.get("doc_id")
        if type(case_id) is not str or _CASE_ID_RE.fullmatch(case_id) is None:
            raise ValueError("cases.jsonl contains a noncanonical doc_id")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != all_membership_cases:
        raise ValueError("cases.jsonl does not equal the complete 42-case membership union")

    for kind in ("query", "passage"):
        seen_identity: set[str] = set()
        rows_by_role: dict[str, list[bytes]] = {role: [] for role in _ROLES}
        field = "query_id" if kind == "query" else "passage_id"
        for record, raw_line in rows_by_kind[kind]:
            case_id = record.get("doc_id")
            identity = record.get(field)
            if type(case_id) is not str or case_id not in all_membership_cases:
                raise ValueError(f"{kind} row references an unknown case")
            if type(identity) is not str or not identity:
                raise TypeError(f"{kind} row has an invalid {field}")
            if identity in seen_identity:
                raise ValueError(f"Duplicate {field}: {identity}")
            seen_identity.add(identity)
            matching_roles = [
                role
                for role in _ROLES
                if case_id in loaded.memberships.for_role(role)
            ]
            if len(matching_roles) != 1:
                raise ValueError(f"{kind} row does not map to exactly one role")
            rows_by_role[matching_roles[0]].append(raw_line)

        for role in _ROLES:
            expected = loaded.value["membership"][role]
            actual_count = len(rows_by_role[role])
            expected_count = expected[f"{kind}_count"]
            if actual_count != expected_count:
                raise ValueError(
                    f"{role} {kind} count mismatch: "
                    f"expected={expected_count}, found={actual_count}"
                )
            actual_subset_sha256 = _sha256(b"".join(rows_by_role[role]))
            expected_subset_sha256 = expected[f"{kind}_subset_sha256"]
            if actual_subset_sha256 != expected_subset_sha256:
                raise ValueError(
                    f"{role} {kind} raw-line subset hash mismatch: "
                    f"expected={expected_subset_sha256}, found={actual_subset_sha256}"
                )


def load_corrected_legacy_config(
    path: Path,
    *,
    expected_sha256: str | None = CORRECTED_LEGACY_CONFIG_SHA256,
    dataset_dir: Path | None = None,
) -> LoadedCorrectedLegacyConfig:
    """Load the approved design, memberships, and optionally its staged data."""

    if expected_sha256 is not None:
        _require_sha256(expected_sha256, name="expected_sha256")
    path = Path(path)
    value, config_sha256 = _load_canonical_json_object(path)
    if expected_sha256 is not None and config_sha256 != expected_sha256:
        raise ValueError(
            "Corrected-legacy config hash mismatch: "
            f"expected={expected_sha256}, found={config_sha256}"
        )
    validate_corrected_legacy_config(value)
    memberships = load_corrected_legacy_memberships(value, config_dir=path.parent)
    loaded = LoadedCorrectedLegacyConfig(
        value=value,
        config_sha256=config_sha256,
        memberships=memberships,
    )
    if dataset_dir is not None:
        validate_corrected_legacy_dataset(loaded, dataset_dir=dataset_dir)
    return loaded
