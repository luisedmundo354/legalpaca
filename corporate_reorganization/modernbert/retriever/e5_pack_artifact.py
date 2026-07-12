from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .data import QueryExample
from .provenance import EXPECTED_DATASET_MANIFEST_SHA256
from .query_packing import (
    FOCUS_PRESERVING_PACK_PROTOCOL,
    PackedQuery,
    pack_focus_preserving_query,
    packed_query_inventory_sha256,
)


E5_PACK_ARTIFACT_SCHEMA_VERSION = 1
E5_PACK_ARTIFACT_PROTOCOL = "frozen_e5_flat_plain_focus_pack_v1"
EXPECTED_QUERY_SOURCE_SHA256 = "bcc6e7573009329f50aaa42a483981e9e30c6e3060984dd840f1c0d7e6f66279"
EXPECTED_E5_SNAPSHOT_MANIFEST_SHA256 = (
    "7629cf8c8bf60569d72f653d21a4c47a8fa806d8fd907db05c65a3288b24b635"
)
EXPECTED_E5_SNAPSHOT_TREE_SHA256 = (
    "1181a9758ea858d6679df0e04f6ac67b26dab90e91f63e76238c2eecec1c1a61"
)
EXPECTED_E5_MODEL_ID = "intfloat/e5-base-v2"
EXPECTED_E5_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


@dataclass(frozen=True)
class ValidatedE5PackArtifact:
    root: Path
    manifest_sha256: str
    packed_query_inventory_sha256: str
    packed_queries: tuple[PackedQuery, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_manifest_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _packed_row(packed: PackedQuery) -> dict[str, object]:
    return {
        "query_id": packed.query_id,
        "contract_sha256": packed.contract_sha256,
        "input_ids": list(packed.input_ids),
        "selected_content_tokens": [
            {
                "unit_id": unit_id,
                "selected": selected,
                "full": full,
            }
            for unit_id, selected, full in packed.selected_content_tokens
        ],
        "root_included": packed.root_included,
        "context_step_positions": list(packed.context_step_positions),
    }


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def build_e5_pack_artifact(
    *,
    queries: Sequence[QueryExample],
    tokenizer: Any,
    output_dir: Path,
) -> dict[str, object]:
    if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)):
        raise TypeError("queries must be a sequence")
    sorted_queries = tuple(sorted(queries, key=lambda query: query.query_id))
    if len(sorted_queries) != 490 or len({query.query_id for query in sorted_queries}) != 490:
        raise ValueError("E5 pack artifact requires exactly 490 unique corrected queries")
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or output_dir.exists():
        raise FileExistsError(f"E5 pack output must be a new absent path: {output_dir}")
    output_dir.mkdir(parents=True)

    packed_queries = tuple(
        pack_focus_preserving_query(query, tokenizer=tokenizer)
        for query in sorted_queries
    )
    row_bytes = tuple(
        (_canonical_json(_packed_row(packed)) + "\n").encode("utf-8")
        for packed in packed_queries
    )
    queries_payload = b"".join(row_bytes)
    queries_path = output_dir / "packed_queries.jsonl"
    _write_new_file(queries_path, queries_payload)
    manifest = {
        "schema_version": E5_PACK_ARTIFACT_SCHEMA_VERSION,
        "artifact_protocol": E5_PACK_ARTIFACT_PROTOCOL,
        "packing_protocol": FOCUS_PRESERVING_PACK_PROTOCOL,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "query_source": {
            "path": "queries/all.jsonl",
            "records": 490,
            "sha256": EXPECTED_QUERY_SOURCE_SHA256,
        },
        "e5_snapshot": {
            "model_id": EXPECTED_E5_MODEL_ID,
            "revision": EXPECTED_E5_REVISION,
            "manifest_sha256": EXPECTED_E5_SNAPSHOT_MANIFEST_SHA256,
            "tree_sha256": EXPECTED_E5_SNAPSHOT_TREE_SHA256,
        },
        "query_view": "flat_plain",
        "fit_views": ["flat_plain"],
        "query_count": 490,
        "packed_query_inventory_sha256": packed_query_inventory_sha256(packed_queries),
        "packed_queries_file": {
            "path": "packed_queries.jsonl",
            "records": 490,
            "size": len(queries_payload),
            "sha256": _sha256_bytes(queries_payload),
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_new_file(manifest_path, _canonical_manifest_bytes(manifest))
    directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def _load_canonical_manifest(path: Path) -> tuple[dict[str, object], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"E5 pack manifest must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"E5 pack manifest is invalid JSON: {path}") from error
    if type(value) is not dict or raw != _canonical_manifest_bytes(value):
        raise ValueError(f"E5 pack manifest is not canonical JSON: {path}")
    return value, _sha256_bytes(raw)


def _validate_manifest_schema(manifest: dict[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "artifact_protocol",
        "packing_protocol",
        "dataset_manifest_sha256",
        "query_source",
        "e5_snapshot",
        "query_view",
        "fit_views",
        "query_count",
        "packed_query_inventory_sha256",
        "packed_queries_file",
    }
    if set(manifest) != expected_keys:
        raise ValueError("E5 pack manifest schema changed")
    expected_scalars = {
        "schema_version": E5_PACK_ARTIFACT_SCHEMA_VERSION,
        "artifact_protocol": E5_PACK_ARTIFACT_PROTOCOL,
        "packing_protocol": FOCUS_PRESERVING_PACK_PROTOCOL,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "query_view": "flat_plain",
        "fit_views": ["flat_plain"],
        "query_count": 490,
    }
    for name, expected in expected_scalars.items():
        if manifest[name] != expected or type(manifest[name]) is not type(expected):
            raise ValueError(f"E5 pack manifest {name} changed")
    if manifest["query_source"] != {
        "path": "queries/all.jsonl",
        "records": 490,
        "sha256": EXPECTED_QUERY_SOURCE_SHA256,
    }:
        raise ValueError("E5 pack query-source identity changed")
    if manifest["e5_snapshot"] != {
        "model_id": EXPECTED_E5_MODEL_ID,
        "revision": EXPECTED_E5_REVISION,
        "manifest_sha256": EXPECTED_E5_SNAPSHOT_MANIFEST_SHA256,
        "tree_sha256": EXPECTED_E5_SNAPSHOT_TREE_SHA256,
    }:
        raise ValueError("E5 pack snapshot identity changed")
    for name in ("packed_query_inventory_sha256",):
        value = manifest[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"E5 pack manifest {name} is not lowercase SHA-256")
    file_record = manifest["packed_queries_file"]
    if (
        type(file_record) is not dict
        or set(file_record) != {"path", "records", "size", "sha256"}
        or file_record["path"] != "packed_queries.jsonl"
        or file_record["records"] != 490
        or type(file_record["size"]) is not int
        or file_record["size"] < 1
        or type(file_record["sha256"]) is not str
        or len(file_record["sha256"]) != 64
    ):
        raise ValueError("E5 pack query-file record changed")


def _packed_from_row(row: dict[str, object]) -> PackedQuery:
    expected_keys = {
        "query_id",
        "contract_sha256",
        "input_ids",
        "selected_content_tokens",
        "root_included",
        "context_step_positions",
    }
    if set(row) != expected_keys:
        raise ValueError("E5 packed-query row schema changed")
    selections = row["selected_content_tokens"]
    if type(selections) is not list:
        raise ValueError("E5 packed-query selections must be a list")
    selected_records: list[tuple[str, int, int]] = []
    for record in selections:
        if type(record) is not dict or set(record) != {"unit_id", "selected", "full"}:
            raise ValueError("E5 packed-query selection schema changed")
        if (
            type(record["unit_id"]) is not str
            or type(record["selected"]) is not int
            or type(record["full"]) is not int
            or not 1 <= record["selected"] <= record["full"]
        ):
            raise ValueError("E5 packed-query selection is invalid")
        selected_records.append(
            (record["unit_id"], record["selected"], record["full"])
        )
    input_ids = row["input_ids"]
    if (
        type(input_ids) is not list
        or not input_ids
        or len(input_ids) > 512
        or any(type(value) is not int or value < 0 for value in input_ids)
    ):
        raise ValueError("E5 packed-query input_ids are invalid")
    if (
        type(row["query_id"]) is not str
        or not row["query_id"]
        or type(row["contract_sha256"]) is not str
        or len(row["contract_sha256"]) != 64
        or type(row["root_included"]) is not bool
        or row["root_included"] is not True
        or type(row["context_step_positions"]) is not list
        or any(type(value) is not int or value < 0 for value in row["context_step_positions"])
        or row["context_step_positions"]
        != sorted(set(row["context_step_positions"]))
    ):
        raise ValueError("E5 packed-query identity fields are invalid")
    return PackedQuery(
        query_id=row["query_id"],
        protocol=FOCUS_PRESERVING_PACK_PROTOCOL,
        output_view="flat_plain",
        fit_views=("flat_plain",),
        rendered_text="",
        input_ids=tuple(input_ids),
        selected_content_tokens=tuple(selected_records),
        root_included=True,
        context_step_positions=tuple(row["context_step_positions"]),
        contract_sha256=row["contract_sha256"],
    )


def validate_e5_pack_artifact(
    artifact_dir: Path,
    *,
    expected_manifest_sha256: str,
    queries: Sequence[QueryExample],
    tokenizer: Any,
) -> ValidatedE5PackArtifact:
    if (
        type(expected_manifest_sha256) is not str
        or len(expected_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
    ):
        raise ValueError("expected_manifest_sha256 must be lowercase SHA-256")
    artifact_dir = Path(artifact_dir)
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError(f"E5 pack artifact must be a real directory: {artifact_dir}")
    if sorted(path.name for path in artifact_dir.iterdir()) != [
        "manifest.json",
        "packed_queries.jsonl",
    ]:
        raise ValueError("E5 pack artifact inventory changed")
    manifest, manifest_sha256 = _load_canonical_manifest(artifact_dir / "manifest.json")
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"E5 pack manifest hash changed: actual={manifest_sha256}, "
            f"expected={expected_manifest_sha256}"
        )
    _validate_manifest_schema(manifest)
    queries_path = artifact_dir / "packed_queries.jsonl"
    if queries_path.is_symlink() or not queries_path.is_file():
        raise ValueError("E5 packed-query file must be a regular non-symlink file")
    file_record = manifest["packed_queries_file"]
    if (
        queries_path.stat().st_size != file_record["size"]
        or _sha256_file(queries_path) != file_record["sha256"]
    ):
        raise ValueError("E5 packed-query file bytes changed")
    raw_lines = queries_path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != 490:
        raise ValueError("E5 packed-query row count changed")
    packed_rows: list[PackedQuery] = []
    for position, raw_line in enumerate(raw_lines):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"E5 packed-query row {position} is invalid JSON") from error
        if type(row) is not dict or raw_line != (_canonical_json(row) + "\n").encode("utf-8"):
            raise ValueError(f"E5 packed-query row {position} is not canonical JSONL")
        packed_rows.append(_packed_from_row(row))
    if packed_query_inventory_sha256(packed_rows) != manifest["packed_query_inventory_sha256"]:
        raise ValueError("E5 packed-query inventory hash changed")

    sorted_queries = tuple(sorted(queries, key=lambda query: query.query_id))
    if len(sorted_queries) != 490:
        raise ValueError("E5 pack verification requires exactly 490 source queries")
    recomputed = tuple(
        pack_focus_preserving_query(query, tokenizer=tokenizer)
        for query in sorted_queries
    )
    if [_packed_row(value) for value in recomputed] != [
        _packed_row(value) for value in packed_rows
    ]:
        raise ValueError("E5 packed-query rows disagree with exact semantic recomputation")
    return ValidatedE5PackArtifact(
        root=artifact_dir,
        manifest_sha256=manifest_sha256,
        packed_query_inventory_sha256=manifest["packed_query_inventory_sha256"],
        packed_queries=recomputed,
    )
