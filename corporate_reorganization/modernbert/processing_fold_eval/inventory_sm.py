"""Strict Phase-1 inventory and storage probe for one retrieval fold."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


MODERNBERT_DIR = Path(__file__).resolve().parents[1]
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

from processing_fold_eval.archive_bridge import (  # noqa: E402
    _file_identity,
    _open_absent_regular,
    _open_directory_snapshot,
    _read_descriptor_exact,
    _rename_no_replace,
    _write_descriptor_exact,
    build_fold_archive_inventory_receipt,
    load_fold_archive_input_manifest,
    validate_fold_archive_inventory_receipt,
)
from processing_fold_eval.image_smoke import (  # noqa: E402
    PROCESSING_LAYOUT,
    validate_image_runtime,
)
from retriever.bm25 import (  # noqa: E402
    BM25_INDEX_ARGUMENTS,
    build_bm25_index,
    validate_bm25_runtime,
)
from retriever.data import PassageIndexTable, load_corpus, load_queries  # noqa: E402
from retriever.evaluation import build_canonical_evaluation_data  # noqa: E402
from retriever.provenance import (  # noqa: E402
    EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_FOLD_MANIFEST_SHA256,
    EXPECTED_PASSAGE_INDEX_SHA256,
)
from retriever.staged_data import validate_staged_dataset_and_fold  # noqa: E402


PHASE1_STORAGE_PROTOCOL = "retrieval_cv_fold_bm25_storage_v1"
PHASE1_OUTPUT_PROTOCOL = "retrieval_cv_fold_inventory_output_v1"
IMAGE_CONTRACT_PATH = Path(
    "/opt/program/modernbert/processing_fold_eval/image_contract.json"
)
PROCESSING_ROOT = Path("/opt/ml/processing")
WORK_PARENT = Path(PROCESSING_LAYOUT["work_parent"])
BM25_SCRATCH_DIRS = (
    WORK_PARENT / "bm25-inventory-a",
    WORK_PARENT / "bm25-inventory-b",
)
ARCHIVE_INVENTORY_NAME = "archive_inventory.json"
BM25_STORAGE_NAME = "bm25_storage.json"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"


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


def _document_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Input must be one singly-linked regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
            after.st_nlink,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_mode,
            metadata.st_nlink,
        ):
            raise RuntimeError(f"Input changed while hashed: {path}")
    finally:
        os.close(descriptor)
    live = path.lstat()
    if (
        live.st_dev,
        live.st_ino,
        live.st_size,
        live.st_mtime_ns,
        live.st_ctime_ns,
        live.st_mode,
        live.st_nlink,
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_nlink,
    ):
        raise RuntimeError(f"Input path changed while hashed: {path}")
    return digest.hexdigest()


def _require_canonical_absolute(path: Path, *, name: str) -> Path:
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve(strict=False) != path
        or path.as_posix().startswith("//")
    ):
        raise ValueError(f"{name} must be one canonical absolute path")
    return path


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = _require_canonical_absolute(Path(path), name=name)
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{name} contains a symlink component: {current}")
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be one real directory: {path}")
    return path


def _assert_no_hardlinks_or_special(root: Path, *, name: str) -> None:
    root = _require_real_directory(root, name=name)
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix() if path != root else "."
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{name} contains a symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ValueError(f"{name} contains a hard-linked file: {relative}")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{name} contains a special entry: {relative}")


def _secure_create_work_parent() -> Path:
    processing = _require_real_directory(PROCESSING_ROOT, name="Processing root")
    if WORK_PARENT.parent != processing:
        raise RuntimeError("Bound work parent left the Processing root")
    if WORK_PARENT.exists() or WORK_PARENT.is_symlink():
        raise FileExistsError("Phase-1 work parent must be initially absent")
    parent_descriptor = os.open(
        processing, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        os.mkdir(WORK_PARENT.name, mode=0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        metadata = os.stat(
            WORK_PARENT.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink != 2:
            raise RuntimeError("Phase-1 work parent creation was not stable")
    finally:
        os.close(parent_descriptor)
    live = WORK_PARENT.lstat()
    if (
        stat.S_ISLNK(live.st_mode)
        or not stat.S_ISDIR(live.st_mode)
        or live.st_dev != metadata.st_dev
        or live.st_ino != metadata.st_ino
    ):
        raise RuntimeError("Phase-1 work parent path changed after creation")
    return WORK_PARENT


def _fold_global_test_data(
    *,
    input_manifest: Mapping[str, Any],
    dataset_dir: Path,
    fold_manifest_path: Path,
):
    fold_manifest = validate_staged_dataset_and_fold(
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
        expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        expected_fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
        expected_dataset_manifest_logical_path=EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
    )
    outer_fold = input_manifest["outer_fold"]
    rotation = fold_manifest["rotations"][outer_fold]
    if rotation["outer_fold"] != outer_fold:
        raise RuntimeError("Frozen fold rotation disagrees with the archive manifest")
    test = rotation["test"]
    corpus_by_passage_id = load_corpus(dataset_dir)
    all_queries = load_queries(dataset_dir, "all")
    if len(corpus_by_passage_id) != 5_286 or len(all_queries) != 490:
        raise RuntimeError("Phase-1 requires exact corrected 5,286/490 data")
    passage_index = PassageIndexTable(corpus_by_passage_id)
    if passage_index.sha256 != EXPECTED_PASSAGE_INDEX_SHA256:
        raise RuntimeError("Phase-1 corrected passage index changed")
    evaluation_data = build_canonical_evaluation_data(
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        evaluated_case_ids=tuple(test["case_ids"]),
        role="test",
        regime_name="fold_global",
    )
    if (
        evaluation_data.query_count != test["queries"]
        or evaluation_data.passage_count != test["passages"]
    ):
        raise RuntimeError("Phase-1 fold-global test inventory changed")
    return fold_manifest, corpus_by_passage_id, evaluation_data


def _preflight_bm25_inputs(
    *,
    input_manifest: Mapping[str, Any],
    dataset_dir: Path,
    fold_manifest_path: Path,
) -> None:
    """Reject invalid Phase-1 data and scratch state before archive scanning."""

    dataset_dir = _require_real_directory(dataset_dir, name="Phase-1 dataset")
    _assert_no_hardlinks_or_special(dataset_dir, name="Phase-1 dataset")
    fold_manifest_path = _require_canonical_absolute(
        fold_manifest_path, name="Phase-1 fold manifest"
    )
    fold_metadata = fold_manifest_path.lstat()
    if (
        stat.S_ISLNK(fold_metadata.st_mode)
        or not stat.S_ISREG(fold_metadata.st_mode)
        or fold_metadata.st_nlink != 1
    ):
        raise ValueError("Phase-1 fold manifest must be singly linked and regular")
    processing = _require_real_directory(PROCESSING_ROOT, name="Processing root")
    if WORK_PARENT.parent != processing:
        raise RuntimeError("Bound work parent left the Processing root")
    if WORK_PARENT.exists() or WORK_PARENT.is_symlink():
        raise FileExistsError("Phase-1 work parent must be initially absent")
    _fold_global_test_data(
        input_manifest=input_manifest,
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
    )


def _statvfs_payload(path: Path) -> dict[str, int]:
    value = os.statvfs(path)
    return {
        "block_size": value.f_bsize,
        "fragment_size": value.f_frsize,
        "blocks": value.f_blocks,
        "blocks_free": value.f_bfree,
        "blocks_available": value.f_bavail,
        "capacity_bytes": value.f_blocks * value.f_frsize,
        "free_bytes": value.f_bfree * value.f_frsize,
        "available_bytes": value.f_bavail * value.f_frsize,
    }


def _measure_tree(root: Path) -> dict[str, Any]:
    root = _require_real_directory(root, name="BM25 scratch tree")
    records: list[dict[str, Any]] = []
    allocated_bytes = 0
    logical_bytes = 0
    file_count = 0
    directory_count = 0
    for path in (root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())):
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        allocated = metadata.st_blocks * 512
        if allocated < 0 or allocated % 512:
            raise RuntimeError("BM25 scratch allocation is not reported in 512-byte blocks")
        allocated_bytes += allocated
        logical_bytes += metadata.st_size
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"BM25 scratch contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            directory_count += 1
            records.append(
                {
                    "allocated_bytes": allocated,
                    "kind": "directory",
                    "logical_size": metadata.st_size,
                    "path": relative,
                }
            )
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise RuntimeError(f"BM25 scratch contains a hard link: {relative}")
            file_count += 1
            records.append(
                {
                    "allocated_bytes": allocated,
                    "kind": "file",
                    "logical_size": metadata.st_size,
                    "path": relative,
                }
            )
        else:
            raise RuntimeError(f"BM25 scratch contains a special entry: {relative}")
    return {
        "allocated_bytes": allocated_bytes,
        "directory_count": directory_count,
        "file_count": file_count,
        "allocation_tree_sha256": _document_sha256(records),
        "logical_bytes": logical_bytes,
        "records": records,
    }


def _seal(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result[field] = _document_sha256(result)
    return result


def build_bm25_storage_receipt(
    *,
    input_manifest: Mapping[str, Any],
    dataset_dir: Path,
    fold_manifest_path: Path,
    archive_inventory: Mapping[str, Any],
    image_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_dir = _require_real_directory(dataset_dir, name="Phase-1 dataset")
    _assert_no_hardlinks_or_special(dataset_dir, name="Phase-1 dataset")
    fold_manifest_path = _require_canonical_absolute(
        fold_manifest_path, name="Phase-1 fold manifest"
    )
    fold_metadata = fold_manifest_path.lstat()
    if (
        stat.S_ISLNK(fold_metadata.st_mode)
        or not stat.S_ISREG(fold_metadata.st_mode)
        or fold_metadata.st_nlink != 1
    ):
        raise ValueError("Phase-1 fold manifest must be singly linked and regular")
    fold_manifest, corpus_by_passage_id, evaluation_data = _fold_global_test_data(
        input_manifest=input_manifest,
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
    )
    before = _statvfs_payload(PROCESSING_ROOT)
    work_parent = _secure_create_work_parent()
    passage_ids = evaluation_data.passage_ids
    passage_texts = tuple(
        corpus_by_passage_id[passage_id].text for passage_id in passage_ids
    )
    runtime = validate_bm25_runtime().to_payload()
    trees: list[dict[str, Any]] = []
    replicas: list[dict[str, Any]] = []
    for ordinal, scratch_dir in enumerate(BM25_SCRATCH_DIRS, start=1):
        index_dir = build_bm25_index(
            passage_ids=passage_ids,
            passage_texts=passage_texts,
            scratch_dir=scratch_dir,
        )
        if index_dir != scratch_dir / "index":
            raise RuntimeError("Pinned BM25 builder returned an unexpected index path")
        tree = _measure_tree(scratch_dir)
        trees.append(tree)
        replicas.append(
            {
                "ordinal": ordinal,
                "scratch_path": str(scratch_dir),
                "allocation_tree_sha256": tree["allocation_tree_sha256"],
            }
        )
    if trees[0] != trees[1]:
        raise RuntimeError("Independent BM25 storage probes produced different allocations")
    after = _statvfs_payload(work_parent)
    outer_fold = input_manifest["outer_fold"]
    test = fold_manifest["rotations"][outer_fold]["test"]
    final_fold_manifest, _, final_evaluation_data = _fold_global_test_data(
        input_manifest=input_manifest,
        dataset_dir=dataset_dir,
        fold_manifest_path=fold_manifest_path,
    )
    if (
        final_fold_manifest != fold_manifest
        or final_evaluation_data.case_ids_sha256
        != evaluation_data.case_ids_sha256
        or final_evaluation_data.query_ids_sha256
        != evaluation_data.query_ids_sha256
        or final_evaluation_data.passage_ids_sha256
        != evaluation_data.passage_ids_sha256
        or final_evaluation_data.candidate_pools_sha256
        != evaluation_data.candidate_pools_sha256
        or final_evaluation_data.contract_sha256 != evaluation_data.contract_sha256
    ):
        raise RuntimeError("Phase-1 dataset or fold identity changed during BM25 probing")
    fold_manifest_sha256 = _sha256_file(fold_manifest_path)
    if fold_manifest_sha256 != EXPECTED_FOLD_MANIFEST_SHA256:
        raise RuntimeError("Phase-1 fold manifest changed after final validation")
    return _seal(
        {
            "schema_version": 1,
            "protocol": PHASE1_STORAGE_PROTOCOL,
            "experiment_id": input_manifest["experiment_id"],
            "outer_fold": outer_fold,
            "role": "test",
            "regime": "fold_global",
            "archive_input_manifest_sha256": archive_inventory[
                "input_manifest_sha256"
            ],
            "archive_inventory_receipt_sha256": archive_inventory[
                "receipt_sha256"
            ],
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "fold_manifest_sha256": fold_manifest_sha256,
            "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
            "case_ids": list(test["case_ids"]),
            "case_ids_sha256": evaluation_data.case_ids_sha256,
            "query_count": evaluation_data.query_count,
            "query_ids_sha256": evaluation_data.query_ids_sha256,
            "passage_count": evaluation_data.passage_count,
            "passage_ids_sha256": evaluation_data.passage_ids_sha256,
            "candidate_pools_sha256": evaluation_data.candidate_pools_sha256,
            "evaluation_contract_sha256": evaluation_data.contract_sha256,
            "bm25_index_arguments": list(BM25_INDEX_ARGUMENTS),
            "bm25_runtime": runtime,
            "bm25_replicas": replicas,
            "bm25_allocation_tree": trees[0],
            "filesystem_before": before,
            "filesystem_after": after,
            "image_runtime": copy.deepcopy(dict(image_runtime)),
        },
        field="receipt_sha256",
    )


def _write_payload_at(stage, name: str, payload: bytes) -> None:
    descriptor = _open_absent_regular(stage.descriptor, name)
    try:
        _write_descriptor_exact(descriptor, payload, name=f"Phase-1 output {name}")
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        identity = _file_identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError(f"Phase-1 output file became unsafe: {name}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if _read_descriptor_exact(
            descriptor, len(payload), name=f"Phase-1 output readback {name}"
        ) != payload or _file_identity(os.fstat(descriptor)) != identity:
            raise RuntimeError(f"Phase-1 output readback changed: {name}")
        stage.assert_stable()
        child = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
        if _file_identity(child) != identity:
            raise RuntimeError(f"Phase-1 output entry changed: {name}")
    finally:
        os.close(descriptor)


def _verify_payload_at(stage, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=stage.descriptor,
    )
    try:
        metadata = os.fstat(descriptor)
        identity = _file_identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError(f"Published Phase-1 output became unsafe: {name}")
        if _read_descriptor_exact(
            descriptor, len(payload), name=f"Published Phase-1 output {name}"
        ) != payload:
            raise RuntimeError(f"Published Phase-1 output changed: {name}")
        stage.assert_stable()
        child = os.stat(name, dir_fd=stage.descriptor, follow_symlinks=False)
        if _file_identity(child) != identity:
            raise RuntimeError(f"Published Phase-1 output entry changed: {name}")
    finally:
        os.close(descriptor)


def _publish_payloads(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    incomplete = output_dir.with_name(f".{output_dir.name}.incomplete")
    with _open_directory_snapshot(
        output_dir.parent, name="Phase-1 output parent"
    ) as publication_parent:
        os.mkdir(incomplete.name, mode=0o700, dir_fd=publication_parent.descriptor)
        created = os.stat(
            incomplete.name,
            dir_fd=publication_parent.descriptor,
            follow_symlinks=False,
        )
        created_identity = (created.st_dev, created.st_ino, created.st_mode)
        if not stat.S_ISDIR(created.st_mode) or stat.S_IMODE(created.st_mode) != 0o700:
            raise RuntimeError("Incomplete Phase-1 output creation changed")
        publication_parent.assert_stable()
        with _open_directory_snapshot(
            incomplete, name="Incomplete Phase-1 output"
        ) as staging:
            if staging.identity != created_identity:
                raise RuntimeError("Incomplete Phase-1 output was replaced before open")
            for name, payload in payloads.items():
                publication_parent.assert_stable()
                staging.assert_stable()
                _write_payload_at(staging, name, payload)
            if set(os.listdir(staging.descriptor)) != set(payloads):
                raise RuntimeError("Incomplete Phase-1 output inventory changed")
            os.fsync(staging.descriptor)
            publication_parent.assert_stable()
            staging.assert_stable()
            _rename_no_replace(
                publication_parent.descriptor,
                incomplete.name,
                output_dir.name,
            )
            staging.rebind(output_dir, name="Published Phase-1 output")
            os.fsync(publication_parent.descriptor)
            publication_parent.assert_stable()
            if set(os.listdir(staging.descriptor)) != set(payloads):
                raise RuntimeError("Published Phase-1 output inventory changed")
            for name, payload in payloads.items():
                _verify_payload_at(staging, name, payload)


def publish_phase1_output(
    *,
    output_dir: Path,
    input_manifest: Mapping[str, Any],
    archive_inventory: Mapping[str, Any],
    bm25_storage: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = _require_canonical_absolute(output_dir, name="Phase-1 output")
    if output_dir != Path(PROCESSING_LAYOUT["evidence_output_dir"]):
        raise ValueError("Phase-1 output must use the contract-bound evidence path")
    parent = _require_real_directory(output_dir.parent, name="Phase-1 output parent")
    if parent != Path(PROCESSING_LAYOUT["output_parent"]):
        raise ValueError("Phase-1 output parent left the Processing contract")
    incomplete = output_dir.with_name(f".{output_dir.name}.incomplete")
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or incomplete.exists()
        or incomplete.is_symlink()
    ):
        raise FileExistsError("Phase-1 output and sibling incomplete path must be absent")
    payloads = {
        ARCHIVE_INVENTORY_NAME: _canonical_bytes(archive_inventory),
        BM25_STORAGE_NAME: _canonical_bytes(bm25_storage),
    }
    files = [
        {"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        for name, payload in sorted(payloads.items())
    ]
    artifact_manifest = _seal(
        {
            "schema_version": 1,
            "protocol": PHASE1_OUTPUT_PROTOCOL,
            "experiment_id": input_manifest["experiment_id"],
            "outer_fold": input_manifest["outer_fold"],
            "archive_input_manifest_sha256": archive_inventory[
                "input_manifest_sha256"
            ],
            "archive_inventory_receipt_sha256": archive_inventory[
                "receipt_sha256"
            ],
            "bm25_storage_receipt_sha256": bm25_storage["receipt_sha256"],
            "files": files,
        },
        field="artifact_manifest_sha256",
    )
    published_payloads = {
        **payloads,
        ARTIFACT_MANIFEST_NAME: _canonical_bytes(artifact_manifest),
    }
    _publish_payloads(output_dir, published_payloads)
    return artifact_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory 12 fold archives and measure exact test BM25 storage.",
        allow_abbrev=False,
    )
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--scratch-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    archive_manifest_path = _require_canonical_absolute(
        args.archive_manifest, name="Phase-1 archive manifest"
    )
    if archive_manifest_path != Path(PROCESSING_LAYOUT["archive_manifest_path"]):
        raise ValueError("Phase-1 archive manifest left the contract-bound mount")
    if Path(args.dataset_dir) != Path(PROCESSING_LAYOUT["dataset_dir"]):
        raise ValueError("Phase-1 dataset left the contract-bound mount")
    if Path(args.fold_manifest) != Path(PROCESSING_LAYOUT["fold_manifest_path"]):
        raise ValueError("Phase-1 fold manifest left the contract-bound control path")
    if Path(args.scratch_parent) != WORK_PARENT:
        raise ValueError("Phase-1 scratch parent left the contract-bound work path")
    if Path(args.output_dir) != Path(PROCESSING_LAYOUT["evidence_output_dir"]):
        raise ValueError("Phase-1 output left the contract-bound evidence path")
    output_dir = Path(args.output_dir)
    incomplete_output = output_dir.with_name(f".{output_dir.name}.incomplete")
    _require_real_directory(output_dir.parent, name="Phase-1 output parent")
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or incomplete_output.exists()
        or incomplete_output.is_symlink()
    ):
        raise FileExistsError("Phase-1 output must be initially absent")
    input_manifest = load_fold_archive_input_manifest(archive_manifest_path)
    if Path(input_manifest["archive_root"]) != archive_manifest_path.parent:
        raise ValueError("Phase-1 archive root left the contract-bound mount")
    _preflight_bm25_inputs(
        input_manifest=input_manifest,
        dataset_dir=args.dataset_dir,
        fold_manifest_path=args.fold_manifest,
    )
    image_runtime = validate_image_runtime(IMAGE_CONTRACT_PATH)
    archive_inventory = build_fold_archive_inventory_receipt(input_manifest)
    validate_fold_archive_inventory_receipt(
        archive_inventory, input_manifest=input_manifest
    )
    bm25_storage = build_bm25_storage_receipt(
        input_manifest=input_manifest,
        dataset_dir=args.dataset_dir,
        fold_manifest_path=args.fold_manifest,
        archive_inventory=archive_inventory,
        image_runtime=image_runtime,
    )
    artifact_manifest = publish_phase1_output(
        output_dir=args.output_dir,
        input_manifest=input_manifest,
        archive_inventory=archive_inventory,
        bm25_storage=bm25_storage,
    )
    print(
        json.dumps(
            artifact_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
