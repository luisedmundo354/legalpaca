from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from .data import PassageIndexTable
from .sampling import validate_sampling_trace


INVALID_PASSAGE_INDEX = -1
TRACE_ARTIFACT_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new_file(path: Path, content: str) -> None:
    """Publish complete bytes atomically while refusing an existing target."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite trace artifact: {path}")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"Refusing stale trace temporary file: {temporary_path}")

    published = False
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary_path, path)
        published = True
        temporary_path.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if published and (path.exists() or path.is_symlink()):
            path.unlink()
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()
        _fsync_directory(path.parent)
        raise


def _failure_payload(context: str, error: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "context": context,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _raise_collective_failure(payload: Mapping[str, Any]) -> None:
    raise RuntimeError(
        f"{payload['context']} failed on rank 0: "
        f"{payload['error_type']}: {payload['message']}"
    )


def _unpadded_row(tensor: torch.Tensor, row_index: int, *, name: str) -> list[int]:
    if tensor.dtype != torch.long or tensor.ndim != 2:
        raise TypeError(f"{name} must be a rank-2 torch.long tensor")
    row = tensor[row_index].detach().cpu()
    valid = row.ne(INVALID_PASSAGE_INDEX)
    count = int(valid.sum().item())
    expected = torch.arange(row.numel()) < count
    if not torch.equal(valid, expected):
        raise ValueError(f"{name} row {row_index} has non-suffix padding")
    values = [int(value) for value in row[:count].tolist()]
    if not values:
        raise ValueError(f"{name} row {row_index} is empty")
    return values


class CandidateTraceStore:
    """Exclusive rank-sharded trace writer and strict post-training merger."""

    def __init__(
        self,
        output_dir: Path,
        *,
        passage_index_table: PassageIndexTable,
        rank: int,
        world_size: int,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Candidate trace persistence requires an initialized process group")
        if rank != dist.get_rank() or world_size != dist.get_world_size():
            raise RuntimeError("Candidate trace rank/world_size disagree with the process group")
        if not isinstance(passage_index_table, PassageIndexTable):
            raise TypeError("passage_index_table must be a PassageIndexTable")
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(f"Trace output root must be a real existing directory: {output_dir}")

        self.output_dir = output_dir
        self.passage_index_table = passage_index_table
        self.rank = rank
        self.world_size = world_size
        self.trace_dir = output_dir / "candidate_traces"
        status: list[object] = [None]
        if rank == 0:
            try:
                self.trace_dir.mkdir(exist_ok=False)
                _fsync_directory(output_dir)
                status[0] = {"ok": True}
            except BaseException as error:
                status[0] = _failure_payload("Candidate trace directory creation", error)
        dist.broadcast_object_list(status, src=0)
        if type(status[0]) is not dict:
            raise RuntimeError("Candidate trace directory creation returned malformed status")
        if status[0].get("ok") is not True:
            _raise_collective_failure(status[0])
        if not self.trace_dir.is_dir() or self.trace_dir.is_symlink():
            raise RuntimeError(f"Candidate trace directory was not created safely: {self.trace_dir}")

        self.shard_path = self.trace_dir / f"rank-{rank:05d}.jsonl"
        stream = None
        local_status: dict[str, Any]
        try:
            stream = self.shard_path.open("x", encoding="utf-8", newline="\n")
            local_status = {"ok": True, "rank": rank}
        except BaseException as error:
            local_status = {
                **_failure_payload("Candidate trace shard creation", error),
                "rank": rank,
            }
        gathered_status: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_status, local_status)
        failures = [
            item
            for item in gathered_status
            if type(item) is not dict or item.get("ok") is not True
        ]
        if failures:
            if stream is not None:
                stream.close()
            raise RuntimeError(f"Candidate trace shard creation failed collectively: {failures}")
        if stream is None:
            raise RuntimeError("Candidate trace shard stream was not created")
        self._stream = stream
        self._seen_keys: set[tuple[int, str]] = set()
        self._closed = False

    def record_batch(
        self,
        traces: object,
        *,
        candidate_passage_indices: torch.Tensor,
        positive_passage_indices: torch.Tensor,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record candidate traces after finalization")
        if type(traces) is not list or not traces:
            raise TypeError("sampling_traces must be a non-empty exact list")
        if candidate_passage_indices.shape[0] != len(traces):
            raise ValueError("Candidate index rows do not align with sampling traces")
        if positive_passage_indices.shape[0] != len(traces):
            raise ValueError("Positive index rows do not align with sampling traces")

        for row_index, trace in enumerate(traces):
            validate_sampling_trace(trace)
            candidate_indices = _unpadded_row(
                candidate_passage_indices,
                row_index,
                name="candidate_passage_indices",
            )
            positive_indices = _unpadded_row(
                positive_passage_indices,
                row_index,
                name="positive_passage_indices",
            )
            expected_candidates = self.passage_index_table.indices_for_ids(
                trace["candidate_passage_ids"]
            )
            expected_positives = self.passage_index_table.indices_for_ids(
                trace["positive_passage_ids"]
            )
            if candidate_indices != expected_candidates:
                raise ValueError(
                    f"Trace/index candidate mismatch for query_id={trace['query_id']!r}"
                )
            if positive_indices != expected_positives:
                raise ValueError(
                    f"Trace/index positive mismatch for query_id={trace['query_id']!r}"
                )

            key = (trace["epoch"], trace["query_id"])
            if key in self._seen_keys:
                raise ValueError(f"Duplicate rank-local sampling trace key={key!r}")
            self._seen_keys.add(key)
            self._stream.write(_canonical_json(trace) + "\n")
        self._stream.flush()

    def _close_shard(self) -> None:
        if self._closed:
            raise RuntimeError("Candidate trace shard was already finalized")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True

    def _validate_manifest_payload(
        self,
        manifest: object,
        *,
        expected_epochs: int,
        query_ids: list[str],
    ) -> None:
        expected_keys = {
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
        if type(manifest) is not dict or set(manifest) != expected_keys:
            raise RuntimeError("Merged trace manifest fields do not match schema")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != TRACE_ARTIFACT_SCHEMA_VERSION
        ):
            raise RuntimeError("Merged trace manifest schema version changed")
        if type(manifest["merge_order"]) is not list or manifest["merge_order"] != [
            "epoch",
            "query_id",
        ]:
            raise RuntimeError("Merged trace ordering contract changed")
        if (
            type(manifest["epochs"]) is not int
            or manifest["epochs"] != expected_epochs
            or type(manifest["queries_per_epoch"]) is not int
            or manifest["queries_per_epoch"] != len(query_ids)
        ):
            raise RuntimeError("Merged trace manifest coverage dimensions changed")
        if manifest["passage_index_sha256"] != self.passage_index_table.sha256:
            raise RuntimeError("Merged trace manifest passage-index digest mismatch")
        expected_query_id_sha256 = hashlib.sha256(
            _canonical_json(sorted(query_ids)).encode("utf-8")
        ).hexdigest()
        if (
            type(manifest["query_ids_sha256"]) is not str
            or manifest["query_ids_sha256"] != expected_query_id_sha256
        ):
            raise RuntimeError("Merged trace manifest query-ID digest mismatch")
        expected_count = expected_epochs * len(query_ids)
        if type(manifest["record_count"]) is not int or manifest["record_count"] != expected_count:
            raise RuntimeError(
                f"Merged trace manifest has {manifest['record_count']} records; "
                f"expected {expected_count}"
            )
        merged_record = manifest["merged"]
        if (
            type(merged_record) is not dict
            or set(merged_record) != {"path", "record_count", "size", "sha256"}
            or merged_record["path"] != "sampling_traces.jsonl"
            or type(merged_record["record_count"]) is not int
            or merged_record["record_count"] != expected_count
            or type(merged_record["size"]) is not int
            or merged_record["size"] < 1
            or type(merged_record["sha256"]) is not str
            or len(merged_record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in merged_record["sha256"])
        ):
            raise RuntimeError("Merged sampling trace manifest record is invalid")
        shards = manifest["shards"]
        if type(shards) is not list or len(shards) != self.world_size:
            raise RuntimeError("Merged trace manifest shard count changed")
        if [record.get("rank") for record in shards if type(record) is dict] != list(
            range(self.world_size)
        ):
            raise RuntimeError("Merged trace manifest shard ranks changed")
        for rank, record in enumerate(shards):
            if (
                type(record) is not dict
                or set(record) != {"rank", "path", "record_count", "size", "sha256"}
                or type(record["rank"]) is not int
                or record["rank"] != rank
                or record["path"] != f"rank-{rank:05d}.jsonl"
                or type(record["record_count"]) is not int
                or record["record_count"] < 0
                or type(record["size"]) is not int
                or record["size"] < 0
                or type(record["sha256"]) is not str
                or len(record["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in record["sha256"])
            ):
                raise RuntimeError(f"Merged trace manifest shard record is invalid for rank={rank}")
        if sum(record["record_count"] for record in shards) != expected_count:
            raise RuntimeError("Merged trace manifest shard counts do not sum to total coverage")

    def _merge_rank_zero(
        self,
        *,
        expected_epochs: int,
        query_ids: list[str],
    ) -> dict[str, Any]:
        expected_shard_names = [f"rank-{rank:05d}.jsonl" for rank in range(self.world_size)]
        actual_entries = sorted(self.trace_dir.iterdir(), key=lambda path: path.name)
        actual_names = [path.name for path in actual_entries]
        if actual_names != expected_shard_names:
            raise RuntimeError(
                f"Candidate trace shard inventory mismatch: actual={actual_names}, "
                f"expected={expected_shard_names}"
            )
        if any(path.is_symlink() or not path.is_file() for path in actual_entries):
            raise RuntimeError("Every candidate trace shard must be a regular non-symlink file")

        records_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
        shard_records: list[dict[str, Any]] = []
        for shard_rank, shard_name in enumerate(expected_shard_names):
            shard_path = self.trace_dir / shard_name
            count = 0
            with shard_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.endswith("\n") or not line.strip():
                        raise ValueError(
                            f"Malformed candidate trace line {shard_name}:{line_number}"
                        )
                    trace = json.loads(line)
                    validate_sampling_trace(trace)
                    key = (trace["epoch"], trace["query_id"])
                    if key in records_by_key:
                        raise ValueError(f"Duplicate cross-rank sampling trace key={key!r}")
                    records_by_key[key] = trace
                    count += 1
            shard_records.append(
                {
                    "rank": shard_rank,
                    "path": shard_name,
                    "record_count": count,
                    "size": shard_path.stat().st_size,
                    "sha256": _sha256_file(shard_path),
                }
            )

        expected_keys = {
            (epoch, query_id)
            for epoch in range(expected_epochs)
            for query_id in query_ids
        }
        actual_keys = set(records_by_key)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise RuntimeError(
                "Sampling trace coverage mismatch: "
                f"missing={missing[:20]}, extra={extra[:20]}, "
                f"missing_count={len(missing)}, extra_count={len(extra)}"
            )

        merged_path = self.trace_dir / "sampling_traces.jsonl"
        manifest_path = self.trace_dir / "manifest.json"
        merged_content = "".join(
            _canonical_json(records_by_key[key]) + "\n"
            for key in sorted(records_by_key)
        )
        try:
            _publish_new_file(merged_path, merged_content)
            query_id_payload = _canonical_json(sorted(query_ids)).encode("utf-8")
            manifest = {
                "schema_version": TRACE_ARTIFACT_SCHEMA_VERSION,
                "merge_order": ["epoch", "query_id"],
                "epochs": expected_epochs,
                "queries_per_epoch": len(query_ids),
                "record_count": len(records_by_key),
                "query_ids_sha256": hashlib.sha256(query_id_payload).hexdigest(),
                "passage_index_sha256": self.passage_index_table.sha256,
                "merged": {
                    "path": merged_path.name,
                    "record_count": len(records_by_key),
                    "size": merged_path.stat().st_size,
                    "sha256": _sha256_file(merged_path),
                },
                "shards": shard_records,
            }
            manifest_content = (
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            _publish_new_file(manifest_path, manifest_content)
            published_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._validate_manifest_payload(
                published_manifest,
                expected_epochs=expected_epochs,
                query_ids=query_ids,
            )
            if (
                published_manifest["merged"]["size"] != merged_path.stat().st_size
                or published_manifest["merged"]["sha256"] != _sha256_file(merged_path)
            ):
                raise RuntimeError("Published merged trace bytes do not match the manifest")
            return published_manifest
        except BaseException:
            for artifact_path in (manifest_path, merged_path):
                if artifact_path.exists() or artifact_path.is_symlink():
                    artifact_path.unlink()
            _fsync_directory(self.trace_dir)
            raise

    def finalize(
        self,
        *,
        expected_epochs: int,
        expected_query_ids: Sequence[str],
    ) -> dict[str, Any]:
        if type(expected_epochs) is not int or expected_epochs < 1:
            raise ValueError("expected_epochs must be a positive exact int")
        query_ids = list(expected_query_ids)
        if (
            not query_ids
            or any(type(query_id) is not str or not query_id for query_id in query_ids)
            or len(query_ids) != len(set(query_ids))
        ):
            raise ValueError("expected_query_ids must be non-empty, unique exact strings")

        try:
            self._close_shard()
            local_status: dict[str, Any] = {"ok": True, "rank": self.rank}
        except BaseException as error:
            local_status = {
                **_failure_payload("Candidate trace shard finalization", error),
                "rank": self.rank,
            }
        gathered_status: list[object] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_status, local_status)
        failures = [
            item
            for item in gathered_status
            if type(item) is not dict or item.get("ok") is not True
        ]
        if failures:
            raise RuntimeError(
                f"Candidate trace shard finalization failed collectively: {failures}"
            )

        merge_status: list[object] = [None]
        if self.rank == 0:
            try:
                manifest = self._merge_rank_zero(
                    expected_epochs=expected_epochs,
                    query_ids=query_ids,
                )
                merge_status[0] = {"ok": True, "manifest": manifest}
            except BaseException as error:
                merge_status[0] = _failure_payload("Candidate trace merge", error)
        dist.broadcast_object_list(merge_status, src=0)
        if type(merge_status[0]) is not dict:
            raise RuntimeError("Candidate trace merge returned a malformed collective status")
        if merge_status[0].get("ok") is not True:
            _raise_collective_failure(merge_status[0])
        manifest = merge_status[0].get("manifest")
        self._validate_manifest_payload(
            manifest,
            expected_epochs=expected_epochs,
            query_ids=query_ids,
        )
        return manifest
