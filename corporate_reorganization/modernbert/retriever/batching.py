from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


DUMMY_QUERY_INDEX = -1
BATCH_ORDER_ALGORITHM = "sha256_query_order_v1"


def _require_exact_int(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _query_order_digest(*, experiment_seed: int, epoch: int, query_id: str) -> bytes:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": BATCH_ORDER_ALGORITHM,
                "epoch": epoch,
                "experiment_seed": experiment_seed,
                "query_id": query_id,
            }
        )
    ).digest()


class GlobalQueryBatchSampler:
    """Build rank-ordered local batches that Accelerate shards exactly once."""

    drop_last = False

    def __init__(
        self,
        query_ids: Sequence[str],
        *,
        experiment_seed: int,
        world_size: int,
        per_device_batch_size: int,
    ) -> None:
        self.query_ids = tuple(query_ids)
        if not self.query_ids:
            raise ValueError("query_ids must not be empty")
        if any(type(query_id) is not str or not query_id or query_id.strip() != query_id for query_id in self.query_ids):
            raise ValueError("Every query_id must be a non-empty, whitespace-trimmed string")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("query_ids contains duplicates")

        self.experiment_seed = _require_exact_int(
            "experiment_seed",
            experiment_seed,
            minimum=0,
        )
        self.world_size = _require_exact_int("world_size", world_size, minimum=1)
        self.batch_size = _require_exact_int(
            "per_device_batch_size",
            per_device_batch_size,
            minimum=1,
        )
        self.epoch = 0

        final_remainder = len(self.query_ids) % self.global_microbatch_size
        if final_remainder and final_remainder < self.world_size:
            raise ValueError(
                "The final global microbatch would give at least one rank no real query: "
                f"queries={len(self.query_ids)}, world_size={self.world_size}, "
                f"per_device_batch_size={self.batch_size}, remainder={final_remainder}. "
                "Step 4 forbids all-dummy ranks."
            )

    @property
    def global_microbatch_size(self) -> int:
        return self.world_size * self.batch_size

    @property
    def prepared_batches_per_rank(self) -> int:
        return (len(self.query_ids) + self.global_microbatch_size - 1) // self.global_microbatch_size

    @property
    def num_sentinel_rows(self) -> int:
        return self.prepared_batches_per_rank * self.global_microbatch_size - len(self.query_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _require_exact_int("epoch", epoch, minimum=0)

    def ordered_real_indices(self) -> list[int]:
        return sorted(
            range(len(self.query_ids)),
            key=lambda index: (
                _query_order_digest(
                    experiment_seed=self.experiment_seed,
                    epoch=self.epoch,
                    query_id=self.query_ids[index],
                ),
                self.query_ids[index],
            ),
        )

    def batches(self) -> list[list[int]]:
        ordered = self.ordered_real_indices()
        raw_batches: list[list[int]] = []
        global_size = self.global_microbatch_size

        for offset in range(0, len(ordered), global_size):
            global_indices = ordered[offset : offset + global_size]
            if len(global_indices) == global_size:
                for rank in range(self.world_size):
                    start = rank * self.batch_size
                    raw_batches.append(global_indices[start : start + self.batch_size])
                continue

            rank_batches: list[list[int]] = [[] for _ in range(self.world_size)]
            for position, index in enumerate(global_indices):
                rank_batches[position % self.world_size].append(index)
            for rank_batch in rank_batches:
                rank_batch.extend([DUMMY_QUERY_INDEX] * (self.batch_size - len(rank_batch)))
                raw_batches.append(rank_batch)

        self._validate_batches(raw_batches)
        return raw_batches

    def _validate_batches(self, raw_batches: Sequence[Sequence[int]]) -> None:
        if len(raw_batches) != len(self):
            raise RuntimeError(
                f"Internal batch count mismatch: generated={len(raw_batches)}, expected={len(self)}"
            )
        if len(raw_batches) % self.world_size != 0:
            raise RuntimeError("Raw batch count is not divisible by world_size")
        if any(len(batch) != self.batch_size for batch in raw_batches):
            raise RuntimeError("Generated a non-full local batch")

        flattened = [index for batch in raw_batches for index in batch]
        real_indices = [index for index in flattened if index != DUMMY_QUERY_INDEX]
        if sorted(real_indices) != list(range(len(self.query_ids))):
            raise RuntimeError("Generated batches do not contain each real query index exactly once")
        if len(flattened) - len(real_indices) != self.num_sentinel_rows:
            raise RuntimeError("Generated batch sentinel count does not match the minimal padding count")

        for global_batch_offset in range(0, len(raw_batches), self.world_size):
            rank_group = raw_batches[global_batch_offset : global_batch_offset + self.world_size]
            if any(all(index == DUMMY_QUERY_INDEX for index in batch) for batch in rank_group):
                raise RuntimeError("Generated an all-dummy rank batch, which Step 4 forbids")

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches()

    def __len__(self) -> int:
        return self.prepared_batches_per_rank * self.world_size


class SentinelQueryDataset:
    """Map the one reserved sentinel index to a record with no scientific content."""

    def __init__(self, dataset: Any, *, epoch_target: Any | None = None) -> None:
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("Wrapped dataset must implement __len__ and __getitem__")
        if len(dataset) < 1:
            raise ValueError("Wrapped dataset must not be empty")
        if epoch_target is not None and not hasattr(epoch_target, "set_epoch"):
            raise TypeError("epoch_target must implement set_epoch")
        self.dataset = dataset
        self.epoch_target = epoch_target

    def set_epoch(self, epoch: int) -> None:
        if not hasattr(self.dataset, "set_epoch"):
            raise TypeError("Wrapped dataset does not implement set_epoch")
        self.dataset.set_epoch(epoch)
        if self.epoch_target is not None:
            self.epoch_target.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if isinstance(idx, bool):
            raise TypeError("Dataset index must be an integer, not bool")
        try:
            index = operator.index(idx)
        except TypeError as exc:
            raise TypeError(f"Dataset index must be an integer; got {type(idx).__name__}") from exc

        if index == DUMMY_QUERY_INDEX:
            return {"is_dummy": True}
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Dataset index out of range: {index}")

        record = self.dataset[index]
        if not isinstance(record, Mapping):
            raise TypeError(f"Wrapped dataset returned {type(record).__name__}, expected a mapping")
        if "is_dummy" in record:
            raise ValueError("Wrapped real record must not define reserved field is_dummy")
        return {**record, "is_dummy": False}
