from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from typing import Final

from .batching import DUMMY_QUERY_INDEX


EXPECTED_QUERY_COUNT: Final[int] = 418
EXPECTED_WORLD_SIZE: Final[int] = 4
EXPECTED_PER_DEVICE_BATCH_SIZE: Final[int] = 4
GLOBAL_MICROBATCH_SIZE: Final[int] = 16
PREPARED_BATCHES_PER_RANK: Final[int] = 27
SENTINEL_ROW_COUNT: Final[int] = 14
TAIL_REAL_QUERY_COUNTS: Final[tuple[int, int]] = (9, 9)
GRADIENT_ACCUMULATION_STEPS: Final[int] = 8
BATCH_ORDER_ALGORITHM: Final[str] = "sha256_corrected_legacy_query_order_v1"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_int(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def _query_order_digest(*, experiment_seed: int, epoch: int, query_id: str) -> bytes:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "algorithm": BATCH_ORDER_ALGORITHM,
                "epoch": epoch,
                "experiment_seed": experiment_seed,
                "query_id": query_id,
            }
        )
    ).digest()


class CorrectedLegacyQueryBatchSampler:
    """Exact four-rank 418-query plan with the frozen 9+9 tail rebalance."""

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
        if len(self.query_ids) != EXPECTED_QUERY_COUNT:
            raise ValueError(
                f"Corrected legacy batcher requires exactly {EXPECTED_QUERY_COUNT} query IDs; "
                f"got {len(self.query_ids)}"
            )
        if any(
            type(query_id) is not str or not query_id or query_id.strip() != query_id
            for query_id in self.query_ids
        ):
            raise ValueError("Every query_id must be a non-empty, whitespace-trimmed exact string")
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("Corrected legacy batcher query_ids contain duplicates")
        self.experiment_seed = _require_exact_int(
            "experiment_seed", experiment_seed, minimum=0
        )
        if world_size != EXPECTED_WORLD_SIZE:
            raise ValueError(
                f"Corrected legacy batcher requires world_size={EXPECTED_WORLD_SIZE}; "
                f"got {world_size!r}"
            )
        if per_device_batch_size != EXPECTED_PER_DEVICE_BATCH_SIZE:
            raise ValueError(
                "Corrected legacy batcher requires "
                f"per_device_batch_size={EXPECTED_PER_DEVICE_BATCH_SIZE}; "
                f"got {per_device_batch_size!r}"
            )
        self.world_size = EXPECTED_WORLD_SIZE
        self.batch_size = EXPECTED_PER_DEVICE_BATCH_SIZE
        self.epoch = 0

    @property
    def prepared_batches_per_rank(self) -> int:
        return PREPARED_BATCHES_PER_RANK

    @property
    def num_sentinel_rows(self) -> int:
        return SENTINEL_ROW_COUNT

    @property
    def global_real_query_counts(self) -> tuple[int, ...]:
        return (GLOBAL_MICROBATCH_SIZE,) * 25 + TAIL_REAL_QUERY_COUNTS

    @property
    def optimizer_window_real_query_counts(self) -> tuple[int, ...]:
        counts = self.global_real_query_counts
        return tuple(
            sum(counts[offset : offset + GRADIENT_ACCUMULATION_STEPS])
            for offset in range(0, len(counts), GRADIENT_ACCUMULATION_STEPS)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _require_exact_int("epoch", epoch, minimum=0)

    def ordered_real_indices(self) -> list[int]:
        return sorted(
            range(EXPECTED_QUERY_COUNT),
            key=lambda index: (
                _query_order_digest(
                    experiment_seed=self.experiment_seed,
                    epoch=self.epoch,
                    query_id=self.query_ids[index],
                ),
                self.query_ids[index],
            ),
        )

    def _rank_batches_for_global_indices(self, global_indices: Sequence[int]) -> list[list[int]]:
        if not global_indices or len(global_indices) > GLOBAL_MICROBATCH_SIZE:
            raise RuntimeError("Invalid corrected-legacy global microbatch size")
        rank_batches: list[list[int]] = [[] for _ in range(EXPECTED_WORLD_SIZE)]
        for position, query_index in enumerate(global_indices):
            rank_batches[position % EXPECTED_WORLD_SIZE].append(query_index)
        for rank_batch in rank_batches:
            rank_batch.extend(
                [DUMMY_QUERY_INDEX] * (EXPECTED_PER_DEVICE_BATCH_SIZE - len(rank_batch))
            )
        return rank_batches

    def batches(self) -> list[list[int]]:
        ordered = self.ordered_real_indices()
        global_batches: list[list[int]] = [
            ordered[offset : offset + GLOBAL_MICROBATCH_SIZE]
            for offset in range(0, 25 * GLOBAL_MICROBATCH_SIZE, GLOBAL_MICROBATCH_SIZE)
        ]
        tail = ordered[25 * GLOBAL_MICROBATCH_SIZE :]
        if len(tail) != sum(TAIL_REAL_QUERY_COUNTS):
            raise RuntimeError("Corrected-legacy tail must contain exactly 18 real queries")
        global_batches.extend([tail[:9], tail[9:]])

        raw_batches = [
            rank_batch
            for global_batch in global_batches
            for rank_batch in self._rank_batches_for_global_indices(global_batch)
        ]
        self._validate_batches(raw_batches)
        return raw_batches

    def _validate_batches(self, raw_batches: Sequence[Sequence[int]]) -> None:
        if len(raw_batches) != PREPARED_BATCHES_PER_RANK * EXPECTED_WORLD_SIZE:
            raise RuntimeError("Corrected-legacy raw batch count changed")
        if any(len(batch) != EXPECTED_PER_DEVICE_BATCH_SIZE for batch in raw_batches):
            raise RuntimeError("Corrected-legacy batcher produced a non-full local batch")
        flattened = [index for batch in raw_batches for index in batch]
        real_indices = [index for index in flattened if index != DUMMY_QUERY_INDEX]
        if sorted(real_indices) != list(range(EXPECTED_QUERY_COUNT)):
            raise RuntimeError("Corrected-legacy batches do not cover each query exactly once")
        if len(flattened) - len(real_indices) != SENTINEL_ROW_COUNT:
            raise RuntimeError("Corrected-legacy sentinel count changed")

        global_counts: list[int] = []
        for offset in range(0, len(raw_batches), EXPECTED_WORLD_SIZE):
            rank_group = raw_batches[offset : offset + EXPECTED_WORLD_SIZE]
            if any(all(index == DUMMY_QUERY_INDEX for index in batch) for batch in rank_group):
                raise RuntimeError("Corrected-legacy batcher produced an all-sentinel rank")
            global_counts.append(
                sum(index != DUMMY_QUERY_INDEX for batch in rank_group for index in batch)
            )
        if tuple(global_counts) != self.global_real_query_counts:
            raise RuntimeError("Corrected-legacy global real-query counts changed")
        if self.optimizer_window_real_query_counts != (128, 128, 128, 34):
            raise RuntimeError("Corrected-legacy optimizer-window query counts changed")

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches()

    def __len__(self) -> int:
        return PREPARED_BATCHES_PER_RANK * EXPECTED_WORLD_SIZE
