from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as autograd_all_gather


INVALID_PASSAGE_INDEX = -1


@dataclass(frozen=True)
class GlobalCandidatePlan:
    """Identical cross-rank ownership plan for one controlled microbatch."""

    global_unique_indices: torch.Tensor
    local_owned_indices: torch.Tensor
    padded_owner_indices: torch.Tensor
    gathered_passage_indices: torch.Tensor
    valid_passage_mask: torch.Tensor
    rank: int
    world_size: int
    max_owned_count: int


def _require_distributed(group=None) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Controlled passage sharing requires an initialized process group")
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if world_size < 2:
        raise RuntimeError("Controlled passage sharing requires world_size >= 2")
    return rank, world_size


def _validate_padded_candidate_rows(
    candidate_indices: torch.Tensor,
    *,
    corpus_size: int,
) -> torch.Tensor:
    if type(corpus_size) is not int or corpus_size < 1:
        raise ValueError("corpus_size must be a positive exact int")
    if not torch.is_tensor(candidate_indices):
        raise TypeError("candidate_indices must be a tensor")
    if candidate_indices.dtype != torch.long or candidate_indices.ndim != 2:
        raise TypeError(
            "candidate_indices must be a rank-2 torch.long tensor; "
            f"got dtype={candidate_indices.dtype}, shape={tuple(candidate_indices.shape)}"
        )
    if candidate_indices.shape[0] < 1 or candidate_indices.shape[1] < 1:
        raise ValueError(f"candidate_indices must be non-empty; got {tuple(candidate_indices.shape)}")
    if ((candidate_indices < INVALID_PASSAGE_INDEX) | (candidate_indices >= corpus_size)).any():
        raise ValueError(
            f"candidate_indices must contain only -1 padding or values in [0, {corpus_size})"
        )

    valid = candidate_indices.ne(INVALID_PASSAGE_INDEX)
    valid_counts = valid.sum(dim=1)
    expected_valid = torch.arange(
        candidate_indices.shape[1],
        device=candidate_indices.device,
    ).unsqueeze(0) < valid_counts.unsqueeze(1)
    if not torch.equal(valid, expected_valid):
        raise ValueError("candidate_indices must use contiguous suffix-only -1 padding")
    if (valid_counts < 1).any():
        raise ValueError("Every local query must contribute at least one candidate index")

    for row_index in range(candidate_indices.shape[0]):
        row = candidate_indices[row_index, : int(valid_counts[row_index].item())]
        if torch.unique(row).numel() != row.numel():
            raise ValueError(f"candidate_indices row {row_index} contains duplicates")
    return torch.unique(candidate_indices[valid], sorted=True)


def _all_gather_unique_index_vectors(
    local_unique: torch.Tensor,
    *,
    group=None,
) -> tuple[torch.Tensor, ...]:
    _, world_size = _require_distributed(group)
    local_count = torch.tensor([local_unique.numel()], dtype=torch.long, device=local_unique.device)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count, group=group)
    counts = [int(count.item()) for count in gathered_counts]
    if any(count < 1 for count in counts):
        raise RuntimeError(f"Every rank must contribute candidates; gathered counts={counts}")

    max_count = max(counts)
    padded = torch.full(
        (max_count,),
        INVALID_PASSAGE_INDEX,
        dtype=torch.long,
        device=local_unique.device,
    )
    padded[: local_unique.numel()] = local_unique
    gathered_padded = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_padded, padded, group=group)

    gathered: list[torch.Tensor] = []
    for source_rank, (row, count) in enumerate(zip(gathered_padded, counts)):
        if not row[count:].eq(INVALID_PASSAGE_INDEX).all():
            raise RuntimeError(f"Rank {source_rank} candidate gather has non-padding after count")
        values = row[:count]
        if (values < 0).any() or torch.unique(values).numel() != values.numel():
            raise RuntimeError(f"Rank {source_rank} candidate gather is not a unique valid vector")
        if values.numel() > 1 and not (values[1:] > values[:-1]).all():
            raise RuntimeError(f"Rank {source_rank} candidate gather is not strictly sorted")
        gathered.append(values)
    return tuple(gathered)


def build_global_candidate_plan(
    candidate_indices: torch.Tensor,
    *,
    corpus_size: int,
    group=None,
) -> GlobalCandidatePlan:
    """Gather, globally deduplicate, and assign balanced deterministic owners."""

    rank, world_size = _require_distributed(group)
    local_unique = _validate_padded_candidate_rows(
        candidate_indices,
        corpus_size=corpus_size,
    )
    gathered_local_unique = _all_gather_unique_index_vectors(local_unique, group=group)
    global_unique = torch.unique(torch.cat(gathered_local_unique), sorted=True)
    if global_unique.numel() < 1:
        raise RuntimeError("Globally unique candidate table is empty")
    if (global_unique < 0).any() or (global_unique >= corpus_size).any():
        raise RuntimeError("Globally unique candidate table contains an out-of-range index")

    max_owned_count = (global_unique.numel() + world_size - 1) // world_size
    local_owned = global_unique[rank::world_size]
    padded_owned = torch.full(
        (max_owned_count,),
        INVALID_PASSAGE_INDEX,
        dtype=torch.long,
        device=candidate_indices.device,
    )
    padded_owned[: local_owned.numel()] = local_owned

    gathered_owned_rows = [torch.empty_like(padded_owned) for _ in range(world_size)]
    dist.all_gather(gathered_owned_rows, padded_owned, group=group)
    owner_matrix = torch.stack(gathered_owned_rows, dim=0)
    for owner_rank in range(world_size):
        expected = torch.full_like(padded_owned, INVALID_PASSAGE_INDEX)
        expected_values = global_unique[owner_rank::world_size]
        expected[: expected_values.numel()] = expected_values
        if not torch.equal(owner_matrix[owner_rank], expected):
            raise RuntimeError(f"Rank {owner_rank} did not follow the frozen round-robin owner plan")

    gathered_passage_indices = owner_matrix.reshape(-1)
    valid_passage_mask = gathered_passage_indices.ne(INVALID_PASSAGE_INDEX)
    real_gathered = gathered_passage_indices[valid_passage_mask]
    if real_gathered.numel() != global_unique.numel():
        raise RuntimeError("Owner table changed the number of globally unique candidates")
    if not torch.equal(torch.sort(real_gathered).values, global_unique):
        raise RuntimeError("Owner table does not contain every globally unique candidate exactly once")

    return GlobalCandidatePlan(
        global_unique_indices=global_unique,
        local_owned_indices=local_owned,
        padded_owner_indices=padded_owned,
        gathered_passage_indices=gathered_passage_indices,
        valid_passage_mask=valid_passage_mask,
        rank=rank,
        world_size=world_size,
        max_owned_count=max_owned_count,
    )


def gather_owned_embeddings(
    owned_embeddings: torch.Tensor,
    plan: GlobalCandidatePlan,
    *,
    group=None,
) -> torch.Tensor:
    """Autograd-gather owner embeddings in exact alignment with the plan."""

    rank, world_size = _require_distributed(group)
    if rank != plan.rank or world_size != plan.world_size:
        raise RuntimeError("GlobalCandidatePlan belongs to a different process group")
    if not torch.is_tensor(owned_embeddings) or owned_embeddings.ndim != 2:
        raise TypeError("owned_embeddings must be a rank-2 tensor")
    if not owned_embeddings.is_floating_point():
        raise TypeError("owned_embeddings must have a floating dtype")
    if owned_embeddings.device != plan.local_owned_indices.device:
        raise ValueError("owned_embeddings and passage-index plan must share a device")
    if owned_embeddings.shape[0] != plan.local_owned_indices.numel():
        raise ValueError(
            f"Owned embedding rows={owned_embeddings.shape[0]} do not match "
            f"owned passage count={plan.local_owned_indices.numel()}"
        )
    if owned_embeddings.shape[1] < 1:
        raise ValueError("owned_embeddings must have a positive embedding dimension")
    if not owned_embeddings.requires_grad:
        raise ValueError("owned_embeddings must retain an autograd graph")
    if not torch.isfinite(owned_embeddings).all():
        raise FloatingPointError("owned_embeddings contains non-finite values")

    padding_rows = plan.max_owned_count - owned_embeddings.shape[0]
    if padding_rows < 0:
        raise RuntimeError("Owned embedding rows exceed the frozen padded owner width")
    padding = owned_embeddings.new_zeros((padding_rows, owned_embeddings.shape[1]))
    padded_embeddings = torch.cat((owned_embeddings, padding), dim=0)
    if tuple(padded_embeddings.shape) != (
        plan.max_owned_count,
        owned_embeddings.shape[1],
    ):
        raise RuntimeError("Padded owner embedding shape changed unexpectedly")

    gathered = autograd_all_gather(padded_embeddings, group=group)
    if type(gathered) is not tuple or len(gathered) != world_size:
        raise RuntimeError("Autograd all_gather returned an unexpected result")
    passage_embeddings = torch.stack(gathered, dim=0).reshape(
        world_size * plan.max_owned_count,
        owned_embeddings.shape[1],
    )
    if passage_embeddings.shape[0] != plan.gathered_passage_indices.numel():
        raise RuntimeError("Gathered embedding and passage-index tables are misaligned")
    return passage_embeddings
