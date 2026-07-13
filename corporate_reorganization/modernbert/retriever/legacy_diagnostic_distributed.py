"""Cross-rank occurrence accounting for corrected legacy diagnostic training."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .distributed import GlobalCandidatePlan


def gather_global_candidate_multiplicities(
    candidate_passage_indices: torch.Tensor,
    candidate_multiplicities: torch.Tensor,
    plan: GlobalCandidatePlan,
    *,
    corpus_size: int,
    expected_occurrences_per_query: int = 64,
    group=None,
) -> torch.Tensor:
    """Return integer occurrence counts aligned with the padded owner table."""

    if type(corpus_size) is not int or corpus_size < 1:
        raise ValueError("corpus_size must be a positive exact int")
    if type(expected_occurrences_per_query) is not int or expected_occurrences_per_query != 64:
        raise ValueError("Corrected legacy occurrence count must be exactly 64")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Corrected legacy multiplicity gathering requires distributed init")
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    if world_size != 4 or plan.rank != rank or plan.world_size != world_size:
        raise RuntimeError("Corrected legacy multiplicity plan requires the same four-rank group")
    if (
        not torch.is_tensor(candidate_passage_indices)
        or candidate_passage_indices.dtype != torch.long
        or candidate_passage_indices.ndim != 2
        or not torch.is_tensor(candidate_multiplicities)
        or candidate_multiplicities.dtype != torch.long
        or candidate_multiplicities.shape != candidate_passage_indices.shape
        or candidate_multiplicities.device != candidate_passage_indices.device
    ):
        raise TypeError("Corrected legacy candidate indices/multiplicities must align as long tensors")
    if candidate_passage_indices.shape[0] < 1 or candidate_passage_indices.shape[1] < 1:
        raise ValueError("Corrected legacy candidate rows must be non-empty")
    valid = candidate_passage_indices.ne(-1)
    expected_valid = torch.arange(
        candidate_passage_indices.shape[1],
        device=candidate_passage_indices.device,
    ).unsqueeze(0) < valid.sum(dim=1).unsqueeze(1)
    if not torch.equal(valid, expected_valid):
        raise ValueError("Corrected legacy candidate indices require suffix-only -1 padding")
    if not torch.equal(candidate_multiplicities.ne(0), valid):
        raise ValueError("Corrected legacy multiplicity zero padding does not align with indices")
    if (candidate_multiplicities[valid] < 1).any():
        raise ValueError("Corrected legacy real candidate multiplicities must be positive")
    if not candidate_multiplicities.sum(dim=1).eq(expected_occurrences_per_query).all():
        raise ValueError("Each corrected legacy candidate row must represent 64 occurrences")
    if ((candidate_passage_indices < -1) | (candidate_passage_indices >= corpus_size)).any():
        raise ValueError("Corrected legacy candidate index is outside the corpus")
    for row_index in range(candidate_passage_indices.shape[0]):
        row = candidate_passage_indices[row_index, valid[row_index]]
        if torch.unique(row).numel() != row.numel():
            raise ValueError(f"Corrected legacy candidate row {row_index} is not unique")

    local_dense = torch.zeros(
        (corpus_size,),
        dtype=torch.long,
        device=candidate_passage_indices.device,
    )
    local_dense.scatter_add_(
        0,
        candidate_passage_indices[valid],
        candidate_multiplicities[valid],
    )
    global_dense = local_dense.clone()
    dist.all_reduce(global_dense, op=dist.ReduceOp.SUM, group=group)

    aligned = torch.zeros_like(plan.gathered_passage_indices)
    aligned[plan.valid_passage_mask] = global_dense[
        plan.gathered_passage_indices[plan.valid_passage_mask]
    ]
    if (aligned[plan.valid_passage_mask] < 1).any():
        raise RuntimeError("A globally owned corrected legacy passage has zero multiplicity")
    if not aligned[~plan.valid_passage_mask].eq(0).all():
        raise RuntimeError("Corrected legacy multiplicity padding is nonzero")
    local_occurrences = torch.tensor(
        [candidate_multiplicities.sum().item()],
        dtype=torch.long,
        device=candidate_passage_indices.device,
    )
    dist.all_reduce(local_occurrences, op=dist.ReduceOp.SUM, group=group)
    if int(aligned.sum().item()) != int(local_occurrences.item()):
        raise RuntimeError("Corrected legacy global multiplicities lost candidate occurrences")
    return aligned


__all__ = ["gather_global_candidate_multiplicities"]
