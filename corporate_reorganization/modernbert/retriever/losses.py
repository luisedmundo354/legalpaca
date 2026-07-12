from __future__ import annotations

from typing import Tuple

import torch


INVALID_PASSAGE_INDEX = -1


def build_positive_mask(
    passage_id_hashes: torch.Tensor,
    positive_id_hashes: torch.Tensor,
) -> torch.Tensor:
    positive_id_hashes = positive_id_hashes.to(device=passage_id_hashes.device)
    passage_id_hashes = passage_id_hashes.to(device=positive_id_hashes.device)

    valid_pos = positive_id_hashes.ne(-1).unsqueeze(-1)
    same_id = positive_id_hashes.unsqueeze(-1).eq(passage_id_hashes.view(1, 1, -1))
    return (same_id & valid_pos).any(dim=1)


def build_index_positive_mask(
    passage_indices: torch.Tensor,
    positive_indices: torch.Tensor,
    valid_passage_mask: torch.Tensor,
) -> torch.Tensor:
    """Build query-specific targets over a padded globally unique index table."""

    if passage_indices.dtype != torch.long or passage_indices.ndim != 1:
        raise TypeError("passage_indices must be a rank-1 torch.long tensor")
    if positive_indices.dtype != torch.long or positive_indices.ndim != 2:
        raise TypeError("positive_indices must be a rank-2 torch.long tensor")
    if valid_passage_mask.dtype != torch.bool or valid_passage_mask.ndim != 1:
        raise TypeError("valid_passage_mask must be a rank-1 torch.bool tensor")
    if passage_indices.shape != valid_passage_mask.shape:
        raise ValueError("passage_indices and valid_passage_mask must have the same shape")
    if passage_indices.device != positive_indices.device or passage_indices.device != valid_passage_mask.device:
        raise ValueError("Passage indices, positive indices, and valid mask must share a device")
    if passage_indices.numel() < 1 or positive_indices.shape[0] < 1 or positive_indices.shape[1] < 1:
        raise ValueError("Index positive-mask tensors must be non-empty")
    if not valid_passage_mask.any():
        raise ValueError("Global passage table has no valid entries")
    if not torch.equal(valid_passage_mask, passage_indices.ne(INVALID_PASSAGE_INDEX)):
        raise ValueError("valid_passage_mask does not exactly identify -1 passage padding")
    if (passage_indices[valid_passage_mask] < 0).any():
        raise ValueError("Valid passage entries must be non-negative")
    if torch.unique(passage_indices[valid_passage_mask]).numel() != int(valid_passage_mask.sum().item()):
        raise ValueError("Global valid passage indices must be unique")
    if (positive_indices < INVALID_PASSAGE_INDEX).any():
        raise ValueError("positive_indices may contain only non-negative values and -1 padding")

    valid_positive = positive_indices.ne(INVALID_PASSAGE_INDEX)
    positive_counts = valid_positive.sum(dim=1)
    if (positive_counts < 1).any():
        raise ValueError("Every valid query must contain at least one all-gold index")
    expected_positive = torch.arange(
        positive_indices.shape[1],
        device=positive_indices.device,
    ).unsqueeze(0) < positive_counts.unsqueeze(1)
    if not torch.equal(valid_positive, expected_positive):
        raise ValueError("positive_indices must use contiguous suffix-only -1 padding")
    for row_index in range(positive_indices.shape[0]):
        row = positive_indices[row_index, : int(positive_counts[row_index].item())]
        if torch.unique(row).numel() != row.numel():
            raise ValueError(f"positive_indices row {row_index} contains duplicates")

    matches = positive_indices.unsqueeze(-1).eq(passage_indices.view(1, 1, -1))
    positive_mask = (
        matches
        & valid_positive.unsqueeze(-1)
        & valid_passage_mask.view(1, 1, -1)
    ).any(dim=1)
    if not positive_mask.any(dim=1).all():
        raise ValueError("At least one query has no positive in the globally unique candidate table")
    return positive_mask


def multi_positive_nce_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    per_query_loss = -(numerator - denominator)
    return per_query_loss.mean(), per_query_loss.detach()


def multi_positive_nce_loss_sum(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or positive_mask.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError(
            f"logits and positive_mask must be equal rank-2 tensors; "
            f"got logits={tuple(logits.shape)}, positive_mask={tuple(positive_mask.shape)}"
        )
    if logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError(f"NCE tensors must be non-empty; got shape={tuple(logits.shape)}")
    if positive_mask.dtype != torch.bool:
        raise TypeError(f"positive_mask must have dtype=torch.bool; got {positive_mask.dtype}")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every valid query must have at least one positive passage")

    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    per_query_loss = -(numerator - denominator)
    if not torch.isfinite(per_query_loss).all():
        raise FloatingPointError("Non-finite per-query retrieval loss")
    return per_query_loss.sum(), per_query_loss.detach()


def masked_multi_positive_nce_loss_sum(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_passage_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if valid_passage_mask.dtype != torch.bool or valid_passage_mask.ndim != 1:
        raise TypeError("valid_passage_mask must be a rank-1 torch.bool tensor")
    if logits.ndim != 2 or logits.shape[1] != valid_passage_mask.numel():
        raise ValueError("valid_passage_mask width must equal the logits passage dimension")
    if valid_passage_mask.device != logits.device:
        raise ValueError("valid_passage_mask and logits must share a device")
    if positive_mask.shape != logits.shape or positive_mask.device != logits.device:
        raise ValueError("positive_mask must match the logits shape and device")
    if (positive_mask & ~valid_passage_mask.unsqueeze(0)).any():
        raise ValueError("An invalid padded passage column was marked positive")
    if not valid_passage_mask.any():
        raise ValueError("At least one denominator passage must be valid")

    masked_logits = logits.masked_fill(~valid_passage_mask.unsqueeze(0), float("-inf"))
    return multi_positive_nce_loss_sum(masked_logits, positive_mask)
