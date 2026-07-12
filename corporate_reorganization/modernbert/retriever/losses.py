from __future__ import annotations

from typing import Tuple

import torch


def build_positive_mask(
    passage_id_hashes: torch.Tensor,
    positive_id_hashes: torch.Tensor,
) -> torch.Tensor:
    positive_id_hashes = positive_id_hashes.to(device=passage_id_hashes.device)
    passage_id_hashes = passage_id_hashes.to(device=positive_id_hashes.device)

    valid_pos = positive_id_hashes.ne(-1).unsqueeze(-1)
    same_id = positive_id_hashes.unsqueeze(-1).eq(passage_id_hashes.view(1, 1, -1))
    return (same_id & valid_pos).any(dim=1)


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
