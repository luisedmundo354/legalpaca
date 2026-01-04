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

