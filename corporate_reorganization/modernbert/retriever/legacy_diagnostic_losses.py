from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollapsedCandidateOccurrences:
    unique_passage_indices: tuple[int, ...]
    multiplicities: tuple[int, ...]
    total_occurrences: int


def collapse_candidate_occurrence_indices(
    candidate_rows: Sequence[Sequence[int]],
    *,
    corpus_size: int,
) -> CollapsedCandidateOccurrences:
    if type(corpus_size) is not int or corpus_size < 1:
        raise ValueError("corpus_size must be a positive exact int")
    if not isinstance(candidate_rows, Sequence) or not candidate_rows:
        raise ValueError("candidate_rows must be a non-empty sequence")

    counts: Counter[int] = Counter()
    for row_index, row in enumerate(candidate_rows):
        if not isinstance(row, Sequence) or not row:
            raise ValueError(f"candidate_rows[{row_index}] must be a non-empty sequence")
        for column_index, passage_index in enumerate(row):
            if type(passage_index) is not int:
                raise TypeError(
                    f"candidate_rows[{row_index}][{column_index}] must be an exact int"
                )
            if passage_index < 0 or passage_index >= corpus_size:
                raise ValueError(
                    f"candidate passage index {passage_index} is outside [0, {corpus_size})"
                )
            counts[passage_index] += 1

    unique_indices = tuple(sorted(counts))
    multiplicities = tuple(counts[index] for index in unique_indices)
    total = sum(multiplicities)
    if total < 1 or any(multiplicity < 1 for multiplicity in multiplicities):
        raise RuntimeError("Collapsed corrected-legacy occurrence table is internally invalid")
    return CollapsedCandidateOccurrences(unique_indices, multiplicities, total)


def multiplicity_aware_multi_positive_nce_loss_sum(
    unique_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    occurrence_multiplicities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NCE over unique columns, exactly representing repeated occurrence columns.

    Adding log(m) to a unique column is the log-sum-exp identity for m identical
    explicit columns. The weight is applied to both the all-candidate denominator
    and the query-specific positive numerator.
    """

    if not torch.is_tensor(unique_logits) or unique_logits.ndim != 2:
        raise TypeError("unique_logits must be a rank-2 tensor")
    if not unique_logits.is_floating_point():
        raise TypeError("unique_logits must have a floating dtype")
    if unique_logits.shape[0] < 1 or unique_logits.shape[1] < 1:
        raise ValueError("unique_logits must be non-empty")
    if not torch.isfinite(unique_logits).all():
        raise FloatingPointError("unique_logits contains non-finite values")
    if not torch.is_tensor(positive_mask) or positive_mask.dtype != torch.bool:
        raise TypeError("positive_mask must be a torch.bool tensor")
    if positive_mask.shape != unique_logits.shape:
        raise ValueError("positive_mask must have exactly the unique_logits shape")
    if positive_mask.device != unique_logits.device:
        raise ValueError("positive_mask and unique_logits must share a device")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every query must have at least one positive unique column")

    if not torch.is_tensor(occurrence_multiplicities):
        raise TypeError("occurrence_multiplicities must be a tensor")
    if occurrence_multiplicities.dtype != torch.long or occurrence_multiplicities.ndim != 1:
        raise TypeError("occurrence_multiplicities must be a rank-1 torch.long tensor")
    if occurrence_multiplicities.numel() != unique_logits.shape[1]:
        raise ValueError("occurrence_multiplicities must align with unique logit columns")
    if occurrence_multiplicities.device != unique_logits.device:
        raise ValueError("occurrence_multiplicities and unique_logits must share a device")
    if (occurrence_multiplicities < 1).any():
        raise ValueError("Every unique candidate column must have multiplicity >= 1")

    log_multiplicity = occurrence_multiplicities.to(dtype=unique_logits.dtype).log()
    occurrence_weighted_logits = unique_logits + log_multiplicity.unsqueeze(0)
    positive_logits = occurrence_weighted_logits.masked_fill(~positive_mask, float("-inf"))
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(occurrence_weighted_logits, dim=1)
    per_query_loss = -(numerator - denominator)
    if not torch.isfinite(per_query_loss).all():
        raise FloatingPointError("Multiplicity-aware corrected-legacy loss is non-finite")
    return per_query_loss.sum(), per_query_loss.detach()
