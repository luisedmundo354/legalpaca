from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as torch_nn_func
from transformers import PreTrainedModel


class DualEncoderRetriever(torch.nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        *,
        slot_token_id: int,
        temperature: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.slot_token_id = int(slot_token_id)
        self.temperature = float(temperature)

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        passage_input_ids: Optional[torch.Tensor] = None,
        passage_attention_mask: Optional[torch.Tensor] = None,
        **unused: Dict,
    ) -> Dict[str, torch.Tensor]:
        if unused:
            raise TypeError(f"Unexpected retriever forward inputs: {sorted(unused)}")
        named_inputs = {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "passage_input_ids": passage_input_ids,
            "passage_attention_mask": passage_attention_mask,
        }
        missing = [name for name, value in named_inputs.items() if value is None]
        if missing:
            raise ValueError(
                "DualEncoderRetriever.forward requires one complete query/passage batch; "
                f"missing={missing}"
            )
        return {
            "query_embeddings": self.encode_queries(
                query_input_ids,
                query_attention_mask,
            ),
            "passage_embeddings": self.encode_passages(
                passage_input_ids,
                passage_attention_mask,
            ),
        }

    def encode_queries(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        slot_mask = input_ids.eq(self.slot_token_id)
        slot_idx = slot_mask.float().argmax(dim=1)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        slot_emb = last_hidden[batch_idx, slot_idx, :]
        return torch_nn_func.normalize(slot_emb, p=2, dim=-1)

    def encode_passages(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.clone()
        if mask.size(1) > 0:
            mask[:, 0] = 0
        mask = mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_pooled = summed / denom
        return torch_nn_func.normalize(mean_pooled, p=2, dim=-1)
