import torch
import torch.nn.functional as F
from typing import Optional

class PrefixSuffixModel(torch.nn.Module):
    """
    Inference-friendly wrapper holding two towers.
    - No DeepSpeed imports here.
    - Temperature kept optional; not used at inference.
    - Provides encode_{prefix,suffix} helpers returning [B, H] tensors.
    """
    def __init__(
            self,
            args: Optional[object] = None,
            prefix_enc: Optional[torch.nn.Module] = None,
            suffix_enc: Optional[torch.nn.Module] = None,
            temperature: Optional[float] = None,
    ):
        super().__init__()
        self.prefix_enc = prefix_enc
        self.suffix_enc = suffix_enc
        if temperature is None:
            temperature = float(getattr(args, "temperature", 0.01)) if args is not None else 0.01
        self.temp = float(temperature)

    # ---- Inference helpers -------------------------------------------------
    @torch.no_grad()
    def encode_prefix(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pooling: str = "mean",     # "mean" or "cls"
            normalize: bool = True,
    ) -> torch.Tensor:
        return self._encode(self.prefix_enc, input_ids, attention_mask, pooling, normalize)

    @torch.no_grad()
    def encode_suffix(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pooling: str = "mean",
            normalize: bool = True,
    ) -> torch.Tensor:
        return self._encode(self.suffix_enc, input_ids, attention_mask, pooling, normalize)

    def _encode(
            self,
            enc_model: torch.nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pooling: str,
            normalize: bool,
    ) -> torch.Tensor:
        """
        Returns a [batch, hidden] embedding tensor.
        Works whether the tower is an AutoModelForMaskedLM (has .model)
        or a bare encoder (no .model).
        """
        # If you used AutoModelForMaskedLM, the encoder lives at .model
        base = getattr(enc_model, "model", enc_model)

        outputs = base(input_ids=input_ids, attention_mask=attention_mask)
        # Standard token embeddings from HF models:
        # shape [B, T, H]
        last_hidden = outputs.last_hidden_state  # documented in Transformers "Model outputs". :contentReference[oaicite:2]{index=2}

        if pooling.lower() == "cls":
            pooled = last_hidden[:, 0, :]  # take [CLS]
        else:
            # mean-pool over valid tokens only (mask==1)
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
            summed = (last_hidden * mask).sum(dim=1)                  # [B,H]
            denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B,1]
            pooled = summed / denom                                   # [B,H]
            # (This is the typical sentence-embeddings recipe.) :contentReference[oaicite:3]{index=3}

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled
