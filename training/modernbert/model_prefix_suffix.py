import torch, torch.nn.functional as F
from torch.distributed.nn.functional import all_gather
import torch.distributed as dist
from deepspeed import comm as ds_comm

def infonce_with_ddp(prefix, pos, temp=0.05):
    device = prefix.device
    pos = pos.to(device)

    # L2‑normalise
    prefix, pos = map(lambda t: F.normalize(t, p=2, dim=-1), (prefix, pos))

    # print("Prefix device:", prefix.device, "Pos device:", pos.device)

    world_size = ds_comm.get_world_size()
    rank = ds_comm.get_rank()

    # all‑gather pos
    pos_buf   = [torch.zeros_like(pos) for _ in range(world_size)]
    ds_comm.all_gather(pos_buf, pos)
    pos_all   = torch.cat(pos_buf, dim=0)

    logits   = prefix @ pos_all.T / temp
    labels   = torch.arange(prefix.size(0), device=device) + rank * prefix.size(0)

    # print("Gathered pos_all device:", pos_all.device)

    pre_buf   = [torch.zeros_like(prefix) for _ in range(world_size)]
    ds_comm.all_gather(pre_buf, prefix)
    prefix_all = torch.cat(pre_buf, dim=0)
    logits_t   = pos @ prefix_all.T / temp

    # print("DEBUG shapes:", prefix_all.shape, logits.shape, labels.shape)
    # print("Labels device:", labels.device, "Logits device:", logits_t.device, "Logits_t device:", logits_t.device)

    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits_t, labels))


class PrefixSuffixModel(torch.nn.Module):
    """
    Container for separate prefix and suffix encoders with a temperature.
    Loss computation is delegated to the ContrastiveTrainer using infonce_with_ddp.
    """
    def __init__(self, args, prefix_enc, suffix_enc):
        super().__init__()
        self.prefix_enc = prefix_enc
        self.suffix_enc = suffix_enc
        self.temp = args.temperature

    def forward(
                self,
                prefix_input_ids=None,
                prefix_attention_mask=None,
                pos_input_ids=None,
                pos_attention_mask=None,
                **unused,
    ):
        return {
                "prefix_input_ids": prefix_input_ids,
                "prefix_attention_mask": prefix_attention_mask,
                "pos_input_ids": pos_input_ids,
                "pos_attention_mask": pos_attention_mask,
            }
