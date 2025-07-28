# trainer_contrastive.py
from transformers import Trainer
import torch
from model_prefix_suffix import infonce_with_ddp  # InfoNCE loss with DDP support

class ContrastiveTrainer(Trainer):
    def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
            **kwargs,
    ):

        engine = model
        device = next(engine.parameters()).device

        prefix = {k[7:]: v.to(device) for k, v in inputs.items()
                  if k.startswith("prefix_")}
        pos    = {k[4:]: v.to(device) for k, v in inputs.items()
                  if k.startswith("pos_")}

        pid = prefix.get('input_ids')
        pam = prefix.get('attention_mask')
        print(f"DEBUG shapes: {pid.shape}, {pam.shape}")
        print(f"Sum attention masks: {prefix['attention_mask'].sum()}")
        # Ensure rotary-embedding buffers live on GPU
        try:
            rot_p = engine.module.prefix_enc.model.rotary_emb
            rot_p.cos = rot_p.cos.cuda()
            rot_p.sin = rot_p.sin.cuda()
        except AttributeError:
            pass
        out_p = engine.module.prefix_enc(**prefix, output_hidden_states=True, return_dict=True)

        try:
            rot_s = engine.module.suffix_enc.model.rotary_emb
            rot_s.cos = rot_s.cos.cuda()
            rot_s.sin = rot_s.sin.cuda()
        except AttributeError:
            pass
        out_s = engine.module.suffix_enc(**pos, output_hidden_states=True, return_dict=True)

        print(f"DBG out_p.hidden_states shape: {len(out_p.hidden_states)}")
        print(f"DBG out_s.hidden_states shape: {len(out_s.hidden_states)}")

        hs_p = out_p.hidden_states[-1]
        hs_s = out_s.hidden_states[-1]

        print("Prefix hidden_states[-1].shape:", hs_p.shape)
        print("Suffix hidden_states[-1].shape:", hs_s.shape)

        emb_p = hs_p[:, 0, :]
        emb_s = hs_s[:, 0, :]

        print(f"DBG emb_p/emb_s shapes: {emb_p.shape}, {emb_s.shape}")

        # use the temperature stored on the PrefixSuffixModel instance
        loss = infonce_with_ddp(emb_p, emb_s, temp=model.module.temp)
        loss = loss.to(self.args.device)
        return (loss, {"loss": loss}) if return_outputs else loss
