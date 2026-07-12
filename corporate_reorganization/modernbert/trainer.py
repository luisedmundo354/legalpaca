from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState

from retriever.eval import evaluate_retrieval
from retriever.losses import build_positive_mask, multi_positive_nce_loss


class MultiPositiveContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        retrieval_eval_config: Optional["RetrievalEvalConfig"] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.retrieval_eval_config = retrieval_eval_config

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            world_size = int(self.args.world_size)
            rank = 0
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()

            if world_size > 1:
                dataloader_params["sampler"] = DistributedSampler(
                    eval_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                dataloader_params["sampler"] = SequentialSampler(eval_dataset)

            dataloader_params["drop_last"] = False

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs: Any,
    ):
        retriever = model.module if hasattr(model, "module") else model
        device = next(retriever.parameters()).device

        query_input_ids = inputs["query_input_ids"].to(device)
        query_attention_mask = inputs["query_attention_mask"].to(device)
        passage_input_ids = inputs["passage_input_ids"].to(device)
        passage_attention_mask = inputs["passage_attention_mask"].to(device)
        passage_id_hashes = inputs["passage_id_hashes"].to(device)
        positive_id_hashes = inputs["positive_id_hashes"].to(device)

        query_embeddings = retriever.encode_queries(query_input_ids, query_attention_mask)
        passage_embeddings = retriever.encode_passages(passage_input_ids, passage_attention_mask)

        passage_embeddings_all = passage_embeddings
        passage_id_hashes_all = passage_id_hashes
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            gathered_passage_embeddings = [torch.zeros_like(passage_embeddings) for _ in range(world_size)]
            dist.all_gather(gathered_passage_embeddings, passage_embeddings)
            gathered_passage_embeddings[rank] = passage_embeddings
            passage_embeddings_all = torch.cat(gathered_passage_embeddings, dim=0)

            gathered_passage_hashes = [torch.zeros_like(passage_id_hashes) for _ in range(world_size)]
            dist.all_gather(gathered_passage_hashes, passage_id_hashes)
            passage_id_hashes_all = torch.cat(gathered_passage_hashes, dim=0)

        logits = (query_embeddings @ passage_embeddings_all.T) / float(retriever.temperature)
        logits = logits.float()
        positive_mask = build_positive_mask(passage_id_hashes_all, positive_id_hashes)

        if not positive_mask.any(dim=1).all():
            raise ValueError("At least one query has no positives in the candidate passage batch")

        loss, per_query_loss = multi_positive_nce_loss(logits, positive_mask)
        loss = loss.to(device)

        if return_outputs:
            return loss, {"loss": loss.detach(), "per_query_loss": per_query_loss}
        return loss

    def training_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)
        return loss.to(self.args.device)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        model = self.model
        retriever = model.module if hasattr(model, "module") else model
        was_training = retriever.training

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self.is_world_process_zero:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"SM_METRIC {name}={float(value)}")

        if self.retrieval_eval_config is None:
            retriever.train(was_training)
            return metrics

        if not self.is_world_process_zero:
            retriever.train(was_training)
            return metrics

        tokenizer = self.tokenizer
        if tokenizer is None:
            retriever.train(was_training)
            return metrics

        retrieval_result = evaluate_retrieval(
            retriever,
            tokenizer,
            processed_dir=self.retrieval_eval_config.processed_dir,
            split=self.retrieval_eval_config.split,
            max_len_query=self.retrieval_eval_config.max_len_query,
            max_len_passage=self.retrieval_eval_config.max_len_passage,
            query_batch_size=self.retrieval_eval_config.query_batch_size,
            passage_batch_size=self.retrieval_eval_config.passage_batch_size,
            ks=self.retrieval_eval_config.ks,
            query_view=self.retrieval_eval_config.query_view,
            regime_name=self.retrieval_eval_config.regime_name,
        )

        retrieval_metrics = dict(retrieval_result.metrics)
        metrics.update(retrieval_metrics)
        self.log(retrieval_metrics)
        for name, value in retrieval_metrics.items():
            if isinstance(value, (int, float)):
                print(f"SM_METRIC {name}={float(value)}")

        retriever.train(was_training)
        return metrics


@dataclass(frozen=True)
class RetrievalEvalConfig:
    processed_dir: Path
    split: str
    max_len_query: int
    max_len_passage: int
    query_batch_size: int
    passage_batch_size: int
    ks: Sequence[int] = (1, 5, 10, 20)
    query_view: str = "structured"
    regime_name: str = "same_case_legacy"


class SetEpochCallback(TrainerCallback):
    def __init__(self, train_dataset: Any):
        self.train_dataset = train_dataset

    def on_epoch_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not hasattr(self.train_dataset, "set_epoch"):
            return
        epoch_idx = int(state.epoch or 0)
        self.train_dataset.set_epoch(epoch_idx)
