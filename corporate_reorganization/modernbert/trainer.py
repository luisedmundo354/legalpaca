from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
from accelerate.utils import DistributedType

from retriever.batching import GlobalQueryBatchSampler, SentinelQueryDataset
from retriever.data import PassageIndexTable
from retriever.distributed import build_global_candidate_plan, gather_owned_embeddings
from retriever.eval import evaluate_retrieval
from retriever.losses import (
    build_index_positive_mask,
    build_positive_mask,
    masked_multi_positive_nce_loss_sum,
    multi_positive_nce_loss,
)
from retriever.traces import CandidateTraceStore


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


class ControlledRetrievalTrainer(MultiPositiveContrastiveTrainer):
    EXPECTED_QUERIES = 294
    EXPECTED_WORLD_SIZE = 4
    EXPECTED_PER_DEVICE_BATCH = 4
    EXPECTED_GRADIENT_ACCUMULATION = 8
    EXPECTED_EPOCHS = 20
    EXPECTED_PREPARED_BATCHES = 19
    EXPECTED_WINDOW_MICROBATCHES = (8, 8, 3)
    EXPECTED_WINDOW_VALID_QUERIES = (128, 128, 38)
    EXPECTED_UPDATES_PER_EPOCH = 3
    EXPECTED_TOTAL_UPDATES = 60

    def __init__(
        self,
        *args: Any,
        experiment_seed: int,
        passage_index_table: PassageIndexTable,
        max_len_passage: int,
        **kwargs: Any,
    ) -> None:
        if type(experiment_seed) is not int or experiment_seed < 0:
            raise ValueError("experiment_seed must be a non-negative exact int")
        if not isinstance(passage_index_table, PassageIndexTable):
            raise TypeError("passage_index_table must be a PassageIndexTable")
        if len(passage_index_table) != 5_286:
            raise RuntimeError(
                f"Controlled corpus passage index must contain exactly 5,286 rows; "
                f"got {len(passage_index_table)}"
            )
        if type(max_len_passage) is not int or max_len_passage != 500:
            raise RuntimeError("Controlled max passage length must be the frozen exact value 500")
        self.experiment_seed = experiment_seed
        self.passage_index_table = passage_index_table
        self.max_len_passage = max_len_passage
        self._global_batch_sampler: GlobalQueryBatchSampler | None = None
        self._window_epoch: int | None = None
        self._window_index = 0
        super().__init__(*args, **kwargs)

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError("Controlled retrieval training requires the frozen DeepSpeed runtime")
        if self.accelerator.num_processes != self.EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"Controlled retrieval requires world_size={self.EXPECTED_WORLD_SIZE}; "
                f"got {self.accelerator.num_processes}"
            )
        if self.args.per_device_train_batch_size != self.EXPECTED_PER_DEVICE_BATCH:
            raise RuntimeError("Controlled per-device query batch must be exactly 4")
        if self.args.gradient_accumulation_steps != self.EXPECTED_GRADIENT_ACCUMULATION:
            raise RuntimeError("Controlled gradient accumulation must be exactly 8")
        if self.args.dataloader_num_workers != 0:
            raise RuntimeError("Controlled retrieval requires dataloader_num_workers=0")
        if self.args.dataloader_drop_last:
            raise RuntimeError("Controlled retrieval forbids dataloader_drop_last")
        if not self.model_accepts_loss_kwargs:
            raise RuntimeError(
                "Transformers 4.49 must detect the retriever loss-kwargs contract; "
                "the controlled training_step relies on its no-extra-GAS-division path"
            )
        if self.processing_class is None:
            raise RuntimeError("Controlled retrieval requires an explicit local tokenizer")
        self._candidate_trace_store = CandidateTraceStore(
            Path(self.args.output_dir),
            passage_index_table=self.passage_index_table,
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Controlled retrieval training requires a train_dataset")
        if len(self.train_dataset) != self.EXPECTED_QUERIES:
            raise RuntimeError(
                f"Controlled fold must contain {self.EXPECTED_QUERIES} training queries; "
                f"got {len(self.train_dataset)}"
            )
        queries = getattr(self.train_dataset, "queries", None)
        if not isinstance(queries, list) or len(queries) != len(self.train_dataset):
            raise TypeError("Controlled train_dataset must expose one ordered queries list")
        query_ids = [query.query_id for query in queries]

        batch_sampler = GlobalQueryBatchSampler(
            query_ids,
            experiment_seed=self.experiment_seed,
            world_size=self.EXPECTED_WORLD_SIZE,
            per_device_batch_size=self.EXPECTED_PER_DEVICE_BATCH,
        )
        # Accelerate 1.4's BatchSamplerShard does not forward set_epoch() to a
        # custom batch sampler. DataLoaderShard does forward it to the dataset,
        # so this wrapper deliberately updates both scientific sampling and the
        # global query order from that one epoch notification.
        wrapped_dataset = SentinelQueryDataset(
            self.train_dataset,
            epoch_target=batch_sampler,
        )
        dataloader = DataLoader(
            wrapped_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
        )
        prepared = self.accelerator.prepare(dataloader)
        if len(batch_sampler) != 76:
            raise RuntimeError(f"Expected 76 raw rank-ordered batches; got {len(batch_sampler)}")
        if len(prepared) != self.EXPECTED_PREPARED_BATCHES:
            raise RuntimeError(
                f"Prepared dataloader must have {self.EXPECTED_PREPARED_BATCHES} batches/rank; "
                f"got {len(prepared)}"
            )
        self._global_batch_sampler = batch_sampler
        return prepared

    @staticmethod
    def _forward_controlled_batch(
        model,
        *,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            passage_input_ids=passage_input_ids,
            passage_attention_mask=passage_attention_mask,
        )
        if type(outputs) is not dict or set(outputs) != {
            "query_embeddings",
            "passage_embeddings",
        }:
            raise TypeError(
                "Controlled DeepSpeed forward must return exactly query_embeddings and "
                "passage_embeddings"
            )
        query_embeddings = outputs["query_embeddings"]
        passage_embeddings = outputs["passage_embeddings"]
        if (
            not torch.is_tensor(query_embeddings)
            or query_embeddings.ndim != 2
            or not torch.is_tensor(passage_embeddings)
            or passage_embeddings.ndim != 2
            or query_embeddings.shape[1] != passage_embeddings.shape[1]
        ):
            raise TypeError("Controlled DeepSpeed forward returned invalid embedding tensors")
        return query_embeddings, passage_embeddings

    @staticmethod
    def _tokenize_owned_passages(
        tokenizer,
        passage_texts: list[str],
        *,
        max_len_passage: int,
    ) -> Mapping[str, torch.Tensor]:
        original_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = "right"
        try:
            tokens = tokenizer(
                passage_texts,
                truncation=True,
                max_length=max_len_passage,
                padding=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.truncation_side = original_truncation_side
        if (
            not isinstance(tokens, Mapping)
            or not torch.is_tensor(tokens.get("input_ids"))
            or not torch.is_tensor(tokens.get("attention_mask"))
            or tokens["input_ids"].shape != tokens["attention_mask"].shape
        ):
            raise TypeError("Passage tokenizer returned invalid input tensors")
        return tokens

    def set_initial_training_values(self, args, dataloader, total_train_batch_size: int):
        if args.max_steps >= 0:
            raise RuntimeError("Controlled retrieval forbids max_steps; use exactly 20 complete epochs")
        if float(args.num_train_epochs) != float(self.EXPECTED_EPOCHS):
            raise RuntimeError(f"Controlled retrieval requires exactly {self.EXPECTED_EPOCHS} epochs")
        if len(dataloader) != self.EXPECTED_PREPARED_BATCHES:
            raise RuntimeError("Controlled prepared dataloader length changed before scheduling")
        if total_train_batch_size != 128:
            raise RuntimeError(
                f"Controlled nominal global query batch must be 128; got {total_train_batch_size}"
            )
        num_examples = self.num_examples(dataloader)
        if num_examples != self.EXPECTED_QUERIES:
            raise RuntimeError(
                f"Trainer reports {num_examples} examples; expected {self.EXPECTED_QUERIES}"
            )
        return (
            self.EXPECTED_EPOCHS,
            self.EXPECTED_UPDATES_PER_EPOCH,
            self.EXPECTED_QUERIES,
            self.EXPECTED_QUERIES * self.EXPECTED_EPOCHS,
            True,
            self.EXPECTED_PREPARED_BATCHES,
            self.EXPECTED_TOTAL_UPDATES,
        )

    @staticmethod
    def _exact_scalar_count(batch: Dict[str, Any]) -> int:
        if "valid_query_count" not in batch:
            raise KeyError("Controlled batch is missing valid_query_count")
        value = batch["valid_query_count"]
        if not torch.is_tensor(value) or value.numel() != 1 or value.dtype != torch.long:
            raise TypeError("valid_query_count must be one torch.long scalar")
        count = int(value.item())
        if count < 1 or count > ControlledRetrievalTrainer.EXPECTED_PER_DEVICE_BATCH:
            raise ValueError(f"Invalid local valid_query_count={count}")
        return count

    def _reduce_window_counts(self, *, local_valid_count: int, local_microbatches: int) -> int:
        device = self.args.device
        micro_min = torch.tensor(local_microbatches, dtype=torch.long, device=device)
        micro_max = micro_min.clone()
        dist.all_reduce(micro_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(micro_max, op=dist.ReduceOp.MAX)
        if int(micro_min.item()) != local_microbatches or int(micro_max.item()) != local_microbatches:
            raise RuntimeError("Ranks fetched different numbers of microbatches for one optimizer window")

        global_valid = torch.tensor(local_valid_count, dtype=torch.long, device=device)
        dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
        global_count = int(global_valid.item())
        if global_count < 1:
            raise RuntimeError("Optimizer window contains no valid queries")
        return global_count

    def get_batch_samples(self, epoch_iterator, num_batches):
        del num_batches
        batch_samples = []
        for _ in range(self.EXPECTED_GRADIENT_ACCUMULATION):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        if not batch_samples:
            raise RuntimeError("Trainer requested an empty controlled optimizer window")

        local_valid_count = sum(self._exact_scalar_count(batch) for batch in batch_samples)
        global_valid_count = self._reduce_window_counts(
            local_valid_count=local_valid_count,
            local_microbatches=len(batch_samples),
        )

        if self._global_batch_sampler is None:
            raise RuntimeError("Controlled global batch sampler was not initialized")
        epoch = self._global_batch_sampler.epoch
        if self._window_epoch != epoch:
            self._window_epoch = epoch
            self._window_index = 0
        if self._window_index >= len(self.EXPECTED_WINDOW_VALID_QUERIES):
            raise RuntimeError(f"Observed too many optimizer windows in epoch={epoch}")
        expected_microbatches = self.EXPECTED_WINDOW_MICROBATCHES[self._window_index]
        expected_valid = self.EXPECTED_WINDOW_VALID_QUERIES[self._window_index]
        if len(batch_samples) != expected_microbatches or global_valid_count != expected_valid:
            raise RuntimeError(
                f"Controlled optimizer window {self._window_index} in epoch={epoch} changed: "
                f"microbatches={len(batch_samples)} expected={expected_microbatches}, "
                f"valid_queries={global_valid_count} expected={expected_valid}"
            )

        for index, batch in enumerate(batch_samples):
            batch["global_window_valid_count"] = global_valid_count
            batch["is_window_end"] = index == len(batch_samples) - 1
        self._window_index += 1
        return batch_samples, global_valid_count

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs: Any,
    ):
        del kwargs
        if type(num_items_in_batch) is not int or num_items_in_batch < 1:
            raise TypeError("Controlled compute_loss requires a positive exact global window count")
        if inputs.get("global_window_valid_count") != num_items_in_batch:
            raise RuntimeError("Batch/global optimizer-window counts disagree")

        retriever = model.module if hasattr(model, "module") else model
        device = next(retriever.parameters()).device
        query_input_ids = inputs["query_input_ids"].to(device)
        query_attention_mask = inputs["query_attention_mask"].to(device)
        candidate_passage_indices = inputs["candidate_passage_indices"].to(device)
        positive_passage_indices = inputs["positive_passage_indices"].to(device)

        local_valid_count = self._exact_scalar_count(inputs)
        if (
            query_input_ids.ndim != 2
            or query_attention_mask.shape != query_input_ids.shape
            or query_input_ids.shape[0] != local_valid_count
        ):
            raise RuntimeError(
                "Tokenized controlled query tensors do not align with "
                f"valid_query_count={local_valid_count}"
            )
        if (
            candidate_passage_indices.dtype != torch.long
            or candidate_passage_indices.ndim != 2
            or candidate_passage_indices.shape[0] != local_valid_count
            or positive_passage_indices.dtype != torch.long
            or positive_passage_indices.ndim != 2
            or positive_passage_indices.shape[0] != local_valid_count
        ):
            raise RuntimeError("Controlled candidate/positive index rows do not align with queries")
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() != self.EXPECTED_WORLD_SIZE
        ):
            raise RuntimeError("Controlled loss requires the initialized four-rank distributed group")

        candidate_plan = build_global_candidate_plan(
            candidate_passage_indices,
            corpus_size=len(self.passage_index_table),
        )
        if candidate_plan.local_owned_indices.numel() < 1:
            raise RuntimeError(
                "Controlled production microbatch assigned no real passage to this rank"
            )
        owned_passage_texts = [
            self.passage_index_table.text_for_index(int(passage_index))
            for passage_index in candidate_plan.local_owned_indices.detach().cpu().tolist()
        ]
        tokenizer = self.processing_class
        owned_passage_tokens = self._tokenize_owned_passages(
            tokenizer,
            owned_passage_texts,
            max_len_passage=self.max_len_passage,
        )
        passage_input_ids = owned_passage_tokens["input_ids"].to(device)
        passage_attention_mask = owned_passage_tokens["attention_mask"].to(device)
        if passage_input_ids.shape[0] != candidate_plan.local_owned_indices.numel():
            raise RuntimeError("Passage tokenizer changed the owned-passage row count")
        query_embeddings, owned_passage_embeddings = self._forward_controlled_batch(
            model,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            passage_input_ids=passage_input_ids,
            passage_attention_mask=passage_attention_mask,
        )
        if query_embeddings.shape[0] != local_valid_count:
            raise RuntimeError("DeepSpeed forward changed the valid query row count")
        if owned_passage_embeddings.shape[0] != candidate_plan.local_owned_indices.numel():
            raise RuntimeError("DeepSpeed forward changed the owned passage row count")
        passage_embeddings_all = gather_owned_embeddings(
            owned_passage_embeddings,
            candidate_plan,
        )

        logits = (
            (query_embeddings @ passage_embeddings_all.T) / float(retriever.temperature)
        ).float()
        positive_mask = build_index_positive_mask(
            candidate_plan.gathered_passage_indices,
            positive_passage_indices,
            candidate_plan.valid_passage_mask,
        )
        local_loss_sum, per_query_loss = masked_multi_positive_nce_loss_sum(
            logits,
            positive_mask,
            candidate_plan.valid_passage_mask,
        )
        scaled_loss = local_loss_sum * (self.EXPECTED_WORLD_SIZE / num_items_in_batch)

        if return_outputs:
            return scaled_loss, {
                "loss": scaled_loss.detach(),
                "local_loss_sum": local_loss_sum.detach(),
                "per_query_loss": per_query_loss,
                "local_valid_query_count": local_valid_count,
                "global_window_valid_count": num_items_in_batch,
                "global_unique_passage_count": candidate_plan.global_unique_indices.numel(),
            }
        return scaled_loss

    def training_step(
        self,
        model,
        inputs: Dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        marker = inputs.get("is_window_end")
        if type(marker) is not bool:
            raise TypeError("Controlled batch is missing exact boolean is_window_end")
        if bool(self.accelerator.sync_gradients) != marker:
            raise RuntimeError(
                "Trainer/controlled window boundary mismatch: "
                f"sync_gradients={self.accelerator.sync_gradients}, marker={marker}"
            )
        if not hasattr(model, "set_gradient_accumulation_boundary"):
            raise TypeError("Controlled DeepSpeed model lacks set_gradient_accumulation_boundary")
        sampling_traces = inputs.get("sampling_traces")
        if (
            type(sampling_traces) is not list
            or len(sampling_traces) != inputs["candidate_passage_indices"].shape[0]
            or len(sampling_traces) != inputs["positive_passage_indices"].shape[0]
        ):
            raise RuntimeError("Controlled sampling trace rows do not align with index rows")
        self._candidate_trace_store.record_batch(
            sampling_traces,
            candidate_passage_indices=inputs["candidate_passage_indices"],
            positive_passage_indices=inputs["positive_passage_indices"],
        )
        model.set_gradient_accumulation_boundary(marker)
        training_inputs = dict(inputs)
        training_inputs.pop("sampling_traces")
        loss = Trainer.training_step(
            self,
            model,
            training_inputs,
            num_items_in_batch=num_items_in_batch,
        )
        return loss.to(self.args.device)

    def finalize_sampling_traces(self) -> dict[str, Any]:
        queries = getattr(self.train_dataset, "queries", None)
        if not isinstance(queries, list) or len(queries) != self.EXPECTED_QUERIES:
            raise TypeError("Controlled train dataset lost its ordered query inventory")
        return self._candidate_trace_store.finalize(
            expected_epochs=self.EXPECTED_EPOCHS,
            expected_query_ids=[query.query_id for query in queries],
        )


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
