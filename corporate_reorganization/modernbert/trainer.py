from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import ExportableState
from accelerate.utils import DistributedType

from retriever.batching import GlobalQueryBatchSampler, SentinelQueryDataset
from retriever.checkpointing import (
    CheckpointSelection,
    VALIDATION_PRIMARY_METRIC as CHECKPOINT_VALIDATION_PRIMARY_METRIC,
    VALIDATION_SECONDARY_METRIC as CHECKPOINT_VALIDATION_SECONDARY_METRIC,
    ValidationMetadataStore,
    canonical_json,
    choose_better_checkpoint,
    load_controlled_checkpoint,
    retain_best_and_last_checkpoints,
    save_controlled_checkpoint,
)
from retriever.data import PassageIndexTable
from retriever.distributed import build_global_candidate_plan, gather_owned_embeddings
from retriever.eval import evaluate_retrieval
from retriever.evaluation import (
    FoldGlobalValidationData,
    FoldGlobalValidationResult,
    VALIDATION_FORWARD_STEPS,
    VALIDATION_PRIMARY_METRIC,
    VALIDATION_SECONDARY_METRIC,
    evaluate_fold_global_distributed,
)
from retriever.losses import (
    build_index_positive_mask,
    build_positive_mask,
    masked_multi_positive_nce_loss_sum,
    multi_positive_nce_loss,
)
from retriever.traces import CandidateTraceStore


def _coordinated_local_operation(context: str, operation):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(f"{context} requires an initialized process group")
    rank = dist.get_rank()
    try:
        value = operation()
        status: dict[str, Any] = {"ok": True, "rank": rank}
    except BaseException as error:
        value = None
        status = {
            "ok": False,
            "rank": rank,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    gathered: list[object] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, status)
    failures = [
        item
        for item in gathered
        if type(item) is not dict or item.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"{context} failed collectively: {failures}")
    return value


def _require_identical_string_across_ranks(context: str, value: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{context} requires a non-empty exact string")
    gathered: list[object] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    if gathered != [value] * dist.get_world_size():
        raise RuntimeError(f"{context} differs across ranks: {gathered}")


def _common_trainer_state_sha256(state: TrainerState, *, rank: int) -> str:
    if not isinstance(state, TrainerState):
        raise TypeError("Controlled Trainer state must be TrainerState")
    if type(rank) is not int or rank not in range(ControlledRetrievalTrainer.EXPECTED_WORLD_SIZE):
        raise ValueError("Controlled Trainer-state rank must be an exact integer 0 through 3")
    payload = asdict(state)
    rank_flags = {
        "is_local_process_zero": payload.pop("is_local_process_zero"),
        "is_world_process_zero": payload.pop("is_world_process_zero"),
    }
    expected_rank_flags = {
        "is_local_process_zero": rank == 0,
        "is_world_process_zero": rank == 0,
    }
    if rank_flags != expected_rank_flags:
        raise RuntimeError(
            f"Trainer state rank flags changed: actual={rank_flags}, "
            f"expected={expected_rank_flags}"
        )
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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
        validation_data: FoldGlobalValidationData,
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
        if (
            CHECKPOINT_VALIDATION_PRIMARY_METRIC != VALIDATION_PRIMARY_METRIC
            or CHECKPOINT_VALIDATION_SECONDARY_METRIC != VALIDATION_SECONDARY_METRIC
        ):
            raise RuntimeError("Evaluator and checkpoint model-selection metric keys disagree")
        if not isinstance(validation_data, FoldGlobalValidationData):
            raise TypeError("validation_data must be FoldGlobalValidationData")
        if validation_data.role != "validation" or validation_data.query_count != 98:
            raise RuntimeError("Controlled validation must be the exact 98-query validation role")
        if validation_data.passage_count not in {1_054, 1_055, 1_060, 1_062}:
            raise RuntimeError(
                "Controlled validation passage count is outside the frozen fold inventory"
            )
        self.experiment_seed = experiment_seed
        self.passage_index_table = passage_index_table
        self.validation_data = validation_data
        self.max_len_passage = max_len_passage
        self._global_batch_sampler: GlobalQueryBatchSampler | None = None
        self._window_epoch: int | None = None
        self._window_index = 0
        self._pending_validation_result: FoldGlobalValidationResult | None = None
        self._pending_selection: CheckpointSelection | None = None
        self._pending_best: CheckpointSelection | None = None
        self._pending_is_new_best: bool | None = None
        self._best_validation_result: FoldGlobalValidationResult | None = None
        self._last_checkpoint_dir: str | None = None
        self._retained_checkpoint_dirs: tuple[str, ...] = ()
        self._evaluated_epochs: set[int] = set()
        self._checkpoint_manifest: dict[str, Any] | None = None
        self._engine_generation = 1
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
        if self.eval_dataset is None:
            raise RuntimeError("Controlled retrieval requires a non-null validation dataset marker")
        if self.retrieval_eval_config is not None:
            raise RuntimeError("Controlled retrieval forbids the legacy retrieval_eval_config")
        if str(self.args.eval_strategy.value) != "epoch":
            raise RuntimeError("Controlled retrieval requires eval_strategy='epoch'")
        if str(self.args.save_strategy.value) != "epoch":
            raise RuntimeError("Controlled retrieval requires save_strategy='epoch'")
        if self.args.eval_on_start or float(self.args.eval_delay) != 0.0:
            raise RuntimeError("Controlled retrieval requires epoch-only validation with no delay")
        if self.args.load_best_model_at_end:
            raise RuntimeError("Controlled ZeRO-3 forbids stock load_best_model_at_end")
        if self.args.save_only_model:
            raise RuntimeError("Controlled checkpoints must retain optimizer and scheduler state")
        if self.args.save_total_limit is not None:
            raise RuntimeError("Controlled checkpoint retention is implemented collectively")
        if (
            self.args.metric_for_best_model
            != VALIDATION_PRIMARY_METRIC.removeprefix("eval_")
            or self.args.greater_is_better is not True
        ):
            raise RuntimeError("Controlled Trainer model-selection arguments changed")
        self._candidate_trace_store = CandidateTraceStore(
            Path(self.args.output_dir),
            passage_index_table=self.passage_index_table,
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
        )
        self._validation_metadata_store = ValidationMetadataStore(
            Path(self.args.output_dir),
            expected_epochs=self.EXPECTED_EPOCHS,
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

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        del eval_dataset
        raise RuntimeError(
            "Controlled retrieval uses exact fold-global validation, not a Trainer eval dataloader"
        )

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
    ) -> None:
        if type(epoch) is not int or epoch not in range(self.EXPECTED_EPOCHS):
            raise RuntimeError(f"Controlled Trainer received invalid zero-based epoch={epoch!r}")
        final_step_save_signal = (
            self.state.global_step == self.EXPECTED_TOTAL_UPDATES
            and self.state.max_steps == self.EXPECTED_TOTAL_UPDATES
            and float(self.state.epoch) == float(self.EXPECTED_EPOCHS)
            and self.control.should_training_stop is True
            and self.control.should_save is True
            and self.control.should_evaluate is False
        )
        if final_step_save_signal:
            # Transformers 4.49 requests a terminal save from on_step_end before
            # its epoch-end callback requests evaluation. Defer only that exact
            # signal so the final checkpoint is selected after validation.
            self.control.should_save = False
        elif self.control.should_save is not self.control.should_evaluate:
            raise RuntimeError(
                "Controlled epoch validation/save flags diverged outside the exact final-step "
                "deferral"
            )
        if self.control.should_save:
            self._completed_epoch_number()
        Trainer._maybe_log_save_evaluate(
            self,
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
        )

    def _completed_epoch_number(self) -> int:
        epoch = self.state.epoch
        if type(epoch) not in (int, float) or not math.isfinite(float(epoch)):
            raise RuntimeError(f"Trainer state has invalid completed epoch={epoch!r}")
        epoch_number = int(epoch)
        if float(epoch) != float(epoch_number):
            raise RuntimeError(f"Validation must run only after a complete epoch; got {epoch}")
        if epoch_number < 1 or epoch_number > self.EXPECTED_EPOCHS:
            raise RuntimeError(f"Completed epoch is outside 1..{self.EXPECTED_EPOCHS}: {epoch}")
        expected_step = epoch_number * self.EXPECTED_UPDATES_PER_EPOCH
        if self.state.global_step != expected_step:
            raise RuntimeError(
                f"Completed epoch {epoch_number} has global_step={self.state.global_step}; "
                f"expected {expected_step}"
            )
        return epoch_number

    def _run_controlled_validation(self, model) -> FoldGlobalValidationResult:
        return evaluate_fold_global_distributed(
            model,
            self.processing_class,
            validation_data=self.validation_data,
            passage_index_table=self.passage_index_table,
            max_len_query=4_096,
            max_len_passage=self.max_len_passage,
            forward_steps=VALIDATION_FORWARD_STEPS,
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is not None and eval_dataset is not self.eval_dataset:
            raise ValueError("Controlled validation dataset cannot be replaced at runtime")
        if ignore_keys not in (None, []):
            raise ValueError("Controlled validation does not accept ignore_keys")
        if metric_key_prefix != "eval":
            raise ValueError("Controlled validation metric prefix must be exactly 'eval'")
        if self.model_wrapped is self.model or self.deepspeed is None:
            raise RuntimeError("Controlled validation requires the active DeepSpeed engine")
        epoch_number = self._completed_epoch_number()
        if epoch_number in self._evaluated_epochs:
            raise RuntimeError(f"Controlled validation epoch {epoch_number} was already evaluated")
        if self._pending_validation_result is not None or self._pending_selection is not None:
            raise RuntimeError("A prior validation result has not completed checkpoint publication")

        result = self._run_controlled_validation(self.model_wrapped)
        metrics = dict(result.metrics)
        if set(metrics) != set(result.metrics):
            raise RuntimeError("Controlled validation metric mapping changed during copy")
        self._pending_validation_result = result
        self._evaluated_epochs.add(epoch_number)
        # Transformers 4.49 mutates the mapping passed to Trainer.log by adding
        # epoch metadata. Keep the canonical broadcast result byte-for-byte
        # unchanged for checkpoint selection and provenance.
        self.log(dict(metrics))
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        if self.is_world_process_zero():
            for name in sorted(metrics):
                print(f"SM_METRIC {name}={metrics[name]}")
        return metrics

    def _determine_best_metric(self, metrics, trial):
        if trial is not None:
            raise RuntimeError("Controlled retrieval forbids hyperparameter-search trials")
        if self._pending_validation_result is None:
            raise RuntimeError("Checkpoint selection requires a pending controlled validation result")
        if type(metrics) is not dict or metrics != self._pending_validation_result.metrics:
            raise RuntimeError("Trainer selection metrics differ from the broadcast validation result")
        epoch_number = self._completed_epoch_number()
        primary = metrics.get(VALIDATION_PRIMARY_METRIC)
        secondary = metrics.get(VALIDATION_SECONDARY_METRIC)
        if type(primary) is not float or not math.isfinite(primary):
            raise RuntimeError("Controlled primary validation metric must be a finite exact float")
        if type(secondary) is not float or not math.isfinite(secondary):
            raise RuntimeError("Controlled secondary validation metric must be a finite exact float")
        candidate = CheckpointSelection(
            schema_version=1,
            epoch=epoch_number,
            global_step=self.state.global_step,
            checkpoint_dir=f"checkpoint-{self.state.global_step}",
            deepspeed_tag=f"global_step{self.state.global_step}",
            primary_metric=primary,
            secondary_metric=secondary,
            ranking_sha256=self._pending_validation_result.ranking_sha256,
        )
        best, is_new_best = choose_better_checkpoint(
            self._validation_metadata_store.best,
            candidate,
        )
        decision_payload = {
            "candidate": candidate.to_payload(),
            "best": best.to_payload(),
            "is_new_best": is_new_best,
        }
        decision_sha256 = hashlib.sha256(
            canonical_json(decision_payload).encode("utf-8")
        ).hexdigest()
        gathered_decisions: list[object] = [None for _ in range(self.EXPECTED_WORLD_SIZE)]
        dist.all_gather_object(gathered_decisions, decision_sha256)
        if gathered_decisions != [decision_sha256] * self.EXPECTED_WORLD_SIZE:
            raise RuntimeError(f"Checkpoint selection differs across ranks: {gathered_decisions}")

        self.state.best_metric = best.primary_metric
        self.state.best_model_checkpoint = str(Path(self.args.output_dir) / best.checkpoint_dir)
        self._pending_selection = candidate
        self._pending_best = best
        self._pending_is_new_best = is_new_best
        return is_new_best

    def _prepare_trainer_state_for_checkpoint(self) -> None:
        if self._pending_selection is None or self._pending_best is None:
            raise RuntimeError("Cannot serialize Trainer state before checkpoint selection")
        self.store_flos()
        callbacks = [
            callback
            for callback in self.callback_handler.callbacks + [self.control]
            if isinstance(callback, ExportableState)
        ]
        if callbacks != [self.control] or not isinstance(self.control, TrainerControl):
            raise RuntimeError(
                "Controlled checkpointing requires TrainerControl as the only ExportableState"
            )
        expected_control = {
            "should_training_stop": self._pending_selection.epoch == self.EXPECTED_EPOCHS,
            "should_epoch_stop": False,
            "should_save": True,
            "should_evaluate": False,
            "should_log": False,
        }
        actual_control = self.control.state().get("args")
        if actual_control != expected_control:
            raise RuntimeError(
                f"TrainerControl flags changed at checkpoint serialization: "
                f"actual={actual_control}, expected={expected_control}"
            )
        self.state.stateful_callbacks = {"TrainerControl": self.control.state()}
        expected_best_path = str(
            Path(self.args.output_dir) / self._pending_best.checkpoint_dir
        )
        if self.state.best_model_checkpoint != expected_best_path:
            raise RuntimeError("Trainer best checkpoint path changed before serialization")
        if self.state.best_metric != self._pending_best.primary_metric:
            raise RuntimeError("Trainer best metric changed before serialization")

    def _save_checkpoint(self, model, trial) -> None:
        if trial is not None:
            raise RuntimeError("Controlled retrieval forbids trial-specific checkpoints")
        if model is not self.model_wrapped or model is not self.deepspeed:
            raise RuntimeError("Controlled checkpoint must save the active top-level engine")
        if (
            self._pending_validation_result is None
            or self._pending_selection is None
            or self._pending_best is None
            or type(self._pending_is_new_best) is not bool
        ):
            raise RuntimeError("Controlled checkpoint is missing validation/selection state")
        _coordinated_local_operation(
            "Trainer checkpoint-state serialization preflight",
            self._prepare_trainer_state_for_checkpoint,
        )
        trainer_state_common_sha256 = _coordinated_local_operation(
            "Trainer state digest preparation",
            lambda: _common_trainer_state_sha256(
                self.state,
                rank=dist.get_rank(),
            ),
        )
        _require_identical_string_across_ranks(
            "Common Trainer state digest",
            trainer_state_common_sha256,
        )
        result = self._pending_validation_result
        candidate = self._pending_selection
        controlled_state = {
            "schema_version": 1,
            "experiment_seed": self.experiment_seed,
            "selection_candidate": candidate.to_payload(),
            "best_after_epoch": self._pending_best.to_payload(),
            "validation": {
                "metrics": dict(result.metrics),
                "ranking_sha256": result.ranking_sha256,
                "case_ids_sha256": result.case_ids_sha256,
                "query_ids_sha256": result.query_ids_sha256,
                "passage_ids_sha256": result.passage_ids_sha256,
                "validation_contract_sha256": result.validation_contract_sha256,
            },
            "passage_index_sha256": self.passage_index_table.sha256,
            "trainer_state_common_sha256": trainer_state_common_sha256,
        }
        checkpoint_metadata = save_controlled_checkpoint(
            output_dir=Path(self.args.output_dir),
            engine=model,
            scheduler=self.lr_scheduler,
            trainer_state=self.state,
            training_args=self.args,
            selection=candidate,
            client_state={"controlled_state": controlled_state},
            expected_world_size=self.EXPECTED_WORLD_SIZE,
        )
        best, is_new_best, _ = self._validation_metadata_store.register_checkpoint(
            candidate=candidate,
            validation_result=result.to_payload(),
            checkpoint_metadata=checkpoint_metadata,
        )
        if best != self._pending_best or is_new_best != self._pending_is_new_best:
            raise RuntimeError("Published checkpoint selection differs from previewed decision")
        if is_new_best:
            self._best_validation_result = result
        self._last_checkpoint_dir = candidate.checkpoint_dir
        self._retained_checkpoint_dirs = retain_best_and_last_checkpoints(
            Path(self.args.output_dir),
            best_checkpoint_dir=best.checkpoint_dir,
            last_checkpoint_dir=candidate.checkpoint_dir,
        )
        self._pending_validation_result = None
        self._pending_selection = None
        self._pending_best = None
        self._pending_is_new_best = None

    def finalize_checkpoint_selection(self) -> dict[str, Any]:
        def validate_local_finalization_state() -> None:
            if self._pending_validation_result is not None or self._pending_selection is not None:
                raise RuntimeError(
                    "Cannot finalize with an unpublished validation/checkpoint result"
                )
            if self._last_checkpoint_dir is None or not self._retained_checkpoint_dirs:
                raise RuntimeError("Controlled training produced no retained checkpoints")

        _coordinated_local_operation(
            "Checkpoint-history finalization preflight",
            validate_local_finalization_state,
        )
        manifest = self._validation_metadata_store.finalize(
            retained_checkpoint_dirs=self._retained_checkpoint_dirs,
        )
        self._checkpoint_manifest = manifest
        return manifest

    def release_current_deepspeed_engine(self) -> None:
        engine = self.deepspeed
        model = self.model
        optimizer = self.optimizer
        scheduler = self.lr_scheduler
        _coordinated_local_operation(
            "DeepSpeed engine release preflight",
            lambda: (
                True
                if engine is not None and engine is self.model_wrapped
                else (_ for _ in ()).throw(
                    RuntimeError(
                        "No active DeepSpeed engine is available for collective release"
                    )
                )
            ),
        )
        self.accelerator.wait_for_everyone()

        def release_local_objects() -> None:
            self.model = None
            self.model_wrapped = None
            self.deepspeed = None
            self.optimizer = None
            self.lr_scheduler = None
            self.callback_handler.model = None
            self.callback_handler.optimizer = None
            self.callback_handler.lr_scheduler = None
            self.callback_handler.train_dataloader = None
            self.callback_handler.eval_dataloader = None
            self.accelerator.free_memory(engine, model, optimizer, scheduler)
            gc.collect()
            torch.cuda.empty_cache()
            if (
                self.accelerator.deepspeed_engine_wrapped is not None
                or self.accelerator._models
                or self.accelerator._optimizers
                or self.accelerator._schedulers
                or self.accelerator._dataloaders
            ):
                raise RuntimeError("Accelerate retained objects after DeepSpeed engine release")

        _coordinated_local_operation(
            "DeepSpeed engine release",
            release_local_objects,
        )
        release_local_objects = None
        engine = model = optimizer = scheduler = None
        self.accelerator.wait_for_everyone()

    def prepare_fresh_deepspeed_engine(self, fresh_model) -> None:
        def validate_fresh_preconditions() -> None:
            if (
                self.deepspeed is not None
                or self.model is not None
                or self.model_wrapped is not None
            ):
                raise RuntimeError(
                    "Fresh engine preparation requires the prior engine to be released"
                )
            if self._checkpoint_manifest is None or self._validation_metadata_store.best is None:
                raise RuntimeError(
                    "Fresh engine preparation requires finalized checkpoint selection"
                )
            if self._engine_generation != 1:
                raise RuntimeError(
                    "Fresh engine preparation permits exactly one Engine-B construction"
                )

        _coordinated_local_operation(
            "Fresh DeepSpeed engine preflight",
            validate_fresh_preconditions,
        )
        from transformers.integrations.deepspeed import deepspeed_init

        def initialize_local_objects():
            if not isinstance(fresh_model, torch.nn.Module):
                raise TypeError("Fresh controlled model must be a torch module")
            partitioned = [
                name
                for name, parameter in fresh_model.named_parameters()
                if hasattr(parameter, "ds_id")
            ]
            if partitioned:
                raise RuntimeError(
                    "Fresh controlled model was already ZeRO-partitioned: "
                    f"{partitioned[:5]}"
                )
            self.model = fresh_model
            self.model_wrapped = fresh_model
            self.callback_handler.model = fresh_model
            self.optimizer = None
            self.lr_scheduler = None
            fresh_model.train()
            return deepspeed_init(
                self,
                num_training_steps=self.EXPECTED_TOTAL_UPDATES,
            )

        optimizer, scheduler = _coordinated_local_operation(
            "Fresh DeepSpeed optimizer/scheduler initialization",
            initialize_local_objects,
        )
        fresh_model.train()
        engine, prepared_optimizer = self.accelerator.prepare(fresh_model, optimizer)
        self.model = fresh_model
        self.model_wrapped = engine
        self.deepspeed = engine
        self.optimizer = prepared_optimizer
        self.lr_scheduler = scheduler
        self.callback_handler.model = fresh_model
        self.callback_handler.optimizer = prepared_optimizer
        self.callback_handler.lr_scheduler = scheduler

        def validate_prepared_engine() -> None:
            if engine.module is not fresh_model:
                raise RuntimeError("Fresh DeepSpeed engine does not wrap the constructed model")
            if int(engine.zero_optimization_stage()) != 3 or engine.bfloat16_enabled() is not True:
                raise RuntimeError("Fresh DeepSpeed engine changed the ZeRO-3/BF16 contract")
            dp_world_size = engine.dp_world_size
            if callable(dp_world_size):
                dp_world_size = dp_world_size()
            if dp_world_size != self.EXPECTED_WORLD_SIZE:
                raise RuntimeError("Fresh DeepSpeed engine changed data-parallel world size")
            if int(engine.global_steps) != 0:
                raise RuntimeError("Fresh DeepSpeed engine must start at global step zero")
            if engine.lr_scheduler is not None or scheduler is None:
                raise RuntimeError("Fresh Trainer scheduler must remain external to DeepSpeed")

        _coordinated_local_operation(
            "Fresh DeepSpeed engine validation",
            validate_prepared_engine,
        )
        self._engine_generation = 2

    def load_and_verify_best_checkpoint(self) -> dict[str, Any]:
        best = self._validation_metadata_store.best

        def validate_reload_preconditions() -> None:
            if best is None or self._best_validation_result is None:
                raise RuntimeError("No selected best checkpoint/result is available for reload")
            if self._checkpoint_manifest is None:
                raise RuntimeError("Checkpoint history must be finalized before best reload")
            if self.deepspeed is None or self.model_wrapped is not self.deepspeed:
                raise RuntimeError("Best reload requires a prepared pristine DeepSpeed engine")

        _coordinated_local_operation(
            "Best-checkpoint reload preflight",
            validate_reload_preconditions,
        )
        load_metadata = load_controlled_checkpoint(
            checkpoint_root=Path(self.args.output_dir) / best.checkpoint_dir,
            engine=self.deepspeed,
            scheduler=self.lr_scheduler,
            selection=best,
            expected_world_size=self.EXPECTED_WORLD_SIZE,
            restore_rng=True,
        )
        reloaded_result = self._run_controlled_validation(self.deepspeed)
        expected_payload = self._best_validation_result.to_payload()
        actual_payload = reloaded_result.to_payload()
        if canonical_json(actual_payload) != canonical_json(expected_payload):
            raise RuntimeError("Fresh-engine best validation does not exactly reproduce selection")
        return {
            **load_metadata,
            "selection": best.to_payload(),
            "validation_result": actual_payload,
        }

    def finalize_sampling_traces(self) -> dict[str, Any]:
        def ordered_query_ids() -> list[str]:
            queries = getattr(self.train_dataset, "queries", None)
            if not isinstance(queries, list) or len(queries) != self.EXPECTED_QUERIES:
                raise TypeError("Controlled train dataset lost its ordered query inventory")
            query_ids = [query.query_id for query in queries]
            if any(type(query_id) is not str or not query_id for query_id in query_ids):
                raise TypeError("Controlled train query IDs changed before trace finalization")
            return query_ids

        query_ids = _coordinated_local_operation(
            "Candidate-trace finalization preflight",
            ordered_query_ids,
        )
        query_inventory_sha256 = hashlib.sha256(
            canonical_json(query_ids).encode("utf-8")
        ).hexdigest()
        _require_identical_string_across_ranks(
            "Candidate-trace query inventory",
            query_inventory_sha256,
        )
        return self._candidate_trace_store.finalize(
            expected_epochs=self.EXPECTED_EPOCHS,
            expected_query_ids=query_ids,
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
