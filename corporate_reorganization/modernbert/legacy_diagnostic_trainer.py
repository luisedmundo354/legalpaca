"""Sealed DeepSpeed trainer for the corrected legacy-style diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader
from transformers import Trainer

from retriever.batching import SentinelQueryDataset
from retriever.data import PassageIndexTable
from retriever.distributed import build_global_candidate_plan, gather_owned_embeddings
from retriever.corrected_legacy_evaluation import (
    CorrectedLegacyValidationEvidenceData,
    evaluate_corrected_legacy_validation_evidence_distributed,
)
from retriever.legacy_diagnostic_batching import CorrectedLegacyQueryBatchSampler
from retriever.legacy_diagnostic_distributed import gather_global_candidate_multiplicities
from retriever.legacy_diagnostic_losses import multiplicity_aware_multi_positive_nce_loss_sum
from retriever.losses import build_index_positive_mask


EXPECTED_QUERIES = 418
EXPECTED_WORLD_SIZE = 4
EXPECTED_PER_DEVICE_BATCH = 4
EXPECTED_GRADIENT_ACCUMULATION = 8
EXPECTED_PREPARED_BATCHES = 27
EXPECTED_WINDOW_MICROBATCHES = (8, 8, 8, 3)
EXPECTED_WINDOW_VALID_QUERIES = (128, 128, 128, 34)
EXPECTED_EPOCHS = 20
EXPECTED_UPDATES_PER_EPOCH = 4
EXPECTED_TOTAL_UPDATES = 80


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class CorrectedLegacyDiagnosticTrainer(Trainer):
    """Exact 418-query trainer; intentionally separate from controlled training."""

    def __init__(
        self,
        *args: Any,
        experiment_seed: int,
        passage_index_table: PassageIndexTable,
        validation_data: CorrectedLegacyValidationEvidenceData,
        max_len_passage: int,
        **kwargs: Any,
    ) -> None:
        if type(experiment_seed) is not int or experiment_seed != 17:
            raise ValueError("Corrected legacy trainer seed must be exactly 17")
        if not isinstance(passage_index_table, PassageIndexTable) or len(passage_index_table) != 5_286:
            raise ValueError("Corrected legacy trainer requires the exact 5,286-passage index")
        if type(max_len_passage) is not int or max_len_passage != 500:
            raise ValueError("Corrected legacy max passage length must be exactly 500")
        if (
            not isinstance(validation_data, CorrectedLegacyValidationEvidenceData)
            or len(validation_data.case_ids) != 4
            or len(validation_data.queries) != 32
            or len(validation_data.passage_indices) != 398
        ):
            raise ValueError("Corrected legacy validation must be exactly 4 cases/32 queries/398 passages")
        self.experiment_seed = experiment_seed
        self.passage_index_table = passage_index_table
        self.validation_data = validation_data
        self.max_len_passage = max_len_passage
        self._global_batch_sampler: CorrectedLegacyQueryBatchSampler | None = None
        self._window_epoch: int | None = None
        self._window_index = 0
        self._trace_epoch: int | None = None
        self._trace_microbatch_index = 0
        self._trace_lines: list[str] = []
        self._trace_query_ids_by_epoch: dict[int, set[str]] = {}
        self._validation_records: list[dict[str, Any]] = []
        self._evaluated_epochs: set[int] = set()
        super().__init__(*args, **kwargs)

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            raise RuntimeError("Corrected legacy training requires DeepSpeed")
        if self.accelerator.num_processes != EXPECTED_WORLD_SIZE:
            raise RuntimeError("Corrected legacy training requires exactly four processes")
        if self.args.per_device_train_batch_size != EXPECTED_PER_DEVICE_BATCH:
            raise RuntimeError("Corrected legacy per-device query batch must be exactly four")
        if self.args.gradient_accumulation_steps != EXPECTED_GRADIENT_ACCUMULATION:
            raise RuntimeError("Corrected legacy gradient accumulation must be exactly eight")
        if self.args.dataloader_num_workers != 0 or self.args.dataloader_persistent_workers:
            raise RuntimeError("Corrected legacy training requires a synchronous dataloader")
        if self.args.dataloader_drop_last:
            raise RuntimeError("Corrected legacy training forbids dropping final rows")
        if not self.model_accepts_loss_kwargs:
            raise RuntimeError("Corrected legacy Trainer requires the exact loss-kwargs contract")
        if self.processing_class is None:
            raise RuntimeError("Corrected legacy training requires an explicit local tokenizer")
        if self.eval_dataset is None:
            raise RuntimeError("Corrected legacy training requires a validation marker")
        if str(self.args.eval_strategy.value) != "epoch" or str(self.args.save_strategy.value) != "no":
            raise RuntimeError("Corrected legacy training requires epoch evaluation and no checkpoints")
        if self.args.load_best_model_at_end or self.args.metric_for_best_model is not None:
            raise RuntimeError("Corrected legacy training forbids best-model selection")
        if self.args.save_total_limit is not None:
            raise RuntimeError("Corrected legacy training forbids checkpoint retention settings")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or len(self.train_dataset) != EXPECTED_QUERIES:
            raise ValueError("Corrected legacy train dataset must contain exactly 418 queries")
        queries = getattr(self.train_dataset, "queries", None)
        if not isinstance(queries, tuple) or len(queries) != EXPECTED_QUERIES:
            raise TypeError("Corrected legacy train dataset must expose its ordered query tuple")
        query_ids = [query.query_id for query in queries]
        batch_sampler = CorrectedLegacyQueryBatchSampler(
            query_ids,
            experiment_seed=self.experiment_seed,
            world_size=EXPECTED_WORLD_SIZE,
            per_device_batch_size=EXPECTED_PER_DEVICE_BATCH,
        )
        wrapped = SentinelQueryDataset(self.train_dataset, epoch_target=batch_sampler)
        prepared = self.accelerator.prepare(
            DataLoader(
                wrapped,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=False,
            )
        )
        if len(batch_sampler) != EXPECTED_PREPARED_BATCHES * EXPECTED_WORLD_SIZE:
            raise RuntimeError("Corrected legacy raw batch count changed")
        if len(prepared) != EXPECTED_PREPARED_BATCHES:
            raise RuntimeError("Corrected legacy prepared dataloader must contain 27 batches/rank")
        self._global_batch_sampler = batch_sampler
        return prepared

    def set_initial_training_values(self, args, dataloader, total_train_batch_size: int):
        if args.max_steps >= 0:
            raise RuntimeError("Corrected legacy training forbids max_steps")
        if float(args.num_train_epochs) != float(EXPECTED_EPOCHS):
            raise RuntimeError("Corrected legacy training requires exactly 20 epochs")
        if len(dataloader) != EXPECTED_PREPARED_BATCHES or total_train_batch_size != 128:
            raise RuntimeError("Corrected legacy Trainer schedule inputs changed")
        if self.num_examples(dataloader) != EXPECTED_QUERIES:
            raise RuntimeError("Corrected legacy Trainer example count changed")
        return (
            EXPECTED_EPOCHS,
            EXPECTED_UPDATES_PER_EPOCH,
            EXPECTED_QUERIES,
            EXPECTED_QUERIES * EXPECTED_EPOCHS,
            True,
            EXPECTED_PREPARED_BATCHES,
            EXPECTED_TOTAL_UPDATES,
        )

    @staticmethod
    def _valid_query_count(batch: Mapping[str, Any]) -> int:
        value = batch.get("valid_query_count")
        if not torch.is_tensor(value) or value.dtype != torch.long or value.numel() != 1:
            raise TypeError("Corrected legacy valid_query_count must be one long scalar")
        result = int(value.item())
        if result < 1 or result > EXPECTED_PER_DEVICE_BATCH:
            raise ValueError("Corrected legacy local valid query count is outside 1..4")
        return result

    def _reduce_window_count(self, local_valid: int, local_microbatches: int) -> int:
        device = self.args.device
        minimum = torch.tensor(local_microbatches, dtype=torch.long, device=device)
        maximum = minimum.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(minimum.item()) != local_microbatches or int(maximum.item()) != local_microbatches:
            raise RuntimeError("Corrected legacy ranks fetched different window lengths")
        count = torch.tensor(local_valid, dtype=torch.long, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return int(count.item())

    def get_batch_samples(self, epoch_iterator, num_batches):
        del num_batches
        batches = []
        for _ in range(EXPECTED_GRADIENT_ACCUMULATION):
            try:
                batches.append(next(epoch_iterator))
            except StopIteration:
                break
        if not batches:
            raise RuntimeError("Trainer requested an empty corrected legacy optimizer window")
        global_valid = self._reduce_window_count(
            sum(self._valid_query_count(batch) for batch in batches),
            len(batches),
        )
        if self._global_batch_sampler is None:
            raise RuntimeError("Corrected legacy batch sampler is not initialized")
        epoch = self._global_batch_sampler.epoch
        if self._window_epoch != epoch:
            self._window_epoch = epoch
            self._window_index = 0
        if self._window_index >= len(EXPECTED_WINDOW_VALID_QUERIES):
            raise RuntimeError("Corrected legacy epoch produced too many optimizer windows")
        expected = (
            EXPECTED_WINDOW_MICROBATCHES[self._window_index],
            EXPECTED_WINDOW_VALID_QUERIES[self._window_index],
        )
        if (len(batches), global_valid) != expected:
            raise RuntimeError(
                f"Corrected legacy window {self._window_index} changed: "
                f"actual={(len(batches), global_valid)}, expected={expected}"
            )
        for position, batch in enumerate(batches):
            batch["global_window_valid_count"] = global_valid
            batch["is_window_end"] = position == len(batches) - 1
        self._window_index += 1
        return batches, global_valid

    @staticmethod
    def _tokenize_passages(tokenizer, texts: list[str]) -> Mapping[str, torch.Tensor]:
        original_side = tokenizer.truncation_side
        tokenizer.truncation_side = "right"
        try:
            tokens = tokenizer(
                texts,
                truncation=True,
                max_length=500,
                padding=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.truncation_side = original_side
        if (
            not isinstance(tokens, Mapping)
            or not torch.is_tensor(tokens.get("input_ids"))
            or not torch.is_tensor(tokens.get("attention_mask"))
            or tokens["input_ids"].shape != tokens["attention_mask"].shape
        ):
            raise TypeError("Corrected legacy passage tokenizer returned malformed tensors")
        return tokens

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs: Any,
    ):
        del kwargs
        if type(num_items_in_batch) is not int or num_items_in_batch < 1:
            raise TypeError("Corrected legacy loss requires a global optimizer-window count")
        if inputs.get("global_window_valid_count") != num_items_in_batch:
            raise RuntimeError("Corrected legacy batch/window counts disagree")
        retriever = model.module if hasattr(model, "module") else model
        device = next(retriever.parameters()).device
        query_input_ids = inputs["query_input_ids"].to(device)
        query_attention_mask = inputs["query_attention_mask"].to(device)
        candidate_indices = inputs["candidate_passage_indices"].to(device)
        candidate_multiplicities = inputs["candidate_multiplicities"].to(device)
        positive_indices = inputs["positive_passage_indices"].to(device)
        local_valid = self._valid_query_count(inputs)
        if query_input_ids.shape != query_attention_mask.shape or query_input_ids.shape[0] != local_valid:
            raise RuntimeError("Corrected legacy query tensors do not align with real rows")
        if candidate_indices.shape != candidate_multiplicities.shape:
            raise RuntimeError("Corrected legacy candidate/multiplicity shapes disagree")
        if candidate_indices.shape[0] != local_valid or positive_indices.shape[0] != local_valid:
            raise RuntimeError("Corrected legacy index rows do not align with queries")
        if not dist.is_initialized() or dist.get_world_size() != EXPECTED_WORLD_SIZE:
            raise RuntimeError("Corrected legacy loss requires the four-rank group")

        plan = build_global_candidate_plan(
            candidate_indices,
            corpus_size=len(self.passage_index_table),
        )
        global_multiplicities = gather_global_candidate_multiplicities(
            candidate_indices,
            candidate_multiplicities,
            plan,
            corpus_size=len(self.passage_index_table),
        )
        if plan.local_owned_indices.numel() < 1:
            raise RuntimeError("Corrected legacy owner rank received no real passage")
        passage_tokens = self._tokenize_passages(
            self.processing_class,
            [
                self.passage_index_table.text_for_index(int(index))
                for index in plan.local_owned_indices.detach().cpu().tolist()
            ],
        )
        outputs = model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            passage_input_ids=passage_tokens["input_ids"].to(device),
            passage_attention_mask=passage_tokens["attention_mask"].to(device),
        )
        if type(outputs) is not dict or set(outputs) != {"query_embeddings", "passage_embeddings"}:
            raise TypeError("Corrected legacy top-level forward schema changed")
        query_embeddings = outputs["query_embeddings"]
        owned_embeddings = outputs["passage_embeddings"]
        if query_embeddings.shape[0] != local_valid:
            raise RuntimeError("Corrected legacy forward changed query row count")
        if owned_embeddings.shape[0] != plan.local_owned_indices.numel():
            raise RuntimeError("Corrected legacy forward changed owned passage count")
        passage_embeddings = gather_owned_embeddings(owned_embeddings, plan)
        logits = (
            (query_embeddings @ passage_embeddings.T) / float(retriever.temperature)
        ).float()
        positive_mask = build_index_positive_mask(
            plan.gathered_passage_indices,
            positive_indices,
            plan.valid_passage_mask,
        )
        valid = plan.valid_passage_mask
        local_loss_sum, per_query_loss = multiplicity_aware_multi_positive_nce_loss_sum(
            logits[:, valid],
            positive_mask[:, valid],
            global_multiplicities[valid],
        )
        scaled_loss = local_loss_sum * (EXPECTED_WORLD_SIZE / num_items_in_batch)
        if return_outputs:
            return scaled_loss, {
                "loss": scaled_loss.detach(),
                "local_loss_sum": local_loss_sum.detach(),
                "per_query_loss": per_query_loss,
                "global_occurrence_count": int(global_multiplicities.sum().item()),
                "global_unique_passage_count": int(valid.sum().item()),
            }
        return scaled_loss

    def _record_traces(self, traces: object) -> None:
        if type(traces) is not list or not traces:
            raise TypeError("Corrected legacy batch must contain sampling traces")
        epoch_values = {trace.get("epoch") for trace in traces if type(trace) is dict}
        if len(epoch_values) != 1:
            raise RuntimeError("Corrected legacy batch traces disagree on epoch")
        epoch = next(iter(epoch_values))
        if type(epoch) is not int or epoch not in range(EXPECTED_EPOCHS):
            raise RuntimeError("Corrected legacy trace epoch is outside 0..19")
        if self._trace_epoch != epoch:
            if self._trace_epoch is not None and epoch != self._trace_epoch + 1:
                raise RuntimeError("Corrected legacy trace epochs are not contiguous")
            self._trace_epoch = epoch
            self._trace_microbatch_index = 0
        if self._trace_microbatch_index >= EXPECTED_PREPARED_BATCHES:
            raise RuntimeError("Corrected legacy rank produced too many trace microbatches")
        query_ids = self._trace_query_ids_by_epoch.setdefault(epoch, set())
        for local_row, trace in enumerate(traces):
            query_id = trace["query_id"]
            if query_id in query_ids:
                raise RuntimeError("Corrected legacy rank repeated a query trace in one epoch")
            query_ids.add(query_id)
            record = {
                "rank": self.accelerator.process_index,
                "prepared_microbatch_index": self._trace_microbatch_index,
                "local_row": local_row,
                "trace": trace,
            }
            self._trace_lines.append(_canonical_json(record) + "\n")
        self._trace_microbatch_index += 1

    def training_step(
        self,
        model,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        marker = inputs.get("is_window_end")
        if type(marker) is not bool or bool(self.accelerator.sync_gradients) != marker:
            raise RuntimeError("Corrected legacy optimizer boundary marker changed")
        if not hasattr(model, "set_gradient_accumulation_boundary"):
            raise TypeError("Corrected legacy DeepSpeed model lacks boundary control")
        self._record_traces(inputs.get("sampling_traces"))
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
        raise RuntimeError("Corrected legacy validation uses the sealed distributed evaluator")

    def _completed_epoch(self) -> int:
        epoch = self.state.epoch
        if type(epoch) not in (int, float) or not math.isfinite(float(epoch)):
            raise RuntimeError("Corrected legacy Trainer epoch is invalid")
        number = int(epoch)
        if float(epoch) != float(number) or number not in range(1, EXPECTED_EPOCHS + 1):
            raise RuntimeError("Corrected legacy validation must follow a complete epoch")
        if self.state.global_step != number * EXPECTED_UPDATES_PER_EPOCH:
            raise RuntimeError("Corrected legacy epoch/global-step chronology changed")
        return number

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if eval_dataset is not None and eval_dataset is not self.eval_dataset:
            raise ValueError("Corrected legacy validation dataset cannot be replaced")
        if ignore_keys not in (None, []) or metric_key_prefix != "eval":
            raise ValueError("Corrected legacy validation invocation changed")
        if self.model_wrapped is self.model or self.deepspeed is None:
            raise RuntimeError("Corrected legacy validation requires the active DeepSpeed engine")
        epoch = self._completed_epoch()
        if epoch in self._evaluated_epochs:
            raise RuntimeError("Corrected legacy epoch was evaluated twice")
        result = evaluate_corrected_legacy_validation_evidence_distributed(
            self.model_wrapped,
            self.processing_class,
            validation_data=self.validation_data,
            passage_index_table=self.passage_index_table,
        )
        record = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": self.state.global_step,
            "validation_result": result.to_payload(),
        }
        record["record_sha256"] = hashlib.sha256(
            _canonical_json(record).encode("utf-8")
        ).hexdigest()
        self._validation_records.append(record)
        self._evaluated_epochs.add(epoch)
        metrics = {
            f"eval_validation_{name}": value for name, value in result.metrics.items()
        }
        self.log(dict(metrics))
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        return metrics

    def local_trace_shard(self) -> dict[str, Any]:
        if self._trace_epoch != EXPECTED_EPOCHS - 1:
            raise RuntimeError("Corrected legacy traces did not reach epoch 19")
        if self._trace_microbatch_index != EXPECTED_PREPARED_BATCHES:
            raise RuntimeError("Corrected legacy final trace epoch lacks 27 microbatches")
        if len(self._trace_query_ids_by_epoch) != EXPECTED_EPOCHS:
            raise RuntimeError("Corrected legacy trace history does not contain 20 epochs")
        payload = "".join(self._trace_lines)
        return {
            "rank": self.accelerator.process_index,
            "record_count": len(self._trace_lines),
            "query_counts_by_epoch": [
                len(self._trace_query_ids_by_epoch[epoch]) for epoch in range(EXPECTED_EPOCHS)
            ],
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "jsonl": payload,
        }

    def validation_history(self) -> tuple[dict[str, Any], ...]:
        if len(self._validation_records) != EXPECTED_EPOCHS:
            raise RuntimeError("Corrected legacy validation history must contain 20 records")
        if [record["epoch"] for record in self._validation_records] != list(range(1, 21)):
            raise RuntimeError("Corrected legacy validation epochs changed")
        if [record["global_step"] for record in self._validation_records] != list(range(4, 81, 4)):
            raise RuntimeError("Corrected legacy validation steps changed")
        return tuple(self._validation_records)


__all__ = ["CorrectedLegacyDiagnosticTrainer"]
