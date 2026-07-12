from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from .hashing import stable_int64_hash
from .markup import SLOT_TOKEN
from .sampling import validate_sampling_trace


INVALID_PASSAGE_INDEX = -1


class RetrievalBatchCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        passage_text_by_passage_id: Dict[str, str],
        *,
        max_len_query: int,
        max_len_passage: int,
    ):
        self.tokenizer = tokenizer
        self.passage_text_by_passage_id = passage_text_by_passage_id
        self.max_len_query = int(max_len_query)
        self.max_len_passage = int(max_len_passage)
        self.slot_token_id = int(self.tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
        if self.slot_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"{SLOT_TOKEN} is not in the tokenizer vocabulary")

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("RetrievalBatchCollator received an empty batch")
        for row_index, example in enumerate(examples):
            if type(example.get("is_dummy")) is not bool:
                raise ValueError(
                    f"Batch row {row_index} must define reserved boolean field is_dummy"
                )
        real_examples = [example for example in examples if not example["is_dummy"]]
        if not real_examples:
            raise ValueError(
                "Step 4 forbids an all-dummy local batch; the global batch plan must give every rank a real query"
            )

        query_texts: List[str] = [str(ex["query_text"]) for ex in real_examples]

        positive_ids_per_query: List[List[str]] = [
            [str(x) for x in (ex.get("positive_passage_ids") or [])] for ex in real_examples
        ]
        if any(len(x) < 1 for x in positive_ids_per_query):
            raise ValueError("Found a query with empty positive_passage_ids")

        candidate_ids_per_query: List[List[str]] = [
            [str(x) for x in (ex.get("candidate_passage_ids") or [])] for ex in real_examples
        ]
        if any(len(x) < 1 for x in candidate_ids_per_query):
            raise ValueError("Found a query with empty candidate_passage_ids")

        flat_candidate_ids: List[str] = [pid for row in candidate_ids_per_query for pid in row]
        passage_texts: List[str] = [self.passage_text_by_passage_id[pid] for pid in flat_candidate_ids]

        self.tokenizer.truncation_side = "left"
        query_tok = self.tokenizer(
            query_texts,
            truncation=True,
            max_length=self.max_len_query,
            padding=True,
            return_tensors="pt",
        )
        slot_counts = (query_tok["input_ids"] == self.slot_token_id).sum(dim=1).tolist()
        if any(c != 1 for c in slot_counts):
            bad = [i for i, c in enumerate(slot_counts) if c != 1]
            raise ValueError(f"Each query must contain exactly one {SLOT_TOKEN}; bad_rows={bad}")

        self.tokenizer.truncation_side = "right"
        passage_tok = self.tokenizer(
            passage_texts,
            truncation=True,
            max_length=self.max_len_passage,
            padding=True,
            return_tensors="pt",
        )

        positive_hashes: List[List[int]] = [
            [stable_int64_hash(pid) for pid in pids] for pids in positive_ids_per_query
        ]
        max_pos = max(len(x) for x in positive_hashes)
        pos_hash_tensor = torch.full((len(positive_hashes), max_pos), -1, dtype=torch.long)
        for row_idx, row in enumerate(positive_hashes):
            if not row:
                continue
            pos_hash_tensor[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)

        passage_hash_tensor = torch.tensor([stable_int64_hash(pid) for pid in flat_candidate_ids], dtype=torch.long)

        return {
            "query_input_ids": query_tok["input_ids"],
            "query_attention_mask": query_tok["attention_mask"],
            "passage_input_ids": passage_tok["input_ids"],
            "passage_attention_mask": passage_tok["attention_mask"],
            "passage_id_hashes": passage_hash_tensor,
            "positive_id_hashes": pos_hash_tensor,
            "valid_query_count": torch.tensor(len(real_examples), dtype=torch.long),
        }


class ControlledRetrievalBatchCollator:
    """Query-only collator for the integer-indexed controlled experiment."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        corpus_size: int,
        max_len_query: int,
    ) -> None:
        if type(corpus_size) is not int or corpus_size < 1:
            raise ValueError("corpus_size must be a positive exact int")
        if type(max_len_query) is not int or max_len_query < 1:
            raise ValueError("max_len_query must be a positive exact int")
        self.tokenizer = tokenizer
        self.corpus_size = corpus_size
        self.max_len_query = max_len_query
        self.slot_token_id = int(self.tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
        if self.slot_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"{SLOT_TOKEN} is not in the tokenizer vocabulary")

    def _validate_index_list(self, name: str, value: object) -> List[int]:
        if type(value) is not list or not value:
            raise ValueError(f"{name} must be a non-empty exact list")
        indices: List[int] = []
        for position, passage_index in enumerate(value):
            if type(passage_index) is not int:
                raise TypeError(
                    f"{name}[{position}] must be an exact int, "
                    f"not {type(passage_index).__name__}"
                )
            if passage_index < 0 or passage_index >= self.corpus_size:
                raise ValueError(
                    f"{name}[{position}]={passage_index} is outside "
                    f"[0, {self.corpus_size})"
                )
            indices.append(passage_index)
        if len(indices) != len(set(indices)):
            raise ValueError(f"{name} contains duplicate passage indices")
        return indices

    @staticmethod
    def _pad_indices(rows: List[List[int]]) -> torch.Tensor:
        width = max(len(row) for row in rows)
        result = torch.full(
            (len(rows), width),
            INVALID_PASSAGE_INDEX,
            dtype=torch.long,
        )
        for row_index, row in enumerate(rows):
            result[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        return result

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not examples:
            raise ValueError("ControlledRetrievalBatchCollator received an empty batch")
        for row_index, example in enumerate(examples):
            if type(example.get("is_dummy")) is not bool:
                raise ValueError(
                    f"Batch row {row_index} must define reserved boolean field is_dummy"
                )
        real_examples = [example for example in examples if not example["is_dummy"]]
        if not real_examples:
            raise ValueError(
                "Step 4 forbids an all-dummy local batch; the global batch plan must give every rank a real query"
            )

        query_texts: List[str] = []
        positive_indices_per_query: List[List[int]] = []
        candidate_indices_per_query: List[List[int]] = []
        sampling_traces: List[Dict[str, Any]] = []
        for row_index, example in enumerate(real_examples):
            query_text = example.get("query_text")
            if type(query_text) is not str or not query_text:
                raise ValueError(f"Real batch row {row_index} has invalid query_text")
            positives = self._validate_index_list(
                f"batch[{row_index}].positive_passage_indices",
                example.get("positive_passage_indices"),
            )
            candidates = self._validate_index_list(
                f"batch[{row_index}].candidate_passage_indices",
                example.get("candidate_passage_indices"),
            )
            if not set(positives).intersection(candidates):
                raise ValueError(
                    f"Real batch row {row_index} has no sampled candidate among its all-gold targets"
                )

            trace = example.get("sampling_trace")
            validate_sampling_trace(trace)
            if trace["query_id"] != example.get("query_id"):
                raise ValueError(f"Real batch row {row_index} trace/query_id mismatch")
            if trace["doc_id"] != example.get("doc_id"):
                raise ValueError(f"Real batch row {row_index} trace/doc_id mismatch")
            if trace["trace_sha256"] != example.get("sampling_trace_sha256"):
                raise ValueError(f"Real batch row {row_index} trace checksum field mismatch")

            query_texts.append(query_text)
            positive_indices_per_query.append(positives)
            candidate_indices_per_query.append(candidates)
            sampling_traces.append(trace)

        self.tokenizer.truncation_side = "left"
        query_tok = self.tokenizer(
            query_texts,
            truncation=True,
            max_length=self.max_len_query,
            padding=True,
            return_tensors="pt",
        )
        slot_counts = (query_tok["input_ids"] == self.slot_token_id).sum(dim=1).tolist()
        if any(count != 1 for count in slot_counts):
            bad = [index for index, count in enumerate(slot_counts) if count != 1]
            raise ValueError(f"Each query must contain exactly one {SLOT_TOKEN}; bad_rows={bad}")

        return {
            "query_input_ids": query_tok["input_ids"],
            "query_attention_mask": query_tok["attention_mask"],
            "candidate_passage_indices": self._pad_indices(candidate_indices_per_query),
            "positive_passage_indices": self._pad_indices(positive_indices_per_query),
            "sampling_traces": sampling_traces,
            "valid_query_count": torch.tensor(len(real_examples), dtype=torch.long),
        }
