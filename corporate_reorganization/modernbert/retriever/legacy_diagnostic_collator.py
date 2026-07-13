"""Strict query-only collator for the corrected legacy-style diagnostic."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .data import PassageIndexTable
from .legacy_diagnostic_sampling import (
    CANDIDATE_OCCURRENCES_PER_QUERY,
    validate_legacy_diagnostic_trace,
)
from .markup import SLOT_TOKEN


INVALID_PASSAGE_INDEX = -1


class CorrectedLegacyDiagnosticCollator:
    def __init__(
        self,
        tokenizer,
        *,
        passage_index_table: PassageIndexTable,
        max_len_query: int,
    ) -> None:
        if not isinstance(passage_index_table, PassageIndexTable):
            raise TypeError("passage_index_table must be a PassageIndexTable")
        if len(passage_index_table) != 5_286:
            raise ValueError("Corrected legacy collator requires the exact 5,286-passage index")
        if type(max_len_query) is not int or max_len_query != 4_096:
            raise ValueError("Corrected legacy max query length must be exactly 4096")
        self.tokenizer = tokenizer
        self.passage_index_table = passage_index_table
        self.max_len_query = max_len_query
        self.slot_token_id = int(tokenizer.convert_tokens_to_ids(SLOT_TOKEN))
        if self.slot_token_id == tokenizer.unk_token_id:
            raise ValueError(f"{SLOT_TOKEN} is not in the tokenizer vocabulary")

    def _indices(self, name: str, value: object, *, unique: bool) -> list[int]:
        if type(value) is not list or not value:
            raise ValueError(f"{name} must be a non-empty exact list")
        result: list[int] = []
        for position, item in enumerate(value):
            if type(item) is not int:
                raise TypeError(f"{name}[{position}] must be an exact int")
            self.passage_index_table.id_for_index(item)
            result.append(item)
        if unique and len(result) != len(set(result)):
            raise ValueError(f"{name} must contain unique indices")
        return result

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], *, padding: int) -> torch.Tensor:
        width = max(map(len, rows))
        result = torch.full((len(rows), width), padding, dtype=torch.long)
        for row_index, row in enumerate(rows):
            result[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        return result

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(examples, Sequence) or not examples:
            raise ValueError("Corrected legacy collator received an empty batch")
        for row_index, example in enumerate(examples):
            if not isinstance(example, Mapping) or type(example.get("is_dummy")) is not bool:
                raise TypeError(
                    f"Corrected legacy batch row {row_index} must define boolean is_dummy"
                )
        real_examples = [example for example in examples if not example["is_dummy"]]
        if not real_examples:
            raise ValueError("Corrected legacy batching forbids an all-sentinel local batch")

        query_texts: list[str] = []
        positive_rows: list[list[int]] = []
        candidate_rows: list[list[int]] = []
        multiplicity_rows: list[list[int]] = []
        traces: list[dict[str, Any]] = []
        for row_index, example in enumerate(real_examples):
            query_text = example.get("query_text")
            if type(query_text) is not str or not query_text:
                raise ValueError(f"Corrected legacy batch row {row_index} has invalid query_text")
            positives = self._indices(
                f"batch[{row_index}].positive_passage_indices",
                example.get("positive_passage_indices"),
                unique=True,
            )
            candidates = self._indices(
                f"batch[{row_index}].unique_candidate_passage_indices",
                example.get("unique_candidate_passage_indices"),
                unique=True,
            )
            multiplicities = example.get("candidate_multiplicities")
            if (
                type(multiplicities) is not list
                or len(multiplicities) != len(candidates)
                or any(type(value) is not int or value < 1 for value in multiplicities)
                or sum(multiplicities) != CANDIDATE_OCCURRENCES_PER_QUERY
            ):
                raise ValueError(
                    f"Corrected legacy batch row {row_index} has invalid candidate multiplicities"
                )
            if candidates != sorted(candidates):
                raise ValueError("Corrected legacy unique candidate indices must be sorted")
            if not set(positives).intersection(candidates):
                raise ValueError("Corrected legacy candidate row contains no selected gold")

            occurrence_indices = self._indices(
                f"batch[{row_index}].candidate_passage_occurrence_indices",
                example.get("candidate_passage_occurrence_indices"),
                unique=False,
            )
            if len(occurrence_indices) != CANDIDATE_OCCURRENCES_PER_QUERY:
                raise ValueError("Corrected legacy occurrence row must contain exactly 64 indices")
            occurrence_counts = Counter(occurrence_indices)
            if candidates != sorted(occurrence_counts) or multiplicities != [
                occurrence_counts[index] for index in candidates
            ]:
                raise ValueError("Corrected legacy unique candidates do not reconstruct occurrences")

            trace = example.get("sampling_trace")
            validate_legacy_diagnostic_trace(trace)
            if trace["query_id"] != example.get("query_id"):
                raise ValueError("Corrected legacy trace/query identity mismatch")
            if trace["doc_id"] != example.get("doc_id"):
                raise ValueError("Corrected legacy trace/case identity mismatch")
            if trace["trace_sha256"] != example.get("sampling_trace_sha256"):
                raise ValueError("Corrected legacy trace checksum field mismatch")
            trace_occurrence_indices = self.passage_index_table.indices_for_ids(
                [record["passage_id"] for record in trace["occurrences"]]
            )
            if trace_occurrence_indices != occurrence_indices:
                raise ValueError("Corrected legacy trace and occurrence indices disagree")

            query_texts.append(query_text)
            positive_rows.append(positives)
            candidate_rows.append(candidates)
            multiplicity_rows.append(list(multiplicities))
            traces.append(trace)

        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            query_tokens = self.tokenizer(
                query_texts,
                truncation=True,
                max_length=self.max_len_query,
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.truncation_side = original_side
        if (
            not isinstance(query_tokens, Mapping)
            or not torch.is_tensor(query_tokens.get("input_ids"))
            or not torch.is_tensor(query_tokens.get("attention_mask"))
            or query_tokens["input_ids"].dtype != torch.long
            or query_tokens["attention_mask"].dtype != torch.long
            or query_tokens["input_ids"].shape != query_tokens["attention_mask"].shape
            or query_tokens["input_ids"].shape[0] != len(real_examples)
        ):
            raise TypeError("Corrected legacy query tokenizer returned malformed tensors")
        slot_counts = query_tokens["input_ids"].eq(self.slot_token_id).sum(dim=1)
        if not slot_counts.eq(1).all():
            bad_rows = slot_counts.ne(1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Each corrected legacy query must retain one {SLOT_TOKEN}; bad={bad_rows}")

        return {
            "query_input_ids": query_tokens["input_ids"],
            "query_attention_mask": query_tokens["attention_mask"],
            "candidate_passage_indices": self._pad(
                candidate_rows,
                padding=INVALID_PASSAGE_INDEX,
            ),
            "candidate_multiplicities": self._pad(multiplicity_rows, padding=0),
            "positive_passage_indices": self._pad(
                positive_rows,
                padding=INVALID_PASSAGE_INDEX,
            ),
            "sampling_traces": traces,
            "valid_query_count": torch.tensor(len(real_examples), dtype=torch.long),
        }


__all__ = ["CorrectedLegacyDiagnosticCollator"]
