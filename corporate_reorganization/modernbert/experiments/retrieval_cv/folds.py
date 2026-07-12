"""Freeze and validate deterministic case-disjoint retrieval folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FOLD_CAPACITIES: Tuple[int, ...] = (9, 9, 8, 8, 8)
EXPECTED_CASES = 42
EXPECTED_QUERIES = 490
EXPECTED_PASSAGES = 5_286
EXPECTED_FINAL_QUERY_LOAD = 98
EXPECTED_FINAL_PASSAGE_LOADS = (1_054, 1_060, 1_055, 1_055, 1_062)
IDEAL_OBJECTIVE = Fraction(2, 5)


@dataclass(frozen=True)
class CaseLoad:
    case_id: str
    queries: int
    passages: int


@dataclass
class FoldState:
    fold_id: int
    capacity: int
    case_ids: List[str] = field(default_factory=list)
    queries: int = 0
    passages: int = 0

    def copy(self) -> "FoldState":
        return FoldState(
            fold_id=self.fold_id,
            capacity=self.capacity,
            case_ids=list(self.case_ids),
            queries=self.queries,
            passages=self.passages,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _case_number(case_id: str) -> int:
    if not case_id or not case_id.isdecimal() or str(int(case_id)) != case_id:
        raise ValueError(f"Case ID must be a canonical non-negative integer string: {case_id!r}")
    return int(case_id)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank JSONL line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not a JSON object")
            yield value


def validate_case_loads(
    case_loads: Sequence[CaseLoad],
    *,
    capacities: Sequence[int],
) -> None:
    if not capacities or any(int(capacity) < 1 for capacity in capacities):
        raise ValueError(f"Fold capacities must be positive: {capacities}")
    if sum(int(capacity) for capacity in capacities) != len(case_loads):
        raise ValueError(
            f"Fold capacities sum to {sum(capacities)}, but there are {len(case_loads)} cases"
        )

    case_ids = [load.case_id for load in case_loads]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs in case loads")
    for load in case_loads:
        _case_number(load.case_id)
        if load.queries < 1:
            raise ValueError(f"Case {load.case_id} has no queries")
        if load.passages < 1:
            raise ValueError(f"Case {load.case_id} has no passages")


def _objective(
    folds: Sequence[FoldState],
    *,
    total_queries: int,
    total_passages: int,
) -> Fraction:
    if total_queries < 1 or total_passages < 1:
        raise ValueError("Objective totals must be positive")
    return sum(
        (
            Fraction(fold.queries, total_queries) ** 2
            + Fraction(fold.passages, total_passages) ** 2
        )
        for fold in folds
    )


def _priority(
    load: CaseLoad,
    *,
    total_queries: int,
    total_passages: int,
) -> Fraction:
    return max(
        Fraction(load.queries, total_queries),
        Fraction(load.passages, total_passages),
    )


def _fold_record(
    fold: FoldState,
    *,
    total_queries: int,
    total_passages: int,
) -> Dict[str, Any]:
    return {
        "fold_id": fold.fold_id,
        "capacity": fold.capacity,
        "num_cases": len(fold.case_ids),
        "case_ids": sorted(fold.case_ids, key=_case_number),
        "queries": fold.queries,
        "passages": fold.passages,
        "query_share": _fraction_text(Fraction(fold.queries, total_queries)),
        "passage_share": _fraction_text(Fraction(fold.passages, total_passages)),
    }


def assign_cases_greedily(
    case_loads: Sequence[CaseLoad],
    *,
    capacities: Sequence[int],
) -> Tuple[List[FoldState], List[Dict[str, Any]], List[str]]:
    validate_case_loads(case_loads, capacities=capacities)
    total_queries = sum(load.queries for load in case_loads)
    total_passages = sum(load.passages for load in case_loads)
    loads_by_id = {load.case_id: load for load in case_loads}
    case_order = sorted(
        loads_by_id,
        key=lambda case_id: (
            -_priority(
                loads_by_id[case_id],
                total_queries=total_queries,
                total_passages=total_passages,
            ),
            _case_number(case_id),
        ),
    )

    folds = [
        FoldState(fold_id=fold_id, capacity=int(capacity))
        for fold_id, capacity in enumerate(capacities)
    ]
    trace: List[Dict[str, Any]] = []
    for step, case_id in enumerate(case_order, start=1):
        load = loads_by_id[case_id]
        choices: List[Tuple[Fraction, int, int]] = []
        for fold in folds:
            if len(fold.case_ids) >= fold.capacity:
                continue
            trial = [item.copy() for item in folds]
            candidate = trial[fold.fold_id]
            candidate.case_ids.append(case_id)
            candidate.queries += load.queries
            candidate.passages += load.passages
            choices.append(
                (
                    _objective(
                        trial,
                        total_queries=total_queries,
                        total_passages=total_passages,
                    ),
                    len(fold.case_ids),
                    fold.fold_id,
                )
            )
        if not choices:
            raise ValueError(f"No non-full fold remains for case {case_id}")
        chosen_objective, prior_case_count, chosen_fold_id = min(choices)
        chosen_fold = folds[chosen_fold_id]
        chosen_fold.case_ids.append(case_id)
        chosen_fold.queries += load.queries
        chosen_fold.passages += load.passages
        trace.append(
            {
                "step": step,
                "case_id": case_id,
                "queries": load.queries,
                "passages": load.passages,
                "priority": _fraction_text(
                    _priority(
                        load,
                        total_queries=total_queries,
                        total_passages=total_passages,
                    )
                ),
                "chosen_fold_id": chosen_fold_id,
                "chosen_fold_prior_case_count": prior_case_count,
                "objective_after": _fraction_text(chosen_objective),
            }
        )

    for fold in folds:
        if len(fold.case_ids) != fold.capacity:
            raise ValueError(
                f"Fold {fold.fold_id} has {len(fold.case_ids)} cases, expected {fold.capacity}"
            )
    return folds, trace, case_order


def refine_pair_swaps(
    folds: Sequence[FoldState],
    *,
    case_loads: Sequence[CaseLoad],
) -> Tuple[List[FoldState], List[Dict[str, Any]]]:
    refined = [fold.copy() for fold in folds]
    loads_by_id = {load.case_id: load for load in case_loads}
    folded_case_ids = [
        case_id for fold in refined for case_id in fold.case_ids
    ]
    if (
        set(loads_by_id) != set(folded_case_ids)
        or len(folded_case_ids) != len(loads_by_id)
    ):
        raise ValueError("Fold membership and case loads do not cover the same cases")
    for expected_fold_id, fold in enumerate(refined):
        if fold.fold_id != expected_fold_id:
            raise ValueError(
                f"Fold position {expected_fold_id} has fold_id={fold.fold_id}"
            )
        if len(fold.case_ids) != fold.capacity:
            raise ValueError(
                f"Fold {fold.fold_id} has {len(fold.case_ids)} cases, "
                f"expected capacity {fold.capacity}"
            )
        expected_queries = sum(
            loads_by_id[case_id].queries for case_id in fold.case_ids
        )
        expected_passages = sum(
            loads_by_id[case_id].passages for case_id in fold.case_ids
        )
        if (fold.queries, fold.passages) != (
            expected_queries,
            expected_passages,
        ):
            raise ValueError(
                f"Fold {fold.fold_id} stored loads do not match its cases: "
                f"stored={(fold.queries, fold.passages)}, "
                f"expected={(expected_queries, expected_passages)}"
            )
    total_queries = sum(load.queries for load in case_loads)
    total_passages = sum(load.passages for load in case_loads)
    swaps: List[Dict[str, Any]] = []

    while True:
        current_objective = _objective(
            refined,
            total_queries=total_queries,
            total_passages=total_passages,
        )
        improvements: List[
            Tuple[Fraction, int, int, int, int, str, str]
        ] = []
        for lower_fold_id in range(len(refined)):
            for higher_fold_id in range(lower_fold_id + 1, len(refined)):
                lower_fold = refined[lower_fold_id]
                higher_fold = refined[higher_fold_id]
                for lower_case_id in sorted(lower_fold.case_ids, key=_case_number):
                    for higher_case_id in sorted(higher_fold.case_ids, key=_case_number):
                        lower_load = loads_by_id[lower_case_id]
                        higher_load = loads_by_id[higher_case_id]
                        trial = [fold.copy() for fold in refined]
                        trial_lower = trial[lower_fold_id]
                        trial_higher = trial[higher_fold_id]
                        trial_lower.queries += higher_load.queries - lower_load.queries
                        trial_lower.passages += higher_load.passages - lower_load.passages
                        trial_higher.queries += lower_load.queries - higher_load.queries
                        trial_higher.passages += lower_load.passages - higher_load.passages
                        candidate_objective = _objective(
                            trial,
                            total_queries=total_queries,
                            total_passages=total_passages,
                        )
                        if candidate_objective < current_objective:
                            improvements.append(
                                (
                                    candidate_objective,
                                    lower_fold_id,
                                    higher_fold_id,
                                    _case_number(lower_case_id),
                                    _case_number(higher_case_id),
                                    lower_case_id,
                                    higher_case_id,
                                )
                            )
        if not improvements:
            break

        (
            next_objective,
            lower_fold_id,
            higher_fold_id,
            _,
            _,
            lower_case_id,
            higher_case_id,
        ) = min(improvements)
        lower_load = loads_by_id[lower_case_id]
        higher_load = loads_by_id[higher_case_id]
        lower_fold = refined[lower_fold_id]
        higher_fold = refined[higher_fold_id]
        lower_fold.case_ids.remove(lower_case_id)
        higher_fold.case_ids.remove(higher_case_id)
        lower_fold.case_ids.append(higher_case_id)
        higher_fold.case_ids.append(lower_case_id)
        lower_fold.queries += higher_load.queries - lower_load.queries
        lower_fold.passages += higher_load.passages - lower_load.passages
        higher_fold.queries += lower_load.queries - higher_load.queries
        higher_fold.passages += lower_load.passages - higher_load.passages
        swaps.append(
            {
                "step": len(swaps) + 1,
                "lower_fold_id": lower_fold_id,
                "higher_fold_id": higher_fold_id,
                "case_from_lower_fold": lower_case_id,
                "case_from_higher_fold": higher_case_id,
                "objective_before": _fraction_text(current_objective),
                "objective_after": _fraction_text(next_objective),
            }
        )

    return refined, swaps


def _verify_dataset_file_hashes(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    output_files = manifest.get("output_files")
    if not isinstance(output_files, dict):
        raise ValueError("Dataset manifest output_files must be an object")
    expected_paths = {
        "cases.jsonl",
        "corpus.jsonl",
        "queries/all.jsonl",
        "pools/candidates_by_case.json",
        "pools/candidates_global.json",
    }
    if set(output_files) != expected_paths:
        raise ValueError(
            f"Dataset output file set changed: expected={sorted(expected_paths)}, "
            f"found={sorted(output_files)}"
        )
    for relative_path, record in output_files.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid dataset output record for {relative_path}")
        path = dataset_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset output: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != record.get("sha256"):
            raise ValueError(
                f"Dataset output hash mismatch for {relative_path}: "
                f"expected={record.get('sha256')}, found={actual_sha256}"
            )


def load_case_loads(dataset_dir: Path) -> Tuple[List[CaseLoad], Dict[str, Any]]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError(
            f"Expected dataset schema_version=2, found {manifest.get('schema_version')!r}"
        )
    _verify_dataset_file_hashes(dataset_dir, manifest)

    counts = manifest.get("counts")
    diagnostics = manifest.get("diagnostics")
    if not isinstance(counts, dict) or not isinstance(diagnostics, dict):
        raise ValueError("Dataset manifest counts and diagnostics must be objects")
    if counts.get("cases") != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} cases, found {counts.get('cases')!r}")
    if counts.get("queries") != EXPECTED_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_QUERIES} queries, found {counts.get('queries')!r}"
        )
    if counts.get("passages") != EXPECTED_PASSAGES:
        raise ValueError(
            f"Expected {EXPECTED_PASSAGES} passages, found {counts.get('passages')!r}"
        )

    manifest_query_counts = diagnostics.get("query_counts_by_case")
    manifest_passage_counts = diagnostics.get("passage_counts_by_case")
    if not isinstance(manifest_query_counts, dict) or not isinstance(
        manifest_passage_counts, dict
    ):
        raise ValueError("Dataset manifest lacks per-case query/passage counts")

    case_ids: List[str] = []
    for record in _iter_jsonl(dataset_dir / "cases.jsonl"):
        case_id = str(record.get("doc_id"))
        _case_number(case_id)
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs in cases.jsonl")

    query_counts = {case_id: 0 for case_id in case_ids}
    query_ids = set()
    for record in _iter_jsonl(dataset_dir / "queries/all.jsonl"):
        query_id = str(record.get("query_id"))
        case_id = str(record.get("doc_id"))
        if query_id in query_ids:
            raise ValueError(f"Duplicate query ID: {query_id}")
        if case_id not in query_counts:
            raise ValueError(f"Query references unknown case {case_id}: {query_id}")
        query_ids.add(query_id)
        query_counts[case_id] += 1

    passage_counts = {case_id: 0 for case_id in case_ids}
    passage_ids = set()
    for record in _iter_jsonl(dataset_dir / "corpus.jsonl"):
        passage_id = str(record.get("passage_id"))
        case_id = str(record.get("doc_id"))
        if passage_id in passage_ids:
            raise ValueError(f"Duplicate passage ID: {passage_id}")
        if case_id not in passage_counts:
            raise ValueError(f"Passage references unknown case {case_id}: {passage_id}")
        passage_ids.add(passage_id)
        passage_counts[case_id] += 1

    normalized_manifest_queries = {
        str(case_id): int(value)
        for case_id, value in manifest_query_counts.items()
    }
    normalized_manifest_passages = {
        str(case_id): int(value)
        for case_id, value in manifest_passage_counts.items()
    }
    if query_counts != normalized_manifest_queries:
        raise ValueError("Per-case query counts disagree with dataset manifest")
    if passage_counts != normalized_manifest_passages:
        raise ValueError("Per-case passage counts disagree with dataset manifest")
    if sum(query_counts.values()) != EXPECTED_QUERIES:
        raise ValueError("Query readback total changed")
    if sum(passage_counts.values()) != EXPECTED_PASSAGES:
        raise ValueError("Passage readback total changed")

    case_loads = [
        CaseLoad(
            case_id=case_id,
            queries=query_counts[case_id],
            passages=passage_counts[case_id],
        )
        for case_id in sorted(case_ids, key=_case_number)
    ]
    validate_case_loads(case_loads, capacities=FOLD_CAPACITIES)

    repo_root = _repo_root().resolve()
    try:
        relative_manifest_path = manifest_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Dataset manifest must be inside repository root {repo_root}: {manifest_path}"
        ) from exc
    dataset_record = {
        "dataset_schema_version": 2,
        "dataset_manifest_path": relative_manifest_path.as_posix(),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "output_sha256": {
            relative_path: str(record["sha256"])
            for relative_path, record in sorted(manifest["output_files"].items())
        },
    }
    return case_loads, dataset_record


def _folds_record(
    folds: Sequence[FoldState],
    *,
    total_queries: int,
    total_passages: int,
) -> List[Dict[str, Any]]:
    return [
        _fold_record(
            fold,
            total_queries=total_queries,
            total_passages=total_passages,
        )
        for fold in folds
    ]


def _role_record(
    fold_ids: Sequence[int],
    folds: Sequence[FoldState],
) -> Dict[str, Any]:
    selected = [folds[fold_id] for fold_id in fold_ids]
    case_ids = sorted(
        [case_id for fold in selected for case_id in fold.case_ids],
        key=_case_number,
    )
    return {
        "fold_ids": list(fold_ids),
        "num_cases": len(case_ids),
        "case_ids": case_ids,
        "queries": sum(fold.queries for fold in selected),
        "passages": sum(fold.passages for fold in selected),
    }


def _validate_final_folds(
    folds: Sequence[FoldState],
    *,
    case_loads: Sequence[CaseLoad],
) -> None:
    if len(folds) != len(FOLD_CAPACITIES):
        raise ValueError(f"Expected {len(FOLD_CAPACITIES)} folds, found {len(folds)}")
    expected_case_ids = {load.case_id for load in case_loads}
    found_case_ids = [case_id for fold in folds for case_id in fold.case_ids]
    if set(found_case_ids) != expected_case_ids or len(found_case_ids) != len(
        expected_case_ids
    ):
        raise ValueError("Final folds are not an exhaustive one-to-one case partition")
    for expected_fold_id, (fold, capacity) in enumerate(
        zip(folds, FOLD_CAPACITIES)
    ):
        if fold.fold_id != expected_fold_id:
            raise ValueError(
                f"Fold position {expected_fold_id} has fold_id={fold.fold_id}"
            )
        if fold.capacity != capacity or len(fold.case_ids) != capacity:
            raise ValueError(
                f"Fold {fold.fold_id} capacity mismatch: "
                f"capacity={fold.capacity}, cases={len(fold.case_ids)}, expected={capacity}"
            )

    if [fold.queries for fold in folds] != [EXPECTED_FINAL_QUERY_LOAD] * 5:
        raise ValueError(
            f"Refined fold query loads changed: {[fold.queries for fold in folds]}"
        )
    if tuple(fold.passages for fold in folds) != EXPECTED_FINAL_PASSAGE_LOADS:
        raise ValueError(
            f"Refined fold passage loads changed: {[fold.passages for fold in folds]}"
        )


def build_fold_manifest(dataset_dir: Path) -> Dict[str, Any]:
    case_loads, dataset_record = load_case_loads(dataset_dir)
    total_queries = sum(load.queries for load in case_loads)
    total_passages = sum(load.passages for load in case_loads)
    greedy_folds, greedy_trace, case_order = assign_cases_greedily(
        case_loads,
        capacities=FOLD_CAPACITIES,
    )
    refined_folds, swaps = refine_pair_swaps(
        greedy_folds,
        case_loads=case_loads,
    )
    _validate_final_folds(refined_folds, case_loads=case_loads)

    greedy_objective = _objective(
        greedy_folds,
        total_queries=total_queries,
        total_passages=total_passages,
    )
    refined_objective = _objective(
        refined_folds,
        total_queries=total_queries,
        total_passages=total_passages,
    )
    if refined_objective >= greedy_objective:
        raise ValueError("Pair-swap refinement did not improve the greedy objective")

    rotations = []
    fold_ids = list(range(len(refined_folds)))
    for outer_fold in fold_ids:
        test_fold = outer_fold
        validation_fold = (outer_fold + 1) % len(refined_folds)
        train_folds = [
            fold_id
            for fold_id in fold_ids
            if fold_id not in {test_fold, validation_fold}
        ]
        train = _role_record(train_folds, refined_folds)
        validation = _role_record([validation_fold], refined_folds)
        test = _role_record([test_fold], refined_folds)
        if set(train["case_ids"]) & set(validation["case_ids"]):
            raise ValueError(f"Outer fold {outer_fold}: train/validation overlap")
        if set(train["case_ids"]) & set(test["case_ids"]):
            raise ValueError(f"Outer fold {outer_fold}: train/test overlap")
        if set(validation["case_ids"]) & set(test["case_ids"]):
            raise ValueError(f"Outer fold {outer_fold}: validation/test overlap")
        if len(train["case_ids"]) + len(validation["case_ids"]) + len(test["case_ids"]) != EXPECTED_CASES:
            raise ValueError(f"Outer fold {outer_fold}: roles are not exhaustive")
        rotations.append(
            {
                "outer_fold": outer_fold,
                "train": train,
                "validation": validation,
                "test": test,
            }
        )

    case_load_records = {
        load.case_id: {
            "queries": load.queries,
            "passages": load.passages,
            "priority": _fraction_text(
                _priority(
                    load,
                    total_queries=total_queries,
                    total_passages=total_passages,
                )
            ),
        }
        for load in sorted(case_loads, key=lambda item: _case_number(item.case_id))
    }
    source_path = Path(__file__).resolve()
    source_relative_path = source_path.relative_to(_repo_root().resolve()).as_posix()
    return {
        "schema_version": 1,
        "manifest_type": "retrieval_case_folds",
        "generator": {
            "source_path": source_relative_path,
            "source_sha256": _sha256(source_path),
        },
        "dataset": dataset_record,
        "totals": {
            "cases": len(case_loads),
            "queries": total_queries,
            "passages": total_passages,
        },
        "algorithm": {
            "version": "greedy_best_pair_swap_v1",
            "capacities": list(FOLD_CAPACITIES),
            "case_priority": {
                "formula": "max(case_queries/total_queries, case_passages/total_passages)",
                "order": "descending priority, then numeric case ID",
                "arithmetic": "exact rational",
            },
            "objective": {
                "formula": "sum_f[(fold_queries/total_queries)^2 + (fold_passages/total_passages)^2]",
                "ideal_equal_load_value": _fraction_text(IDEAL_OBJECTIVE),
                "arithmetic": "exact rational",
            },
            "greedy_tie_break": [
                "resulting objective",
                "current fold case count",
                "fold ID",
            ],
            "refinement": {
                "operation": "cross-fold one-case-for-one-case swap",
                "acceptance": "strict objective decrease only",
                "selection": "best objective decrease",
                "tie_break": [
                    "resulting objective",
                    "lower fold ID",
                    "higher fold ID",
                    "numeric case ID from lower fold",
                    "numeric case ID from higher fold",
                ],
                "stop": "no strictly improving pair swap remains",
            },
        },
        "case_priority_order": case_order,
        "case_loads": case_load_records,
        "greedy": {
            "objective": _fraction_text(greedy_objective),
            "objective_excess_over_equal_load": _fraction_text(
                greedy_objective - IDEAL_OBJECTIVE
            ),
            "folds": _folds_record(
                greedy_folds,
                total_queries=total_queries,
                total_passages=total_passages,
            ),
            "assignment_trace": greedy_trace,
        },
        "pair_swap_refinement": {
            "num_swaps": len(swaps),
            "swaps": swaps,
            "objective": _fraction_text(refined_objective),
            "objective_excess_over_equal_load": _fraction_text(
                refined_objective - IDEAL_OBJECTIVE
            ),
            "strict_pairwise_local_optimum": True,
        },
        "folds": _folds_record(
            refined_folds,
            total_queries=total_queries,
            total_passages=total_passages,
        ),
        "rotations": rotations,
    }


def _canonical_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(_canonical_json_text(value))


def freeze_fold_manifest(*, dataset_dir: Path, output_path: Path) -> Dict[str, Any]:
    if os.path.lexists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing fold manifest: {output_path}")
    manifest = build_fold_manifest(dataset_dir)
    _write_json(manifest, output_path)
    return manifest


def validate_frozen_fold_manifest(
    *,
    dataset_dir: Path,
    fold_manifest_path: Path,
) -> Dict[str, Any]:
    if not fold_manifest_path.is_file():
        raise FileNotFoundError(f"Missing fold manifest: {fold_manifest_path}")
    stored = _load_json(fold_manifest_path)
    expected = build_fold_manifest(dataset_dir)
    if stored != expected:
        raise ValueError(
            f"Frozen fold manifest does not exactly match deterministic regeneration: "
            f"{fold_manifest_path}"
        )
    expected_bytes = _canonical_json_text(expected).encode("utf-8")
    if fold_manifest_path.read_bytes() != expected_bytes:
        raise ValueError(
            f"Frozen fold manifest does not use canonical deterministic bytes: "
            f"{fold_manifest_path}"
        )
    return stored


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or validate deterministic retrieval CV folds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--dataset-dir", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)
    validate_parser.add_argument("--folds", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        freeze_fold_manifest(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
        )
    elif args.command == "validate":
        validate_frozen_fold_manifest(
            dataset_dir=args.dataset_dir,
            fold_manifest_path=args.folds,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
