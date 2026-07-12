"""Step-9 scientific readback gate for five fold evaluations.

This module intentionally performs no statistical aggregation. Case/seed
aggregation, bootstrap intervals, contrasts, and figures belong to Step 12.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...retriever.data import (
    CorpusPassage,
    PassageIndexTable,
    QueryExample,
    load_corpus,
    load_queries,
)
from ...retriever.provenance import EXPECTED_DATASET_MANIFEST_LOGICAL_PATH
from ...retriever.regimes import CANONICAL_CANDIDATE_REGIMES
from ...retriever.staged_data import validate_staged_dataset_and_fold


EXPECTED_EXPERIMENT_CONFIG_SHA256 = (
    "e51f4e8097f8888adda0382dd5c9377d7fd7417e0356b176f50ab37f7002aa96"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "cce04197b7f92c851c8e1e0b1fc0ff3f2757911d646a0079236c03070442e4be"
)
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "469858f2f8e42d0b19e53ee71af690f722482120348a2fe9719b99104758e00d"
)
EXPECTED_PASSAGE_INDEX_SHA256 = (
    "641b7a6f9f77d308b9b2b4b38ab2318ffdbc61af4b4ad718caf0d3ad571ec43d"
)
EXPECTED_TOTALS = {"cases": 42, "queries": 490, "passages": 5_286}
BM25_SYSTEM_TYPE = "bm25_pyserini"
E5_SYSTEM_TYPE = "e5_base_v2"
FIXED_BASE_SYSTEM_TYPE = "fixed_untrained_modernbert_artifact"
CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE = "controlled_dual_encoder_artifact"

EXPECTED_SYSTEM_CONTRACT = tuple(
    sorted(
        [
            ("bm25_flat_plain", BM25_SYSTEM_TYPE, "flat_plain"),
            ("e5_base_v2_flat_plain", E5_SYSTEM_TYPE, "flat_plain"),
            (
                "modernbert_base_flat_masked",
                FIXED_BASE_SYSTEM_TYPE,
                "flat_masked",
            ),
            *[
                (
                    f"{view}_{sampler}_seed{seed}",
                    CONTROLLED_DUAL_ENCODER_SYSTEM_TYPE,
                    view,
                )
                for view in ("flat_masked", "structured")
                for sampler in ("global_uniform", "local_unique")
                for seed in (17, 29, 43)
            ],
        ],
        key=lambda record: record[0],
    )
)
EXPECTED_SYSTEM_IDS = tuple(record[0] for record in EXPECTED_SYSTEM_CONTRACT)
EXPECTED_RESULT_RECORDS_PER_FOLD = len(EXPECTED_SYSTEM_CONTRACT) * len(
    CANONICAL_CANDIDATE_REGIMES
)


@dataclass(frozen=True)
class _InspectedEvaluationBundle:
    record: dict[str, Any]
    case_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    passage_ids: tuple[str, ...]
    runtime_identity: dict[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evaluation artifact must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Evaluation artifact is not valid JSON: {path}") from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError(f"Evaluation artifact must be one canonical JSON object: {path}")
    return value


def _lower_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def validate_published_evaluation_bundle(**kwargs):
    """Load the ML-backed strict validator only when scientific readback runs."""

    from ...retriever.evaluator import validate_published_evaluation_bundle as validate

    return validate(**kwargs)


def _inspect_evaluation_bundle(
    output_dir: Path,
    *,
    all_queries: Sequence[QueryExample],
    corpus_by_passage_id: Mapping[str, CorpusPassage],
    expected_case_ids_by_fold: Mapping[int, tuple[str, ...]],
) -> _InspectedEvaluationBundle:
    from ...retriever.evaluator import EvaluationIdentity

    output_dir = Path(output_dir)
    config = _load_canonical(output_dir / "evaluation_config.json")
    identity_payload = config.get("identity")
    expected_identity_keys = {
        "dataset_manifest_sha256",
        "evaluation_plan_sha256",
        "experiment_config_sha256",
        "experiment_id",
        "fold_manifest_sha256",
        "outer_fold",
        "passage_index_sha256",
        "role",
    }
    if type(identity_payload) is not dict or set(identity_payload) != expected_identity_keys:
        raise ValueError("Evaluation identity schema changed")
    identity = EvaluationIdentity(**identity_payload)
    expected_identity = {
        "experiment_id": "arr_retrieval_cv_v1",
        "experiment_config_sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "fold_manifest_sha256": EXPECTED_FOLD_MANIFEST_SHA256,
        "passage_index_sha256": EXPECTED_PASSAGE_INDEX_SHA256,
        "role": "test",
    }
    for key, expected in expected_identity.items():
        if identity_payload[key] != expected:
            raise ValueError(f"Evaluation identity {key} changed")
    plan_hash = _lower_sha256(
        identity.evaluation_plan_sha256,
        name="evaluation_plan_sha256",
    )

    runtime_identity = config.get("runtime_identity")
    if type(runtime_identity) is not dict or not runtime_identity:
        raise ValueError("Evaluation runtime_identity must be one non-empty object")
    reconstructed = validate_published_evaluation_bundle(
        output_dir=output_dir,
        identity=identity,
        runtime_identity=runtime_identity,
        all_queries=all_queries,
        corpus_by_passage_id=corpus_by_passage_id,
        expected_system_contract=EXPECTED_SYSTEM_CONTRACT,
    )
    if len(reconstructed) != EXPECTED_RESULT_RECORDS_PER_FOLD:
        raise ValueError("Evaluation result coverage changed")

    case_ids = tuple(config["case_ids"])
    query_ids = tuple(config["query_ids"])
    passage_ids = tuple(config["passage_ids"])
    expected_case_ids = expected_case_ids_by_fold[identity.outer_fold]
    if case_ids != expected_case_ids:
        raise ValueError(
            f"Evaluation fold {identity.outer_fold} does not contain its exact test cases"
        )
    system_ids = tuple(system["system_id"] for system in config["systems"])
    if system_ids != EXPECTED_SYSTEM_IDS:
        raise ValueError("Evaluation system inventory changed")
    regime_names = tuple(regime["regime_name"] for regime in config["regimes"])
    if regime_names != CANONICAL_CANDIDATE_REGIMES:
        raise ValueError("Evaluation regime inventory changed")

    ranking_row_count = len((output_dir / "rankings.jsonl").read_bytes().splitlines())
    return _InspectedEvaluationBundle(
        record={
            "artifact_manifest_sha256": _sha256(
                output_dir / "artifact_manifest.json"
            ),
            "case_count": len(case_ids),
            "case_ids_sha256": _lower_sha256(
                config["case_ids_sha256"], name="case_ids_sha256"
            ),
            "evaluation_plan_sha256": plan_hash,
            "outer_fold": identity.outer_fold,
            "passage_count": len(passage_ids),
            "passage_ids_sha256": _lower_sha256(
                config["passage_ids_sha256"], name="passage_ids_sha256"
            ),
            "query_count": len(query_ids),
            "query_ids_sha256": _lower_sha256(
                config["query_ids_sha256"], name="query_ids_sha256"
            ),
            "ranking_row_count": ranking_row_count,
            "regime_names": list(regime_names),
            "result_record_count": len(reconstructed),
            "system_ids": list(system_ids),
        },
        case_ids=case_ids,
        query_ids=query_ids,
        passage_ids=passage_ids,
        runtime_identity=runtime_identity,
    )


def _require_exact_partition(
    inventories: Sequence[tuple[str, ...]],
    *,
    expected: set[str],
    name: str,
) -> None:
    seen: set[str] = set()
    for outer_fold, inventory in enumerate(inventories):
        overlap = seen.intersection(inventory)
        if overlap:
            raise ValueError(
                f"Evaluation {name} inventories overlap at fold {outer_fold}: "
                f"{sorted(overlap)}"
            )
        seen.update(inventory)
    if seen != expected:
        raise ValueError(
            f"Evaluation {name} inventories do not form the exact five-fold partition"
        )


def build_evaluation_index(
    output_dirs: Sequence[Path],
    *,
    dataset_dir: Path,
    fold_manifest_path: Path,
) -> dict[str, Any]:
    """Validate five canonical test bundles against the frozen staged study."""

    if len(output_dirs) != 5:
        raise ValueError("Step-9 evaluation index requires exactly five complete folds")
    fold_manifest = validate_staged_dataset_and_fold(
        dataset_dir=Path(dataset_dir),
        fold_manifest_path=Path(fold_manifest_path),
        expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        expected_fold_manifest_sha256=EXPECTED_FOLD_MANIFEST_SHA256,
        expected_dataset_manifest_logical_path=EXPECTED_DATASET_MANIFEST_LOGICAL_PATH,
    )
    if fold_manifest.get("totals") != EXPECTED_TOTALS:
        raise ValueError("Frozen fold-manifest totals changed")

    corpus = load_corpus(Path(dataset_dir))
    all_queries = load_queries(Path(dataset_dir), "all")
    passage_index = PassageIndexTable(corpus)
    if passage_index.sha256 != EXPECTED_PASSAGE_INDEX_SHA256:
        raise ValueError("Staged corpus passage-index SHA-256 changed")
    query_ids = [query.query_id for query in all_queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Staged queries contain duplicate query IDs")

    rotations = fold_manifest.get("rotations")
    if type(rotations) is not list or len(rotations) != 5:
        raise ValueError("Frozen fold manifest must contain exactly five rotations")
    expected_case_ids_by_fold: dict[int, tuple[str, ...]] = {}
    for outer_fold, rotation in enumerate(rotations):
        if type(rotation) is not dict or rotation.get("outer_fold") != outer_fold:
            raise ValueError("Frozen fold rotation identity/order changed")
        test_role = rotation.get("test")
        if type(test_role) is not dict or type(test_role.get("case_ids")) is not list:
            raise ValueError("Frozen fold test-role inventory changed")
        case_ids = tuple(sorted(test_role["case_ids"]))
        if (
            not case_ids
            or any(type(case_id) is not str or not case_id for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
        ):
            raise ValueError("Frozen fold test case IDs changed")
        expected_case_ids_by_fold[outer_fold] = case_ids

    inspected = [
        _inspect_evaluation_bundle(
            Path(output_dir),
            all_queries=all_queries,
            corpus_by_passage_id=corpus,
            expected_case_ids_by_fold=expected_case_ids_by_fold,
        )
        for output_dir in output_dirs
    ]
    inspected.sort(key=lambda bundle: bundle.record["outer_fold"])
    records = [bundle.record for bundle in inspected]
    if [record["outer_fold"] for record in records] != list(range(5)):
        raise ValueError("Evaluation index must cover outer folds 0..4 exactly once")
    if len({record["artifact_manifest_sha256"] for record in records}) != 5:
        raise ValueError("Each fold evaluation must have a distinct artifact manifest")
    if len({record["evaluation_plan_sha256"] for record in records}) != 5:
        raise ValueError("Each fold evaluation must have a distinct evaluation plan")
    if any(
        bundle.runtime_identity != inspected[0].runtime_identity
        for bundle in inspected[1:]
    ):
        raise ValueError("Evaluation runtime identity changed across folds")

    expected_case_ids = set().union(*map(set, expected_case_ids_by_fold.values()))
    corpus_case_ids = {passage.doc_id for passage in corpus.values()}
    query_case_ids = {query.doc_id for query in all_queries}
    if corpus_case_ids != expected_case_ids or query_case_ids != expected_case_ids:
        raise ValueError("Staged case inventories do not match the five test folds")
    if {
        "cases": len(expected_case_ids),
        "queries": len(query_ids),
        "passages": len(corpus),
    } != EXPECTED_TOTALS:
        raise ValueError("Staged study counts changed")

    _require_exact_partition(
        [bundle.case_ids for bundle in inspected],
        expected=expected_case_ids,
        name="case",
    )
    _require_exact_partition(
        [bundle.query_ids for bundle in inspected],
        expected=set(query_ids),
        name="query",
    )
    _require_exact_partition(
        [bundle.passage_ids for bundle in inspected],
        expected=set(corpus),
        name="passage",
    )
    for bundle in inspected:
        outer_fold = bundle.record["outer_fold"]
        test_role = rotations[outer_fold]["test"]
        if (
            bundle.record["case_count"] != test_role["num_cases"]
            or bundle.record["query_count"] != test_role["queries"]
            or bundle.record["passage_count"] != test_role["passages"]
            or bundle.record["ranking_row_count"]
            != EXPECTED_RESULT_RECORDS_PER_FOLD * test_role["queries"]
        ):
            raise ValueError(f"Evaluation fold {outer_fold} role counts changed")

    return {
        "schema_version": 1,
        "manifest_type": "arr_retrieval_five_fold_evaluation_index",
        "experiment_id": "arr_retrieval_cv_v1",
        "folds": records,
        "statistics_computed": False,
    }
