"""Production entry path for the corrected legacy-style retrieval diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


_DATASET_FILES = (
    "cases.jsonl",
    "corpus.jsonl",
    "dataset_manifest.json",
    "pools/candidates_by_case.json",
    "pools/candidates_global.json",
    "queries/all.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"Corrected legacy artifact requires a non-empty regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_new_text(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite corrected legacy artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size < 1:
        raise ValueError(f"Corrected legacy input must be a non-empty regular file: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite copied corrected legacy input: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if destination.read_bytes() != source.read_bytes():
        raise RuntimeError(f"Corrected legacy input copy changed bytes: {source}")


def _subset_bytes(jsonl_path: Path, case_ids: set[str]) -> tuple[bytes, int]:
    selected: list[bytes] = []
    with jsonl_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError(f"Malformed corrected legacy JSONL line {jsonl_path}:{line_number}")
            record = json.loads(line)
            if type(record) is not dict or type(record.get("doc_id")) not in (str, int):
                raise ValueError(f"Corrected legacy JSONL row lacks doc_id: {jsonl_path}:{line_number}")
            if str(record["doc_id"]) in case_ids:
                selected.append(line)
    if not selected:
        raise ValueError(f"Corrected legacy subset is empty: {jsonl_path}")
    return b"".join(selected), len(selected)


def _merge_trace_artifacts(
    trainer,
    *,
    trace_dir: Path,
    gathered: list[object],
) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    expected_queries = {query.query_id for query in trainer.train_dataset.queries}
    coverage = {epoch: set() for epoch in range(20)}
    for expected_rank, record in enumerate(gathered):
        if type(record) is not dict or record.get("rank") != expected_rank:
            raise RuntimeError("Corrected legacy trace shard metadata changed")
        path = trace_dir / record["path"]
        if _sha256(path) != record["sha256"] or path.stat().st_size != record["size"]:
            raise RuntimeError("Corrected legacy trace shard identity changed")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != record["record_count"]:
            raise RuntimeError("Corrected legacy trace shard row count changed")
        for line in lines:
            value = json.loads(line)
            if type(value) is not dict or set(value) != {
                "rank", "prepared_microbatch_index", "local_row", "trace"
            }:
                raise RuntimeError("Corrected legacy trace record schema changed")
            trace = value["trace"]
            if value["rank"] != expected_rank:
                raise RuntimeError("Corrected legacy trace row rank changed")
            epoch = trace["epoch"]
            query_id = trace["query_id"]
            if query_id in coverage[epoch]:
                raise RuntimeError("Corrected legacy trace duplicated a query within an epoch")
            coverage[epoch].add(query_id)
            all_records.append(value)
    if any(coverage[epoch] != expected_queries for epoch in range(20)):
        raise RuntimeError("Corrected legacy traces do not cover all 418 queries in every epoch")
    if len(all_records) != 8_360:
        raise RuntimeError("Corrected legacy trace history must contain exactly 8,360 records")
    all_records.sort(key=lambda value: (value["trace"]["epoch"], value["trace"]["query_id"]))
    merged_path = trace_dir / "sampling_traces.jsonl"
    _write_new_text(merged_path, "".join(_canonical_json(value) + "\n" for value in all_records))
    manifest = {
        "schema_version": 1,
        "record_count": 8_360,
        "epochs": 20,
        "queries_per_epoch": 418,
        "merge_order": ["epoch", "query_id"],
        "shards": gathered,
        "merged": _file_record(merged_path, trace_dir),
    }
    manifest_path = trace_dir / "manifest.json"
    _write_new_text(manifest_path, _canonical_json(manifest, indent=2) + "\n")
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def _publish_trace_artifacts(
    trainer,
    output_dir: Path,
    torch_module,
    *,
    collective_local_call,
    rank_zero_call,
) -> dict[str, Any]:
    trace_dir = output_dir / "candidate_traces"
    rank = torch_module.distributed.get_rank()
    world_size = torch_module.distributed.get_world_size()
    rank_zero_call("Corrected legacy trace directory publication", trace_dir.mkdir)

    def publish_local_shard() -> dict[str, Any]:
        shard = trainer.local_trace_shard()
        shard_path = trace_dir / f"rank-{rank:05d}.jsonl"
        _write_new_text(shard_path, shard.pop("jsonl"))
        if _sha256(shard_path) != shard["sha256"]:
            raise RuntimeError("Corrected legacy trace shard changed during publication")
        return {**shard, "path": shard_path.name, "size": shard_path.stat().st_size}

    local_record = collective_local_call(
        torch_module.distributed,
        "Corrected legacy local trace-shard publication",
        publish_local_shard,
    )
    gathered: list[object] = [None for _ in range(world_size)]
    torch_module.distributed.all_gather_object(gathered, local_record)
    return rank_zero_call(
        "Corrected legacy trace merge publication",
        lambda: _merge_trace_artifacts(
            trainer,
            trace_dir=trace_dir,
            gathered=gathered,
        ),
    )


def _publish_validation_history(trainer, output_dir: Path) -> dict[str, Any]:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir()
    records = trainer.validation_history()
    files = []
    for expected_epoch, record in enumerate(records, start=1):
        if record["epoch"] != expected_epoch or record["global_step"] != expected_epoch * 4:
            raise RuntimeError("Corrected legacy validation chronology changed")
        path = validation_dir / f"epoch-{expected_epoch:03d}.json"
        _write_new_text(path, _canonical_json(record, indent=2) + "\n")
        files.append(_file_record(path, validation_dir))
    manifest = {
        "schema_version": 1,
        "epochs": 20,
        "global_steps": list(range(4, 81, 4)),
        "model_selection": "none_final_epoch_only",
        "files": files,
        "records_sha256": hashlib.sha256(
            _canonical_json(list(records)).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = validation_dir / "manifest.json"
    _write_new_text(manifest_path, _canonical_json(manifest, indent=2) + "\n")
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def _publish_input_evidence(args, loaded_config, output_dir: Path) -> dict[str, Any]:
    input_dir = output_dir / "inputs"
    data_output = input_dir / "data"
    data_output.mkdir(parents=True)
    for relative in _DATASET_FILES:
        _copy_new(args.data_dir / relative, data_output / relative)

    membership_output = input_dir / "corrected_legacy_membership"
    membership_output.mkdir()
    config_root = args.corrected_legacy_config.parent
    role_files: dict[str, Path] = {}
    for role in ("train", "validation", "test"):
        source = config_root / loaded_config.value["membership"][role]["membership_path"]
        destination = membership_output / f"{role}_cases.txt"
        _copy_new(source, destination)
        role_files[role] = destination

    subset_output = input_dir / "subsets"
    subset_output.mkdir()
    subset_records = []
    for role in ("train", "validation", "test"):
        case_ids = set(loaded_config.memberships.for_role(role))
        expected = loaded_config.value["membership"][role]
        for kind, relative, output_name, count_key, hash_key in (
            ("queries", "queries/all.jsonl", f"{role}_queries.jsonl", "query_count", "query_subset_sha256"),
            ("corpus", "corpus.jsonl", f"{role}_corpus.jsonl", "passage_count", "passage_subset_sha256"),
        ):
            payload, count = _subset_bytes(args.data_dir / relative, case_ids)
            path = subset_output / output_name
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if count != expected[count_key] or _sha256(path) != expected[hash_key]:
                raise RuntimeError(f"Corrected legacy {role} {kind} subset identity changed")
            subset_records.append({"role": role, "kind": kind, "count": count, **_file_record(path, input_dir)})

    settings_path = input_dir / "settings.json"
    _copy_new(args.corrected_legacy_config, settings_path)
    data_records = [_file_record(data_output / relative, input_dir) for relative in _DATASET_FILES]
    membership_records = [
        {"role": role, **_file_record(role_files[role], input_dir)}
        for role in ("train", "validation", "test")
    ]
    manifest = {
        "schema_version": 1,
        "config_sha256": loaded_config.config_sha256,
        "data": data_records,
        "membership": membership_records,
        "subsets": subset_records,
        "settings": _file_record(settings_path, input_dir),
    }
    manifest_path = input_dir / "manifest.json"
    _write_new_text(manifest_path, _canonical_json(manifest, indent=2) + "\n")
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def _publish_evaluation(
    results: Mapping[str, Any],
    *,
    output_dir: Path,
    query_view: str,
    model_record: Mapping[str, Any],
    test_contract_sha256: str,
) -> dict[str, Any]:
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir()
    payloads = {regime: result.to_payload() for regime, result in results.items()}
    results_path = evaluation_dir / "results.json"
    _write_new_text(
        results_path,
        _canonical_json({"schema_version": 1, "results": payloads}, indent=2) + "\n",
    )
    ranking_rows = []
    for regime in results:
        for ranking in payloads[regime]["rankings"]:
            ranking_rows.append({"regime_name": regime, **ranking})
    if len(ranking_rows) != 160:
        raise RuntimeError("Corrected legacy final evaluation must contain 160 query rankings")
    rankings_path = evaluation_dir / "rankings.jsonl"
    _write_new_text(rankings_path, "".join(_canonical_json(row) + "\n" for row in ranking_rows))
    config = {
        "schema_version": 1,
        "evaluation_type": "corrected_legacy_diagnostic_test",
        "query_view": query_view,
        "system_count": 1,
        "regimes": list(results),
        "query_rankings": 160,
        "test_contract_sha256": test_contract_sha256,
        "final_model": dict(model_record),
    }
    config_path = evaluation_dir / "evaluation_config.json"
    _write_new_text(config_path, _canonical_json(config, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "evaluation_config": _file_record(config_path, evaluation_dir),
        "results": _file_record(results_path, evaluation_dir),
        "rankings": _file_record(rankings_path, evaluation_dir),
    }
    manifest_path = evaluation_dir / "artifact_manifest.json"
    _write_new_text(manifest_path, _canonical_json(manifest, indent=2) + "\n")
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def run_corrected_legacy_diagnostic(args, *, training_launch_provenance: Mapping[str, Any]) -> int:
    """Execute one sealed flat or structured corrected legacy diagnostic job."""

    import numpy
    import torch
    import transformers
    from safetensors.torch import load_model, save_model
    from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config
    from transformers.utils import is_flash_attn_2_available

    import train_sm as controlled_entry
    from experiments.retrieval_cv.corrected_legacy_config import load_corrected_legacy_config
    from legacy_diagnostic_trainer import CorrectedLegacyDiagnosticTrainer
    from retriever.checkpointing import rank_zero_call
    from retriever.corrected_legacy_evaluation import (
        build_corrected_legacy_test_data,
        build_corrected_legacy_validation_evidence_data,
        evaluate_corrected_legacy_test_distributed,
    )
    from retriever.data import PassageIndexTable, load_queries
    from retriever.legacy_diagnostic_collator import CorrectedLegacyDiagnosticCollator
    from retriever.legacy_diagnostic_data import load_corrected_legacy_data
    from retriever.legacy_diagnostic_sampling import CorrectedLegacyDiagnosticDataset
    from retriever.markup import SLOT_TOKEN, all_markup_tokens
    from retriever.models import DualEncoderRetriever
    from retriever.provenance import (
        EXPECTED_BASE_TRAINING_IMAGE,
        EXPECTED_DATASET_MANIFEST_SHA256,
        EXPECTED_DERIVED_TRAINING_IMAGE,
        EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256,
        EXPECTED_PASSAGE_INDEX_SHA256,
        EXPECTED_RUNTIME_VERSIONS,
        load_snapshot_manifest,
        validate_preimport_environment,
        validate_snapshot_directory,
    )
    from retriever.staged_data import validate_staged_dataset

    validate_staged_dataset(
        dataset_dir=args.data_dir,
        expected_dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
    )
    loaded_config = load_corrected_legacy_config(
        args.corrected_legacy_config,
        dataset_dir=args.data_dir,
    )
    design = loaded_config.value
    if args.query_view not in design["query_views"] or args.base_seed != design["training"]["seed"]:
        raise RuntimeError("Corrected legacy CLI and frozen design disagree")
    expected_run_id = {
        "flat_masked": "corrected-legacy-flat",
        "structured": "corrected-legacy-structured",
    }[args.query_view]
    expected_launch_keys = {
        "bootstrap_protocol",
        "source_bundle",
        "training_image_contract_sha256",
        "training_plan_sha256",
        "training_request_payload_sha256",
        "training_run_id",
        "training_staging_receipt_sha256",
    }
    if (
        type(training_launch_provenance) is not dict
        or set(training_launch_provenance) != expected_launch_keys
        or training_launch_provenance["training_run_id"] != expected_run_id
    ):
        raise RuntimeError("Corrected legacy launch provenance is incomplete or mismatched")
    local_rank, rank, world_size = controlled_entry._configure_distributed_environment(torch)
    controlled_entry._configure_determinism(
        experiment_seed=args.base_seed,
        torch_module=torch,
        numpy_module=numpy,
        transformers_module=transformers,
    )
    if world_size != 4 or not is_flash_attn_2_available():
        raise RuntimeError("Corrected legacy runtime requires four ranks and FlashAttention 2")
    controlled_entry._validate_deepspeed_config(args.deepspeed_config)
    snapshot_manifest = load_snapshot_manifest(args.snapshot_manifest)
    validate_snapshot_directory(args.base_model_dir, snapshot_manifest)
    controlled_entry._prepare_output_directory(args.output_dir)

    corrected_data = load_corrected_legacy_data(args.data_dir)
    passage_index = PassageIndexTable(corrected_data.corpus_by_passage_id)
    if passage_index.sha256 != EXPECTED_PASSAGE_INDEX_SHA256:
        raise RuntimeError("Corrected legacy passage-index identity changed")
    all_queries = load_queries(args.data_dir, "all")
    shared_queries = {query.query_id: query for query in all_queries}
    strict_queries = {
        query.query_id: query
        for split in corrected_data.queries_by_split.values()
        for query in split
    }
    if set(shared_queries) != set(strict_queries) or any(
        shared_queries[query_id].doc_id != strict_queries[query_id].doc_id
        or tuple(shared_queries[query_id].positive_passage_ids)
        != strict_queries[query_id].positive_passage_ids
        for query_id in shared_queries
    ):
        raise RuntimeError("Corrected legacy strict and shared query loaders disagree")
    validation_membership = loaded_config.memberships.validation
    validation_data = build_corrected_legacy_validation_evidence_data(
        all_queries=all_queries,
        corpus_by_passage_id=corrected_data.corpus_by_passage_id,
        passage_index_table=passage_index,
        validation_case_ids=validation_membership,
        query_view=args.query_view,
    )
    test_data = build_corrected_legacy_test_data(
        all_queries=all_queries,
        corpus_by_passage_id=corrected_data.corpus_by_passage_id,
        passage_index_table=passage_index,
        test_case_ids=loaded_config.memberships.test,
        query_view=args.query_view,
    )
    train_dataset = CorrectedLegacyDiagnosticDataset(
        corrected_data,
        experiment_seed=args.base_seed,
        query_view=args.query_view,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_dir),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    slot_token_id = controlled_entry._add_and_validate_markup_tokens(
        tokenizer,
        markup_tokens=all_markup_tokens(),
        slot_token=SLOT_TOKEN,
    )
    model = controlled_entry._build_controlled_retriever(
        base_model_dir=args.base_model_dir,
        tokenizer_size=len(tokenizer),
        slot_token_id=slot_token_id,
        temperature=design["training"]["temperature"],
        auto_config_class=AutoConfig,
        auto_model_class=AutoModel,
        retriever_class=DualEncoderRetriever,
    )
    collator = CorrectedLegacyDiagnosticCollator(
        tokenizer,
        passage_index_table=passage_index,
        max_len_query=design["training"]["max_query_tokens"],
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=20,
        per_device_train_batch_size=4,
        learning_rate=design["training"]["learning_rate"],
        warmup_ratio=design["training"]["warmup_ratio"],
        weight_decay=design["training"]["weight_decay"],
        adam_beta1=design["training"]["adam_beta1"],
        adam_beta2=design["training"]["adam_beta2"],
        adam_epsilon=design["training"]["adam_epsilon"],
        optim=design["training"]["optimizer"],
        lr_scheduler_type=design["training"]["lr_scheduler_type"],
        max_grad_norm=design["training"]["max_grad_norm"],
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        eval_on_start=False,
        eval_delay=0,
        save_strategy="no",
        logging_steps=1,
        bf16=True,
        tf32=False,
        deepspeed=str(args.deepspeed_config),
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=False,
        metric_for_best_model=None,
        save_total_limit=None,
        local_rank=local_rank,
        seed=17,
        data_seed=17,
        full_determinism=False,
        accelerator_config={
            "split_batches": False,
            "dispatch_batches": False,
            "even_batches": False,
            "use_seedable_sampler": False,
        },
    )
    trainer = CorrectedLegacyDiagnosticTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_data.queries,
        data_collator=collator,
        processing_class=tokenizer,
        experiment_seed=17,
        passage_index_table=passage_index,
        validation_data=validation_data,
        max_len_passage=500,
    )
    model = None
    validate_preimport_environment(17)
    controlled_entry._validate_determinism_state(torch)
    trainer.train()
    if trainer.state.global_step != 80 or float(trainer.state.epoch) != 20.0:
        raise RuntimeError("Corrected legacy training did not complete 20 epochs/80 updates")
    active_engine = trainer.deepspeed
    if active_engine is None or trainer.model_wrapped is not active_engine:
        raise RuntimeError(
            "Corrected legacy finalization requires model_wrapped to be the active "
            "DeepSpeed engine"
        )
    validation_history = trainer.validation_history()
    final_results = evaluate_corrected_legacy_test_distributed(
        active_engine,
        tokenizer,
        test_data=test_data,
        passage_index_table=passage_index,
    )
    trace_manifest = _publish_trace_artifacts(
        trainer,
        args.output_dir,
        torch,
        collective_local_call=controlled_entry._collective_local_call,
        rank_zero_call=rank_zero_call,
    )
    torch.distributed.barrier()
    gathered_state = trainer.accelerator.get_state_dict(active_engine)

    def validate_gathered_state() -> dict[str, int]:
        if rank == 0:
            state = controlled_entry._validate_gathered_bf16_state_dict(gathered_state, torch)
            return {"rank": rank, "tensor_count": len(state)}
        if gathered_state is not None:
            raise RuntimeError("Nonzero rank received corrected legacy gathered state")
        return {"rank": rank, "tensor_count": 0}

    controlled_entry._collective_local_call(
        torch.distributed,
        "Corrected legacy final-state validation",
        validate_gathered_state,
    )
    unset_hf_deepspeed_config()

    def publish_artifacts() -> dict[str, Any]:
        exact_state = controlled_entry._validate_gathered_bf16_state_dict(gathered_state, torch)
        cpu_tokenizer = AutoTokenizer.from_pretrained(
            str(args.base_model_dir),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        cpu_slot_id = controlled_entry._add_and_validate_markup_tokens(
            cpu_tokenizer,
            markup_tokens=all_markup_tokens(),
            slot_token=SLOT_TOKEN,
        )
        if cpu_slot_id != slot_token_id or len(cpu_tokenizer) != len(tokenizer):
            raise RuntimeError("Corrected legacy CPU tokenizer contract changed")
        cpu_model = controlled_entry._build_controlled_retriever(
            base_model_dir=args.base_model_dir,
            tokenizer_size=len(cpu_tokenizer),
            slot_token_id=cpu_slot_id,
            temperature=design["training"]["temperature"],
            auto_config_class=AutoConfig,
            auto_model_class=AutoModel,
            retriever_class=DualEncoderRetriever,
            torch_dtype=torch.bfloat16,
        )
        incompatible = cpu_model.load_state_dict(exact_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Corrected legacy strict final-state load was incomplete")
        tensor_count = controlled_entry._require_model_matches_state_dict(cpu_model, exact_state, torch)
        exact_state.clear()
        model_record = controlled_entry._publish_new_binary(
            args.output_dir / "model.safetensors",
            lambda path: save_model(
                cpu_model,
                str(path),
                metadata={
                    "format": "pt",
                    "weight_dtype": "bfloat16",
                    "source": "active_engine_epoch_20_zero3_gathered_state",
                },
            ),
        )
        roundtrip_model = controlled_entry._build_controlled_retriever(
            base_model_dir=args.base_model_dir,
            tokenizer_size=len(cpu_tokenizer),
            slot_token_id=cpu_slot_id,
            temperature=design["training"]["temperature"],
            auto_config_class=AutoConfig,
            auto_model_class=AutoModel,
            retriever_class=DualEncoderRetriever,
            torch_dtype=torch.bfloat16,
        )
        load_model(roundtrip_model, str(args.output_dir / "model.safetensors"), strict=True)
        if controlled_entry._require_model_matches_state_dict(
            roundtrip_model,
            cpu_model.state_dict(),
            torch,
        ) != tensor_count:
            raise RuntimeError("Corrected legacy safetensors round trip changed tensor count")
        tokenizer_record = controlled_entry._publish_pretrained_directory(
            args.output_dir / "tokenizer",
            lambda path: cpu_tokenizer.save_pretrained(str(path)),
        )
        encoder_record = controlled_entry._publish_pretrained_directory(
            args.output_dir / "encoder_config",
            lambda path: cpu_model.encoder.config.save_pretrained(str(path)),
        )
        wrapper = {
            "schema_version": 1,
            "architecture": "DualEncoderRetriever",
            "artifact_type": "corrected_legacy_diagnostic_retriever",
            "query_view": args.query_view,
            "slot_token": SLOT_TOKEN,
            "slot_token_id": cpu_slot_id,
            "temperature": design["training"]["temperature"],
            "tokenizer_size": len(cpu_tokenizer),
            "weight_dtype": "bfloat16",
            "model_artifact_protocol": design["training"]["final_model"],
        }
        wrapper_path = args.output_dir / "wrapper_config.json"
        _write_new_text(wrapper_path, _canonical_json(wrapper, indent=2) + "\n")
        wrapper_record = _file_record(wrapper_path, args.output_dir)

        inputs_manifest = _publish_input_evidence(args, loaded_config, args.output_dir)
        validation_manifest = _publish_validation_history(trainer, args.output_dir)
        evaluation_manifest = _publish_evaluation(
            final_results,
            output_dir=args.output_dir,
            query_view=args.query_view,
            model_record=model_record,
            test_contract_sha256=test_data.contract_sha256,
        )
        explanation_path = args.output_dir / "setting_explanation.md"
        _write_new_text(
            explanation_path,
            "# Corrected legacy-style diagnostic\n\n"
            + design["setting_explanation"]
            + "\n\nThe job completed 20 full epochs and exported the active epoch-20 model. "
            "No best-epoch selection or checkpoint reload was performed.\n",
        )
        run = {
            "schema_version": 1,
            "diagnostic_id": design["diagnostic_id"],
            "label": design["label"],
            "run_id": expected_run_id,
            "run_kind": "corrected_legacy_diagnostic",
            "query_view": args.query_view,
            "seed": 17,
            "schedule": {"epochs": 20, "updates_per_epoch": 4, "total_updates": 80},
            "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
            "training_image": EXPECTED_DERIVED_TRAINING_IMAGE,
            "training_base_image": EXPECTED_BASE_TRAINING_IMAGE,
            "training_image_runtime_inventory_sha256": (
                EXPECTED_DERIVED_TRAINING_IMAGE_RUNTIME_INVENTORY_SHA256
            ),
            "training_launch_provenance": dict(training_launch_provenance),
            "snapshot": {
                "manifest_sha256": _sha256(args.snapshot_manifest),
                "tree_sha256": snapshot_manifest["tree_sha256"],
            },
            "config_sha256": loaded_config.config_sha256,
            "passage_index_sha256": passage_index.sha256,
            "validation_records": len(validation_history),
            "candidate_traces": {
                "record_count": trace_manifest["record_count"],
                "manifest_sha256": trace_manifest["manifest_sha256"],
            },
            "final_model": {**model_record, "tensor_count": tensor_count},
            "inputs_manifest_sha256": inputs_manifest["manifest_sha256"],
            "validation_manifest_sha256": validation_manifest["manifest_sha256"],
            "evaluation_manifest_sha256": evaluation_manifest["manifest_sha256"],
            "reporting_boundary": design["reporting_boundary"],
        }
        run_path = args.output_dir / "corrected_legacy_run.json"
        _write_new_text(run_path, _canonical_json(run, indent=2) + "\n")

        expected_top_level = {
            "candidate_traces",
            "corrected_legacy_run.json",
            "encoder_config",
            "evaluation",
            "inputs",
            "model.safetensors",
            "setting_explanation.md",
            "tokenizer",
            "validation",
            "wrapper_config.json",
        }
        actual_top_level = {entry.name for entry in args.output_dir.iterdir()}
        if actual_top_level != expected_top_level or any(
            entry.is_symlink() for entry in args.output_dir.iterdir()
        ):
            raise RuntimeError("Corrected legacy top-level artifact inventory changed")
        artifact_manifest = {
            "schema_version": 1,
            "commit_marker": True,
            "artifact_type": "corrected_legacy_diagnostic_retriever",
            "run": _file_record(run_path, args.output_dir),
            "model": model_record,
            "tokenizer": tokenizer_record,
            "encoder_config": encoder_record,
            "wrapper": wrapper_record,
            "setting_explanation": _file_record(explanation_path, args.output_dir),
            "inputs_manifest": _file_record(args.output_dir / "inputs/manifest.json", args.output_dir),
            "trace_manifest": _file_record(
                args.output_dir / "candidate_traces/manifest.json", args.output_dir
            ),
            "validation_manifest": _file_record(
                args.output_dir / "validation/manifest.json", args.output_dir
            ),
            "evaluation_manifest": _file_record(
                args.output_dir / "evaluation/artifact_manifest.json", args.output_dir
            ),
        }
        artifact_path = args.output_dir / "artifact_manifest.json"
        _write_new_text(artifact_path, _canonical_json(artifact_manifest, indent=2) + "\n")
        if {entry.name for entry in args.output_dir.iterdir()} != {
            *expected_top_level,
            "artifact_manifest.json",
        }:
            artifact_path.unlink()
            raise RuntimeError("Corrected legacy commit-marker publication changed inventory")
        return {
            "artifact_manifest_sha256": _sha256(artifact_path),
            "model_sha256": model_record["sha256"],
            "run_sha256": _sha256(run_path),
        }

    published = rank_zero_call("Corrected legacy artifact publication", publish_artifacts)
    if set(published) != {"artifact_manifest_sha256", "model_sha256", "run_sha256"}:
        raise RuntimeError("Corrected legacy publication result schema changed")
    torch.distributed.barrier()
    return 0


__all__ = ["run_corrected_legacy_diagnostic"]
