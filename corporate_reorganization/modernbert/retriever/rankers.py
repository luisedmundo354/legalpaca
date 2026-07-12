from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


RANKER_SCORE_PROTOCOL = "complete_cpu_float32_scores_v1"


def _exact_ids(values: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list or tuple")
    normalized: list[str] = []
    for position, value in enumerate(values):
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError(f"{name}[{position}] must be a non-empty exact string")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} contains duplicate identities")
    if normalized != sorted(normalized):
        raise ValueError(f"{name} must be lexicographically sorted")
    return tuple(normalized)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive exact integer")
    return value


def _explicit_device(device: object, torch_module) -> Any:
    if type(device) is not str or not device or device.strip() != device:
        raise ValueError("device must be an explicit non-empty exact string")
    if device == "auto":
        raise ValueError("device='auto' is forbidden for controlled evaluation")
    resolved = torch_module.device(device)
    if resolved.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported controlled evaluation device: {device}")
    return resolved


def validate_complete_score_matrix(
    scores,
    *,
    query_ids: Sequence[str],
    passage_ids: Sequence[str],
    torch_module,
):
    """Return an owned CPU-float32 matrix after exact completeness checks."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    if not torch_module.is_tensor(scores):
        raise TypeError("scores must be a torch tensor")
    if scores.ndim != 2 or not scores.is_floating_point():
        raise TypeError("scores must be a rank-2 floating tensor")
    expected_shape = (len(normalized_query_ids), len(normalized_passage_ids))
    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            f"Complete score matrix has shape={tuple(scores.shape)}; expected={expected_shape}"
        )
    result = scores.detach().to(device="cpu", dtype=torch_module.float32).contiguous()
    if tuple(result.shape) != expected_shape or result.dtype != torch_module.float32:
        raise RuntimeError("Score conversion changed the complete matrix contract")
    if not bool(torch_module.isfinite(result).all().item()):
        raise FloatingPointError("Complete score matrix contains a non-finite value")
    return result.clone()


def score_embedding_matrices(
    *,
    query_embeddings,
    passage_embeddings,
    query_ids: Sequence[str],
    passage_ids: Sequence[str],
    torch_module,
):
    """Score complete embeddings in CPU float32, matching controlled validation."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    for name, tensor, rows in (
        ("query_embeddings", query_embeddings, len(normalized_query_ids)),
        ("passage_embeddings", passage_embeddings, len(normalized_passage_ids)),
    ):
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if tensor.ndim != 2 or not tensor.is_floating_point() or tensor.shape[0] != rows:
            raise ValueError(f"{name} has an invalid shape or dtype")
        if tensor.shape[1] < 1 or not bool(torch_module.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{name} is empty or non-finite")
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Query and passage embedding dimensions disagree")
    query_cpu = query_embeddings.detach().to(device="cpu", dtype=torch_module.float32)
    passage_cpu = passage_embeddings.detach().to(device="cpu", dtype=torch_module.float32)
    return validate_complete_score_matrix(
        query_cpu @ passage_cpu.T,
        query_ids=normalized_query_ids,
        passage_ids=normalized_passage_ids,
        torch_module=torch_module,
    )


def _tokenize_exact(
    tokenizer,
    texts: tuple[str, ...],
    *,
    truncation_side: str,
    max_length: int,
    torch_module,
) -> Mapping[str, Any]:
    if not texts or any(type(text) is not str or not text.strip() for text in texts):
        raise ValueError("Ranker tokenizer inputs must be non-empty exact strings")
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be exactly 'left' or 'right'")
    _positive_int(max_length, name="max_length")
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        tokens = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = original_side
    if not isinstance(tokens, Mapping):
        raise TypeError("Tokenizer did not return a mapping")
    input_ids = tokens.get("input_ids")
    attention_mask = tokens.get("attention_mask")
    if (
        not torch_module.is_tensor(input_ids)
        or input_ids.dtype != torch_module.long
        or input_ids.ndim != 2
        or input_ids.shape[0] != len(texts)
        or not torch_module.is_tensor(attention_mask)
        or attention_mask.dtype != torch_module.long
        or attention_mask.shape != input_ids.shape
    ):
        raise TypeError("Tokenizer returned malformed input_ids/attention_mask")
    if bool((attention_mask.sum(dim=1) < 1).any().item()):
        raise RuntimeError("Tokenizer produced an empty encoded row")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _validate_embedding_batch(
    embeddings,
    *,
    rows: int,
    name: str,
    torch_module,
):
    if (
        not torch_module.is_tensor(embeddings)
        or embeddings.ndim != 2
        or not embeddings.is_floating_point()
        or embeddings.shape[0] != rows
        or embeddings.shape[1] < 1
    ):
        raise TypeError(f"{name} returned malformed embeddings")
    if not bool(torch_module.isfinite(embeddings).all().item()):
        raise FloatingPointError(f"{name} returned non-finite embeddings")
    return embeddings.detach().to(device="cpu", dtype=torch_module.float32)


def score_loaded_dual_encoder(
    *,
    model,
    tokenizer,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    passage_ids: Sequence[str],
    passage_texts: Sequence[str],
    slot_token_id: int,
    query_batch_size: int,
    passage_batch_size: int,
    max_len_query: int,
    max_len_passage: int,
    device: str,
    torch_module,
):
    """Encode one strict local DualEncoder artifact and return complete scores."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    query_text_tuple = tuple(query_texts)
    passage_text_tuple = tuple(passage_texts)
    if len(query_text_tuple) != len(normalized_query_ids):
        raise ValueError("query_texts must align exactly with query_ids")
    if len(passage_text_tuple) != len(normalized_passage_ids):
        raise ValueError("passage_texts must align exactly with passage_ids")
    if type(slot_token_id) is not int or slot_token_id < 0:
        raise ValueError("slot_token_id must be a nonnegative exact integer")
    query_batch_size = _positive_int(query_batch_size, name="query_batch_size")
    passage_batch_size = _positive_int(passage_batch_size, name="passage_batch_size")
    max_len_query = _positive_int(max_len_query, name="max_len_query")
    max_len_passage = _positive_int(max_len_passage, name="max_len_passage")
    resolved_device = _explicit_device(device, torch_module)
    try:
        parameter_device = next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise TypeError("Dual encoder must expose at least one parameter") from error
    if parameter_device != resolved_device:
        raise RuntimeError(
            f"Loaded dual encoder is on {parameter_device}; expected explicit {resolved_device}"
        )
    if not hasattr(model, "encode_queries") or not hasattr(model, "encode_passages"):
        raise TypeError("Loaded model is not a DualEncoderRetriever")

    query_embeddings: list[Any] = []
    passage_embeddings: list[Any] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch_module.no_grad():
            for start in range(0, len(query_text_tuple), query_batch_size):
                texts = query_text_tuple[start : start + query_batch_size]
                tokens = _tokenize_exact(
                    tokenizer,
                    texts,
                    truncation_side="left",
                    max_length=max_len_query,
                    torch_module=torch_module,
                )
                slot_counts = tokens["input_ids"].eq(slot_token_id).sum(dim=1)
                if not bool(slot_counts.eq(1).all().item()):
                    raise RuntimeError(
                        "Every controlled query must retain exactly one slot token after truncation"
                    )
                embeddings = model.encode_queries(
                    tokens["input_ids"].to(resolved_device),
                    tokens["attention_mask"].to(resolved_device),
                )
                query_embeddings.append(
                    _validate_embedding_batch(
                        embeddings,
                        rows=len(texts),
                        name="Dual-encoder query path",
                        torch_module=torch_module,
                    )
                )
            for start in range(0, len(passage_text_tuple), passage_batch_size):
                texts = passage_text_tuple[start : start + passage_batch_size]
                tokens = _tokenize_exact(
                    tokenizer,
                    texts,
                    truncation_side="right",
                    max_length=max_len_passage,
                    torch_module=torch_module,
                )
                pooling_counts = tokens["attention_mask"].clone()
                if pooling_counts.shape[1] > 0:
                    pooling_counts[:, 0] = 0
                if bool(pooling_counts.sum(dim=1).lt(1).any().item()):
                    raise RuntimeError("A passage has no token left after first-token exclusion")
                embeddings = model.encode_passages(
                    tokens["input_ids"].to(resolved_device),
                    tokens["attention_mask"].to(resolved_device),
                )
                passage_embeddings.append(
                    _validate_embedding_batch(
                        embeddings,
                        rows=len(texts),
                        name="Dual-encoder passage path",
                        torch_module=torch_module,
                    )
                )
    finally:
        model.train(was_training)
    return score_embedding_matrices(
        query_embeddings=torch_module.cat(query_embeddings, dim=0),
        passage_embeddings=torch_module.cat(passage_embeddings, dim=0),
        query_ids=normalized_query_ids,
        passage_ids=normalized_passage_ids,
        torch_module=torch_module,
    )


def score_loaded_mean_pool_encoder(
    *,
    model,
    tokenizer,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    passage_ids: Sequence[str],
    passage_texts: Sequence[str],
    query_prefix: str,
    passage_prefix: str,
    query_truncation_side: str,
    query_batch_size: int,
    passage_batch_size: int,
    max_len_query: int,
    max_len_passage: int,
    device: str,
    torch_module,
):
    """Score an explicitly loaded dense baseline with normalized mean pooling."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    if query_truncation_side not in {"left", "right"}:
        raise ValueError("query_truncation_side must be explicitly 'left' or 'right'")
    if type(query_prefix) is not str or type(passage_prefix) is not str:
        raise TypeError("Dense prefixes must be exact strings")
    raw_query_texts = tuple(query_texts)
    raw_passage_texts = tuple(passage_texts)
    if len(raw_query_texts) != len(normalized_query_ids) or any(
        type(text) is not str or not text.strip() for text in raw_query_texts
    ):
        raise ValueError("query_texts must align exactly with query_ids")
    if len(raw_passage_texts) != len(normalized_passage_ids) or any(
        type(text) is not str or not text.strip() for text in raw_passage_texts
    ):
        raise ValueError("passage_texts must align exactly with passage_ids")
    prefixed_queries = tuple(f"{query_prefix}{text}" for text in raw_query_texts)
    prefixed_passages = tuple(f"{passage_prefix}{text}" for text in raw_passage_texts)
    query_batch_size = _positive_int(query_batch_size, name="query_batch_size")
    passage_batch_size = _positive_int(passage_batch_size, name="passage_batch_size")
    max_len_query = _positive_int(max_len_query, name="max_len_query")
    max_len_passage = _positive_int(max_len_passage, name="max_len_passage")
    resolved_device = _explicit_device(device, torch_module)
    try:
        parameter_device = next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise TypeError("Dense encoder must expose at least one parameter") from error
    if parameter_device != resolved_device:
        raise RuntimeError(
            f"Loaded dense encoder is on {parameter_device}; expected {resolved_device}"
        )

    def encode(
        texts: tuple[str, ...],
        *,
        batch_size: int,
        truncation_side: str,
        max_length: int,
        name: str,
    ):
        batches: list[Any] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = _tokenize_exact(
                tokenizer,
                batch_texts,
                truncation_side=truncation_side,
                max_length=max_length,
                torch_module=torch_module,
            )
            input_ids = tokens["input_ids"].to(resolved_device)
            attention_mask = tokens["attention_mask"].to(resolved_device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if (
                not torch_module.is_tensor(last_hidden)
                or last_hidden.ndim != 3
                or last_hidden.shape[:2] != input_ids.shape
                or not last_hidden.is_floating_point()
            ):
                raise TypeError(f"{name} encoder returned malformed hidden states")
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
            denominators = mask.sum(dim=1)
            if bool(denominators.le(0).any().item()):
                raise RuntimeError(f"{name} encoder received an empty pooling mask")
            pooled = (last_hidden * mask).sum(dim=1) / denominators
            embeddings = torch_module.nn.functional.normalize(pooled, p=2, dim=-1)
            batches.append(
                _validate_embedding_batch(
                    embeddings,
                    rows=len(batch_texts),
                    name=name,
                    torch_module=torch_module,
                )
            )
        return torch_module.cat(batches, dim=0)

    was_training = bool(model.training)
    model.eval()
    try:
        with torch_module.no_grad():
            query_embeddings = encode(
                prefixed_queries,
                batch_size=query_batch_size,
                truncation_side=query_truncation_side,
                max_length=max_len_query,
                name="Dense query path",
            )
            passage_embeddings = encode(
                prefixed_passages,
                batch_size=passage_batch_size,
                truncation_side="right",
                max_length=max_len_passage,
                name="Dense passage path",
            )
    finally:
        model.train(was_training)
    return score_embedding_matrices(
        query_embeddings=query_embeddings,
        passage_embeddings=passage_embeddings,
        query_ids=normalized_query_ids,
        passage_ids=normalized_passage_ids,
        torch_module=torch_module,
    )


def _pad_pretokenized_queries(
    tokenizer,
    sequences: tuple[tuple[int, ...], ...],
    *,
    torch_module,
) -> Mapping[str, Any]:
    if getattr(tokenizer, "padding_side", None) != "right":
        raise RuntimeError("Pinned E5 tokenizer must use right padding")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise RuntimeError("Pinned E5 tokenizer has no exact pad token ID")
    if not sequences:
        raise ValueError("Pretokenized E5 query batch must not be empty")
    max_length = max(len(sequence) for sequence in sequences)
    if max_length > 512 or any(
        not sequence
        or any(type(token_id) is not int or token_id < 0 for token_id in sequence)
        for sequence in sequences
    ):
        raise ValueError("Pretokenized E5 query batch contains an invalid sequence")
    input_rows = [
        [*sequence, *([pad_token_id] * (max_length - len(sequence)))]
        for sequence in sequences
    ]
    mask_rows = [
        [*([1] * len(sequence)), *([0] * (max_length - len(sequence)))]
        for sequence in sequences
    ]
    input_ids = torch_module.tensor(input_rows, dtype=torch_module.long)
    attention_mask = torch_module.tensor(mask_rows, dtype=torch_module.long)
    token_type_ids = torch_module.zeros_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def score_loaded_e5_encoder(
    *,
    model,
    tokenizer,
    query_ids: Sequence[str],
    packed_query_input_ids: Sequence[Sequence[int]],
    passage_ids: Sequence[str],
    passage_texts: Sequence[str],
    passage_prefix: str,
    query_batch_size: int,
    passage_batch_size: int,
    max_len_passage: int,
    device: str,
    torch_module,
):
    """Score exact prepacked E5 queries without a second prefix or truncation."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    packed_queries = tuple(tuple(sequence) for sequence in packed_query_input_ids)
    if len(packed_queries) != len(normalized_query_ids):
        raise ValueError("packed_query_input_ids must align exactly with query_ids")
    if any(
        len(sequence) < 4
        or len(sequence) > 512
        or tuple(sequence[1:3]) != (23_032, 1_024)
        or sequence[0] != getattr(tokenizer, "cls_token_id", None)
        or sequence[-1] != getattr(tokenizer, "sep_token_id", None)
        for sequence in packed_queries
    ):
        raise ValueError("Pretokenized E5 query violates prefix, special-token, or length contract")
    raw_passage_texts = tuple(passage_texts)
    if len(raw_passage_texts) != len(normalized_passage_ids) or any(
        type(text) is not str or not text.strip() for text in raw_passage_texts
    ):
        raise ValueError("passage_texts must align exactly with passage_ids")
    if passage_prefix != "passage: " or type(passage_prefix) is not str:
        raise ValueError("Pinned E5 passage prefix must be exact 'passage: '")
    original_truncation_side = getattr(tokenizer, "truncation_side", None)
    if original_truncation_side not in {"left", "right"}:
        raise RuntimeError("Pinned E5 tokenizer has no exact truncation side")
    prefixed_passages = tuple(f"{passage_prefix}{text}" for text in raw_passage_texts)
    query_batch_size = _positive_int(query_batch_size, name="query_batch_size")
    passage_batch_size = _positive_int(passage_batch_size, name="passage_batch_size")
    if type(max_len_passage) is not int or max_len_passage != 500:
        raise ValueError("Pinned E5 max_len_passage must be exact integer 500")
    resolved_device = _explicit_device(device, torch_module)
    try:
        parameter_device = next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise TypeError("E5 encoder must expose at least one parameter") from error
    if parameter_device != resolved_device:
        raise RuntimeError(
            f"Loaded E5 encoder is on {parameter_device}; expected {resolved_device}"
        )

    def encode_tokens(tokens: Mapping[str, Any], *, rows: int, name: str):
        required = {"input_ids", "attention_mask", "token_type_ids"}
        if set(tokens) != required:
            raise ValueError(f"{name} token mapping changed: keys={sorted(tokens)}")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_type_ids = tokens["token_type_ids"]
        if (
            not torch_module.is_tensor(input_ids)
            or input_ids.dtype != torch_module.long
            or input_ids.ndim != 2
            or input_ids.shape[0] != rows
            or not torch_module.is_tensor(attention_mask)
            or attention_mask.dtype != torch_module.long
            or attention_mask.shape != input_ids.shape
            or not torch_module.is_tensor(token_type_ids)
            or token_type_ids.dtype != torch_module.long
            or token_type_ids.shape != input_ids.shape
            or bool(token_type_ids.ne(0).any().item())
        ):
            raise TypeError(f"{name} tokenizer tensors are malformed")
        device_tokens = {
            key: value.to(resolved_device)
            for key, value in tokens.items()
        }
        outputs = model(**device_tokens, return_dict=True)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if (
            not torch_module.is_tensor(last_hidden)
            or last_hidden.ndim != 3
            or last_hidden.shape[:2] != device_tokens["input_ids"].shape
            or not last_hidden.is_floating_point()
        ):
            raise TypeError(f"{name} encoder returned malformed hidden states")
        mask = device_tokens["attention_mask"].unsqueeze(-1).type_as(last_hidden)
        denominators = mask.sum(dim=1)
        if bool(denominators.le(0).any().item()):
            raise RuntimeError(f"{name} encoder received an empty pooling mask")
        pooled = (last_hidden * mask).sum(dim=1) / denominators
        embeddings = torch_module.nn.functional.normalize(pooled, p=2, dim=-1)
        return _validate_embedding_batch(
            embeddings,
            rows=rows,
            name=name,
            torch_module=torch_module,
        )

    query_embeddings: list[Any] = []
    passage_embeddings: list[Any] = []
    was_training = bool(model.training)
    model.eval()
    tokenizer.truncation_side = "right"
    try:
        with torch_module.no_grad():
            for start in range(0, len(packed_queries), query_batch_size):
                batch = packed_queries[start : start + query_batch_size]
                query_embeddings.append(
                    encode_tokens(
                        _pad_pretokenized_queries(
                            tokenizer,
                            batch,
                            torch_module=torch_module,
                        ),
                        rows=len(batch),
                        name="E5 query path",
                    )
                )
            for start in range(0, len(prefixed_passages), passage_batch_size):
                texts = prefixed_passages[start : start + passage_batch_size]
                tokens = tokenizer(
                    list(texts),
                    truncation=True,
                    max_length=max_len_passage,
                    padding=True,
                    return_tensors="pt",
                    return_token_type_ids=True,
                )
                if not isinstance(tokens, Mapping):
                    raise TypeError("E5 passage tokenizer did not return a mapping")
                passage_embeddings.append(
                    encode_tokens(
                        {
                            "input_ids": tokens.get("input_ids"),
                            "attention_mask": tokens.get("attention_mask"),
                            "token_type_ids": tokens.get("token_type_ids"),
                        },
                        rows=len(texts),
                        name="E5 passage path",
                    )
                )
    finally:
        tokenizer.truncation_side = original_truncation_side
        model.train(was_training)
    return score_embedding_matrices(
        query_embeddings=torch_module.cat(query_embeddings, dim=0),
        passage_embeddings=torch_module.cat(passage_embeddings, dim=0),
        query_ids=normalized_query_ids,
        passage_ids=normalized_passage_ids,
        torch_module=torch_module,
    )


def complete_bm25_scores_from_hits(
    *,
    query_ids: Sequence[str],
    passage_ids: Sequence[str],
    hits_by_query: Mapping[str, Sequence[Mapping[str, object]]],
    torch_module,
):
    """Fill the BM25 zero tail before the canonical stable ranking step."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    if not isinstance(hits_by_query, Mapping) or set(hits_by_query) != set(
        normalized_query_ids
    ):
        raise ValueError("BM25 hit rows must cover exactly the query inventory")
    passage_set = set(normalized_passage_ids)
    score_rows: list[list[float]] = []
    for query_id in normalized_query_ids:
        raw_hits = hits_by_query[query_id]
        if not isinstance(raw_hits, (list, tuple)):
            raise TypeError(f"BM25 hits for query {query_id!r} must be a list or tuple")
        score_by_passage_id: dict[str, float] = {}
        for position, hit in enumerate(raw_hits):
            if type(hit) is not dict or set(hit) != {"passage_id", "score"}:
                raise ValueError(
                    f"BM25 hit {position} for query {query_id!r} has an invalid schema"
                )
            passage_id = hit["passage_id"]
            score = hit["score"]
            if type(passage_id) is not str or passage_id not in passage_set:
                raise ValueError(
                    f"BM25 query {query_id!r} returned a foreign passage {passage_id!r}"
                )
            if passage_id in score_by_passage_id:
                raise ValueError(
                    f"BM25 query {query_id!r} returned duplicate passage {passage_id!r}"
                )
            if type(score) not in {int, float} or not math.isfinite(float(score)):
                raise FloatingPointError(
                    f"BM25 query {query_id!r} returned a non-finite score"
                )
            score_by_passage_id[passage_id] = float(score)
        score_rows.append(
            [score_by_passage_id.get(passage_id, 0.0) for passage_id in normalized_passage_ids]
        )
    return validate_complete_score_matrix(
        torch_module.tensor(score_rows, dtype=torch_module.float32),
        query_ids=normalized_query_ids,
        passage_ids=normalized_passage_ids,
        torch_module=torch_module,
    )


def search_bm25_complete_scores(
    *,
    searcher,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    passage_ids: Sequence[str],
    torch_module,
):
    """Adapt an explicitly constructed Lucene searcher to the complete score protocol."""

    normalized_query_ids = _exact_ids(query_ids, name="query_ids")
    normalized_passage_ids = _exact_ids(passage_ids, name="passage_ids")
    texts = tuple(query_texts)
    if len(texts) != len(normalized_query_ids) or any(
        type(text) is not str or not text.strip() for text in texts
    ):
        raise ValueError("BM25 query_texts must align with query_ids and be non-empty")
    hits_by_query: dict[str, list[dict[str, object]]] = {}
    for query_id, query_text in zip(normalized_query_ids, texts):
        raw_hits = searcher.search(query_text, k=len(normalized_passage_ids))
        if not isinstance(raw_hits, (list, tuple)):
            raise TypeError("Lucene search must return a list or tuple")
        hits_by_query[query_id] = [
            {"passage_id": str(hit.docid), "score": float(hit.score)}
            for hit in raw_hits
        ]
    return complete_bm25_scores_from_hits(
        query_ids=normalized_query_ids,
        passage_ids=normalized_passage_ids,
        hits_by_query=hits_by_query,
        torch_module=torch_module,
    )
