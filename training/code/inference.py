import os, sys
import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from safetensors.torch import load_file
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Your wrapper must NOT import deepspeed at inference time
from model_prefix_suffix import PrefixSuffixModel


@torch.no_grad()
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # mask: B x T x 1
    mask = attention_mask.unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _coerce_to_text(data) -> str:
    """
    Accepts bytes or str for:
      - raw text (common with Batch Transform + input_filter="$.inputs")
      - JSON: {"inputs": "..."} or ["...", ...] or a scalar "..."
    Returns a single string to embed.
    """
    # 1) decode only if needed (prevents "'str' has no attribute 'decode'")
    if isinstance(data, (bytes, bytearray)):
        logger.info(f"[transform_fn] data sample (bytes)={repr(data[:120])}")
        s = data.decode("utf-8", errors="replace")
    elif isinstance(data, str):
        logger.info(f"[transform_fn] data sample (str)={repr(str(data)[:120])}")
        s = data
    else:
        # very rare path: already-deserialized object
        return json.dumps(data, ensure_ascii=False)

    s = s.strip()
    # 2) try JSON; fall back to raw text
    try:
        payload = json.loads(s)
    except json.JSONDecodeError:
        return s

    # 3) normalize to a single string
    if isinstance(payload, dict):
        if "inputs" in payload:
            v = payload["inputs"]
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        # fallback: first string value
        for v in payload.values():
            if isinstance(v, str):
                return v
        return json.dumps(payload, ensure_ascii=False)

    if isinstance(payload, list):
        # Batch Transform uses strategy="SingleRecord", but if a list arrives, embed first item
        return payload[0] if payload and isinstance(payload[0], str) else " ".join(map(str, payload))

    return "" if payload is None else str(payload)


def model_fn(model_dir: str):
    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(model_dir)

    # --- build towers from LOCAL config (no downloads in the container) ---
    prefix_cfg = AutoConfig.from_pretrained(os.path.join(model_dir, "prefix_encoder"))
    suffix_cfg = AutoConfig.from_pretrained(os.path.join(model_dir, "suffix_encoder"))
    m1 = AutoModelForMaskedLM.from_config(prefix_cfg)
    m2 = AutoModelForMaskedLM.from_config(suffix_cfg)

    # --- wrap and load consolidated weights from model.safetensors ---
    temp = float(os.getenv("TEMPERATURE", "0.01"))
    wrapper = PrefixSuffixModel(args=None, prefix_enc=m1, suffix_enc=m2, temperature=temp)

    weights_path = os.path.join(model_dir, "model.safetensors")
    state = load_file(weights_path, device="cpu")  # keys: 'prefix_enc.*' and 'suffix_enc.*'
    # strict=False tolerates tied/shared weights to avoid safetensors shared-tensor complaints
    wrapper.load_state_dict(state, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper.to(device).eval()

    # knobs from env (set them in HuggingFaceModel(..., env={...}))
    return {
        "tokenizer": tok,
        "wrapper": wrapper,
        "device": device,
        "tower": os.getenv("TOWER", "suffix").lower(),     # "prefix" or "suffix"
        "pooling": os.getenv("POOLING", "mean").lower(),   # "mean" or "cls"
        "normalize": os.getenv("NORMALIZE", "true").lower() == "true",
        "max_len": int(os.getenv("MAX_LEN", "512")),
    }


@torch.no_grad()
def _embed(state, text: str) -> list[float]:
    tok = state["tokenizer"]
    wrapper = state["wrapper"]
    device = state["device"]

    toks = tok(text, return_tensors="pt", padding=False, truncation=True, max_length=state["max_len"])
    toks = {k: v.to(device) for k, v in toks.items()}

    # Forward through the chosen tower; request hidden states for pooling
    if state["tower"] == "prefix":
        out = wrapper.prefix_enc(**toks, output_hidden_states=True, return_dict=True)
    else:
        out = wrapper.suffix_enc(**toks, output_hidden_states=True, return_dict=True)

    last = out.hidden_states[-1]
    if state["pooling"] == "cls":
        pooled = last[:, 0, :]
    else:
        pooled = mean_pool(last, toks["attention_mask"])

    if state["normalize"]:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    return pooled[0].detach().cpu().numpy().tolist()


def transform_fn(model, data, content_type, accept):
    """
    IMPORTANT: must return (body, content_type), not just a string.
    """
    logger.info(f"[transform_fn] content type: {content_type} accept: {accept} data_type={type(data)}")

    text = _coerce_to_text(data)
    if not text:
        logger.info(f"[transform_fn] text when it is not text after to_text {text}")
        body = json.dumps({"error": "Empty or unparsable payload"})
        # newline-terminated for Batch Transform assemble_with='Line'
        return body + "\n", "application/json"

    logger.info(f"[transform_fn] text when it is text after to_text {text}")
    embedding = _embed(model, text)

    if len(embedding) == 1:
        logger.info(f"embedding[0] returned {type(embedding)}")
        body = json.dumps({"embedding": embedding[0]})
    else:
        logger.info(f"embedding returned {type(embedding)}")
        body = embedding

    #try:
    #    json.loads(body)
    #except Exception as e:
    #    print(f"[ERR] json.loads failed: {e} head={body[:120]!r}", file=sys.stderr, flush=True)
    #    raise

    # 6) short debug peek (shows if a stray quote snuck in)
    print(f"[dbg] body_head={body[:80]!r} len={len(body)}", file=sys.stderr, flush=True)

    logger.info(f"Embedding: {embedding[:30]}")
    logger.info(f"[transform_fn] returning first5={body[:80]!r} len={len(body)}")

    return body, accept
