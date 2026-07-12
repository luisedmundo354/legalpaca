from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_model, save_model
from transformers import AutoConfig, AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
MODERNBERT_DIR = REPO_ROOT / "corporate_reorganization/modernbert"
if str(MODERNBERT_DIR) not in sys.path:
    sys.path.insert(0, str(MODERNBERT_DIR))

import train_sm as controlled_train  # noqa: E402
from retriever.markup import SLOT_TOKEN, all_markup_tokens  # noqa: E402
from retriever.models import DualEncoderRetriever  # noqa: E402


class ExactSnapshotTokenizerRuntimeTest(unittest.TestCase):
    @staticmethod
    def snapshot_dir() -> Path:
        value = os.environ.get("ARR_TOKENIZER_DIR")
        if not value:
            raise RuntimeError(
                "ARR_TOKENIZER_DIR must name the exact frozen ModernBERT snapshot; "
                "this required suite never skips"
            )
        return Path(value)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            str(self.snapshot_dir()),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )

    def test_transformers_449_exact_extension_contract(self) -> None:
        markup_tokens = all_markup_tokens()
        tokenizer = self.load_tokenizer()
        self.assertEqual(len(tokenizer), 50_368)
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": markup_tokens}
        )
        self.assertEqual(len(markup_tokens), 19)
        self.assertEqual(added, 19)
        self.assertEqual(len(tokenizer), 50_386)
        self.assertEqual(len(tokenizer) - 50_368, 18)

        fresh = self.load_tokenizer()
        slot_token_id = controlled_train._add_and_validate_markup_tokens(
            fresh,
            markup_tokens=markup_tokens,
            slot_token=SLOT_TOKEN,
        )
        self.assertEqual(len(fresh), 50_386)
        self.assertNotEqual(slot_token_id, fresh.unk_token_id)

    def test_frozen_config_is_overridden_to_deterministic_flash_attention(self) -> None:
        config = AutoConfig.from_pretrained(
            str(self.snapshot_dir()),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.assertEqual(config.model_type, "modernbert")
        self.assertIs(config.deterministic_flash_attn, False)
        self.assertIs(config.reference_compile, None)
        configured = controlled_train._enable_deterministic_modernbert_flash_attention(config)
        self.assertIs(configured.deterministic_flash_attn, True)
        self.assertIs(configured.reference_compile, False)

    def test_exact_model_resolves_all_attention_layers_to_deterministic_fa2(self) -> None:
        config = controlled_train._enable_deterministic_modernbert_flash_attention(
            AutoConfig.from_pretrained(
                str(self.snapshot_dir()),
                local_files_only=True,
                trust_remote_code=False,
            )
        )
        # The local verification host has no GPU. Patch only the availability
        # predicate used by Transformers' FA2 loader; weights remain on CPU and
        # no CUDA kernel is executed.
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            encoder = AutoModel.from_pretrained(
                str(self.snapshot_dir()),
                config=config,
                attn_implementation="flash_attention_2",
                local_files_only=True,
                trust_remote_code=False,
            )
        controlled_train._validate_loaded_modernbert_attention(encoder)
        attention_modules = [
            module for module in encoder.modules() if hasattr(module, "deterministic_flash_attn")
        ]
        self.assertEqual(len(attention_modules), 22)

        tokenizer = self.load_tokenizer()
        controlled_train._add_and_validate_markup_tokens(
            tokenizer,
            markup_tokens=all_markup_tokens(),
            slot_token=SLOT_TOKEN,
        )
        encoder.resize_token_embeddings(len(tokenizer))
        self.assertEqual(encoder.config.vocab_size, 50_386)
        self.assertIs(encoder.config.reference_compile, False)

    def test_final_bf16_factory_and_tied_safetensors_round_trip_are_exact(self) -> None:
        tokenizer = self.load_tokenizer()
        slot_token_id = controlled_train._add_and_validate_markup_tokens(
            tokenizer,
            markup_tokens=all_markup_tokens(),
            slot_token=SLOT_TOKEN,
        )

        def build_model():
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                return controlled_train._build_controlled_retriever(
                    base_model_dir=self.snapshot_dir(),
                    tokenizer_size=len(tokenizer),
                    slot_token_id=slot_token_id,
                    temperature=0.07,
                    auto_config_class=AutoConfig,
                    auto_model_class=AutoModel,
                    retriever_class=DualEncoderRetriever,
                    torch_dtype=torch.bfloat16,
                )

        source_model = build_model()
        state = source_model.state_dict()
        self.assertEqual(len(state), 134)
        self.assertEqual(
            {tensor.dtype for tensor in state.values() if tensor.is_floating_point()},
            {torch.bfloat16},
        )
        self.assertFalse(any(hasattr(parameter, "ds_id") for parameter in source_model.parameters()))
        controlled_train._validate_gathered_bf16_state_dict(state, torch)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.safetensors"
            save_model(source_model, str(model_path))
            reloaded_model = build_model()
            missing, unexpected = load_model(
                reloaded_model,
                model_path,
                strict=True,
                device="cpu",
            )
            self.assertFalse(missing)
            self.assertFalse(unexpected)
            self.assertEqual(
                controlled_train._require_models_bitwise_equal(
                    source_model,
                    reloaded_model,
                    torch,
                ),
                134,
            )


if __name__ == "__main__":
    unittest.main()
