"""
JSON-based contrastive dataset for dual-encoder training.

Reads training examples from JSON files with 'examples': [ {prefix, positive} ]
and tokenizes them into fixed-length tensors.
"""
import os
import glob
import json

import torch
from torch.utils.data import Dataset


class JsonContrastiveDataset(Dataset):
    def __init__(self, json_dir, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_len = args.max_seq_length
        examples = []
        for path in sorted(glob.glob(os.path.join(json_dir, '*_train.json'))):
            with open(path, 'r', encoding='utf8') as f:
                data = json.load(f)
            for ex in data.get('examples', []):
                examples.append(ex)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # tokenize prefix and positive (suffix)
        p_tok = self.tokenizer(
            ex['prefix'],
            padding='max_length', truncation=True, max_length=self.max_len,
            return_tensors='pt'
        )
        s_tok = self.tokenizer(
            ex['positive'],
            padding='max_length', truncation=True, max_length=self.max_len,
            return_tensors='pt'
        )
        # convert to 1D tensors
        return {
            'prefices':      p_tok['input_ids'].squeeze(0),
            'prefix_masks':  p_tok['attention_mask'].squeeze(0),
            'suffices':      s_tok['input_ids'].squeeze(0),
            'suffix_masks':  s_tok['attention_mask'].squeeze(0)
        }
