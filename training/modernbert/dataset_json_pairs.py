import os, glob, json, torch
from torch.utils.data import Dataset

class JsonPairDataset(Dataset):
    def __init__(self, directory, tokenizer, max_len=512):
        self.records = []
        for fp in glob.glob(os.path.join(directory, "*.jsonl")):
            with open(fp, "r", encoding="utf-8") as f:
                self.records.extend(json.loads(l) for l in f)
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        pre = self.tok(rec["prefix"],  truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        pos = self.tok(rec["positive"], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {
            "prefix_input_ids": pre["input_ids"].squeeze(0),
            "prefix_attention_mask": pre["attention_mask"].squeeze(0),
            "pos_input_ids": pos["input_ids"].squeeze(0),
            "pos_attention_mask": pos["attention_mask"].squeeze(0),
        }
