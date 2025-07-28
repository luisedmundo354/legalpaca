import json, random, argparse, pathlib, os, math, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True,
                   help="Folder that contains one source‑document JSON file per opinion/ECJ judgement")
    p.add_argument("--outdir",     required=True,
                   help="Destination folder that will get train.jsonl / val.jsonl / test.jsonl")
    p.add_argument("--val_frac",   type=float, default=0.10)
    p.add_argument("--test_frac",  type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # --- 1️⃣  discover all JSON files (=documents) -----------------------------
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json")])
    if not files:
        sys.exit(f"No *.json files found under {args.input_dir}")

    random.shuffle(files)                                     # reproducible because of seed

    # --- 2️⃣  compute split sizes ---------------------------------------------
    n_docs   = len(files)
    n_val    = math.ceil(n_docs * args.val_frac)
    n_test   = math.ceil(n_docs * args.test_frac)
    n_train  = n_docs - n_val - n_test
    splits_by_file = {
        "val":   set(files[:n_val]),
        "test":  set(files[n_val:n_val+n_test]),
        "train": set(files[n_val+n_test:]),
    }

    print(f"Docs → train={n_train}, val={n_val}, test={n_test}")

    # --- 3️⃣  aggregate examples per split ------------------------------------
    buckets = {k: [] for k in ["train", "val", "test"]}
    corpus_lines = []                       # optional: accumulate target_set strings

    for fname in files:
        with open(os.path.join(args.input_dir, fname), encoding="utf-8") as fh:
            blob = json.load(fh)

        examples   = blob.get("examples", [])
        target_set = blob.get("target_set", [])

        split = next(k for k, bag in splits_by_file.items() if fname in bag)
        for ex in examples:
            buckets[split].append({"prefix": ex["prefix"], "positive": ex["positive"]})

        corpus_lines.extend(target_set)     # we keep *all* lines for evaluation

    # --- 4️⃣  write out jsonl files -------------------------------------------
    out_path = pathlib.Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split, rows in buckets.items():
        outfile = out_path / f"{split}.jsonl"
        with outfile.open("w", encoding="utf-8") as fh:
            for obj in rows:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"{split:>5s}: wrote {len(rows):,} lines to {outfile}")

    # also dump the evaluation corpus once
    if corpus_lines:
        corpus_file = out_path / "target_set.txt"
        corpus_file.write_text("\n".join(corpus_lines), encoding="utf-8")
        print(f"Saved full evaluation corpus → {corpus_file}")

if __name__ == "__main__":
    main()
