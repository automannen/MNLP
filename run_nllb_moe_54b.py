"""Translate selected directions with facebook/nllb-moe-54b, loading the model once.

Writes outputs/nllb-moe-54b/bf16/<src>-<tgt>.jsonl in the same schema as run_evaluation.py,
so evaluate.py can score them. Directions whose output file already exists are skipped, so a
timed-out job can simply be resubmitted.
"""

import argparse
import json
import os
import time

from transformers import AutoTokenizer

from benchmark_nllb_moe_54b import DTYPES, load_benchmark_model, translate_pairs
from dataset_pipeline import build_translation_pair, load_config, load_flores_language, validate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="facebook/nllb-moe-54b")
    parser.add_argument("--precision", choices=list(DTYPES), default="bf16")
    parser.add_argument("--directions", nargs="+", required=True, help="SRC-TGT FLORES codes, e.g. eng_Latn-bak_Cyrl")
    parser.add_argument("--n_sentences", type=int, help="only the first N sentences per direction (default: all)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    split = config["dataset"]["split"]
    model_name = args.checkpoint.split("/")[-1]

    # Data first: a missing offline cache should fail in seconds, not after the long model load.
    data = {}
    for direction in args.directions:
        src_lang, tgt_lang = direction.split("-")
        pairs = build_translation_pair(
            load_flores_language(src_lang, split, config["dataset"]["data_dir"], config["dataset"]["name"]),
            load_flores_language(tgt_lang, split, config["dataset"]["data_dir"], config["dataset"]["name"]),
            src_lang, tgt_lang,
        )
        validate_pair(pairs, src_lang, tgt_lang)
        data[direction] = pairs[:args.n_sentences] if args.n_sentences else pairs

    print(f"[moe54b] loading {args.checkpoint} ({args.precision}) ...")
    load_start = time.time()
    model = load_benchmark_model(args.checkpoint, DTYPES[args.precision])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    print(f"[moe54b] loaded in {time.time() - load_start:.0f}s")

    for direction, pairs in data.items():
        src_lang, tgt_lang = direction.split("-")
        path = os.path.join(args.outputs_dir, model_name, args.precision, f"{direction}.jsonl")
        if os.path.exists(path):
            print(f"[moe54b] {direction}: {path} exists, skipping")
            continue

        start = time.time()
        hyps, _ = translate_pairs(model, tokenizer, pairs, src_lang, tgt_lang, args.batch_size, args.max_length)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for p, hyp in zip(pairs, hyps):
                f.write(json.dumps({
                    "id": p["id"], "src_lang": src_lang, "tgt_lang": tgt_lang,
                    "model": model_name, "quant": args.precision,
                    "source": p["source"], "reference": p["reference"], "hypothesis": hyp,
                }, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)  # only complete directions count as done
        print(f"[moe54b] {direction}: {len(pairs)} sentences in {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
