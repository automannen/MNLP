"""End-to-end data pipeline entry point.

For every configured translation direction this:
  1. loads (downloading + caching locally if needed) the required FLORES languages,
  2. aligns source/target sentences into the standard record format,
  3. validates the aligned data (unique ids, no empty sentences, ...),
  4. saves the standardized data to processed_dir as JSONL,
  5. builds a ready-to-use NLLB DataLoader.

Run with: python main.py [--config config.yaml]
"""

import argparse

from transformers import AutoTokenizer

from dataset_pipeline import (
    build_translation_pair,
    create_flores_dataloader,
    load_config,
    load_flores_language,
    save_pair_jsonl,
    validate_pair,
)


def run(config_path="config.yaml"):
    config = load_config(config_path)

    dataset_name = config["dataset"]["name"]
    split = config["dataset"]["split"]
    data_dir = config["dataset"]["data_dir"]
    processed_dir = config["processed_dir"]
    batch_size = config["dataloader"]["batch_size"]
    max_length = config["dataloader"]["max_length"]
    checkpoint = config["model"]["checkpoint"]
    directions = config["directions"]

    # Step 1: load (download once, then reuse local cache for) every required language.
    required_langs = sorted({lang for pair in directions for lang in pair})
    language_data = {}
    for lang in required_langs:
        print(f"[data] loading {lang} ({split}) ...")
        language_data[lang] = load_flores_language(lang, split, data_dir, dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    dataloaders = {}
    for src_lang, tgt_lang in directions:
        direction = f"{src_lang}-{tgt_lang}"
        print(f"[data] building direction {direction} ...")

        pairs = build_translation_pair(
            language_data[src_lang], language_data[tgt_lang], src_lang, tgt_lang
        )
        validate_pair(pairs, src_lang, tgt_lang)

        out_path = f"{processed_dir}/{split}/{direction}.jsonl"
        save_pair_jsonl(pairs, out_path)
        print(f"[data] saved {len(pairs)} validated examples -> {out_path}")

        dataloaders[direction] = create_flores_dataloader(
            pairs, tokenizer, src_lang, batch_size=batch_size, max_length=max_length
        )

    print(f"[done] {len(dataloaders)} direction(s) ready: {', '.join(dataloaders)}")
    return dataloaders


def main():
    parser = argparse.ArgumentParser(description="Run the FLORES data pipeline end to end")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    args = parser.parse_args()

    run(args.config)


if __name__ == "__main__":
    main()
