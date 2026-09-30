"""FLORES-200 loading/caching, alignment into a standard format, validation, and
the NLLB DataLoader used downstream for inference/training."""

import json
import os

import yaml
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

MAX_DIRECTIONS = 100


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    directions = config.get("directions") or []
    if not directions:
        raise ValueError("config must declare at least one translation direction")
    if len(directions) > MAX_DIRECTIONS:
        raise ValueError(
            f"config declares {len(directions)} directions, at most {MAX_DIRECTIONS} are allowed"
        )

    return config


def get_required_languages(config):
    langs = set()
    for src_lang, tgt_lang in config["directions"]:
        langs.add(src_lang)
        langs.add(tgt_lang)
    return sorted(langs)


def _flores_language_path(data_dir, split, lang):
    return os.path.join(data_dir, split, lang)


def load_flores_language(lang, split, data_dir, dataset_name="openlanguagedata/flores_plus"):
    """Load a single FLORES language subset, downloading and caching it locally
    the first time it's needed. Subsequent calls reuse the local cache."""
    path = _flores_language_path(data_dir, split, lang)

    if os.path.isdir(path):
        return Dataset.load_from_disk(path)

    data = load_dataset(dataset_name, lang, split=split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.save_to_disk(path)
    return data


def build_translation_pair(source_data, target_data, src_lang, tgt_lang):
    """Align two FLORES language subsets by sentence id into the standard
    {id, src_lang, tgt_lang, source, reference} record format."""
    target_lookup = {(row["split"], row["id"]): row["text"] for row in target_data}

    pairs = []
    for row in source_data:
        key = (row["split"], row["id"])
        if key not in target_lookup:
            continue

        pairs.append(
            {
                "id": row["id"],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source": row["text"],
                "reference": target_lookup[key],
            }
        )

    return pairs


def validate_pair(pairs, src_lang, tgt_lang):
    """Data validation checklist (README section 21) run before expensive experiments."""
    if not pairs:
        raise ValueError(f"no aligned examples found for {src_lang} -> {tgt_lang}")

    ids = [example["id"] for example in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sentence ids found for {src_lang} -> {tgt_lang}")

    for example in pairs:
        if not example["source"].strip():
            raise ValueError(f"empty source sentence (id={example['id']}) for {src_lang} -> {tgt_lang}")
        if not example["reference"].strip():
            raise ValueError(f"empty reference sentence (id={example['id']}) for {src_lang} -> {tgt_lang}")

    return True


def save_pair_jsonl(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in pairs:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def create_flores_dataloader(
    pairs,
    tokenizer,
    src_lang,
    batch_size=16,
    max_length=512,
):
    """Build a DataLoader for one translation direction from already
    aligned/validated pairs (see build_translation_pair / validate_pair)."""
    dataset = Dataset.from_dict(
        {
            "id": [example["id"] for example in pairs],
            "source": [example["source"] for example in pairs],
            "reference": [example["reference"] for example in pairs],
        }
    )

    tokenizer.src_lang = src_lang

    def collate_fn(examples):
        source_texts = [example["source"] for example in examples]

        model_inputs = tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "references": [example["reference"] for example in examples],
            "sources": source_texts,
            "ids": [example["id"] for example in examples],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
