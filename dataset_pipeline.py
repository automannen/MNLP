from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


def create_flores_dataloader(
    tokenizer,
    source_lang,
    target_lang,
    split="devtest",
    batch_size=16,
    max_length=512,
):
    # Load each language separately
    source_data = load_dataset(
        "openlanguagedata/flores_plus",
        source_lang,
        split=split,
    )

    target_data = load_dataset(
        "openlanguagedata/flores_plus",
        target_lang,
        split=split,
    )

    # Align translations using sentence IDs
    target_lookup = {
        (row["split"], row["id"]): row["text"]
        for row in target_data
    }

    ids = []
    sources = []
    references = []

    for row in source_data:
        key = (row["split"], row["id"])

        if key in target_lookup:
            ids.append(row["id"])
            sources.append(row["text"])
            references.append(target_lookup[key])

    dataset = Dataset.from_dict({
        "id": ids,
        "source": sources,
        "reference": references,
    })

    # Tell NLLB which source language is being tokenized
    tokenizer.src_lang = source_lang

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