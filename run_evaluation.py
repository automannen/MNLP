import argparse
import json
import os
from argparse import Namespace

from transformers import AutoTokenizer

import evaluate as ev
from dataset_pipeline import load_config
from main import build_data_loaders
from model_pipeline import load_model, translate_batch

COMET_MODEL = "Unbabel/wmt22-comet-da"


def translate_all(dataloaders, model, flores_to_nllb, tokenizer, model_name, quant, outputs_dir,
                  max_length, limit_batches=None, num_beams=None):
    for direction, dataloader in dataloaders.items():
        src_lang, tgt_lang = direction.split("-")
        
        if src_lang in flores_to_nllb:
            src_lang_model = flores_to_nllb[src_lang]
        else: 
            src_lang_model = src_lang
        
        if tgt_lang in flores_to_nllb:
            tgt_lang_model = flores_to_nllb[tgt_lang]
        else:
            tgt_lang_model = tgt_lang
            
        tokenizer.src_lang = src_lang_model

        records = []
        for i, batch in enumerate(dataloader):
            if limit_batches and i >= limit_batches:
                break
            hyps = translate_batch(model, batch, tokenizer, tgt_lang_model, max_length=max_length,
                                   num_beams=num_beams)
            for sent_id, source, reference, hyp in zip(
                batch["ids"], batch["sources"], batch["references"], hyps
            ):
                records.append({
                    "id": sent_id,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "model": model_name,
                    "quant": quant,
                    "source": source,
                    "reference": reference,
                    "hypothesis": hyp,
                })

        path = os.path.join(outputs_dir, model_name, quant, f"{direction}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[translate] {direction}: {len(records)} sentences -> {path}")
        print(f"            example: {records[0]['hypothesis'][:80]!r}")


def main():
    parser = argparse.ArgumentParser(description="Translate with the team pipeline and evaluate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoints", nargs="+",
                        help="one or more HF model ids to translate with (default: model.checkpoint in config)")
    parser.add_argument("--quant", default="fp32",
                        help="label for this run's precision (the team's load_model is full precision)")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out_csv", default="results/scores.csv")
    parser.add_argument("--max_length", type=int, default=256, help="max output length in generate()")
    parser.add_argument("--num_beams", type=int,
                        help="beam size in generate() (default: the model's own, greedy for NLLB-200); "
                             "give the run a matching --quant label, e.g. fp32_beam4")
    parser.add_argument("--limit_batches", type=int, help="only translate the first N batches per direction (test)")
    parser.add_argument("--skip_translation", action="store_true", help="only evaluate existing outputs")
    parser.add_argument("--prepare_only", action="store_true",
                        help="download data, NLLB and COMET, then stop (run on the login node)")
    parser.add_argument("--no_comet", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoints = args.checkpoints or [config["model"]["checkpoint"]]

    for checkpoint in checkpoints:
        if args.skip_translation and not args.prepare_only:
            break
        model_name = checkpoint.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        dataloaders = build_data_loaders(
            config["directions"], tokenizer,
            config["dataset"]["split"], config["dataset"]["data_dir"], config["dataset"]["name"],
            checkpoint, config["processed_dir"],
            config["dataloader"]["batch_size"], config["dataloader"]["max_length"],
        )

        if args.prepare_only:
            from transformers import AutoModelForSeq2SeqLM
            print(f"[prepare] downloading {checkpoint} ...")
            AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            continue

        print(f"[translate] model {checkpoint}")
        model = load_model(checkpoint)
        translate_all(dataloaders, model, config["model"]["language_code_mapping"], tokenizer, model_name, args.quant,
                      args.outputs_dir, args.max_length, args.limit_batches, args.num_beams)
        del model

    if args.prepare_only:
        from comet import download_model, load_from_checkpoint
        print("[prepare] downloading COMET ...")
        load_from_checkpoint(download_model(COMET_MODEL))
        print("[prepare] done - now submit the GPU job")
        return

    ev.evaluate(Namespace(
        outputs_dir=args.outputs_dir,
        baseline="fp32",
        out_csv=args.out_csv,
        bootstrap_samples=args.bootstrap_samples,
        seed=12345,
        no_comet=args.no_comet,
        comet_model=COMET_MODEL,
        comet_batch_size=32,
        gpus=1,
    ))


if __name__ == "__main__":
    main()