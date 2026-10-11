"""Feasibility/timing benchmark for facebook/nllb-moe-54b on a single Snellius H100.

This is NOT the full FLORES experiment. It only:
  1. loads the 54B checkpoint (bf16 by default, device_map="auto") and records
     how/where it was placed (fully on GPU vs CPU/disk offload),
  2. runs a small sanity check + timing run on ONE already-configured translation
     direction, reusing the same FLORES loading/alignment code as the rest of
     the pipeline (dataset_pipeline.py) so results stay comparable,
  3. extrapolates the measured per-sentence throughput to the full experiment
     (all directions x all examples in config.yaml) so the team can decide
     whether the full 54B run is worth the compute.

Run with --load_only first (see benchmark_54b_loadcheck.job) to check the model
even fits before spending GPU time on the timing run (benchmark_54b_timing.job).

Usage:
    python benchmark_nllb_moe_54b.py --load_only
    python benchmark_nllb_moe_54b.py --n_sanity 10 --n_timing 100 --batch_size 4
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dataset_pipeline import build_translation_pair, load_config, load_flores_language, validate_pair

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="used for FLORES split/data_dir/dataset "
                         "name and to count directions/examples for the full-experiment extrapolation")
    parser.add_argument("--checkpoint", default="facebook/nllb-moe-54b")
    parser.add_argument("--precision", choices=list(DTYPES), default="bf16")
    parser.add_argument("--direction", help="SRC-TGT FLORES codes, e.g. eng_Latn-swh_Latn "
                         "(default: first direction in --config)")
    parser.add_argument("--split", help="default: dataset.split in --config")
    parser.add_argument("--batch_size", type=int, default=4, help="54B is much bigger than the "
                         "600M/1.3B models, so this defaults lower than config.yaml's dataloader.batch_size")
    parser.add_argument("--max_length", type=int, default=256, help="matches run_evaluation.py's default")
    parser.add_argument("--n_sanity", type=int, default=10, help="sentences translated first, not timed, "
                         "for manual inspection (README §32 step 1)")
    parser.add_argument("--n_timing", type=int, default=100, help="sentences used for the actual timing run")
    parser.add_argument("--load_only", action="store_true", help="load the model, record device map / "
                         "peak VRAM, then stop (no inference) - see benchmark_54b_loadcheck.job")
    parser.add_argument("--out_json", default="results/benchmark_nllb_moe_54b_h100.json")
    return parser.parse_args()


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "count": torch.cuda.device_count(),
        "total_memory_gb": round(props.total_memory / 1e9, 2),
    }


def get_cpu_ram_gb():
    """Total host RAM in GB, read from /proc/meminfo (Linux/Snellius only)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024**2, 2)
    except OSError:
        pass
    return None


def peak_gpu_gb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def summarize_device_map(model):
    """hf_device_map is only set when device_map="auto" actually had to split the model."""
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return {"device_map_module_counts": None, "cpu_offload": False, "disk_offload": False}
    counts = Counter(str(device) for device in device_map.values())
    return {
        "device_map_module_counts": dict(counts),
        "cpu_offload": any(str(d) == "cpu" for d in device_map.values()),
        "disk_offload": any(str(d) == "disk" for d in device_map.values()),
    }


def load_benchmark_model(checkpoint, dtype):
    """AutoModelForSeq2SeqLM covers NLLB-MoE's architecture too; fall back to the
    explicit class if a given transformers version doesn't route it automatically."""
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=dtype, device_map="auto")
    except (ValueError, KeyError):
        from transformers import NllbMoeForConditionalGeneration
        return NllbMoeForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=dtype, device_map="auto")


def iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def translate_pairs(model, tokenizer, pairs, src_lang, tgt_lang, batch_size, max_length):
    """Same generation settings as model_pipeline.translate_batch (forced_bos_token_id per
    direction), but batches straight from the pair dicts and also counts generated tokens."""
    tokenizer.src_lang = src_lang
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    input_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    hypotheses = []
    total_tokens = 0
    for batch in iter_batches(pairs, batch_size):
        sources = [p["source"] for p in batch]
        inputs = tokenizer(sources, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to(input_device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, forced_bos_token_id=tgt_token_id, max_length=max_length,
            )
        total_tokens += int((generated != tokenizer.pad_token_id).sum().item())
        hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return hypotheses, total_tokens


def save_json(path, result):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def print_summary(result):
    load = result.get("load", {})
    bench = result.get("benchmark", {})
    extrap = result.get("extrapolation", {})
    print("\n" + "=" * 60)
    print(f"Model:      {result['model']}")
    print(f"GPU:        {result['gpu']}")
    print(f"Precision:  {result['precision']}")
    print(f"\nModel load:\n  {load.get('seconds')}s, peak VRAM {load.get('peak_gpu_memory_gb')} GB")
    print(f"  cpu_offload={load.get('cpu_offload')} disk_offload={load.get('disk_offload')}")
    if bench:
        print(f"\nBenchmark ({result['benchmark_config']['direction']}, "
              f"{bench['n_sentences_timed']} sentences, batch_size="
              f"{result['benchmark_config']['batch_size']}):")
        print(f"  inference time: {bench['inference_seconds']}s")
        print(f"  throughput:     {bench['sentences_per_second']} sentences/s, "
              f"{bench['tokens_per_second']} tokens/s")
        print(f"  peak VRAM:      {bench['peak_gpu_memory_gb']} GB")
    if extrap:
        print(f"\nExtrapolation ({extrap['total_directions_in_config']} directions x "
              f"{extrap['examples_per_direction']} examples):")
        print(f"  one direction:   ~{extrap['estimated_hours_one_direction']} h")
        print(f"  full experiment: ~{extrap['estimated_hours_full_experiment']} h "
              f"(~{extrap['estimated_gpu_hours_full_experiment_single_h100']} single-H100 GPU-hours)")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)

    direction = args.direction or "-".join(config["directions"][0])
    src_lang, tgt_lang = direction.split("-")
    split = args.split or config["dataset"]["split"]
    data_dir = config["dataset"]["data_dir"]
    dataset_name = config["dataset"]["name"]

    result = {
        "model": args.checkpoint,
        "precision": args.precision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": get_gpu_info(),
        "cpu_ram_gb": get_cpu_ram_gb(),
        "benchmark_config": {
            "direction": direction, "split": split, "batch_size": args.batch_size,
            "max_length": args.max_length, "n_sanity": args.n_sanity, "n_timing": args.n_timing,
        },
    }

    # Data first: a missing offline cache should fail in seconds, not after the 16 min model load.
    print(f"[benchmark] loading FLORES data for {direction} ({split}) ...")
    source_data = load_flores_language(src_lang, split, data_dir, dataset_name)
    target_data = load_flores_language(tgt_lang, split, data_dir, dataset_name)
    pairs = build_translation_pair(source_data, target_data, src_lang, tgt_lang)
    validate_pair(pairs, src_lang, tgt_lang)
    result["examples_available_in_split"] = len(pairs)

    print(f"[benchmark] loading {args.checkpoint} ({args.precision}) with device_map='auto' ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    try:
        model = load_benchmark_model(args.checkpoint, DTYPES[args.precision])
    except Exception as e:  # capture the exact error instead of retrying expensive jobs
        result["error"] = f"model load failed: {type(e).__name__}: {e}"
        result["load"] = {"seconds": round(time.time() - load_start, 1),
                           "peak_gpu_memory_gb": peak_gpu_gb()}
        save_json(args.out_json, result)
        print(f"[benchmark] model load FAILED: {result['error']}")
        return
    model.eval()
    load_seconds = time.time() - load_start

    result["load"] = {"seconds": round(load_seconds, 1), "peak_gpu_memory_gb": peak_gpu_gb(),
                       **summarize_device_map(model)}
    print(f"[benchmark] loaded in {load_seconds:.1f}s, peak VRAM {result['load']['peak_gpu_memory_gb']} GB")
    print(f"[benchmark] device map: {result['load']}")
    save_json(args.out_json, result)  # save progress in case inference crashes/times out

    if args.load_only:
        print(f"[benchmark] --load_only set, wrote {args.out_json}, stopping before inference")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    sanity_pairs = pairs[:args.n_sanity]
    timing_pairs = pairs[args.n_sanity:args.n_sanity + args.n_timing]
    if len(timing_pairs) < args.n_timing:
        print(f"[benchmark] warning: only {len(timing_pairs)} examples available for timing "
              f"(requested {args.n_timing})")

    print(f"[benchmark] sanity check on {len(sanity_pairs)} sentences (not timed) ...")
    sanity_hyps, _ = translate_pairs(model, tokenizer, sanity_pairs, src_lang, tgt_lang,
                                      args.batch_size, args.max_length)
    for p, hyp in list(zip(sanity_pairs, sanity_hyps))[:5]:
        print(f"  src={p['source'][:80]!r}\n  ref={p['reference'][:80]!r}\n  hyp={hyp[:80]!r}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"[benchmark] timing run on {len(timing_pairs)} sentences ...")
    infer_start = time.time()
    timing_hyps, total_tokens = translate_pairs(model, tokenizer, timing_pairs, src_lang, tgt_lang,
                                                 args.batch_size, args.max_length)
    inference_seconds = time.time() - infer_start

    n = len(timing_pairs)
    avg_seconds_per_sentence = inference_seconds / n if n else 0
    result["benchmark"] = {
        "n_sentences_timed": n,
        "inference_seconds": round(inference_seconds, 2),
        "generated_tokens": total_tokens,
        "sentences_per_second": round(n / inference_seconds, 4) if inference_seconds else 0,
        "tokens_per_second": round(total_tokens / inference_seconds, 2) if inference_seconds else 0,
        "avg_seconds_per_sentence": round(avg_seconds_per_sentence, 4),
        "peak_gpu_memory_gb": peak_gpu_gb(),
    }

    model_name = args.checkpoint.split("/")[-1]
    # Same schema as outputs/<model>/<quant>/<direction>.jsonl (README §8/§18).
    result["sample_translations"] = [
        {"id": p["id"], "src_lang": src_lang, "tgt_lang": tgt_lang, "model": model_name,
         "quant": args.precision, "source": p["source"], "reference": p["reference"], "hypothesis": hyp}
        for p, hyp in list(zip(timing_pairs, timing_hyps))[:5]
    ]

    total_directions = len(config["directions"])
    examples_per_direction = len(pairs)  # every direction is aligned to the same FLORES ids
    est_one_direction = avg_seconds_per_sentence * examples_per_direction
    est_full = est_one_direction * total_directions
    result["extrapolation"] = {
        "examples_per_direction": examples_per_direction,
        "total_directions_in_config": total_directions,
        "estimated_seconds_one_direction": round(est_one_direction, 1),
        "estimated_hours_one_direction": round(est_one_direction / 3600, 2),
        "estimated_seconds_full_experiment": round(est_full, 1),
        "estimated_hours_full_experiment": round(est_full / 3600, 2),
        "estimated_gpu_hours_full_experiment_single_h100": round(est_full / 3600, 2),
        "assumptions": (
            "Linear extrapolation from one direction/batch size/precision; assumes constant "
            "throughput across directions and languages and a single H100 with no queueing. "
            "Excludes COMET/chrF++/TER scoring time and per-job model (re)load time."
        ),
    }

    save_json(args.out_json, result)
    print_summary(result)


if __name__ == "__main__":
    main()
