"""Efficiency evaluation of a distilled NLLB student against its teacher.

Two steps, because teacher and student are measured in separate GPU jobs:

  1. measure: load ONE model and record its size, speed and memory
     -> results/efficiency_<model>_<precision>.json
  2. report: combine those files with the translations already in outputs/ into
     compression ratio, speedup and quality retained (per direction and per
     resource group, with paired bootstrap significance)
     -> results/efficiency_report_{summary,quality,groups}.csv

Run measure with identical settings (GPU type, directions, batch size, beam size,
max length) for every model, otherwise the speedup is not a fair comparison;
report warns when they differ.

Usage:
    python evaluate_efficiency.py measure --checkpoint facebook/nllb-200-distilled-1.3B
    python evaluate_efficiency.py measure --checkpoint facebook/nllb-moe-54b --precision bf16
    python evaluate_efficiency.py report --teacher nllb-moe-54b --students nllb-200-distilled-1.3B
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sacrebleu.metrics import BLEU, CHRF
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoTokenizer

import evaluate as ev
from benchmark_nllb_moe_54b import (DTYPES, get_gpu_info, load_benchmark_model, peak_gpu_gb, save_json,
                                    summarize_device_map)
from dataset_pipeline import build_translation_pair, load_config, load_flores_language, validate_pair

# NLLB-MoE routes every token to its top-2 experts, so only 2 experts per MoE layer are used.
MOE_ACTIVE_EXPERTS = 2

# High/low resource as in the NLLB-200 paper (low = under 1M bitext sentences).
# Override with a `resource_levels:` mapping in config.yaml.
RESOURCE_LEVELS = {
    "eng_Latn": "high", "rus_Cyrl": "high", "arb_Arab": "high", "cmn_Hant": "high", "swh_Latn": "high",
    "bak_Cyrl": "low", "yue_Hant": "low", "ace_Arab": "low",
}

# A percentage of the teacher's score only makes sense on a scale with a true zero, which COMET lacks.
RETAINED_PCT_METRICS = {"chrF++", "spBLEU"}


def efficiency_path(results_dir, model, quant):
    return os.path.join(results_dir, f"efficiency_{model}_{quant}.json")


# ---------------------------------------------------------------------------
# measure: size
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Total parameters, and the parameters one token can actually pass through
    (for a dense model both are the same)."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleDict):
            experts = list(experts.values())
        if isinstance(experts, (list, nn.ModuleList)) and len(experts) > MOE_ACTIVE_EXPERTS:
            per_expert = sum(p.numel() for p in experts[0].parameters())
            inactive += (len(experts) - MOE_ACTIVE_EXPERTS) * per_expert
    return total, total - inactive


def weights_gb(model):
    """Memory taken by the weights at the loaded precision."""
    return round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9, 3)


def checkpoint_disk_gb(checkpoint):
    """Size of the weight files on disk (local folder or HF cache)."""
    try:
        folder = checkpoint if os.path.isdir(checkpoint) else snapshot_download(checkpoint, local_files_only=True)
    except Exception:
        return None
    files = (glob.glob(os.path.join(folder, "**", "*.safetensors"), recursive=True)
             or glob.glob(os.path.join(folder, "**", "*.bin"), recursive=True))
    return round(sum(os.path.getsize(f) for f in files) / 1e9, 3) if files else None


def describe_architecture(model):
    config = model.config
    num_experts = getattr(config, "num_experts", None)
    return {
        "model_type": config.model_type,
        "kind": "MoE" if num_experts else "dense",
        "encoder_layers": config.encoder_layers,
        "decoder_layers": config.decoder_layers,
        "d_model": config.d_model,
        "encoder_ffn_dim": config.encoder_ffn_dim,
        "decoder_ffn_dim": config.decoder_ffn_dim,
        "num_experts": num_experts,
    }


# ---------------------------------------------------------------------------
# measure: speed, memory, energy, FLOPs
# ---------------------------------------------------------------------------

def sync():
    """CUDA calls are asynchronous: wait for them so the timer measures the real work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def translate_timed(model, tokenizer, pairs, src_lang, tgt_lang, batch_size, args):
    """Seconds (tokenize + generate + decode) and generated tokens for every batch.
    Same generation settings as model_pipeline.translate_batch."""
    tokenizer.src_lang = src_lang
    gen_kwargs = {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(tgt_lang), "max_length": args.max_length}
    if args.num_beams:
        gen_kwargs["num_beams"] = args.num_beams
    input_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seconds, tokens = [], []
    for i in range(0, len(pairs), batch_size):
        sources = [p["source"] for p in pairs[i:i + batch_size]]
        sync()
        start = time.perf_counter()
        inputs = tokenizer(sources, return_tensors="pt", padding=True, truncation=True,
                           max_length=args.max_input_length).to(input_device)
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        tokenizer.batch_decode(generated, skip_special_tokens=True)
        sync()
        seconds.append(time.perf_counter() - start)
        tokens.append(int((generated != tokenizer.pad_token_id).sum().item()))
    return seconds, tokens


class GpuPowerSampler:
    """Polls nvidia-smi in a background thread; mean power x elapsed time = GPU energy.
    Covers the GPU(s) only, so it undercounts a model that is partly offloaded to CPU."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.watts = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                ).stdout
                self.watts.append(sum(float(x) for x in out.split()))
            except (OSError, ValueError, subprocess.SubprocessError):
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def mean_watts(self):
        return statistics.mean(self.watts) if self.watts else None


def measure_throughput(model, tokenizer, data, args):
    """Tokens/s at --batch_size, averaged over --runs timed passes after one warm-up batch."""
    first_direction, first_pairs = next(iter(data.items()))
    translate_timed(model, tokenizer, first_pairs[:args.batch_size], *first_direction.split("-"),
                    args.batch_size, args)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    n_sentences = sum(len(pairs) for pairs in data.values())
    tokens_per_second, sentences_per_second, total_seconds, total_tokens = [], [], 0.0, 0
    with GpuPowerSampler() as power:
        for run in range(args.runs):
            run_seconds, run_tokens = 0.0, 0
            for direction, pairs in data.items():
                seconds, tokens = translate_timed(model, tokenizer, pairs, *direction.split("-"),
                                                  args.batch_size, args)
                run_seconds += sum(seconds)
                run_tokens += sum(tokens)
            tokens_per_second.append(run_tokens / run_seconds)
            sentences_per_second.append(n_sentences / run_seconds)
            total_seconds += run_seconds
            total_tokens += run_tokens
            print(f"[measure] run {run + 1}/{args.runs}: {tokens_per_second[-1]:.1f} tokens/s")

    watts = power.mean_watts()
    return {
        "batch_size": args.batch_size,
        "runs": args.runs,
        "n_sentences": n_sentences,
        "generated_tokens_per_run": total_tokens // args.runs,
        "tokens_per_second": round(statistics.mean(tokens_per_second), 2),
        "tokens_per_second_std": round(statistics.stdev(tokens_per_second), 2) if args.runs > 1 else None,
        "sentences_per_second": round(statistics.mean(sentences_per_second), 3),
        "peak_gpu_memory_gb": peak_gpu_gb(),
        "mean_gpu_watts": round(watts, 1) if watts else None,
        "joules_per_1k_tokens": round(1000 * watts * total_seconds / total_tokens, 1) if watts else None,
    }


def measure_latency(model, tokenizer, data, args):
    """Per-sentence latency at batch size 1, which large-batch throughput hides."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    for direction, pairs in data.items():
        seconds, _ = translate_timed(model, tokenizer, pairs[:args.n_latency], *direction.split("-"), 1, args)
        latencies.extend(seconds)
    ms = 1000 * np.array(latencies)
    return {
        "n_sentences": len(latencies),
        "mean_ms": round(float(ms.mean()), 1),
        "median_ms": round(float(np.percentile(ms, 50)), 1),
        "p90_ms": round(float(np.percentile(ms, 90)), 1),
        "p99_ms": round(float(np.percentile(ms, 99)), 1),
        "peak_gpu_memory_gb": peak_gpu_gb(),
    }


def measure_flops(model, tokenizer, data, args):
    """FLOPs of generate() at batch size 1 (no padding), encoder cost included.
    torch's counter only knows matmul-like ops, so this is a lower bound."""
    direction, pairs = next(iter(data.items()))
    pairs = pairs[:args.n_flops]
    with FlopCounterMode(display=False) as counter:
        _, tokens = translate_timed(model, tokenizer, pairs, *direction.split("-"), 1, args)
    flops = counter.get_total_flops()
    return {
        "n_sentences": len(pairs),
        "gflops_per_token": round(flops / sum(tokens) / 1e9, 3),
        "gflops_per_sentence": round(flops / len(pairs) / 1e9, 2),
    }


def measure(args):
    config = load_config(args.config)
    dataset = config["dataset"]
    args.max_input_length = config["dataloader"]["max_length"]
    directions = args.directions or ["-".join(config["directions"][0])]
    model_name = args.checkpoint.split("/")[-1]
    out_json = efficiency_path(args.results_dir, model_name, args.precision)

    # Data and language codes first: these should fail in seconds, not after a long model load.
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    data = {}
    for direction in directions:
        src_lang, tgt_lang = direction.split("-")
        for lang in (src_lang, tgt_lang):
            if tokenizer.convert_tokens_to_ids(lang) == tokenizer.unk_token_id:
                raise ValueError(f"{lang} is not a language code of {args.checkpoint}'s tokenizer, so "
                                 "translations (and therefore timings) for it would be meaningless")
        pairs = build_translation_pair(
            load_flores_language(src_lang, dataset["split"], dataset["data_dir"], dataset["name"]),
            load_flores_language(tgt_lang, dataset["split"], dataset["data_dir"], dataset["name"]),
            src_lang, tgt_lang,
        )
        validate_pair(pairs, src_lang, tgt_lang)
        data[direction] = pairs[:args.n_sentences]

    print(f"[measure] loading {args.checkpoint} ({args.precision}) ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    model = load_benchmark_model(args.checkpoint, DTYPES[args.precision])
    model.eval()

    parameters, active_parameters = count_parameters(model)
    result = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "quant": args.precision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": get_gpu_info(),
        "load": {"seconds": round(time.time() - load_start, 1), "peak_gpu_memory_gb": peak_gpu_gb(),
                 **summarize_device_map(model)},
        "architecture": describe_architecture(model),
        "size": {
            "parameters": parameters,
            "active_parameters_per_token": active_parameters,
            "weights_gb": weights_gb(model),
            "disk_gb": checkpoint_disk_gb(args.checkpoint),
        },
        "settings": {
            "directions": directions, "split": dataset["split"], "n_sentences_per_direction": args.n_sentences,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "max_input_length": args.max_input_length,
            "num_beams": args.num_beams or model.generation_config.num_beams,
        },
        "distillation_gpu_hours": args.distillation_gpu_hours,
    }
    save_json(out_json, result)  # save progress after every stage in case a long job times out
    print(f"[measure] {parameters / 1e6:.0f}M parameters ({active_parameters / 1e6:.0f}M active per token)")

    result["throughput"] = measure_throughput(model, tokenizer, data, args)
    save_json(out_json, result)

    result["latency"] = measure_latency(model, tokenizer, data, args)
    save_json(out_json, result)
    print(f"[measure] latency at batch size 1: median {result['latency']['median_ms']} ms, "
          f"p90 {result['latency']['p90_ms']} ms")

    if args.flops:
        result["flops"] = measure_flops(model, tokenizer, data, args)
        save_json(out_json, result)
        print(f"[measure] {result['flops']['gflops_per_token']} GFLOPs per generated token")

    print(f"[measure] wrote {out_json}")


# ---------------------------------------------------------------------------
# report: quality retained
# ---------------------------------------------------------------------------

def resolve_run(spec, groups):
    """'model' or 'model:quant' -> (model, quant); the quant may be left out if outputs/ has only one."""
    model, _, quant = spec.partition(":")
    quants = sorted({q for (m, q, _, _) in groups if m == model})
    if not quants:
        raise ValueError(f"no translations of {model} found in outputs/")
    if not quant:
        if len(quants) > 1:
            raise ValueError(f"{model} has outputs for {quants}, pick one with {model}:<quant>")
        quant = quants[0]
    return model, quant


def select_common(groups, runs):
    """Keep only the directions every run has translated, and within them only the
    sentence ids every run has, so all models are scored on exactly the same sentences."""
    per_run = [{(s, t) for (m, q, s, t) in groups if (m, q) == run} for run in runs]
    directions = sorted(set.intersection(*per_run))
    skipped = sorted(set.union(*per_run) - set(directions))
    if skipped:
        print(f"[report] skipping {len(skipped)} direction(s) not translated by every model, "
              f"e.g. {'-'.join(skipped[0])}")
    if not directions:
        raise ValueError("teacher and students have no translated direction in common")

    selected = {}
    for direction in directions:
        ids = set.intersection(*[{r["id"] for r in groups[(*run, *direction)]} for run in runs])
        for run in runs:
            selected[(*run, *direction)] = [r for r in groups[(*run, *direction)] if r["id"] in ids]
    return selected, directions


def score_runs(groups, args):
    """Per-sentence statistics for every run and metric (see evaluate.score_runs), with spBLEU added."""
    metrics = {"chrF++": CHRF(word_order=2), "spBLEU": BLEU(tokenize="flores200")}
    stats, score_fns = {}, {}
    for name, metric in metrics.items():
        score_fns[name] = ev.sacrebleu_score_fn(metric)
        for key, records in groups.items():
            stats[(key, name)] = ev.sacrebleu_sentence_stats(
                metric, [r["hypothesis"] for r in records], [r["reference"] for r in records])

    if not args.no_comet:
        print(f"[report] scoring with COMET ({args.comet_model}) ...")
        comet_scores = ev.comet_segment_scores(groups, args.comet_model, args.comet_batch_size, args.gpus)
        score_fns["COMET"] = ev.mean_score
        for key, seg_scores in comet_scores.items():
            stats[(key, "COMET")] = seg_scores
    return stats, score_fns


def holm(p_values):
    """Holm correction for running many significance tests at once."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running_max = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


def quality_rows(groups, directions, teacher, students, stats, score_fns, levels, args):
    """One row per student, direction and metric: student vs teacher on the same sentences."""
    rows = []
    for student in students:
        student_rows = []
        for src_lang, tgt_lang in directions:
            teacher_key, student_key = (*teacher, src_lang, tgt_lang), (*student, src_lang, tgt_lang)
            for name, score_fn in score_fns.items():
                teacher_score = score_fn(stats[(teacher_key, name)])
                student_score = score_fn(stats[(student_key, name)])
                ci_low, ci_high, p_value = ev.paired_bootstrap(
                    score_fn, stats[(teacher_key, name)], stats[(student_key, name)],
                    args.bootstrap_samples, args.seed,
                )
                retained = name in RETAINED_PCT_METRICS and teacher_score
                student_rows.append({
                    "teacher": teacher[0],
                    "model": student[0],
                    "direction": f"{src_lang}-{tgt_lang}",
                    "resource_group": f"{levels.get(src_lang, 'unknown')}->{levels.get(tgt_lang, 'unknown')}",
                    "metric": name,
                    "teacher_score": round(teacher_score, 3),
                    "model_score": round(student_score, 3),
                    # Negative = this model is worse than the teacher.
                    "delta": round(student_score - teacher_score, 3) + 0.0,
                    "retained_pct": round(100 * student_score / teacher_score, 2) if retained else "",
                    "ci_low": round(ci_low, 3) + 0.0,
                    "ci_high": round(ci_high, 3) + 0.0,
                    "p_value": round(p_value, 4),
                    "n_sentences": len(groups[student_key]),
                })
        for row, p_holm in zip(student_rows, holm([r["p_value"] for r in student_rows])):
            row["p_holm"] = round(p_holm, 4)
        rows.extend(student_rows)
    return rows


def group_rows(rows):
    """Average the per-direction rows per resource group (and over all directions),
    keeping the worst direction, which the average hides."""
    buckets = {}
    for row in rows:
        for group in ("all", row["resource_group"]):
            buckets.setdefault((row["model"], group, row["metric"]), []).append(row)

    out = []
    for (student, group, metric), members in buckets.items():
        retained = [r["retained_pct"] for r in members if r["retained_pct"] != ""]
        worst = min(members, key=lambda r: r["retained_pct"] if retained else r["delta"])
        out.append({
            "model": student,
            "resource_group": group,
            "metric": metric,
            "n_directions": len(members),
            "mean_teacher_score": round(ev.mean_score([r["teacher_score"] for r in members]), 3),
            "mean_model_score": round(ev.mean_score([r["model_score"] for r in members]), 3),
            "mean_delta": round(ev.mean_score([r["delta"] for r in members]), 3),
            "mean_retained_pct": round(ev.mean_score(retained), 2) if retained else "",
            "n_significant_holm": sum(r["p_holm"] < 0.05 for r in members),
            "worst_direction": worst["direction"],
            "worst_delta": worst["delta"],
            "worst_retained_pct": worst["retained_pct"],
        })
    return out


# ---------------------------------------------------------------------------
# report: efficiency summary
# ---------------------------------------------------------------------------

def load_efficiency(results_dir, run):
    path = efficiency_path(results_dir, *run)
    if not os.path.exists(path):
        print(f"[warn] {path} not found: run `measure` for {run[0]} (its efficiency columns are left empty)")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def warn_if_not_comparable(teacher_eff, student_eff, student):
    """Speedup only means something if both models were measured under the same conditions."""
    differences = []
    for label, get in [
        ("GPU", lambda e: e["gpu"].get("name")),
        ("precision", lambda e: e["quant"]),
        ("CPU offload", lambda e: e["load"].get("cpu_offload")),
        ("settings", lambda e: e["settings"]),
    ]:
        if get(teacher_eff) != get(student_eff):
            differences.append(f"{label}: teacher={get(teacher_eff)} student={get(student_eff)}")
    if differences:
        print(f"[warn] {student} was not measured like the teacher, so its speedup is not like-for-like:")
        for difference in differences:
            print(f"         {difference}")


def ratio(numerator, denominator, digits=2):
    return round(numerator / denominator, digits) if numerator and denominator else ""


def summary_rows(teacher, students, roles, results_dir, groups):
    """One row per model with every efficiency number next to its average quality."""
    teacher_eff = load_efficiency(results_dir, teacher)
    overall = {(g["model"], g["metric"]): g for g in groups if g["resource_group"] == "all"}
    metrics = sorted({g["metric"] for g in groups})

    rows = []
    for run in [teacher, *students]:
        is_teacher = run == teacher
        eff = teacher_eff if is_teacher else load_efficiency(results_dir, run)
        if eff and teacher_eff and not is_teacher:
            warn_if_not_comparable(teacher_eff, eff, run[0])
        size, arch = eff.get("size", {}), eff.get("architecture", {})
        throughput, latency = eff.get("throughput", {}), eff.get("latency", {})
        teacher_size, teacher_throughput = teacher_eff.get("size", {}), teacher_eff.get("throughput", {})

        row = {
            "model": run[0],
            "quant": run[1],
            "role": roles[run],
            "architecture": f"{arch['kind']} {arch['encoder_layers']}enc/{arch['decoder_layers']}dec "
                            f"d={arch['d_model']}" if arch else "",
            "parameters_m": ratio(size.get("parameters"), 1e6, 1),
            "active_parameters_m": ratio(size.get("active_parameters_per_token"), 1e6, 1),
            "compression_ratio": ratio(teacher_size.get("parameters"), size.get("parameters")),
            "compression_ratio_active": ratio(teacher_size.get("active_parameters_per_token"),
                                              size.get("active_parameters_per_token")),
            "disk_gb": size.get("disk_gb", ""),
            "peak_gpu_memory_gb": throughput.get("peak_gpu_memory_gb", ""),
            "batch_size": throughput.get("batch_size", ""),
            "tokens_per_second": throughput.get("tokens_per_second", ""),
            "speedup": ratio(throughput.get("tokens_per_second"), teacher_throughput.get("tokens_per_second")),
            "latency_median_ms": latency.get("median_ms", ""),
            "latency_p90_ms": latency.get("p90_ms", ""),
            "joules_per_1k_tokens": throughput.get("joules_per_1k_tokens", ""),
            "gflops_per_token": eff.get("flops", {}).get("gflops_per_token", ""),
            "distillation_gpu_hours": eff.get("distillation_gpu_hours", ""),
        }
        for metric in metrics:
            # Every student is scored on the same directions, so any student's row has the teacher's mean.
            group = overall[(students[0][0] if is_teacher else run[0], metric)]
            row[metric] = group["mean_teacher_score" if is_teacher else "mean_model_score"]
            if metric in RETAINED_PCT_METRICS:
                row[f"{metric}_retained_pct"] = 100.0 if is_teacher else group["mean_retained_pct"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[report] wrote {len(rows)} rows -> {path}")


def report(args):
    levels = {**RESOURCE_LEVELS, **(load_config(args.config).get("resource_levels") or {})}
    all_groups = ev.load_outputs(args.outputs_dir)
    teacher = resolve_run(args.teacher, all_groups)
    # Baselines (e.g. a non-distilled model of the same size) are compared to the teacher
    # exactly like students; only their label in the summary differs.
    roles = {teacher: "teacher"}
    roles.update({resolve_run(spec, all_groups): "student" for spec in args.students})
    roles.update({resolve_run(spec, all_groups): "baseline" for spec in args.baselines})
    students = [run for run in roles if run != teacher]

    groups, directions = select_common(all_groups, [teacher, *students])
    print(f"[report] {len(students)} model(s) vs {teacher[0]} on {len(directions)} direction(s)")
    stats, score_fns = score_runs(groups, args)

    quality = quality_rows(groups, directions, teacher, students, stats, score_fns, levels, args)
    grouped = group_rows(quality)
    summary = summary_rows(teacher, students, roles, args.results_dir, grouped)
    write_csv(f"{args.out_prefix}_quality.csv", quality)
    write_csv(f"{args.out_prefix}_groups.csv", grouped)
    write_csv(f"{args.out_prefix}_summary.csv", summary)

    print(f"\n{'model':<26}{'role':<10}{'params(M)':>10}{'compr.':>8}{'mem(GB)':>9}{'tok/s':>9}{'speedup':>9}"
          f"{'p50(ms)':>9}{'p90(ms)':>9}{'chrF++':>8}{'retained%':>11}")
    for r in summary:
        print(f"{r['model'][:25]:<26}{r['role']:<10}{r['parameters_m']:>10}{r['compression_ratio']:>8}"
              f"{r['peak_gpu_memory_gb']:>9}{r['tokens_per_second']:>9}{r['speedup']:>9}"
              f"{r['latency_median_ms']:>9}{r['latency_p90_ms']:>9}{r['chrF++']:>8}"
              f"{r['chrF++_retained_pct']:>11}")

    print(f"\ndelta = model minus {teacher[0]}, negative = worse than the teacher; sig = directions with Holm p < 0.05")
    print(f"{'model':<26}{'group':<12}{'metric':<8}{'dirs':>5}{'delta':>8}{'retained%':>11}{'sig':>5}"
          f"  worst direction")
    for g in grouped:
        worst = g["worst_retained_pct"] if g["worst_retained_pct"] != "" else g["worst_delta"]
        print(f"{g['model'][:25]:<26}{g['resource_group']:<12}{g['metric']:<8}{g['n_directions']:>5}"
              f"{g['mean_delta']:>8}{g['mean_retained_pct']:>11}{g['n_significant_holm']:>5}"
              f"  {g['worst_direction']} ({worst})")


def main():
    parser = argparse.ArgumentParser(description="Efficiency of distilled NLLB students vs their teacher")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="size, speed and memory of ONE model (needs a GPU job)")
    m.add_argument("--config", default="config.yaml")
    m.add_argument("--checkpoint", required=True, help="HF model id, e.g. facebook/nllb-200-distilled-1.3B")
    m.add_argument("--precision", choices=list(DTYPES), default="fp32",
                   help="also used as the quant label, as in outputs/<model>/<quant>/")
    m.add_argument("--directions", nargs="+", help="SRC-TGT FLORES codes (default: first direction in --config)")
    m.add_argument("--n_sentences", type=int, default=200, help="sentences per direction in each throughput run")
    m.add_argument("--n_latency", type=int, default=50, help="sentences per direction timed at batch size 1")
    m.add_argument("--batch_size", type=int, default=16)
    m.add_argument("--runs", type=int, default=3, help="timed throughput passes to average over")
    m.add_argument("--num_beams", type=int, help="default: the model's own generation config, as in the pipeline")
    m.add_argument("--max_length", type=int, default=256, help="matches run_evaluation.py's default")
    m.add_argument("--flops", action="store_true", help="also count FLOPs per generated token")
    m.add_argument("--n_flops", type=int, default=8, help="sentences used for --flops")
    m.add_argument("--distillation_gpu_hours", type=float,
                   help="cost of producing this student (incl. teacher-generated data); cannot be measured "
                        "here, it is only stored so that it shows up in the report")
    m.add_argument("--results_dir", default="results")
    m.set_defaults(func=measure)

    r = sub.add_parser("report", help="combine measure files and outputs/ into teacher-vs-student tables")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--teacher", required=True, help="model name as in outputs/, optionally model:quant")
    r.add_argument("--students", nargs="+", required=True,
                   help="distilled model(s), names as in outputs/, optionally model:quant")
    r.add_argument("--baselines", nargs="*", default=[],
                   help="non-distilled model(s) to compare to the teacher in the same tables")
    r.add_argument("--outputs_dir", default="outputs", help="folder with inference JSONL files")
    r.add_argument("--results_dir", default="results", help="folder with the measure files")
    r.add_argument("--out_prefix", default="results/efficiency_report")
    r.add_argument("--bootstrap_samples", type=int, default=1000)
    r.add_argument("--seed", type=int, default=12345)
    r.add_argument("--no_comet", action="store_true", help="skip COMET (e.g. quick test without GPU)")
    r.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    r.add_argument("--comet_batch_size", type=int, default=32)
    r.add_argument("--gpus", type=int, default=1, help="GPUs for COMET (0 = CPU)")
    r.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
