import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict

from sacrebleu.metrics import CHRF, TER

REQUIRED_FIELDS = {"id", "src_lang", "tgt_lang", "model", "quant", "source", "reference", "hypothesis"}

# For TER lower is better; for everything else higher is better.
HIGHER_IS_BETTER = {"chrF++": True, "TER": False, "COMET": True}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_outputs(outputs_dir):
    """Group all records by (model, quant, src_lang, tgt_lang), sorted by id."""
    groups = defaultdict(list)
    paths = sorted(glob.glob(os.path.join(outputs_dir, "**", "*.jsonl"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no .jsonl files found under {outputs_dir}")

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = REQUIRED_FIELDS - record.keys()
                if missing:
                    raise ValueError(f"{path}:{line_no} is missing fields {sorted(missing)}")
                key = (record["model"], record["quant"], record["src_lang"], record["tgt_lang"])
                groups[key].append(record)

    for key, records in groups.items():
        records.sort(key=lambda r: r["id"])
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sentence ids in run {key}")

    return groups


def script_of(lang_code):
    """FLORES codes look like 'swh_Latn': the part after '_' is the script."""
    return lang_code.split("_")[-1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_sacrebleu_metrics():
    return {
        # chrF++: character n-grams plus word uni/bigrams (word_order=2).
        "chrF++": CHRF(word_order=2),
        # Chinese/Cantonese have no spaces, so default TER mostly counts spaces the model
        # inserts; normalized + asian_support tokenizes CJK per character.
        "TER": TER(normalized=True, asian_support=True),
    }


def sacrebleu_sentence_stats(metric, hypotheses, references):
    """Per-sentence sufficient statistics, so corpus scores can be recomputed
    on bootstrap resamples without re-tokenizing. Uses sacrebleu internals
    (the same ones sacrebleu's own significance testing uses)."""
    return metric._extract_corpus_statistics(hypotheses, [references])


def sacrebleu_score_fn(metric):
    return lambda stats: metric._aggregate_and_compute(stats).score


def comet_segment_scores(groups, model_name, batch_size, gpus):
    """COMET score per sentence for every run, scaled to 0-100."""
    from comet import download_model, load_from_checkpoint

    comet_model = load_from_checkpoint(download_model(model_name))
    scores = {}
    for key, records in groups.items():
        data = [
            {"src": r["source"], "mt": r["hypothesis"], "ref": r["reference"]}
            for r in records
        ]
        output = comet_model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
        scores[key] = [100 * s for s in output.scores]
    return scores


def mean_score(values):
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def paired_bootstrap(score_fn, base_stats, sys_stats, n_samples, seed):
    """Resample the same sentence indices for baseline and quantized run and
    look at the distribution of (quantized score - baseline score)."""
    rng = random.Random(seed)
    n = len(base_stats)
    deltas = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        sys_score = score_fn([sys_stats[i] for i in idx])
        base_score = score_fn([base_stats[i] for i in idx])
        deltas.append(sys_score - base_score)

    deltas.sort()
    ci_low = deltas[int(0.025 * n_samples)]
    ci_high = deltas[int(0.975 * n_samples) - 1]
    share_le_zero = sum(d <= 0 for d in deltas) / n_samples
    share_ge_zero = sum(d >= 0 for d in deltas) / n_samples
    p_value = min(1.0, 2 * min(share_le_zero, share_ge_zero))
    return ci_low, ci_high, p_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_runs(groups, args):
    """Per-sentence statistics for every run and metric, plus the function that
    turns a list of those statistics into a corpus score."""
    metrics = make_sacrebleu_metrics()
    stats = {}          # (run_key, metric_name) -> per-sentence stats
    score_fns = {}      # metric_name -> function(stats) -> corpus score
    for name, metric in metrics.items():
        score_fns[name] = sacrebleu_score_fn(metric)
        for key, records in groups.items():
            hyps = [r["hypothesis"] for r in records]
            refs = [r["reference"] for r in records]
            stats[(key, name)] = sacrebleu_sentence_stats(metric, hyps, refs)

    if not args.no_comet:
        print(f"[eval] scoring with COMET ({args.comet_model}) ...")
        comet_scores = comet_segment_scores(groups, args.comet_model, args.comet_batch_size, args.gpus)
        score_fns["COMET"] = mean_score
        for key, seg_scores in comet_scores.items():
            stats[(key, "COMET")] = seg_scores

    return stats, score_fns


def evaluate(args):
    groups = load_outputs(args.outputs_dir)
    print(f"[eval] found {len(groups)} run(s)")

    # 1. Per-sentence statistics for every run and metric.
    stats, score_fns = score_runs(groups, args)

    # 2. Corpus scores and degradation relative to the baseline run.
    rows = []
    for key in sorted(groups):
        model, quant, src_lang, tgt_lang = key
        records = groups[key]
        base_key = (model, args.baseline, src_lang, tgt_lang)
        has_baseline = base_key in groups and quant != args.baseline

        if base_key in groups and [r["id"] for r in groups[base_key]] != [r["id"] for r in records]:
            raise ValueError(f"run {key} and its baseline {base_key} contain different sentence ids")

        hyp_tokens = [r["hyp_tokens"] for r in records if "hyp_tokens" in r]
        latencies = [r["latency_s"] for r in records if "latency_s" in r]

        for name, score_fn in score_fns.items():
            score = score_fn(stats[(key, name)])
            row = {
                "model": model,
                "quant": quant,
                "direction": f"{src_lang}-{tgt_lang}",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_script": script_of(src_lang),
                "tgt_script": script_of(tgt_lang),
                "metric": name,
                "higher_is_better": HIGHER_IS_BETTER[name],
                "score": round(score, 3),
                "baseline_score": "",
                "degradation": "",
                "degradation_pct": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "n_sentences": len(records),
                "mean_hyp_tokens": round(mean_score(hyp_tokens), 2) if hyp_tokens else "",
                "mean_latency_s": round(mean_score(latencies), 5) if latencies else "",
            }

            if has_baseline:
                base_score = score_fn(stats[(base_key, name)])
                # Make "positive = worse" hold for every metric, including TER.
                sign = 1 if HIGHER_IS_BETTER[name] else -1
                degradation = sign * (base_score - score)
                ci_low, ci_high, p_value = paired_bootstrap(
                    score_fn, stats[(base_key, name)], stats[(key, name)],
                    args.bootstrap_samples, args.seed,
                )
                row.update({
                    "baseline_score": round(base_score, 3),
                    "degradation": round(degradation, 3) + 0.0,  # +0.0 avoids printing -0.0
                    "degradation_pct": round(100 * degradation / base_score, 2) if base_score else "",
                    # Interval of (quantized - baseline) in the metric's own direction.
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "p_value": round(p_value, 4),
                })
            rows.append(row)

    # 3. Save and summarize.
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {len(rows)} rows -> {args.out_csv}")

    print(f"\n{'model':<26}{'quant':<8}{'direction':<20}{'metric':<8}{'score':>8}{'degr.':>8}{'p':>8}")
    for r in rows:
        print(f"{r['model'][:25]:<26}{r['quant']:<8}{r['direction']:<20}{r['metric']:<8}"
              f"{r['score']:>8}{str(r['degradation']):>8}{str(r['p_value']):>8}")

    missing = sorted({(m, s, t) for (m, q, s, t) in groups if (m, args.baseline, s, t) not in groups})
    if missing:
        print(f"\n[warn] no '{args.baseline}' baseline for: {missing} (degradation left empty)")

def get_model_quant(groups, current_model):
    quants = {quant for (model, quant, _, _) in groups if model == current_model}
    
    if len(quants) != 1:
        raise ValueError(f"Expected one quant level for {current_model} but found {sorted(quants)}")
    
    return next(iter(quants))

def compare(args):
    """Paired bootstrap between two models (both at the --baseline precision),
    per direction and metric. Positive gain = model_b is better than model_a."""
    model_a, model_b = args.compare
    groups = {
        key: records for key, records in load_outputs(args.outputs_dir).items()
        if key[0] in (model_a, model_b)
    }
    
    directions_a = {(src, tgt) for (model, _, src, tgt) in groups if model == model_a}
    
    directions_b = {(src, tgt) for (model, _, src, tgt) in groups if model == model_b}
    
    # take intersection since 54B model has less directions
    directions = sorted(directions_a & directions_b)
    
    quant_model_a = get_model_quant(groups, model_a)
    quant_model_b = get_model_quant(groups, model_b)

    if not directions:
        raise ValueError(f"need '{args.baseline}' outputs of both models for every direction, missing: "
                         f"{missing or [model_a, model_b]}")
    print(f"[compare] {model_b} vs {model_a} on {len(directions)} direction(s)")

    stats, score_fns = score_runs(groups, args)
    n_tests = len(directions) * len(score_fns)
    

    rows = []
    for src_lang, tgt_lang in directions:
        key_a = (model_a, quant_model_a, src_lang, tgt_lang)
        key_b = (model_b, quant_model_b, src_lang, tgt_lang)
        if [r["id"] for r in groups[key_a]] != [r["id"] for r in groups[key_b]]:
            raise ValueError(f"{key_a} and {key_b} contain different sentence ids")

        for name, score_fn in score_fns.items():
            score_a = score_fn(stats[(key_a, name)])
            score_b = score_fn(stats[(key_b, name)])
            ci_low, ci_high, p_value = paired_bootstrap(
                score_fn, stats[(key_a, name)], stats[(key_b, name)],
                args.bootstrap_samples, args.seed,
            )
            # Make "positive = model_b better" hold for every metric, including TER.
            sign = 1 if HIGHER_IS_BETTER[name] else -1
            gain_low, gain_high = sorted((sign * ci_low, sign * ci_high))
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "quant_a": quant_model_a,
                "quant_b": quant_model_b,
                "direction": f"{src_lang}-{tgt_lang}",
                "metric": name,
                "higher_is_better": HIGHER_IS_BETTER[name],
                "score_a": round(score_a, 3),
                "score_b": round(score_b, 3),
                "gain_b_over_a": round(sign * (score_b - score_a), 3) + 0.0,
                "ci_low": round(gain_low, 3) + 0.0,
                "ci_high": round(gain_high, 3) + 0.0,
                "p_value": round(p_value, 4),
                # Bonferroni: corrects for running n_tests tests at once.
                "p_bonferroni": round(min(1.0, p_value * n_tests), 4),
                "n_sentences": len(groups[key_a]),
            })

    out_csv = args.compare_csv or os.path.join("results", f"compare_{model_a}_vs_{model_b}.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] wrote {len(rows)} rows -> {out_csv}")

    print(f"\ngain = {model_b} minus {model_a}, positive = {model_b} better")
    print(f"{'direction':<20}{'metric':<8}{'gain':>8}{'95% CI':>20}{'p':>8}{'p_bonf':>8}")
    for r in rows:
        ci = f"[{r['ci_low']}, {r['ci_high']}]"
        print(f"{r['direction']:<20}{r['metric']:<8}{r['gain_b_over_a']:>8}{ci:>20}"
              f"{r['p_value']:>8}{r['p_bonferroni']:>8}")


def main():
    parser = argparse.ArgumentParser(description="Score MT outputs and quantization degradation")
    parser.add_argument("--outputs_dir", default="outputs", help="folder with inference JSONL files")
    parser.add_argument("--baseline", default="fp32", help="quant label of the unquantized reference run")
    parser.add_argument("--out_csv", default="results/scores.csv")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no_comet", action="store_true", help="skip COMET (e.g. quick test without GPU)")
    parser.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet_batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1, help="GPUs for COMET (0 = CPU)")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="instead of the normal evaluation, test whether MODEL_B differs from MODEL_A "
                             "(model names as in outputs/, e.g. nllb-200-1.3B nllb-200-distilled-1.3B)")
    parser.add_argument("--compare_csv", help="output of --compare (default: results/compare_<A>_vs_<B>.csv)")
    args = parser.parse_args()
    if args.compare:
        compare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()