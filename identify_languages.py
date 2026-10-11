"""Check which language the saved translations are actually written in.

Runs two fastText language-identification models over the hypotheses (and, as a
sanity check, the references) in an outputs directory:
  - lid218:  Meta's NLLB language identifier (~200 labels, no Jawi Malay label)
  - glotlid: GlotLID (~2100 labels, includes zsm_Arab, min_Arab, zgh_Tfng)

With --compare MODEL_A MODEL_B it instead runs a paired bootstrap between two models on
how often their output is identified as the target language, per direction.

Run once with --prepare_only where internet is available (e.g. the Snellius login
node) to cache both models, since compute nodes have no internet access.
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import fasttext
from huggingface_hub import hf_hub_download

from evaluate import mean_score, paired_bootstrap

LID_MODELS = {
    "lid218": "facebook/fasttext-language-identification",
    "glotlid": "cis-lmu/glotlid",
}

# FLORES code -> label the identifier uses for that language, where they differ.
LABEL_ALIASES = {
    "lid218": {"cmn_Hant": "zho_Hant"},
    "glotlid": {"cmn_Hant": "cmn_Hani", "yue_Hant": "yue_Hani"},
}

FIELDS = ["lid_model", "model", "quant", "direction", "src_lang", "tgt_lang", "text",
          "n_sentences", "expected_label", "pct_expected", "label", "pct"]


def load_lid_model(name):
    # fastText prints a deprecation warning on every load_model call.
    fasttext.FastText.eprint = lambda x: None
    return fasttext.load_model(hf_hub_download(LID_MODELS[name], "model.bin"))


def identify(lid_model, texts):
    """Top-1 language label for each text."""
    labels, _ = lid_model.predict([t.replace("\n", " ") for t in texts], k=1)
    return [label[0].replace("__label__", "") for label in labels]


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize(counts, expected, top_k):
    """Rows of (expected_label, pct_expected, label, pct) for the top_k labels."""
    n = sum(counts.values())
    pct_expected = round(100 * counts[expected] / n, 1)
    return [(expected, pct_expected, label, round(100 * count / n, 1))
            for label, count in counts.most_common(top_k)]


def run(lid_name, outputs_dir, top_k):
    lid_model = load_lid_model(lid_name)
    aliases = LABEL_ALIASES[lid_name]

    rows = []
    ref_counts = {}
    pooled = defaultdict(Counter)
    paths = sorted(glob.glob(os.path.join(outputs_dir, "*", "*", "*.jsonl")))
    for path in paths:
        model, quant = path.split(os.sep)[-3:-1]
        direction = os.path.basename(path)[:-len(".jsonl")]
        src_lang, tgt_lang = direction.split("-")
        expected = aliases.get(tgt_lang, tgt_lang)
        records = load_records(path)

        counts = Counter(identify(lid_model, [r["hypothesis"] for r in records]))
        pooled[(model, quant, tgt_lang)] += counts
        for row in summarize(counts, expected, top_k):
            rows.append([lid_name, model, quant, direction, src_lang, tgt_lang, "hypothesis",
                         len(records), *row])

        # References are the same for every model, so identify each target language once.
        if tgt_lang not in ref_counts:
            ref_counts[tgt_lang] = Counter(identify(lid_model, [r["reference"] for r in records]))
            for row in summarize(ref_counts[tgt_lang], expected, top_k):
                rows.append([lid_name, "reference", "", direction, src_lang, tgt_lang, "reference",
                             len(records), *row])

    print(f"\n[langid] {lid_name}: {len(paths)} files from {outputs_dir}")
    for tgt_lang in sorted(ref_counts):
        expected = aliases.get(tgt_lang, tgt_lang)
        print(f"  {tgt_lang}")
        print(f"    {'reference':30s}{format_counts(ref_counts[tgt_lang], expected)}")
        for (model, quant, tgt), counts in sorted(pooled.items()):
            if tgt == tgt_lang:
                print(f"    {model + ' ' + quant:30s}{format_counts(counts, expected)}")
    return rows


def format_counts(counts, expected):
    n = sum(counts.values())
    others = ", ".join(f"{label} {100 * count / n:.0f}%" for label, count in counts.most_common(5)
                       if label != expected and count / n >= 0.02)
    return f"{100 * counts[expected] / n:5.1f}% {expected}" + (f" | {others}" if others else "")


def compare(lid_names, outputs_dir, model_a, model_b, bootstrap_samples, seed, out_csv):
    """Paired bootstrap between two models on the share of sentences identified as the
    target language, per direction and identifier. Positive gain = model_b is more often
    in the target language than model_a."""
    paths = {}
    for model in (model_a, model_b):
        for path in sorted(glob.glob(os.path.join(outputs_dir, model, "*", "*.jsonl"))):
            quant = path.split(os.sep)[-2]
            paths[(model, os.path.basename(path)[:-len(".jsonl")])] = (quant, path)
    directions = sorted({d for m, d in paths if m == model_a} & {d for m, d in paths if m == model_b})
    if not directions:
        raise ValueError(f"no direction with outputs of both {model_a} and {model_b} in {outputs_dir}")
    print(f"[compare] {model_b} vs {model_a} on {len(directions)} direction(s)")
    n_tests = len(directions) * len(lid_names)

    rows = []
    for lid_name in lid_names:
        lid_model = load_lid_model(lid_name)
        for direction in directions:
            expected = LABEL_ALIASES[lid_name].get(direction.split("-")[1], direction.split("-")[1])
            (quant_a, path_a), (quant_b, path_b) = paths[(model_a, direction)], paths[(model_b, direction)]
            records_a, records_b = load_records(path_a), load_records(path_b)
            if [r["id"] for r in records_a] != [r["id"] for r in records_b]:
                raise ValueError(f"{path_a} and {path_b} contain different sentence ids")

            # Per sentence: 100 if identified as the target language, else 0 (mean = percentage).
            hits_a = [100.0 * (label == expected)
                      for label in identify(lid_model, [r["hypothesis"] for r in records_a])]
            hits_b = [100.0 * (label == expected)
                      for label in identify(lid_model, [r["hypothesis"] for r in records_b])]
            ci_low, ci_high, p_value = paired_bootstrap(mean_score, hits_a, hits_b, bootstrap_samples, seed)
            rows.append({
                "lid_model": lid_name,
                "model_a": model_a,
                "model_b": model_b,
                "quant_a": quant_a,
                "quant_b": quant_b,
                "direction": direction,
                "metric": "pct_target_lang",
                "higher_is_better": True,
                "score_a": round(mean_score(hits_a), 3),
                "score_b": round(mean_score(hits_b), 3),
                "gain_b_over_a": round(mean_score(hits_b) - mean_score(hits_a), 3) + 0.0,
                "ci_low": round(ci_low, 3) + 0.0,
                "ci_high": round(ci_high, 3) + 0.0,
                "p_value": round(p_value, 4),
                # Bonferroni: corrects for running n_tests tests at once.
                "p_bonferroni": round(min(1.0, p_value * n_tests), 4),
                "n_sentences": len(records_a),
            })

    out_csv = out_csv or os.path.join("results", f"langid_compare_{model_a}_vs_{model_b}.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] wrote {len(rows)} rows -> {out_csv}")

    print(f"\ngain = {model_b} minus {model_a} in % of sentences identified as the target language")
    print(f"{'lid':<9}{'direction':<20}{'a':>7}{'b':>7}{'gain':>8}{'95% CI':>18}{'p':>8}{'p_bonf':>8}")
    for r in rows:
        ci = f"[{r['ci_low']}, {r['ci_high']}]"
        print(f"{r['lid_model']:<9}{r['direction']:<20}{r['score_a']:>7.1f}{r['score_b']:>7.1f}"
              f"{r['gain_b_over_a']:>8.1f}{ci:>18}{r['p_value']:>8}{r['p_bonferroni']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify the language of saved translations with lid218 and GlotLID"
    )
    parser.add_argument("--outputs_dir", default="outputs_devtest",
                        help="directory with <model>/<quant>/<src>-<tgt>.jsonl translations")
    parser.add_argument("--lid_models", nargs="+", default=list(LID_MODELS), choices=list(LID_MODELS))
    parser.add_argument("--csv", default="results/langid_devtest.csv", help="where to write the table")
    parser.add_argument("--top_k", type=int, default=5,
                        help="number of most frequent labels to keep per file")
    parser.add_argument("--prepare_only", action="store_true",
                        help="only download and cache the language-ID models, then exit")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="paired bootstrap between two models instead of the per-file table")
    parser.add_argument("--compare_csv",
                        help="output of --compare (default: results/langid_compare_<A>_vs_<B>.csv)")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.prepare_only:
        for name in args.lid_models:
            print(f"[langid] cached {name} at {hf_hub_download(LID_MODELS[name], 'model.bin')}")
        return

    if args.compare:
        compare(args.lid_models, args.outputs_dir, *args.compare, args.bootstrap_samples, args.seed,
                args.compare_csv)
        return

    rows = []
    for name in args.lid_models:
        rows += run(name, args.outputs_dir, args.top_k)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"\n[langid] wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
