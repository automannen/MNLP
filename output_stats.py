"""Surface statistics of the saved translations: repetitions, question marks and length.

For every <model>/<quant>/<src>-<tgt>.jsonl in an outputs directory, and with the reference
as the baseline for what is normal in that language:
  - repetition:     share of 4-grams that already occurred earlier in the same sentence, and
                    share of sentences stuck in a loop (a span repeated 3+ times in a row)
  - question marks: number per 100 sentences, and share of sentences with more than the reference
  - length:         hypothesis/reference length ratio, and share of sentences that are much
                    longer or much shorter than their reference

Units are whitespace-separated words, or characters for scripts written without spaces.
Lengths are always in non-whitespace characters, so they are comparable across scripts.
Nothing is translated again; CPU only and takes under a minute.
"""

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict

from evaluate import script_of
from identify_languages import load_records

# Scripts written without spaces between words: count repetitions over characters.
CHAR_SCRIPTS = {"Hant", "Hans", "Hani", "Jpan", "Thai", "Laoo", "Khmr", "Mymr", "Tibt"}

QUESTION_MARKS = "?？؟⁇"

NGRAM = 4
# A loop is a span of up to MAX_PERIOD units repeated back-to-back MIN_COPIES+ times.
MAX_PERIOD = 20
MIN_COPIES = 3
# In character mode also require the loop to cover this many characters ("1000" is not a loop).
MIN_LOOP_CHARS = 6

# A sentence is "long"/"short" if its length is beyond this factor of the reference length.
LENGTH_FACTOR = 1.5

FIELDS = ["model", "quant", "direction", "src_lang", "tgt_lang", "unit", "n_sentences",
          "rep_ngram_pct", "ref_rep_ngram_pct", "pct_sent_loop", "ref_pct_sent_loop",
          "mean_loop_copies", "qmarks_per_100", "ref_qmarks_per_100", "pct_sent_extra_qmarks",
          "len_ratio", "mean_len", "ref_mean_len", "pct_sent_long", "pct_sent_short"]


def units_of(text, char_mode):
    return [c for c in text if not c.isspace()] if char_mode else text.split()


def repeated_ngrams(units, n=NGRAM):
    """(repeated, total): n-grams that already occurred earlier in the sentence."""
    total = max(0, len(units) - n + 1)
    return total - len({tuple(units[i:i + n]) for i in range(total)}), total


def loop_copies(units, char_mode):
    """Largest number of back-to-back copies of one span, or 0 if nothing loops."""
    best = 0
    for period in range(1, min(MAX_PERIOD, len(units) // MIN_COPIES) + 1):
        run = 0
        for i in range(period, len(units)):
            run = run + 1 if units[i] == units[i - period] else 0
            copies = run // period + 1
            if copies > best and (not char_mode or copies * period >= MIN_LOOP_CHARS):
                best = copies
    return best if best >= MIN_COPIES else 0


def count_file(records, char_mode):
    """Raw counts for one file; kept as sums so files can be pooled before taking ratios."""
    c = Counter(n=len(records))
    for r in records:
        hyp, ref = r["hypothesis"], r["reference"]
        hyp_units, ref_units = units_of(hyp, char_mode), units_of(ref, char_mode)

        repeated, total = repeated_ngrams(hyp_units)
        c["rep"] += repeated
        c["ngrams"] += total
        repeated, total = repeated_ngrams(ref_units)
        c["ref_rep"] += repeated
        c["ref_ngrams"] += total

        copies = loop_copies(hyp_units, char_mode)
        c["loops"] += copies > 0
        c["loop_copies"] += copies
        c["ref_loops"] += loop_copies(ref_units, char_mode) > 0

        hyp_q = sum(hyp.count(q) for q in QUESTION_MARKS)
        ref_q = sum(ref.count(q) for q in QUESTION_MARKS)
        c["qmarks"] += hyp_q
        c["ref_qmarks"] += ref_q
        c["extra_qmarks"] += hyp_q > ref_q

        hyp_len, ref_len = len("".join(hyp.split())), len("".join(ref.split()))
        c["len"] += hyp_len
        c["ref_len"] += ref_len
        c["long"] += hyp_len > LENGTH_FACTOR * ref_len
        c["short"] += hyp_len < ref_len / LENGTH_FACTOR
    return c


def pct(part, whole):
    return round(100 * part / whole, 2) if whole else 0.0


def stats(c):
    return {
        "n_sentences": c["n"],
        "rep_ngram_pct": pct(c["rep"], c["ngrams"]),
        "ref_rep_ngram_pct": pct(c["ref_rep"], c["ref_ngrams"]),
        "pct_sent_loop": pct(c["loops"], c["n"]),
        "ref_pct_sent_loop": pct(c["ref_loops"], c["n"]),
        # Average over the looping sentences only.
        "mean_loop_copies": round(c["loop_copies"] / c["loops"], 1) if c["loops"] else 0.0,
        "qmarks_per_100": pct(c["qmarks"], c["n"]),
        "ref_qmarks_per_100": pct(c["ref_qmarks"], c["n"]),
        "pct_sent_extra_qmarks": pct(c["extra_qmarks"], c["n"]),
        "len_ratio": round(c["len"] / c["ref_len"], 3) if c["ref_len"] else 0.0,
        "mean_len": round(c["len"] / c["n"], 1),
        "ref_mean_len": round(c["ref_len"] / c["n"], 1),
        "pct_sent_long": pct(c["long"], c["n"]),
        "pct_sent_short": pct(c["short"], c["n"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Repetition, question-mark and length statistics of saved translations"
    )
    parser.add_argument("--outputs_dir", default="outputs_devtest",
                        help="directory with <model>/<quant>/<src>-<tgt>.jsonl translations")
    parser.add_argument("--csv", default="results/output_stats_devtest.csv",
                        help="where to write the per-direction table")
    parser.add_argument("--model", default=None,
                    help="evaluate only this model (default: all models)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.outputs_dir, args.model or "*", "*", "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no .jsonl files found under {args.outputs_dir}")

    rows = []
    pooled = defaultdict(Counter)
    for path in paths:
        model, quant = path.split(os.sep)[-3:-1]
        direction = os.path.basename(path)[:-len(".jsonl")]
        src_lang, tgt_lang = direction.split("-")
        char_mode = script_of(tgt_lang) in CHAR_SCRIPTS
        counts = count_file(load_records(path), char_mode)
        pooled[(tgt_lang, model, quant)] += counts
        rows.append({"model": model, "quant": quant, "direction": direction, "src_lang": src_lang,
                     "tgt_lang": tgt_lang, "unit": "char" if char_mode else "word", **stats(counts)})

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stats] wrote {len(rows)} rows -> {args.csv}")

    print(f"\npooled over source languages; (ref) = same statistic on the references")
    print(f"{'tgt_lang':<10}{'model':<32}{'rep4%':>7}{'(ref)':>7}{'loop%':>7}{'(ref)':>7}"
          f"{'?/100':>8}{'(ref)':>7}{'extra?%':>9}{'len':>7}{'long%':>7}{'short%':>8}")
    for (tgt_lang, model, quant), counts in sorted(pooled.items()):
        s = stats(counts)
        print(f"{tgt_lang:<10}{model + ' ' + quant:<32}{s['rep_ngram_pct']:>7.1f}"
              f"{s['ref_rep_ngram_pct']:>7.1f}{s['pct_sent_loop']:>7.1f}{s['ref_pct_sent_loop']:>7.1f}"
              f"{s['qmarks_per_100']:>8.1f}{s['ref_qmarks_per_100']:>7.1f}"
              f"{s['pct_sent_extra_qmarks']:>9.1f}{s['len_ratio']:>7.2f}"
              f"{s['pct_sent_long']:>7.1f}{s['pct_sent_short']:>8.1f}")


if __name__ == "__main__":
    main()
