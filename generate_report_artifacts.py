"""Generate traceable result tables and the two required figures from saved experiment files.

Usage:
    python generate_report_artifacts.py scores.csv runs/test.jsonl report_artifacts

The script refuses incomplete paired data instead of silently comparing different question sets.
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from statistics import NormalDist

import matplotlib
import matplotlib.pyplot as plt


CONDITIONS = ("baseline", "length_matched", "filtered")
METRICS = ("hallucination_rate", "answer_correctness", "faithfulness", "context_precision", "context_recall")


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_mean_ci(values, seed=42, repetitions=10000):
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions)]
    return percentile(means, 0.025), percentile(means, 0.975)


def load_scores(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    by_qid = defaultdict(dict)
    for row in rows:
        by_qid[row["qid"]][row["condition"]] = row
    incomplete = {qid: sorted(set(CONDITIONS) - set(items)) for qid, items in by_qid.items()
                  if set(items) != set(CONDITIONS)}
    if incomplete:
        sample = list(incomplete.items())[:5]
        raise ValueError(f"Incomplete condition sets; examples: {sample}")
    return rows


def load_runs(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [row for row in rows if "condition" in row]
    by_key = {}
    duplicates = []
    for row in rows:
        condition = row.get("condition")
        qid = str(row.get("qid", ""))
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected condition {condition!r} for qid {qid!r}")
        key = (qid, condition)
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row
    if duplicates:
        raise ValueError(
            "Duplicate qid-condition rows in runtime log; use the cleaned unique log. "
            f"Examples: {duplicates[:5]}"
        )
    qids = sorted({key[0] for key in by_key})
    incomplete = {
        qid: sorted(set(CONDITIONS) - {condition for (row_qid, condition) in by_key if row_qid == qid})
        for qid in qids
    }
    incomplete = {qid: missing for qid, missing in incomplete.items() if missing}
    if incomplete:
        raise ValueError(f"Incomplete runtime condition sets; examples: {list(incomplete.items())[:5]}")
    return rows


def finite(values):
    return [value for value in values if math.isfinite(value)]


def rankdata_absolute(differences):
    ordered = sorted(enumerate(differences), key=lambda item: abs(item[1]))
    ranks = {}
    tie_correction = 0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and abs(ordered[j][1]) == abs(ordered[i][1]):
            j += 1
        rank = (i + 1 + j) / 2
        for index, _ in ordered[i:j]:
            ranks[index] = rank
        tie_size = j - i
        tie_correction += tie_size ** 3 - tie_size
        i = j
    return ranks, tie_correction


def wilcoxon_normal(differences):
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 0.0, 0.0, 1.0, 0
    ranks, tie_correction = rankdata_absolute(nonzero)
    w_plus = sum(ranks[i] for i, value in enumerate(nonzero) if value > 0)
    w_minus = sum(ranks[i] for i, value in enumerate(nonzero) if value < 0)
    n = len(nonzero)
    expected = n * (n + 1) / 4
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24
    z_value = max(0.0, abs(w_plus - expected) - 0.5) / math.sqrt(variance)
    p_value = min(1.0, 2 * (1 - NormalDist().cdf(z_value)))
    return min(w_plus, w_minus), z_value, p_value, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores")
    parser.add_argument("jsonl")
    parser.add_argument("outdir")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    scores = load_scores(args.scores)
    runs = load_runs(args.jsonl)

    score_values = defaultdict(lambda: defaultdict(list))
    for row in scores:
        condition = row["condition"]
        faithfulness = float(row["faithfulness"])
        if math.isfinite(faithfulness):
            score_values[condition]["hallucination_rate"].append(float(faithfulness < 1.0))
        for metric in METRICS[1:]:
            value = float(row[metric])
            if math.isfinite(value):
                score_values[condition][metric].append(value)

    summary_path = os.path.join(args.outdir, "main_metrics.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "metric", "n_valid", "mean", "ci95_low", "ci95_high"])
        writer.writeheader()
        for condition in CONDITIONS:
            for metric in METRICS:
                values = finite(score_values[condition][metric])
                lo, hi = bootstrap_mean_ci(values)
                writer.writerow({"condition": condition, "metric": metric, "n_valid": len(values),
                                 "mean": round(statistics.mean(values), 6) if values else "nan",
                                 "ci95_low": round(lo, 6) if values else "nan",
                                 "ci95_high": round(hi, 6) if values else "nan"})

    runtime = defaultdict(lambda: defaultdict(list))
    runtime_by_qid = defaultdict(dict)
    for row in runs:
        condition = row["condition"]
        qid = str(row["qid"])
        measures = {
            "latency_s": float(row["latency_s"]),
            "prompt_tokens": float(row["prompt_tokens"]),
            "chunks": float(len(row["contexts"])),
        }
        runtime_by_qid[qid][condition] = measures
        for measure, value in measures.items():
            runtime[condition][measure].append(value)

    efficiency_path = os.path.join(args.outdir, "efficiency_metrics.csv")
    with open(efficiency_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "measure", "n", "mean", "sd", "median", "q1", "q3"])
        writer.writeheader()
        for condition in CONDITIONS:
            for measure in ("latency_s", "prompt_tokens", "chunks"):
                values = sorted(runtime[condition][measure])
                writer.writerow({"condition": condition, "measure": measure, "n": len(values),
                                 "mean": round(statistics.mean(values), 4),
                                 "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                                 "median": round(statistics.median(values), 4),
                                 "q1": round(percentile(values, 0.25), 4),
                                 "q3": round(percentile(values, 0.75), 4)})

    comparison_path = os.path.join(args.outdir, "efficiency_comparisons.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as handle:
        fields = ["condition_a", "condition_b", "measure", "n_pairs", "mean_a", "mean_b",
                  "difference_b_minus_a", "percent_change", "wilcoxon_w", "z_approx", "p_raw", "n_nonzero"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition_a, condition_b in (("baseline", "filtered"), ("length_matched", "filtered")):
            for measure in ("latency_s", "prompt_tokens", "chunks"):
                qids = sorted(runtime_by_qid)
                values_a = [runtime_by_qid[qid][condition_a][measure] for qid in qids]
                values_b = [runtime_by_qid[qid][condition_b][measure] for qid in qids]
                differences = [b - a for a, b in zip(values_a, values_b)]
                w_value, z_value, p_value, n_nonzero = wilcoxon_normal(differences)
                mean_a = statistics.mean(values_a)
                mean_b = statistics.mean(values_b)
                writer.writerow({
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "measure": measure,
                    "n_pairs": len(qids),
                    "mean_a": round(mean_a, 6),
                    "mean_b": round(mean_b, 6),
                    "difference_b_minus_a": round(mean_b - mean_a, 6),
                    "percent_change": round((mean_b - mean_a) / mean_a * 100, 6),
                    "wilcoxon_w": round(w_value, 6),
                    "z_approx": round(z_value, 6),
                    "p_raw": f"{p_value:.10g}",
                    "n_nonzero": n_nonzero,
                })

    labels = ["Hallucination\nrate", "Answer\ncorrectness", "Faithfulness", "Context\nprecision", "Context\nrecall"]
    x = list(range(len(METRICS)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ci, condition in enumerate(CONDITIONS):
        means, lower, upper = [], [], []
        for metric in METRICS:
            values = score_values[condition][metric]
            mean_value = statistics.mean(values)
            lo, hi = bootstrap_mean_ci(values)
            means.append(mean_value)
            lower.append(mean_value - lo)
            upper.append(hi - mean_value)
        positions = [value + (ci - 1) * width for value in x]
        ax.bar(positions, means, width, label=condition.replace("_", " "),
               yerr=[lower, upper], capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean with bootstrap 95% CI")
    ax.set_title("Paired Test Results by Experimental Condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "figure_1_main_metrics.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    label_argument = "tick_labels" if tuple(int(part) for part in matplotlib.__version__.split(".")[:2]) >= (3, 9) else "labels"
    ax.boxplot([runtime[condition]["latency_s"] for condition in CONDITIONS],
               showfliers=True, **{label_argument: [condition.replace("_", "\n") for condition in CONDITIONS]})
    ax.set_ylabel("Generation latency (seconds per query)")
    ax.set_title("Generation Latency Distribution")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "figure_2_latency_distribution.png"), dpi=220)
    plt.close(fig)

    print(summary_path)
    print(efficiency_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
