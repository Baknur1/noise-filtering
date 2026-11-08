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
    return [row for row in rows if "condition" in row]


def finite(values):
    return [value for value in values if math.isfinite(value)]


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
    for row in runs:
        condition = row["condition"]
        runtime[condition]["latency_s"].append(float(row["latency_s"]))
        if row.get("prompt_tokens") is not None:
            runtime[condition]["prompt_tokens"].append(float(row["prompt_tokens"]))
        runtime[condition]["chunks"].append(float(len(row["contexts"])))

    efficiency_path = os.path.join(args.outdir, "efficiency_metrics.csv")
    with open(efficiency_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "measure", "n", "mean", "median", "q1", "q3"])
        writer.writeheader()
        for condition in CONDITIONS:
            for measure in ("latency_s", "prompt_tokens", "chunks"):
                values = sorted(runtime[condition][measure])
                writer.writerow({"condition": condition, "measure": measure, "n": len(values),
                                 "mean": round(statistics.mean(values), 4),
                                 "median": round(statistics.median(values), 4),
                                 "q1": round(percentile(values, 0.25), 4),
                                 "q3": round(percentile(values, 0.75), 4)})

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


if __name__ == "__main__":
    main()
