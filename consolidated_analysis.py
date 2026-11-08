"""Run all paired statistical comparisons with explicit correction families.

Usage:
    python consolidated_analysis.py scores.csv analysis_output

Outputs:
    statistical_comparisons.csv
    hypothesis_decisions.csv
    analysis_manifest.json

The script never replaces NaN values or silently changes the paired sample.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import NormalDist, mean


CONDITIONS = ("baseline", "length_matched", "filtered")
METRICS = ("faithfulness", "answer_correctness", "context_precision", "context_recall")
PAIRS = (
    ("baseline", "filtered"),
    ("length_matched", "filtered"),
    ("baseline", "length_matched"),
)


def holm_adjust(pvalues):
    count = len(pvalues)
    order = sorted(range(count), key=pvalues.__getitem__)
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_mean_ci(differences, repetitions=10000, seed=42):
    rng = random.Random(seed)
    sampled = [mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions)]
    return percentile(sampled, 0.025), percentile(sampled, 0.975)


def wilcoxon(differences):
    nonzero = [(index, value) for index, value in enumerate(differences) if value != 0]
    if not nonzero:
        return {"statistic": 0.0, "w_plus": 0.0, "w_minus": 0.0, "z": 0.0, "p": 1.0, "n_nonzero": 0}
    ordered = sorted(nonzero, key=lambda item: abs(item[1]))
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
    n = len(nonzero)
    w_plus = sum(ranks[index] for index, value in nonzero if value > 0)
    w_minus = sum(ranks[index] for index, value in nonzero if value < 0)
    expected = n * (n + 1) / 4
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24
    if variance <= 0:
        z_value, p_value = 0.0, 1.0
    else:
        z_value = max(0.0, abs(w_plus - expected) - 0.5) / math.sqrt(variance)
        p_value = min(1.0, 2 * (1 - NormalDist().cdf(z_value)))
    return {"statistic": min(w_plus, w_minus), "w_plus": w_plus, "w_minus": w_minus,
            "z": z_value, "p": p_value, "n_nonzero": n}


def mcnemar(table):
    baseline_only_failure = table[0][1]
    comparison_only_failure = table[1][0]
    discordant = baseline_only_failure + comparison_only_failure
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, k) for k in range(min(baseline_only_failure, comparison_only_failure) + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return discordant, p_value


def load_scores(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    required = {"qid", "condition", *METRICS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"scores.csv must contain columns: {sorted(required)}")
    paired = defaultdict(dict)
    duplicates = []
    for row in rows:
        key = (row["qid"], row["condition"])
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"Unexpected condition {row['condition']!r} for qid {row['qid']!r}")
        if row["condition"] in paired[row["qid"]]:
            duplicates.append(key)
        paired[row["qid"]][row["condition"]] = {metric: float(row[metric]) for metric in METRICS}
    if duplicates:
        raise ValueError(f"Duplicate qid-condition rows; examples: {duplicates[:5]}")
    incomplete = {qid: sorted(set(CONDITIONS) - set(items)) for qid, items in paired.items()
                  if set(items) != set(CONDITIONS)}
    if incomplete:
        raise ValueError(f"Incomplete condition sets; examples: {list(incomplete.items())[:5]}")
    return paired, len(rows)


def valid_pairs(paired, condition_a, condition_b, metric):
    qids = []
    for qid, conditions in paired.items():
        a = conditions[condition_a][metric]
        b = conditions[condition_b][metric]
        if math.isfinite(a) and math.isfinite(b):
            qids.append(qid)
    return qids


def continuous_result(paired, condition_a, condition_b, metric):
    qids = valid_pairs(paired, condition_a, condition_b, metric)
    if not qids:
        raise ValueError(f"No valid paired values for {condition_a}, {condition_b}, {metric}")
    values_a = [paired[qid][condition_a][metric] for qid in qids]
    values_b = [paired[qid][condition_b][metric] for qid in qids]
    differences = [b - a for a, b in zip(values_a, values_b)]
    test = wilcoxon(differences)
    ci_low, ci_high = bootstrap_mean_ci(differences)
    difference_mean = mean(differences)
    if len(differences) > 1:
        sd = math.sqrt(sum((value - difference_mean) ** 2 for value in differences) / (len(differences) - 1))
    else:
        sd = 0.0
    effect = difference_mean / sd if sd > 0 else math.nan
    return {
        "condition_a": condition_a, "condition_b": condition_b, "metric": metric,
        "test": "Wilcoxon signed-rank normal approximation with tie correction",
        "n_total_qids": len(paired), "n_valid_pairs": len(qids), "n_missing_pairs": len(paired) - len(qids),
        "mean_a": mean(values_a), "mean_b": mean(values_b), "difference_b_minus_a": difference_mean,
        "ci95_low": ci_low, "ci95_high": ci_high, "effect_size_dz": effect,
        "statistic": test["statistic"], "w_plus": test["w_plus"], "w_minus": test["w_minus"],
        "z_approx": test["z"], "n_nonzero": test["n_nonzero"], "p_raw": test["p"],
    }


def hallucination_result(paired, condition_a, condition_b, threshold):
    qids = valid_pairs(paired, condition_a, condition_b, "faithfulness")
    values_a = [int(paired[qid][condition_a]["faithfulness"] < threshold) for qid in qids]
    values_b = [int(paired[qid][condition_b]["faithfulness"] < threshold) for qid in qids]
    table = [
        [sum(a == 0 and b == 0 for a, b in zip(values_a, values_b)),
         sum(a == 0 and b == 1 for a, b in zip(values_a, values_b))],
        [sum(a == 1 and b == 0 for a, b in zip(values_a, values_b)),
         sum(a == 1 and b == 1 for a, b in zip(values_a, values_b))],
    ]
    discordant, p_value = mcnemar(table)
    differences = [b - a for a, b in zip(values_a, values_b)]
    ci_low, ci_high = bootstrap_mean_ci(differences)
    metric_name = "hallucination_rate" if threshold == 1.0 else f"hallucination_rate_faithfulness_lt_{threshold:g}"
    return {
        "condition_a": condition_a, "condition_b": condition_b, "metric": metric_name,
        "test": "Exact two-sided McNemar",
        "n_total_qids": len(paired), "n_valid_pairs": len(qids), "n_missing_pairs": len(paired) - len(qids),
        "mean_a": mean(values_a), "mean_b": mean(values_b), "difference_b_minus_a": mean(differences),
        "ci95_low": ci_low, "ci95_high": ci_high, "effect_size_dz": math.nan,
        "statistic": discordant, "w_plus": math.nan, "w_minus": math.nan, "z_approx": math.nan,
        "n_nonzero": discordant, "p_raw": p_value,
        "mcnemar_00": table[0][0], "mcnemar_01": table[0][1],
        "mcnemar_10": table[1][0], "mcnemar_11": table[1][1],
    }


def assign_families(results):
    for result in results:
        metric = result["metric"]
        if metric in ("hallucination_rate", "answer_correctness"):
            result["correction_family"] = "confirmatory_primary_all_pairs"
        elif metric == "hallucination_rate_faithfulness_lt_0.8":
            result["correction_family"] = "sensitivity_hallucination_all_pairs"
        else:
            result["correction_family"] = "secondary_metrics_all_pairs"
    families = defaultdict(list)
    for index, result in enumerate(results):
        families[result["correction_family"]].append(index)
    for family, indices in families.items():
        adjusted = holm_adjust([results[index]["p_raw"] for index in indices])
        for index, p_adjusted in zip(indices, adjusted):
            results[index]["p_holm"] = p_adjusted
            results[index]["significant_0_05"] = p_adjusted < 0.05
    return {family: len(indices) for family, indices in families.items()}


def find_result(results, condition_a, condition_b, metric):
    return next(result for result in results
                if result["condition_a"] == condition_a and result["condition_b"] == condition_b
                and result["metric"] == metric)


def hypothesis_rows(results):
    h1 = find_result(results, "baseline", "filtered", "hallucination_rate")
    h2 = find_result(results, "baseline", "filtered", "answer_correctness")
    h3_correctness = find_result(results, "length_matched", "filtered", "answer_correctness")
    h3_faithfulness = find_result(results, "length_matched", "filtered", "faithfulness")
    h4 = find_result(results, "baseline", "filtered", "context_recall")
    h1_supported = h1["difference_b_minus_a"] <= -0.05 and h1["p_holm"] < 0.05
    h2_supported = (h2["difference_b_minus_a"] > 0 and h2["ci95_low"] > 0 and h2["p_holm"] < 0.05)
    h3_supported = (h3_correctness["difference_b_minus_a"] > 0 and h3_correctness["p_holm"] < 0.05
                    and h3_faithfulness["difference_b_minus_a"] > 0 and h3_faithfulness["p_holm"] < 0.05)
    return [
        {"hypothesis": "H1", "decision": "Supported" if h1_supported else "Not supported",
         "rule": "Filtered hallucination rate decreases by at least 0.05 and Holm-adjusted McNemar p < 0.05",
         "evidence": f"difference={h1['difference_b_minus_a']:.6f}; p_holm={h1['p_holm']:.6g}"},
        {"hypothesis": "H2", "decision": "Supported" if h2_supported else "Not supported",
         "rule": "Correctness difference > 0, CI excludes 0, and Holm-adjusted Wilcoxon p < 0.05",
         "evidence": f"difference={h2['difference_b_minus_a']:.6f}; CI=[{h2['ci95_low']:.6f}, {h2['ci95_high']:.6f}]; p_holm={h2['p_holm']:.6g}"},
        {"hypothesis": "H3", "decision": "Supported" if h3_supported else "Not supported",
         "rule": "Filtered exceeds Top-4 control in correctness and faithfulness with adjusted p < 0.05",
         "evidence": f"correctness_diff={h3_correctness['difference_b_minus_a']:.6f}; correctness_p={h3_correctness['p_holm']:.6g}; faithfulness_diff={h3_faithfulness['difference_b_minus_a']:.6f}; faithfulness_p={h3_faithfulness['p_holm']:.6g}"},
        {"hypothesis": "H4", "decision": "Exploratory only",
         "rule": "No non-inferiority margin was frozen, so non-significance cannot prove evidence preservation",
         "evidence": f"context_recall_diff={h4['difference_b_minus_a']:.6f}; CI=[{h4['ci95_low']:.6f}, {h4['ci95_high']:.6f}]; p_holm={h4['p_holm']:.6g}"},
    ]


def clean(value):
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.10g}"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores")
    parser.add_argument("outdir")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paired, input_rows = load_scores(args.scores)

    results = []
    for condition_a, condition_b in PAIRS:
        results.append(hallucination_result(paired, condition_a, condition_b, 1.0))
        results.append(hallucination_result(paired, condition_a, condition_b, 0.8))
        for metric in METRICS:
            results.append(continuous_result(paired, condition_a, condition_b, metric))
    family_sizes = assign_families(results)
    decisions = hypothesis_rows(results)

    comparison_path = outdir / "statistical_comparisons.csv"
    fieldnames = sorted({key for result in results for key in result})
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: clean(result.get(key, "")) for key in fieldnames})

    decision_path = outdir / "hypothesis_decisions.csv"
    with decision_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hypothesis", "decision", "rule", "evidence"])
        writer.writeheader()
        writer.writerows(decisions)

    manifest = {
        "input_scores": str(Path(args.scores).resolve()),
        "input_rows": input_rows,
        "unique_qids": len(paired),
        "conditions": list(CONDITIONS),
        "pairs": [list(pair) for pair in PAIRS],
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 42,
        "holm_family_sizes": family_sizes,
        "notes": [
            "Differences are condition_b minus condition_a.",
            "Wilcoxon p-values use a normal approximation with tie correction and continuity correction.",
            "H4 remains exploratory because the frozen protocol does not specify a non-inferiority margin.",
        ],
    }
    manifest_path = outdir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(comparison_path)
    print(decision_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
