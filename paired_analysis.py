"""
Paired statistical analysis (Assignment 3, Part 5).
Input CSV columns: qid, condition, faithfulness, answer_correctness, context_precision, context_recall
(one row per question x condition; per-sample scores exported from the RAGAS evaluation run).
Usage: python paired_analysis.py scores.csv [baseline] [filtered]
"""
import random
import sys
from math import comb
from statistics import mean

import csv
from statistics import NormalDist


def holm_correction(pvalues, alpha=0.05):
    """Return Holm-adjusted rejection decisions and p-values."""
    count = len(pvalues)
    order = sorted(range(count), key=pvalues.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running_max)
    return [value <= alpha for value in adjusted], adjusted


def wilcoxon_pvalue(d):
    """Calculate a two-sided signed-rank p-value without SciPy."""
    nonzero = [(i, value) for i, value in enumerate(d) if value != 0]
    if not nonzero:
        return 1.0

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
    positive_rank_sum = sum(ranks[index] for index, value in nonzero if value > 0)
    expected = n * (n + 1) / 4
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24
    if variance <= 0:
        return 1.0
    z = max(0.0, abs(positive_rank_sum - expected) - 0.5) / variance ** 0.5
    return min(1.0, 2 * (1 - NormalDist().cdf(z)))

def boot_ci(d, n=10000, seed=42):
    rng = random.Random(seed)
    means = [mean(rng.choices(d, k=len(d))) for _ in range(n)]
    means.sort()

    def percentile(p):
        position = (len(means) - 1) * p / 100
        lower = int(position)
        upper = min(lower + 1, len(means) - 1)
        return means[lower] + (means[upper] - means[lower]) * (position - lower)

    return percentile(2.5), percentile(97.5)

def mcnemar_exact_pvalue(table):
    """Calculate a two-sided exact McNemar p-value."""
    discordant_a, discordant_b = table[0][1], table[1][0]
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(discordant_a, discordant_b) + 1))
    return min(1.0, 2 * tail / (2 ** n))

def compare(df, a="baseline", b="filtered", hall_thr=1.0):
    paired = {}
    metrics = ("faithfulness", "answer_correctness", "context_precision", "context_recall")
    for row in df:
        paired.setdefault(row["qid"], {})[row["condition"]] = {
            metric: float(row[metric]) for metric in metrics
        }
    qids = [qid for qid, conditions in paired.items() if a in conditions and b in conditions]
    if not qids:
        raise ValueError(f"No questions have scores for both {a!r} and {b!r}")

    out, pvals = [], {}
    isnan = lambda v: v != v
    n_all = len(qids)
    def valid(metric):
        return [q for q in qids if not isnan(paired[q][a][metric]) and not isnan(paired[q][b][metric])]
    print(f"paired questions: {n_all}")
    # binary hallucination indicator: faithfulness < hall_thr (any unsupported claim)
    qf = valid("faithfulness")
    print(f"faithfulness: {len(qf)}/{n_all} pairs valid ({n_all - len(qf)} dropped: judge returned NaN)")
    ha = [int(paired[qid][a]["faithfulness"] < hall_thr) for qid in qf]
    hb = [int(paired[qid][b]["faithfulness"] < hall_thr) for qid in qf]
    tab = [[sum(x == 0 and y == 0 for x, y in zip(ha, hb)), sum(x == 0 and y == 1 for x, y in zip(ha, hb))],
           [sum(x == 1 and y == 0 for x, y in zip(ha, hb)), sum(x == 1 and y == 1 for x, y in zip(ha, hb))]]
    mc_pvalue = mcnemar_exact_pvalue(tab)
    print(f"Hallucination rate {a}: {mean(ha):.3f}  {b}: {mean(hb):.3f}  McNemar exact p={mc_pvalue:.4f}")
    pvals["hallucination_rate"] = mc_pvalue
    for m in ["answer_correctness", "faithfulness", "context_precision", "context_recall"]:
        qm = valid(m)
        if m != "faithfulness":
            print(f"{m}: {len(qm)}/{n_all} pairs valid")
        x = [paired[qid][a][m] for qid in qm]
        y = [paired[qid][b][m] for qid in qm]
        d = [yb - xa for xa, yb in zip(x, y)]
        p = wilcoxon_pvalue(d)
        lo, hi = boot_ci(d)
        d_mean = mean(d)
        d_sd = (sum((value - d_mean) ** 2 for value in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0
        dz = d_mean / d_sd if d_sd > 0 else float("nan")
        print(f"{m:20s} {a}={mean(x):.3f} {b}={mean(y):.3f} diff={d_mean:+.3f} "
              f"95%CI[{lo:+.3f},{hi:+.3f}] dz={dz:.2f} Wilcoxon p={p:.4f}")
        pvals[m] = p
    prim = ["hallucination_rate", "answer_correctness"]           # primary outcomes -> Holm correction
    rej, adj = holm_correction([pvals[k] for k in prim])
    for k, r_, p_ in zip(prim, rej, adj):
        print(f"PRIMARY {k}: Holm-adjusted p={p_:.4f} significant={r_}")

if __name__ == "__main__":
    with open(sys.argv[1], newline="", encoding="utf-8") as scores_file:
        df = list(csv.DictReader(scores_file))
    compare(df, *(sys.argv[2:4] if len(sys.argv) > 3 else ("baseline", "filtered")))
