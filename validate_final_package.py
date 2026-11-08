"""Validate Assignment 4 evidence before report finalisation.

Usage:
    python validate_final_package.py --run runs/test.jsonl --scores scores.csv \
        --errors error_analysis_completed.csv

Exit code 0 means all machine-checkable gates passed. Exit code 1 means at
least one required gate failed. Human checks such as Moodle timestamps, team
approval, literature comparison, and reproduction signatures remain manual.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = {"baseline", "length_matched", "filtered"}


def result(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"{status:4s} | {name} | {detail}")
    return passed


def validate_run(path):
    checks = []
    lines = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    manifests = [row for row in lines if "condition" not in row]
    records = [row for row in lines if "condition" in row]
    checks.append(result("one manifest", len(manifests) == 1, f"found {len(manifests)}"))
    if not manifests:
        return checks, set()
    manifest = manifests[0]
    checks.append(result("test split", manifest.get("split") == "test", str(manifest.get("split"))))
    checks.append(result("frozen N", manifest.get("n") == 200, str(manifest.get("n"))))
    checks.append(result("frozen tau", float(manifest.get("tau", math.nan)) == -2.0, str(manifest.get("tau"))))
    checks.append(result("frozen control m", manifest.get("m") == 4, str(manifest.get("m"))))
    pairs = [(str(row.get("qid")), row.get("condition")) for row in records]
    checks.append(result("600 condition records", len(records) == 600, str(len(records))))
    checks.append(result("no duplicate run pairs", len(pairs) == len(set(pairs)), f"unique {len(set(pairs))}"))
    qids = {qid for qid, _ in pairs}
    checks.append(result("200 unique qids", len(qids) == 200, str(len(qids))))
    by_qid = defaultdict(set)
    for qid, condition in pairs:
        by_qid[qid].add(condition)
    incomplete = [qid for qid, conditions in by_qid.items() if conditions != CONDITIONS]
    checks.append(result("three conditions per qid", not incomplete, f"incomplete {len(incomplete)}"))
    expected_hash = hashlib.sha256("\n".join(sorted(qids)).encode()).hexdigest()
    checks.append(result("qid SHA-256", manifest.get("qid_sha256") == expected_hash,
                         f"manifest={manifest.get('qid_sha256')} recomputed={expected_hash}"))
    required_fields = {"question", "reference", "contexts", "chunk_ids", "answer", "latency_s"}
    incomplete_rows = sum(not required_fields.issubset(row) for row in records)
    checks.append(result("required run fields", incomplete_rows == 0, f"incomplete rows {incomplete_rows}"))
    return checks, qids


def validate_scores(path, run_qids):
    checks = []
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    required = {"qid", "condition", "faithfulness", "answer_correctness", "context_precision", "context_recall"}
    checks.append(result("score columns", bool(rows) and required.issubset(rows[0]), str(sorted(rows[0]) if rows else [])))
    pairs = [(row["qid"], row["condition"]) for row in rows]
    checks.append(result("600 score records", len(rows) == 600, str(len(rows))))
    checks.append(result("no duplicate score pairs", len(pairs) == len(set(pairs)), f"unique {len(set(pairs))}"))
    score_qids = {qid for qid, _ in pairs}
    checks.append(result("score qids match run", score_qids == run_qids,
                         f"missing={len(run_qids-score_qids)} extra={len(score_qids-run_qids)}"))
    by_qid = defaultdict(set)
    for qid, condition in pairs:
        by_qid[qid].add(condition)
    incomplete = [qid for qid, conditions in by_qid.items() if conditions != CONDITIONS]
    checks.append(result("scored three conditions per qid", not incomplete, f"incomplete {len(incomplete)}"))
    nan_counts = Counter()
    for row in rows:
        for metric in required - {"qid", "condition"}:
            try:
                if not math.isfinite(float(row[metric])):
                    nan_counts[metric] += 1
            except (ValueError, TypeError):
                nan_counts[metric] += 1
    checks.append(result("NaN values documented", not nan_counts, dict(nan_counts)))
    return checks


def validate_errors(path):
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    required = {"qid", "condition", "error_type", "evidence_excerpt", "possible_cause"}
    complete = [row for row in rows if required.issubset(row) and all(row.get(field, "").strip() for field in required)]
    return [result("30 manually completed error rows", len(complete) >= 30,
                   f"complete {len(complete)} of {len(rows)}")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--errors", required=True)
    args = parser.parse_args()
    checks, qids = validate_run(args.run)
    checks.extend(validate_scores(args.scores, qids))
    checks.extend(validate_errors(args.errors))
    print("\nManual gates still required: Moodle timestamp; team-size approval; verified Assignment 1 sources; repository URL; LICENSE; tag assignment-4; signed Appendix C.")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
