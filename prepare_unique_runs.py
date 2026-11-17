#!/usr/bin/env python3
"""Create a de-duplicated runtime log while preserving the original raw log.

The historical test log contains repeated qid-condition records. This utility keeps
the first complete record for each pair and writes a new manifest that documents the
cleanup. It never changes the original file.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = ("baseline", "length_matched", "filtered")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    source_manifest = next((row for row in rows if "condition" not in row), {})
    by_key = {}
    counts = Counter()
    for row in rows:
        if "condition" not in row:
            continue
        key = (str(row["qid"]), row["condition"])
        counts[key] += 1
        by_key.setdefault(key, row)

    by_qid = defaultdict(set)
    for qid, condition in by_key:
        by_qid[qid].add(condition)
    incomplete = {qid: sorted(set(CONDITIONS) - conditions) for qid, conditions in by_qid.items()
                  if conditions != set(CONDITIONS)}
    if incomplete:
        raise ValueError(f"Incomplete condition sets: {list(incomplete.items())[:5]}")

    qids = sorted(by_qid)
    duplicate_rows_removed = sum(value - 1 for value in counts.values())
    manifest = {
        **source_manifest,
        "n": len(qids),
        "unique_qids": len(qids),
        "source_file": str(Path(args.input).name),
        "cleanup_policy": "first complete record retained for each qid-condition pair",
        "duplicate_rows_removed": duplicate_rows_removed,
        "qid_sha256_unique": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
        "warning": (
            "The original log sampled 200 rows but contained repeated question IDs. "
            "This cleaned file supports analysis of the 135 unique observed questions only."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for qid in qids:
            for condition in CONDITIONS:
                handle.write(json.dumps(by_key[(qid, condition)], ensure_ascii=False) + "\n")
    print(f"Wrote {len(qids) * len(CONDITIONS)} records for {len(qids)} unique qids")
    print(f"Removed {duplicate_rows_removed} repeated records; original preserved at {args.input}")


if __name__ == "__main__":
    main()
