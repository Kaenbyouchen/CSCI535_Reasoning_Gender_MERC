#!/usr/bin/env python3
"""
Aggregate stats from reasoning/cot_audited.jsonl.

  python analysis/summarize_reasoning_audit.py --in result/.../cot_audited.jsonl
  python analysis/summarize_reasoning_audit.py --in result/.../cot_audited.jsonl --out-csv result/Reasoning_Audit_Summary.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="cot_audited.jsonl")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional output CSV")
    args = ap.parse_args()
    p = args.in_path
    if not p.is_absolute():
        p = ROOT / p
    n = 0
    bias_c = 0
    st_sum = 0
    st_dist = Counter()
    by_gender = defaultdict(lambda: [0, 0])
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        a = o.get("audit") or {}
        n += 1
        bg = bool(a.get("contains_gender_bias", False))
        if bg:
            bias_c += 1
        try:
            sc = int(a.get("stereotype_score", 0))
        except (TypeError, ValueError):
            sc = 0
        st_sum += sc
        st_dist[sc] += 1
        g = o.get("gender", "") or "unk"
        by_gender[g][0] += 1
        if bg:
            by_gender[g][1] += 1

    lines = [
        f"file={p}",
        f"n_lines={n}",
        f"contains_gender_bias rate={bias_c / n * 100:.1f}%" if n else "n=0",
        f"mean_stereotype_score={st_sum / n:.3f}" if n else "mean_st=nan",
        f"stereotype_score_hist={dict(st_dist)}",
        "per_speaker_gender (total, bias_count, rate%):",
    ]
    for g, (t, b) in sorted(by_gender.items()):
        r = 100.0 * b / t if t else 0.0
        lines.append(f"  {g}: {t}, {b}, {r:.1f}%")
    out_s = "\n".join(lines)
    print(out_s)
    if args.out_csv and n:
        out = args.out_csv
        if not out.is_absolute():
            out = ROOT / out
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["n", "bias_rate", "mean_stereotype", "n_score0", "n_score1", "n_score2"])
            w.writerow(
                [n, bias_c / n if n else 0, st_sum / n if n else 0,
                 st_dist.get(0, 0), st_dist.get(1, 0), st_dist.get(2, 0)])


if __name__ == "__main__":
    main()
