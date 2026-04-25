"""
shift_matrix.py — Pretty-print the BL->CF prediction shift matrix per modality.

Reads result/MELD_CF_shift_matrices.json (produced by build_counterfactual_summary.py)
and prints a human-readable confusion-style table for each modality.

Usage:
    python analysis/shift_matrix.py
    python analysis/shift_matrix.py --shift-json result/MELD_CF_shift_matrices.json
"""

import argparse
import json
from pathlib import Path


EMOTIONS = ["Anger", "Disgust", "Sadness", "Joy", "Neutral", "Surprise", "Fear"]


def print_matrix(title, mat, n_paired):
    print()
    print("=" * 70)
    print(f" {title}    (n_paired_samples={n_paired})")
    print(" rows = baseline pred,  cols = counterfactual pred")
    print("=" * 70)
    header = "BL\\CF      " + "".join(f"{e[:7]:>9}" for e in EMOTIONS) + "    total"
    print(header)
    for src in EMOTIONS:
        row = mat.get(src, {})
        total = sum(row.get(d, 0) for d in EMOTIONS)
        diag = row.get(src, 0)
        line = f"{src:<10}"
        for dst in EMOTIONS:
            v = row.get(dst, 0)
            cell = f"{v:>9d}" if dst != src else f"[{v:>7d}]"
            line += cell
        line += f"    {total:>5d}  (kept={diag/total*100 if total else 0:5.1f}%)"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-json", default="result/MELD_CF_shift_matrices.json")
    args = ap.parse_args()

    p = Path(args.shift_json)
    if not p.exists():
        print(f"Not found: {p}. Run build_counterfactual_summary.py first.")
        return
    data = json.load(open(p, encoding="utf-8"))
    if not data:
        print("No shift matrices in file.")
        return
    for modality, payload in data.items():
        mat = payload.get("row=BL_pred, col=CF_pred", {})
        n_paired = payload.get("n_paired_samples", 0)
        print_matrix(modality, mat, n_paired)
    print()


if __name__ == "__main__":
    main()
