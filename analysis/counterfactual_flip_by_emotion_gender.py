#!/usr/bin/env python3
"""
Per-emotion and per-speaker-gender breakdown for MELD counterfactual (BL vs CF) runs.

Reads the same result/ cells as build_counterfactual_summary.py and writes:

  result/MELD_CF_transition_by_gender.csv
      modality, gender, bl_pred, cf_pred, count
      Full 7×7 transition counts, stratified by original speaker gender.

  result/MELD_CF_fliprate_by_bl_pred_and_gender.csv
      modality, gender, bl_pred, n_paired, n_flip, flip_rate
      Flip rate when conditioning on BL prediction.

  result/MELD_CF_fliprate_by_gold_and_gender.csv
      modality, gender, gold_emotion, n_paired, n_flip, flip_rate
      Flip rate when conditioning on gold label.

  result/MELD_CF_pred_distribution_by_gender.csv
      modality, gender, condition, emotion, count, proportion
      Marginal predicted-emotion distribution (BL vs CF) per gender.

Usage:
  python analysis/counterfactual_flip_by_emotion_gender.py
  python analysis/counterfactual_flip_by_emotion_gender.py --result-dir result
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import build_counterfactual_summary as bcs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="result")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Default: same as --result-dir",
    )
    args = ap.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    transition_rows: list[dict] = []
    flip_bl_rows: list[dict] = []
    flip_gold_rows: list[dict] = []
    dist_rows: list[dict] = []

    for modality in bcs.MODALITY_KEYS:
        bl_dir = bcs.find_latest_cell(result_dir, "BL", modality)
        cf_dir = bcs.find_latest_cell(result_dir, "CF", modality)
        if bl_dir is None or cf_dir is None:
            print(f"[SKIP] {modality}: missing BL or CF run dir")
            continue

        bl_pred = bcs.load_predictions(bl_dir)
        cf_pred = bcs.load_predictions(cf_dir)
        common = sorted(set(bl_pred) & set(cf_pred))

        # --- transition counts & flip by bl_pred & gold (per gender)
        trans_cnt: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        bl_bucket: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
        gold_bucket: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
        dist_bl: dict[str, Counter[str]] = defaultdict(Counter)
        dist_cf: dict[str, Counter[str]] = defaultdict(Counter)

        for k in common:
            br = bl_pred[k]
            cr = cf_pred[k]
            g = (br.get("gender") or "").strip().lower()
            if g not in ("male", "female"):
                g = "unknown"
            bp = (br.get("pred_emotion") or "").strip()
            cp = (cr.get("pred_emotion") or "").strip()
            gold = (br.get("gold_emotion") or "").strip()
            flip = 1 if bp != cp else 0

            if bp in bcs.EMOTIONS and cp in bcs.EMOTIONS:
                trans_cnt[g][(bp, cp)] += 1
            if bp in bcs.EMOTIONS:
                bl_bucket[g][(bp, flip)] += 1
            if gold in bcs.EMOTIONS:
                gold_bucket[g][(gold, flip)] += 1
            if bp in bcs.EMOTIONS:
                dist_bl[g][bp] += 1
            if cp in bcs.EMOTIONS:
                dist_cf[g][cp] += 1

        for g in sorted(trans_cnt.keys()):
            for (bp, cp), c in sorted(trans_cnt[g].items()):
                transition_rows.append({
                    "modality": modality,
                    "gender": g,
                    "bl_pred": bp,
                    "cf_pred": cp,
                    "count": c,
                })

        for g in sorted(bl_bucket.keys()):
            by_pred: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for (emo, flip), c in bl_bucket[g].items():
                by_pred[emo][0] += c
                by_pred[emo][1] += c * flip
            for emo in bcs.EMOTIONS:
                n_tot, n_flip = by_pred.get(emo, [0, 0])
                if n_tot == 0:
                    continue
                flip_rate = n_flip / n_tot
                flip_bl_rows.append({
                    "modality": modality,
                    "gender": g,
                    "bl_pred": emo,
                    "n_paired": n_tot,
                    "n_flip": n_flip,
                    "flip_rate": round(flip_rate, 6),
                })

        for g in sorted(gold_bucket.keys()):
            by_gold: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for (emo, flip), c in gold_bucket[g].items():
                by_gold[emo][0] += c
                by_gold[emo][1] += c * flip
            for emo in bcs.EMOTIONS:
                n_tot, n_flip = by_gold.get(emo, [0, 0])
                if n_tot == 0:
                    continue
                flip_gold_rows.append({
                    "modality": modality,
                    "gender": g,
                    "gold_emotion": emo,
                    "n_paired": n_tot,
                    "n_flip": n_flip,
                    "flip_rate": round(n_flip / n_tot, 6),
                })

        for g in sorted(set(dist_bl.keys()) | set(dist_cf.keys())):
            for cond, dist in (("BL", dist_bl[g]), ("CF", dist_cf[g])):
                tot = sum(dist.values())
                if tot == 0:
                    continue
                for emo in bcs.EMOTIONS:
                    cnt = dist.get(emo, 0)
                    dist_rows.append({
                        "modality": modality,
                        "gender": g,
                        "condition": cond,
                        "emotion": emo,
                        "count": cnt,
                        "proportion": round(cnt / tot, 6) if tot else 0.0,
                    })

    def _write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {path} ({len(rows)} rows)")

    t_path = out_dir / "MELD_CF_transition_by_gender.csv"
    _write(
        t_path,
        ["modality", "gender", "bl_pred", "cf_pred", "count"],
        transition_rows,
    )

    fb_path = out_dir / "MELD_CF_fliprate_by_bl_pred_and_gender.csv"
    _write(
        fb_path,
        ["modality", "gender", "bl_pred", "n_paired", "n_flip", "flip_rate"],
        flip_bl_rows,
    )

    fg_path = out_dir / "MELD_CF_fliprate_by_gold_and_gender.csv"
    _write(
        fg_path,
        ["modality", "gender", "gold_emotion", "n_paired", "n_flip", "flip_rate"],
        flip_gold_rows,
    )

    d_path = out_dir / "MELD_CF_pred_distribution_by_gender.csv"
    _write(
        d_path,
        ["modality", "gender", "condition", "emotion", "count", "proportion"],
        dist_rows,
    )


if __name__ == "__main__":
    main()
