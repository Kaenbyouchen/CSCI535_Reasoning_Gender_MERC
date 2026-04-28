#!/usr/bin/env python3
"""
Bar charts: one figure per modality — flip_rate for each BL-predicted emotion,
grouped by speaker gender (male / female).

Input:  result/MELD_CF_fliprate_by_bl_pred_and_gender.csv
Output: result/MELD_CF_plots/fliprate_<modality_sanitized>.png

  python analysis/plot_cf_fliprate_bars.py
  python analysis/plot_cf_fliprate_bars.py --csv result/MELD_CF_fliprate_by_bl_pred_and_gender.csv --out-dir result/MELD_CF_plots
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

EMOTIONS = ["Anger", "Disgust", "Sadness", "Joy", "Neutral", "Surprise", "Fear"]
MODALITY_ORDER = ["audio", "text+audio", "audio+video", "text+audio+video"]


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "result" / "MELD_CF_fliprate_by_bl_pred_and_gender.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "result" / "MELD_CF_plots",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else _ROOT / args.csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib\n" + str(e)
        ) from e

    # modality -> gender -> bl_pred -> {flip_rate, n_paired}
    data: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict),
    )
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mod = (row.get("modality") or "").strip()
            gen = (row.get("gender") or "").strip().lower()
            bp = (row.get("bl_pred") or "").strip()
            if not mod or gen not in ("male", "female") or bp not in EMOTIONS:
                continue
            data[mod][gen][bp] = {
                "flip_rate": float(row["flip_rate"]),
                "n_paired": int(row["n_paired"]),
            }

    for mod in MODALITY_ORDER:
        if mod not in data:
            continue
        genders = ["female", "male"]
        x = np.arange(len(EMOTIONS))
        width = 0.36
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

        for gi, g in enumerate(genders):
            heights = []
            for emo in EMOTIONS:
                cell = data[mod].get(g, {}).get(emo)
                heights.append(cell["flip_rate"] if cell else 0.0)
            offset = (gi - 0.5) * width
            bars = ax.bar(
                x + offset,
                heights,
                width,
                label=g,
                color="#4C72B0" if g == "female" else "#DD8452",
            )
            # annotate n_paired on top (small)
            for bar, emo in zip(bars, EMOTIONS):
                cell = data[mod].get(g, {}).get(emo)
                if not cell:
                    continue
                h = bar.get_height()
                ax.annotate(
                    f"n={cell['n_paired']}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        ax.set_ylabel("Flip rate (BL pred ≠ CF pred)")
        ax.set_xlabel("BL-predicted emotion")
        ax.set_xticks(x)
        ax.set_xticklabels(EMOTIONS, rotation=25, ha="right")
        ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.08))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(title="Speaker gender", loc="upper right")
        ax.set_title(
            f"MELD counterfactual — flip rate by BL prediction\nmodality: {mod}",
        )

        out_path = out_dir / f"fliprate_{_sanitize(mod)}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {out_path}")

    print(f"Done. Plots under {out_dir}")


if __name__ == "__main__":
    main()
