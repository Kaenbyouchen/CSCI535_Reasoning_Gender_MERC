"""
build_counterfactual_summary.py — Aggregate the 8 MELD CF eval cells into a
single comparison table.

Inputs (auto-discovered under result/):
    qwen25_omni7b_cf_MELD_CF_BL_audio_*/
    qwen25_omni7b_cf_MELD_CF_BL_text+audio_*/
    qwen25_omni7b_cf_MELD_CF_BL_audio+video_*/
    qwen25_omni7b_cf_MELD_CF_BL_text+audio+video_*/
    qwen25_omni7b_cf_MELD_CF_CF_audio_*/
    qwen25_omni7b_cf_MELD_CF_CF_text+audio_*/
    qwen25_omni7b_cf_MELD_CF_CF_audio+video_*/
    qwen25_omni7b_cf_MELD_CF_CF_text+audio+video_*/

Each must contain:
    metrics.json
    predictions_detailed.csv
    logits.npz                (optional — only used for KL/JSD)

Outputs:
    result/MELD_Counterfactual_Summary.csv     (one row per (modality, condition))
    result/MELD_CF_shift_matrices.json         (BL→CF prediction shift counts per modality)

Usage:
    python analysis/build_counterfactual_summary.py
    python analysis/build_counterfactual_summary.py --result-dir result
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


EMOTIONS = ["Anger", "Disgust", "Sadness", "Joy", "Neutral", "Surprise", "Fear"]
MODALITY_KEYS = ["audio", "text+audio", "audio+video", "text+audio+video"]
CONDITION_TAGS = ["BL", "CF"]


def find_latest_cell(result_dir, condition, modality):
    """Find most-recent run dir matching the (condition, modality) cell."""
    pat = re.compile(
        rf"qwen25_omni7b_cf_MELD_CF_{condition}_{re.escape(modality)}_(\d{{8}}_\d{{6}})$"
    )
    candidates = []
    for d in Path(result_dir).iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m:
            candidates.append((m.group(1), d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def load_predictions(run_dir):
    """Return dict[(dia, utt) -> row]."""
    p = run_dir / "predictions_detailed.csv"
    if not p.exists():
        return {}
    out = {}
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (str(row["dialogue_id"]).strip(),
                   str(row["utterance_id"]).strip())
            out[key] = row
    return out


def load_logits(run_dir):
    """Return (np.ndarray [N,7], list of (dia,utt)) or (None, None)."""
    p = run_dir / "logits.npz"
    if not p.exists():
        return None, None
    try:
        import numpy as np
        z = np.load(p, allow_pickle=False)
        logits = z["logits"]
        dia = list(z["dialogue_ids"])
        utt = list(z["utterance_ids"])
        keys = [(str(d), str(u)) for d, u in zip(dia, utt)]
        return logits, keys
    except Exception as exc:
        print(f"[WARN] cannot load logits from {p}: {exc}")
        return None, None


def softmax(x):
    import numpy as np
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_divergence(p_logits, q_logits):
    """KL(P || Q), both 1-D logits."""
    import numpy as np
    if not (np.isfinite(p_logits).all() and np.isfinite(q_logits).all()):
        return float("nan")
    p = softmax(p_logits)
    q = softmax(q_logits)
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def metrics_to_row(modality, condition, n_samples, m, gender_m, extra=None):
    """One row for the summary CSV."""
    row = {
        "modality": modality,
        "condition": condition,
        "n_samples": n_samples,
        "acc": round(m.get("overall_accuracy", 0.0), 4),
        "wF1": round(m.get("weighted_f1", 0.0), 4),
    }
    for g in ["male", "female"]:
        gm = gender_m.get(g, {})
        row[f"acc_{g}"] = round(gm.get("overall_accuracy", 0.0), 4)
        row[f"wF1_{g}"] = round(gm.get("weighted_f1", 0.0), 4)
    for emo in EMOTIONS:
        row[f"f1_{emo}"] = round(m.get("per_emotion", {}).get(emo, {}).get("f1", 0.0), 4)
    if extra:
        row.update(extra)
    else:
        row["flip_rate"] = ""
        row["flip_rate_male"] = ""
        row["flip_rate_female"] = ""
        row["mean_KL_BL_to_CF"] = ""
    return row


def compute_pair_extras(bl_run_dir, cf_run_dir):
    """Compute flip rate + mean KL over BL↔CF paired samples."""
    bl_pred = load_predictions(bl_run_dir)
    cf_pred = load_predictions(cf_run_dir)
    common = set(bl_pred) & set(cf_pred)
    flips = 0
    flips_by_gender = Counter()
    n_by_gender = Counter()
    for k in common:
        b = bl_pred[k]
        c = cf_pred[k]
        g = (b.get("gender") or "").strip()
        n_by_gender[g] += 1
        if b["pred_emotion"] != c["pred_emotion"]:
            flips += 1
            flips_by_gender[g] += 1

    flip_rate = flips / len(common) if common else float("nan")
    flip_male = (flips_by_gender["male"] / n_by_gender["male"]
                 if n_by_gender["male"] else float("nan"))
    flip_female = (flips_by_gender["female"] / n_by_gender["female"]
                   if n_by_gender["female"] else float("nan"))

    # KL divergence (best-effort, only if both have logits).
    bl_logits, bl_keys = load_logits(bl_run_dir)
    cf_logits, cf_keys = load_logits(cf_run_dir)
    mean_kl = float("nan")
    if bl_logits is not None and cf_logits is not None:
        try:
            import numpy as np
            bl_idx = {k: i for i, k in enumerate(bl_keys)}
            cf_idx = {k: i for i, k in enumerate(cf_keys)}
            kls = []
            for k in common:
                if k in bl_idx and k in cf_idx:
                    val = kl_divergence(bl_logits[bl_idx[k]],
                                         cf_logits[cf_idx[k]])
                    if math.isfinite(val):
                        kls.append(val)
            if kls:
                mean_kl = float(np.mean(kls))
        except Exception as exc:
            print(f"[WARN] KL compute failed: {exc}")

    return {
        "flip_rate": round(flip_rate, 4) if math.isfinite(flip_rate) else "",
        "flip_rate_male": round(flip_male, 4) if math.isfinite(flip_male) else "",
        "flip_rate_female": round(flip_female, 4) if math.isfinite(flip_female) else "",
        "mean_KL_BL_to_CF": round(mean_kl, 4) if math.isfinite(mean_kl) else "",
    }


def build_shift_matrix(bl_run_dir, cf_run_dir):
    """7x7 counts: rows=BL.pred, cols=CF.pred."""
    bl_pred = load_predictions(bl_run_dir)
    cf_pred = load_predictions(cf_run_dir)
    common = set(bl_pred) & set(cf_pred)
    mat = {emo: {emo2: 0 for emo2 in EMOTIONS} for emo in EMOTIONS}
    for k in common:
        b = bl_pred[k]["pred_emotion"]
        c = cf_pred[k]["pred_emotion"]
        if b in mat and c in mat[b]:
            mat[b][c] += 1
    return {"row=BL_pred, col=CF_pred": mat,
             "n_paired_samples": len(common)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="result")
    ap.add_argument("--out-csv", default="result/MELD_Counterfactual_Summary.csv")
    ap.add_argument("--shift-json", default="result/MELD_CF_shift_matrices.json")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    rows = []
    shift_dump = {}
    missing = []

    for modality in MODALITY_KEYS:
        bl_dir = find_latest_cell(result_dir, "BL", modality)
        cf_dir = find_latest_cell(result_dir, "CF", modality)
        if bl_dir is None:
            missing.append(("BL", modality))
        if cf_dir is None:
            missing.append(("CF", modality))

        # Per-cell metrics (always emit a row even if missing, for visibility).
        for cond, run_dir in [("BL", bl_dir), ("CF", cf_dir)]:
            if run_dir is None:
                rows.append({"modality": modality, "condition": cond,
                             "n_samples": "MISSING"})
                continue
            mj = json.load(open(run_dir / "metrics.json", encoding="utf-8"))
            extra = None
            if cond == "CF" and bl_dir is not None:
                extra = compute_pair_extras(bl_dir, run_dir)
            row = metrics_to_row(
                modality, cond, mj.get("n_samples_evaluated", ""),
                mj.get("overall", {}), mj.get("by_gender", {}), extra)
            row["run_dir"] = str(run_dir)
            rows.append(row)

        if bl_dir is not None and cf_dir is not None:
            shift_dump[modality] = build_shift_matrix(bl_dir, cf_dir)

    # Columns
    cols = ["modality", "condition", "n_samples", "acc", "wF1",
            "acc_male", "acc_female", "wF1_male", "wF1_female"]
    for emo in EMOTIONS:
        cols.append(f"f1_{emo}")
    cols += ["flip_rate", "flip_rate_male", "flip_rate_female",
             "mean_KL_BL_to_CF", "run_dir"]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    with open(args.shift_json, "w", encoding="utf-8") as f:
        json.dump(shift_dump, f, indent=2, ensure_ascii=False)

    print(f"\nSummary written to: {out_csv}")
    print(f"Shift matrices written to: {args.shift_json}")
    if missing:
        print("\n[WARN] missing cells:")
        for cond, mod in missing:
            print(f"  - {cond} | {mod}")


if __name__ == "__main__":
    main()
