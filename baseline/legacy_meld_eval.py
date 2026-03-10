import argparse
import csv
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from baseline import bc_LSTM
from gpt52 import (
    EMOTIONS,
    YAML,
    append_summary,
    build_run_dir,
    build_speaker_labeler,
    compute_metrics,
    compute_pred_distribution,
    normalize_emotion,
    normalize_gender,
    read_yaml,
    write_json,
)


def load_label_index(mode):
    data_path = Path(f"./data/pickles/data_{mode.lower()}.p")
    with data_path.open("rb") as f:
        packed = pickle.load(f)
    # packed format in MELD baseline:
    # [revs, W, word_idx_map, vocab, max_sentence_length, label_index]
    label_index = packed[5]
    idx_to_label = {v: k for k, v in label_index.items()}
    return idx_to_label


def normalize_legacy_label(label):
    txt = str(label).strip().lower()
    alias = {
        "ang": "Anger",
        "anger": "Anger",
        "dis": "Disgust",
        "disgust": "Disgust",
        "sad": "Sadness",
        "sadness": "Sadness",
        "joy": "Joy",
        "happiness": "Joy",
        "happy": "Joy",
        "neu": "Neutral",
        "neutral": "Neutral",
        "sur": "Surprise",
        "surprise": "Surprise",
        "fear": "Fear",
        "fea": "Fear",
    }
    if txt in alias:
        return alias[txt]
    return normalize_emotion(str(label))


def modality_to_list(modality):
    m = str(modality).strip().lower()
    if m == "text":
        return ["text"]
    if m == "audio":
        return ["audio"]
    if m == "bimodal":
        return ["text", "audio"]
    raise ValueError("modality must be one of: text, audio, bimodal")


def required_artifacts(modality, classification_mode, weights_template):
    mode = str(classification_mode).strip().lower()
    req = [
        Path(f"./data/pickles/data_{mode}.p"),
        Path(str(weights_template).format(modality=modality)),
    ]
    if modality == "audio":
        req.append(Path(f"./data/pickles/audio_embeddings_feature_selection_{mode}.pkl"))
    elif modality == "bimodal":
        req.extend(
            [
                Path(f"./data/pickles/text_{mode}.pkl"),
                Path(f"./data/pickles/audio_{mode}.pkl"),
            ]
        )
    return req


def assert_required_artifacts_exist(modality, classification_mode, weights_template):
    req = required_artifacts(modality, classification_mode, weights_template)
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        hint = (
            "Missing legacy baseline artifacts.\n"
            "Please place MELD feature/model files under data/pickles and data/models.\n"
            "Missing paths:\n- " + "\n- ".join(missing)
        )
        raise FileNotFoundError(hint)


def build_prediction_map(test_dialogue_ids, test_mask, pred_logits, idx_to_label):
    pred_map = {}
    keys = list(test_dialogue_ids.keys())
    for i, dialogue_id in enumerate(keys):
        utts = test_dialogue_ids[dialogue_id]
        for j, utt in enumerate(utts):
            if test_mask[i, j] != 1:
                continue
            pred_idx = int(np.argmax(pred_logits[i, j]))
            raw_lab = idx_to_label.get(pred_idx, "neutral")
            pred_map[f"{dialogue_id}_{utt}"] = normalize_legacy_label(raw_lab)
    return pred_map


def main():
    parser = argparse.ArgumentParser(description="Evaluate original MELD baseline with unified JSON outputs.")
    parser.add_argument("--config", default="yaml/legacy_baseline_MERC.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--modality",
        choices=["text", "audio", "bimodal"],
        default=None,
        help="Override modality from config.",
    )
    parser.add_argument(
        "--speaker-mode",
        choices=["name", "anon", "none"],
        default=None,
        help="Speaker identity mode: name (raw), anon (P1/P2), or none (Unknown).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate first N rows from test csv (optional smoke test).",
    )
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    model_alias = str(cfg["model"]["alias"]).strip()
    modality = args.modality or str(cfg["eval"].get("modality", "text")).strip().lower()
    if modality not in {"text", "audio", "bimodal"}:
        raise ValueError("eval.modality must be one of: text, audio, bimodal")
    modalities = modality_to_list(modality)
    model_name = str(cfg["model"].get("legacy_model_name", "bcLSTM")).strip()
    dataset = cfg["dataset"]["name"]
    task = cfg["task"]["name"]
    origin_dataset = bool(cfg["dataset"]["is_origin"])

    test_csv = Path(cfg["paths"]["test_csv"])
    result_root = Path(cfg["paths"]["result_dir"])
    result_root.mkdir(parents=True, exist_ok=True)
    run_alias = f"{model_alias}_{modality}"
    run_dir = build_run_dir(result_root, run_alias, dataset, task, origin_dataset)

    cfg_speaker_mode = str(cfg.get("eval", {}).get("speaker_mode", "name")).lower()
    speaker_mode = args.speaker_mode or cfg_speaker_mode
    if speaker_mode not in {"name", "anon", "none"}:
        raise ValueError("speaker_mode must be one of: name, anon, none")
    speaker_label_of = build_speaker_labeler(speaker_mode)

    cfg_max_samples = cfg.get("eval", {}).get("max_samples")
    max_samples = args.max_samples if args.max_samples is not None else cfg_max_samples
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0.")

    # Persist effective runtime config for reproducibility.
    cfg.setdefault("eval", {})
    cfg["eval"]["modality"] = modality
    cfg["eval"]["speaker_mode"] = speaker_mode
    cfg["eval"]["max_samples"] = max_samples
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        YAML.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    weights_tpl = str(
        cfg["model"].get(
            "weights_path_template",
            "./data/models/{modality}_weights_emotion.hdf5",
        )
    )
    classification_mode = str(cfg["model"].get("classification_mode", "Emotion"))
    assert_required_artifacts_exist(modality, classification_mode, weights_tpl)

    # Build original architecture from source code, then load legacy weights.
    legacy_args = SimpleNamespace(classify=classification_mode, modality=modality)
    legacy_model = bc_LSTM(legacy_args)
    legacy_model.load_data()

    weights_path = Path(weights_tpl.format(modality=modality))
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "Please make sure legacy MELD baseline files are placed under data/models."
        )

    if modality == "text":
        model = legacy_model.get_text_model()
    elif modality == "audio":
        model = legacy_model.get_audio_model()
    else:
        model = legacy_model.get_bimodal_model()
    model.load_weights(str(weights_path))
    pred_logits = model.predict(legacy_model.test_x, verbose=0)
    idx_to_label = load_label_index(classification_mode)
    pred_map = build_prediction_map(
        legacy_model.data.test_dialogue_ids,
        legacy_model.test_mask,
        pred_logits,
        idx_to_label,
    )

    records = []
    gold = []
    pred = []
    skipped_no_gender = 0
    missing_pred = 0

    with test_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_samples is not None and idx > max_samples:
                break
            dialogue_id = str(row.get("Dialogue_ID", "")).strip()
            utterance_id = str(row.get("Utterance_ID", "")).strip()
            key = f"{dialogue_id}_{utterance_id}"

            speaker_raw = str(row.get("Speaker", "")).strip()
            speaker = speaker_label_of(dialogue_id, speaker_raw)
            utterance = str(row.get("Utterance", "")).strip()
            gender = normalize_gender(row.get("Gender"))
            if not gender:
                skipped_no_gender += 1
                continue

            if key not in pred_map:
                missing_pred += 1
                continue

            g = normalize_emotion(row.get("Emotion", "Neutral"))
            p = pred_map[key]
            gold.append(g)
            pred.append(p)
            records.append(
                {
                    "index": idx,
                    "speaker": speaker_raw,
                    "speaker_label": speaker,
                    "gender": gender,
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "utterance": utterance,
                    "context_history_count": 0,
                    "gold_emotion": g,
                    "pred_emotion": p,
                    "is_correct": int(g == p),
                    "raw_model_output": f"legacy_label={p}",
                }
            )

    overall_metrics = compute_metrics(gold, pred, EMOTIONS)
    by_gender_idx = defaultdict(list)
    for i, r in enumerate(records):
        by_gender_idx[r["gender"]].append(i)

    gender_metrics = {}
    gender_pred_distributions = {}
    for gender, idxs in by_gender_idx.items():
        g_sub = [gold[i] for i in idxs]
        p_sub = [pred[i] for i in idxs]
        gender_metrics[gender] = compute_metrics(g_sub, p_sub, EMOTIONS)
        gender_pred_distributions[gender] = compute_pred_distribution(p_sub, EMOTIONS, g_sub)

    metrics = {
        "model": model_name,
        "model_alias": run_alias,
        "dataset": dataset,
        "task": task,
        "legacy_modality": modality,
        "speaker_mode": speaker_mode,
        "modalities": modalities,
        "overall": overall_metrics,
        "by_gender": gender_metrics,
        "predicted_emotion_distribution": {
            "overall": compute_pred_distribution(pred, EMOTIONS, gold),
            "by_gender": gender_pred_distributions,
        },
        "skipped_no_gender": skipped_no_gender,
        "missing_prediction_rows": missing_pred,
    }

    pred_csv = run_dir / "predictions_detailed.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    write_json(run_dir / "metrics.json", metrics)
    summary_entry = {
        "run_dir": str(run_dir),
        "model": model_name,
        "model_alias": run_alias,
        "dataset": dataset,
        "task": task,
        "legacy_modality": modality,
        "speaker_mode": speaker_mode,
        "modalities": modalities,
        "overall_accuracy": overall_metrics["overall_accuracy"],
        "overall_weighted_f1": overall_metrics["weighted_f1"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_summary(result_root, summary_entry)
    print(f"Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
