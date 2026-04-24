"""
iemocap_eval.py — IEMOCAP Multimodal Emotion Recognition Evaluation
Supports GPT (OpenAI), Gemini, and Qwen2.5-Omni via YAML configuration.

Usage (run from project root: CSCI535_Reasoning_Gender_MERC/):
    python baseline/iemocap_eval.py --config yaml/gpt52_IEMOCAP.yaml
    python baseline/iemocap_eval.py --config yaml/gemini3_IEMOCAP.yaml
    python baseline/iemocap_eval.py --config yaml/qwen25_IEMOCAP.yaml
    python baseline/iemocap_eval.py --config yaml/qwen25_IEMOCAP.yaml --max-samples 20
    python baseline/iemocap_eval.py --config yaml/qwen25_IEMOCAP.yaml --modalities text
"""

import argparse
import base64
import csv
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ─────────────────────────────────────────────────────────────────────────────
# PyYAML safe import (avoid picking up local ./yaml/ directory as module)
# ─────────────────────────────────────────────────────────────────────────────

def _import_pyyaml():
    original = list(sys.path)
    cwd = str(Path.cwd().resolve())
    root = str(Path(__file__).resolve().parents[1])
    sys.path = [p for p in sys.path if p not in {"", cwd, root}]
    try:
        mod = importlib.import_module("yaml")
    except ImportError as exc:
        raise ImportError("Please install pyyaml: pip install pyyaml") from exc
    finally:
        sys.path = original
    return mod

YAML = _import_pyyaml()


# ─────────────────────────────────────────────────────────────────────────────
# Inlined utilities (no gpt52 import to avoid openai dependency)
# ─────────────────────────────────────────────────────────────────────────────

def read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return YAML.safe_load(f)

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def b64_of_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def render_template(template, values):
    if not template:
        return ""
    val_map = {str(k): str(v) for k, v in (values or {}).items()}
    return re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}").sub(
        lambda m: val_map.get(m.group(1), m.group(0)), str(template))

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError("ffmpeg not found.") from exc

def extract_frames(video_path, out_dir, fps):
    """Extract video frames as JPEG files. Used for OpenAI/Gemini."""
    frame_dir = Path(out_dir) / f"{Path(video_path).stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.jpg"))
    if existing:
        return existing
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-vf", f"fps={float(fps):g}", str(frame_dir / "frame_%04d.jpg")]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return sorted(frame_dir.glob("*.jpg"))

def prepare_wav(wav_path, out_dir):
    """Normalize WAV to 16 kHz mono."""
    out = Path(out_dir) / f"{Path(wav_path).stem}_16k.wav"
    if out.exists():
        return out
    cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-ac", "1", "-ar", "16000", str(out)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except subprocess.CalledProcessError:
        return None

def extract_utt_wav(wav_path, start, end, out_dir, utt_id):
    """Cut a single utterance segment from dialogue WAV (16kHz mono)."""
    out = Path(out_dir) / f"{utt_id}_16k.wav"
    if out.exists():
        return out
    cmd = ["ffmpeg", "-y", "-i", str(wav_path),
           "-ss", str(start), "-to", str(end),
           "-ac", "1", "-ar", "16000", str(out)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except subprocess.CalledProcessError:
        return None

def extract_utt_video(avi_path, start, end, out_dir, utt_id):
    """Cut a single utterance segment from dialogue AVI."""
    out = Path(out_dir) / f"{utt_id}.mp4"
    if out.exists():
        return out
    cmd = ["ffmpeg", "-y", "-i", str(avi_path),
           "-ss", str(start), "-to", str(end),
           "-c:v", "libx264", "-preset", "fast", str(out)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except subprocess.CalledProcessError:
        return None

def build_speaker_labeler(mode):
    mode = str(mode or "anon").strip().lower()
    per_dialogue = defaultdict(dict)
    def get_label(dialogue_id, speaker_name):
        raw = str(speaker_name or "").strip()
        if mode == "name":
            return raw or "Unknown"
        if mode == "none":
            return "Unknown"
        mapping = per_dialogue[str(dialogue_id)]
        key = raw or "__unknown__"
        if key not in mapping:
            mapping[key] = f"P{len(mapping) + 1}"
        return mapping[key]
    return get_label

def compute_metrics(gold, pred, labels):
    n = len(gold)
    if n == 0:
        return {"overall_accuracy": 0.0, "weighted_f1": 0.0,
                "per_emotion": {}, "total_samples": 0}
    overall_acc = sum(g == p for g, p in zip(gold, pred)) / n
    per_label = {}
    wf1_sum, total_support = 0.0, 0
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for g in gold if g == lab)
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1,
                          "accuracy": (tp + tn) / n, "support": support}
        wf1_sum += f1 * support
        total_support += support
    return {"overall_accuracy": overall_acc,
            "weighted_f1": wf1_sum / total_support if total_support else 0.0,
            "per_emotion": per_label, "total_samples": n}

def build_run_dir(result_root, model_alias, dataset, task, origin):
    tag = "Origin" if origin else "Processed"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    d   = Path(result_root) / f"{model_alias}_{dataset}_{task}_{tag}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP Constants
# ─────────────────────────────────────────────────────────────────────────────

IEMOCAP_EMO_ABBR = {
    "neu": "Neutral",      "ang": "Anger",        "fru": "Frustration",
    "hap": "Happiness",    "exc": "Excited",       "sad": "Sadness",
    "fea": "Fear",         "dis": "Disgust",       "sur": "Surprise",
    "xxx": "Other",
}

_EMO_ALIASES = {
    "neutral": "Neutral",
    "anger": "Anger",         "angry": "Anger",
    "frustration": "Frustration", "frustrated": "Frustration",
    "happiness": "Happiness", "happy": "Happiness", "joy": "Happiness",
    "excited": "Excited",     "excitement": "Excited",
    "sadness": "Sadness",     "sad": "Sadness",
    "fear": "Fear",           "fearful": "Fear",    "afraid": "Fear",
    "disgust": "Disgust",     "disgusted": "Disgust",
    "surprise": "Surprise",   "surprised": "Surprise",
    "other": "Other",
}

TEXT_OMITTED = "[TRANSCRIPT_OMITTED]"


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_emo_eval(path):
    pat = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[")
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out.append({"utterance_id": m.group(3),
                             "start": float(m.group(1)), "end": float(m.group(2)),
                             "emotion_abbr": m.group(4).lower()})
    return out

def _parse_transcription(path):
    pat = re.compile(r"^(\S+)\s+\[[\d.\-]+\]:\s*(.+)$")
    trans = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                trans[m.group(1)] = m.group(2).strip()
    return trans

def _gender_from_utt_id(utt_id):
    last = utt_id.split("_")[-1]
    if last.startswith("F"):
        return "F", "female"
    if last.startswith("M"):
        return "M", "male"
    return "Unknown", ""

def load_iemocap(data_dir, valid_labels, filter_other=True, sessions=None):
    data_dir  = Path(data_dir)
    session_list = sessions if sessions else list(range(1, 6))
    dialogues = []
    for s in session_list:
        emo_dir   = data_dir / f"Session{s}" / "dialog" / "EmoEvaluation"
        trans_dir = data_dir / f"Session{s}" / "dialog" / "transcriptions"
        wav_dir   = data_dir / f"Session{s}" / "dialog" / "wav"
        avi_dir   = data_dir / f"Session{s}" / "dialog" / "avi" / "DivX"
        if not emo_dir.exists():
            print(f"[WARN] Session{s} EmoEvaluation not found, skipping.")
            continue
        for emo_file in sorted(emo_dir.glob("*.txt")):
            did  = emo_file.stem
            raw  = _parse_emo_eval(emo_file)
            if not raw:
                continue
            trans = {}
            tf = trans_dir / f"{did}.txt"
            if tf.exists():
                trans = _parse_transcription(tf)
            utts = []
            for u in raw:
                emo = IEMOCAP_EMO_ABBR.get(u["emotion_abbr"], "Other")
                if filter_other and emo == "Other":
                    continue
                if emo not in valid_labels:
                    continue
                spk, gender = _gender_from_utt_id(u["utterance_id"])
                utts.append({"utterance_id": u["utterance_id"],
                              "start": u["start"], "end": u["end"],
                              "speaker": spk, "gender": gender,
                              "emotion": emo,
                              "text": trans.get(u["utterance_id"], ""),
                              "dialogue_id": did})
            if not utts:
                continue
            wav_path = wav_dir / f"{did}.wav"
            avi_path = avi_dir / f"{did}.avi"
            dialogues.append({"dialogue_id": did, "session": s,
                               "wav_path": wav_path if wav_path.exists() else None,
                               "avi_path": avi_path if avi_path.exists() else None,
                               "utterances": utts})
    return dialogues


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _norm_emo(text, labels):
    if not text:
        return labels[0]
    t = str(text).strip().lower().replace('"', "").replace("'", "")
    if t in _EMO_ALIASES and _EMO_ALIASES[t] in labels:
        return _EMO_ALIASES[t]
    for k, v in _EMO_ALIASES.items():
        if k in t and v in labels:
            return v
    return labels[0]

def parse_pred_dialogue(raw, utt_indices, labels):
    txt = (raw or "").strip()
    ids = list(utt_indices)
    label_pat = "|".join(re.escape(l) for l in labels)

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", txt)
    if fence_m:
        txt = fence_m.group(1).strip()

    try:
        parsed = json.loads(txt)
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and isinstance(parsed.get("predictions"), list):
        items = parsed["predictions"]
    elif isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        pred_map = {str(k).strip(): _norm_emo(str(v), labels) for k, v in parsed.items()}
        if all(uid in pred_map for uid in ids):
            return [pred_map[uid] for uid in ids]
        items = []
    else:
        items = []

    pred_map, seq = {}, []
    for it in items:
        if isinstance(it, str):
            seq.append(_norm_emo(it, labels)); continue
        if not isinstance(it, dict):
            continue
        emo = it.get("emotion", it.get("label", ""))
        uid = str(it.get("utterance_id", it.get("id", ""))).strip()
        if emo:
            seq.append(_norm_emo(str(emo), labels))
        if uid and emo:
            # Store under both raw key ("U02") and stripped numeric key ("02")
            pred_map[uid] = _norm_emo(str(emo), labels)
            stripped = uid.lstrip("Uu")
            if stripped != uid:
                pred_map[stripped] = pred_map[uid]

    if pred_map and all(uid in pred_map for uid in ids):
        return [pred_map[uid] for uid in ids]
    if len(seq) >= len(ids):
        return seq[:len(ids)]
    hits = re.findall(f"({label_pat})", txt, re.IGNORECASE)
    if len(hits) >= len(ids):
        return [_norm_emo(x, labels) for x in hits[:len(ids)]]
    one = _norm_emo(txt, labels)
    return [one] * len(ids)


def parse_pred_utterance(raw, labels):
    """Parse a single emotion label from model output (JSON or plain text)."""
    txt = (raw or "").strip()
    fence_m = re.search(r"```(?:\w+)?\s*([\s\S]+?)\s*```", txt)
    if fence_m:
        txt = fence_m.group(1).strip()
    # Try JSON first: {"emotion": "Frustration"}
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            emo = obj.get("emotion", obj.get("label", ""))
            if emo:
                return _norm_emo(str(emo), labels)
    except Exception:
        pass
    # Fallback: plain text
    return _norm_emo(txt, labels)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Output
# ─────────────────────────────────────────────────────────────────────────────

def build_modality_str(modalities, fps):
    return "+".join(f"video@{fps:g}fps" if m == "video" else m for m in modalities)

def build_csv_row(model_name, modality_str, gender_metrics, labels):
    row = {"model": model_name, "modalities": modality_str}
    # Interleave: male_acc, female_acc, overall_acc, male_wf1, female_wf1, overall_wf1, ...
    for metric in ["acc", "wf1"]:
        key = "overall_accuracy" if metric == "acc" else "weighted_f1"
        for g in ["male", "female", "overall"]:
            m = gender_metrics.get(g, {})
            row[f"{g}_{metric}"] = round(m.get(key, 0.0), 4)
    for lab in labels:
        for g in ["male", "female", "overall"]:
            m = gender_metrics.get(g, {})
            row[f"{g}_{lab}_f1"] = round(
                m.get("per_emotion", {}).get(lab, {}).get("f1", 0.0), 4)
    return row

def write_iemocap_summary_csv(result_dir, row_data, labels):
    csv_path = Path(result_dir) / "IEMOCAP_Summary.csv"
    header   = ["model", "modalities"]
    for metric in ["acc", "wf1"]:
        header += [f"{g}_{metric}" for g in ["male", "female", "overall"]]
    for lab in labels:
        header += [f"{g}_{lab}_f1" for g in ["male", "female", "overall"]]
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row_data)
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on IEMOCAP for MERC.")
    parser.add_argument("--config",       required=True,
                        help="Path to YAML config (e.g. yaml/qwen25_IEMOCAP.yaml).")
    parser.add_argument("--max-samples",  type=int,   default=None,
                        help="Limit total utterances (smoke test).")
    parser.add_argument("--speaker-mode", choices=["name", "anon", "none"], default=None)
    parser.add_argument("--modalities",   nargs="+",  default=None,
                        help="Override modalities, e.g. --modalities text audio")
    parser.add_argument("--fps",          type=float, default=None,
                        help="Override frame_sample_fps.")
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    model_type        = str(cfg["model"].get("type", "openai")).lower()
    model_alias       = cfg["model"]["alias"]
    temperature       = float(cfg["model"].get("temperature",       0.0))
    max_output_tokens = int  (cfg["model"].get("max_output_tokens", 1024))
    retries           = int  (cfg["model"].get("max_retries",       3))
    sleep_s           = float(cfg["model"].get("retry_sleep_seconds", 2.0))
    sample_retries    = int  (cfg.get("eval", {}).get("sample_retries", 2))

    modalities       = [m.lower() for m in (args.modalities or cfg["eval"]["modalities"])]
    frame_sample_fps = float(args.fps or cfg["eval"].get("frame_sample_fps", 1.0))
    speaker_mode     = args.speaker_mode or str(cfg["eval"].get("speaker_mode", "anon")).lower()
    log_every_n      = int(cfg["eval"].get("log_every_n", 10))
    max_samples      = args.max_samples if args.max_samples is not None \
                       else cfg["eval"].get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
    include_text = "text" in modalities

    labels       = list(cfg["task"]["labels"])
    filter_other = bool(cfg["dataset"].get("filter_other", True))
    data_dir     = Path(cfg["dataset"]["data_dir"])
    result_root  = Path(cfg["paths"]["result_dir"])
    result_root.mkdir(parents=True, exist_ok=True)

    if any(m in {"audio", "video"} for m in modalities):
        ensure_ffmpeg()

    context_window = int(cfg["eval"].get("context_window", 3))

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt_cfg    = cfg.get("prompt", {}).get("utterance", {})
    label_list    = ", ".join(labels)
    system_prompt = prompt_cfg.get("system", (
        f"You are an expert evaluator for Multimodal Emotion Recognition in Conversation (MERC).\n"
        f"You must classify the target utterance into exactly one emotion from:\n"
        f"{label_list}.\n"
        f"Use all provided modalities if available.\n"
        f'Output strictly JSON only, with this schema: {{"emotion": "<one of the labels>"}}'
    ))
    user_tmpl      = prompt_cfg.get("user_template", (
        "Target utterance information:\n"
        "- Speaker: {target_speaker}\n"
        "- Text: {target_text}\n"
        "Please predict the emotion label for this utterance."
    ))
    history_tmpl   = prompt_cfg.get("history_prefix_template", (
        "Dialogue history (previous utterances from same dialogue):\n"
        "{history_block}\n\n"
        "{user_text}"
    ))
    audio_prefix = prompt_cfg.get("audio_prefix_template", "Audio of TARGET utterance:")
    video_prefix = prompt_cfg.get("video_prefix_template", "Video of TARGET utterance:")

    # ── Model init ────────────────────────────────────────────────────────
    model_name     = None
    client         = None
    qwen_model     = None
    qwen_processor = None
    use_audio_in_video = False
    return_audio       = False

    if model_type == "openai":
        from openai import OpenAI
        model_name = cfg["model"]["api_model_name"]
        api_key    = os.environ.get(cfg["model"]["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"Env var {cfg['model']['api_key_env']} not set.")
        client = OpenAI(api_key=api_key)
        print(f"Using OpenAI model: {model_name}")

    elif model_type == "gemini":
        from google import genai
        model_name = cfg["model"]["api_model_name"]
        api_key    = os.environ.get(cfg["model"]["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"Env var {cfg['model']['api_key_env']} not set.")
        client = genai.Client(api_key=api_key)
        print(f"Using Gemini model: {model_name}")

    elif model_type == "qwen":
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info as _qwen_pmm
        globals()["_qwen_pmm"] = _qwen_pmm

        model_name         = cfg["model"]["hf_model_name"]
        use_audio_in_video = bool(cfg["model"].get("use_audio_in_video", False))
        return_audio       = bool(cfg["model"].get("return_audio",       False))

        dtype_str = str(cfg["model"].get("torch_dtype", "bf16")).lower()
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}.get(dtype_str, "auto")
        model_kwargs = {"torch_dtype": dtype,
                        "device_map":  cfg["model"].get("device_map", "auto"),
                        "low_cpu_mem_usage": bool(cfg["model"].get("low_cpu_mem_usage", True))}
        attn = cfg["model"].get("attn_implementation")
        if attn:
            model_kwargs["attn_implementation"] = attn
        cache_dir = cfg["model"].get("cache_dir")
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        hf_token = os.environ.get(cfg["model"].get("hf_token_env", "HF_TOKEN"), None) or None
        if hf_token:
            model_kwargs["token"] = hf_token

        print(f"Loading Qwen model: {model_name} ...")
        try:
            qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs)
        except ImportError as exc:
            if model_kwargs.get("attn_implementation") == "flash_attention_2":
                print("[WARN] flash_attention_2 unavailable, retrying without it.")
                kw2 = {k: v for k, v in model_kwargs.items() if k != "attn_implementation"}
                qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name, **kw2)
            else:
                raise
        proc_kw = {}
        if cache_dir:
            proc_kw["cache_dir"] = cache_dir
        if hf_token:
            proc_kw["token"] = hf_token
        qwen_processor = Qwen2_5OmniProcessor.from_pretrained(model_name, **proc_kw)
        if cfg["model"].get("disable_talker", True):
            qwen_model.disable_talker()
        print("Qwen model loaded.")
    else:
        raise ValueError(f"Unknown model.type '{model_type}'. Choose: openai, gemini, qwen")

    # ── Load IEMOCAP ──────────────────────────────────────────────────────
    print("Loading IEMOCAP data ...")
    test_sessions = cfg["dataset"].get("test_sessions", None)
    dialogues = load_iemocap(data_dir, labels, filter_other=filter_other,
                             sessions=test_sessions)
    if not dialogues:
        raise RuntimeError(f"No dialogues loaded. Check dataset.data_dir: {data_dir}")
    total_utts = sum(len(d["utterances"]) for d in dialogues)
    print(f"Loaded {len(dialogues)} dialogues, {total_utts} utterances.")

    if max_samples is not None:
        kept, count = [], 0
        for d in dialogues:
            if count >= max_samples:
                break
            kept.append(d)
            count += len(d["utterances"])
        dialogues = kept
        print(f"Demo mode: {len(dialogues)} dialogues (~{count} utterances).")

    # ── Run directory ─────────────────────────────────────────────────────
    run_dir   = build_run_dir(result_root, model_alias, "IEMOCAP", "MERC", True)
    tmp_media = run_dir / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)

    speaker_label_of = build_speaker_labeler(speaker_mode)
    records          = []
    gold_all, pred_all            = [], []
    gold_by_gender, pred_by_gender = defaultdict(list), defaultdict(list)

    pbar = tqdm(total=sum(len(d["utterances"]) for d in dialogues),
                desc=f"IEMOCAP [{model_alias}]") if tqdm else None
    skipped_failed = 0
    utt_count = 0

    # ── Evaluation loop (utterance-level) ────────────────────────────────
    for di, dialogue in enumerate(dialogues):
        did      = dialogue["dialogue_id"]
        utts     = dialogue["utterances"]
        wav_path = dialogue["wav_path"]
        avi_path = dialogue["avi_path"]

        for ui, u in enumerate(utts):
            tgt_spk  = speaker_label_of(did, u["speaker"])
            tgt_text = u["text"] if include_text else TEXT_OMITTED

            # Build user_text from target utterance
            user_text = render_template(user_tmpl, {
                "target_speaker": tgt_spk,
                "target_text":    tgt_text,
            })

            # Prepend dialogue history if available (same as MELD)
            ctx_utts = utts[max(0, ui - context_window): ui]
            if ctx_utts:
                history_lines = []
                for cu in ctx_utts:
                    spk  = speaker_label_of(did, cu["speaker"])
                    text = cu["text"] if include_text else TEXT_OMITTED
                    history_lines.append(f"Speaker={spk}: {text}")
                history_block = "\n".join(history_lines)
                user_text = render_template(history_tmpl, {
                    "history_block": history_block,
                    "user_text":     user_text,
                })

            success, raw_output = False, ""

            for attempt in range(sample_retries):
                try:
                    # ── OpenAI ────────────────────────────────────────────
                    if model_type == "openai":
                        use_chat = "audio-preview" in model_name.lower()
                        content  = [{"type": "input_text", "text": user_text}]

                        if "video" in modalities and avi_path:
                            seg_v = extract_utt_video(avi_path, u["start"], u["end"],
                                                      tmp_media, u["utterance_id"])
                            if seg_v:
                                frames = extract_frames(seg_v, tmp_media, frame_sample_fps)
                                if frames:
                                    content.append({"type": "input_text", "text": video_prefix})
                                    for fr in frames:
                                        content.append({"type": "input_image",
                                                        "image_url": f"data:image/jpeg;base64,{b64_of_file(fr)}"})

                        if "audio" in modalities and wav_path:
                            seg_a = extract_utt_wav(wav_path, u["start"], u["end"],
                                                    tmp_media, u["utterance_id"])
                            if seg_a:
                                content.append({"type": "input_text", "text": audio_prefix})
                                content.append({"type": "input_audio",
                                                "audio": {"data": b64_of_file(seg_a), "format": "wav"}})

                        ok = False
                        for _ in range(retries):
                            try:
                                if use_chat:
                                    chat_c = []
                                    for item in content:
                                        t = item["type"]
                                        if t == "input_text":
                                            chat_c.append({"type": "text", "text": item["text"]})
                                        elif t == "input_image":
                                            chat_c.append({"type": "image_url", "image_url": {"url": item["image_url"]}})
                                        elif t == "input_audio":
                                            aud = item["audio"]
                                            chat_c.append({"type": "input_audio",
                                                            "input_audio": {"data": aud["data"], "format": aud["format"]}})
                                    resp = client.chat.completions.create(
                                        model=model_name, temperature=temperature,
                                        max_tokens=max_output_tokens, modalities=["text"],
                                        messages=[{"role": "system", "content": system_prompt},
                                                  {"role": "user",   "content": chat_c}])
                                    raw_output = resp.choices[0].message.content or ""
                                    if isinstance(raw_output, list):
                                        raw_output = "\n".join(
                                            c.get("text", "") if isinstance(c, dict) else str(c)
                                            for c in raw_output)
                                    raw_output = str(raw_output).strip()
                                else:
                                    resp = client.responses.create(
                                        model=model_name, temperature=temperature,
                                        max_output_tokens=max_output_tokens,
                                        input=[{"role": "system",
                                                "content": [{"type": "input_text", "text": system_prompt}]},
                                               {"role": "user", "content": content}])
                                    raw_output = getattr(resp, "output_text", None) or ""
                                    if not raw_output:
                                        try:
                                            data = resp.model_dump()
                                            raw_output = "\n".join(
                                                c.get("text", "")
                                                for out in data.get("output", [])
                                                for c in out.get("content", [])
                                                if c.get("type") == "output_text").strip()
                                        except Exception:
                                            raw_output = str(resp)
                                ok = True; break
                            except Exception as exc:
                                raw_output = f"API_ERROR: {exc}"
                                time.sleep(sleep_s)
                        if not ok:
                            raise RuntimeError(raw_output)

                    # ── Gemini ────────────────────────────────────────────
                    elif model_type == "gemini":
                        from google.genai import types as gt
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE
                        timeout_s = float(cfg["model"].get("request_timeout_seconds", 90))

                        parts = [gt.Part(text=f"{system_prompt}\n\n{user_text}")]
                        if "video" in modalities and avi_path:
                            seg_v = extract_utt_video(avi_path, u["start"], u["end"],
                                                      tmp_media, u["utterance_id"])
                            if seg_v:
                                frames = extract_frames(seg_v, tmp_media, frame_sample_fps)
                                if frames:
                                    parts.append(gt.Part(text=video_prefix))
                                    for fr in frames:
                                        parts.append(gt.Part(inline_data=gt.Blob(
                                            mime_type="image/jpeg", data=Path(fr).read_bytes())))
                        if "audio" in modalities and wav_path:
                            seg_a = extract_utt_wav(wav_path, u["start"], u["end"],
                                                    tmp_media, u["utterance_id"])
                            if seg_a:
                                parts.append(gt.Part(text=audio_prefix))
                                parts.append(gt.Part(inline_data=gt.Blob(
                                    mime_type="audio/wav", data=Path(seg_a).read_bytes())))
                        contents = [gt.Content(role="user", parts=parts)]

                        ok = False
                        for _ in range(retries):
                            try:
                                ex  = ThreadPoolExecutor(max_workers=1)
                                fut = ex.submit(client.models.generate_content,
                                                model=model_name, contents=contents,
                                                config=gt.GenerateContentConfig(
                                                    temperature=temperature,
                                                    max_output_tokens=max_output_tokens))
                                try:
                                    resp = fut.result(timeout=timeout_s)
                                except FTE as e:
                                    fut.cancel()
                                    raise TimeoutError(f"Gemini timeout after {timeout_s}s") from e
                                finally:
                                    ex.shutdown(wait=False)
                                raw_output = getattr(resp, "text", None) or ""
                                if not raw_output:
                                    try:
                                        cands = getattr(resp, "candidates", [])
                                        raw_output = "\n".join(
                                            getattr(p, "text", "")
                                            for p in (cands[0].content.parts if cands else []))
                                    except Exception:
                                        raw_output = str(resp)
                                ok = True; break
                            except Exception as exc:
                                raw_output = f"API_ERROR: {exc}"
                                time.sleep(sleep_s)
                        if not ok:
                            raise RuntimeError(raw_output)

                    # ── Qwen ──────────────────────────────────────────────
                    elif model_type == "qwen":
                        import torch
                        _pmm = globals()["_qwen_pmm"]

                        user_content = [{"type": "text", "text": user_text}]

                        if "video" in modalities and avi_path:
                            seg_v = extract_utt_video(avi_path, u["start"], u["end"],
                                                      tmp_media, u["utterance_id"])
                            if seg_v:
                                user_content.append({"type": "text", "text": video_prefix})
                                user_content.append({"type": "video", "video": str(seg_v),
                                                     "fps": frame_sample_fps})

                        if "audio" in modalities and wav_path:
                            seg_a = extract_utt_wav(wav_path, u["start"], u["end"],
                                                    tmp_media, u["utterance_id"])
                            if seg_a:
                                user_content.append({"type": "text", "text": audio_prefix})
                                user_content.append({"type": "audio", "audio": str(seg_a)})

                        conversation = [
                            {"role": "system",
                             "content": [{"type": "text", "text": system_prompt}]},
                            {"role": "user", "content": user_content},
                        ]

                        text_in = qwen_processor.apply_chat_template(
                            conversation, add_generation_prompt=True, tokenize=False)
                        audios, images, videos = _pmm(
                            conversation, use_audio_in_video=use_audio_in_video)
                        inputs = qwen_processor(
                            text=text_in, audio=audios, images=images, videos=videos,
                            return_tensors="pt", padding=True,
                            use_audio_in_video=use_audio_in_video,
                        ).to(qwen_model.device)

                        gen_kwargs = {"max_new_tokens": max_output_tokens,
                                      "use_audio_in_video": use_audio_in_video,
                                      "return_audio": return_audio,
                                      "do_sample": False}
                        if temperature and float(temperature) > 0:
                            gen_kwargs["do_sample"]   = True
                            gen_kwargs["temperature"] = float(temperature)

                        with torch.inference_mode():
                            generated = qwen_model.generate(**inputs, **gen_kwargs)
                        if isinstance(generated, tuple):
                            generated = generated[0]
                        input_len  = inputs.input_ids.shape[-1]
                        raw_output = qwen_processor.batch_decode(
                            generated[:, input_len:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0].strip()

                    success = True
                    break

                except Exception as exc:
                    last_err = str(exc)
                    if attempt < sample_retries - 1:
                        time.sleep(sleep_s)
                    else:
                        skipped_failed += 1
                        print(f"[WARN] Skip utt={u['utterance_id']} (attempt {attempt+1}): {last_err}")

            pred = parse_pred_utterance(raw_output, labels) if success else labels[0]
            gold = u["emotion"]
            gold_all.append(gold)
            pred_all.append(pred)
            gold_by_gender[u["gender"]].append(gold)
            pred_by_gender[u["gender"]].append(pred)
            records.append({
                "dialogue_id":      did,
                "utterance_id":     u["utterance_id"],
                "speaker":          u["speaker"],
                "gender":           u["gender"],
                "start":            u["start"],
                "end":              u["end"],
                "text":             u["text"],
                "gold_emotion":     gold,
                "pred_emotion":     pred,
                "is_correct":       int(gold == pred),
                "raw_model_output": raw_output,
            })

            utt_count += 1
            if pbar:
                pbar.update(1)
            if utt_count % log_every_n == 0 and records:
                acc = sum(r["is_correct"] for r in records) / len(records)
                print(f"  [utt {utt_count}] running acc={acc:.4f}")

    if pbar:
        pbar.close()

    # ── Metrics ───────────────────────────────────────────────────────────
    overall_m = compute_metrics(gold_all, pred_all, labels)
    male_m    = compute_metrics(gold_by_gender["male"],   pred_by_gender["male"],   labels)
    female_m  = compute_metrics(gold_by_gender["female"], pred_by_gender["female"], labels)
    gender_metrics = {"male": male_m, "female": female_m, "overall": overall_m}

    # ── Save artifacts ────────────────────────────────────────────────────
    write_json(run_dir / "metrics.json", {
        "model": model_name, "model_alias": model_alias, "dataset": "IEMOCAP",
        "modalities": modalities, "frame_sample_fps": frame_sample_fps,
        "speaker_mode": speaker_mode, "labels": labels,
        "skipped_failed": skipped_failed,
        "overall": overall_m,
        "by_gender": {"male": male_m, "female": female_m},
    })

    if records:
        with open(run_dir / "predictions_detailed.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        YAML.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # ── IEMOCAP_Summary.csv ───────────────────────────────────────────────
    modality_str = build_modality_str(modalities, frame_sample_fps)
    csv_row      = build_csv_row(model_name, modality_str, gender_metrics, labels)
    summary_path = write_iemocap_summary_csv(result_root, csv_row, labels)

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Model     : {model_name}")
    print(f"Modalities: {modality_str}")
    print(f"Overall   : Acc={overall_m['overall_accuracy']:.4f}  WF1={overall_m['weighted_f1']:.4f}  (n={overall_m['total_samples']})")
    print(f"Male      : Acc={male_m['overall_accuracy']:.4f}  WF1={male_m['weighted_f1']:.4f}  (n={male_m['total_samples']})")
    print(f"Female    : Acc={female_m['overall_accuracy']:.4f}  WF1={female_m['weighted_f1']:.4f}  (n={female_m['total_samples']})")
    print(f"Results   → {run_dir}")
    print(f"Summary   → {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
