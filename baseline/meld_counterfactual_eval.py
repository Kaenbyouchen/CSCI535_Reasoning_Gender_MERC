"""
meld_counterfactual_eval.py — MELD Counterfactual evaluation for Qwen2.5-Omni-7B.

Runs Qwen2.5-Omni on the MELD test subset that has voice-converted (gender-swapped)
audio available, in **utterance mode**. Supports both original audio and
counterfactual audio across any combination of {text, audio, video}.

Per cell, the script writes:
  - predictions_detailed.csv  (per-sample gold/pred, plus original_gender / target_gender)
  - logits.npz                (per-sample 7-class logits at the emotion-word position; best effort)
  - metrics.json              (overall + by-gender + per-emotion + by-emotion x gender)

Usage:
    python baseline/meld_counterfactual_eval.py \
        --config yaml/qwen25_MELD_counterfactual.yaml \
        --audio-source original \
        --modalities audio
    python baseline/meld_counterfactual_eval.py \
        --config yaml/qwen25_MELD_counterfactual.yaml \
        --audio-source counterfactual \
        --modalities text audio video
"""

import argparse
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
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS = ["Anger", "Disgust", "Sadness", "Joy", "Neutral", "Surprise", "Fear"]
TEXT_OMITTED = "[TRANSCRIPT_OMITTED]"

_EMO_ALIASES = {
    "anger": "Anger", "angry": "Anger",
    "disgust": "Disgust", "disgusted": "Disgust",
    "sadness": "Sadness", "sad": "Sadness",
    "joy": "Joy", "happy": "Joy", "happiness": "Joy",
    "neutral": "Neutral",
    "surprise": "Surprise", "surprised": "Surprise",
    "fear": "Fear", "afraid": "Fear", "fearful": "Fear",
}


def read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return YAML.safe_load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def render_template(template, values):
    if not template:
        return ""
    val_map = {str(k): str(v) for k, v in (values or {}).items()}
    return re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}").sub(
        lambda m: val_map.get(m.group(1), m.group(0)), str(template))


def normalize_gender(text):
    if not text:
        return ""
    t = str(text).strip().lower()
    if t in {"male", "m", "1"}:
        return "male"
    if t in {"female", "f", "2"}:
        return "female"
    return ""


def normalize_emotion(text):
    if not text:
        return "Neutral"
    t = str(text).strip().lower().replace('"', "").replace("'", "")
    if t in _EMO_ALIASES:
        return _EMO_ALIASES[t]
    for k, v in _EMO_ALIASES.items():
        if k in t:
            return v
    return "Neutral"


def parse_pred(raw_text):
    txt = (raw_text or "").strip()
    fence_m = re.search(r"```(?:\w+)?\s*([\s\S]+?)\s*```", txt)
    if fence_m:
        txt = fence_m.group(1).strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            emo = obj.get("emotion", obj.get("label", ""))
            if emo:
                return normalize_emotion(str(emo))
    except Exception:
        pass
    m = re.search(r"(Anger|Disgust|Sadness|Joy|Neutral|Surprise|Fear)", txt, flags=re.IGNORECASE)
    if m:
        return normalize_emotion(m.group(1))
    return normalize_emotion(txt)


def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError("ffmpeg required but not found in PATH.") from exc


def normalize_wav(wav_path, out_dir):
    """Re-encode any wav to 16kHz mono."""
    out = Path(out_dir) / f"{Path(wav_path).stem}_16k.wav"
    if out.exists():
        return out
    cmd = ["ffmpeg", "-y", "-i", str(wav_path),
           "-ac", "1", "-ar", "16000", str(out)]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except subprocess.CalledProcessError:
        return None


def extract_audio_from_video(video_path, out_dir):
    """Extract 16kHz mono wav from .mp4."""
    out = Path(out_dir) / f"{Path(video_path).stem}_16k.wav"
    if out.exists():
        return out
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-ac", "1", "-ar", "16000", str(out)]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out
    except subprocess.CalledProcessError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Clip resolver — find .mp4 inside MELD.Raw/ or its tar.gz
# ─────────────────────────────────────────────────────────────────────────────

class ClipResolver:
    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        self.local_map = {}
        self.tar_index = {}
        self.cache_dir = Path(os.environ.get("MELD_CF_TMP", "/tmp")) / f"meld_cf_{os.getpid()}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_local_mp4()

    def _index_local_mp4(self):
        if not self.raw_dir.exists():
            return
        for p in self.raw_dir.rglob("*.mp4"):
            name = p.name
            if name.startswith("._"):
                continue
            self.local_map[name] = p

    def _build_tar_index(self, split):
        if split in self.tar_index:
            return
        import tarfile
        tar_path = self.raw_dir / f"{split}.tar.gz"
        index = {}
        tar_obj = None
        if tar_path.exists():
            tar_obj = tarfile.open(tar_path, "r:gz")
            for member in tar_obj.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".mp4"):
                    continue
                base = Path(member.name).name
                if base.startswith("._"):
                    continue
                if base not in index:
                    index[base] = member.name
        self.tar_index[split] = (tar_obj, index)

    def resolve(self, split, dialogue_id, utterance_id):
        base = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        if base in self.local_map and self.local_map[base].exists():
            return self.local_map[base]
        self._build_tar_index(split)
        tar_obj, index = self.tar_index.get(split, (None, {}))
        if not tar_obj or base not in index:
            return None
        dst = self.cache_dir / base
        if dst.exists():
            return dst
        member = tar_obj.getmember(index[base])
        fobj = tar_obj.extractfile(member)
        if fobj is None:
            return None
        with dst.open("wb") as out:
            out.write(fobj.read())
        return dst


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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

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
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for g in gold if g == lab)
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1,
                          "accuracy": (tp + tn) / n, "support": support}
        wf1_sum += f1 * support
        total_support += support
    return {"overall_accuracy": overall_acc,
            "weighted_f1": wf1_sum / total_support if total_support else 0.0,
            "per_emotion": per_label, "total_samples": n}


def compute_pred_distribution(pred, labels):
    n = len(pred)
    counts = {lab: 0 for lab in labels}
    for p in pred:
        if p in counts:
            counts[p] += 1
    return {"total_samples": n,
            "emotions": {lab: {"count": c,
                                "ratio_percent": (c * 100.0 / n) if n else 0.0}
                          for lab, c in counts.items()}}


def build_run_dir(result_root, model_alias, modality_tag, audio_source_tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_alias}_MELD_CF_{audio_source_tag}_{modality_tag}_{ts}"
    d = Path(result_root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_summary(result_dir, entry):
    p = Path(result_dir) / "result_summary.json"
    data = []
    if p.exists():
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception:
            data = []
    data.append(entry)
    write_json(p, data)


# ─────────────────────────────────────────────────────────────────────────────
# Logit extraction (best-effort)
# ─────────────────────────────────────────────────────────────────────────────

def build_emotion_first_token_ids(tokenizer):
    """Return {emotion -> first-token-id when emotion is generated after `"`}.

    We tokenize each emotion with a leading `"` so the BPE merges that the model
    would actually see at generation time are reproduced. We then take the
    first token AFTER the opening quote.
    """
    quote_ids = tokenizer('"', add_special_tokens=False).input_ids
    if not quote_ids:
        return {}
    quote_id = quote_ids[0]
    out = {}
    for emo in EMOTIONS:
        ids = tokenizer(f'"{emo}"', add_special_tokens=False).input_ids
        if not ids:
            continue
        if ids[0] == quote_id and len(ids) >= 2:
            out[emo] = int(ids[1])
        else:
            out[emo] = int(ids[0])
    return out


def extract_emotion_logits(scores, new_token_ids, tokenizer, emotion_first_ids):
    """Best-effort: find generation step right after `{"emotion": "` and read
    logits for each emotion's first-token id. Returns dict[emotion -> float] or
    NaNs if not parseable.

    scores: tuple of [vocab] tensors, one per generated token.
    new_token_ids: list[int], generated token ids in order.
    """
    import math
    nan = float("nan")
    if not scores or not new_token_ids:
        return {emo: nan for emo in EMOTIONS}

    # Walk tokens, decode incrementally, find position of `"emotion"` then the
    # next position whose token id is one of the candidate first-token ids.
    target_set = set(emotion_first_ids.values())
    seen_marker = False
    target_pos = None
    for i in range(len(new_token_ids)):
        text_so_far = tokenizer.decode(new_token_ids[: i + 1], skip_special_tokens=True)
        if not seen_marker:
            if '"emotion"' in text_so_far.lower() or "emotion:" in text_so_far.lower():
                seen_marker = True
            continue
        # already seen marker; first token whose id is one of the candidates wins
        if int(new_token_ids[i]) in target_set:
            target_pos = i
            break
    if target_pos is None or target_pos >= len(scores):
        return {emo: nan for emo in EMOTIONS}

    step_logits = scores[target_pos][0]  # [vocab]
    out = {}
    for emo in EMOTIONS:
        tid = emotion_first_ids.get(emo)
        if tid is None:
            out[emo] = nan
            continue
        try:
            out[emo] = float(step_logits[tid].item())
        except Exception:
            out[emo] = nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MELD Counterfactual eval (Qwen2.5-Omni).")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--audio-source", choices=["original", "counterfactual"],
                        default=None, help="Where audio waveforms come from.")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="e.g. --modalities audio video")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-logits", action="store_true",
                        help="Skip logit extraction (saves a tiny bit of compute).")
    parser.add_argument("--speaker-mode", choices=["name", "anon", "none"], default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    # ── Resolve config ────────────────────────────────────────────────────
    model_name = cfg["model"]["hf_model_name"]
    model_alias = cfg["model"]["alias"]
    modalities = [m.lower() for m in (args.modalities or cfg["eval"]["modalities"])]
    audio_source = args.audio_source or str(cfg["eval"].get("audio_source", "original"))
    if audio_source not in {"original", "counterfactual"}:
        raise ValueError("audio_source must be 'original' or 'counterfactual'.")
    speaker_mode = (args.speaker_mode
                    or str(cfg["eval"].get("speaker_mode", "anon")).lower())
    save_logits = (not args.no_logits) and bool(cfg["eval"].get("save_logits", True))
    max_samples = (args.max_samples if args.max_samples is not None
                   else cfg["eval"].get("max_samples"))
    if max_samples is not None:
        max_samples = int(max_samples)

    include_text = "text" in modalities
    include_audio = "audio" in modalities
    include_video = "video" in modalities
    if not (include_text or include_audio or include_video):
        raise ValueError("At least one modality required.")

    if include_audio or include_video:
        ensure_ffmpeg()

    test_csv = Path(cfg["paths"]["test_csv"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    cf_meta_path = Path(cfg["paths"]["cf_metadata_csv"])
    result_root = Path(cfg["paths"]["result_dir"])
    result_root.mkdir(parents=True, exist_ok=True)

    if not cf_meta_path.exists():
        raise FileNotFoundError(f"CF metadata not found: {cf_meta_path}")

    # ── Build run dir ─────────────────────────────────────────────────────
    modality_tag = "+".join(modalities)
    audio_tag = "BL" if audio_source == "original" else "CF"
    run_dir = build_run_dir(result_root, model_alias, modality_tag, audio_tag)
    tmp_media = run_dir / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)

    # ── Load CF metadata, build (dia, utt) -> info map ───────────────────
    cf_map = {}
    cf_skipped = 0
    with cf_meta_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("conversion_status", "")).strip().lower() != "success":
                cf_skipped += 1
                continue
            key = (str(row["dialogue_id"]).strip(), str(row["utterance_id"]).strip())
            cf_audio_rel = str(row.get("converted_audio_path", "")).strip().lstrip("./")
            cf_audio_path = (Path(cf_audio_rel) if cf_audio_rel else None)
            cf_map[key] = {
                "speaker": row.get("speaker", "").strip(),
                "emotion": row.get("emotion", "").strip(),
                "gender": normalize_gender(row.get("gender", "")),
                "target_gender": normalize_gender(row.get("target_gender", "")),
                "cf_audio_path": cf_audio_path,
            }
    print(f"[CF metadata] {len(cf_map)} success rows ({cf_skipped} skipped).")

    # ── Load MELD test csv, filter to CF rows ─────────────────────────────
    rows_kept = []
    with test_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (str(row["Dialogue_ID"]).strip(),
                   str(row["Utterance_ID"]).strip())
            if key not in cf_map:
                continue
            row["_cf"] = cf_map[key]
            rows_kept.append(row)
    if max_samples:
        rows_kept = rows_kept[:max_samples]
    print(f"[Eval set] {len(rows_kept)} utterances (intersect MELD test with CF success set).")

    # ── Save run config ───────────────────────────────────────────────────
    cfg_dump = {
        "model": cfg["model"],
        "task": cfg["task"],
        "paths": {k: str(v) for k, v in cfg["paths"].items()},
        "eval": {**cfg.get("eval", {}),
                 "modalities": modalities,
                 "audio_source": audio_source,
                 "speaker_mode": speaker_mode,
                 "save_logits": save_logits,
                 "max_samples": max_samples,
                 "n_samples_filtered": len(rows_kept)},
    }
    with (run_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        YAML.safe_dump(cfg_dump, f, sort_keys=False, allow_unicode=True)

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt_cfg = cfg.get("prompt", {}).get("utterance", {}) or {}
    label_list = ", ".join(EMOTIONS)
    system_prompt = prompt_cfg.get("system", (
        "You are an expert evaluator for Multimodal Emotion Recognition in Conversation (MERC).\n"
        f"You must classify the target utterance into exactly one emotion from:\n{label_list}.\n"
        "Use all provided modalities if available.\n"
        'Output strictly JSON only, with this schema: {"emotion": "<one of the seven labels>"}'
    ))
    user_tmpl = prompt_cfg.get("user_template", (
        "Target utterance information:\n"
        "- Speaker: {speaker}\n"
        "- Dialogue_ID: {dialogue_id}\n"
        "- Utterance_ID: {utterance_id}\n"
        "- Text: {utterance}\n"
        "Please predict the emotion label for this utterance."
    ))
    audio_prefix = prompt_cfg.get("audio_prefix_template", "Audio of TARGET utterance:")
    video_prefix = prompt_cfg.get("video_prefix_template", "Video of TARGET utterance:")

    # Video sampling — must satisfy qwen-omni-utils VIDEO_MIN_PIXELS (128*28*28).
    # Smaller max_pixels causes smart_resize to raise and all video samples fail.
    _vmin = 128 * 28 * 28
    video_fps = float(cfg.get("eval", {}).get("video_fps", 1.0))
    video_max_pixels = int(cfg.get("eval", {}).get("video_max_pixels", _vmin))
    if video_max_pixels < _vmin:
        print(f"[WARN] video_max_pixels={video_max_pixels} < {_vmin} (qwen VIDEO_MIN_PIXELS); "
              f"clamping to {_vmin}.")
        video_max_pixels = _vmin
    video_max_frames = cfg.get("eval", {}).get("video_max_frames")
    video_max_frames = (int(video_max_frames) if video_max_frames is not None else None)

    # ── Load Qwen ─────────────────────────────────────────────────────────
    import torch
    from transformers import (Qwen2_5OmniForConditionalGeneration,
                              Qwen2_5OmniProcessor)
    from qwen_omni_utils import process_mm_info as _qwen_pmm

    dtype_str = str(cfg["model"].get("torch_dtype", "bf16")).lower()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}.get(dtype_str, "auto")
    use_audio_in_video = bool(cfg["model"].get("use_audio_in_video", False))
    return_audio = bool(cfg["model"].get("return_audio", False))
    max_output_tokens = int(cfg["model"].get("max_output_tokens", 64))
    temperature = float(cfg["model"].get("temperature", 0.0))
    sample_retries = int(cfg.get("eval", {}).get("sample_retries", 2))
    sleep_s = float(cfg["model"].get("retry_sleep_seconds", 2.0))
    log_every_n = int(cfg.get("eval", {}).get("log_every_n", 50))

    model_kwargs = {"torch_dtype": dtype,
                    "device_map": cfg["model"].get("device_map", "auto"),
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

    print(f"Loading Qwen model: {model_name} (dtype={dtype_str}, "
          f"attn={model_kwargs.get('attn_implementation', 'default')}) ...")
    try:
        qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs)
    except (ImportError, ValueError, RuntimeError) as exc:
        msg = str(exc).lower()
        if (model_kwargs.get("attn_implementation") == "flash_attention_2"
                and ("flash" in msg or "import" in msg)):
            print(f"[WARN] flash_attention_2 unavailable ({exc}); retrying with sdpa.")
            kw2 = dict(model_kwargs)
            kw2["attn_implementation"] = "sdpa"
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

    # First-token ids for each emotion (best-effort logit extraction).
    tokenizer = qwen_processor.tokenizer
    emotion_first_ids = build_emotion_first_token_ids(tokenizer)
    print(f"[logits] emotion first-token ids: {emotion_first_ids}")

    # ── Eval loop ─────────────────────────────────────────────────────────
    speaker_label_of = build_speaker_labeler(speaker_mode)

    resolver = ClipResolver(raw_dir)
    records = []
    gold_all, pred_all = [], []
    logits_buffer = []
    skipped_failed = 0
    skipped_audio_missing = 0
    skipped_video_missing = 0

    pbar = (tqdm(total=len(rows_kept), desc=f"MELD-CF [{audio_tag} | {modality_tag}]")
            if tqdm else None)

    for idx, row in enumerate(rows_kept, start=1):
        dialogue_id = str(row["Dialogue_ID"]).strip()
        utterance_id = str(row["Utterance_ID"]).strip()
        speaker_raw = str(row.get("Speaker", "")).strip()
        speaker_label = speaker_label_of(dialogue_id, speaker_raw)
        utterance = str(row.get("Utterance", "")).strip()
        gold = normalize_emotion(row.get("Emotion", ""))
        cf_info = row["_cf"]
        original_gender = cf_info["gender"]
        target_gender = cf_info["target_gender"]

        utt_text_for_prompt = utterance if include_text else TEXT_OMITTED
        user_text = render_template(user_tmpl, {
            "speaker": speaker_label,
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "utterance": utt_text_for_prompt,
        })

        # Build user_content (multimodal).
        user_content = [{"type": "text", "text": user_text}]

        clip = resolver.resolve("test", dialogue_id, utterance_id)

        if include_video:
            if clip is None:
                skipped_video_missing += 1
            else:
                user_content.append({"type": "text", "text": video_prefix})
                vitem = {
                    "type": "video", "video": str(clip),
                    "fps": video_fps,
                    "max_pixels": video_max_pixels,
                }
                if video_max_frames is not None:
                    vitem["max_frames"] = video_max_frames
                user_content.append(vitem)

        if include_audio:
            wav_path = None
            if audio_source == "original":
                if clip is not None:
                    wav_path = extract_audio_from_video(clip, tmp_media)
            else:
                cf_wav = cf_info["cf_audio_path"]
                if cf_wav is not None and Path(cf_wav).exists():
                    wav_path = normalize_wav(cf_wav, tmp_media)
            if wav_path is None:
                skipped_audio_missing += 1
            else:
                user_content.append({"type": "text", "text": audio_prefix})
                user_content.append({"type": "audio", "audio": str(wav_path)})

        conversation = [
            {"role": "system",
             "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        raw_output = ""
        emo_logits = {emo: float("nan") for emo in EMOTIONS}
        success = False
        last_err = ""

        for attempt in range(sample_retries):
            try:
                text_in = qwen_processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False)
                audios, images, videos = _qwen_pmm(
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
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = float(temperature)
                if save_logits:
                    gen_kwargs["output_scores"] = True
                    gen_kwargs["return_dict_in_generate"] = True

                with torch.inference_mode():
                    generated = qwen_model.generate(**inputs, **gen_kwargs)

                if save_logits and hasattr(generated, "sequences"):
                    seq = generated.sequences
                    scores = generated.scores
                else:
                    seq = generated[0] if isinstance(generated, tuple) else generated
                    scores = None

                if isinstance(seq, tuple):
                    seq = seq[0]

                input_len = inputs.input_ids.shape[-1]
                new_ids = seq[0, input_len:].tolist()
                raw_output = qwen_processor.batch_decode(
                    seq[:, input_len:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()

                if save_logits and scores is not None:
                    emo_logits = extract_emotion_logits(
                        scores, new_ids, tokenizer, emotion_first_ids)

                success = True
                break
            except Exception as exc:
                last_err = str(exc)
                if attempt < sample_retries - 1:
                    time.sleep(sleep_s)

        if not success:
            skipped_failed += 1
            print(f"[WARN] idx={idx} dia={dialogue_id} utt={utterance_id}: {last_err}")
            pred = "Neutral"
        else:
            pred = parse_pred(raw_output)

        gold_all.append(gold)
        pred_all.append(pred)
        records.append({
            "index": idx,
            "speaker": speaker_raw,
            "speaker_label": speaker_label,
            "gender": original_gender,
            "target_gender": target_gender,
            "audio_source": audio_source,
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "utterance": utterance,
            "gold_emotion": gold,
            "pred_emotion": pred,
            "is_correct": int(gold == pred),
            "raw_model_output": raw_output,
        })
        logits_buffer.append([emo_logits.get(e, float("nan")) for e in EMOTIONS])

        if pbar:
            pbar.update(1)
        elif idx % log_every_n == 0:
            running_acc = sum(r["is_correct"] for r in records) / len(records)
            print(f"  [{idx}/{len(rows_kept)}] running acc={running_acc:.4f}")

    if pbar:
        pbar.close()

    # ── Metrics ───────────────────────────────────────────────────────────
    overall_m = compute_metrics(gold_all, pred_all, EMOTIONS)
    by_gender_idx = defaultdict(list)
    for i, r in enumerate(records):
        by_gender_idx[r["gender"]].append(i)
    gender_metrics = {}
    gender_pred_dist = {}
    for g, idxs in by_gender_idx.items():
        if not g:
            continue
        gs = [gold_all[i] for i in idxs]
        ps = [pred_all[i] for i in idxs]
        gender_metrics[g] = compute_metrics(gs, ps, EMOTIONS)
        gender_pred_dist[g] = compute_pred_distribution(ps, EMOTIONS)

    metrics = {
        "model": model_name,
        "model_alias": model_alias,
        "dataset": "MELD",
        "task": "MERC_CF",
        "audio_source": audio_source,
        "modalities": modalities,
        "speaker_mode": speaker_mode,
        "use_audio_in_video": use_audio_in_video,
        "n_samples_evaluated": len(records),
        "skipped_failed_after_retries": skipped_failed,
        "missing_modality_stats": {
            "audio_missing": skipped_audio_missing,
            "video_missing": skipped_video_missing,
        },
        "overall": overall_m,
        "by_gender": gender_metrics,
        "predicted_emotion_distribution": {
            "overall": compute_pred_distribution(pred_all, EMOTIONS),
            "by_gender": gender_pred_dist,
        },
    }

    # ── Save artefacts ────────────────────────────────────────────────────
    pred_csv = run_dir / "predictions_detailed.csv"
    if records:
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    write_json(run_dir / "metrics.json", metrics)

    if save_logits and logits_buffer:
        try:
            import numpy as np
            arr = np.array(logits_buffer, dtype=np.float32)  # [N, 7]
            np.savez_compressed(
                run_dir / "logits.npz",
                logits=arr,
                emotions=np.array(EMOTIONS),
                dialogue_ids=np.array([r["dialogue_id"] for r in records]),
                utterance_ids=np.array([r["utterance_id"] for r in records]),
                gold=np.array([r["gold_emotion"] for r in records]),
                pred=np.array([r["pred_emotion"] for r in records]),
                gender=np.array([r["gender"] for r in records]),
                target_gender=np.array([r["target_gender"] for r in records]),
            )
        except Exception as exc:
            print(f"[WARN] could not save logits.npz: {exc}")

    summary_entry = {
        "run_dir": str(run_dir),
        "model": model_name,
        "model_alias": model_alias,
        "dataset": "MELD",
        "task": "MERC_CF",
        "audio_source": audio_source,
        "modalities": modalities,
        "n_samples_evaluated": len(records),
        "overall_accuracy": overall_m["overall_accuracy"],
        "overall_weighted_f1": overall_m["weighted_f1"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_summary(result_root, summary_entry)
    print(f"\nDone. cell={audio_tag}/{modality_tag}  acc={overall_m['overall_accuracy']:.4f}  "
          f"wF1={overall_m['weighted_f1']:.4f}  results→ {run_dir}")


if __name__ == "__main__":
    main()
