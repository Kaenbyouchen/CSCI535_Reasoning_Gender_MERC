import argparse
import base64
import csv
import importlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError("Please install openai sdk: pip install openai") from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


EMOTIONS = ["Anger", "Disgust", "Sadness", "Joy", "Neutral", "Surprise", "Fear"]
EMO_SET_LOWER = {e.lower(): e for e in EMOTIONS}
TEXT_OMITTED_TOKEN = "[TRANSCRIPT_OMITTED]"
DEFAULT_SYSTEM_PROMPT = """You are an expert evaluator for Multimodal Emotion Recognition in Conversation (MERC).
You must classify the target utterance into exactly one emotion from:
Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear.
Use all provided modalities (text/audio/video) if available.
Output strictly JSON only, with this schema:
{"emotion": "<one of the seven labels>", "reason": "<short reason>"}"""
DEFAULT_USER_TEMPLATE = """Target utterance information:
- Speaker: {speaker}
- Dialogue_ID: {dialogue_id}
- Utterance_ID: {utterance_id}
- Text: {utterance}
Please predict the emotion label for this utterance."""
DEFAULT_DIALOGUE_SYSTEM_PROMPT = """You are an expert evaluator for Multimodal Emotion Recognition in Conversation (MERC).
Given a full dialogue with multiple utterances, predict one emotion label for EACH utterance.
Valid labels: Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear.
Output strictly JSON only with schema:
{"predictions":[{"utterance_id":"<id>","emotion":"<one label>","reason":"<short reason>"}]}"""
DEFAULT_DIALOGUE_USER_TEMPLATE = """Dialogue_ID: {dialogue_id}
Utterance count: {utterance_count}
Dialogue utterances in chronological order:
{dialogue_block}

Return strictly JSON with schema:
{"predictions":[{"utterance_id":"<id>","emotion":"<one of seven labels>","reason":"<short reason>"}]}"""
DEFAULT_HISTORY_PREFIX_TEMPLATE = """Dialogue history (previous utterances from same dialogue):
{history_block}

{user_text}"""
DEFAULT_DIALOGUE_VIDEO_PREFIX_TEMPLATE = "Video frames for U{utterance_id}:"
DEFAULT_DIALOGUE_AUDIO_PREFIX_TEMPLATE = "Audio for U{utterance_id}:"

# 各种 gender 写法统一成 male/female
def normalize_gender(text):
    if not text:
        return ""
    t = str(text).strip().lower()
    if t in {"male", "m", "1", "男", "男生"}:
        return "male"
    if t in {"female", "f", "2", "女", "女生"}:
        return "female"
    return ""

# 安全导入 PyYAML，避免被项目目录下同名 yaml 目录干扰
def import_pyyaml():
    """
    Avoid importing local ./yaml directory as python module.
    """
    original = list(sys.path)
    cwd = str(Path.cwd().resolve())
    script_root = str(Path(__file__).resolve().parents[1])
    pruned = [p for p in sys.path if p not in {"", cwd, script_root}]
    sys.path = pruned
    try:
        yaml_mod = importlib.import_module("yaml")
    except ImportError as exc:
        raise ImportError("Please install pyyaml: pip install pyyaml") from exc
    finally:
        sys.path = original
    if not hasattr(yaml_mod, "safe_load"):
        raise RuntimeError(
            "Imported module 'yaml' does not provide safe_load. "
            "Please install PyYAML and avoid naming conflicts."
        )
    return yaml_mod


YAML = import_pyyaml()

# 各种 emotion 写法统一成 Anger/Disgust/Sadness/Joy/Neutral/Surprise/Fear
def normalize_emotion(text):
    if not text:
        return "Neutral"
    t = text.strip().lower()
    t = t.replace('"', "").replace("'", "")
    alias = {
        "anger": "Anger",
        "angry": "Anger",
        "disgust": "Disgust",
        "disgusted": "Disgust",
        "sad": "Sadness",
        "sadness": "Sadness",
        "joy": "Joy",
        "happy": "Joy",
        "happiness": "Joy",
        "neutral": "Neutral",
        "surprise": "Surprise",
        "surprised": "Surprise",
        "fear": "Fear",
        "afraid": "Fear",
    }
    if t in alias:
        return alias[t]
    for key, val in alias.items():
        if key in t:
            return val
    return "Neutral"

# 读取 yaml 文件
def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return YAML.safe_load(f)

# 写入 json 文件
def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# 获取 prompt 配置
def get_prompt_bundle(cfg):
    prompt_cfg = cfg.get("prompt", {}) or {}
    utterance_cfg = prompt_cfg.get("utterance", {}) or {}
    dialogue_cfg = prompt_cfg.get("dialogue", {}) or {}
    return {
        # Prefer nested prompt blocks; fall back to legacy flat keys.
        "system": utterance_cfg.get("system", prompt_cfg.get("system", DEFAULT_SYSTEM_PROMPT)),
        "user_template": utterance_cfg.get(
            "user_template", prompt_cfg.get("user_template", DEFAULT_USER_TEMPLATE)
        ),
        "history_prefix_template": utterance_cfg.get(
            "history_prefix_template",
            prompt_cfg.get("history_prefix_template", DEFAULT_HISTORY_PREFIX_TEMPLATE),
        ),
        "dialogue_system": dialogue_cfg.get(
            "system", prompt_cfg.get("dialogue_system", DEFAULT_DIALOGUE_SYSTEM_PROMPT)
        ),
        "dialogue_user_template": dialogue_cfg.get(
            "user_template",
            prompt_cfg.get("dialogue_user_template", DEFAULT_DIALOGUE_USER_TEMPLATE),
        ),
        "dialogue_video_prefix_template": dialogue_cfg.get(
            "video_prefix_template",
            prompt_cfg.get("dialogue_video_prefix_template", DEFAULT_DIALOGUE_VIDEO_PREFIX_TEMPLATE),
        ),
        "dialogue_audio_prefix_template": dialogue_cfg.get(
            "audio_prefix_template",
            prompt_cfg.get("dialogue_audio_prefix_template", DEFAULT_DIALOGUE_AUDIO_PREFIX_TEMPLATE),
        ),
    }

# 渲染模板
def render_template(template, values):
    if template is None:
        return ""
    text = str(template)
    val_map = {str(k): str(v) for k, v in (values or {}).items()}
    pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    return pattern.sub(lambda m: val_map.get(m.group(1), m.group(0)), text)

# 确保 ffmpeg 可用
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError("ffmpeg is required for audio/video processing but not found.") from exc

# 视频剪辑解析器
class ClipResolver:
    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(tempfile.mkdtemp(prefix="meld_gpt52_"))
        self.local_map = {}
        self.tar_index = {}
        self._index_local_mp4()

    # 索引本地 mp4 文件
    def _index_local_mp4(self):
        for p in self.raw_dir.rglob("*.mp4"):
            name = p.name
            if name.startswith("._"):
                continue
            self.local_map[name] = p

    # 索引 tar 包里的 mp4 文件
    def _build_tar_index(self, split):
        if split in self.tar_index:
            return
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

    # 解析视频文件
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

# 提取音频文件
def extract_audio_wav(video_path, out_dir):
    out_path = Path(out_dir) / f"{Path(video_path).stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    except subprocess.CalledProcessError:
        return None

# 提取视频帧
def extract_frames(video_path, out_dir, frame_sample_fps):
    frame_dir = Path(out_dir) / f"{Path(video_path).stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frame_dir / "frame_%03d.jpg")
    fps_val = float(frame_sample_fps)
    if fps_val <= 0:
        raise ValueError("frame_sample_fps must be > 0.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps_val:g}",
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    return sorted(frame_dir.glob("*.jpg"))

# 将文件编码为 base64
def b64_of_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 将响应转换为文本
def response_to_text(resp):
    text = getattr(resp, "output_text", None)
    if text:
        return text
    try:
        data = resp.model_dump()
    except Exception:
        return ""
    outputs = data.get("output", [])
    chunks = []
    for out in outputs:
        for c in out.get("content", []):
            if c.get("type") == "output_text":
                chunks.append(c.get("text", ""))
    return "\n".join(chunks).strip()

# 将聊天完成响应转换为文本
def chat_completion_to_text(resp):
    try:
        msg = resp.choices[0].message
    except Exception:
        return ""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                chunks.append(item.get("text", ""))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    return str(content).strip()


# 解析预测结果
def parse_prediction(raw_text):
    txt = (raw_text or "").strip()
    try:
        maybe = json.loads(txt)
        if isinstance(maybe, dict) and "emotion" in maybe:
            return normalize_emotion(str(maybe["emotion"]))
    except Exception:
        pass
    m = re.search(r"(Anger|Disgust|Sadness|Joy|Neutral|Surprise|Fear)", txt, flags=re.IGNORECASE)
    if m:
        return normalize_emotion(m.group(1))
    return normalize_emotion(txt)

# 构建历史块
def build_history_block(history_rows):
    if not history_rows:
        return "None"
    lines = []
    for item in history_rows:
        lines.append(
            f"- [U{item['utterance_id']}] Speaker={item['speaker']}: {item['utterance']}"
        )
    return "\n".join(lines)

# 构建说话人标签器
def build_speaker_labeler(speaker_mode):
    mode = str(speaker_mode or "name").strip().lower()
    if mode not in {"name", "anon", "none"}:
        raise ValueError("speaker_mode must be one of: name, anon, none")
    per_dialogue = defaultdict(dict)

    def get_label(dialogue_id, speaker_name):
        raw = str(speaker_name or "").strip()
        if mode == "name":
            return raw if raw else "Unknown"
        if mode == "none":
            return "Unknown"
        did = str(dialogue_id or "").strip()
        key = raw if raw else "__unknown__"
        mapping = per_dialogue[did]
        if key not in mapping:
            mapping[key] = f"P{len(mapping) + 1}"
        return mapping[key]

    return get_label

# 解析对话预测结果
def parse_dialogue_predictions(raw_text, dialogue_rows):
    txt = (raw_text or "").strip()
    ids = [str(r.get("Utterance_ID", "")).strip() for r in dialogue_rows]
    try:
        parsed = json.loads(txt)
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and isinstance(parsed.get("predictions"), list):
        items = parsed["predictions"]
    elif isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        # Fallback: {"0":"Joy","1":"Neutral",...}
        if all(isinstance(k, str) for k in parsed.keys()):
            pred_map = {str(k).strip(): normalize_emotion(str(v)) for k, v in parsed.items()}
            out = []
            for uid in ids:
                if uid not in pred_map:
                    raise ValueError(f"Missing utterance_id={uid} in dialogue prediction output.")
                out.append(pred_map[uid])
            return out
        items = []
    else:
        items = []

    pred_map = {}
    seq_preds = []
    for it in items:
        if isinstance(it, str):
            seq_preds.append(normalize_emotion(it))
            continue
        if not isinstance(it, dict):
            continue
        emo = it.get("emotion", it.get("label", it.get("pred_emotion", "")))
        if emo:
            seq_preds.append(normalize_emotion(str(emo)))
        uid = str(it.get("utterance_id", it.get("id", ""))).strip()
        if uid and emo:
            pred_map[uid] = normalize_emotion(str(emo))

    if pred_map and all(uid in pred_map for uid in ids):
        return [pred_map[uid] for uid in ids]

    if len(seq_preds) >= len(ids):
        return seq_preds[: len(ids)]

    emo_hits = re.findall(r"(Anger|Disgust|Sadness|Joy|Neutral|Surprise|Fear)", txt, flags=re.IGNORECASE)
    if len(emo_hits) >= len(ids):
        return [normalize_emotion(x) for x in emo_hits[: len(ids)]]

    # Last-resort fallback: parse one label and broadcast to keep pipeline robust.
    one = parse_prediction(txt)
    return [one for _ in ids]


def compute_metrics(gold, pred, labels):
    n = len(gold)
    overall_acc = sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0
    per_label = {}
    weighted_f1_sum = 0.0
    total_support = 0
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        tn = n - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = (tp + tn) / n if n else 0.0
        support = sum(1 for g in gold if g == lab)
        per_label[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
            "support": support,
        }
        weighted_f1_sum += f1 * support
        total_support += support
    weighted_f1 = weighted_f1_sum / total_support if total_support else 0.0
    return {
        "overall_accuracy": overall_acc,
        "weighted_f1": weighted_f1,
        "per_emotion": per_label,
        "total_samples": n,
    }


def compute_pred_distribution(pred, labels, gold=None):
    total = len(pred)
    counts = {lab: 0 for lab in labels}
    for p in pred:
        if p in counts:
            counts[p] += 1
    gold_counts = None
    if gold is not None:
        gold_counts = {lab: 0 for lab in labels}
        for g in gold:
            if g in gold_counts:
                gold_counts[g] += 1
    emotions = {}
    if total == 0:
        for lab in labels:
            item = {"count": 0, "ratio_percent": 0.0}
            if gold_counts is not None:
                gt = gold_counts[lab]
                item["gt_count"] = gt
                item["pred_over_gt_fraction"] = f"0/{gt}"
                item["pred_over_gt_ratio"] = (0.0 if gt > 0 else None)
            emotions[lab] = item
        return {"total_samples": 0, "emotions": emotions}

    running = 0.0
    for i, lab in enumerate(labels):
        c = counts[lab]
        if i < len(labels) - 1:
            ratio = (c * 100.0) / total
            running += ratio
        else:
            # Ensure percentages sum to exactly 100 for readability.
            ratio = 100.0 - running
        item = {"count": c, "ratio_percent": ratio}
        if gold_counts is not None:
            gt = gold_counts[lab]
            item["gt_count"] = gt
            item["pred_over_gt_fraction"] = f"{c}/{gt}"
            item["pred_over_gt_ratio"] = ((c / gt) if gt > 0 else None)
        emotions[lab] = item
    return {"total_samples": total, "emotions": emotions}


def build_run_dir(base_result_dir, model_name, dataset, task, origin):
    tag = "Origin" if origin else "Processed"
    base = f"{model_name}_{dataset}_{task}_{tag}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_result_dir) / f"{base}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_summary(result_dir, entry):
    summary_path = Path(result_dir) / "result_summary.json"
    data = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    data.append(entry)
    write_json(summary_path, data)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.2 on MELD test set for MERC.")
    parser.add_argument("--config", default="yaml/gpt52_MERC.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate first N utterances from test csv (demo/smoke test).",
    )
    parser.add_argument(
        "--context-window-max",
        type=int,
        default=None,
        help="Use up to K previous utterances from same dialogue as text context. 0 disables context.",
    )
    parser.add_argument(
        "--inference-unit",
        choices=["utterance", "dialogue"],
        default=None,
        help="Inference granularity: utterance (default) or dialogue.",
    )
    parser.add_argument(
        "--speaker-mode",
        choices=["name", "anon", "none"],
        default=None,
        help="Speaker identity mode: name (raw), anon (P1/P2), or none (Unknown).",
    )
    args = parser.parse_args()

    # 读取 yaml 配置
    cfg = read_yaml(args.config)
    # 获取评估模式
    modalities = [m.lower() for m in cfg["eval"]["modalities"]]
    include_text_input = "text" in modalities
    model_name = cfg["model"]["api_model_name"]
    # 获取模型别名
    model_alias = cfg["model"]["alias"]
    dataset = cfg["dataset"]["name"]
    # 获取任务名称
    task = cfg["task"]["name"]
    # 获取原始数据集
    origin_dataset = bool(cfg["dataset"]["is_origin"])
    # 确保 ffmpeg 可用
    # 如果包含音频或视频，则确保 ffmpeg 可用
    if any(m in {"audio", "video"} for m in modalities):
        ensure_ffmpeg()

    # 读取测试 csv 文件
    test_csv = Path(cfg["paths"]["test_csv"])
    # 读取原始数据集目录
    raw_dir = Path(cfg["paths"]["raw_dir"])
    # 读取结果目录
    result_root = Path(cfg["paths"]["result_dir"])
    # 创建结果目录
    result_root.mkdir(parents=True, exist_ok=True)

    # 构建运行目录
    run_dir = build_run_dir(result_root, model_alias, dataset, task, origin_dataset)
    # 创建 OpenAI 客户端

    client = OpenAI(api_key=os.environ.get(cfg["model"]["api_key_env"], ""))
    # 如果 API 密钥环境变量不存在，则抛出异常
    if not os.environ.get(cfg["model"]["api_key_env"]):
        raise RuntimeError(f"Environment variable {cfg['model']['api_key_env']} is missing.")

    # 创建视频剪辑解析器
    resolver = ClipResolver(raw_dir)
    # 创建临时媒体目录
    tmp_media = run_dir / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)

    # 创建记录列表
    records = []
    # 创建黄金标签列表
    gold = []
    # 创建预测标签列表
    pred = []

    # 获取最大重试次数
    retries = int(cfg["model"].get("max_retries", 3))
    # 获取重试睡眠时间
    sleep_s = float(cfg["model"].get("retry_sleep_seconds", 2.0))
    # 获取温度
    temperature = float(cfg["model"].get("temperature", 0.0))
    # 获取最大输出 token 数
    max_output_tokens = int(cfg["model"].get("max_output_tokens", 64))
    # 获取样本重试次数
    sample_retries = int(cfg.get("eval", {}).get("sample_retries", 2))
    # 获取文本仅 fallback 模型
    text_only_fallback_model = cfg["model"].get("text_only_fallback_model", "gpt-4o")
    # 获取 prompt 配置
    prompt_bundle = get_prompt_bundle(cfg)
    # 获取系统提示
    system_prompt = prompt_bundle["system"]
    # 获取用户提示
    user_tmpl = prompt_bundle["user_template"]
    # 获取对话系统提示
    dialogue_system_prompt = prompt_bundle["dialogue_system"]
    # 获取对话用户提示
    dialogue_user_tmpl = prompt_bundle["dialogue_user_template"]
    # 获取历史前缀提示
    history_prefix_tmpl = prompt_bundle["history_prefix_template"]
    # 获取对话视频前缀提示
    dialogue_video_prefix_tmpl = prompt_bundle["dialogue_video_prefix_template"]
    # 获取对话音频前缀提示
    dialogue_audio_prefix_tmpl = prompt_bundle["dialogue_audio_prefix_template"]

    # 获取总行数
    def get_total_rows(csv_file):
        with csv_file.open(newline="", encoding="utf-8") as fcount:
            # header line not counted as sample
            return max(sum(1 for _ in fcount) - 1, 0)

    # 获取最大样本数
    cfg_max_samples = cfg.get("eval", {}).get("max_samples")
    # 获取最大样本数
    max_samples = args.max_samples if args.max_samples is not None else cfg_max_samples
    # 如果最大样本数不为空，则转换为整数
    if max_samples is not None:
        max_samples = int(max_samples)
        # 如果最大样本数小于等于 0，则抛出异常
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0.")
        # 打印 demo 模式启用信息
        print(f"Demo mode enabled: evaluating first {max_samples} samples.")

    # 获取上下文窗口最大值
    cfg_context_window_max = cfg.get("eval", {}).get("context_window_max", 0)
    # 获取上下文窗口最大值
    context_window_max = (
        args.context_window_max if args.context_window_max is not None else cfg_context_window_max
    )
    # 转换为整数
    context_window_max = int(context_window_max or 0)
    # 如果上下文窗口最大值小于 0，则抛出异常
    if context_window_max < 0:
        raise ValueError("context_window_max must be >= 0.")
    # 获取视频采样率
    frame_sample_fps = float(cfg.get("eval", {}).get("frame_sample_fps", 1.0))
    # 如果视频采样率小于等于 0，则抛出异常
    if frame_sample_fps <= 0:
        raise ValueError("frame_sample_fps must be > 0.")
    inference_unit = args.inference_unit or str(cfg.get("eval", {}).get("inference_unit", "utterance"))
    if inference_unit not in {"utterance", "dialogue"}:
        raise ValueError("inference_unit must be 'utterance' or 'dialogue'.")
    cfg_speaker_mode = str(cfg.get("eval", {}).get("speaker_mode", "name")).lower()
    speaker_mode = args.speaker_mode or cfg_speaker_mode
    if speaker_mode not in {"name", "anon", "none"}:
        raise ValueError("speaker_mode must be one of: name, anon, none")
    speaker_label_of = build_speaker_labeler(speaker_mode)

    # Persist effective runtime overrides in run config.
    cfg.setdefault("eval", {})
    cfg["eval"]["max_samples"] = max_samples
    cfg["eval"]["context_window_max"] = context_window_max
    cfg["eval"]["frame_sample_fps"] = frame_sample_fps
    cfg["eval"]["inference_unit"] = inference_unit
    cfg["eval"]["speaker_mode"] = speaker_mode

    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        YAML.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    total_rows = get_total_rows(test_csv)
    target_total = min(total_rows, max_samples) if max_samples is not None else total_rows
    pbar = tqdm(total=target_total, desc="Evaluating MELD") if tqdm is not None else None

    use_chat_audio_path = "audio-preview" in model_name.lower()
    dialogue_histories = defaultdict(list)

    if inference_unit == "dialogue":
        with test_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for idx, row in enumerate(reader, start=1):
                if max_samples is not None and idx > max_samples:
                    break
                row["_index"] = idx
                rows.append(row)

        grouped = defaultdict(list)
        skipped_no_gender = 0
        for row in rows:
            gender = normalize_gender(row.get("Gender"))
            if not gender:
                skipped_no_gender += 1
                continue
            grouped[str(row.get("Dialogue_ID", "")).strip()].append(row)
        dialogues = list(grouped.values())
        pbar_total = sum(len(x) for x in dialogues)
        if pbar is not None and pbar.total != pbar_total:
            pbar.total = pbar_total
            pbar.refresh()

        skipped_failed = 0
        skipped_video_for_missing = 0
        skipped_audio_for_missing = 0
        audio_input_enabled = "audio" in modalities
        audio_disabled_reason = ""
        warned_text_only_fallback = False

        for drows in dialogues:
            success = False
            last_error = ""
            dialogue_id = str(drows[0].get("Dialogue_ID", "")).strip()
            for attempt in range(sample_retries):
                try:
                    block_lines = []
                    for r in drows:
                        speaker_label = speaker_label_of(
                            dialogue_id,
                            str(r.get("Speaker", "")).strip(),
                        )
                        utt_text = (
                            str(r.get("Utterance", "")).strip()
                            if include_text_input
                            else TEXT_OMITTED_TOKEN
                        )
                        block_lines.append(
                            f"- [U{r['Utterance_ID']}] Speaker={speaker_label}: {utt_text}"
                        )
                    dialogue_block = "\n".join(block_lines)
                    user_text = render_template(
                        dialogue_user_tmpl,
                        {
                            "dialogue_id": dialogue_id,
                            "utterance_count": len(drows),
                            "dialogue_block": dialogue_block,
                        },
                    )
                    content = [{"type": "input_text", "text": user_text}]
                    audio_attached_by_uid = {}
                    video_frames_by_uid = {}
                    for r in drows:
                        uid = str(r.get("Utterance_ID", "")).strip()
                        audio_attached_by_uid[uid] = False
                        video_frames_by_uid[uid] = 0

                    for r in drows:
                        uid = str(r.get("Utterance_ID", "")).strip()
                        clip = resolver.resolve("test", r["Dialogue_ID"], r["Utterance_ID"])
                        if "video" in modalities:
                            if clip is None:
                                skipped_video_for_missing += 1
                            else:
                                frames = extract_frames(clip, tmp_media, frame_sample_fps)
                                if not frames:
                                    skipped_video_for_missing += 1
                                else:
                                    video_frames_by_uid[uid] = len(frames)
                                    content.append(
                                        {
                                            "type": "input_text",
                                            "text": render_template(
                                                dialogue_video_prefix_tmpl,
                                                {"utterance_id": r["Utterance_ID"]},
                                            ),
                                        }
                                    )
                                    for fr in frames:
                                        content.append(
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/jpeg;base64,{b64_of_file(fr)}",
                                            }
                                        )
                        if audio_input_enabled and "audio" in modalities:
                            if clip is None:
                                skipped_audio_for_missing += 1
                            else:
                                wav = extract_audio_wav(clip, tmp_media)
                                if wav is None:
                                    skipped_audio_for_missing += 1
                                else:
                                    audio_attached_by_uid[uid] = True
                                    content.append(
                                        {
                                            "type": "input_text",
                                            "text": render_template(
                                                dialogue_audio_prefix_tmpl,
                                                {"utterance_id": r["Utterance_ID"]},
                                            ),
                                        }
                                    )
                                    content.append(
                                        {
                                            "type": "input_audio",
                                            "audio": {"data": b64_of_file(wav), "format": "wav"},
                                        }
                                    )

                    raw_output = ""
                    ok = False
                    for _ in range(retries):
                        try:
                            has_audio_input = any(
                                isinstance(x, dict) and x.get("type") == "input_audio" for x in content
                            )
                            call_model_name = model_name
                            if use_chat_audio_path and not has_audio_input:
                                if text_only_fallback_model:
                                    call_model_name = text_only_fallback_model
                                    if not warned_text_only_fallback:
                                        print(
                                            "[WARN] audio-preview model received text-only request; "
                                            f"fallback to '{text_only_fallback_model}' for text-only samples."
                                        )
                                        warned_text_only_fallback = True
                                else:
                                    raise RuntimeError(
                                        "audio-preview model requires audio input. "
                                        "Set eval.modalities to include audio, or configure "
                                        "model.text_only_fallback_model."
                                    )

                            if use_chat_audio_path and has_audio_input:
                                chat_user_content = []
                                for item in content:
                                    if item.get("type") == "input_text":
                                        chat_user_content.append({"type": "text", "text": item.get("text", "")})
                                    elif item.get("type") == "input_image":
                                        chat_user_content.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": item.get("image_url", "")},
                                            }
                                        )
                                    elif item.get("type") == "input_audio":
                                        aud = item.get("audio", {})
                                        chat_user_content.append(
                                            {
                                                "type": "input_audio",
                                                "input_audio": {
                                                    "data": aud.get("data", ""),
                                                    "format": aud.get("format", "wav"),
                                                },
                                            }
                                        )
                                chat_resp = client.chat.completions.create(
                                    model=call_model_name,
                                    messages=[
                                        {"role": "system", "content": dialogue_system_prompt},
                                        {"role": "user", "content": chat_user_content},
                                    ],
                                    temperature=temperature,
                                    max_tokens=max_output_tokens,
                                    modalities=["text"],
                                )
                                raw_output = chat_completion_to_text(chat_resp)
                            else:
                                resp = client.responses.create(
                                    model=call_model_name,
                                    input=[
                                        {
                                            "role": "system",
                                            "content": [
                                                {"type": "input_text", "text": dialogue_system_prompt}
                                            ],
                                        },
                                        {"role": "user", "content": content},
                                    ],
                                    temperature=temperature,
                                    max_output_tokens=max_output_tokens,
                                )
                                raw_output = response_to_text(resp)
                            ok = True
                            break
                        except Exception as exc:
                            raw_output = f"API_ERROR: {exc}"
                            time.sleep(sleep_s)
                    if not ok:
                        raise RuntimeError(raw_output)

                    preds = parse_dialogue_predictions(raw_output, drows)
                    for i, r in enumerate(drows):
                        g = normalize_emotion(r["Emotion"])
                        p = preds[i]
                        gold.append(g)
                        pred.append(p)
                        records.append(
                            {
                                "index": r["_index"],
                                "speaker": str(r.get("Speaker", "")).strip(),
                                "speaker_label": speaker_label_of(
                                    dialogue_id,
                                    str(r.get("Speaker", "")).strip(),
                                ),
                                "gender": normalize_gender(r.get("Gender")),
                                "dialogue_id": str(r.get("Dialogue_ID", "")).strip(),
                                "utterance_id": str(r.get("Utterance_ID", "")).strip(),
                                "utterance": str(r.get("Utterance", "")).strip(),
                                "context_history_count": len(drows) - 1,
                                "gold_emotion": g,
                                "pred_emotion": p,
                                "is_correct": int(g == p),
                                "resolved_system_prompt": dialogue_system_prompt,
                                "resolved_user_prompt": user_text,
                                "audio_attached": bool(
                                    audio_attached_by_uid.get(
                                        str(r.get("Utterance_ID", "")).strip(), False
                                    )
                                ),
                                "video_attached": int(
                                    video_frames_by_uid.get(
                                        str(r.get("Utterance_ID", "")).strip(), 0
                                    )
                                    > 0
                                ),
                                "video_frames_count": int(
                                    video_frames_by_uid.get(
                                        str(r.get("Utterance_ID", "")).strip(), 0
                                    )
                                ),
                                "raw_model_output": raw_output,
                            }
                        )
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if "Invalid value: 'input_audio'" in last_error and audio_input_enabled:
                        audio_input_enabled = False
                        audio_disabled_reason = (
                            "Model/API does not support input_audio content type. "
                            "Fallback to non-audio modalities."
                        )
                        print("[WARN] input_audio unsupported by API. Disable audio and retrying...")
                    if attempt < sample_retries - 1:
                        time.sleep(sleep_s)

            if not success:
                skipped_failed += len(drows)
                print(
                    f"[WARN] Skip dialogue={dialogue_id}, utt_count={len(drows)}, reason={last_error}"
                )
            if pbar is not None:
                pbar.update(len(drows))

        if pbar is not None:
            pbar.close()

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
            "model_alias": model_alias,
            "dataset": dataset,
            "task": task,
            "inference_unit": inference_unit,
            "speaker_mode": speaker_mode,
            "modalities": modalities,
            "context_window_max": context_window_max,
            "frame_sample_fps": frame_sample_fps,
            "effective_modalities": [m for m in modalities if not (m == "audio" and not audio_input_enabled)],
            "audio_input_enabled": audio_input_enabled,
            "audio_disabled_reason": audio_disabled_reason,
            "overall": overall_metrics,
            "by_gender": gender_metrics,
            "predicted_emotion_distribution": {
                "overall": compute_pred_distribution(pred, EMOTIONS, gold),
                "by_gender": gender_pred_distributions,
            },
            "skipped_no_gender": skipped_no_gender,
            "skipped_failed_after_retries": skipped_failed,
            "missing_modality_stats": {
                "video_missing_or_unreadable": skipped_video_for_missing,
                "audio_missing_or_unreadable": skipped_audio_for_missing,
            },
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
            "model_alias": model_alias,
            "dataset": dataset,
            "task": task,
            "inference_unit": inference_unit,
            "speaker_mode": speaker_mode,
            "modalities": modalities,
            "context_window_max": context_window_max,
            "frame_sample_fps": frame_sample_fps,
            "overall_accuracy": overall_metrics["overall_accuracy"],
            "overall_weighted_f1": overall_metrics["weighted_f1"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        append_summary(result_root, summary_entry)
        print(f"Done. Results saved to: {run_dir}")
        return

    with test_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        skipped_no_gender = 0
        skipped_failed = 0
        skipped_video_for_missing = 0
        skipped_audio_for_missing = 0
        audio_input_enabled = "audio" in modalities
        audio_disabled_reason = ""
        warned_text_only_fallback = False
        for idx, row in enumerate(reader, start=1):
            if max_samples is not None and idx > max_samples:
                break
            dialogue_id = str(row.get("Dialogue_ID", "")).strip()
            utterance_id = str(row.get("Utterance_ID", "")).strip()
            speaker_raw = str(row.get("Speaker", "")).strip()
            speaker = speaker_label_of(dialogue_id, speaker_raw)
            utterance = str(row.get("Utterance", "")).strip()
            utterance_for_prompt = utterance if include_text_input else TEXT_OMITTED_TOKEN
            history_rows = dialogue_histories[dialogue_id]
            if context_window_max > 0:
                history_slice = history_rows[-context_window_max:]
            else:
                history_slice = []
            history_block = build_history_block(history_slice)

            gender = normalize_gender(row.get("Gender"))
            if not gender:
                skipped_no_gender += 1
                history_rows.append(
                    {
                        "utterance_id": utterance_id,
                        "speaker": speaker,
                        "utterance": utterance_for_prompt,
                    }
                )
                if pbar is not None:
                    pbar.update(1)
                continue

            success = False
            last_error = ""
            for attempt in range(sample_retries):
                try:
                    split = "test"
                    clip = resolver.resolve(split, row["Dialogue_ID"], row["Utterance_ID"])

                    user_text = render_template(
                        user_tmpl,
                        {
                            "utterance": utterance_for_prompt,
                            "speaker": speaker,
                            "dialogue_id": dialogue_id,
                            "utterance_id": utterance_id,
                            "history_block": history_block,
                            "history_utterances": history_block,
                            "history_count": len(history_slice),
                            "context_window_max": context_window_max,
                        },
                    )
                    if context_window_max > 0 and "{history_block}" not in user_tmpl:
                        user_text = render_template(
                            history_prefix_tmpl,
                            {
                                "history_block": history_block,
                                "user_text": user_text,
                            },
                        )
                    content = [{"type": "input_text", "text": user_text}]
                    sample_video_frames_count = 0
                    sample_audio_attached = False

                    if "video" in modalities:
                        if clip is None:
                            skipped_video_for_missing += 1
                        else:
                            frames = extract_frames(clip, tmp_media, frame_sample_fps)
                            if not frames:
                                skipped_video_for_missing += 1
                            else:
                                sample_video_frames_count = len(frames)
                                for fr in frames:
                                    content.append(
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:image/jpeg;base64,{b64_of_file(fr)}",
                                        }
                                    )

                    if audio_input_enabled and "audio" in modalities:
                        if clip is None:
                            skipped_audio_for_missing += 1
                        else:
                            wav = extract_audio_wav(clip, tmp_media)
                            if wav is None:
                                skipped_audio_for_missing += 1
                            else:
                                sample_audio_attached = True
                                content.append(
                                    {
                                        "type": "input_audio",
                                        "audio": {"data": b64_of_file(wav), "format": "wav"},
                                    }
                                )

                    messages = [
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": content},
                    ]

                    raw_output = ""
                    ok = False
                    for _ in range(retries):
                        try:
                            has_audio_input = any(
                                isinstance(x, dict) and x.get("type") == "input_audio" for x in content
                            )

                            call_model_name = model_name
                            if use_chat_audio_path and not has_audio_input:
                                if text_only_fallback_model:
                                    call_model_name = text_only_fallback_model
                                    if not warned_text_only_fallback:
                                        print(
                                            "[WARN] audio-preview model received text-only request; "
                                            f"fallback to '{text_only_fallback_model}' for text-only samples."
                                        )
                                        warned_text_only_fallback = True
                                else:
                                    raise RuntimeError(
                                        "audio-preview model requires audio input. "
                                        "Set eval.modalities to include audio, or configure "
                                        "model.text_only_fallback_model."
                                    )

                            if use_chat_audio_path and has_audio_input:
                                chat_user_content = []
                                for item in content:
                                    if item.get("type") == "input_text":
                                        chat_user_content.append(
                                            {"type": "text", "text": item.get("text", "")}
                                        )
                                    elif item.get("type") == "input_image":
                                        chat_user_content.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": item.get("image_url", "")},
                                            }
                                        )
                                    elif item.get("type") == "input_audio":
                                        aud = item.get("audio", {})
                                        chat_user_content.append(
                                            {
                                                "type": "input_audio",
                                                "input_audio": {
                                                    "data": aud.get("data", ""),
                                                    "format": aud.get("format", "wav"),
                                                },
                                            }
                                        )

                                chat_messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": chat_user_content},
                                ]
                                chat_resp = client.chat.completions.create(
                                    model=call_model_name,
                                    messages=chat_messages,
                                    temperature=temperature,
                                    max_tokens=max_output_tokens,
                                    modalities=["text"],
                                )
                                raw_output = chat_completion_to_text(chat_resp)
                            else:
                                resp = client.responses.create(
                                    model=call_model_name,
                                    input=messages,
                                    temperature=temperature,
                                    max_output_tokens=max_output_tokens,
                                )
                                raw_output = response_to_text(resp)
                            ok = True
                            break
                        except Exception as exc:
                            raw_output = f"API_ERROR: {exc}"
                            time.sleep(sleep_s)

                    if not ok:
                        raise RuntimeError(raw_output)

                    p = parse_prediction(raw_output)
                    g = normalize_emotion(row["Emotion"])
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
                            "context_history_count": len(history_slice),
                            "gold_emotion": g,
                            "pred_emotion": p,
                            "is_correct": int(g == p),
                            "resolved_system_prompt": system_prompt,
                            "resolved_user_prompt": user_text,
                            "audio_attached": bool(sample_audio_attached),
                            "video_attached": int(sample_video_frames_count > 0),
                            "video_frames_count": int(sample_video_frames_count),
                            "raw_model_output": raw_output,
                        }
                    )
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if "Invalid value: 'input_audio'" in last_error and audio_input_enabled:
                        audio_input_enabled = False
                        audio_disabled_reason = (
                            "Model/API does not support input_audio content type. "
                            "Fallback to non-audio modalities."
                        )
                        print("[WARN] input_audio unsupported by API. Disable audio and retrying...")
                    if attempt < sample_retries - 1:
                        time.sleep(sleep_s)

            if not success:
                skipped_failed += 1
                print(
                    f"[WARN] Skip sample idx={idx}, dia={row['Dialogue_ID']}, utt={row['Utterance_ID']}, reason={last_error}"
                )

            history_rows.append(
                {
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "utterance": utterance_for_prompt,
                }
            )

            if pbar is not None:
                pbar.update(1)
            elif idx % int(cfg["eval"].get("log_every_n", 50)) == 0:
                print(f"Processed {idx} samples...")

    if pbar is not None:
        pbar.close()

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
        "model_alias": model_alias,
        "dataset": dataset,
        "task": task,
        "inference_unit": inference_unit,
        "speaker_mode": speaker_mode,
        "modalities": modalities,
        "context_window_max": context_window_max,
        "frame_sample_fps": frame_sample_fps,
        "effective_modalities": [m for m in modalities if not (m == "audio" and not audio_input_enabled)],
        "audio_input_enabled": audio_input_enabled,
        "audio_disabled_reason": audio_disabled_reason,
        "overall": overall_metrics,
        "by_gender": gender_metrics,
        "predicted_emotion_distribution": {
            "overall": compute_pred_distribution(pred, EMOTIONS, gold),
            "by_gender": gender_pred_distributions,
        },
        "skipped_no_gender": skipped_no_gender,
        "skipped_failed_after_retries": skipped_failed,
        "missing_modality_stats": {
            "video_missing_or_unreadable": skipped_video_for_missing,
            "audio_missing_or_unreadable": skipped_audio_for_missing,
        },
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
        "model_alias": model_alias,
        "dataset": dataset,
        "task": task,
        "inference_unit": inference_unit,
        "speaker_mode": speaker_mode,
        "modalities": modalities,
        "context_window_max": context_window_max,
        "frame_sample_fps": frame_sample_fps,
        "overall_accuracy": overall_metrics["overall_accuracy"],
        "overall_weighted_f1": overall_metrics["weighted_f1"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_summary(result_root, summary_entry)
    print(f"Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
