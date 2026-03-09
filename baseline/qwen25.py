import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise ImportError("Please install PyTorch: pip install torch") from exc

try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
except ImportError as exc:
    raise ImportError(
        "Please install transformers>=4.51.3 with Qwen2.5-Omni support."
    ) from exc

try:
    from qwen_omni_utils import process_mm_info
except ImportError as exc:
    raise ImportError(
        "Please install qwen-omni-utils: pip install qwen-omni-utils -U"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from gpt52 import (
    EMOTIONS,
    ClipResolver,
    TEXT_OMITTED_TOKEN,
    YAML,
    append_summary,
    build_run_dir,
    build_speaker_labeler,
    compute_metrics,
    compute_pred_distribution,
    ensure_ffmpeg,
    extract_audio_wav,
    get_prompt_bundle,
    normalize_emotion,
    normalize_gender,
    parse_dialogue_predictions,
    parse_prediction,
    read_yaml,
    render_template,
    write_json,
)


def parse_torch_dtype(dtype_text):
    dt = str(dtype_text or "auto").strip().lower()
    if dt == "auto":
        return "auto"
    if dt in {"fp16", "float16"}:
        return torch.float16
    if dt in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dt in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("model.torch_dtype must be one of: auto, fp16, bf16, fp32")


def build_history_block(history_rows):
    if not history_rows:
        return "None"
    lines = []
    for item in history_rows:
        lines.append(
            f"- [U{item['utterance_id']}] Speaker={item['speaker']}: {item['utterance']}"
        )
    return "\n".join(lines)


def decode_generation_text(processor, generated, inputs):
    text_ids = generated
    if isinstance(generated, tuple):
        text_ids = generated[0]
    if not torch.is_tensor(text_ids):
        return str(text_ids).strip()

    input_len = 0
    if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
        input_len = int(inputs.input_ids.shape[-1])
    new_token_ids = text_ids[:, input_len:]
    text = processor.batch_decode(
        new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if text:
        return str(text[0]).strip()
    return ""


def run_qwen_generate(
    model,
    processor,
    conversation,
    use_audio_in_video,
    max_output_tokens,
    temperature,
    return_audio,
):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device)
    if hasattr(model, "dtype"):
        try:
            inputs = inputs.to(model.dtype)
        except Exception:
            # Some integer tensors should keep integer dtype.
            pass

    gen_kwargs = {
        "max_new_tokens": int(max_output_tokens),
        "use_audio_in_video": bool(use_audio_in_video),
        "return_audio": bool(return_audio),
    }
    if temperature and float(temperature) > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    return decode_generation_text(processor, generated, inputs)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni-7B on MELD test set for MERC."
    )
    parser.add_argument("--config", default="yaml/qwen25_MERC.yaml", help="Path to YAML config.")
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

    cfg = read_yaml(args.config)
    modalities = [m.lower() for m in cfg["eval"]["modalities"]]
    include_text_input = "text" in modalities
    model_name = cfg["model"]["hf_model_name"]
    model_alias = cfg["model"]["alias"]
    dataset = cfg["dataset"]["name"]
    task = cfg["task"]["name"]
    origin_dataset = bool(cfg["dataset"]["is_origin"])

    if any(m in {"audio", "video"} for m in modalities):
        ensure_ffmpeg()

    test_csv = Path(cfg["paths"]["test_csv"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    result_root = Path(cfg["paths"]["result_dir"])
    result_root.mkdir(parents=True, exist_ok=True)

    run_dir = build_run_dir(result_root, model_alias, dataset, task, origin_dataset)

    resolver = ClipResolver(raw_dir)
    tmp_media = run_dir / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)

    records = []
    gold = []
    pred = []

    retries = int(cfg["model"].get("max_retries", 2))
    sleep_s = float(cfg["model"].get("retry_sleep_seconds", 2.0))
    temperature = float(cfg["model"].get("temperature", 0.0))
    max_output_tokens = int(cfg["model"].get("max_output_tokens", 128))
    sample_retries = int(cfg.get("eval", {}).get("sample_retries", 2))
    sample_timeout_s = float(cfg.get("eval", {}).get("sample_timeout_seconds", 600.0))
    stall_log_seconds = float(cfg.get("eval", {}).get("stall_log_seconds", 180.0))
    if sample_timeout_s <= 0:
        raise ValueError("eval.sample_timeout_seconds must be > 0.")
    if stall_log_seconds <= 0:
        raise ValueError("eval.stall_log_seconds must be > 0.")

    use_audio_in_video = bool(cfg["model"].get("use_audio_in_video", True))
    return_audio = bool(cfg["model"].get("return_audio", False))
    hf_token = os.environ.get(str(cfg["model"].get("hf_token_env", "HF_TOKEN")), None)
    if hf_token == "":
        hf_token = None

    prompt_bundle = get_prompt_bundle(cfg)
    system_prompt = prompt_bundle["system"]
    user_tmpl = prompt_bundle["user_template"]
    dialogue_system_prompt = prompt_bundle["dialogue_system"]
    dialogue_user_tmpl = prompt_bundle["dialogue_user_template"]
    history_prefix_tmpl = prompt_bundle["history_prefix_template"]
    dialogue_video_prefix_tmpl = prompt_bundle["dialogue_video_prefix_template"]
    dialogue_audio_prefix_tmpl = prompt_bundle["dialogue_audio_prefix_template"]

    cfg_max_samples = cfg.get("eval", {}).get("max_samples")
    max_samples = args.max_samples if args.max_samples is not None else cfg_max_samples
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0.")
        print(f"Demo mode enabled: evaluating first {max_samples} samples.")

    cfg_context_window_max = cfg.get("eval", {}).get("context_window_max", 0)
    context_window_max = (
        args.context_window_max if args.context_window_max is not None else cfg_context_window_max
    )
    context_window_max = int(context_window_max or 0)
    if context_window_max < 0:
        raise ValueError("context_window_max must be >= 0.")
    inference_unit = args.inference_unit or str(cfg.get("eval", {}).get("inference_unit", "utterance"))
    if inference_unit not in {"utterance", "dialogue"}:
        raise ValueError("inference_unit must be 'utterance' or 'dialogue'.")
    cfg_speaker_mode = str(cfg.get("eval", {}).get("speaker_mode", "name")).lower()
    speaker_mode = args.speaker_mode or cfg_speaker_mode
    if speaker_mode not in {"name", "anon", "none"}:
        raise ValueError("speaker_mode must be one of: name, anon, none")
    speaker_label_of = build_speaker_labeler(speaker_mode)

    cfg.setdefault("eval", {})
    cfg["eval"]["max_samples"] = max_samples
    cfg["eval"]["context_window_max"] = context_window_max
    cfg["eval"]["inference_unit"] = inference_unit
    cfg["eval"]["speaker_mode"] = speaker_mode
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        YAML.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    dtype = parse_torch_dtype(cfg["model"].get("torch_dtype", "auto"))
    device_map = cfg["model"].get("device_map", "auto")
    attn_impl = cfg["model"].get("attn_implementation", None)
    low_cpu_mem_usage = bool(cfg["model"].get("low_cpu_mem_usage", True))
    cache_dir = cfg["model"].get("cache_dir", None)

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if hf_token:
        model_kwargs["token"] = hf_token

    print(f"Loading model: {model_name}")
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    except KeyError as exc:
        raise RuntimeError(
            "Current transformers version does not support Qwen2.5-Omni.\n"
            "Please install:\n"
            "pip uninstall -y transformers\n"
            "pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview\n"
            "pip install accelerate"
        ) from exc
    processor_kwargs = {}
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir
    if hf_token:
        processor_kwargs["token"] = hf_token
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, **processor_kwargs)

    if bool(cfg["model"].get("disable_talker", True)):
        model.disable_talker()

    def get_total_rows(csv_file):
        with csv_file.open(newline="", encoding="utf-8") as fcount:
            return max(sum(1 for _ in fcount) - 1, 0)

    total_rows = get_total_rows(test_csv)
    target_total = min(total_rows, max_samples) if max_samples is not None else total_rows
    pbar = tqdm(total=target_total, desc="Evaluating MELD") if tqdm is not None else None
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

        for drows in dialogues:
            success = False
            last_error = ""
            dialogue_id = str(drows[0].get("Dialogue_ID", "")).strip()
            dialogue_started = time.time()
            warned_stall = False
            for attempt in range(sample_retries):
                try:
                    elapsed = time.time() - dialogue_started
                    if elapsed >= sample_timeout_s:
                        raise TimeoutError(
                            f"Dialogue timed out after {elapsed:.1f}s (limit={sample_timeout_s:.1f}s)"
                        )
                    if elapsed >= stall_log_seconds and not warned_stall:
                        print(
                            f"[WARN] dialogue={dialogue_id} still running for {elapsed:.1f}s, continuing..."
                        )
                        warned_stall = True

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
                    user_content = [{"type": "text", "text": user_text}]
                    for r in drows:
                        clip = resolver.resolve("test", r["Dialogue_ID"], r["Utterance_ID"])
                        if "video" in modalities:
                            if clip is None:
                                skipped_video_for_missing += 1
                            else:
                                user_content.append(
                                    {
                                        "type": "text",
                                        "text": render_template(
                                            dialogue_video_prefix_tmpl,
                                            {"utterance_id": r["Utterance_ID"]},
                                        ),
                                    }
                                )
                                user_content.append({"type": "video", "video": str(clip)})
                        if "audio" in modalities:
                            if clip is None:
                                skipped_audio_for_missing += 1
                            else:
                                wav = extract_audio_wav(clip, tmp_media)
                                if wav is None:
                                    skipped_audio_for_missing += 1
                                else:
                                    user_content.append(
                                        {
                                            "type": "text",
                                            "text": render_template(
                                                dialogue_audio_prefix_tmpl,
                                                {"utterance_id": r["Utterance_ID"]},
                                            ),
                                        }
                                    )
                                    user_content.append({"type": "audio", "audio": str(wav)})

                    conversation = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": dialogue_system_prompt}],
                        },
                        {"role": "user", "content": user_content},
                    ]

                    raw_output = ""
                    ok = False
                    for _ in range(retries):
                        try:
                            raw_output = run_qwen_generate(
                                model=model,
                                processor=processor,
                                conversation=conversation,
                                use_audio_in_video=use_audio_in_video,
                                max_output_tokens=max_output_tokens,
                                temperature=temperature,
                                return_audio=return_audio,
                            )
                            ok = True
                            break
                        except Exception as exc:
                            raw_output = f"INFERENCE_ERROR: {exc}"
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
                                "raw_model_output": raw_output,
                            }
                        )
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < sample_retries - 1:
                        time.sleep(sleep_s)

            if not success:
                skipped_failed += len(drows)
                print(f"[WARN] Skip dialogue={dialogue_id}, utt_count={len(drows)}, reason={last_error}")
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
            gender_pred_distributions[gender] = compute_pred_distribution(p_sub, EMOTIONS)

        metrics = {
            "model": model_name,
            "model_alias": model_alias,
            "dataset": dataset,
            "task": task,
            "inference_unit": inference_unit,
            "speaker_mode": speaker_mode,
            "modalities": modalities,
            "context_window_max": context_window_max,
            "use_audio_in_video": use_audio_in_video,
            "overall": overall_metrics,
            "by_gender": gender_metrics,
            "predicted_emotion_distribution": {
                "overall": compute_pred_distribution(pred, EMOTIONS),
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
            history_slice = history_rows[-context_window_max:] if context_window_max > 0 else []
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
            sample_started = time.time()
            warned_stall = False
            for attempt in range(sample_retries):
                try:
                    elapsed = time.time() - sample_started
                    if elapsed >= sample_timeout_s:
                        raise TimeoutError(
                            f"Sample timed out after {elapsed:.1f}s (limit={sample_timeout_s:.1f}s)"
                        )
                    if elapsed >= stall_log_seconds and not warned_stall:
                        print(
                            f"[WARN] sample idx={idx} still running for {elapsed:.1f}s, continuing..."
                        )
                        warned_stall = True

                    clip = resolver.resolve("test", row["Dialogue_ID"], row["Utterance_ID"])
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

                    user_content = [{"type": "text", "text": user_text}]
                    if "video" in modalities:
                        if clip is None:
                            skipped_video_for_missing += 1
                        else:
                            user_content.append({"type": "video", "video": str(clip)})

                    if "audio" in modalities:
                        if clip is None:
                            skipped_audio_for_missing += 1
                        else:
                            wav = extract_audio_wav(clip, tmp_media)
                            if wav is None:
                                skipped_audio_for_missing += 1
                            else:
                                user_content.append({"type": "audio", "audio": str(wav)})

                    conversation = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}],
                        },
                        {"role": "user", "content": user_content},
                    ]

                    raw_output = ""
                    ok = False
                    for _ in range(retries):
                        try:
                            raw_output = run_qwen_generate(
                                model=model,
                                processor=processor,
                                conversation=conversation,
                                use_audio_in_video=use_audio_in_video,
                                max_output_tokens=max_output_tokens,
                                temperature=temperature,
                                return_audio=return_audio,
                            )
                            ok = True
                            break
                        except Exception as exc:
                            raw_output = f"INFERENCE_ERROR: {exc}"
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
                            "raw_model_output": raw_output,
                        }
                    )
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
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
        gender_pred_distributions[gender] = compute_pred_distribution(p_sub, EMOTIONS)

    metrics = {
        "model": model_name,
        "model_alias": model_alias,
        "dataset": dataset,
        "task": task,
        "inference_unit": inference_unit,
        "speaker_mode": speaker_mode,
        "modalities": modalities,
        "context_window_max": context_window_max,
        "use_audio_in_video": use_audio_in_video,
        "overall": overall_metrics,
        "by_gender": gender_metrics,
        "predicted_emotion_distribution": {
            "overall": compute_pred_distribution(pred, EMOTIONS),
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
        "overall_accuracy": overall_metrics["overall_accuracy"],
        "overall_weighted_f1": overall_metrics["weighted_f1"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_summary(result_root, summary_entry)
    print(f"Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
