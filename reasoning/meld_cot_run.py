#!/usr/bin/env python3
"""
Task 3 — CoT (chain-of-thought) with Qwen2.5-Omni.
  - MELD:   CF-success ∩ MELD test (same as counterfactual eval); optional BL/CF pred CSVs for flip subsample
  - IEMOCAP: test_sessions from yaml (e.g. Session 5), same utterance format as iemocap_eval

Usage (from project root):
  python reasoning/meld_cot_run.py --config yaml/qwen25_MELD_reasoning.yaml
  python reasoning/meld_cot_run.py --config yaml/qwen25_IEMOCAP_reasoning.yaml --modalities text audio
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_mcf():
    p = _ROOT / "baseline" / "meld_counterfactual_eval.py"
    spec = importlib.util.spec_from_file_location("mcf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_ice():
    p = _ROOT / "baseline" / "iemocap_eval.py"
    spec = importlib.util.spec_from_file_location("ice", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mcf = _load_mcf()
_ice = None


def _ice_mod():
    global _ice
    if _ice is None:
        _ice = _load_ice()
    return _ice


read_yaml = mcf.read_yaml
YAML = mcf.YAML
parse_pred = mcf.parse_pred
normalize_emotion = mcf.normalize_emotion
normalize_gender = mcf.normalize_gender
render_template = mcf.render_template
EMOTIONS = mcf.EMOTIONS
TEXT_OMITTED = mcf.TEXT_OMITTED
ClipResolver = mcf.ClipResolver
ensure_ffmpeg = mcf.ensure_ffmpeg
normalize_wav = mcf.normalize_wav
extract_audio_from_video = mcf.extract_audio_from_video
build_speaker_labeler = mcf.build_speaker_labeler


def _load_pred_csv(path: Path) -> dict[tuple[str, str], str]:
    if not path or not Path(path).exists():
        return {}
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (str(row.get("dialogue_id", row.get("Dialogue_ID", ""))).strip(),
                 str(row.get("utterance_id", row.get("Utterance_ID", ""))).strip())
            if k[0] and k[1]:
                out[k] = normalize_emotion(row.get("pred_emotion", ""))
    return out


def _stratify_fill(rows, n: int, rng: random.Random, is_meld: bool) -> list:
    if n <= 0 or not rows:
        return []
    g = defaultdict(list)
    for r in rows:
        if is_meld:
            e = normalize_emotion(r.get("Emotion", ""))
            gen = (r.get("_cf") or {}).get("gender") or "unk"
        else:
            e = str(r.get("Emotion", "")).strip()
            gen = str(r.get("gender") or "unk")
        g[(e, gen)].append(r)
    for k in g:
        rng.shuffle(g[k])
    keys = [k for k, v in g.items() if v]
    if not keys:
        rng.shuffle(rows)
        return rows[:n]
    rng.shuffle(keys)
    out = []
    i = 0
    while len(out) < n and any(g[k] for k in keys):
        k = keys[i % len(keys)]
        if g[k]:
            out.append(g[k].pop(0))
        i += 1
        if i > len(keys) * 5000:
            break
    if len(out) < n:
        rem = [x for k in keys for x in g[k]]
        rng.shuffle(rem)
        out.extend(rem[:(n - len(out))])
    return out[:n]


def build_iemocap_rows(cfg) -> list[dict]:
    ice = _ice_mod()
    ds = cfg.get("dataset", {})
    data_dir = _ROOT / ds.get("data_dir", "data/IEMOCAP_full_release")
    labels = list(cfg.get("task", {}).get("labels", []))
    if not labels:
        raise ValueError("IEMOCAP: task.labels is required in YAML.")
    sessions = ds.get("test_sessions", [5])
    flt = bool(ds.get("filter_other", True))
    dials = ice.load_iemocap(data_dir, set(labels), filter_other=flt, sessions=sessions)
    rows = []
    for d in dials:
        for u in d["utterances"]:
            rows.append({
                "Dialogue_ID": d["dialogue_id"],
                "Utterance_ID": u["utterance_id"],
                "Speaker": u.get("speaker", "Unknown"),
                "Emotion": u["emotion"],
                "Utterance": u.get("text", ""),
                "gender": u.get("gender", ""),
                "_iem": {"dialogue": d, "utt": u},
            })
    return rows


def subsample_rows_meld(
    all_rows: list[dict],
    n: int,
    seed: int,
    min_flips: int,
    bl_map: dict,
    cf_map: dict,
) -> list[dict]:
    rng = random.Random(seed)
    kfn = lambda r: (str(r["Dialogue_ID"]).strip(), str(r["Utterance_ID"]).strip())
    flip_keys: set = set()
    for r in all_rows:
        k = kfn(r)
        if k in bl_map and k in cf_map and bl_map[k] != cf_map[k]:
            flip_keys.add(k)
    flip_rows = [r for r in all_rows if kfn(r) in flip_keys]
    rng.shuffle(flip_rows)
    take_fl = min(min_flips, len(flip_rows), n)
    selected = [flip_rows[i] for i in range(take_fl)]
    skeys = {kfn(r) for r in selected}
    rest = [r for r in all_rows if kfn(r) not in skeys]
    need = n - len(selected)
    if need > 0:
        more = _stratify_fill(rest, need, rng, True)
        for r in more:
            if kfn(r) not in skeys:
                skeys.add(kfn(r))
                selected.append(r)
    if len(selected) < n:
        pool = [r for r in all_rows if kfn(r) not in skeys]
        rng.shuffle(pool)
        for r in pool:
            if len(selected) >= n:
                break
            selected.append(r)
    rng.shuffle(selected)
    return selected[:n]


def build_run_dir(result_root, model_alias, modality_tag: str, dataset_tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_alias}_{dataset_tag}_REASON_{modality_tag}_{ts}"
    d = Path(result_root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="yaml/qwen25_MELD_reasoning.yaml", help="YAML path")
    ap.add_argument("--modalities", nargs="*", default=None, help="Override e.g. text audio video")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--no-audit-hint", action="store_true", help="If set, do not add flip metadata file")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfg = read_yaml(_ROOT / args.config if not Path(args.config).is_absolute() else args.config)

    model = cfg.get("model", {})
    model_name = model["hf_model_name"]
    model_alias = model.get("alias", "qwen25_reason")
    paths = cfg.get("paths", {})
    sub = cfg.get("subsample", {})
    n_samples = int(args.n_samples or sub.get("n_samples", 80))
    seed = int(sub.get("random_seed", 42))
    is_meld = str(cfg.get("dataset", {}).get("name", "MELD")).strip().upper() == "MELD"
    task_labels = list(cfg.get("task", {}).get("labels", list(EMOTIONS)))

    if is_meld:
        audio_source = str(sub.get("audio_source", "counterfactual")).lower()
        if audio_source not in ("original", "counterfactual"):
            raise ValueError("MELD subsample.audio_source must be original or counterfactual")
        min_flips = int(sub.get("min_flips_in_batch", 0))
    else:
        audio_source = "original"
        min_flips = 0

    modalities = [m.lower() for m in (args.modalities or cfg.get("eval", {}).get("modalities", ["text", "audio"]))]
    if not modalities:
        raise ValueError("No modalities")
    include_text = "text" in modalities
    include_audio = "audio" in modalities
    include_video = "video" in modalities
    if not (include_text or include_audio or include_video):
        raise ValueError("Need at least one of text, audio, video")
    if include_audio or include_video:
        ensure_ffmpeg()

    result_root = Path(paths.get("result_dir", "result"))
    all_rows: list[dict] = []
    bl_pred_map: dict = {}
    cf_pred_map: dict = {}
    raw_dir: Path = Path(".")

    if is_meld:
        test_csv = Path(paths["test_csv"])
        raw_dir = Path(paths["raw_dir"])
        cf_meta = Path(paths["cf_metadata_csv"])
        cf_map: dict = {}
        with cf_meta.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("conversion_status", "")).strip().lower() != "success":
                    continue
                key = (str(row["dialogue_id"]).strip(), str(row["utterance_id"]).strip())
                cf_rel = str(row.get("converted_audio_path", "")).strip().lstrip("./")
                cf_p = Path(cf_rel) if cf_rel else None
                cf_map[key] = {
                    "speaker": row.get("speaker", "").strip(),
                    "gender": normalize_gender(row.get("gender", "")),
                    "target_gender": normalize_gender(row.get("target_gender", "")),
                    "cf_audio_path": cf_p,
                }
        with test_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (str(row["Dialogue_ID"]).strip(), str(row["Utterance_ID"]).strip())
                if key in cf_map:
                    row["_cf"] = cf_map[key]
                    all_rows.append(row)
        bl_path = paths.get("predictions_bl_csv")
        cf_pred_path = paths.get("predictions_cf_csv")
        if isinstance(bl_path, str) and str(bl_path).lower() in ("null", "none", ""):
            bl_path = None
        if isinstance(cf_pred_path, str) and str(cf_pred_path).lower() in ("null", "none", ""):
            cf_pred_path = None
        if bl_path:
            bl_path = (Path(bl_path) if Path(bl_path).is_absolute() else _ROOT / bl_path)
        if cf_pred_path:
            cf_pred_path = (Path(cf_pred_path) if Path(cf_pred_path).is_absolute() else _ROOT / cf_pred_path)
        bl_pred_map = _load_pred_csv(Path(bl_path)) if bl_path else {}
        cf_pred_map = _load_pred_csv(Path(cf_pred_path)) if cf_pred_path else {}
        if min_flips and (not bl_pred_map or not cf_pred_map):
            print("[WARN] min_flips>0 but BL/CF prediction CSVs missing; flip-aware subsample degraded.")
    else:
        all_rows = build_iemocap_rows(cfg)
        # For video/audio paths, iemocap load uses data_dir; expose raw_dir for any relative resolve
        raw_dir = _ROOT / cfg.get("dataset", {}).get("data_dir", "data/IEMOCAP_full_release")

    if not all_rows:
        raise RuntimeError("No utterances in pool (check paths / sessions / MELD CF filter).")
    n_cap = min(n_samples, len(all_rows))
    if n_cap >= len(all_rows):
        rows_kept = all_rows
    elif is_meld and sub.get("stratify_emotion_gender", True):
        rows_kept = subsample_rows_meld(
            all_rows, n_cap, seed, min_flips, bl_pred_map, cf_pred_map)
    elif (not is_meld) and sub.get("stratify_emotion_gender", True):
        rows_kept = _stratify_fill(
            all_rows, n_cap, random.Random(seed), is_meld=False)
    else:
        rng = random.Random(seed)
        pool = all_rows[:]
        rng.shuffle(pool)
        rows_kept = pool[:n_cap] if n_cap < len(all_rows) else all_rows
    if is_meld:
        print(f"[Reason] MELD: {len(rows_kept)} utterances (from {len(all_rows)} CF-success test rows).")
    else:
        print(f"[Reason] IEMOCAP: {len(rows_kept)} utterances (pool {len(all_rows)} in selected sessions).")

    modality_tag = "+".join(modalities)
    dtag = "MELD" if is_meld else "IEMOCAP"
    run_dir = build_run_dir(result_root, model_alias, modality_tag, dtag)
    ice = _ice_mod()
    tmp_media = run_dir / "tmp_media"
    tmp_media.mkdir(parents=True, exist_ok=True)

    prompt_cfg = cfg.get("prompt", {}) or {}
    system_cot = prompt_cfg.get("system_cot", "")
    user_tmpl = prompt_cfg.get("user_cot_template", prompt_cfg.get("user_template", ""))
    audio_prefix = prompt_cfg.get("audio_prefix_template", "Audio of TARGET utterance:")
    video_prefix = prompt_cfg.get("video_prefix_template", "Video of TARGET utterance:")

    _vmin = 128 * 28 * 28
    evc = cfg.get("eval", {})
    video_fps = float(evc.get("video_fps", 1.0))
    video_max_pixels = int(evc.get("video_max_pixels", _vmin))
    if video_max_pixels < _vmin:
        video_max_pixels = _vmin
    v_max_fr = evc.get("video_max_frames")
    video_max_frames = int(v_max_fr) if v_max_fr is not None else None

    speaker_mode = str(evc.get("speaker_mode", "anon" if is_meld else "none"))
    speaker_of = build_speaker_labeler(speaker_mode)
    resolver = ClipResolver(raw_dir) if is_meld else None

    sample_records = []
    for row in rows_kept:
        key = (str(row["Dialogue_ID"]).strip(), str(row["Utterance_ID"]).strip())
        if is_meld:
            is_fl = bool(bl_pred_map and cf_pred_map and
                         bl_pred_map.get(key) != cf_pred_map.get(key))
            sample_records.append({
                "dialogue_id": key[0], "utterance_id": key[1],
                "in_flip_pool": is_fl and key in bl_pred_map and key in cf_pred_map,
            })
        else:
            sample_records.append({
                "dialogue_id": key[0], "utterance_id": key[1],
            })
    with (run_dir / "subsample_manifest.json").open("w", encoding="utf-8") as f:
        man = {
            "dataset": dtag, "n": len(rows_kept), "rows": sample_records,
            "modality": modality_tag, "audio_source": audio_source,
        }
        json.dump(man, f, indent=2, ensure_ascii=False)

    with (run_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        dump = {
            "dataset": cfg.get("dataset", {"name": "MELD"}),
            "model": model, "task": cfg.get("task", {}), "paths": {k: str(v) for k, v in paths.items()},
            "subsample": sub, "eval": {**evc, "modalities": modalities},
        }
        YAML.safe_dump(dump, f, sort_keys=False, allow_unicode=True)

    # Load Qwen
    import torch
    from transformers import (Qwen2_5OmniForConditionalGeneration,
                              Qwen2_5OmniProcessor)
    from qwen_omni_utils import process_mm_info as _qwen_pmm

    dtype_str = str(model.get("torch_dtype", "bf16")).lower()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}.get(dtype_str, "auto")
    use_aiv = bool(model.get("use_audio_in_video", False))
    return_audio = bool(model.get("return_audio", False))
    max_new = int(model.get("max_cot_output_tokens", 1024))
    temperature = float(model.get("temperature", 0.0))
    retries = int(model.get("max_retries", 2))
    sleep_s = float(model.get("retry_sleep_seconds", 2.0))
    log_every = int(evc.get("log_every_n", 5))

    model_kwargs = {"torch_dtype": dtype, "device_map": model.get("device_map", "auto"),
                    "low_cpu_mem_usage": bool(model.get("low_cpu_mem_usage", True))}
    if model.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model["attn_implementation"]
    cdir = model.get("cache_dir")
    if cdir:
        model_kwargs["cache_dir"] = cdir
    tok = os.environ.get(model.get("hf_token_env", "HF_TOKEN") or "HF_TOKEN", None) or None
    if tok:
        model_kwargs["token"] = tok

    print(f"Loading Qwen: {model_name} ...")
    try:
        qm = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    except (ImportError, ValueError, RuntimeError) as e:
        msg = str(e).lower()
        if (model_kwargs.get("attn_implementation") == "flash_attention_2" and
                "flash" in msg):
            kw2 = {**model_kwargs, "attn_implementation": "sdpa"}
            qm = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kw2)
        else:
            raise
    proc_kw = {}
    if cdir:
        proc_kw["cache_dir"] = cdir
    if tok:
        proc_kw["token"] = tok
    qproc = Qwen2_5OmniProcessor.from_pretrained(model_name, **proc_kw)
    if model.get("disable_talker", True):
        qm.disable_talker()
    print("Model ready.")

    out_jsonl = run_dir / "cot_generations.jsonl"
    f_out = out_jsonl.open("w", encoding="utf-8")
    t0 = time.time()

    for i, row in enumerate(rows_kept, start=1):
        d_id = str(row["Dialogue_ID"]).strip()
        u_id = str(row["Utterance_ID"]).strip()
        spk = str(row.get("Speaker", "")).strip()
        sl = speaker_of(d_id, spk)
        utt = str(row.get("Utterance", "")).strip()
        utt_for = utt if include_text else TEXT_OMITTED
        is_flip = False
        d_rec, uu = None, None

        if is_meld:
            gold = normalize_emotion(row.get("Emotion", ""))
            cfi = row["_cf"]
            ogen = cfi.get("gender", "")
            tgen = cfi.get("target_gender", "")
            use_cf = audio_source == "counterfactual"
            used_cf_audio = "no"
            if use_cf and cfi.get("cf_audio_path") and cfi["cf_audio_path"].exists():
                used_cf_audio = "yes"
            is_flip = bool(
                bl_pred_map and cf_pred_map
                and bl_pred_map.get((d_id, u_id)) != cf_pred_map.get((d_id, u_id))
            )
        else:
            gold = str(row.get("Emotion", "")).strip()
            ogen = str(row.get("gender", ""))
            tgen = "n/a"
            used_cf_audio = "n/a"
            iem = row["_iem"]
            d_rec, uu = iem["dialogue"], iem["utt"]

        user_text = render_template(user_tmpl, {
            "speaker": sl, "dialogue_id": d_id, "utterance_id": u_id, "utterance": utt_for,
            "gold_emotion": gold, "gender": ogen or "unknown",
            "target_gender": tgen or "n/a", "used_cf_audio": used_cf_audio,
        })

        user_content = [{"type": "text", "text": user_text}]
        if is_meld and resolver is not None:
            clip = resolver.resolve("test", d_id, u_id)
        else:
            clip = None

        if is_meld:
            if include_video:
                if clip is None:
                    user_content.append({"type": "text", "text": f"{video_prefix} (missing file)"})
                else:
                    user_content.append({"type": "text", "text": video_prefix})
                    vitem = {"type": "video", "video": str(clip), "fps": video_fps,
                              "max_pixels": video_max_pixels}
                    if video_max_frames is not None:
                        vitem["max_frames"] = video_max_frames
                    user_content.append(vitem)
            if include_audio:
                wav = None
                if audio_source == "original" and clip is not None:
                    wav = extract_audio_from_video(clip, tmp_media)
                elif audio_source == "counterfactual" and cfi.get("cf_audio_path"):
                    p = cfi["cf_audio_path"]
                    if p and Path(p).exists():
                        wav = normalize_wav(p, tmp_media)
                if wav is None:
                    user_content.append({"type": "text", "text": f"{audio_prefix} (unavailable; check paths)"})
                else:
                    user_content.append({"type": "text", "text": audio_prefix})
                    user_content.append({"type": "audio", "audio": str(wav)})
        else:
            wav_path, avi_path = d_rec.get("wav_path"), d_rec.get("avi_path")
            if include_video and avi_path and Path(avi_path).exists():
                seg_v = ice.extract_utt_video(avi_path, uu["start"], uu["end"], tmp_media, uu["utterance_id"])
                if not seg_v:
                    user_content.append({"type": "text", "text": f"{video_prefix} (extract failed)"})
                else:
                    user_content.append({"type": "text", "text": video_prefix})
                    vitem = {"type": "video", "video": str(seg_v), "fps": video_fps,
                              "max_pixels": video_max_pixels}
                    if video_max_frames is not None:
                        vitem["max_frames"] = video_max_frames
                    user_content.append(vitem)
            elif include_video:
                user_content.append({"type": "text", "text": f"{video_prefix} (missing .avi)"})
            if include_audio and wav_path and Path(wav_path).exists():
                seg_a = ice.extract_utt_wav(wav_path, uu["start"], uu["end"],
                                            tmp_media, uu["utterance_id"])
                if not seg_a:
                    user_content.append({"type": "text", "text": f"{audio_prefix} (extract failed)"})
                else:
                    user_content.append({"type": "text", "text": audio_prefix})
                    user_content.append({"type": "audio", "audio": str(seg_a)})
            elif include_audio:
                user_content.append({"type": "text", "text": f"{audio_prefix} (missing .wav)"})

        convo = [
            {"role": "system", "content": [{"type": "text", "text": system_cot}]},
            {"role": "user", "content": user_content},
        ]
        raw, err = "", ""
        for _ in range(retries):
            try:
                t_in = qproc.apply_chat_template(
                    convo, add_generation_prompt=True, tokenize=False)
                aud, img, vid = _qwen_pmm(convo, use_audio_in_video=use_aiv)
                ins = qproc(
                    text=t_in, audio=aud, images=img, videos=vid,
                    return_tensors="pt", padding=True, use_audio_in_video=use_aiv,
                ).to(qm.device)
                gkw = {"max_new_tokens": max_new, "use_audio_in_video": use_aiv,
                       "return_audio": return_audio, "do_sample": bool(temperature > 0)}
                if temperature and temperature > 0:
                    gkw["temperature"] = float(temperature)
                with torch.inference_mode():
                    out = qm.generate(**ins, **gkw)
                seq = out[0] if not hasattr(out, "sequences") else out.sequences
                if isinstance(seq, (tuple, list)) and not hasattr(seq, "shape"):
                    seq = seq[0]
                li = int(ins.input_ids.shape[-1])
                if isinstance(seq, torch.Tensor) and seq.dim() == 1:
                    gen_ids = seq[li:]
                    raw = qproc.batch_decode(
                        gen_ids.unsqueeze(0), skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0].strip()
                else:
                    raw = qproc.batch_decode(
                        seq[:, li:], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0].strip()
                break
            except Exception as exc:
                err = str(exc)
                time.sleep(sleep_s)
        else:
            raw = ""

        if is_meld:
            pred_emo = parse_pred(raw) if raw else ""
        else:
            pred_emo = ice.parse_pred_utterance(raw, task_labels) if raw else ""

        rec = {
            "dataset": dtag, "dialogue_id": d_id, "utterance_id": u_id,
            "utterance": utt, "gold_emotion": gold, "pred_emotion": pred_emo,
            "raw_cot": raw, "error": err if (not raw and err) else "",
            "model": model_name, "modalities": modalities, "audio_source": audio_source,
            "used_cf_audio": used_cf_audio, "gender": ogen, "target_gender": tgen,
            "is_bl_cf_flip": is_flip if is_meld else None, "index": i,
        }
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()
        if i % log_every == 0 or i == len(rows_kept):
            dt = time.time() - t0
            print(f"  [{i}/{len(rows_kept)}] {dt:.0f}s elapsed, last pred={pred_emo}")

    f_out.close()
    out_abs = out_jsonl.resolve()
    try:
        out_rel = str(out_abs.relative_to(_ROOT))
    except ValueError:
        out_rel = str(out_abs)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "n_written": len(rows_kept), "out": out_rel,
        }, f, indent=2)
    print(f"Done. {out_jsonl}")


if __name__ == "__main__":
    main()
