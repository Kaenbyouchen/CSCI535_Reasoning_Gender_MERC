#!/usr/bin/env python3
"""
Text-only input attribution (gradient × input embedding) for Qwen2.5-Omni + MERC CoT prompt.

Input:  Phase-1 --cot-jsonl (cot_generations.jsonl from meld_cot_run.py; any modality run is ok,
        but this script only uses text fields to rebuild the *user* block as in Phase-1 text-only:
        system + user text, no audio/video tensors).

Output:
  <out_dir>/per_sample/{index:04d}_{safe_id}.csv   — one row per *prompt* subword (see mask column)
  <out_dir>/summary.csv                          — one row per sample + aggregate row

Loss:  -log p(first BPE of gold emotion | context), where context = chat_template string + a short
        forced "assistant" stub ending at ### 4) Final {"emotion": "  (same for MELD & IEMOCAP).

Gradient (preferred): input_ids forward + embedding output hook + grad×embedding (same idea as
        grad×input). Qwen2.5-Omni often still raises NotImplementedError during full backward.

Fallback: leave-one-token occlusion — replace each in-prompt token with a mask id, |Δ NLL| vs baseline
        (no autograd). Use --no-occlusion-fallback to disable and surface errors instead.

  python reasoning/text_input_attribution.py \\
    --cot-jsonl result/.../cot_generations.jsonl --config yaml/qwen25_MELD_reasoning.yaml
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_gender_words = frozenset({
    "he", "she", "her", "his", "him", "man", "men", "woman", "women", "boy", "girl", "male", "female",
    "guy", "guys", "girls", "boys", "husband", "wife", "mother", "father", "son", "daughter", "sister", "brother",
    "gender", "sex", "feminine", "masculine", "lady", "ladies", "gentleman", "mom", "dad", "girly",
    "target", "conversion", "counterfactual", "trans", "queer", "mr", "mrs", "ms", "sir", "madam",
})


def _mcf():
    p = _ROOT / "baseline" / "meld_counterfactual_eval.py"
    spec = importlib.util.spec_from_file_location("mcf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mcf = _mcf()
read_yaml = mcf.read_yaml
render_template = mcf.render_template
normalize_emotion = mcf.normalize_emotion


def _is_gender_token(s: str) -> bool:
    t = s.replace("Ġ", "").replace("▁", "").strip().lower()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    if not t:
        return False
    if t in _gender_words:
        return True
    for g in _gender_words:
        if g in t and len(t) <= max(len(g) + 2, 6):
            return True
    return False


def _build_user_text(
    rec: dict,
    user_tmpl: str,
    *,
    ablate_gender: bool,
) -> str:
    d_id = str(rec.get("dialogue_id", "")).strip()
    u_id = str(rec.get("utterance_id", "")).strip()
    spk = str(rec.get("Speaker", "")).strip()
    utt = str(rec.get("utterance", "")).strip()
    g = str(rec.get("gender", "")).strip()
    tg = str(rec.get("target_gender", "")).strip()
    ucf = str(rec.get("used_cf_audio", "")).strip()
    is_m = str(rec.get("dataset", "")).strip().upper() == "MELD"
    gold = normalize_emotion(rec.get("gold_emotion", "")) if is_m else str(rec.get("gold_emotion", "")).strip()
    if ablate_gender:
        g, tg = "Omitted", "Omitted"
        ucf = ucf if ucf in ("n/a", "N/A", "") else "Omitted"
    return render_template(user_tmpl, {
        "speaker": spk, "dialogue_id": d_id, "utterance_id": u_id, "utterance": utt,
        "gold_emotion": gold, "gender": g or "unknown", "target_gender": tg or "n/a", "used_cf_audio": ucf,
    })


def _chat_template_string(raw) -> str:
    """apply_chat_template(..., tokenize=False) may return str or list (processor version)."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        if not raw:
            return ""
        if all(isinstance(x, str) for x in raw):
            return "".join(raw)
        return "".join(str(x) for x in raw)
    return str(raw)


def _resolve_mask_token_id(tok) -> int:
    for mid in (tok.mask_token_id, tok.unk_token_id, tok.pad_token_id):
        if mid is not None and int(mid) >= 0:
            return int(mid)
    ids = tok.encode(".", add_special_tokens=False)
    if ids:
        return int(ids[0])
    return 0


def _compute_nll(
    qm,
    qproc,
    input_ids: torch.Tensor,
    attn: torch.Tensor,
    gold_emotion: str,
    use_aiv: bool,
    tid0: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Single forward, no grad. Returns (nll scalar tensor, first subword id of gold emotion)."""
    with torch.no_grad():
        fwd_kw = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "use_cache": False,
            "return_dict": True,
        }
        try:
            out = qm(**fwd_kw, use_audio_in_video=use_aiv)
        except TypeError:
            out = qm(**fwd_kw)
        logits = out.logits[0, -1, :]
        if tid0 is None:
            g = (gold_emotion or "").strip()
            ids_emo = qproc.tokenizer.encode(g, add_special_tokens=False)
            if not ids_emo:
                raise ValueError("empty gold emotion id")
            tid0 = int(ids_emo[0])
        nll = -F.log_softmax(logits.float(), dim=0)[tid0]
    return nll, tid0


def _pmask_tensor(off, split: int, L: int, ref: torch.Tensor) -> torch.Tensor:
    pmask: list[float] = []
    for i in range(L):
        if off is not None and i < len(off):
            a, _b = int(off[i][0]), int(off[i][1])
            in_prompt = 1.0 if a < split else 0.0
        else:
            in_prompt = 1.0
        pmask.append(in_prompt)
    return ref.new_tensor(pmask)


def _attrib_fallback_ok(e: BaseException) -> bool:
    if isinstance(e, NotImplementedError):
        return True
    if isinstance(e, RuntimeError):
        m = str(e)
        return (
            "Embedding output grad hook did not run" in m
            or "cannot attribute" in m.lower()
        )
    return False


def _forward_loss_occlusion(
    qm,
    qproc,
    t_in: str,
    suffix: str,
    gold_emotion: str,
    use_aiv: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """|Δ NLL| when each in-prompt token is replaced by mask id (no backward)."""
    dev = next(qm.parameters()).device
    full = t_in + suffix
    toks = qproc.tokenizer(
        full, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True,
    )
    input_ids = toks["input_ids"].to(dev)
    attn = toks["attention_mask"].to(dev)
    off = toks.get("offset_mapping")
    off = off[0] if off is not None else None
    split = len(t_in)
    L = int(input_ids.shape[1])
    nll_base, tid0 = _compute_nll(qm, qproc, input_ids, attn, gold_emotion, use_aiv, tid0=None)
    mask_id = _resolve_mask_token_id(qproc.tokenizer)
    sal = torch.zeros(L, device=dev, dtype=torch.float32)
    for i in range(L):
        if off is not None and i < len(off):
            a, _b = int(off[i][0]), int(off[i][1])
            if a >= split:
                continue
        ids2 = input_ids.clone()
        if int(ids2[0, i].item()) == mask_id:
            continue
        ids2[0, i] = mask_id
        nll_i, _ = _compute_nll(qm, qproc, ids2, attn, gold_emotion, use_aiv, tid0=tid0)
        sal[i] = (nll_i - nll_base).abs()
    pmask_t = _pmask_tensor(off, split, L, sal)
    return nll_base.detach(), sal.detach(), pmask_t, tid0


def _build_forced_assistant_suffix(utterance: str) -> str:
    """Minimal stub so model predicts the JSON emotion; keeps ### 1–4 structure."""
    u = (utterance or "").strip()
    return (
        f"\n### 1) Text\n{u}\n\n"
        "### 2) Acoustic\nNot provided.\n\n"
        "### 3) Integrate\nN/A\n\n"
        '### 4) Final\n{"emotion": "'
    )


def _forward_loss_gradient(
    qm,
    qproc,
    t_in: str,
    suffix: str,
    gold_emotion: str,
    use_aiv: bool,
) -> tuple[torch.Tensor, torch.Tensor, "torch.Tensor", int]:
    dev = next(qm.parameters()).device
    full = t_in + suffix
    toks = qproc.tokenizer(
        full, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True,
    )
    input_ids = toks["input_ids"].to(dev)
    attn = toks["attention_mask"].to(dev)
    off = toks.get("offset_mapping")
    if off is not None:
        off = off[0]  # (L,2)
    else:
        print("WARN: tokenizer returned no offset_mapping; in_prompt column is 1 for all tokens.",
              file=sys.stderr)
    split = len(t_in)

    emb_w = qm.get_input_embeddings()
    saved_emb: list[torch.Tensor] = []
    saved_grad: list[torch.Tensor] = []
    emb_out: torch.Tensor | None = None

    def _fwd_hook(_mod, _inp, out):
        # If the hook runs more than once, keep the last capture (should be rare).
        saved_emb[:] = [out]

    h_fwd = emb_w.register_forward_hook(_fwd_hook)
    was_req = emb_w.weight.requires_grad
    emb_w.weight.requires_grad_(True)
    th = None
    try:
        with torch.set_grad_enabled(True):
            fwd_kw = {
                "input_ids": input_ids,
                "attention_mask": attn,
                "use_cache": False,
                "return_dict": True,
            }
            try:
                out = qm(**fwd_kw, use_audio_in_video=use_aiv)
            except TypeError:
                out = qm(**fwd_kw)
            logits = out.logits[0, -1, :]
            g = (gold_emotion or "").strip()
            ids_emo = qproc.tokenizer.encode(g, add_special_tokens=False)
            if not ids_emo:
                raise ValueError("empty gold emotion id")
            tid0 = int(ids_emo[0])
            nll = -F.log_softmax(logits.float(), dim=0)[tid0]
            if not saved_emb:
                raise RuntimeError("Forward hook did not capture embedding output.")
            emb_out = saved_emb[0]
            if not emb_out.requires_grad:
                raise RuntimeError("Embedding output has requires_grad=False; cannot attribute.")
            th = emb_out.register_hook(lambda gr: saved_grad.append(gr))
            try:
                nll.backward()
            finally:
                if th is not None:
                    th.remove()
                    th = None
    finally:
        h_fwd.remove()
        emb_w.weight.requires_grad_(was_req)
        if emb_w.weight.grad is not None:
            emb_w.weight.grad = None

    if not saved_grad:
        raise RuntimeError("Embedding output grad hook did not run (backward may be unsupported on this path).")
    assert emb_out is not None
    grad = saved_grad[0][0]  # (L, h)
    emb = emb_out.detach()[0].float()
    sal = (grad.float() * emb).sum(dim=-1).abs()  # (L,)
    L = int(sal.shape[0])
    pmask_t = _pmask_tensor(off, split, L, sal)
    return nll.detach(), sal.detach(), pmask_t, tid0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-jsonl", type=Path, required=True, help="Path to Phase-1 cot_generations.jsonl")
    ap.add_argument("--config", type=Path, default=Path("yaml/qwen25_MELD_reasoning.yaml"))
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="Default: result/text_attribution/<jsonl_stem>_<ablate>/",
    )
    ap.add_argument("--ablate-gender-meta", action="store_true", help="Omit gender/target_gender in user block")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all lines")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config-iemocap", type=Path, default=Path("yaml/qwen25_IEMOCAP_reasoning.yaml"),
                    help="If set, use for lines with dataset=IEMOCAP when not using --one-config")
    ap.add_argument("--one-config", action="store_true", help="Use only --config for all (ignore IEMOCAP template)")
    ap.add_argument(
        "--no-occlusion-fallback",
        action="store_true",
        help="If gradient attribution fails, abort instead of leave-one-token occlusion",
    )
    args = ap.parse_args()
    os.chdir(_ROOT)
    device = torch.device(args.device)
    p_jsonl = args.cot_jsonl if args.cot_jsonl.is_absolute() else _ROOT / args.cot_jsonl
    if not p_jsonl.is_file():
        print(f"Missing: {p_jsonl}", file=sys.stderr)
        sys.exit(1)

    cfg_p = args.config if args.config.is_absolute() else _ROOT / args.config
    cfg_meld = read_yaml(cfg_p)
    cfg_ip = (args.config_iemocap if args.config_iemocap.is_absolute() else
              _ROOT / args.config_iemocap)
    cfg_iemo = read_yaml(cfg_ip)

    prompt_cfg = cfg_meld.get("prompt", {}) or {}
    system_cot = prompt_cfg.get("system_cot", "")
    user_tmpl_meld = prompt_cfg.get("user_cot_template", "")

    prompt_i = cfg_iemo.get("prompt", {}) or {}
    user_tmpl_iemo = prompt_i.get("user_cot_template", user_tmpl_meld)
    system_cot_iemo = prompt_i.get("system_cot", system_cot)

    model = cfg_meld.get("model", {})
    model_name = model["hf_model_name"]
    use_aiv = bool(model.get("use_audio_in_video", False))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        str(model.get("torch_dtype", "bf16")).lower(), "auto"
    )
    m_kw = {
        "torch_dtype": dtype, "device_map": "auto" if device.type == "cuda" else None,
        "low_cpu_mem_usage": True, "attn_implementation": model.get("attn_implementation", "sdpa"),
    }
    if device.type == "cpu":
        m_kw.pop("device_map", None)
    cdir = model.get("cache_dir")
    if cdir:
        m_kw["cache_dir"] = cdir
    tok = os.environ.get(model.get("hf_token_env", "HF_TOKEN") or "HF_TOKEN", None) or None
    if tok:
        m_kw["token"] = tok

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    print(f"Loading {model_name} ...")
    qm = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **m_kw)
    p_kw = {k: v for k, v in [("cache_dir", cdir), ("token", tok)] if v}
    qproc = Qwen2_5OmniProcessor.from_pretrained(model_name, **p_kw)
    if model.get("disable_talker", True):
        qm.disable_talker()
    qm.eval()
    dev = next(qm.parameters()).device
    for p in qm.parameters():
        p.requires_grad = False

    stem = p_jsonl.stem
    abtag = "ablate" if args.ablate_gender_meta else "full"
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = _ROOT / "result" / "text_attribution" / f"{stem}_{abtag}"
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    (out_dir / "per_sample").mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample"
    print(f"Output -> {out_dir}")

    rows = []
    with p_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n_total = len(rows)
    n_cap = n_total if not args.max_samples else min(args.max_samples, n_total)
    rows = rows[:n_cap]

    summary_path = out_dir / "summary.csv"
    all_summ = []

    for idx, rec in enumerate(rows, start=1):
        dtag = str(rec.get("dataset", "MELD")).strip().upper()
        is_iemo = dtag == "IEMOCAP" and (not args.one_config)
        user_tmpl = user_tmpl_meld if not is_iemo else user_tmpl_iemo
        g_raw = str(rec.get("gold_emotion", "")).strip()
        gold = normalize_emotion(g_raw) if dtag == "MELD" else g_raw
        d_id = str(rec.get("dialogue_id", "")).strip()
        u_id = str(rec.get("utterance_id", "")).strip()
        utt = str(rec.get("utterance", "")).strip()

        user_text = _build_user_text(rec, user_tmpl, ablate_gender=bool(args.ablate_gender_meta))
        user_content = [{"type": "text", "text": user_text}]
        sc = system_cot_iemo if is_iemo else system_cot
        convo = [
            {"role": "system", "content": [{"type": "text", "text": sc}]},
            {"role": "user", "content": user_content},
        ]

        t_in = _chat_template_string(
            qproc.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
        )
        if not t_in or not t_in.strip():
            print(f"[{idx}] skip: empty template", file=sys.stderr)
            continue
        suff = _build_forced_assistant_suffix(utt)
        attrib_method = ""
        try:
            nll, sal, pmask, _tid0 = _forward_loss_gradient(qm, qproc, t_in, suff, gold, use_aiv)
            attrib_method = "grad"
        except Exception as e:
            if args.no_occlusion_fallback or not _attrib_fallback_ok(e):
                err_msg = f"{type(e).__name__}: {e!s}" if str(e) else f"{type(e).__name__}: (no message) {e!r}"
                print(f"[{idx}] {d_id}/{u_id} err: {err_msg}", file=sys.stderr)
                all_summ.append({
                    "index": idx, "dialogue_id": d_id, "utterance_id": u_id, "dataset": dtag, "gold_emotion": gold,
                    "nll": "", "in_prompt_sal_sum": "", "in_prompt_sal_gender": "", "frac_sal_in_gender": "",
                    "n_prompt_tokens": "", "n_gender_in_prompt": "", "attrib_method": "",
                    "error": err_msg[:500],
                })
                continue
            print(
                f"[{idx}] {d_id}/{u_id} grad failed ({type(e).__name__}); using occlusion",
                file=sys.stderr,
            )
            try:
                nll, sal, pmask, _tid0 = _forward_loss_occlusion(qm, qproc, t_in, suff, gold, use_aiv)
                attrib_method = "occlusion"
            except Exception as e2:
                err_msg = f"{type(e2).__name__}: {e2!s}" if str(e2) else f"{type(e2).__name__}: (no message) {e2!r}"
                print(f"[{idx}] {d_id}/{u_id} err: {err_msg}", file=sys.stderr)
                all_summ.append({
                    "index": idx, "dialogue_id": d_id, "utterance_id": u_id, "dataset": dtag, "gold_emotion": gold,
                    "nll": "", "in_prompt_sal_sum": "", "in_prompt_sal_gender": "", "frac_sal_in_gender": "",
                    "n_prompt_tokens": "", "n_gender_in_prompt": "", "attrib_method": "",
                    "error": err_msg[:500],
                })
                continue
        toks = qproc.tokenizer(t_in + suff, return_tensors="pt", return_offset_mapping=True, add_special_tokens=True)
        input_ids = toks["input_ids"][0]
        tks = qproc.tokenizer.convert_ids_to_tokens(input_ids)
        L = int(pmask.shape[0])
        sal_list = [float(sal[i].item()) for i in range(L)]
        if len(tks) < L:
            tks = tks + [""] * (L - len(tks))
        in_s = 0.0
        in_g = 0.0
        n_pr = 0
        n_gm = 0
        p_csv = out_dir / "per_sample" / f"{idx:04d}_d{d_id}u{u_id}.csv".replace(" ", "_")
        with p_csv.open("w", encoding="utf-8", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(
                ("token_index", "token", "saliency", "in_prompt", "is_gender_lexicon",),
            )
            for i in range(L):
                tk = tks[i] if i < len(tks) else ""
                s = float(sal[i].item()) if i < len(sal_list) else 0.0
                in_p = float(pmask[i].item()) if i < len(pmask) else 0.0
                gflag = 1.0 if _is_gender_token(tk) else 0.0
                w.writerow((i, tk, s, in_p, gflag))
                if in_p > 0.5:
                    in_s += s
                    n_pr += 1
                    if gflag > 0.5:
                        in_g += s
                        n_gm += 1
        num = in_s if in_s > 0 else 1e-8
        frac = in_g / num
        all_summ.append({
            "index": idx, "dialogue_id": d_id, "utterance_id": u_id, "dataset": dtag, "gold_emotion": gold,
            "nll": float(nll.item()), "in_prompt_sal_sum": in_s, "in_prompt_sal_gender": in_g,
            "frac_sal_in_gender": float(frac), "n_prompt_tokens": n_pr, "n_gender_in_prompt": n_gm,
            "attrib_method": attrib_method,
            "error": "OK",
        })
        if idx % 10 == 0:
            print(f"  {idx}/{n_cap} nll={float(nll.item()):.4f} frac_g={frac:.4f}", flush=True)

    ok = []
    for r in all_summ:
        if r.get("error") != "OK":
            continue
        try:
            float(r["nll"])
            float(r["frac_sal_in_gender"])
        except (TypeError, ValueError):
            continue
        ok.append(r)
    if ok:
        m_frac = sum(float(r["frac_sal_in_gender"]) for r in ok) / len(ok)
        m_nll = sum(float(r["nll"]) for r in ok) / len(ok)
    else:
        m_frac, m_nll = float("nan"), float("nan")

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(all_summ[0].keys()) if all_summ else [
            "index", "dialogue_id", "utterance_id", "dataset", "gold_emotion", "nll",
            "in_prompt_sal_sum", "in_prompt_sal_gender", "frac_sal_in_gender",
            "n_prompt_tokens", "n_gender_in_prompt", "attrib_method", "error",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_summ:
            w.writerow(r)
        w.writerow({**{k: "" for k in fields},
                    "index": "MEAN(OK)", "nll": m_nll, "frac_sal_in_gender": m_frac,
                    "in_prompt_sal_sum": "—", "in_prompt_sal_gender": "—", "n_prompt_tokens": "—",
                    "n_gender_in_prompt": "—", "attrib_method": "—",
                    "error": f"n_ok={len(ok)}/n={len(all_summ)}"})
    n_ok = sum(1 for r in all_summ if r.get("error") == "OK")
    print(f"Wrote {summary_path}; per_sample CSVs with data: {n_ok}/{len(all_summ)} -> {per_sample_path}")


if __name__ == "__main__":
    main()
