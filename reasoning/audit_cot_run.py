#!/usr/bin/env python3
"""
Phase 2 — Audit CoT lines from cot_generations.jsonl (OpenAI chat or local Qwen, text-only).

  python reasoning/audit_cot_run.py --cot-jsonl result/.../cot_generations.jsonl
  python reasoning/audit_cot_run.py --cot-jsonl result/.../cot_generations.jsonl --config yaml/qwen25_MELD_reasoning.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_yaml():
    import importlib.util
    p = _ROOT / "baseline" / "meld_counterfactual_eval.py"
    spec = importlib.util.spec_from_file_location("mcf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mcf = _load_yaml()
read_yaml = mcf.read_yaml
YAML = mcf.YAML
render_template = mcf.render_template


def _parse_json_loose(s: str) -> dict:
    s = s.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"stereotype_score": 0, "contains_gender_bias": False, "evidence": "",
            "explanation": "parse failed", "raw": s[:500]}


def run_openai(system: str, user: str, model: str, api_key: str, max_tok: int) -> str:
    from openai import BadRequestError, OpenAI
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_completion_tokens=max_tok,
        )
    except BadRequestError as e:
        em = str(e).lower()
        if "max_completion_tokens" in em and "unsupported" in em:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=max_tok,
            )
        else:
            raise
    return (r.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-jsonl", required=True, type=Path, help="Path to cot_generations.jsonl")
    ap.add_argument("--config", default="yaml/qwen25_MELD_reasoning.yaml", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Default: same dir as cot_audited.jsonl")
    ap.add_argument("--max-lines", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0, help="Skip first N lines")
    ap.add_argument("--backend", choices=["openai", "qwen"], default=None, help="Override YAML")
    args = ap.parse_args()
    os.chdir(_ROOT)
    cfgp = _ROOT / args.config if not args.config.is_absolute() else args.config
    cfg = read_yaml(cfgp)
    base = (args.cot_jsonl if args.cot_jsonl.is_absolute() else _ROOT / args.cot_jsonl)
    if not base.exists():
        print(f"Missing: {base}")
        sys.exit(1)
    out_path = args.out
    if out_path is None:
        out_path = base.parent / "cot_audited.jsonl"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path

    aud = cfg.get("auditor", {})
    backend = args.backend or str(aud.get("backend", "openai"))
    pcfg = cfg.get("prompt", {}) or {}
    sys_a = pcfg.get("system_auditor", "")
    user_tmpl = pcfg.get("user_auditor_template", "")

    qm_name = None
    qm_kw = {}
    if backend == "qwen":
        qm_name = (aud.get("hf_model_name") or cfg.get("model", {}).get("hf_model_name"))
        mdl = cfg.get("model", {})
        import torch
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
            str(mdl.get("torch_dtype", "bf16")).lower(), "auto")
        qm_kw = {"torch_dtype": dtype, "device_map": mdl.get("device_map", "auto"),
                 "low_cpu_mem_usage": True, "attn_implementation": mdl.get("attn_implementation", "sdpa")}
        if mdl.get("cache_dir"):
            qm_kw["cache_dir"] = mdl["cache_dir"]
        tok = os.environ.get(mdl.get("hf_token_env", "HF_TOKEN") or "HF_TOKEN", None) or None
        if tok:
            qm_kw["token"] = tok
        # Load once
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        print(f"Loading Qwen for audit: {qm_name} ...")
        try:
            qm = Qwen2_5OmniForConditionalGeneration.from_pretrained(qm_name, **qm_kw)
        except (ImportError, ValueError, RuntimeError) as e:
            if qm_kw.get("attn_implementation") == "flash_attention_2" and "flash" in str(e).lower():
                qm_kw2 = {**qm_kw, "attn_implementation": "sdpa"}
                qm = Qwen2_5OmniForConditionalGeneration.from_pretrained(qm_name, **qm_kw2)
            else:
                raise
        qproc = Qwen2_5OmniProcessor.from_pretrained(
            qm_name, cache_dir=qm_kw.get("cache_dir"), token=qm_kw.get("token"))
        if mdl.get("disable_talker", True):
            qm.disable_talker()
    else:
        qm = qproc = None
        import os
        kenv = aud.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(kenv, "")
        if not api_key:
            print(f"Set {kenv} for OpenAI backend.")
            sys.exit(1)
        oa_model = aud.get("api_model_name", "gpt-4.1-mini")

    max_tok = int(aud.get("max_output_tokens", 512))
    n_done = 0
    w = out_path.open("w", encoding="utf-8")
    with base.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < args.start:
                continue
            if args.max_lines and n_done >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cot = o.get("raw_cot", "")
            u_text = render_template(user_tmpl, {
                "cot_text": cot,
                "dialogue_id": o.get("dialogue_id", ""),
                "utterance_id": o.get("utterance_id", ""),
                "gold_emotion": o.get("gold_emotion", ""),
                "gender": o.get("gender", ""),
                "target_gender": o.get("target_gender", ""),
                "used_cf_audio": o.get("used_cf_audio", ""),
            })
            try:
                if backend == "openai":
                    raw_a = run_openai(sys_a, u_text, oa_model, api_key, max_tok)
                else:
                    msg = [
                        {"role": "system", "content": [{"type": "text", "text": sys_a}]},
                        {"role": "user", "content": [{"type": "text", "text": u_text}]},
                    ]
                    t_in = qproc.apply_chat_template(
                        msg, add_generation_prompt=True, tokenize=False)
                    # text-only: no media
                    ins = qproc(text=t_in, return_tensors="pt", padding=True).to(qm.device)
                    with __import__("torch").inference_mode():
                        gen = qm.generate(**ins, max_new_tokens=max_tok, do_sample=False,
                                          use_audio_in_video=False, return_audio=False)
                    seq = gen[0] if not hasattr(gen, "sequences") else gen.sequences
                    if hasattr(seq, "shape") and len(seq.shape) == 1:
                        seq = seq.unsqueeze(0)
                    li = ins.input_ids.shape[-1]
                    raw_a = qproc.batch_decode(
                        seq[:, li:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            except Exception as e:
                raw_a = json.dumps({
                    "stereotype_score": 0, "contains_gender_bias": False, "evidence": "",
                    "explanation": f"audit error: {e!s}"})
            try:
                audit = json.loads(raw_a) if raw_a.strip().startswith("{") else _parse_json_loose(raw_a)
            except Exception:
                audit = _parse_json_loose(raw_a)
            o["audit"] = audit
            o["auditor_raw"] = (raw_a[:2000] + "…") if len(raw_a) > 2000 else raw_a
            w.write(json.dumps(o, ensure_ascii=False) + "\n")
            w.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"  audited {n_done} ...")
    w.close()
    with (out_path.parent / "audit_config.yaml").open("w", encoding="utf-8") as cf:
        YAML.safe_dump({
            "backend": backend, "out": str(out_path.relative_to(_ROOT)),
        }, cf, allow_unicode=True)
    print(f"Wrote {n_done} lines -> {out_path}")


if __name__ == "__main__":
    main()
