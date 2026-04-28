#!/usr/bin/env python3
"""
Single-panel saliency plot from one per_sample/*.csv (text_input_attribution).

By default, only tokens whose **tokenizer char offsets overlap the dataset
``utterance`` substring inside the rendered user prompt** ``t_in`` (same
rebuild as attribution: yaml + cot jsonl + ``apply_chat_template``). Use
``--full-prompt-tokens`` to include all in-prompt tokens (metadata, headers, …).

Top-K among **letter-containing** subwords (≥2 A–Z), excluding `<|...|>` chat specials,
**English stopwords**, **JSON-ish** tokens (emotion/json/null/…), and common
**interjections / fillers** (oh, um, yeah, …).

Utterance-only mode **loads the HF processor** once for offsets (even if
``--no-hf-decode`` is set; that flag only affects decoded bar labels).

Decode labels: optional HF `--yaml` + `--cot-jsonl` + processor (see --no-hf-decode).

  python analysis/plot_text_attribution_topk.py \\
    --csv result/text_attribution/cot_generations_full/per_sample/0001_d51u1.csv \\
    --k 10 --out result/text_attribution_plots/topk_0001_d51u1.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Letters-only normalization for noise-word filtering.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "as", "of", "at", "by", "for", "from", "in", "into",
    "to", "on", "with", "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "under", "over", "out", "up", "down", "off", "than", "then", "so", "such", "no", "nor",
    "not", "only", "own", "same", "too", "very", "just", "also", "both", "each", "few", "more", "most",
    "other", "some", "any", "all", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "do", "does", "did", "done", "have", "has", "had", "having", "be", "am", "is",
    "are", "was", "were", "been", "being", "it", "its", "this", "that", "these", "those", "i", "me",
    "my", "we", "us", "our", "you", "your", "they", "them", "their", "what", "which", "who", "whom",
    "whose", "where", "when", "why", "how", "there", "here", "again", "once", "ever", "even", "still",
    "yet", "already", "rather", "quite", "per", "via", "across", "within", "without", "among", "toward",
    "towards", "upon", "until", "while", "although", "though", "because", "since", "unless", "whether",
})
_JSONISH_STOP = frozenset({"emotion", "json", "null", "true", "false"})
# Interjections, backchannels, and high-frequency dialogue fluff (single-token match after letter-stripping).
_FILLER_STOP = frozenset({
    "oh", "ooh", "ah", "ahh", "uh", "um", "umm", "hmm", "hm", "huh", "eh", "ha", "heh", "mm", "mmm",
    "wow", "aww", "ugh", "meh", "oops", "ouch", "phew", "shh", "tsk", "tch",
    "yeah", "yep", "yup", "nah", "nope", "naw", "okay", "ok", "oke", "okey", "kay",
    "hey", "hi", "hello", "bye", "thanks", "thank", "thx", "please", "sorry",
    "well", "like", "kinda", "sorta", "really", "actually", "basically", "literally", "totally", "seriously",
    "right", "sure", "guess", "maybe", "perhaps", "anyway", "whatever", "stuff", "thing", "things",
    "gonna", "wanna", "gotta", "dunno", "lemme", "cmon", "comeon",
})
_NOISE_STOPWORDS = _STOPWORDS | _JSONISH_STOP | _FILLER_STOP


def _letters_only_lower(tok: str) -> str:
    t = (tok or "").replace("Ġ", "").replace("▁", "").strip().lower()
    return re.sub(r"[^a-z]", "", t)


def _is_noise_stopword(tok: str) -> bool:
    w = _letters_only_lower(tok)
    if not w:
        return True
    return w in _NOISE_STOPWORDS


def _load_mcf():
    p = _ROOT / "baseline" / "meld_counterfactual_eval.py"
    spec = importlib.util.spec_from_file_location("mcf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _parse_d_u_from_csv_path(path: Path) -> tuple[str, str] | None:
    """Stem like 0001_d51u1 -> ('51','1'); 0045_d207u0 -> ('207','0')."""
    stem = path.stem
    m = re.search(r"_d(\d+)u(\d+)$", stem)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _build_user_text(rec: dict, user_tmpl: str, *, ablate_gender: bool, normalize_emotion, render_template):
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
    return render_template(
        user_tmpl,
        {
            "speaker": spk,
            "dialogue_id": d_id,
            "utterance_id": u_id,
            "utterance": utt,
            "gold_emotion": gold,
            "gender": g or "unknown",
            "target_gender": tg or "n/a",
            "used_cf_audio": ucf,
        },
    )


def _chat_template_string(raw) -> str:
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


def _build_suff(utterance: str) -> str:
    u = (utterance or "").strip()
    return (
        f"\n### 1) Text\n{u}\n\n"
        "### 2) Acoustic\nNot provided.\n\n"
        "### 3) Integrate\nN/A\n\n"
        '### 4) Final\n{"emotion": "'
    )


def _short_special(tok: str) -> str:
    t = tok.strip()
    aliases = {
        "<|im_start|>": "[im_start]",
        "<|im_end|>": "[im_end]",
        "<|endoftext|>": "[eos]",
    }
    return aliases.get(t, t if len(t) < 28 else (t[:12] + "…"))


def _fmt_raw_piece(s: str) -> str:
    return (s or "").replace("Ġ", "·").replace("▁", "·").strip() or "·"


def _is_chat_special_token(tok: str) -> bool:
    t = (tok or "").strip()
    return "<|" in t or "|>" in t


def _letter_count(tok: str) -> int:
    t = (tok or "").replace("Ġ", "").replace("▁", "")
    return len(re.findall(r"[A-Za-z]", t))


def _utterance_char_spans_in_tin(t_in: str, utterance: str) -> list[tuple[int, int]]:
    """All [start, end) ranges where ``utterance`` occurs as a substring of ``t_in``."""
    u = utterance or ""
    if not u:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        i = t_in.find(u, start)
        if i < 0:
            break
        spans.append((i, i + len(u)))
        start = i + 1
    return spans


def _offset_pairs_from_encoding(enc, *, expected_len: int) -> list[tuple[int, int]] | None:
    """Turn BatchEncoding offset_mapping into ``expected_len`` ``(start, end)`` pairs (Qwen layouts vary)."""
    om_raw = enc.get("offset_mapping")
    if om_raw is None:
        return None
    if hasattr(om_raw, "tolist"):
        om_raw = om_raw.tolist()
    if not isinstance(om_raw, (list, tuple)) or not om_raw:
        return None

    def _each_is_pair_row(seq: list) -> bool:
        return bool(seq) and all(
            isinstance(x, (list, tuple))
            and len(x) == 2
            and all(isinstance(t, int) for t in x)
            for x in seq
        )

    L = int(expected_len)
    cands: list[list] = []
    if _each_is_pair_row(list(om_raw)):
        cands.append(list(om_raw))
    if isinstance(om_raw[0], (list, tuple)) and _each_is_pair_row(list(om_raw[0])):
        cands.append(om_raw[0])

    for seq in cands:
        if not seq:
            continue
        if isinstance(seq[0], int):
            if len(seq) != 2 * L:
                continue
            return [(int(seq[i]), int(seq[i + 1])) for i in range(0, len(seq), 2)]
        om: list[tuple[int, int]] = []
        bad = False
        for it in seq:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                om.append((int(it[0]), int(it[1])))
            elif hasattr(it, "tolist"):
                t = it.tolist()
                if isinstance(t, (list, tuple)) and len(t) >= 2:
                    om.append((int(t[0]), int(t[1])))
                else:
                    bad = True
                    break
            else:
                bad = True
                break
        if not bad and len(om) == L:
            return om
    return None


def _token_indices_overlapping_spans(
    offset_row: list[tuple[int, int]],
    spans: list[tuple[int, int]],
    *,
    clip_end: int,
) -> frozenset[int]:
    """Token i kept iff its char span overlaps any ``spans`` and starts before ``clip_end`` (len(t_in))."""
    kept: set[int] = set()
    for i, (a, b) in enumerate(offset_row):
        if a >= clip_end or b <= a:
            continue
        for s, e in spans:
            if a < e and b > s:
                kept.add(i)
                break
    return frozenset(kept)


def _build_t_in_full_from_jsonl(
    *,
    yaml_path: Path,
    jsonl_path: Path,
    dialogue_id: str,
    utterance_id: str,
    hf_model: str,
):
    """Return (processor, t_in, full, utterance, err). err set if json row missing."""
    mcf = _load_mcf()
    read_yaml = mcf.read_yaml
    render_template = mcf.render_template
    normalize_emotion = mcf.normalize_emotion

    cfg = read_yaml(yaml_path if yaml_path.is_absolute() else _ROOT / yaml_path)
    prompt = cfg.get("prompt", {})
    system_cot = prompt.get("system_cot", "")
    user_tmpl = prompt.get("user_cot_template", "")

    rec = None
    jp = jsonl_path if jsonl_path.is_absolute() else _ROOT / jsonl_path
    with jp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if str(o.get("dialogue_id", "")).strip() == dialogue_id and str(
                o.get("utterance_id", "")
            ).strip() == utterance_id:
                rec = o
                break
    if rec is None:
        return None, "", "", "", "no matching jsonl row"

    user_text = _build_user_text(rec, user_tmpl, ablate_gender=False, normalize_emotion=normalize_emotion, render_template=render_template)
    utt = str(rec.get("utterance", "")).strip()
    convo = [
        {"role": "system", "content": [{"type": "text", "text": system_cot}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    from transformers import Qwen2_5OmniProcessor

    proc = Qwen2_5OmniProcessor.from_pretrained(hf_model)
    t_in = _chat_template_string(
        proc.apply_chat_template(convo, add_generation_prompt=True, tokenize=False),
    )
    full = t_in + _build_suff(utt)
    return proc, t_in, full, utt, ""


def _alignment_input_ids_and_utterance_mask(
    *,
    yaml_path: Path,
    jsonl_path: Path,
    dialogue_id: str,
    utterance_id: str,
    hf_model: str,
) -> tuple[list[int] | None, frozenset[int] | None, str]:
    """``(input_ids, utterance_token_indices, err)``. utterance set empty if text not found in ``t_in``."""
    built = _build_t_in_full_from_jsonl(
        yaml_path=yaml_path,
        jsonl_path=jsonl_path,
        dialogue_id=dialogue_id,
        utterance_id=utterance_id,
        hf_model=hf_model,
    )
    proc, t_in, full, utt, err = built
    if proc is None or err:
        return None, None, err or "build prompt failed"
    if not (utt or "").strip():
        return None, None, "empty utterance in jsonl row"
    tok = proc.tokenizer
    try:
        enc = tok(full, add_special_tokens=True, return_offsets_mapping=True)
    except (TypeError, ValueError) as e:
        return None, None, f"tokenizer encode failed: {e}"
    ids = enc["input_ids"]
    if hasattr(ids, "tolist") and callable(getattr(ids, "tolist", None)):
        raw = ids.tolist()
        if raw and isinstance(raw[0], list):
            ids_list = [int(x) for x in raw[0]]
        else:
            ids_list = [int(x) for x in raw]
    elif isinstance(ids, (list, tuple)):
        if ids and isinstance(ids[0], (list, tuple)):
            ids_list = [int(x) for x in ids[0]]
        else:
            ids_list = [int(x) for x in ids]
    else:
        ids_list = [int(x) for x in ids[0]]
    L_ids = len(ids_list)
    om = _offset_pairs_from_encoding(enc, expected_len=L_ids)
    if om is None:
        return None, None, "could not parse offset_mapping to match input_ids length"
    spans = _utterance_char_spans_in_tin(t_in, utt)
    if not spans:
        return ids_list, frozenset(), ""
    t_split = len(t_in)
    mask = _token_indices_overlapping_spans(om, spans, clip_end=t_split)
    return ids_list, mask, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "result/text_attribution/cot_generations_full/per_sample/0001_d51u1.csv",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--yaml",
        type=Path,
        default=_ROOT / "yaml/qwen25_MELD_reasoning.yaml",
        help="Same as text_input_attribution --config (for decode labels)",
    )
    ap.add_argument(
        "--cot-jsonl",
        type=Path,
        default=_ROOT / "result/qwen25_omni7b_reason_MELD_REASON_text_20260426_165650/cot_generations.jsonl",
        help="Same cot_generations.jsonl used for the attribution run",
    )
    ap.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="HF id for Qwen2_5OmniProcessor (default: read from yaml model.hf_model_name)",
    )
    ap.add_argument(
        "--no-hf-decode",
        action="store_true",
        help="Bar labels use raw SentencePiece only (no decode([id])). Utterance-only mode still loads HF once for offset alignment.",
    )
    ap.add_argument(
        "--full-prompt-tokens",
        action="store_true",
        help="Plot saliency over all in-prompt tokens (default: only tokens overlapping the dataset utterance text inside the user prompt t_in).",
    )
    ap.add_argument(
        "--legacy-global-topk",
        action="store_true",
        help="Old behavior: single panel, global top-K (often only specials / BPE junk)",
    )
    args = ap.parse_args()

    p = args.csv if args.csv.is_absolute() else _ROOT / args.csv
    if not p.is_file():
        raise SystemExit(f"Missing: {p}")

    rows = []
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                sal = float(row["saliency"])
                in_p = float(row["in_prompt"])
            except (KeyError, ValueError):
                continue
            if in_p <= 0.5:
                continue
            tok = row.get("token", "")
            rows.append(
                {
                    "token_index": int(row["token_index"]),
                    "token": tok,
                    "saliency": sal,
                },
            )
    if not rows:
        raise SystemExit("No in_prompt rows in CSV.")

    parsed = _parse_d_u_from_csv_path(p)
    yaml_p = args.yaml if args.yaml.is_absolute() else _ROOT / args.yaml
    jsonl_p = args.cot_jsonl if args.cot_jsonl.is_absolute() else _ROOT / args.cot_jsonl
    hf_name = args.hf_model.strip()
    if not hf_name and yaml_p.is_file():
        cfg0 = _load_mcf().read_yaml(yaml_p)
        hf_name = str((cfg0.get("model") or {}).get("hf_model_name", "")).strip()

    ids_list: list[int] | None = None
    utter_mask: frozenset[int] | None = None
    decode_err = ""
    utter_warn = ""

    can_align = bool(parsed and yaml_p.is_file() and jsonl_p.is_file() and hf_name)

    if not args.full_prompt_tokens:
        if not can_align:
            raise SystemExit(
                "Default: only dataset utterance tokens. Need CSV stem *_d<dialogue>u<utt>.csv plus "
                "--yaml, --cot-jsonl, and hf_model_name (or --hf-model). "
                "Or use --full-prompt-tokens to skip utterance alignment.",
            )
        try:
            ids_list, utter_mask, decode_err = _alignment_input_ids_and_utterance_mask(
                yaml_path=yaml_p,
                jsonl_path=jsonl_p,
                dialogue_id=parsed[0],
                utterance_id=parsed[1],
                hf_model=hf_name,
            )
        except Exception as e:
            raise SystemExit(f"HF alignment failed: {e}") from e
        if decode_err:
            raise SystemExit(decode_err)
        if utter_mask is not None and len(utter_mask) == 0:
            utter_warn = "utterance text not found in rendered t_in (0 overlapping tokens)"
        rows = [r for r in rows if utter_mask is not None and r["token_index"] in utter_mask]
        if not rows:
            raise SystemExit(
                "No CSV rows overlap the dataset utterance span (check yaml/jsonl match the attribution run).",
            )
    elif can_align and not args.no_hf_decode:
        try:
            ids_list, _, decode_err = _alignment_input_ids_and_utterance_mask(
                yaml_path=yaml_p,
                jsonl_path=jsonl_p,
                dialogue_id=parsed[0],
                utterance_id=parsed[1],
                hf_model=hf_name,
            )
            if decode_err:
                ids_list = None
        except Exception as e:
            decode_err = str(e)
            ids_list = None
        if ids_list is None and decode_err == "":
            decode_err = "alignment failed"
    else:
        decode_err = "(--no-hf-decode)" if args.no_hf_decode else "missing yaml/jsonl/hf_model or parse failed"

    # Avoid loading processor twice per label — load once if ids_list ok
    _proc = None

    def label_for_fast(r: dict) -> str:
        nonlocal _proc
        idx = r["token_index"]
        raw = r["token"]
        if ids_list is not None and 0 <= idx < len(ids_list):
            if _proc is None:
                from transformers import Qwen2_5OmniProcessor

                yp = args.yaml if args.yaml.is_absolute() else _ROOT / args.yaml
                cfg = _load_mcf().read_yaml(yp)
                name = args.hf_model.strip() or str(cfg["model"]["hf_model_name"])
                _proc = Qwen2_5OmniProcessor.from_pretrained(name)
            tid = int(ids_list[idx])
            frag = _proc.tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "\\n")
            if not frag.strip():
                frag = _short_special(raw)
            else:
                frag = repr(frag)[1:-1]
            return f"[{idx}] «{frag}»  |  raw:{_fmt_raw_piece(raw)}"
        piece = _short_special(raw) if raw.strip().startswith("<|") else _fmt_raw_piece(raw)
        return f"[{idx}] {piece}"

    def _plot_one_panel(ax, items: list[dict], title: str, *, fontsize: int = 8) -> None:
        if not items:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return
        labels = [label_for_fast(r) for r in reversed(items)]
        vals = [r["saliency"] for r in reversed(items)]
        colors = ["#4c72b0"] * len(items)
        y = range(len(items))
        ax.barh(list(y), vals, color=colors, height=0.72)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=fontsize)
        ax.set_xlabel("Saliency")
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color="k", linewidth=0.4)

    out = args.out
    if out is None:
        out_dir = _ROOT / "result" / "text_attribution_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"topk_{p.stem}.png"
    else:
        out = out if out.is_absolute() else _ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("pip install matplotlib") from e

    sub = f"{p.name}"
    if not args.full_prompt_tokens:
        sub += "  |  tokens ⊆ dataset utterance (char span in t_in)"
    if utter_warn:
        sub += f"  |  {utter_warn}"
    if ids_list is not None and not args.no_hf_decode:
        sub += "  |  «…» = decode([id]); raw = SentencePiece"
    elif args.no_hf_decode:
        sub += "  |  raw labels (--no-hf-decode)"
    else:
        sub += f"  |  decode: {decode_err}"

    if args.legacy_global_topk:
        rows.sort(key=lambda r: r["saliency"], reverse=True)
        top = rows[: max(1, args.k)]
        fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
        _plot_one_panel(ax, top, f"Legacy: global top-{len(top)} by saliency\n{sub}")
    else:
        wordish = [
            r
            for r in rows
            if (not _is_chat_special_token(r["token"]))
            and _letter_count(r["token"]) >= 2
            and (not _is_noise_stopword(r["token"]))
        ]
        wordish.sort(key=lambda r: r["saliency"], reverse=True)
        top_w = wordish[: max(1, args.k)]

        fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
        title_extra = " (dataset utterance only)" if not args.full_prompt_tokens else ""
        _plot_one_panel(
            ax,
            top_w,
            f"Top-{len(top_w)} saliency{title_extra}: ≥2 letters, no <|special|>, no stop/filler/JSON-ish\n{sub}",
            fontsize=7,
        )

    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out}")
    if utter_warn:
        print(f"[WARN] {utter_warn}", flush=True)
    if ids_list is None and args.full_prompt_tokens and not args.no_hf_decode:
        print(f"[WARN] HF decode skipped: {decode_err or '(unknown)'}", flush=True)


if __name__ == "__main__":
    main()
