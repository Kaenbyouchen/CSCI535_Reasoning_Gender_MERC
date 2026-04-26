#!/bin/bash
# Run MELD Task 3 (CoT) multiple times: one run per modality configuration.
# Edit MODALITIES below to add/remove single-modality or combinations.
#
# Usage:
#   conda activate csci535
#   cd <project root>
#   # CoT only (7 GPU runs serially — set aside enough wall time)
#   bash run_reasoning_multi_modalities.sh
#   # Also run OpenAI audit after each CoT (requires OPENAI_API_KEY)
#   RUN_AUDIT=1 bash run_reasoning_multi_modalities.sh
#
# To match flip-aware subsampling with your audio BL/CF runs, set in yaml (paths):
#   predictions_bl_csv, predictions_cf_csv
set -e
cd "$(dirname "$0")"
CONFIG="${CONFIG:-yaml/qwen25_MELD_reasoning.yaml}"

# Each entry: words passed to --modalities (one group per run)
MODALITIES=(
  "text"
  "audio"
  "video"
  "text audio"
  "text video"
  "audio video"
  "text audio video"
)

# Optional: HF / cache (align with your CARC .bashrc)
if [ -z "${HF_HOME}" ] && [ -d "/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface" ]; then
  export HF_HOME="/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export HF_HUB_CACHE="${HF_HOME}/hub"
fi

for m in "${MODALITIES[@]}"; do
  echo ""
  echo "============================================================"
  echo " Modality: $m"
  echo " Started: $(date '+%F %T')"
  echo "============================================================"
  python reasoning/meld_cot_run.py --config "$CONFIG" --modalities $m
  LATEST=$(ls -dt result/qwen25_omni7b_reason_MELD_REASON_* 2>/dev/null | head -1)
  if [ -z "$LATEST" ] || [ ! -f "$LATEST/cot_generations.jsonl" ]; then
    echo "[ERR] no cot_generations.jsonl"
    continue
  fi
  echo "  -> $LATEST"
  if [ "${RUN_AUDIT:-0}" = "1" ]; then
    python reasoning/audit_cot_run.py --cot-jsonl "$LATEST/cot_generations.jsonl" --config "$CONFIG" && \
    python analysis/summarize_reasoning_audit.py --in "$LATEST/cot_audited.jsonl" || true
  fi
done
echo "All runs finished."
