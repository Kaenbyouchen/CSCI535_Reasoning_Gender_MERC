#!/bin/bash
# Phase 1 — CoT generation for MELD and IEMOCAP, seven modality settings (14 GPU runs, serial).
# Modalities: text, audio, video, text+audio, text+video, audio+video, text+audio+video
#
# Usage (project root; conda activate csci535):
#   mkdir -p logs
#   nohup bash run_reasoning_phase1_all_modalities_nohup.sh > "logs/phase1_all_mod_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
#
# Env:
#   N_SAMPLES=100 (default)
#   MELD_CFG=yaml/qwen25_MELD_reasoning.yaml
#   IEMO_CFG=yaml/qwen25_IEMOCAP_reasoning.yaml
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

N_SAMPLES="${N_SAMPLES:-100}"
MELD_CFG="${MELD_CFG:-yaml/qwen25_MELD_reasoning.yaml}"
IEMO_CFG="${IEMO_CFG:-yaml/qwen25_IEMOCAP_reasoning.yaml}"

if [ -z "${HF_HOME:-}" ] && [ -d "/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface" ]; then
  export HF_HOME="/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export HF_HUB_CACHE="${HF_HOME}/hub"
fi

MODALITY_ARGS=(text audio video "text audio" "text video" "audio video" "text audio video")

for m in "${MODALITY_ARGS[@]}"; do
  echo ""
  echo "============================================================"
  echo " Phase 1 | MELD | modalities: $m | $(date '+%F %T')"
  echo "============================================================"
  python reasoning/meld_cot_run.py --config "$MELD_CFG" --modalities $m --n-samples "$N_SAMPLES"
  echo ""
  echo "============================================================"
  echo " Phase 1 | IEMOCAP | modalities: $m | $(date '+%F %T')"
  echo "============================================================"
  python reasoning/meld_cot_run.py --config "$IEMO_CFG" --modalities $m --n-samples "$N_SAMPLES"
done

echo ""
echo "Phase 1 all modalities finished: $(date '+%F %T')"
