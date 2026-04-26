#!/bin/bash
# CoT reasoning: MELD then IEMOCAP, modalities text + audio only.
# Default: 100 stratified samples per dataset (--n-samples; same logic as yaml subsample + meld_cot_run.py).
#
# From project root (e.g. on CARC after conda activate csci535):
#   mkdir -p logs
#   nohup bash run_reasoning_full_text_audio_nohup.sh > logs/reasoning_ta_n100_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f logs/reasoning_ta_n100_*.log
#
# Override count:
#   N_SAMPLES=200 nohup bash run_reasoning_full_text_audio_nohup.sh > logs/... 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

N_SAMPLES="${N_SAMPLES:-100}"
MELD_CFG="${MELD_CFG:-yaml/qwen25_MELD_reasoning.yaml}"
IEMO_CFG="${IEMO_CFG:-yaml/qwen25_IEMOCAP_reasoning.yaml}"

if [ -z "${HF_HOME:-}" ] && [ -d "/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface" ]; then
  export HF_HOME="/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface"
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export HF_HUB_CACHE="${HF_HOME}/hub"
fi

echo "=== MELD reasoning (text+audio), n_samples=${N_SAMPLES} ==="
echo "Started: $(date '+%F %T')"
python reasoning/meld_cot_run.py --config "$MELD_CFG" --modalities text audio --n-samples "$N_SAMPLES"
echo "MELD finished: $(date '+%F %T')"

echo ""
echo "=== IEMOCAP reasoning (text+audio), n_samples=${N_SAMPLES} ==="
echo "Started: $(date '+%F %T')"
python reasoning/meld_cot_run.py --config "$IEMO_CFG" --modalities text audio --n-samples "$N_SAMPLES"
echo "IEMOCAP finished: $(date '+%F %T')"
echo "All done."
