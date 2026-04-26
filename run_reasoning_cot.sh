#!/bin/bash
# Single MELD reasoning run (CoT) using yaml/qwen25_MELD_reasoning.yaml, then optional audit.
#
# Usage (after: conda activate csci535, cd project root):
#   bash run_reasoning_cot.sh
#   RUN_AUDIT=1 bash run_reasoning_cot.sh
#
# For OpenAI audit, export OPENAI_API_KEY first.
set -e
cd "$(dirname "$0")"
CONFIG="${CONFIG:-yaml/qwen25_MELD_reasoning.yaml}"
EXTRA=("$@")
python reasoning/meld_cot_run.py --config "$CONFIG" "${EXTRA[@]}"
LATEST=$(ls -dt result/*_REASON_* 2>/dev/null | head -1)
if [ -z "$LATEST" ] || [ ! -f "$LATEST/cot_generations.jsonl" ]; then
  echo "[run_reasoning_cot] No output dir found."
  exit 1
fi
echo "CoT output: $LATEST/cot_generations.jsonl"
if [ "${RUN_AUDIT:-0}" = "1" ]; then
  python reasoning/audit_cot_run.py --cot-jsonl "$LATEST/cot_generations.jsonl" --config "$CONFIG"
  python analysis/summarize_reasoning_audit.py --in "$LATEST/cot_audited.jsonl" 2>/dev/null || true
fi
