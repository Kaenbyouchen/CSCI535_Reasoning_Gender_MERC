#!/bin/bash
# Phase 2 — Global gender-bias pattern audit: for each modality tag, merge MELD + IEMOCAP
# cot_generations.jsonl (latest matching run dirs) and call OpenAI once per modality.
#
# Requires: OPENAI_API_KEY (or yaml auditor.api_key_env), Phase 1 outputs under result/
#
# Usage:
#   mkdir -p logs
#   nohup bash run_reasoning_phase2_global_audit_nohup.sh > "logs/phase2_global_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
#
# Env:
#   CONFIG=yaml/qwen25_MELD_reasoning.yaml  (auditor.api_model_name + global_pattern_max_output_tokens)
#   GLOBAL_AUDIT_MAX_TOK=4096  (optional; overrides --max-output-tokens when set)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs result/global_bias_audit

CONFIG="${CONFIG:-yaml/qwen25_MELD_reasoning.yaml}"
EXTRA=()
if [ -n "${GLOBAL_AUDIT_MAX_TOK:-}" ]; then
  EXTRA+=(--max-output-tokens "${GLOBAL_AUDIT_MAX_TOK}")
fi

# Tags must match meld_cot_run.py run_dir modality segment (joined with +)
MOD_TAGS=(text audio video "text+audio" "text+video" "audio+video" "text+audio+video")

for tag in "${MOD_TAGS[@]}"; do
  echo ""
  echo "============================================================"
  echo " Phase 2 | global bias audit | modality=${tag} | $(date '+%F %T')"
  echo "============================================================"
  python reasoning/global_bias_audit.py --modality "$tag" --auto-discover --config "$CONFIG" "${EXTRA[@]}"
done

echo ""
echo "Phase 2 global audits finished: $(date '+%F %T')"
