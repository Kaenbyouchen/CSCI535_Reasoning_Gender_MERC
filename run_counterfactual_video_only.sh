#!/bin/bash
# run_counterfactual_video_only.sh — re-run only the 4 video cells with the new
# fps/max_pixels caps. Used after the first run hit OOM on long clips.
#
# Cells:
#   BL × {audio+video, text+audio+video}                (original audio)
#   CF × {audio+video, text+audio+video}                (gender-swapped audio)
#
# Usage (after `conda activate csci535`):
#   bash run_counterfactual_video_only.sh
#
# Estimated runtime on 1× A40: ~3-4h total (~2156 utterances per cell, video subsampled to 1 fps).

set -e
cd "$(dirname "$0")"

# Make sure HF cache does NOT live on /home1 (quota too small).
if [ -z "${HF_HOME}" ]; then
  for cand in \
      "/project2/msoleyma_1026/RuiZhang/hf_cache/huggingface" \
      "/scratch1/${USER}/hf_cache/huggingface"; do
    if [ -d "${cand}" ]; then
      export HF_HOME="${cand}"
      break
    fi
  done
fi
if [ -n "${HF_HOME}" ]; then
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  echo "[env] HF_HOME=${HF_HOME}"
fi

CONFIG="yaml/qwen25_MELD_counterfactual.yaml"
EXTRA_ARGS="$@"

run_cell () {
  local SOURCE="$1"; shift
  local LABEL="$1"; shift
  echo ""
  echo "======================================================"
  echo " Cell: ${SOURCE} | ${LABEL}    (modalities: $@)"
  echo " Started at: $(date '+%F %T')"
  echo "======================================================"
  python baseline/meld_counterfactual_eval.py \
      --config "${CONFIG}" \
      --audio-source "${SOURCE}" \
      --modalities "$@" \
      ${EXTRA_ARGS}
}

echo "==> 4 video cells will run serially (BL video×2 + CF video×2)."

run_cell original       BL_audio_video         audio video
run_cell original       BL_text_audio_video    text audio video
run_cell counterfactual CF_audio_video         audio video
run_cell counterfactual CF_text_audio_video    text audio video

echo ""
echo "======================================================"
echo " All 4 video cells done. Rebuild summary."
echo "======================================================"
python analysis/build_counterfactual_summary.py \
    --result-dir result \
    --out-csv result/MELD_Counterfactual_Summary.csv
echo ""
echo "Summary at: result/MELD_Counterfactual_Summary.csv"
