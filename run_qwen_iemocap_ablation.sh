#!/bin/bash
set -e
cd "$(dirname "$0")"

EXTRA_ARGS="$@"

echo "======================================================"
echo " Qwen2.5-Omni IEMOCAP Ablation (Session 5, 7-class)"
echo " Modalities to add: video (muted) / audio+video / text+video"
echo " Note: video uses use_audio_in_video=false (muted)"
echo "======================================================"

echo ""
echo "[1/3] video only (muted)"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities video \
    $EXTRA_ARGS

echo ""
echo "[2/3] audio + video"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities audio video \
    $EXTRA_ARGS

echo ""
echo "[3/3] text + video"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities text video \
    $EXTRA_ARGS

echo ""
echo "======================================================"
echo " Ablation done. Summary appended → result/IEMOCAP_Summary.csv"
echo "======================================================"
