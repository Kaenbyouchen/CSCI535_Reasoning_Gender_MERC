#!/bin/bash
set -e
cd "$(dirname "$0")"

EXTRA_ARGS="$@"

echo "======================================================"
echo " Qwen2.5-Omni IEMOCAP Evaluation (Session 5, 7-class)"
echo " Modalities: text / audio / text+audio / text+audio+video"
echo "======================================================"

echo ""
echo "[1/4] text"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities text \
    $EXTRA_ARGS

echo ""
echo "[2/4] audio"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities audio \
    $EXTRA_ARGS

echo ""
echo "[3/4] text + audio"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities text audio \
    $EXTRA_ARGS

echo ""
echo "[4/4] text + audio + video"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities text audio video \
    $EXTRA_ARGS

echo ""
echo "======================================================"
echo " All done. Summary → result/IEMOCAP_Summary.csv"
echo "======================================================"
