#!/bin/bash
cd "$(dirname "$0")"
python baseline/iemocap_eval.py \
    --config yaml/qwen25_IEMOCAP.yaml \
    --modalities text audio \
    --max-samples 30
