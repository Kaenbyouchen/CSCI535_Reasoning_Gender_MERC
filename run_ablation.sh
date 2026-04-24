#!/bin/bash
# Usage: activate csci535 env first, then run this script.
#   conda activate csci535
#   bash run_ablation.sh
# The job continues after logout (nohup). Log → qwen_iemocap_ablation.log
cd "$(dirname "$0")"
nohup bash ./run_qwen_iemocap_ablation.sh > qwen_iemocap_ablation.log 2>&1 &
echo "Started. PID: $!  Log: $(pwd)/qwen_iemocap_ablation.log"
