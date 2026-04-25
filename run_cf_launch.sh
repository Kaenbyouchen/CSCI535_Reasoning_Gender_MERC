#!/bin/bash
# run_cf_launch.sh — nohup launcher for run_counterfactual.sh.
#
# Usage:
#   conda activate csci535
#   bash run_cf_launch.sh
#
# The job survives logout. Log → qwen_meld_cf.log
cd "$(dirname "$0")"
nohup bash ./run_counterfactual.sh > qwen_meld_cf.log 2>&1 &
echo "Started. PID: $!  Log: $(pwd)/qwen_meld_cf.log"
echo "Tail with: tail -f $(pwd)/qwen_meld_cf.log"
