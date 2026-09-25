#!/usr/bin/env bash

set -euo pipefail

cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
result="$PWD/results/softwall_same_gpu"
scripts="$PWD/scripts_for_node/softwall_same_gpu"
mkdir -p "$result"

bash "$scripts/run_confirm39_clean_lifecycle.sh" \
    2>&1 | tee "$result/confirm39_execution.log"
bash "$scripts/run_confirm37_background_smoke.sh" \
    2>&1 | tee "$result/confirm37_execution.log"
bash "$scripts/run_confirm40_cold_start_ablation.sh" \
    2>&1 | tee "$result/confirm40_execution.log"
bash "$scripts/run_confirm38_background_comparison.sh" \
    2>&1 | tee "$result/confirm38_execution.log"

echo 'confirm39, confirm37, confirm40, confirm38 follow-on pipeline completed'
