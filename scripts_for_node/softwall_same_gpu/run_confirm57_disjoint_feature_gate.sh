#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output_root="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$output_root"
for seed in 20456001 20457001; do
    if [[ "$seed" == 20456001 ]]; then
        modes=(pre_fading post_fading)
    else
        modes=(post_fading pre_fading)
    fi
    for mode in "${modes[@]}"; do
        output="$output_root/confirm57_feature_job${SLURM_JOB_ID}_seed${seed}_${mode}.json"
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/sweep_dual_receiver_snr.py \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --snr-db=-8.5 --trials 500 --seed "$seed" \
            --noise-reference "$mode" --record-observed-features \
            --prewarm-observed-features --output "$output"
    done
done

python3.11 scripts_for_node/softwall_same_gpu/analyze_confirm57_channel_gate.py \
    --root "$SOFTWALL_ROOT" --job "$SLURM_JOB_ID"
