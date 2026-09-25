#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output_root="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$output_root"
for seed in 20356001 20356002; do
    if [[ "$seed" == 20356001 ]]; then
        modes=(pre_fading post_fading)
    else
        modes=(post_fading pre_fading)
    fi
    for mode in "${modes[@]}"; do
        output="$output_root/channel_feature_job${SLURM_JOB_ID}_seed${seed}_${mode}.json"
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/sweep_dual_receiver_snr.py \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --snr-db=-8.5 --trials 500 --seed "$seed" \
            --noise-reference "$mode" --record-observed-features \
            --output "$output"
    done
done

python3.11 scripts_for_node/softwall_same_gpu/analyze_channel_feature_gate.py \
    --train-pre "$output_root/channel_feature_job${SLURM_JOB_ID}_seed20356001_pre_fading.json" \
    --train-post "$output_root/channel_feature_job${SLURM_JOB_ID}_seed20356001_post_fading.json" \
    --test-pre "$output_root/channel_feature_job${SLURM_JOB_ID}_seed20356002_pre_fading.json" \
    --test-post "$output_root/channel_feature_job${SLURM_JOB_ID}_seed20356002_post_fading.json" \
    --output "$SOFTWALL_ROOT/results/softwall_same_gpu/channel_feature_gate_job${SLURM_JOB_ID}.json"
