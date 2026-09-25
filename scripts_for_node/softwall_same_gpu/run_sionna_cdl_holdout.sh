#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

models=(D E)
for index in 0 1; do
    model="${models[$index]}"
    payload_seed=$((20359000 + index * 1000))
    channel_seed=$((59001 + index * 1000))
    output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/sionna_cdl_${model}_holdout_job${SLURM_JOB_ID}.json"
    shifter_gpu env CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH="/softwall_runtime/sionna_deps:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1" \
        python3 /softwall/probe_sionna_cdl_holdout.py \
        --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
        --output "$output" --model "$model" --iterations-per-snr 50 \
        --payload-seed "$payload_seed" --channel-seed "$channel_seed"
done
