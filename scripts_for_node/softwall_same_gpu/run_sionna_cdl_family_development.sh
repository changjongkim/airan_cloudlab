#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

arms=(A:1 A:10 A:30 A:100 B:100 C:100 D:100 E:100)
index=0
for arm in "${arms[@]}"; do
    model="${arm%%:*}"
    delay="${arm##*:}"
    seed=$((58961 + index))
    output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/sionna_cdl_${model}_delay${delay}ns_development_job${SLURM_JOB_ID}.json"
    shifter_gpu env CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH="/softwall_runtime/sionna_deps:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1" \
        python3 /softwall/probe_sionna_cdl_family_development.py \
        --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
        --output "$output" --model "$model" --delay-spread-ns "$delay" \
        --iterations 10 --payload-seed 20358960 --channel-seed "$seed" \
        --esno-db 30.0
    index=$((index + 1))
done
