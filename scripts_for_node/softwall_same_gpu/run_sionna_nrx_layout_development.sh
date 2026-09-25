#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

for layout in native fortran; do
    output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/sionna_nrx_layout_${layout}_job${SLURM_JOB_ID}.json"
    shifter_gpu env CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH="/softwall_runtime/sionna_deps:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1" \
        python3 /softwall/probe_sionna_nrx_layout_development.py \
        --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
        --output "$output" --layout "$layout" --iterations 20 \
        --payload-seed 20358920 --channel-seed 58921 --esno-db 30.0
done
