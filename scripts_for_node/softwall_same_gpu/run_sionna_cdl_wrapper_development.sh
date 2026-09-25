#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/sionna_cdl_wrapper_development_job${SLURM_JOB_ID}.json"
shifter_gpu env \
    PYTHONPATH="/softwall_runtime/sionna_deps:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1" \
    CUDA_VISIBLE_DEVICES=0 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/probe_sionna_cdl_official_snr_development.py \
    --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
    --output "$output" --iterations-per-snr 100 \
    --payload-seed 20358800 --channel-seed 58801 --neural-path wrapper
