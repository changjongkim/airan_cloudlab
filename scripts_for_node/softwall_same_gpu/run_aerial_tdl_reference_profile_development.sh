#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/aerial_tdl_reference_profile_development_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/probe_aerial_tdl_reference_profile_development.py \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$output" \
    --iterations 10 --payload-seed 20357300 --tdl-seed 57301
