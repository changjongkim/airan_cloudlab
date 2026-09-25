#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/aerial_cdl_reference_development_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/probe_aerial_cdl_reference_development.py \
    --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
    --output "$output" --iterations 20 \
    --payload-seed 20358500 --channel-seed 58501
