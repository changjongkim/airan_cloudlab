#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output="$SOFTWALL_ROOT/results/softwall_multigpu/raw/c163_speculative_parallel_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/probe_c163_speculative_parallel.py \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$output" --warmup 100 --iterations 1000 \
    --seed 20357900 --deadline-ms 4.5
