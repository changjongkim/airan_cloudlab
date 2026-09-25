#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

profiles=(
    reference_mcs7_stream1_tx1_direct
    reference_mcs7_stream1_tx1_wrapper
    reference_mcs7_stream4_tx1_direct
)

for profile in "${profiles[@]}"; do
    output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/nrx_reference_clean_${profile}_attempt3_job${SLURM_JOB_ID}.json"
    shifter_gpu env CUDA_VISIBLE_DEVICES=0 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/probe_nrx_reference_clean_matrix.py \
        --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
        --output "$output" --iterations 10 --seed 20358900 \
        --profile-name "$profile"
done
