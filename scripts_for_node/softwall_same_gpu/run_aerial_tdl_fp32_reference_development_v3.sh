#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

engine_host="$SOFTWALL_ROOT/runtime/softwall_same_gpu/engines/neural_rx_fp32_reference.trt"
build_log="$SOFTWALL_ROOT/runtime/softwall_same_gpu/engines/neural_rx_fp32_reference_python_v3.build.log"
output="$SOFTWALL_ROOT/results/softwall_same_gpu/raw/aerial_tdl_fp32_reference_development_v3_job${SLURM_JOB_ID}.json"

rm -f "$engine_host"
shifter_gpu python3 /softwall/build_nrx_fp32_reference_engine.py \
    --onnx /opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx \
    --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
    >"$build_log" 2>&1

shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/probe_aerial_tdl_tx_contract_development.py \
    --engine /softwall_runtime/engines/neural_rx_fp32_reference.trt \
    --output "$output" \
    --iterations 10 --payload-seed 20357700 --tdl-seed 57701
