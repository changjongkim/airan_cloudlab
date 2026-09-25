#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output_dir="$SOFTWALL_ROOT/results/softwall_multigpu/c166_native_fixture_seed20359400"
mkdir -p "$output_dir"

shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_TASK1:/softwall_task1" \
    --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
    --volume="$output_dir:/fixture" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    --env=CUDA_VISIBLE_DEVICES=0 \
    python3 /softwall/export_c166_native_fixture.py \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output-dir /fixture --seed 20359400 --device 0

echo "C166 native parity fixture exported: $output_dir/manifest.json"
