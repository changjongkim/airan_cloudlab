#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

output="$SOFTWALL_ROOT/results/softwall_multigpu/c167_persistent_conventional_shadow_job${SLURM_JOB_ID}.json"
shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_TASK1:/softwall_task1" \
    --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
    --volume="$(dirname "$output"):/results" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    --env=CUDA_VISIBLE_DEVICES=0 \
    python3 /softwall/c167_persistent_conventional_shadow.py \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "/results/$(basename "$output")" \
        --seed 20359400 --warmup 20 --iterations 300
