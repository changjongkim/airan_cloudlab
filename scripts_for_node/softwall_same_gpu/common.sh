#!/usr/bin/env bash

# This file is sourced by both strict runners and an interactive allocation.
# Callers choose their own shell error policy so a failed probe cannot release
# the allocation shell itself.

readonly SOFTWALL_ROOT=/pscratch/sd/s/sgkim/kcj/airan_cloudlab
readonly AERIAL_REPO="$SOFTWALL_ROOT/third_party/aerial-cuda-accelerated-ran"
readonly AERIAL_IMAGE=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
readonly SOFTWALL_SCRIPTS="$SOFTWALL_ROOT/scripts_for_node/softwall_same_gpu"
readonly SOFTWALL_TASK1="$SOFTWALL_ROOT/scripts_for_node/task1"
readonly SOFTWALL_RUNTIME="$SOFTWALL_ROOT/runtime/softwall_same_gpu"
readonly GPU_LD_PATH=/opt/udiImage/modules/gpu/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib

require_allocation() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || {
        echo "SoftWall runner requires an active Slurm allocation" >&2
        return 2
    }
    [[ "$PWD" == "$SOFTWALL_ROOT" || "$PWD" == "$SOFTWALL_ROOT"/* ]] || {
        echo "working directory must stay under $SOFTWALL_ROOT" >&2
        return 2
    }
}

shifter_gpu() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --volume="$SOFTWALL_ROOT/data/public/burstgpt:/softwall_burstgpt" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=SOFTWALL_NRX_ENGINE=/softwall_runtime/engines/neural_rx_fp16_full.trt \
        --env=CUDA_VISIBLE_DEVICES=0 \
        "$@"
}

shifter_qwen() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONNOUSERSITE=1 \
        --env=PYTHONPATH=/softwall_runtime/python:/softwall \
        --env=HF_HOME=/softwall_runtime/cache/huggingface \
        --env=TMPDIR=/softwall_runtime/tmp \
        --env=HF_HUB_DISABLE_XET=1 \
        --env=CUDA_VISIBLE_DEVICES=0 \
        "$@"
}
