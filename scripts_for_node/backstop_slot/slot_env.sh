#!/usr/bin/env bash
# Shared environment for slot-scale Antiphase runs. Source from an allocation.

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

readonly SLOT_SCRIPTS="$SOFTWALL_ROOT/scripts_for_node/backstop_slot"
readonly SLOT_DATA="$SOFTWALL_ROOT/run_state/backstop_slot/dataset_v1"
readonly SLOT_RESULTS="$SOFTWALL_ROOT/results/backstop_slot"

shifter_slot() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --volume="$SLOT_SCRIPTS:/backstop_slot" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/backstop_slot:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=0,1,2,3 "$@"
}
