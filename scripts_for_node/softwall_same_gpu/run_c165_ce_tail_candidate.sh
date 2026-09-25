#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

if [[ $# -ne 4 ]]; then
    echo "usage: $0 INPUT_MODE SEED ITERATIONS RUN_ROLE" >&2
    exit 2
fi

input_mode="$1"
seed="$2"
iterations="$3"
run_role="$4"
case "$input_mode" in
    persistent_sync|persistent_ordered) ;;
    *) echo "unsupported C165 input mode: $input_mode" >&2; exit 2 ;;
esac
if [[ ! "$seed" =~ ^[0-9]+$ || ! "$iterations" =~ ^[0-9]+$ ]]; then
    echo "seed and iterations must be positive integers" >&2
    exit 2
fi
case "$run_role" in
    ablation|development|holdout) ;;
    *) echo "unsupported C165 run role: $run_role" >&2; exit 2 ;;
esac

label="c165_ce_tail_${input_mode}_${run_role}_seed${seed}_job${SLURM_JOB_ID}"
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
controller_output="$result_root/raw/${label}_controller.json"
worker_output="$result_root/raw/${label}_worker.json"
controller_log="$result_root/raw/${label}_controller.log"
worker_log="$result_root/raw/${label}_worker.log"
tag="${label}_${BASHPID}"
mkdir -p "$result_root/raw" "$state"
rm -f "$state"/cuda_ipc_* "$controller_output" "$worker_output" \
    "$controller_log" "$worker_log"

shifter_p2p() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_VISIBLE_DEVICES=0,1 "$@"
}

controller_pid=""
worker_pid=""
cleanup() {
    [[ -z "$controller_pid" ]] || kill "$controller_pid" 2>/dev/null || true
    [[ -z "$worker_pid" ]] || kill "$worker_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

shifter_p2p python3 /softwall/c163_raw_p2p_controller.py \
    --tag "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$controller_output" --warmup 100 --iterations "$iterations" \
    --deadline-ms 4.5 --seed "$seed" --cpu-affinity-index 0 \
    >"$controller_log" 2>&1 &
controller_pid=$!

shifter_p2p python3 /softwall/c163_raw_p2p_nrx_worker.py \
    --tag "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$worker_output" --seed "$seed" \
    --source-device 0 --destination-device 1 \
    --same-stream --busy-poll --cpu-affinity-index -1 \
    --input-mode "$input_mode" \
    >"$worker_log" 2>&1 &
worker_pid=$!

wait "$controller_pid"
controller_pid=""
wait "$worker_pid"
worker_pid=""
trap - EXIT INT TERM

echo "C165 CE-tail candidate complete: $controller_output"
