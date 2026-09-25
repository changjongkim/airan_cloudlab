#!/usr/bin/env bash

# C153: global certificate -> Qwen lease/fence -> shared cuPHY/P2P recovery.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C153_LABEL:-confirm153_integrated_shared_recovery_job${SLURM_JOB_ID}}
warmup=${SOFTWALL_C153_WARMUP:-10}
seed0=${SOFTWALL_C153_SEED0:-25300051}
seed1=${SOFTWALL_C153_SEED1:-25310051}
channel0=${SOFTWALL_C153_CHANNEL0:-25301001}
channel1=${SOFTWALL_C153_CHANNEL1:-25311001}
snr_db=${SOFTWALL_C153_SNR_DB:-20.0}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
result="$result_root/${label}_result.json"
owner0="$raw/${label}_owner0.json"
owner1="$raw/${label}_owner1.json"
worker="$raw/${label}_worker.json"
qwen="$raw/${label}_qwen.json"
inventory="$raw/${label}_gpu_inventory.csv"
release_file="$state/release.json"
qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c153.sock"
mkdir -p "$raw" "$state" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$state"/cuda_ipc_* "$release_file" "$qwen_socket" \
    "$owner0" "$owner1" "$worker" "$qwen" "$result"

nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

python3.11 "$SOFTWALL_SCRIPTS/build_confirm153_protocol.py" \
    --output "$protocol" --label "$label" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --receiver-seed0 "$seed0" --receiver-seed1 "$seed1" \
    --channel-seed0 "$channel0" --channel-seed1 "$channel1" \
    --snr-db "$snr_db" --warmup "$warmup"

shifter_all() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=0,1,2 "$@"
}

shifter_qwen_gpu2() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/softwall \
        --env=HF_HOME=/softwall_runtime/cache/huggingface \
        --env=TMPDIR=/softwall_runtime/tmp --env=HF_HUB_DISABLE_XET=1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=2 "$@"
}

pids=""
cleanup() {
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    for pid in $pids; do wait "$pid" 2>/dev/null || true; done
    rm -f "$qwen_socket"
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
export SOFTWALL_MPS_GPU=0,1,2
softwall_mps_start
softwall_mps_assert

shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/trace_qwen_worker.py \
    --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 64 \
    --batch-size 1 --warmup-per-length 3 \
    --socket "$qwen_socket" --output "$qwen" &
qwen_pid=$!
pids="$pids $qwen_pid"

for _ in {1..3600}; do
    [[ -S "$qwen_socket" ]] && break
    if ! kill -0 "$qwen_pid" 2>/dev/null; then
        wait "$qwen_pid" || true
        echo "C153 Qwen worker exited before readiness" >&2
        exit 1
    fi
    sleep 0.05
done
[[ -S "$qwen_socket" ]] || { echo "C153 Qwen readiness timeout" >&2; exit 1; }

tag="${label}_${BASHPID}"
shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/integrated_shared_recovery_owner.py \
    --home-id 0 --request-id h0-r2 --source-device 0 --destination-device 2 \
    --tag "${tag}_h0" --ipc-dir "$state" --release-file "$release_file" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt --output "$owner0" \
    --receiver-seed "$seed0" --channel-seed "$channel0" --snr-db "$snr_db" &
pids="$pids $!"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/integrated_shared_recovery_owner.py \
    --home-id 1 --request-id h1-r0 --source-device 1 --destination-device 2 \
    --tag "${tag}_h1" --ipc-dir "$state" --release-file "$release_file" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt --output "$owner1" \
    --receiver-seed "$seed1" --channel-seed "$channel1" --snr-db "$snr_db" &
pids="$pids $!"

for _ in {1..3600}; do
    if [[ -f "$state/cuda_ipc_${tag}_h0.info" \
        && -f "$state/cuda_ipc_${tag}_h1.info" ]]; then
        break
    fi
    sleep 0.05
done
[[ -f "$state/cuda_ipc_${tag}_h0.info" \
    && -f "$state/cuda_ipc_${tag}_h1.info" ]] || {
    echo "C153 owners did not publish IPC handles" >&2
    exit 1
}

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/integrated_shared_recovery_worker.py \
    --tag0 "${tag}_h0" --tag1 "${tag}_h1" \
    --ipc-dir "$state" --release-file "$release_file" --qwen-socket "$qwen_socket" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt --output "$worker" \
    --source-device0 0 --source-device1 1 --destination-device 2 \
    --receiver-seed0 "$seed0" --receiver-seed1 "$seed1" --warmup "$warmup" \
    --release-lead-ms 1000 &
pids="$pids $!"

failure=0
for pid in $pids; do wait "$pid" || failure=1; done
pids=""
[[ "$failure" -eq 0 ]] || { echo "C153 process failure" >&2; exit 1; }

python3.11 "$SOFTWALL_SCRIPTS/analyze_confirm153_integrated_shared_recovery.py" \
    --protocol "$protocol" --owner0 "$owner0" --owner1 "$owner1" \
    --worker "$worker" --qwen "$qwen" --inventory "$inventory" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --output "$result"

cleanup
trap - EXIT INT TERM
echo "C153 integrated shared recovery complete: $result"
