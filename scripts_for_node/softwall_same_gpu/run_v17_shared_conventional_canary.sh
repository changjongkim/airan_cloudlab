#!/usr/bin/env bash

# C149: two RAN homes send PUSCH recovery inputs to one GPU2 cuPHY worker.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

iterations=${SOFTWALL_SHARED_CONV_ITERATIONS:-100}
warmup=${SOFTWALL_SHARED_CONV_WARMUP:-10}
snr_db=${SOFTWALL_SHARED_CONV_SNR_DB:-20.0}
label=${SOFTWALL_SHARED_CONV_LABEL:-confirm149_shared_conv_job${SLURM_JOB_ID}}
seed0=${SOFTWALL_SHARED_CONV_SEED0:-24900051}
seed1=${SOFTWALL_SHARED_CONV_SEED1:-24910051}
channel0=${SOFTWALL_SHARED_CONV_CHANNEL0:-24901000}
channel1=${SOFTWALL_SHARED_CONV_CHANNEL1:-24911000}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
result="$result_root/${label}_result.json"
owner0="$raw/${label}_owner0.json"
owner1="$raw/${label}_owner1.json"
worker="$raw/${label}_worker.json"
inventory="$raw/${label}_gpu_inventory.csv"
mkdir -p "$raw" "$state" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$state"/cuda_ipc_* "$owner0" "$owner1" "$worker" "$result"

nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

python3.11 "$SOFTWALL_SCRIPTS/build_confirm149_protocol.py" \
    --output "$protocol" --label "$label" \
    --iterations "$iterations" --warmup "$warmup" \
    --receiver-seed0 "$seed0" --receiver-seed1 "$seed1" \
    --channel-seed0 "$channel0" --channel-seed1 "$channel1" \
    --snr-db "$snr_db" \
    --owner-source "$SOFTWALL_SCRIPTS/shared_conventional_owner.py" \
    --worker-source "$SOFTWALL_SCRIPTS/shared_conventional_worker.py" \
    --p2p-source "$SOFTWALL_SCRIPTS/multigpu_p2p_ipc_gate.py" \
    --phy-source "$SOFTWALL_SCRIPTS/dual_receiver_phy.py" \
    --ipc-source "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py"

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

pids=""
cleanup() {
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    for pid in $pids; do wait "$pid" 2>/dev/null || true; done
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
export SOFTWALL_MPS_GPU=0,1,2
softwall_mps_start
softwall_mps_assert

tag="${label}_${BASHPID}"
shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/shared_conventional_owner.py \
    --home-id 0 --source-device 0 --destination-device 2 \
    --tag "${tag}_h0" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$owner0" --iterations "$iterations" \
    --receiver-seed "$seed0" --channel-seed-base "$channel0" \
    --snr-db "$snr_db" &
pids="$pids $!"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/shared_conventional_owner.py \
    --home-id 1 --source-device 1 --destination-device 2 \
    --tag "${tag}_h1" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$owner1" --iterations "$iterations" \
    --receiver-seed "$seed1" --channel-seed-base "$channel1" \
    --snr-db "$snr_db" &
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
    echo "shared conventional owners did not publish IPC handles" >&2
    exit 1
}

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/shared_conventional_worker.py \
    --tag0 "${tag}_h0" --tag1 "${tag}_h1" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$worker" --iterations "$iterations" --warmup "$warmup" \
    --source-device0 0 --source-device1 1 --destination-device 2 \
    --receiver-seed0 "$seed0" --receiver-seed1 "$seed1" &
pids="$pids $!"

failure=0
for pid in $pids; do wait "$pid" || failure=1; done
pids=""
[[ "$failure" -eq 0 ]] || {
    echo "shared conventional canary process failed" >&2
    exit 1
}

python3.11 "$SOFTWALL_SCRIPTS/analyze_confirm149_shared_conventional.py" \
    --protocol "$protocol" --owner0 "$owner0" --owner1 "$owner1" \
    --worker "$worker" --inventory "$inventory" --output "$result"

cleanup
trap - EXIT INT TERM
echo "shared conventional canary complete: $result"
