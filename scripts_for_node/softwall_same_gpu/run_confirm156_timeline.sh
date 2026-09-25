#!/usr/bin/env bash

# C156: Nsight timeline conformance for the controlled conditional-open arm.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C156_LABEL:-confirm156_timeline_job${SLURM_JOB_ID}}
seed_base=${SOFTWALL_C156_SEED_BASE:-25600000}
excluded_nodes=${SOFTWALL_C156_EXCLUDED_NODES:-nid001176,nid001348,nid001253,nid001280,nid001032,nid001025}
warmup=${SOFTWALL_C156_WARMUP:-10}
release_lead_ms=${SOFTWALL_C156_RELEASE_LEAD_MS:-3000}
snr_db=${SOFTWALL_C156_SNR_DB:-20.0}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
peer_spec="$result_root/${label}_peer_spec.json"
peer_tsv="$state/peers.tsv"
release_file="$state/release.json"
inventory="$raw/${label}_gpu_inventory.csv"
qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c156.sock"
qwen_output="$raw/${label}_qwen.json"
worker_output="$raw/${label}_worker.json"
qwen_profile="$raw/${label}_qwen_profile"
worker_profile="$raw/${label}_worker_profile"
result="$result_root/${label}_result.json"
mkdir -p "$raw" "$state" "$state/tmp" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"

nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

python3.11 "$SOFTWALL_SCRIPTS/build_confirm156_timeline_protocol.py" \
    --output "$protocol" --label "$label" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --seed-base "$seed_base" --excluded-nodes "$excluded_nodes" \
    --snr-db "$snr_db" --warmup "$warmup" --release-lead-ms "$release_lead_ms"

current_node=$(hostname)
case ",$excluded_nodes," in
    *",$current_node,"*)
        echo "C156 node $current_node is in frozen exclusion set" >&2
        exit 3
        ;;
esac

python3.11 "$SOFTWALL_SCRIPTS/build_confirm154_peer_spec.py" \
    --protocol "$protocol" --branch conditional_open \
    --tag-prefix "$label" --raw-dir "$raw" --output "$peer_spec" > "$peer_tsv"

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
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
export SOFTWALL_MPS_GPU=0,1,2
softwall_mps_start
softwall_mps_assert

shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
    --sample=none --cpuctxsw=none --force-overwrite=true --export=sqlite \
    --output="$qwen_profile" \
    python3 /softwall/trace_qwen_worker.py \
    --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 64 \
    --batch-size 1 --warmup-per-length 3 \
    --socket "$qwen_socket" --output "$qwen_output" &
qwen_pid=$!
pids="$pids $qwen_pid"
for _ in {1..3600}; do
    [[ -S "$qwen_socket" ]] && break
    if ! kill -0 "$qwen_pid" 2>/dev/null; then
        wait "$qwen_pid" || true
        echo "C156 Qwen worker exited before readiness" >&2
        exit 1
    fi
    sleep 0.05
done
[[ -S "$qwen_socket" ]] || { echo "C156 Qwen readiness timeout" >&2; exit 1; }

while IFS=$'\t' read -r home request source receiver_seed channel_seed tag owner_output; do
    [[ -n "$tag" ]] || continue
    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/integrated_shared_recovery_owner.py \
        --home-id "$home" --request-id "$request" \
        --source-device "$source" --destination-device 2 \
        --tag "$tag" --ipc-dir "$state" --release-file "$release_file" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$owner_output" --receiver-seed "$receiver_seed" \
        --channel-seed "$channel_seed" --snr-db "$snr_db" &
    pids="$pids $!"
done < "$peer_tsv"

for _ in {1..7200}; do
    all_ready=1
    while IFS=$'\t' read -r _home _request _source _receiver _channel tag _output; do
        [[ -n "$tag" ]] || continue
        [[ -f "$state/cuda_ipc_${tag}.info" ]] || all_ready=0
    done < "$peer_tsv"
    [[ "$all_ready" -eq 1 ]] && break
    sleep 0.05
done
while IFS=$'\t' read -r _home _request _source _receiver _channel tag _output; do
    [[ -n "$tag" ]] || continue
    [[ -f "$state/cuda_ipc_${tag}.info" ]] || {
        echo "C156 owner $tag did not publish IPC handles" >&2
        exit 1
    }
done < "$peer_tsv"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
    TMPDIR="$state/tmp" \
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
    --sample=none --cpuctxsw=none --force-overwrite=true --export=sqlite \
    --output="$worker_profile" \
    python3 /softwall/integrated_shared_recovery_timeline_worker.py \
    --branch conditional_open --peer-spec "$peer_spec" \
    --ipc-dir "$state" --release-file "$release_file" \
    --qwen-socket "$qwen_socket" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$worker_output" --destination-device 2 \
    --warmup "$warmup" --release-lead-ms "$release_lead_ms" &
pids="$pids $!"

failure=0
for pid in $pids; do wait "$pid" || failure=1; done
pids=""
[[ "$failure" -eq 0 ]] || { echo "C156 process failure" >&2; exit 1; }

[[ -f "${qwen_profile}.sqlite" && -f "${worker_profile}.sqlite" ]] || {
    echo "C156 Nsight SQLite export missing" >&2
    exit 1
}

python3.11 "$SOFTWALL_SCRIPTS/analyze_confirm156_timeline.py" \
    --protocol "$protocol" --peer-spec "$peer_spec" \
    --worker "$worker_output" --qwen "$qwen_output" \
    --worker-sqlite "${worker_profile}.sqlite" \
    --qwen-sqlite "${qwen_profile}.sqlite" \
    --inventory "$inventory" --scripts-root "$SOFTWALL_SCRIPTS" \
    --task1-root "$SOFTWALL_TASK1" --output "$result"

cleanup
trap - EXIT INT TERM
echo "C156 timeline complete: $result"
