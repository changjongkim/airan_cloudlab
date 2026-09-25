#!/usr/bin/env bash

# Actual TensorRT NeuralRx -> V17.1 credit transition -> Qwen/shared recovery.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_ACTUAL_NRX_LABEL:-confirm157a_actual_nrx_job${SLURM_JOB_ID}}
campaign=${SOFTWALL_ACTUAL_NRX_CAMPAIGN:-development}
seed_base=${SOFTWALL_ACTUAL_NRX_SEED_BASE:-25800000}
excluded_nodes=${SOFTWALL_ACTUAL_NRX_EXCLUDED_NODES:-nid001085,nid001064}
warmup=${SOFTWALL_ACTUAL_NRX_WARMUP:-10}
release_lead_ms=${SOFTWALL_ACTUAL_NRX_RELEASE_LEAD_MS:-3000}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
peer_spec="$result_root/${label}_peer_spec.json"
peer_tsv="$state/peers.tsv"
release_file="$state/release.json"
dispatch_file="$state/dispatch.json"
inventory="$raw/${label}_gpu_inventory.csv"
qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/${label}.sock"
qwen_output="$raw/${label}_qwen.json"
nrx_output="$raw/${label}_nrx_worker.json"
coordinator_output="$raw/${label}_coordinator.json"
result="$result_root/${label}_result.json"

mkdir -p "$raw" "$state" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$state"/cuda_ipc_* "$state"/*.json "$qwen_socket" \
    "$qwen_output" "$nrx_output" "$coordinator_output" "$result"
nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

python3.11 "$SOFTWALL_SCRIPTS/build_actual_nrx_integrated_protocol.py" \
    --output "$protocol" --peer-spec "$peer_spec" --peer-tsv "$peer_tsv" \
    --label "$label" --campaign "$campaign" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --raw-dir "$raw" --state-dir "$state" --seed-base "$seed_base" \
    --excluded-nodes "$excluded_nodes" --warmup "$warmup" \
    --release-lead-ms "$release_lead_ms"

current_node=$(hostname)
case ",$excluded_nodes," in
    *",$current_node,"*)
        echo "actual-NRx node $current_node is in frozen exclusion set" >&2
        exit 3
        ;;
esac

shifter_all() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=0,1,2,3 "$@"
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
export SOFTWALL_MPS_GPU=0,1,2,3
softwall_mps_start
softwall_mps_assert

shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/trace_qwen_worker.py \
    --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 64 \
    --batch-size 1 --warmup-per-length 3 --socket "$qwen_socket" \
    --output "$qwen_output" >"$raw/${label}_qwen.log" 2>&1 &
qwen_pid=$!
pids="$pids $qwen_pid"
for _ in {1..3600}; do
    [[ -S "$qwen_socket" ]] && break
    if ! kill -0 "$qwen_pid" 2>/dev/null; then
        wait "$qwen_pid" || true
        echo "actual-NRx Qwen worker exited before readiness" >&2
        exit 1
    fi
    sleep 0.05
done
[[ -S "$qwen_socket" ]] || { echo "actual-NRx Qwen readiness timeout" >&2; exit 1; }

while IFS=$'\t' read -r home request source receiver_seed channel_seed snr \
        nrx_tag recovery_tag ready_file outcome_file owner_output; do
    [[ -n "$nrx_tag" ]] || continue
    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/actual_nrx_integrated_owner.py \
        --home-id "$home" --request-id "$request" --source-device "$source" \
        --nrx-tag "$nrx_tag" --recovery-tag "$recovery_tag" \
        --ipc-dir "$state" --release-file "$release_file" \
        --dispatch-file "$dispatch_file" --ready-file "$ready_file" \
        --outcome-file "$outcome_file" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$owner_output" --receiver-seed "$receiver_seed" \
        --channel-seed "$channel_seed" --snr-db "$snr" \
        >"${owner_output%.json}.log" 2>&1 &
    pids="$pids $!"
done < "$peer_tsv"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/actual_nrx_multi_peer_worker.py \
    --peer-spec "$peer_spec" --ipc-dir "$state" --release-file "$release_file" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$nrx_output" --destination-device 3 \
    >"$raw/${label}_nrx_worker.log" 2>&1 &
pids="$pids $!"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/actual_nrx_integrated_coordinator.py \
    --peer-spec "$peer_spec" --ipc-dir "$state" \
    --release-file "$release_file" --dispatch-file "$dispatch_file" \
    --qwen-socket "$qwen_socket" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$coordinator_output" --destination-device 2 \
    --warmup "$warmup" --release-lead-ms "$release_lead_ms" \
    >"$raw/${label}_coordinator.log" 2>&1 &
pids="$pids $!"

failure=0
for pid in $pids; do wait "$pid" || failure=1; done
pids=""
[[ "$failure" -eq 0 ]] || {
    echo "actual-NRx integrated process failure; artifacts preserved" >&2
    exit 1
}

python3.11 "$SOFTWALL_SCRIPTS/analyze_actual_nrx_integrated.py" \
    --protocol "$protocol" --peer-spec "$peer_spec" \
    --coordinator "$coordinator_output" --nrx-worker "$nrx_output" \
    --qwen "$qwen_output" --inventory "$inventory" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --output "$result"
python3.11 - "$result" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding='utf-8'))
raise SystemExit(0 if value['all_pass'] else 1)
PY

cleanup
trap - EXIT INT TERM
echo "actual-NeuralRx integrated campaign complete: $result"
