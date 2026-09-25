#!/usr/bin/env bash

# C164 first-RAN-request-after-quiescent-MPS-restart boundary campaign.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C164R_LABEL:-c164_mps_restart_job${SLURM_JOB_ID}}
campaign=${SOFTWALL_C164R_CAMPAIGN:-development}
seed_base=${SOFTWALL_C164R_SEED_BASE:-47000000}
excluded_nodes=${SOFTWALL_C164R_EXCLUDED_NODES:-nid002288,nid002100,nid001109,nid001085,nid001064,nid001124,nid001177,nid001308,nid001204,nid001632,nid001824,nid001025,nid001145,nid001069,nid001044,nid001265,nid001353}
iterations=${SOFTWALL_C164R_ITERATIONS:-60}
period_ms=${SOFTWALL_C164R_PERIOD_MS:-180}
warmup=${SOFTWALL_C164R_WARMUP:-10}
release_lead_ms=${SOFTWALL_C164R_RELEASE_LEAD_MS:-3000}
reverse_cases=${SOFTWALL_C164R_REVERSE_CASES:-0}
grid="$SOFTWALL_ROOT/results/softwall_multigpu/c162_feasibility_grid_v1.json"
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
peer_spec="$result_root/${label}_peer_spec.json"
peer_tsv="$state/peers.tsv"
schedule_file="$state/schedule.json"
inventory="$raw/${label}_gpu_inventory.csv"
# Linux AF_UNIX paths are limited to roughly 108 bytes. Keep the per-job
# directory for uniqueness and use a fixed short basename independent of the
# descriptive artifact label.
qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c159q2.sock"
qwen_output="$raw/${label}_qwen.json"
nrx_output="$raw/${label}_nrx_worker.json"
coordinator_output="$raw/${label}_coordinator.json"
before_probe="$raw/${label}_mps_before_probe.json"
before_snapshot="$raw/${label}_mps_before_snapshot.json"
after_probe="$raw/${label}_mps_after_probe.json"
after_snapshot="$raw/${label}_mps_after_snapshot.json"
restart_marker="$raw/${label}_mps_restart_marker.json"
base_result="$result_root/${label}_base_result.json"
result="$result_root/${label}_result.json"

rm -rf "$state"
mkdir -p "$raw" "$state" \
    "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$qwen_socket" "$qwen_output" "$nrx_output" \
    "$coordinator_output" "$before_probe" "$before_snapshot" \
    "$after_probe" "$after_snapshot" "$restart_marker" "$base_result" "$result"
nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

reverse_arg=()
[[ "$reverse_cases" == 1 ]] && reverse_arg+=(--reverse-cases)
python3.11 "$SOFTWALL_SCRIPTS/build_c162_boundary_protocol.py" \
    --output "$protocol" --peer-spec "$peer_spec" --peer-tsv "$peer_tsv" \
    --label "$label" --campaign "$campaign" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --raw-dir "$raw" --state-dir "$state" --grid "$grid" \
    --seed-base "$seed_base" \
    --excluded-nodes "$excluded_nodes" --iterations "$iterations" \
    --period-ms "$period_ms" --warmup "$warmup" \
    --release-lead-ms "$release_lead_ms" "${reverse_arg[@]}"
python3.11 "$SOFTWALL_SCRIPTS/build_c164_restart_protocol.py" \
    --protocol "$protocol" --scripts-root "$SOFTWALL_SCRIPTS"

current_node=$(hostname)
case ",$excluded_nodes," in
    *",$current_node,"*)
        echo "C164 restart node $current_node is in the frozen exclusion set" >&2
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

mps_control_pids() {
    pgrep -u "$USER" -f '(^|/)nvidia-cuda-mps-control( |$)' \
        | paste -sd, - || true
}

mps_server_pids() {
    pgrep -u "$USER" -f '(^|/)nvidia-cuda-mps-server( |$)' \
        | paste -sd, - || true
}

# Establish a real four-GPU CUDA client epoch, prove that both MPS control and
# server processes disappear, then establish a distinct second epoch.  No RAN
# or optional-AI process is launched until this proof has been written.
shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
    python3 /softwall/c164_mps_epoch_probe.py \
    --epoch before_restart --output "$before_probe" \
    >"$raw/${label}_mps_before_probe.log" 2>&1
python3.11 "$SOFTWALL_SCRIPTS/c164_mps_epoch_snapshot.py" \
    --phase before_restart --probe "$before_probe" \
    --control-pids "$(mps_control_pids)" \
    --server-pids "$(mps_server_pids)" --output "$before_snapshot"

stop_started_ns=$(python3.11 -c 'import time; print(time.perf_counter_ns())')
softwall_mps_stop
stop_completed_ns=$(python3.11 -c 'import time; print(time.perf_counter_ns())')
control_absent=true
server_absent=true
softwall_mps_control_ready && control_absent=false
[[ -n "$(mps_server_pids)" ]] && server_absent=false

softwall_mps_start
softwall_mps_assert
shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
    python3 /softwall/c164_mps_epoch_probe.py \
    --epoch after_restart --output "$after_probe" \
    >"$raw/${label}_mps_after_probe.log" 2>&1
python3.11 "$SOFTWALL_SCRIPTS/c164_mps_epoch_snapshot.py" \
    --phase after_restart --probe "$after_probe" \
    --control-pids "$(mps_control_pids)" \
    --server-pids "$(mps_server_pids)" --output "$after_snapshot"
python3.11 "$SOFTWALL_SCRIPTS/build_c164_mps_restart_marker.py" \
    --before "$before_snapshot" --after "$after_snapshot" \
    --stop-started-ns "$stop_started_ns" \
    --stop-completed-ns "$stop_completed_ns" \
    --control-absent "$control_absent" --server-absent "$server_absent" \
    --output "$restart_marker"

shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/c159_q2_qwen_worker.py \
    --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 16,32,64,128,256,512 \
    --batch-size 1 --warmup-per-length 3 --socket "$qwen_socket" \
    --output "$qwen_output" >"$raw/${label}_qwen.log" 2>&1 &
qwen_pid=$!
pids="$pids $qwen_pid"
for _ in {1..3600}; do
    [[ -S "$qwen_socket" ]] && break
    if ! kill -0 "$qwen_pid" 2>/dev/null; then
        wait "$qwen_pid" || true
        echo "C159-Q2 Qwen worker exited before readiness" >&2
        exit 1
    fi
    sleep 0.05
done
[[ -S "$qwen_socket" ]] || { echo "C159-Q2 Qwen readiness timeout" >&2; exit 1; }

while IFS=$'\t' read -r home request source receiver_seed channel_seed_base snr \
        nrx_tag recovery_tag ready_file decision_control owner_output; do
    [[ -n "$nrx_tag" ]] || continue
    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/c159_prestaged_owner.py \
        --home-id "$home" --request-id "$request" --source-device "$source" \
        --nrx-tag "$nrx_tag" --recovery-tag "$recovery_tag" \
        --ipc-dir "$state" --schedule-file "$schedule_file" \
        --decision-control "$decision_control" --ready-file "$ready_file" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$owner_output" --iterations "$iterations" \
        --receiver-seed "$receiver_seed" \
        --channel-seed-base "$channel_seed_base" --snr-db "$snr" \
        >"${owner_output%.json}.log" 2>&1 &
    pids="$pids $!"
done < "$peer_tsv"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/c159_persistent_nrx_worker.py \
    --peer-spec "$peer_spec" --ipc-dir "$state" \
    --schedule-file "$schedule_file" --iterations "$iterations" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$nrx_output" --destination-device 3 \
    >"$raw/${label}_nrx_worker.log" 2>&1 &
pids="$pids $!"

shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/c162_boundary_coordinator.py \
    --peer-spec "$peer_spec" --ipc-dir "$state" \
    --schedule-file "$schedule_file" --qwen-socket "$qwen_socket" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$coordinator_output" --iterations "$iterations" \
    --period-ms "$period_ms" --destination-device 2 \
    --warmup "$warmup" --release-lead-ms "$release_lead_ms" \
    "${reverse_arg[@]}" \
    >"$raw/${label}_coordinator.log" 2>&1 &
pids="$pids $!"

failure=0
for pid in $pids; do wait "$pid" || failure=1; done
pids=""
[[ "$failure" -eq 0 ]] || {
    echo "C164 process failure; artifacts preserved" >&2
    exit 1
}

python3.11 "$SOFTWALL_SCRIPTS/analyze_c162_boundary.py" \
    --protocol "$protocol" --peer-spec "$peer_spec" \
    --coordinator "$coordinator_output" --nrx-worker "$nrx_output" \
    --qwen "$qwen_output" --inventory "$inventory" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --output "$base_result"
python3.11 "$SOFTWALL_SCRIPTS/analyze_c164_restart.py" \
    --protocol "$protocol" --base-result "$base_result" \
    --marker "$restart_marker" --coordinator "$coordinator_output" \
    --nrx-worker "$nrx_output" --qwen "$qwen_output" \
    --scripts-root "$SOFTWALL_SCRIPTS" --output "$result"
python3.11 - "$result" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value["all_pass"] else 1)
PY

cleanup
trap - EXIT INT TERM
echo "C164 MPS-restart physical boundary campaign complete: $result"
