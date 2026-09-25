#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C164R_LABEL:-c164_reconnect_job${SLURM_JOB_ID}}
campaign=${SOFTWALL_C164R_CAMPAIGN:-development}
seed=${SOFTWALL_C164R_SEED:-51000000}
tokens=${SOFTWALL_C164R_TOKENS:-60}
excluded_nodes=${SOFTWALL_C164R_EXCLUDED_NODES:-nid002288,nid002100,nid001109,nid001085,nid001064,nid001124,nid001177,nid001308,nid001204,nid001632,nid001824,nid001025,nid001145,nid001069,nid001044,nid001265,nid001353,nid001169,nid001137,nid001192,nid001200,nid001288,nid001381}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
mandatory_output="$raw/${label}_mandatory.json"
mandatory_ready="$state/mandatory_ready.json"
mandatory_stop="$state/mandatory_stop"
worker_output="$raw/${label}_worker.json"
client_output="$raw/${label}_client.json"
journal_dir="$state/journal"
inventory="$raw/${label}_gpu_inventory.csv"
result="$result_root/${label}_result.json"
qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/${label}.sock"

rm -rf "$state"
mkdir -p "$raw" "$state" "$journal_dir" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$protocol" "$mandatory_output" "$mandatory_ready" "$mandatory_stop" \
  "$worker_output" "$client_output" "$inventory" "$result" "$qwen_socket"
nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

python3.11 "$SOFTWALL_SCRIPTS/build_c164_reconnect_protocol.py" \
  --output "$protocol" --scripts-root "$SOFTWALL_SCRIPTS" \
  --task1-root "$SOFTWALL_TASK1" --label "$label" --campaign "$campaign" \
  --seed "$seed" --tokens "$tokens" --excluded-nodes "$excluded_nodes"

current_node=$(hostname)
case ",$excluded_nodes," in *",$current_node,"*) echo "excluded node $current_node" >&2; exit 3;; esac
worker_epoch=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["worker_epoch"])' "$protocol")
lifecycle_epoch=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lifecycle_epoch"])' "$protocol")

shifter_aerial_gpu2() {
  shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_TASK1:/softwall_task1" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=2 "$@"
}
shifter_qwen_gpu2() {
  shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
    --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/softwall \
    --env=HF_HOME=/softwall_runtime/cache/huggingface --env=TMPDIR=/softwall_runtime/tmp \
    --env=HF_HUB_DISABLE_XET=1 --env=CUDA_MODULE_LOADING=LAZY \
    --env=CUDA_VISIBLE_DEVICES=2 "$@"
}

worker_pid=""; mandatory_pid=""
cleanup() {
  [[ -n "$worker_pid" ]] && kill "$worker_pid" 2>/dev/null || true
  [[ -n "$mandatory_pid" ]] && kill "$mandatory_pid" 2>/dev/null || true
  [[ -n "$worker_pid" ]] && wait "$worker_pid" 2>/dev/null || true
  [[ -n "$mandatory_pid" ]] && wait "$mandatory_pid" 2>/dev/null || true
  softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure; softwall_mps_stop
export SOFTWALL_MPS_GPU=0,1,2,3
softwall_mps_start; softwall_mps_assert

shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
  python3 /softwall/c164_reconnect_qwen_worker.py --model Qwen/Qwen2.5-1.5B \
  --allowed-context-lengths 128,512 --batch-size 1 --warmup-per-length 3 \
  --socket "$qwen_socket" --journal-dir "$journal_dir" \
  --worker-epoch "$worker_epoch" --output "$worker_output" \
  >"$raw/${label}_worker.log" 2>&1 &
worker_pid=$!
for _ in {1..6000}; do
  [[ -S "$qwen_socket" ]] && break
  kill -0 "$worker_pid" 2>/dev/null || { wait "$worker_pid" || true; echo "worker readiness failure" >&2; exit 1; }
  sleep 0.05
done
[[ -S "$qwen_socket" ]] || { echo "worker readiness timeout" >&2; exit 1; }

shifter_aerial_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
  python3 /softwall/c164_mandatory_continuity_runner.py \
  --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
  --output "$mandatory_output" --ready-file "$mandatory_ready" --stop-file "$mandatory_stop" \
  --cells 4 --period-ms 180 --deadline-ms 155 --component-bound-ms 25 --warmup 20 \
  --minimum-iterations 30 --maximum-iterations 5000 --release-lead-ms 2000 --seed "$seed" \
  >"$raw/${label}_mandatory.log" 2>&1 &
mandatory_pid=$!
for _ in {1..6000}; do
  [[ -f "$mandatory_ready" ]] && break
  kill -0 "$mandatory_pid" 2>/dev/null || { wait "$mandatory_pid" || true; echo "mandatory readiness failure" >&2; exit 1; }
  sleep 0.05
done
[[ -f "$mandatory_ready" ]] || { echo "mandatory readiness timeout" >&2; exit 1; }

python3.11 - "$mandatory_ready" <<'PY'
import json,sys,time
d=json.load(open(sys.argv[1])); target=int(d['first_release_ns'])+5*int(d['period_ns'])
while time.perf_counter_ns()<target: time.sleep(.001)
PY

python3.11 "$SOFTWALL_SCRIPTS/c164_reconnect_client.py" --socket "$qwen_socket" \
  --worker-epoch "$worker_epoch" --lifecycle-epoch "$lifecycle_epoch" \
  --tokens "$tokens" --pacing-ms 10 --mandatory-ready "$mandatory_ready" \
  --launch-ahead-ms 10 --output "$client_output"
wait "$worker_pid"; worker_pid=""
touch "$mandatory_stop"
wait "$mandatory_pid"; mandatory_pid=""

python3.11 "$SOFTWALL_SCRIPTS/analyze_c164_reconnect.py" --protocol "$protocol" \
  --mandatory "$mandatory_output" --worker "$worker_output" --client "$client_output" \
  --inventory "$inventory" --scripts-root "$SOFTWALL_SCRIPTS" \
  --task1-root "$SOFTWALL_TASK1" --output "$result"

cleanup; trap - EXIT INT TERM
echo "C164 reconnect campaign complete: $result"
