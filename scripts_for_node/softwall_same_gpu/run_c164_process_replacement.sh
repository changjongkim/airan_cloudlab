#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C164P_LABEL:-c164_process_replacement_job${SLURM_JOB_ID}}
campaign=${SOFTWALL_C164P_CAMPAIGN:-development}
seed=${SOFTWALL_C164P_SEED:-53000000}
episodes=${SOFTWALL_C164P_EPISODES:-8}
excluded_nodes=${SOFTWALL_C164P_EXCLUDED_NODES:-nid002288,nid002100,nid001109,nid001085,nid001064,nid001124,nid001177,nid001308,nid001204,nid001632,nid001824,nid001025,nid001145,nid001069,nid001044,nid001265,nid001353,nid001169,nid001137,nid001192,nid001200,nid001288,nid001381,nid001057,nid001104}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
socket_dir="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
protocol="$result_root/${label}_protocol.json"
mandatory_output="$raw/${label}_mandatory.json"
mandatory_ready="$state/mandatory_ready.json"
mandatory_stop="$state/mandatory_stop"
inventory="$raw/${label}_gpu_inventory.csv"
result="$result_root/${label}_result.json"
episode_table="$state/episodes.tsv"

rm -rf "$state"; mkdir -p "$state" "$raw" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$protocol" "$mandatory_output" "$mandatory_stop" "$inventory" "$result"
nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"
python3.11 "$SOFTWALL_SCRIPTS/build_c164_process_replacement_protocol.py" \
  --output "$protocol" --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
  --raw-dir "$raw" --state-dir "$state" --socket-dir "$socket_dir" \
  --label "$label" --campaign "$campaign" \
  --seed "$seed" --episodes "$episodes" --excluded-nodes "$excluded_nodes"

current_node=$(hostname)
case ",$excluded_nodes," in *",$current_node,"*) echo "excluded node $current_node" >&2; exit 3;; esac

python3.11 - "$protocol" "$episode_table" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
with open(sys.argv[2],'w') as f:
 for e in p['episodes']:
  f.write('\t'.join(str(e[k]) for k in ('episode','fault_mode','context_length','token','old_worker_epoch','new_worker_epoch','socket','ready','worker_output','worker_log','client','journal','before','termination','exit','after','certificate'))+'\n')
PY

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

start_worker() {
  local epoch=$1 socket_path=$2 ready_path=$3 journal_path=$4 output_path=$5 log_path=$6 predecessor=${7:-}
  rm -f "$socket_path" "$ready_path" "$output_path"
  local predecessor_args=()
  [[ -n "$predecessor" ]] && predecessor_args=(--predecessor-journal "$predecessor")
  shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/c164_process_replacement_qwen_worker.py \
    --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 128,512 --warmup-per-length 3 \
    --socket "$socket_path" --journal "$journal_path" --ready-file "$ready_path" \
    --worker-epoch "$epoch" --output "$output_path" "${predecessor_args[@]}" \
    >"$log_path" 2>&1 &
  worker_pid=$!
  for _ in {1..6000}; do
    [[ -S "$socket_path" && -f "$ready_path" ]] && break
    kill -0 "$worker_pid" 2>/dev/null || { wait "$worker_pid" || true; echo "worker readiness failure: $epoch" >&2; return 1; }
    sleep 0.05
  done
  [[ -S "$socket_path" && -f "$ready_path" ]] || { echo "worker readiness timeout: $epoch" >&2; return 1; }
}

previous_journal=""
while IFS=$'\t' read -r episode fault context token old_epoch new_epoch socket_path ready_path \
    worker_output worker_log client journal before termination exit_marker after certificate; do
  start_worker "$old_epoch" "$socket_path" "$ready_path" "$journal" \
    "$worker_output" "$worker_log" "$previous_journal"
  wrapper_pid=$worker_pid
  python3.11 "$SOFTWALL_SCRIPTS/c164_process_replacement_client.py" \
    --socket "$socket_path" --worker-epoch "$old_epoch" --lifecycle-epoch "$seed" \
    --token "$token" --context-length "$context" --fault-mode "$fault" --output "$client"
  internal_worker_pid=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$ready_path")
  python3.11 "$SOFTWALL_SCRIPTS/c164_mps_client_control.py" snapshot \
    --target-pid "$internal_worker_pid" --output "$before"
  python3.11 "$SOFTWALL_SCRIPTS/c164_mps_client_control.py" terminate \
    --before "$before" --output "$termination"
  client_pid=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target"]["pid"])' "$termination")
  kill -9 "$client_pid" 2>/dev/null || true
  kill -9 "$wrapper_pid" 2>/dev/null || true
  set +e
  wait "$wrapper_pid"; wait_status=$?
  set -e
  worker_pid=""
  for _ in {1..200}; do kill -0 "$client_pid" 2>/dev/null || break; sleep 0.01; done
  python3.11 "$SOFTWALL_SCRIPTS/c164_process_exit_marker.py" \
    --client-pid "$client_pid" --wrapper-pid "$wrapper_pid" \
    --wrapper-wait-status "$wait_status" --output "$exit_marker"
  python3.11 "$SOFTWALL_SCRIPTS/c164_mps_client_control.py" snapshot \
    --target-pid "$internal_worker_pid" --output "$after"
  python3.11 "$SOFTWALL_SCRIPTS/build_c164_quiescence_certificate.py" \
    --journal "$journal" --before "$before" --termination "$termination" \
    --exit "$exit_marker" --after "$after" --output "$certificate"
  python3.11 "$SOFTWALL_SCRIPTS/c164_recover_replacement_journal.py" \
    --journal "$journal" --certificate "$certificate" --new-worker-epoch "$new_epoch" >/dev/null
  previous_journal=$journal
done < "$episode_table"

final_epoch=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_worker"]["epoch"])' "$protocol")
final_socket=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_worker"]["socket"])' "$protocol")
final_ready=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_worker"]["ready"])' "$protocol")
final_output=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_worker"]["output"])' "$protocol")
final_log=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_worker"]["log"])' "$protocol")
final_journal=$(python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_worker"]["journal"])' "$protocol")
start_worker "$final_epoch" "$final_socket" "$final_ready" "$final_journal" \
  "$final_output" "$final_log" "$previous_journal"
python3.11 - "$final_socket" <<'PY'
import json,socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect(sys.argv[1])
with s,s.makefile('rwb',buffering=0) as c:
 c.write(b'{"op":"stop"}\n'); r=json.loads(c.readline()); assert r.get('stopped') is True
PY
wait "$worker_pid"; worker_pid=""

touch "$mandatory_stop"; wait "$mandatory_pid"; mandatory_pid=""
python3.11 "$SOFTWALL_SCRIPTS/analyze_c164_process_replacement.py" \
  --protocol "$protocol" --mandatory "$mandatory_output" --inventory "$inventory" \
  --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" --output "$result"

cleanup; trap - EXIT INT TERM
echo "C164 process-replacement campaign complete: $result"
