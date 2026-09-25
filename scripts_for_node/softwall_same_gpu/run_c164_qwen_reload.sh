#!/usr/bin/env bash

# C164 repeated Qwen unload/reload while mandatory four-cell cuPHY continues.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C164Q_LABEL:-c164_qwen_reload_job${SLURM_JOB_ID}}
campaign=${SOFTWALL_C164Q_CAMPAIGN:-development}
seed=${SOFTWALL_C164Q_SEED:-49000000}
episodes=${SOFTWALL_C164Q_EPISODES:-30}
quiet_gap_s=${SOFTWALL_C164Q_QUIET_GAP_S:-1}
excluded_nodes=${SOFTWALL_C164Q_EXCLUDED_NODES:-nid002288,nid002100,nid001109,nid001085,nid001064,nid001124,nid001177,nid001308,nid001204,nid001632,nid001824,nid001025,nid001145,nid001069,nid001044,nid001265,nid001353,nid001169,nid001137,nid001192,nid001200}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
protocol="$result_root/${label}_protocol.json"
mandatory_output="$raw/${label}_mandatory.json"
mandatory_ready="$state/mandatory_ready.json"
mandatory_stop="$state/mandatory_stop"
inventory="$raw/${label}_gpu_inventory.csv"
result="$result_root/${label}_result.json"
qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c164q.sock"

rm -rf "$state"
mkdir -p "$raw" "$state" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -f "$mandatory_output" "$mandatory_ready" "$mandatory_stop" \
    "$inventory" "$result" "$qwen_socket"
nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

python3.11 "$SOFTWALL_SCRIPTS/build_c164_qwen_reload_protocol.py" \
    --output "$protocol" --scripts-root "$SOFTWALL_SCRIPTS" \
    --task1-root "$SOFTWALL_TASK1" \
    --raw-dir "$raw" --label "$label" --campaign "$campaign" \
    --seed "$seed" --episodes "$episodes" \
    --quiet-gap-s "$quiet_gap_s" --excluded-nodes "$excluded_nodes"

current_node=$(hostname)
case ",$excluded_nodes," in
    *",$current_node,"*)
        echo "C164 reload node $current_node is in the frozen exclusion set" >&2
        exit 3
        ;;
esac

shifter_aerial_gpu2() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=2 "$@"
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

mandatory_pid=""
qwen_pid=""
cleanup() {
    [[ -n "$qwen_pid" ]] && kill "$qwen_pid" 2>/dev/null || true
    [[ -n "$mandatory_pid" ]] && kill "$mandatory_pid" 2>/dev/null || true
    [[ -n "$qwen_pid" ]] && wait "$qwen_pid" 2>/dev/null || true
    [[ -n "$mandatory_pid" ]] && wait "$mandatory_pid" 2>/dev/null || true
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
# Keep the daemon's visibility identical to the qualified multi-GPU mode.
# With a daemon restricted to physical GPU2, MPS remaps it to logical GPU0;
# a client that also requests CUDA_VISIBLE_DEVICES=2 then sees no device.
export SOFTWALL_MPS_GPU=0,1,2,3
softwall_mps_start
softwall_mps_assert

shifter_aerial_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 \
    CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/c164_mandatory_continuity_runner.py \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$mandatory_output" --ready-file "$mandatory_ready" \
    --stop-file "$mandatory_stop" --cells 4 --period-ms 180 \
    --deadline-ms 155 --component-bound-ms 25 --warmup 20 \
    --minimum-iterations 30 --maximum-iterations 5000 \
    --release-lead-ms 2000 --seed "$seed" \
    >"$raw/${label}_mandatory.log" 2>&1 &
mandatory_pid=$!

for _ in {1..6000}; do
    [[ -f "$mandatory_ready" ]] && break
    if ! kill -0 "$mandatory_pid" 2>/dev/null; then
        wait "$mandatory_pid" || true
        echo "mandatory continuity runner exited before readiness" >&2
        exit 1
    fi
    sleep 0.05
done
[[ -f "$mandatory_ready" ]] || { echo "mandatory readiness timeout" >&2; exit 1; }

# Leave a quiet same-process prefix before the first reload.
python3.11 - "$mandatory_ready" <<'PY'
import json,sys,time
d=json.load(open(sys.argv[1]))
target=int(d['first_release_ns']) + 5*int(d['period_ns'])
while time.perf_counter_ns() < target:
    time.sleep(.001)
PY

for episode in $(seq 1 "$episodes"); do
    printf -v suffix '%03d' "$episode"
    qwen_output="$raw/${label}_qwen_${suffix}.json"
    event_output="$raw/${label}_reload_episode_${suffix}.json"
    rm -f "$qwen_socket" "$qwen_output" "$event_output"
    launch_ns=$(python3.11 -c 'import time; print(time.perf_counter_ns())')
    shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
        CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/c159_q2_qwen_worker.py \
        --model Qwen/Qwen2.5-1.5B \
        --allowed-context-lengths 16,32,64,128,256,512 \
        --batch-size 1 --warmup-per-length 3 --socket "$qwen_socket" \
        --output "$qwen_output" >"$raw/${label}_qwen_${suffix}.log" 2>&1 &
    qwen_pid=$!
    for _ in {1..6000}; do
        [[ -S "$qwen_socket" ]] && break
        if ! kill -0 "$qwen_pid" 2>/dev/null; then
            wait "$qwen_pid" || true
            echo "Qwen reload episode $episode exited before readiness" >&2
            exit 1
        fi
        sleep 0.05
    done
    [[ -S "$qwen_socket" ]] || { echo "Qwen reload readiness timeout" >&2; exit 1; }
    ready_ns=$(python3.11 -c 'import time; print(time.perf_counter_ns())')
    python3.11 - "$qwen_socket" <<'PY'
import json,socket,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect(sys.argv[1])
with s, s.makefile('rwb',buffering=0) as c:
    c.write(b'{"op":"stop"}\n')
    r=json.loads(c.readline())
    assert r.get('stopped') is True, r
PY
    wait "$qwen_pid"
    qwen_pid=""
    exit_ns=$(python3.11 -c 'import time; print(time.perf_counter_ns())')
    python3.11 "$SOFTWALL_SCRIPTS/c164_qwen_reload_episode.py" \
        --episode "$episode" --launch-ns "$launch_ns" \
        --ready-ns "$ready_ns" --exit-ns "$exit_ns" \
        --qwen-output "$qwen_output" --output "$event_output"
    sleep "$quiet_gap_s"
done

touch "$mandatory_stop"
wait "$mandatory_pid"
mandatory_pid=""

python3.11 "$SOFTWALL_SCRIPTS/analyze_c164_qwen_reload.py" \
    --protocol "$protocol" --mandatory "$mandatory_output" \
    --inventory "$inventory" --scripts-root "$SOFTWALL_SCRIPTS" \
    --task1-root "$SOFTWALL_TASK1" \
    --output "$result"

cleanup
trap - EXIT INT TERM
echo "C164 Qwen-reload mandatory-continuity campaign complete: $result"
