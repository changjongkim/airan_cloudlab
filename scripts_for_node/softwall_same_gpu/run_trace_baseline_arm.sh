#!/usr/bin/env bash

# One independently restarted physical arm of the trace-driven system comparison.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
system=${1:?system must be static, work_conserving, or softwall}
case "$system" in
    static|work_conserving|softwall) ;;
    *) echo "invalid system: $system" >&2; exit 2 ;;
esac

prefix=${SOFTWALL_GATE_PREFIX:?SOFTWALL_GATE_PREFIX is required}
iterations=${SOFTWALL_GATE_ITERATIONS:-340}
payload_seed=${SOFTWALL_PAYLOAD_SEED:-20800051}
channel_seed_base=${SOFTWALL_CHANNEL_SEED_BASE:-20801000}
ai_bounds=${SOFTWALL_AI_BOUND_MAP:-16:35,32:35,64:35,128:40,256:65,512:75}
ai_model=${SOFTWALL_AI_MODEL:-Qwen/Qwen2.5-1.5B}
trace_path=${SOFTWALL_TRACE_PATH:-/softwall_burstgpt/softwall_burst60_prefill_trace.json}
trace_sha=${SOFTWALL_TRACE_SHA256:-da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d}
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
state="$SOFTWALL_ROOT/run_state/softwall_same_gpu/job${SLURM_JOB_ID}_${prefix}"
mkdir -p "$raw" "$state"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}

worker0_pid=""
worker1_pid=""
background_pid=""
tag="trace_${system}_${SLURM_JOB_ID}_${BASHPID}"
# AF_UNIX paths are typically limited to roughly 108 bytes.  Result prefixes
# carry experiment provenance and can be long, so keep the runtime socket name
# independent and short.
background_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/tb_${BASHPID}.sock"
cleanup() {
    for pid in "$worker0_pid" "$worker1_pid" "$background_pid"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    rm -f "$background_socket"
    softwall_mps_stop || true
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
sleep 10
softwall_mps_start
softwall_mps_assert

requal="$raw/${prefix}_requalification.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 100 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 50 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321955 --snr-db -8.5 --channel-seed-base 20359900
python3 - "$requal" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x['iterations'] != 100 or x['deadline_misses']:
    raise SystemExit('trace-arm operational requalification failed')
print('trace-arm operational requalification passed', flush=True)
PY

sleep 10
for cell in 0 1; do
    output="$raw/${prefix}_worker${cell}.json"
    shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/same_request_ipc_worker.py \
        --tag "${tag}_c${cell}" --ipc-dir "$state" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$output" &
    if [[ "$cell" == 0 ]]; then worker0_pid=$!; else worker1_pid=$!; fi
done

shifter_qwen env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/trace_qwen_worker.py \
    --model "$ai_model" --allowed-context-lengths 16,32,64,128,256,512 \
    --batch-size 1 --warmup-per-length 3 \
    --socket "$background_socket" \
    --output "$raw/${prefix}_background.json" &
background_pid=$!
for _ in {1..3000}; do
    [[ -S "$background_socket" ]] && break
    kill -0 "$background_pid" 2>/dev/null || {
        echo "trace background worker exited before ready" >&2
        exit 1
    }
    sleep 0.05
done
[[ -S "$background_socket" ]] || {
    echo 'trace background worker readiness timeout' >&2
    exit 1
}

shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/four_cell_trace_baseline_controller.py \
    --tag-prefix "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --socket "$background_socket" \
    --trace "$trace_path" \
    --trace-sha256 "$trace_sha" --system "$system" \
    --ai-bound-map "$ai_bounds" \
    --output "$raw/${prefix}_controller.json" \
    --iterations "$iterations" --cells 4 --warmup 20 \
    --period-ms 180 --deadline-ms 155 \
    --nrx-bound-ms 45 --conv-bound-ms 12 --commit-guard-ms 2 \
    --endpoint-timeout-ms 100 --ai-guard-ms 2 --ai-rpc-timeout-ms 120 \
    --seed "$payload_seed" --snr-db -8.5 \
    --channel-seed-base "$channel_seed_base" \
    --gate-mode low_threshold --gate-threshold 1.9490545988082886 \
    --gc-mode off --alternate-admission-order \
    --inject-correlated-failure-every 5 --fault-pattern mixed

wait "$worker0_pid"; worker0_pid=""
wait "$worker1_pid"; worker1_pid=""
wait "$background_pid"; background_pid=""
cleanup
trap - EXIT INT TERM
echo "trace baseline arm completed: $system / $prefix"
