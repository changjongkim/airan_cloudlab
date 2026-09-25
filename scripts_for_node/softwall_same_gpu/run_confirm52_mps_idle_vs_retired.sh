#!/usr/bin/env bash
set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
result="$SOFTWALL_ROOT/results/softwall_same_gpu"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}
worker_pid=""
controller_pid=""
socket=""
cleanup() {
    if [[ -n "$controller_pid" ]]; then kill "$controller_pid" 2>/dev/null || true; fi
    if [[ -n "$worker_pid" ]]; then kill "$worker_pid" 2>/dev/null || true; fi
    if [[ -n "$socket" ]]; then rm -f "$socket"; fi
    softwall_mps_stop || true
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

requal="$raw/confirm52_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 60 --deadline-ms 35 --nrx-bound-ms 50 \
    --conv-bound-ms 35 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20322952 --channel-seed-base 20325952
python3 - "$requal" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x['iterations'] != 200 or x['deadline_misses']:
    raise SystemExit('confirm52 operational requalification failed')
print('confirm52 operational requalification passed', flush=True)
PY

for round in {1..10}; do
    seed=$((20322000 + round))
    if ((round % 2)); then
        order="mps_idle mps_ack_only mps_retired"
    else
        order="mps_retired mps_ack_only mps_idle"
    fi
    for condition in $order; do
        sleep 20
        prefix="confirm52_r${round}_${condition}_job${SLURM_JOB_ID}"
        ran_output="$raw/${prefix}_ran.json"
        worker_output="$raw/${prefix}_worker.json"
        launcher_output="$raw/${prefix}_launcher.json"
        state="$SOFTWALL_ROOT/run_state/softwall_same_gpu/job${SLURM_JOB_ID}/confirm52"
        mkdir -p "$state"
        ready_file="$state/${prefix}.ready"
        start_file="$state/${prefix}.start.json"
        socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c52_${BASHPID}_${round}_${condition}.sock"
        rm -f "$socket" "$ran_output" "$worker_output" "$launcher_output" "$ready_file" "$start_file"
        shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
            CUDA_MPS_CLIENT_PRIORITY=1 \
            python3 /softwall/ai_worker.py --kind nrx --mode rpc \
            --repeats 40 --socket "$socket" --output "$worker_output" &
        worker_pid=$!
        for _ in {1..1200}; do
            [[ -S "$socket" ]] && break
            kill -0 "$worker_pid" 2>/dev/null || {
                echo "confirm52 worker exited before ready" >&2
                exit 1
            }
            sleep 0.05
        done
        [[ -S "$socket" ]] || { echo 'confirm52 worker readiness timeout' >&2; exit 1; }
        if [[ "$condition" == mps_idle ]]; then
            lifecycle_arg=--quiesce-before-first-release
        else
            lifecycle_arg=--retire-before-first-release
        fi
        lifecycle_args=()
        if [[ "$condition" != mps_ack_only ]]; then
            lifecycle_args=(--lifecycle-ready-file "$ready_file" --lifecycle-start-file "$start_file")
        fi
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/overrun_controller.py \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --socket "$socket" --output "$ran_output" \
            --iterations 1000 --warmup 20 \
            --period-ms 60 --deadline-ms 35 \
            --timeout-ms 5 --drain-guard-ms 2 \
            --fault-every 100000 --seed "$seed" \
            "${lifecycle_args[@]}" \
            "$lifecycle_arg" &
        controller_pid=$!
        if [[ "$condition" == mps_ack_only ]]; then
            wait "$worker_pid"
            worker_pid=""
            shifter_gpu python3 - "$launcher_output" <<'PY'
import json, os, sys, time
from pathlib import Path
target = Path(sys.argv[1])
temporary = target.with_suffix(target.suffix + '.tmp')
temporary.write_text(json.dumps({
    'worker_state': 'exited',
    'confirmed_ns': time.perf_counter_ns(),
}), encoding='utf-8')
os.replace(temporary, target)
PY
            wait "$controller_pid"
            controller_pid=""
            socket=""
            echo "confirm52 done round=$round condition=$condition"
            continue
        fi
        for _ in {1..1200}; do
            [[ -f "$ready_file" ]] && break
            kill -0 "$controller_pid" 2>/dev/null || {
                echo "confirm52 RAN controller exited before lifecycle ready" >&2
                exit 1
            }
            sleep 0.05
        done
        [[ -f "$ready_file" ]] || { echo 'confirm52 lifecycle readiness timeout' >&2; exit 1; }
        if [[ "$condition" == mps_idle ]]; then
            kill -0 "$worker_pid" 2>/dev/null || {
                echo "confirm52 idle MPS worker died before RAN epoch" >&2
                exit 1
            }
            worker_state=alive
        else
            wait "$worker_pid"
            worker_pid=""
            worker_state=exited
        fi
        shifter_gpu python3 - "$start_file" "$worker_state" <<'PY'
import json, os, sys, time
from pathlib import Path
target = Path(sys.argv[1])
temporary = target.with_suffix(target.suffix + '.tmp')
temporary.write_text(json.dumps({
    'worker_state': sys.argv[2],
    'confirmed_ns': time.perf_counter_ns(),
}), encoding='utf-8')
os.replace(temporary, target)
PY
        wait "$controller_pid"
        controller_pid=""
        if [[ -n "$worker_pid" ]]; then
            wait "$worker_pid"
            worker_pid=""
        fi
        socket=""
        echo "confirm52 done round=$round condition=$condition"
    done
done

shifter_gpu python3 /softwall/analyze_confirm52_mps_idle_vs_retired.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm52_mps_idle_vs_retired_protocol.json" \
    --output "$result/confirm52_mps_idle_vs_retired.md"

cleanup
trap - EXIT INT TERM
echo 'confirm52 MPS idle versus retired completed'
