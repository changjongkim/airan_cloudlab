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
cleanup() {
    softwall_mps_stop || true
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

requal="$raw/confirm48_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 30 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321948 --snr-db -8.5 --channel-seed-base 20339900
python3 - "$requal" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x['iterations'] != 200 or x['deadline_misses']:
    raise SystemExit('confirm48 operational requalification failed')
print('confirm48 operational requalification passed')
PY

early_stopped=0
for round in 1 2 3 4; do
    case "$round" in
        1|4) gc_mode=on ;;
        2|3) gc_mode=off ;;
    esac
    sleep 20
    SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
    BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
    AI_RPC_TIMEOUT_MS=35 PERIOD_MS=90 DEADLINE_MS=80 \
    ENDPOINT_TIMEOUT_MS=30 NRX_BOUND_MS=30 CONV_BOUND_MS=25 \
    COMMIT_GUARD_MS=2 RECOVERY_GAP_AI=1 GC_MODE="$gc_mode" \
    WARMUP=20 SEED=20321148 SNR_DB=-8.5 CHANNEL_SEED_BASE=20339000 \
        bash "$SOFTWALL_SCRIPTS/run_same_request_transaction_gc_probe.sh" \
        "confirm48_gc_r${round}_${gc_mode}" 4000
    if shifter_gpu python3 /softwall/analyze_confirm48_gc_tail_abba.py \
        --raw "$raw" --job "$SLURM_JOB_ID" \
        --protocol "$result/confirm48_gc_tail_abba_protocol.json" \
        --completed-rounds "$round" \
        --output "$result/confirm48_gc_tail_abba.md"; then
        :
    fi
    stop_now=$(python3 - "$result/confirm48_gc_tail_abba.json" <<'PY'
import json, sys
gates = json.load(open(sys.argv[1], encoding='utf-8'))['gates']
ignored = {'full_frozen_order_completed', 'all_pass'}
failed = [name for name, passed in gates.items() if name not in ignored and not passed]
print('1' if failed else '0')
PY
    )
    if [[ "$stop_now" == 1 ]]; then
        echo "confirm48 frozen gate impossible after round=$round; stopping early"
        early_stopped=1
        break
    fi
done

if [[ "$early_stopped" == 1 ]]; then
    echo 'confirm48 early-stop diagnostic complete; frozen hypothesis FAIL'
    exit 1
fi

cleanup
trap - EXIT INT TERM
echo 'confirm48 GC tail ABBA completed'
