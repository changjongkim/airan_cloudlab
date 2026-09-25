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

requal="$raw/confirm50_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 50 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321950 --snr-db -8.5 --channel-seed-base 20341900
python3 - "$requal" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if data['iterations'] != 200 or data['deadline_misses']:
    raise SystemExit('confirm50 operational requalification failed')
print('confirm50 operational requalification passed')
PY

for round in 1 2 3 4; do
    case "$round" in
        1) mode=off; gap=0; seed=20321050; channel=20340000 ;;
        2) mode=on;  gap=1; seed=20321050; channel=20340000 ;;
        3) mode=on;  gap=1; seed=20321051; channel=20341000 ;;
        4) mode=off; gap=0; seed=20321051; channel=20341000 ;;
    esac
    sleep 20
    SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
    BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
    AI_RPC_TIMEOUT_MS=35 PERIOD_MS=90 DEADLINE_MS=80 \
    ENDPOINT_TIMEOUT_MS=30 NRX_BOUND_MS=50 CONV_BOUND_MS=25 \
    COMMIT_GUARD_MS=2 RECOVERY_GAP_AI="$gap" WARMUP=20 SEED="$seed" \
    SNR_DB=-8.5 CHANNEL_SEED_BASE="$channel" \
        bash "$SOFTWALL_SCRIPTS/run_same_request_transaction.sh" \
        "confirm50_r${round}_${mode}" 10000
done

shifter_gpu python3 /softwall/analyze_confirm50_requalified_gap_abba.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm50_requalified_gap_abba_protocol.json" \
    --output "$result/confirm50_requalified_gap_abba.md"

cleanup
trap - EXIT INT TERM
echo 'confirm50 requalified recovery-gap ABBA completed'
