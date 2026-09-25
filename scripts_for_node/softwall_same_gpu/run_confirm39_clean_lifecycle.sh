#!/usr/bin/env bash

# Re-run the previously contaminated lifecycle gates under one exclusive GPU lock.
set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
result="$SOFTWALL_ROOT/results/softwall_same_gpu"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
if [[ "${SOFTWALL_GPU_LOCK_PREHELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || { echo 'pre-held GPU0 lock is absent' >&2; exit 4; }
else
    mkdir "$gpu_lock_dir" 2>/dev/null || {
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    }
fi
cleanup() {
    softwall_mps_stop || true
    rm -f "$gpu_lock_dir/owner"
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

# Fresh epoch operational smoke: do not include these releases in either gate.
requal="$raw/confirm39_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 60 --deadline-ms 35 --nrx-bound-ms 50 \
    --conv-bound-ms 35 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321937 --channel-seed-base 20325937
python3 - "$requal" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if len(d['records']) != 200 or d['deadline_misses']:
    raise SystemExit('confirm39 fresh-epoch requalification failed')
print('confirm39 fresh-epoch requalification passed')
PY
sleep 20

SOFTWALL_GPU_LOCK_HELD=1 CAPS=100,20 FAULT_EVERY=100000 \
RETIRE_BEFORE_FIRST_RELEASE=1 PERIOD_MS=60 DEADLINE_MS=35 \
TIMEOUT_MS=5 DRAIN_GUARD_MS=2 OVERRUN_REPEATS=40 \
SEED_BASE=20320000 \
    bash "$SOFTWALL_SCRIPTS/run_overrun_campaign.sh" \
    5 1000 confirm39_clean_sham_retirement

shifter_gpu python3 /softwall/analyze_overrun.py \
    --raw "$raw" --campaign confirm39_clean_sham_retirement \
    --output "$result/confirm39_clean_sham_retirement.md"

# Remove the previous worker's MPS server state before the separate maintenance gate.
softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert
sleep 20

SOFTWALL_GPU_LOCK_HELD=1 CAPS=100,20 PRE_ITERATIONS=100 \
POST_ITERATIONS=1000 QUIET_SECONDS=20 OVERRUN_REPEATS=40 \
    bash "$SOFTWALL_SCRIPTS/run_maintenance_requalification.sh" \
    3 confirm39_clean_maintenance_requalification

shifter_gpu python3 /softwall/analyze_maintenance_requalification.py \
    --raw "$raw" --campaign confirm39_clean_maintenance_requalification \
    --output "$result/confirm39_clean_maintenance_requalification.md"

shifter_gpu python3 /softwall/analyze_confirm39_lifecycle.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm39_clean_lifecycle_protocol.json" \
    --output "$result/confirm39_clean_lifecycle_audit.md"

cleanup
trap - EXIT INT TERM
echo 'confirm39 clean lifecycle gates completed'
