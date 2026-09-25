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

requal="$raw/confirm42_transaction_gap_smoke_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 30 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321942 --snr-db -8.5 --channel-seed-base 20332900
python3 - "$requal" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result['iterations'] != 200 or result['deadline_misses']:
    raise SystemExit('confirm42 operational requalification failed')
print('confirm42 operational requalification passed')
PY
sleep 20

SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
AI_RPC_TIMEOUT_MS=45 PERIOD_MS=90 DEADLINE_MS=80 \
ENDPOINT_TIMEOUT_MS=30 NRX_BOUND_MS=30 CONV_BOUND_MS=25 \
COMMIT_GUARD_MS=2 RECOVERY_GAP_AI=1 WARMUP=20 SEED=20321042 \
SNR_DB=-8.5 CHANNEL_SEED_BASE=20332000 \
    bash "$SOFTWALL_SCRIPTS/run_same_request_transaction.sh" \
    confirm42_transaction_gap_smoke 500

shifter_gpu python3 /softwall/analyze_transaction_smoke.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm42_transaction_gap_smoke_protocol.json" \
    --output "$result/confirm42_transaction_gap_smoke.md"

cleanup
trap - EXIT INT TERM
echo 'confirm42 transactional gap smoke completed'
