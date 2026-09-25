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

requal="$raw/confirm47_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 20 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321947 --snr-db -8.5 --channel-seed-base 20337900
python3 - "$requal" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if data['iterations'] != 200 or data['deadline_misses']:
    raise SystemExit('confirm47 operational requalification failed')
print('confirm47 operational requalification passed')
PY

for round in 1 2; do
    case "$round" in
        1) seed=20321047; channel=20337000; order="eager external" ;;
        2) seed=20321048; channel=20338000; order="external eager" ;;
    esac
    for condition in $order; do
        sleep 20
        if [[ "$condition" == eager ]]; then
            POLICIES=eager_dual CELLS=1 PERIOD_MS=90 DEADLINE_MS=80 \
            NRX_BOUND_MS=20 CONV_BOUND_MS=25 COMMIT_GUARD_MS=2 \
            FAILURE_EVERY=0 SNR_DB=-8.5 SEED_BASE="$((seed - 1))" \
            CHANNEL_SEED_BASE="$channel" CHANNEL_SEED_STRIDE=0 \
            BACKGROUND_CAP=20 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
            AI_RPC_TIMEOUT_MS=35 QUIET_SECONDS=0 \
                bash "$SOFTWALL_SCRIPTS/run_s2_campaign.sh" \
                1 10000 "confirm47_r${round}_eager_job${SLURM_JOB_ID}"
        else
            SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
            BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
            AI_RPC_TIMEOUT_MS=35 PERIOD_MS=90 DEADLINE_MS=80 \
            ENDPOINT_TIMEOUT_MS=30 NRX_BOUND_MS=50 CONV_BOUND_MS=25 \
            COMMIT_GUARD_MS=2 RECOVERY_GAP_AI=1 WARMUP=20 SEED="$seed" \
            SNR_DB=-8.5 CHANNEL_SEED_BASE="$channel" \
                bash "$SOFTWALL_SCRIPTS/run_same_request_transaction.sh" \
                "confirm47_r${round}_external" 10000
        fi
    done
done

shifter_gpu python3 /softwall/analyze_transaction_vs_eager_bound50.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm47_conservative_bound_protocol.json" \
    --output "$result/confirm47_conservative_bound.md"

cleanup
trap - EXIT INT TERM
echo 'confirm47 conservative-bound transactional external vs eager completed'
