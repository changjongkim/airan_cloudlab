#!/usr/bin/env bash
set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
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

requal="$raw/confirm51_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 20 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321951 --snr-db -8.5 --channel-seed-base 20351900
python3 - "$requal" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if data['iterations'] != 200 or data['deadline_misses']:
    raise SystemExit('confirm51 operational requalification failed')
print('confirm51 operational requalification passed', flush=True)
PY

for round in 1 2; do
    case "$round" in
        1) seed=20321051; channel=20351000; order="conventional_only eager_dual external_transaction" ;;
        2) seed=20321052; channel=20352000; order="external_transaction eager_dual conventional_only" ;;
    esac
    for period in 90 45 25 12; do
        for condition in $order; do
            echo "confirm51 start round=$round period=$period policy=$condition" 
            sleep 20
            if [[ "$condition" == external_transaction ]]; then
                SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
                BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 \
                AI_GUARD_MS=2 AI_RPC_TIMEOUT_MS=35 PERIOD_MS="$period" \
                DEADLINE_MS=80 ENDPOINT_TIMEOUT_MS=30 NRX_BOUND_MS=50 \
                CONV_BOUND_MS=25 COMMIT_GUARD_MS=2 RECOVERY_GAP_AI=1 \
                WARMUP=20 SEED="$seed" SNR_DB=-8.5 \
                CHANNEL_SEED_BASE="$channel" \
                    bash "$SOFTWALL_SCRIPTS/run_same_request_transaction.sh" \
                    "confirm51_r${round}_p${period}_external_job${SLURM_JOB_ID}" 1000
            else
                POLICIES="$condition" CELLS=1 PERIOD_MS="$period" \
                DEADLINE_MS=80 NRX_BOUND_MS=20 CONV_BOUND_MS=25 \
                COMMIT_GUARD_MS=2 FAILURE_EVERY=0 SNR_DB=-8.5 \
                SEED_BASE="$((seed - 1))" CHANNEL_SEED_BASE="$channel" \
                CHANNEL_SEED_STRIDE=0 BACKGROUND_CAP=20 AI_BUDGET_MS=40 \
                AI_GUARD_MS=2 AI_RPC_TIMEOUT_MS=35 QUIET_SECONDS=0 \
                    bash "$SOFTWALL_SCRIPTS/run_s2_campaign.sh" \
                    1 1000 "confirm51_r${round}_p${period}_${condition}_job${SLURM_JOB_ID}"
            fi
            echo "confirm51 done round=$round period=$period policy=$condition"
        done
    done
done

shifter_gpu python3 /softwall/analyze_confirm51_n1_single_cell.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$SOFTWALL_ROOT/results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic_protocol.json" \
    --output "$SOFTWALL_ROOT/results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic.md"

cleanup
trap - EXIT INT TERM
echo 'confirm51 N1 single-cell diagnostic completed'
