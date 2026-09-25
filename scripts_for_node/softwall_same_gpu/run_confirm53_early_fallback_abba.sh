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

requal="$raw/confirm53_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 50 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321953 --snr-db -8.5 --channel-seed-base 20359900
python3 - "$requal" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x['iterations'] != 200 or x['deadline_misses']:
    raise SystemExit('confirm53 operational requalification failed')
print('confirm53 operational requalification passed', flush=True)
PY

for arm in 1 2 3 4; do
    case "$arm" in
        1) timing=early_if_clear; seed=20321053; channel=20353000 ;;
        2) timing=latest; seed=20321053; channel=20353000 ;;
        3) timing=latest; seed=20321054; channel=20354000 ;;
        4) timing=early_if_clear; seed=20321054; channel=20354000 ;;
    esac
    echo "confirm53 start arm=$arm timing=$timing"
    sleep 20
    SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
    BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 \
    AI_GUARD_MS=2 AI_RPC_TIMEOUT_MS=35 PERIOD_MS=45 \
    DEADLINE_MS=80 ENDPOINT_TIMEOUT_MS=30 NRX_BOUND_MS=50 \
    CONV_BOUND_MS=25 COMMIT_GUARD_MS=2 RECOVERY_GAP_AI=1 \
    FALLBACK_TIMING="$timing" WARMUP=20 SEED="$seed" SNR_DB=-8.5 \
    CHANNEL_SEED_BASE="$channel" \
        bash "$SOFTWALL_SCRIPTS/run_same_request_transaction.sh" \
        "confirm53_arm${arm}_${timing}_job${SLURM_JOB_ID}" 1000
    echo "confirm53 done arm=$arm timing=$timing"
    if [[ "$arm" == 2 || "$arm" == 4 ]]; then
        shifter_gpu python3 /softwall/analyze_confirm53_early_fallback_abba.py \
            --raw "$raw" --job "$SLURM_JOB_ID" \
            --protocol "$result/confirm53_early_fallback_abba_protocol.json" \
            --completed-arms "$arm" \
            --output "$result/confirm53_early_fallback_abba.md"
        if [[ "$arm" == 2 ]]; then
            first_pass=$(python3 - "$result/confirm53_early_fallback_abba.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print('1' if report['pair_gates']['pair1'] and not report['errors'] else '0')
PY
            )
            if [[ "$first_pass" == 0 ]]; then
                echo 'confirm53 pair1 frozen gate FAIL; stopping before pair2'
                exit 1
            fi
        fi
    fi
done

cleanup
trap - EXIT INT TERM
echo 'confirm53 early fallback ABBA completed'
