#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure

raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
result="$SOFTWALL_ROOT/results/softwall_same_gpu"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir -p "$raw"
if [[ "${SOFTWALL_GPU_LOCK_PREHELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || {
        echo "pre-held GPU0 lock is absent" >&2
        exit 4
    }
else
    if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    fi
fi
cleanup() {
    softwall_mps_configure || true
    softwall_mps_stop || true
    rm -f "$gpu_lock_dir/owner"
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Keep the MPS control plane and every dependent CUDA client in this same
# persistent Slurm step. A daemon created by a short step is cleaned up by Slurm.
softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

requal="$raw/confirm35_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 20 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321936 --snr-db -8.5 --channel-seed-base 20325900
python3 - "$requal" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if len(data["records"]) != 200 or data["deadline_misses"] != 0:
    raise SystemExit("confirm35 operational requalification failed")
print("confirm35 operational requalification passed")
PY
sleep 20

SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 PERIOD_MS=90 DEADLINE_MS=80 \
ENDPOINT_TIMEOUT_MS=30 WARMUP=20 SEED=20321035 SNR_DB=-8.5 \
CHANNEL_SEED_BASE=20325000 \
    bash "$SOFTWALL_SCRIPTS/run_same_request_ipc.sh" \
    confirm35_same_request_natural_s2 10000
sleep 20

SOFTWALL_GPU_LOCK_HELD=1 ITERATIONS=10000 PERIOD_MS=90 DEADLINE_MS=80 \
SNR_DB=-8.5 SEED=20321035 CHANNEL_SEED_BASE=20325000 QUIET_SECONDS=20 \
    bash "$SOFTWALL_SCRIPTS/run_external_endpoint_baselines.sh" \
    confirm35_same_node_baselines

shifter_gpu python3 /softwall/analyze_same_request_ipc.py \
    --raw "$raw" --campaign confirm35_same_request_natural_s2 \
    --output "$result/confirm35_same_request_natural_s2.md"

external="$raw/confirm35_same_request_natural_s2_cap80_job${SLURM_JOB_ID}_controller.json"
external_worker="$raw/confirm35_same_request_natural_s2_cap80_job${SLURM_JOB_ID}_worker.json"
shifter_gpu python3 /softwall/analyze_external_endpoint_baselines.py \
    --raw "$raw" --campaign confirm35_same_node_baselines \
    --external-controller "$external" \
    --external-worker "$external_worker" \
    --protocol "$result/confirm35_same_node_full_comparison_protocol.json" \
    --output "$result/confirm35_same_node_baselines.md" --margin-pp 0.25

cleanup
trap - EXIT INT TERM
echo "confirm35 same-node full comparison completed"
