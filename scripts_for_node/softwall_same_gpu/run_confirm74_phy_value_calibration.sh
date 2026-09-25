#!/usr/bin/env bash

# Disjoint synthetic PHY-value train/test traces; no online policy in this arm.
set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

trials=${SOFTWALL_VALUE_TRIALS:-5000}
train_seed=${SOFTWALL_VALUE_TRAIN_SEED:-20800001}
test_seed=${SOFTWALL_VALUE_TEST_SEED:-20810001}
prefix=${SOFTWALL_VALUE_PREFIX:-confirm74_value_job${SLURM_JOB_ID}}
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}
trap 'rmdir "$gpu_lock_dir" 2>/dev/null || true' EXIT INT TERM
python3.11 - "$raw/${prefix}_manifest.json" "$trials" "$train_seed" "$test_seed" <<'PY'
import json, os, platform, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "trials": int(sys.argv[2]), "train_seed": int(sys.argv[3]),
    "test_seed": int(sys.argv[4]),
}, indent=2) + "\n")
PY
for split in train test; do
    if [[ "$split" == train ]]; then seed=$train_seed; else seed=$test_seed; fi
    shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/sweep_dual_receiver_snr.py \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --snr-db=-8.5 --trials "$trials" --seed "$seed" \
        --noise-reference pre_fading --record-observed-features \
        --prewarm-observed-features \
        --output "$raw/${prefix}_${split}.json"
done
echo "confirm74 disjoint PHY-value traces completed"
