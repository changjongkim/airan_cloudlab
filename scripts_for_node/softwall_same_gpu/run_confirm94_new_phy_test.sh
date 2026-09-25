#!/usr/bin/env bash

# Independent synthetic PHY trace for a prospectively fixed comparator screen.
set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

seed=${SOFTWALL_PHY_SEED:?}
trials=${SOFTWALL_PHY_TRIALS:?}
prefix=${SOFTWALL_PHY_PREFIX:?}
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}
trap 'rmdir "$gpu_lock_dir" 2>/dev/null || true' EXIT INT TERM

python3.11 - "$raw/${prefix}_manifest.json" "$trials" "$seed" <<'PY'
import json, os, platform, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "host": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "trials_per_snr": int(sys.argv[2]),
    "seed": int(sys.argv[3]),
    "snrs_db": [-10.0, -9.0, -8.0, -7.0],
}, indent=2) + "\n")
PY

shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/sweep_dual_receiver_snr.py \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --snr-db=-10,-9,-8,-7 --trials "$trials" --seed "$seed" \
    --noise-reference pre_fading --record-observed-features \
    --prewarm-observed-features \
    --output "$raw/${prefix}_test.json"

echo "confirm94 independent mixed-SNR PHY trace completed"
