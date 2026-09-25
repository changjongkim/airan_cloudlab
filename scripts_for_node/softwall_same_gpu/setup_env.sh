#!/usr/bin/env bash
set -Eeuo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

mkdir -p "$SOFTWALL_ROOT/third_party" \
    "$SOFTWALL_ROOT/runtime/softwall_same_gpu/engines" \
    "$SOFTWALL_ROOT/run_state/softwall_same_gpu" \
    "$SOFTWALL_ROOT/results/softwall_same_gpu/raw"

if [[ ! -d "$AERIAL_REPO/.git" ]]; then
    git clone --depth 1 --branch 25.3.2 --recurse-submodules --shallow-submodules \
        https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git "$AERIAL_REPO"
fi

actual_commit=$(git -C "$AERIAL_REPO" rev-parse HEAD)
expected_commit=3bf76a43dceb493b00f2ee75fdfbb87038eab7c6
[[ "$actual_commit" == "$expected_commit" ]] || {
    echo "Aerial commit mismatch: $actual_commit" >&2
    exit 3
}

binding="$AERIAL_REPO/pyaerial/src/aerial/pycuphy/_pycuphy.cpython-310-x86_64-linux-gnu.so"
if [[ ! -s "$binding" ]]; then
    shifter_gpu --workdir=/opt/nvidia/cuBB bash -c \
        'cmake -B build -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON && cmake --build build -j 32 -t _pycuphy pycuphycpp'
fi

shifter_gpu python3 -c \
    'import cupy; import aerial; from aerial.phy5g.algorithms import ChannelEstimator; x=cupy.ones(1024,dtype=cupy.float32); assert float(cupy.sum(x).get()) == 1024.0; print("SOFTWALL_ENV_OK")'

manifest="$SOFTWALL_ROOT/run_state/softwall_same_gpu/environment_${SLURM_JOB_ID}.txt"
{
    date -u +started_utc=%FT%TZ
    echo "slurm_job_id=$SLURM_JOB_ID"
    echo "node=$SLURM_JOB_NODELIST"
    echo "aerial_commit=$actual_commit"
    echo "aerial_image=$AERIAL_IMAGE"
    shifterimg images | grep -F "$AERIAL_IMAGE"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,mig.mode.current,compute_mode \
        --format=csv,noheader
} >"$manifest"
echo "environment manifest: $manifest"
