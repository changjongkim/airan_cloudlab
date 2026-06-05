#!/usr/bin/env bash
# Perlmutter no-MIG 실험 — Setup
# 실행: bash 01_setup.sh

set -uo pipefail
AERIAL_IMAGE="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-$SCRATCH/aerial-cuda-accelerated-ran}"
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/airan_cloudlab/results/perlmutter_nomig}"

echo "=== Setup ==="

# 1. Aerial container pull
echo "--- 1) Pulling Aerial Shifter image ---"
if shifterimg images | grep -q "aerial-cuda-accelerated-ran"; then
    echo "Image already pulled."
else
    shifterimg pull "$AERIAL_IMAGE" 2>&1 | tail -3
    echo "Verify:"
    shifterimg images | grep aerial
fi

# 2. Aerial open-source repo
echo
echo "--- 2) Cloning NVIDIA Aerial open-source ---"
if [[ -d "$AERIAL_REPO/.git" ]]; then
    echo "Already cloned at $AERIAL_REPO"
    (cd "$AERIAL_REPO" && git pull && git lfs pull) 2>&1 | tail -3
else
    git clone --recurse-submodules https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git "$AERIAL_REPO"
    (cd "$AERIAL_REPO" && git lfs pull) 2>&1 | tail -3
fi

# 3. Results dir
echo
echo "--- 3) Creating results dir ---"
mkdir -p "$RESULTS_DIR"/{F_nomig,F_nomig_mps,NCU_nomig,P5_nomig,nsys_nomig,logs}
chmod -R 777 "$RESULTS_DIR" 2>/dev/null || true
echo "Results dir: $RESULTS_DIR"
ls "$RESULTS_DIR"

# 4. Test Shifter + Aerial Python
echo
echo "--- 4) Testing Aerial Python in Shifter ---"
shifter --image=$AERIAL_IMAGE --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    python3 -c "
import sys
try:
    from aerial.phy5g.algorithms import ChannelEstimator
    print('OK: pyaerial imports')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
" 2>&1

echo
echo "=== Setup complete ==="
echo "Env vars to use in experiment scripts:"
echo "  AERIAL_IMAGE=$AERIAL_IMAGE"
echo "  AERIAL_REPO=$AERIAL_REPO"
echo "  RESULTS_DIR=$RESULTS_DIR"
