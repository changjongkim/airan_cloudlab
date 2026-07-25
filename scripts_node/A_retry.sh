#!/usr/bin/env bash
# Retry A baseline (L1 alone, full GPU, no MIG, no AI)
# Runs after MIG fully disabled and driver stable
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
GPU=0

# Ensure MIG mode is OFF
state=$(nvidia-smi -i $GPU --query-gpu=mig.mode.current --format=csv,noheader)
echo "current MIG state = $state"
if [[ "$state" == "Enabled" ]]; then
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo systemctl stop nvidia-persistenced 2>/dev/null || true
  sudo rmmod nvidia_drm 2>/dev/null || true
  sudo rmmod nvidia_modeset 2>/dev/null || true
  sudo rmmod nvidia_uvm 2>/dev/null || true
  sudo nvidia-smi -i $GPU -mig 0
  sudo modprobe nvidia_uvm
  sudo systemctl start nvidia-persistenced
  sudo systemctl restart docker
  sleep 15  # longer wait for daemon
fi
echo "MIG now = $(nvidia-smi -i $GPU --query-gpu=mig.mode.current --format=csv,noheader)"

# Verify docker GPU works before running L1
timeout 30 docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L 2>&1 | head -3

echo "=== A: L1 alone on full GPU (no MIG, no AI), 8T8R, 20 cells ==="
env AI=none L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='no-mig' DURATION=15 \
  bash "$SCRIPT_DIR/run_sweep_v2.sh" > v3_A_retry.log 2>&1

TS=$(grep -oP 'results/\K[0-9_]+' v3_A_retry.log | tail -1)
echo "TS=$TS"
python3 /tmp/check.py "$SCRIPT_DIR/results/$TS"
echo "A retry DONE"
