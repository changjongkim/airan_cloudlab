#!/usr/bin/env bash
# Focused heavy workload comparison: MIG vs no-mig with Qwen-7B / HBM 16GB on 8T8R L1.
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
RS="$SCRIPT_DIR/run_sweep_v2.sh"
mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

# Make sure cuPHY 8T8R works; if not, fallback to 4T4R config but big PRB
mark "ENSURE MIG MODE ON"
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader
state=$(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader)
if [[ "$state" != "Enabled" ]]; then
  sudo systemctl stop nvidia-persistenced 2>&1 || true
  sudo rmmod nvidia_drm 2>&1 || true
  sudo rmmod nvidia_modeset 2>&1 || true
  sudo rmmod nvidia_uvm 2>&1 || true
  sudo nvidia-smi -i 0 -mig 1
  sudo modprobe nvidia_uvm
  sudo systemctl start nvidia-persistenced
  sudo systemctl restart docker
  sleep 5
fi

mark "MIG split-50-50 + Qwen-7B + 8T8R L1"
env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-50-50' DURATION=120 \
  bash $RS > heavy_A_qwen7b_mig.log 2>&1

mark "MIG split-50-50 + HBM 16GB + 8T8R L1"
env AI=hbm HBM_ALLOC=16.0 L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-50-50' DURATION=120 \
  bash $RS > heavy_B_hbm16_mig.log 2>&1

# Disable MIG for no-mig comparison
mark "DISABLE MIG MODE"
sudo nvidia-smi mig -i 0 -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i 0 -dgi >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>&1 || true
sudo rmmod nvidia_drm 2>&1 || true
sudo rmmod nvidia_modeset 2>&1 || true
sudo rmmod nvidia_uvm 2>&1 || true
sudo nvidia-smi -i 0 -mig 0
sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
sudo systemctl restart docker
sleep 5

mark "no-mig + Qwen-7B + 8T8R L1"
env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='no-mig' DURATION=120 \
  bash $RS > heavy_C_qwen7b_nomig.log 2>&1

mark "no-mig + HBM 16GB + 8T8R L1"
env AI=hbm HBM_ALLOC=16.0 L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='no-mig' DURATION=120 \
  bash $RS > heavy_D_hbm16_nomig.log 2>&1

mark "DONE"
ls -t results/ | head -10
