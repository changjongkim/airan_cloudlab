#!/usr/bin/env bash
# Baselines A, B, C with heavy workload (8T8R cuPHY L1, Qwen-7B / HBM 16GB)
# A. L1 alone, full GPU (no MIG, no AI)
# B. L1 alone, MIG 3g.20gb (no AI)
# C. L1 + AI on various MIG split ratios
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
RS="$SCRIPT_DIR/run_sweep_v2.sh"
mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

# Step 0: MIG should be ON already. Verify.
state=$(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader)
echo "MIG mode = $state"

# B: L1 alone on 3g.20gb (MIG mode already on)
mark "B: L1 alone on MIG 3g.20gb (no AI), 8T8R, 20 cells"
env AI=none L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-50-50' DURATION=15 \
  bash $RS > v3_B_L1_only_mig.log 2>&1

# C: various MIG split ratios with Qwen
mark "C1: split-40-60 (L1=2g.10gb) + Qwen + 8T8R"
env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-40-60' DURATION=90 \
  bash $RS > v3_C1_s4060_qwen.log 2>&1

mark "C2: split-50-50 + Qwen + 8T8R (already have, repeat for cleanliness)"
env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-50-50' DURATION=90 \
  bash $RS > v3_C2_s5050_qwen.log 2>&1

mark "C3: split-60-40 + Qwen + 8T8R"
env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-60-40' DURATION=90 \
  bash $RS > v3_C3_s6040_qwen.log 2>&1

# Step: Disable MIG for A baseline
mark "DISABLE MIG"
sudo nvidia-smi mig -i 0 -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i 0 -dgi >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo rmmod nvidia_drm 2>/dev/null || true
sudo rmmod nvidia_modeset 2>/dev/null || true
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo nvidia-smi -i 0 -mig 0
sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
sudo systemctl restart docker
sleep 5

# A: L1 alone on full GPU
mark "A: L1 alone on full GPU (no MIG, no AI), 8T8R, 20 cells"
env AI=none L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='no-mig' DURATION=15 \
  bash $RS > v3_A_L1_only_fullGPU.log 2>&1

# Bonus: no-mig + Qwen vs no-mig + HBM16 (we have one but need consistency)
mark "no-mig + Qwen + 8T8R (control)"
env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='no-mig' DURATION=90 \
  bash $RS > v3_nomig_qwen.log 2>&1

mark "ALL v3 DONE"
ls -t results/ | head -10
