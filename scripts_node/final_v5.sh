#!/usr/bin/env bash
# Final: re-enable MIG, run C3 reproducibility (3x split-60-40) + cell-count on split-50-50
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
RS="$SCRIPT_DIR/run_sweep_v2.sh"
mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

mark "ENABLE MIG"
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo rmmod nvidia_drm 2>/dev/null || true
sudo rmmod nvidia_modeset 2>/dev/null || true
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo nvidia-smi -i 0 -mig 1
sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
sudo systemctl restart docker
sleep 10
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader

# C3 reproducibility x3
for i in 1 2 3; do
  mark "C3_repro_$i: split-60-40 + Qwen"
  env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=20 PRESETS='split-60-40' DURATION=90 \
    bash $RS > v5_C3_repro${i}.log 2>&1
done

# Cell-count on split-50-50 + Qwen (cells 1, 4, 10, 20)
for c in 1 4 10 20; do
  mark "cellcount c=$c on split-50-50 + Qwen"
  env AI=qwen7b L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS=$c PRESETS='split-50-50' DURATION=90 \
    bash $RS > v5_cells${c}.log 2>&1
done

mark "v5 DONE"
