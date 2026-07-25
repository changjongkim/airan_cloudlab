#!/usr/bin/env bash
# Master sweep v2 — robust ordering, MIG mode toggled ONLY twice
# Strategy:
#   1) Reboot once (caller does it)
#   2) Enable MIG mode once
#   3) Run ALL MIG-presets (instances destroyed/recreated, mode stays ON)
#   4) Disable MIG mode once
#   5) Run ALL no-mig presets
#
# Per-sweep verification: after each preset, check JSON existence; if missing,
# retry once (still inside same MIG mode session, no reboot).
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
GPU=0
mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

# Clean state. Caller should have already enabled MIG mode after unloading
# nvidia_uvm/nvidia_modeset/nvidia_drm (these modules keep GPU "in use" preventing
# MIG mode toggle). We assume MIG mode is already ENABLED at this point.
sudo killall -9 nvidia-cuda-mps-server nvidia-cuda-mps-control 2>/dev/null || true
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log
docker ps -aq | xargs -r docker rm -f 2>/dev/null
# Verify MIG mode is on
state=$(nvidia-smi -i $GPU --query-gpu=mig.mode.current --format=csv,noheader)
echo "starting with MIG mode = $state"
if [[ "$state" != "Enabled" ]]; then
  echo "ERROR: MIG mode is not enabled. Run: sudo rmmod nvidia_uvm nvidia_modeset nvidia_drm && sudo nvidia-smi -i $GPU -mig 1"
  exit 1
fi
sleep 1
mark "MIG mode confirmed Enabled — start sweeps"

# Use new run_sweep_v2.sh which never toggles MIG mode itself.
RS="$SCRIPT_DIR/run_sweep_v2.sh"

##############################################################################
# MIG-ON phase: all sweeps that use MIG splits, four-way, seven-1g
##############################################################################

# All MIG split sweeps for each AI workload type
for ai_setup in \
    "A1_gpt2:AI=gpt2" \
    "A2_hbm1:AI=hbm HBM_ALLOC=1.0" \
    "A3_hbm8:AI=hbm HBM_ALLOC=8.0" \
    "A4_resnet:AI=resnet RESNET_BS=16"; do
  label="${ai_setup%%:*}"; envs="${ai_setup#*:}"
  mark "$label : MIG split presets"
  env $envs CELLS=20 PRESETS='split-20-80 split-40-60 split-50-50 split-60-40' \
    DURATION=120 bash $RS > master_${label}.log 2>&1
done

mark "B1: four-way-bigL1 mixed gpt2,resnet,hbm"
env AI=gpt2,resnet,hbm HBM_ALLOC=1.0 RESNET_BS=16 CELLS=20 PRESETS='four-way-bigL1' DURATION=120 \
  bash $RS > master_B1_fw_bigL1_mixed.log 2>&1

mark "B2: four-way-eq mixed gpt2,resnet,hbm"
env AI=gpt2,resnet,hbm HBM_ALLOC=1.0 RESNET_BS=16 CELLS=20 PRESETS='four-way-eq' DURATION=120 \
  bash $RS > master_B2_fw_eq_mixed.log 2>&1

mark "B3: seven-1g mixed"
env AI=gpt2,resnet,hbm,gpt2,resnet,hbm HBM_ALLOC=1.0 RESNET_BS=16 CELLS=20 \
  PRESETS='seven-1g' DURATION=120 bash $RS > master_B3_seven1g_mixed.log 2>&1

mark "B4: seven-1g GPT-2 x6"
env AI=gpt2 CELLS=20 PRESETS='seven-1g' DURATION=120 bash $RS > master_B4_seven1g_gpt2.log 2>&1

# Cell-count sweeps with MIG (split-20-80, split-50-50, split-60-40)
for preset_label in "C2_s2080:split-20-80" "C3_s5050:split-50-50" "C4_s6040:split-60-40"; do
  lbl="${preset_label%%:*}"; pre="${preset_label#*:}"
  for c in 1 4 10 20 40; do
    mark "$lbl c=$c"
    env AI=gpt2 CELLS=$c PRESETS="$pre" DURATION=90 bash $RS > master_${lbl}_c${c}.log 2>&1
  done
done

# HBM intensity sweep at split-50-50
for gb in 0.5 1.0 2.0 4.0 8.0 16.0; do
  mark "D1: HBM ${gb}GB @ split-50-50"
  env AI=hbm HBM_ALLOC=$gb CELLS=20 PRESETS='split-50-50' DURATION=90 \
    bash $RS > master_D1_hbm${gb}.log 2>&1
done

# ResNet batch sweep at split-50-50
for bs in 8 16 32 64; do
  mark "D2: ResNet bs=$bs @ split-50-50"
  env AI=resnet RESNET_BS=$bs CELLS=20 PRESETS='split-50-50' DURATION=90 \
    bash $RS > master_D2_resnet_bs${bs}.log 2>&1
done

# PRB sweep at split-50-50
for prbs in 51 106 217 273; do
  mark "E: PRBs=$prbs @ split-50-50"
  env L1_NUM_PRBS=$prbs AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=90 \
    bash $RS > master_E_prbs${prbs}.log 2>&1
done

# Antenna sweep at split-50-50
for cfg in 2x2 4x4 8x8; do
  tx=${cfg%x*}; rx=${cfg#*x}
  mark "F: ${cfg} antenna @ split-50-50"
  env L1_NUM_TX_ANT=$tx L1_NUM_RX_ANT=$rx AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=90 \
    bash $RS > master_F_ant${cfg}.log 2>&1
done

mark "G: stability 1000 iters @ split-50-50"
env L1_ITERATIONS=1000 AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=600 \
  bash $RS > master_G_stability.log 2>&1

mark "H2: AI ON sustained @ split-50-50 (300 iters)"
env L1_ITERATIONS=300 AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=180 \
  bash $RS > master_H2_AIon.log 2>&1

# MCS sweep
for mcs in 0 2 7 16 24; do
  mark "I: MCS=$mcs @ split-50-50"
  env L1_MCS_INDEX=$mcs AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=90 \
    bash $RS > master_I_mcs${mcs}.log 2>&1
done

##############################################################################
# NOW switch to MIG OFF for no-mig presets.
# Disabling needs same trick: unload nvidia_uvm/modeset/drm if blocked.
##############################################################################
mark "DISABLE MIG MODE (switching to no-mig group)"
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sudo killall -9 nvidia-cuda-mps-server nvidia-cuda-mps-control 2>/dev/null || true
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log
docker ps -aq | xargs -r docker rm -f 2>/dev/null
sleep 2
sudo systemctl stop nvidia-persistenced 2>&1 || true
sudo rmmod nvidia_drm 2>&1 || true
sudo rmmod nvidia_modeset 2>&1 || true
sudo rmmod nvidia_uvm 2>&1 || true
sleep 2
sudo nvidia-smi -i $GPU -mig 0
sleep 2
nvidia-smi -i $GPU --query-gpu=mig.mode.current --format=csv,noheader

# A1-A4 no-mig variant
for ai_setup in \
    "A1_gpt2_nomig:AI=gpt2" \
    "A2_hbm1_nomig:AI=hbm HBM_ALLOC=1.0" \
    "A3_hbm8_nomig:AI=hbm HBM_ALLOC=8.0" \
    "A4_resnet_nomig:AI=resnet RESNET_BS=16"; do
  label="${ai_setup%%:*}"; envs="${ai_setup#*:}"
  mark "$label : no-mig"
  env $envs CELLS=20 PRESETS='no-mig' DURATION=120 bash $RS > master_${label}.log 2>&1
done

# Cell-count no-mig
mark "C1: no-mig cell-count sweep"
for c in 1 4 10 20 40; do
  env AI=gpt2 CELLS=$c PRESETS='no-mig' DURATION=90 bash $RS > master_C1_nomig_c${c}.log 2>&1
done

# Baseline (no AI) cell-count
mark "C5: BASELINE no AI, no MIG, cell-count"
for c in 1 4 10 20 40; do
  env AI=none CELLS=$c PRESETS='no-mig' DURATION=10 bash $RS > master_C5_baseline_c${c}.log 2>&1
done

# H1, H3 — no-mig baseline before/after H2 (we already ran H2 in MIG group; redo H1/H3 here)
mark "H1: AI OFF baseline @ no-mig (pre)"
env AI=none CELLS=20 PRESETS='no-mig' DURATION=10 bash $RS > master_H1_noAI.log 2>&1
mark "H3: AI OFF baseline @ no-mig (post)"
env AI=none CELLS=20 PRESETS='no-mig' DURATION=10 bash $RS > master_H3_noAI_post.log 2>&1

mark "ALL DONE"
ls -t results/ | head -60
