#!/usr/bin/env bash
# Master sweep — comprehensive real-cuPHY MIG experiments.
# Phases:
#   A: AI workload sweep × 5 MIG presets  (gpt2, hbm 1GB, hbm 8GB, resnet)
#   B: multi-AI configs (four-way-bigL1, four-way-eq, seven-1g mixed/uniform)
#   C: cell-count sweep at multiple MIG configs
#   D: AI intensity sweep at split-50-50
set -uo pipefail

cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

sudo "$SCRIPT_DIR/02_mig.sh" hard-reset || true
echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log

###########################################################################
# PHASE A — AI workload type sweep across 5 MIG presets
###########################################################################
mark "A1: GPT-2 × MIG-5"
AI=gpt2 CELLS=20 PRESETS='no-mig split-20-80 split-40-60 split-50-50 split-60-40' \
  DURATION=120 ./run_sweep.sh > master_A1_gpt2.log 2>&1

mark "A2: HBM 1GB × MIG-5"
AI=hbm HBM_ALLOC=1.0 CELLS=20 PRESETS='no-mig split-20-80 split-40-60 split-50-50 split-60-40' \
  DURATION=120 ./run_sweep.sh > master_A2_hbm1.log 2>&1

mark "A3: HBM 8GB × MIG-5"
AI=hbm HBM_ALLOC=8.0 CELLS=20 PRESETS='no-mig split-20-80 split-40-60 split-50-50 split-60-40' \
  DURATION=120 ./run_sweep.sh > master_A3_hbm8.log 2>&1

mark "A4: ResNet-50 × MIG-5"
AI=resnet RESNET_BS=16 CELLS=20 PRESETS='no-mig split-20-80 split-40-60 split-50-50 split-60-40' \
  DURATION=120 ./run_sweep.sh > master_A4_resnet.log 2>&1

###########################################################################
# PHASE B — multi-AI MIG configs (mixed workloads)
###########################################################################
mark "B1: four-way-bigL1 mixed gpt2,resnet,hbm"
AI=gpt2,resnet,hbm HBM_ALLOC=1.0 RESNET_BS=16 CELLS=20 PRESETS='four-way-bigL1' \
  DURATION=120 ./run_sweep.sh > master_B1_fw_bigL1_mixed.log 2>&1

mark "B2: four-way-eq mixed gpt2,resnet,hbm"
AI=gpt2,resnet,hbm HBM_ALLOC=1.0 RESNET_BS=16 CELLS=20 PRESETS='four-way-eq' \
  DURATION=120 ./run_sweep.sh > master_B2_fw_eq_mixed.log 2>&1

mark "B3: seven-1g mixed (gpt2,resnet,hbm) x2"
AI=gpt2,resnet,hbm,gpt2,resnet,hbm HBM_ALLOC=1.0 RESNET_BS=16 CELLS=20 \
  PRESETS='seven-1g' DURATION=120 ./run_sweep.sh > master_B3_seven1g_mixed.log 2>&1

mark "B4: seven-1g uniform GPT-2 x6"
AI=gpt2 CELLS=20 PRESETS='seven-1g' DURATION=120 \
  ./run_sweep.sh > master_B4_seven1g_gpt2.log 2>&1

###########################################################################
# PHASE C — cell-count sweep at multiple MIG configs
###########################################################################
mark "C1: GPT-2 + no-mig × cells {1,4,10,20,40}"
for c in 1 4 10 20 40; do
  AI=gpt2 CELLS=$c PRESETS='no-mig' DURATION=90 \
    ./run_sweep.sh > master_C1_nomig_c${c}.log 2>&1
done

mark "C2: GPT-2 + split-20-80 × cells {1,4,10,20,40}"
for c in 1 4 10 20 40; do
  AI=gpt2 CELLS=$c PRESETS='split-20-80' DURATION=90 \
    ./run_sweep.sh > master_C2_s2080_c${c}.log 2>&1
done

mark "C3: GPT-2 + split-50-50 × cells {1,4,10,20,40}"
for c in 1 4 10 20 40; do
  AI=gpt2 CELLS=$c PRESETS='split-50-50' DURATION=90 \
    ./run_sweep.sh > master_C3_s5050_c${c}.log 2>&1
done

mark "C4: GPT-2 + split-60-40 × cells {1,4,10,20,40}"
for c in 1 4 10 20 40; do
  AI=gpt2 CELLS=$c PRESETS='split-60-40' DURATION=90 \
    ./run_sweep.sh > master_C4_s6040_c${c}.log 2>&1
done

mark "C5: BASELINE (no AI) + no-mig × cells {1,4,10,20,40}"
for c in 1 4 10 20 40; do
  AI=none CELLS=$c PRESETS='no-mig' DURATION=10 \
    ./run_sweep.sh > master_C5_baseline_c${c}.log 2>&1
done

###########################################################################
# PHASE D — AI intensity sweep at split-50-50
###########################################################################
mark "D1: HBM alloc sweep @ split-50-50"
for gb in 0.5 1.0 2.0 4.0 8.0 16.0; do
  AI=hbm HBM_ALLOC=$gb CELLS=20 PRESETS='split-50-50' DURATION=90 \
    ./run_sweep.sh > master_D1_hbm${gb}.log 2>&1
done

mark "D2: ResNet batch sweep @ split-50-50"
for bs in 8 16 32 64; do
  AI=resnet RESNET_BS=$bs CELLS=20 PRESETS='split-50-50' DURATION=90 \
    ./run_sweep.sh > master_D2_resnet_bs${bs}.log 2>&1
done

###########################################################################
# PHASE E — PRB allocation sweep (L1 HBM footprint variable)
###########################################################################
mark "E: PRB allocation sweep @ split-50-50 + GPT-2"
for prbs in 51 106 217 273; do
  L1_NUM_PRBS=$prbs AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=90 \
    ./run_sweep.sh > master_E_prbs${prbs}.log 2>&1
done

###########################################################################
# PHASE F — MIMO antenna config sweep (L1 SM intensity variable)
###########################################################################
mark "F: Antenna config sweep @ split-50-50 + GPT-2"
for cfg in 2x2 4x4 8x8; do
  tx=${cfg%x*}; rx=${cfg#*x}
  L1_NUM_TX_ANT=$tx L1_NUM_RX_ANT=$rx AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=90 \
    ./run_sweep.sh > master_F_ant${cfg}.log 2>&1
done

###########################################################################
# PHASE G — long-duration stability (~10 min steady state)
###########################################################################
mark "G: Long-duration stability @ split-50-50 + GPT-2 (1000 iters)"
L1_ITERATIONS=1000 AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=600 \
  ./run_sweep.sh > master_G_stability.log 2>&1

###########################################################################
# PHASE H — AI on/off burst sequence (measure recovery)
###########################################################################
mark "H1: AI OFF baseline @ split-50-50 (no AI)"
AI=none CELLS=20 PRESETS='split-50-50' DURATION=10 \
  ./run_sweep.sh > master_H1_noAI.log 2>&1
mark "H2: AI ON sustained @ split-50-50 + GPT-2 (300 iters)"
L1_ITERATIONS=300 AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=180 \
  ./run_sweep.sh > master_H2_AIon.log 2>&1
mark "H3: AI OFF after AI ran @ split-50-50 (re-check baseline)"
AI=none CELLS=20 PRESETS='split-50-50' DURATION=10 \
  ./run_sweep.sh > master_H3_noAI_post.log 2>&1

###########################################################################
# PHASE I — MCS (coding intensity) sweep
###########################################################################
mark "I: MCS sweep @ split-50-50 + GPT-2"
for mcs in 0 2 7 16 24; do
  L1_MCS_INDEX=$mcs AI=gpt2 CELLS=20 PRESETS='split-50-50' DURATION=90 \
    ./run_sweep.sh > master_I_mcs${mcs}.log 2>&1
done

mark "ALL DONE"
sudo "$SCRIPT_DIR/02_mig.sh" hard-reset
ls -t results/ | head -60
