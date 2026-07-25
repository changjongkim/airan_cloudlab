#!/usr/bin/env bash
# Resume the no-mig batch (since MIG OFF was triggered, no docker restart issue now)
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
RS="$SCRIPT_DIR/run_sweep_v2.sh"

mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

mark "A1 nomig: GPT-2"
env AI=gpt2 CELLS=20 PRESETS='no-mig' DURATION=120 bash $RS > master_A1_gpt2_nomig.log 2>&1

mark "A2 nomig: HBM 1GB"
env AI=hbm HBM_ALLOC=1.0 CELLS=20 PRESETS='no-mig' DURATION=120 bash $RS > master_A2_hbm1_nomig.log 2>&1

mark "A3 nomig: HBM 8GB"
env AI=hbm HBM_ALLOC=8.0 CELLS=20 PRESETS='no-mig' DURATION=120 bash $RS > master_A3_hbm8_nomig.log 2>&1

mark "A4 nomig: ResNet"
env AI=resnet RESNET_BS=16 CELLS=20 PRESETS='no-mig' DURATION=120 bash $RS > master_A4_resnet_nomig.log 2>&1

mark "C1 nomig cell-count"
for c in 1 4 10 20 40; do
  env AI=gpt2 CELLS=$c PRESETS='no-mig' DURATION=90 bash $RS > master_C1_nomig_c${c}.log 2>&1
done

mark "C5 BASELINE no AI no MIG cell-count"
for c in 1 4 10 20 40; do
  env AI=none CELLS=$c PRESETS='no-mig' DURATION=10 bash $RS > master_C5_baseline_c${c}.log 2>&1
done

mark "H1 AI OFF baseline"
env AI=none CELLS=20 PRESETS='no-mig' DURATION=10 bash $RS > master_H1_noAI.log 2>&1
mark "H3 AI OFF baseline (post)"
env AI=none CELLS=20 PRESETS='no-mig' DURATION=10 bash $RS > master_H3_noAI_post.log 2>&1

mark "ALL NO-MIG DONE"
ls -t results/ | head -30
