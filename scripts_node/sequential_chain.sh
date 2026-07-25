#!/usr/bin/env bash
# Sequential chain — Phase 2 → Phase 3 D1 → Phase 4, all on GPU 0 (MIG).
# Same DATE_DIR for unified results.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR
LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/sequential_chain.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%H:%M:%S'; }
sec() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*" | tee -a "$LOG"; }

sec "SEQUENTIAL CHAIN start (GPU 0 MIG)"

# Phase 2
sec "Phase 2: multi-partition (M1-M4)"
GPU=0 DATE_DIR="$DATE_DIR" bash ./phase2_multipartition.sh 2>&1 | tee -a "$LOG"

# Phase 3 D1 only (skip A baseline — already have from earlier baseline runs)
sec "Phase 3 D1a + D1b (skip A baseline)"
N=10 DURATION=30
N=$N PRESET=split-40-60 AI=qwen7b TAG=D1a_4060_qwen DMON=1 DURATION=$DURATION bash ./run_n20.sh 2>&1 | tee -a "$LOG"
N=$N PRESET=split-40-60 AI=none TAG=D1b_4060_alone DMON=0 DURATION=$DURATION bash ./run_n20.sh 2>&1 | tee -a "$LOG"

# Phase 4
sec "Phase 4: AI-RAN workloads (AR1-AR3)"
GPU=0 DATE_DIR="$DATE_DIR" bash ./phase4_airan.sh 2>&1 | tee -a "$LOG"

sec "SEQUENTIAL CHAIN DONE"
echo "Final results:"
ls -d "$HOME/cloudlab_aerial/results/$DATE_DIR/n"* 2>/dev/null | tee -a "$LOG"
