#!/usr/bin/env bash
# Parallel chain — Phase 2/3/4 on GPU 0/1/2 + extra baseline on GPU 3 (no-MIG).
#
# GPU 0 (MIG): Phase 2 multi-partition
# GPU 1 (MIG): Phase 4 AI-RAN workloads (NeuralRx/ChanPred/xApp)
# GPU 2 (MIG): Phase 3 D1 (split-40-60 + Qwen / alone)
# GPU 3 (no-MIG): Extra baseline N=20 (full GPU L1 alone)
#
# Prereq: reboot done, MIG enabled on GPU 0/1/2, MIG disabled on GPU 3.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR
LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/parallel_chain.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%H:%M:%S'; }
sec() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*" | tee -a "$LOG"; }

sec "PARALLEL CHAIN start (GPU 0/1/2 MIG, GPU 3 no-MIG baseline)"

# GPU 0: Phase 2 multi-partition
sec "Launch Phase 2 on GPU 0 (background)"
GPU=0 DATE_DIR="$DATE_DIR" nohup bash ./phase2_multipartition.sh > /tmp/phase2.log 2>&1 &
P2_PID=$!
echo "phase2 pid: $P2_PID" | tee -a "$LOG"

sleep 5

# GPU 1: Phase 4 AI-RAN workloads
sec "Launch Phase 4 on GPU 1 (background)"
GPU=1 DATE_DIR="$DATE_DIR" nohup bash ./phase4_airan.sh > /tmp/phase4.log 2>&1 &
P4_PID=$!
echo "phase4 pid: $P4_PID" | tee -a "$LOG"

sleep 5

# GPU 2: Phase 3 D1a + D1b only (skip A baseline — already have it from GPU 1/3)
sec "Launch Phase 3 D1 on GPU 2 (background, skip A baseline)"
GPU=2 DATE_DIR="$DATE_DIR" nohup bash -c '
set -uo pipefail
cd ~/cloudlab_aerial
N=10 DURATION=30
echo "=== D1a: split-40-60 + qwen7b ==="
N=$N PRESET=split-40-60 AI=qwen7b TAG=D1a_4060_qwen DMON=1 DURATION=$DURATION bash ./run_n20.sh
echo "=== D1b: split-40-60 alone ==="
N=$N PRESET=split-40-60 AI=none TAG=D1b_4060_alone DMON=0 DURATION=$DURATION bash ./run_n20.sh
echo "=== Phase 3 D1 done ==="
' > /tmp/phase3_d1.log 2>&1 &
P3_PID=$!
echo "phase3 D1 pid: $P3_PID" | tee -a "$LOG"

sec "Waiting for all parallel jobs to complete..."
wait $P2_PID; sec "Phase 2 done (exit $?)"
wait $P4_PID; sec "Phase 4 done (exit $?)"
wait $P3_PID; sec "Phase 3 D1 done (exit $?)"

sec "PARALLEL CHAIN DONE"
echo "Results:"
ls -la "$HOME/cloudlab_aerial/results/$DATE_DIR/" | tee -a "$LOG"
