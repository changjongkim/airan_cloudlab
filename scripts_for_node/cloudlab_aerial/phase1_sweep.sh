#!/usr/bin/env bash
# Phase 1 master sweep — runs all P1 sensitivity experiments sequentially.
#
# Implements SENSITIVITY_EXPERIMENTS §A1 + §A2 + bimodal baseline.
# Total: 4 × (N=20 × ~30s + setup) ≈ 1.5 ~ 2 hours.
#
# Results land in:
#   results/<YYYYMMDD>/n20_A0_qwen_baseline/
#   results/<YYYYMMDD>/n20_A1_prefill/
#   results/<YYYYMMDD>/n20_A1_decode/
#   results/<YYYYMMDD>/n20_A2_hbm/
#
# Each contains: run_*.json, run_*.log, summary.txt, dmon.csv, markers.txt.
# Run analyze_run.py on each subdir afterward for cluster verdict.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR
N="${N:-20}"
DURATION="${DURATION:-30}"

LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/phase1_master.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%H:%M:%S'; }
sec() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*" | tee -a "$LOG"; }

sec "PHASE 1 sweep start — DATE_DIR=$DATE_DIR, N=$N, DURATION=${DURATION}s"
sec "Output root: results/$DATE_DIR/"

# ----------------------------------------------------------------------------
# A0 — Baseline bimodal re-confirmation (split-60-40 + Qwen normal)
# Expected: BIMODAL ~50:50, gap ~6.7ms (reproduces N=4 finding)
# ----------------------------------------------------------------------------
sec "A0: split-60-40 + qwen7b (baseline)"
N=$N PRESET=split-60-40 AI=qwen7b TAG=A0_qwen_baseline DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# A1a — Prefill-only (H1 phase hypothesis HIGH-mode trigger)
# Expected (if H1 correct): BIMODAL with HIGH dominant, or UNIMODAL HIGH
# ----------------------------------------------------------------------------
sec "A1a: split-60-40 + qwen7b_prefill (PREFILL burst only)"
N=$N PRESET=split-60-40 AI=qwen7b_prefill TAG=A1_prefill DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# A1b — Decode-only (H1 phase hypothesis LOW-mode trigger)
# Expected (if H1 correct): UNIMODAL LOW (no leakage)
# ----------------------------------------------------------------------------
sec "A1b: split-60-40 + qwen7b_decode (DECODE only, HBM mostly idle)"
N=$N PRESET=split-60-40 AI=qwen7b_decode TAG=A1_decode DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# A2 — Non-phased AI workload (H1 vs H2/H3 separation)
# Expected if H1 correct: HBM stress is UNIMODAL (no phase oscillation)
# Expected if H2/H3 correct: HBM stress is BIMODAL too
# ----------------------------------------------------------------------------
sec "A2: split-60-40 + hbm 16GB (non-phased, deterministic)"
N=$N PRESET=split-60-40 AI=hbm HBM_ALLOC=16.0 TAG=A2_hbm DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
sec "Running bimodal_detect on all four runs"
for tag in A0_qwen_baseline A1_prefill A1_decode A2_hbm; do
  dir="results/$DATE_DIR/n20_$tag"
  if [[ -d "$dir" ]]; then
    echo ""
    echo "----- $tag -----" | tee -a "$LOG"
    python3 "$HOME/analyze_run.py" "$dir" 2>&1 | tee -a "$LOG"
  fi
done

sec "PHASE 1 sweep DONE"
echo ""
echo "Summary of cluster verdicts:"
grep "VERDICT:" "$LOG" | tail -8

echo ""
echo "Next: collect & push results"
echo "  rsync -av results/$DATE_DIR/ <local>:/cloudlab_results/results/$DATE_DIR/"
