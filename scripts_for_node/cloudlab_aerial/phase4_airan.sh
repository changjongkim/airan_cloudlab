#!/usr/bin/env bash
# Phase 4 — AI-RAN workload bimodal universality.
#
# Tests whether the split-60-40 bimodal leakage observed with Qwen-7B also
# appears with real AI-RAN workloads (Neural RX, channel prediction LSTM,
# xApp anomaly detection).
#
# This addresses user's critique: "We only had Qwen — but real AI-RAN
# scenarios use PHY-layer NN + xApps, not LLMs."
#
# Each config N=10. Total ~30 min.
#
# Hypothesis tested: is bimodal a Qwen-phase artifact (H1) or a more
# general MIG-asymmetry artifact (H2/H3)?
#   - If neuralrx/chanpred/xapp also bimodal → H2/H3 (NoC/arbiter)
#   - If only qwen variants bimodal → H1 confirmed (phase-specific)

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR
N="${N:-10}"
DURATION="${DURATION:-30}"

LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/phase4_master.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%H:%M:%S'; }
sec() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*" | tee -a "$LOG"; }

sec "PHASE 4 AI-RAN workload bimodal universality — DATE_DIR=$DATE_DIR, N=$N"

# ----------------------------------------------------------------------------
# AR1 — split-60-40 + Neural RX (PHY-layer NN, in-line scenario)
# Expected: small leakage (NN inference is small + L2-resident)
# ----------------------------------------------------------------------------
sec "AR1: split-60-40 + neuralrx (PHY-layer NN, real AI-RAN in-line)"
N=$N PRESET=split-60-40 AI=neuralrx \
  TAG=AR1_6040_neuralrx DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# AR2 — split-60-40 + Channel Prediction LSTM
# Expected: very small leakage (LSTM is tiny, ~5MB)
# ----------------------------------------------------------------------------
sec "AR2: split-60-40 + chanpred (LSTM channel prediction)"
N=$N PRESET=split-60-40 AI=chanpred \
  TAG=AR2_6040_chanpred DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# AR3 — split-60-40 + xApp anomaly (autoencoder)
# Expected: small leakage. Tests "many small RAN xApps" pattern
# ----------------------------------------------------------------------------
sec "AR3: split-60-40 + xapp_anomaly (rApp telemetry autoencoder)"
N=$N PRESET=split-60-40 AI=xapp_anomaly \
  TAG=AR3_6040_xapp DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
sec "Running bimodal_detect on phase 4 results"
for tag in AR1_6040_neuralrx AR2_6040_chanpred AR3_6040_xapp; do
  dir="results/$DATE_DIR/n${N}_$tag"
  if [[ -d "$dir" ]]; then
    echo ""
    echo "----- $tag -----" | tee -a "$LOG"
    python3 "$HOME/bimodal_detect.py" "$dir" 2>&1 | tee -a "$LOG"
  fi
done

sec "PHASE 4 DONE"
echo "AI-RAN bimodal verdicts:"
grep "VERDICT:" "$LOG" | tail -3

# Cross-reference with phase 1 Qwen results
echo ""
echo "Comparison with Phase 1 Qwen variants:"
echo "If all AI-RAN workloads UNIMODAL but Qwen variants BIMODAL → H1 (phase alignment) confirmed"
echo "If all workloads BIMODAL → H2/H3 (NoC/arbiter, workload-agnostic) confirmed"
echo "If mixed → mechanism is workload-dependent, needs deeper investigation"
