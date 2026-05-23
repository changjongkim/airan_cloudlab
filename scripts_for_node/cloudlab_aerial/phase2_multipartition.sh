#!/usr/bin/env bash
# Phase 2 — Multi-AI partition sweep.
#
# Tests 4 new multi-partition presets to study production AI-RAN scenarios
# (1 L1 + 2-3 AI services). Implements MASTER_SUMMARY §6 multi-partition
# stretch goal + addresses user's "multi-workload" requirement.
#
# Each config N=10 (reduced from 20 to fit in 5h budget alongside phase1).
# Total ~32 min computational + ~12 min MIG reconfig overhead = ~45 min.
#
# Output:
#   results/<YYYYMMDD>/n10_M1_3way_balanced_qwen_small/
#   results/<YYYYMMDD>/n10_M2_3way_L1small_qwen7b/
#   results/<YYYYMMDD>/n10_M3_3way_asym_mixed/
#   results/<YYYYMMDD>/n10_M4_4way_3AI_light/

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR
N="${N:-10}"          # smaller N for multi-partition to fit time budget
DURATION="${DURATION:-30}"

LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/phase2_master.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%H:%M:%S'; }
sec() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*" | tee -a "$LOG"; }

sec "PHASE 2 multi-partition sweep — DATE_DIR=$DATE_DIR, N=$N"

# ----------------------------------------------------------------------------
# M1 — 3way-balanced (2g+2g+3g, L1=3g, AI×2 on 2g instances)
# Production AI-RAN: L1 + 2 xApps (PHY-layer NN + telemetry analytics)
# Question: bimodal disappears with symmetric AI×2?
# Expected: leakage small (symmetric neighbors), NO bimodal
# ----------------------------------------------------------------------------
sec "M1: 3way-balanced + neuralrx,xapp_anomaly (2 AI-RAN services)"
N=$N PRESET=3way-balanced AI=neuralrx,xapp_anomaly \
  TAG=M1_3way_balanced_AIRAN DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# M2 — 3way-L1small (2g+2g+3g, L1=2g, AI on 2g + 3g asymmetric)
# Production scenario: L1 on small partition, heavy LLM + light LSTM
# Expected: high mean (partition cap), possible bimodal on 3g neighbor
# ----------------------------------------------------------------------------
sec "M2: 3way-L1small + chanpred,qwen_small (LSTM + small LLM mixed)"
N=$N PRESET=3way-L1small AI=chanpred,qwen_small \
  TAG=M2_3way_L1small_mixed DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# M3 — 3way-asym (1g+2g+4g, L1=4g, AI on 2g + 1g)
# Production: L1 (RU on big partition) + Neural RX + small xApp
# Expected: dominated by neuralrx on 2g; 1g xApp minor
# ----------------------------------------------------------------------------
sec "M3: 3way-asym + neuralrx,xapp_anomaly (PHY NN + xApp)"
N=$N PRESET=3way-asym AI=neuralrx,xapp_anomaly \
  TAG=M3_3way_asym_AIRAN DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# M4 — 4way-1L1+3AI (4g+1g+1g+1g, L1=4g, AI×3 on 1g)
# Production: L1 + 3 microservices (channel pred + xApp anomaly + RX NN)
# All 3 are small AI-RAN workloads, fit in 1g.5gb each
# Question: leakage proportional to AI count?
# ----------------------------------------------------------------------------
sec "M4: 4way-1L1+3AI + chanpred,xapp_anomaly,gpt2 (3 mixed AI-RAN xApps)"
N=$N PRESET=4way-1L1+3AI AI=chanpred,xapp_anomaly,gpt2 \
  TAG=M4_4way_3xApp DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
sec "Running bimodal_detect on phase 2 results"
for tag in M1_3way_balanced_AIRAN M2_3way_L1small_mixed M3_3way_asym_AIRAN M4_4way_3xApp; do
  dir="results/$DATE_DIR/n10_$tag"
  if [[ -d "$dir" ]]; then
    echo ""
    echo "----- $tag -----" | tee -a "$LOG"
    python3 "$HOME/bimodal_detect.py" "$dir" 2>&1 | tee -a "$LOG"
  fi
done

sec "PHASE 2 DONE"
echo "Verdicts:"
grep "VERDICT:" "$LOG" | tail -8
