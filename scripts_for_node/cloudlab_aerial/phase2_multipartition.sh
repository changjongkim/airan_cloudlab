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
# Question: bimodal disappears with symmetric AI×2 instead of single 4g AI?
# Expected: leakage small (symmetric neighbors), NO bimodal
# ----------------------------------------------------------------------------
sec "M1: 3way-balanced + qwen_small,qwen_small (symmetric 2 AI)"
N=$N PRESET=3way-balanced AI=qwen_small,qwen_small \
  TAG=M1_3way_balanced_qwen_small DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# M2 — 3way-L1small (2g+2g+3g, L1=2g, AI on 2g + 3g asymmetric)
# Question: L1 on SMALL partition with bigger AI neighbor — partition cap effect?
# Expected: high mean (partition cap), possible bimodal on 3g asymmetric neighbor
# ----------------------------------------------------------------------------
sec "M2: 3way-L1small + qwen_small,qwen7b (asymmetric AI sizes)"
N=$N PRESET=3way-L1small AI=qwen_small,qwen7b \
  TAG=M2_3way_L1small_mixed DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# M3 — 3way-asym (1g+2g+4g, L1=4g, AI on 2g + 1g)
# Question: L1 on biggest partition with two smaller AI neighbors
# Expected: leakage from 2g side, 1g side contributes less (small workload)
# ----------------------------------------------------------------------------
sec "M3: 3way-asym + qwen_small,gpt2 (mixed weight AI)"
N=$N PRESET=3way-asym AI=qwen_small,gpt2 \
  TAG=M3_3way_asym_mixed DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# M4 — 4way-1L1+3AI (4g+1g+1g+1g, L1=4g, AI×3 on 1g)
# Question: leakage proportional to AI count? Sum of 3 light AI > single big AI?
# Expected: cumulative leakage, but each 1g neighbor light
# ----------------------------------------------------------------------------
sec "M4: 4way-1L1+3AI + gpt2,resnet,hbm_1g (3 different light AI)"
N=$N PRESET=4way-1L1+3AI AI=gpt2,resnet,hbm_1g \
  TAG=M4_4way_3AI_mix DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
sec "Running bimodal_detect on phase 2 results"
for tag in M1_3way_balanced_qwen_small M2_3way_L1small_mixed M3_3way_asym_mixed M4_4way_3AI_mix; do
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
