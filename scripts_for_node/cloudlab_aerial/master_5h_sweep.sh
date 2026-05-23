#!/usr/bin/env bash
# Master 5-hour sweep — runs phase1 + phase2 + phase3 + phase4
# for the 5/24 10AM-3PM CloudLab reservation window.
#
# Time budget:
#   0:00 ~ 0:05  Setup verification (caller should do scp + MIG enable BEFORE this)
#   0:05 ~ 2:00  Phase 1 (A0/A1a/A1b/A2 N=20)               ~ 1h55m
#   2:00 ~ 2:30  Phase 4 (AR1/AR2/AR3 AI-RAN N=10)          ~ 30m
#   2:30 ~ 3:15  Phase 2 (M1/M2/M3/M4 multi-AI N=10)        ~ 45m
#   3:15 ~ 4:00  Phase 3 (D1ab N=10 + A baseline)           ~ 45m
#   4:00 ~ 4:30  Buffer / re-run failures                   ~ 30m
#   4:30 ~ 5:00  rsync + git push + snapshot   ~ 30m (CALLER DOES)
#
# After this script finishes, caller MUST:
#   1. rsync results to local
#   2. git commit + push
#   3. Save image snapshot via CloudLab portal
#
# Usage:
#   nohup ./master_5h_sweep.sh > /tmp/master_$(date +%H%M).log 2>&1 &
#   tail -f /tmp/master_*.log

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR

START_TS=$(date +%s)
MASTER_LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/master_5h.log"
mkdir -p "$(dirname "$MASTER_LOG")"

elapsed() {
  local now=$(date +%s)
  local delta=$((now - START_TS))
  printf '%02d:%02d' $((delta / 60)) $((delta % 60))
}
sec() {
  printf '\n############################################################\n'
  printf '# [%s elapsed] %s\n' "$(elapsed)" "$*"
  printf '############################################################\n'
} 2>&1 | tee -a "$MASTER_LOG"

sec "MASTER 5H SWEEP START — $(date)"
sec "DATE_DIR=$DATE_DIR"

# ----------------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------------
sec "Pre-flight: nvidia-smi + MIG status"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv,noheader 2>&1 | tee -a "$MASTER_LOG"

mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0)
if [[ "$mig_state" != "Enabled" ]]; then
  sec "MIG not enabled — enabling now"
  bash ./driver_reset.sh 2>&1 | tee -a "$MASTER_LOG"
  sudo nvidia-smi -i 0 -mig 1 2>&1 | tee -a "$MASTER_LOG"
  sleep 3
fi

sec "Pre-flight: docker container check"
docker images | grep -E "aerial|airan" 2>&1 | tee -a "$MASTER_LOG"

# ----------------------------------------------------------------------------
# PHASE 1 — Bimodal mechanism (1h55m budget)
# A0/A1a/A1b/A2 × N=20 on split-60-40
# ----------------------------------------------------------------------------
sec "PHASE 1 START — bimodal mechanism (A0/A1a/A1b/A2 N=20)"
N=20 DURATION=30 bash ./phase1_sweep.sh 2>&1 | tee -a "$MASTER_LOG"
sec "PHASE 1 DONE — elapsed $(elapsed)"

# ----------------------------------------------------------------------------
# PHASE 4 — AI-RAN workload bimodal universality (30m budget)
# AR1/AR2/AR3 on split-60-40 with Neural RX, chanpred LSTM, xApp anomaly
# Critical: tests if bimodal is Qwen-specific (H1) or general (H2/H3)
# ----------------------------------------------------------------------------
sec "PHASE 4 START — AI-RAN workload bimodal universality (AR1/AR2/AR3 N=10)"
N=10 DURATION=30 bash ./phase4_airan.sh 2>&1 | tee -a "$MASTER_LOG"
sec "PHASE 4 DONE — elapsed $(elapsed)"

# ----------------------------------------------------------------------------
# PHASE 2 — Multi-AI partition (45m budget)
# M1/M2/M3/M4 × N=10
# ----------------------------------------------------------------------------
sec "PHASE 2 START — multi-AI partition (M1/M2/M3/M4 N=10)"
N=10 DURATION=30 bash ./phase2_multipartition.sh 2>&1 | tee -a "$MASTER_LOG"
sec "PHASE 2 DONE — elapsed $(elapsed)"

# ----------------------------------------------------------------------------
# PHASE 3 — D1 partition cap separation + A baseline (45m budget)
# ----------------------------------------------------------------------------
sec "PHASE 3 START — D1 + A baseline"
N=10 DURATION=30 bash ./phase3_extras.sh 2>&1 | tee -a "$MASTER_LOG"
sec "PHASE 3 DONE — elapsed $(elapsed)"

# ----------------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------------
sec "MASTER SWEEP COMPLETE — total $(elapsed)"
echo ""
echo "All bimodal verdicts:"
grep -h "VERDICT:" "results/$DATE_DIR"/*master*.log 2>/dev/null

echo ""
echo "Next steps for caller:"
echo "  1. rsync -av results/$DATE_DIR/ to local cloudlab_results/"
echo "  2. git commit + push"
echo "  3. CloudLab portal → Save Image (snapshot)"
echo ""
ls -la "results/$DATE_DIR/" 2>&1 | tee -a "$MASTER_LOG"
