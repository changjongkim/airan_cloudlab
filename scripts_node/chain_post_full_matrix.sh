#!/usr/bin/env bash
# Auto-chain after ai_full_matrix finishes:
#   1. ai_supplement.sh (ResNet + Traffic forecaster)
#   2. l1_multi_ai_matrix.sh (M5-M8)
#   3. nsight_full_matrix.sh (L2 cache profiling)
#
# Waits until ai_full_matrix process is no longer running, then runs each in sequence.
# Logs to /tmp/chain_*.log and continues past failures.

set -uo pipefail
LOG=/tmp/chain.log
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "===== CHAIN START — waiting for ai_full_matrix to finish ====="
while pgrep -fa run_ai_full_matrix >/dev/null; do
  sleep 60
done
log "ai_full_matrix exited; proceeding"

# Stage 1: ai_supplement
log ""
log "===== STAGE 1: ai_supplement (ResNet + Forecaster × 4 partitions) ====="
if bash ~/cloudlab_aerial/run_ai_supplement.sh > /tmp/chain_supplement.log 2>&1; then
  log "  ✓ ai_supplement OK"
else
  log "  ✗ ai_supplement exited non-zero (continuing)"
fi

# Stage 2: l1_multi_ai_matrix
log ""
log "===== STAGE 2: l1_multi_ai_matrix (M5-M8, 10 cells) ====="
if bash ~/cloudlab_aerial/run_l1_multi_ai_matrix.sh > /tmp/chain_multi_ai.log 2>&1; then
  log "  ✓ multi_ai_matrix OK"
else
  log "  ✗ multi_ai_matrix exited non-zero (continuing)"
fi

# Stage 3: nsight_full_matrix
log ""
log "===== STAGE 3: nsight_full_matrix (12 scenarios, L2 hit rate) ====="
if bash ~/cloudlab_aerial/nsight_full_matrix.sh > /tmp/chain_nsight.log 2>&1; then
  log "  ✓ nsight OK"
else
  log "  ✗ nsight exited non-zero"
fi

log ""
log "===== CHAIN COMPLETE ====="
log "Result dirs in ~/cloudlab_aerial/results/$(date +%Y%m%d)/"
