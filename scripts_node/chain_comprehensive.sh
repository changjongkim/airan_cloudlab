#!/usr/bin/env bash
# Comprehensive chain: ALL workloads × ALL P-stages.
# Re-runs P3/P4/P5/P7 with all 8-9 workloads (was 4 each).
# Plus Stage 4b supplement (xapp + forecaster per-op latency).

set -uo pipefail
LOG=/tmp/chain_comprehensive.log
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "===== COMPREHENSIVE CHAIN START ====="

# Stage 4b: xapp + Forecaster per-op latency
log ""
log "===== STAGE 4b: ai_per_op_latency_b (xapp + Forecaster) ====="
if bash ~/cloudlab_aerial/run_ai_per_op_latency_b.sh > /tmp/chain_stage4b.log 2>&1; then
  log "  ✓ Stage 4b OK"
else
  log "  ✗ Stage 4b failed"
fi

# P3 expanded: 8 workloads × 4 partitions
log ""
log "===== P3: partition sweep (8 workloads × 4 partitions) ====="
if bash ~/cloudlab_aerial/run_p3_partition_sweep.sh > /tmp/chain_p3.log 2>&1; then
  log "  ✓ P3 OK"
else
  log "  ✗ P3 failed"
fi

# P4: 9 workloads timeseries
log ""
log "===== P4: L1 timeseries (9 workloads) ====="
if bash ~/cloudlab_aerial/run_p4_l1_timeseries.sh > /tmp/chain_p4.log 2>&1; then
  log "  ✓ P4 OK"
else
  log "  ✗ P4 failed"
fi

# P5: 9 workloads sustained (5min each — total long)
log ""
log "===== P5: sustained 5-min (9 workloads) ====="
if bash ~/cloudlab_aerial/run_p5_sustained.sh > /tmp/chain_p5.log 2>&1; then
  log "  ✓ P5 OK"
else
  log "  ✗ P5 failed"
fi

# P7: 9 workloads PDSCH TX
log ""
log "===== P7: PDSCH TX (9 workloads) ====="
if bash ~/cloudlab_aerial/run_p7_pdsch_tx.sh > /tmp/chain_p7.log 2>&1; then
  log "  ✓ P7 OK"
else
  log "  ✗ P7 failed"
fi

log ""
log "===== COMPREHENSIVE CHAIN DONE ====="
