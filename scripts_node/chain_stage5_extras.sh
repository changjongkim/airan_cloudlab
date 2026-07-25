#!/usr/bin/env bash
# Stage 5+ chain: P3 partition sweep → P4 timeseries → P5 sustained → P7 PDSCH TX.
# Waits for Stage 4 (ai_per_op_latency) to finish, then runs all P-series sequentially.

set -uo pipefail
LOG=/tmp/chain_stage5.log
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "===== STAGE 5+ WATCHER START — waiting for Stage 4 to finish ====="
while true; do
  if pgrep -fa "run_ai_per_op_latency_matrix" >/dev/null; then
    sleep 60
  elif [[ -f /tmp/chain_stage4_run.log ]]; then
    log "Stage 4 done"
    break
  else
    sleep 60
  fi
done

log ""
log "===== P3: Partition size sweep ====="
if bash ~/cloudlab_aerial/run_p3_partition_sweep.sh > /tmp/chain_p3.log 2>&1; then
  log "✓ P3 OK"
else
  log "✗ P3 failed"
fi

log ""
log "===== P4: L1 time-series ====="
if bash ~/cloudlab_aerial/run_p4_l1_timeseries.sh > /tmp/chain_p4.log 2>&1; then
  log "✓ P4 OK"
else
  log "✗ P4 failed"
fi

log ""
log "===== P5: Sustained 5-min ====="
if bash ~/cloudlab_aerial/run_p5_sustained.sh > /tmp/chain_p5.log 2>&1; then
  log "✓ P5 OK"
else
  log "✗ P5 failed"
fi

log ""
log "===== P7: PDSCH TX (other cuPHY workload) ====="
if bash ~/cloudlab_aerial/run_p7_pdsch_tx.sh > /tmp/chain_p7.log 2>&1; then
  log "✓ P7 OK"
else
  log "✗ P7 failed"
fi

log ""
log "===== ALL P-SERIES DONE ====="
