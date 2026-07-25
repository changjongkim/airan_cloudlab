#!/usr/bin/env bash
# Stage 4 chain extension: ai_per_op_latency_matrix.
# Waits for Stage 3 (nsight_full_matrix.sh) to finish, then runs AI per-op latency.
# Run alongside existing chain — picks up after nsight ends.

set -uo pipefail
LOG=/tmp/chain_stage4.log
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "===== STAGE 4 WATCHER START — waiting for nsight to finish ====="
# Wait until nsight script is no longer running (its log will be at /tmp/chain_nsight.log when started)
while true; do
  if pgrep -fa "nsight_full_matrix" >/dev/null; then
    sleep 60
  elif [[ -f /tmp/chain_nsight.log ]]; then
    # nsight already started and now no longer running → done
    log "nsight not running and log exists → assume done"
    break
  else
    # nsight not yet started
    sleep 60
  fi
done

log "ai_per_op_latency_matrix starting"
if bash ~/cloudlab_aerial/run_ai_per_op_latency_matrix.sh > /tmp/chain_stage4_run.log 2>&1; then
  log "✓ Stage 4 OK"
else
  log "✗ Stage 4 exited non-zero"
fi

log "===== STAGE 4 DONE ====="
