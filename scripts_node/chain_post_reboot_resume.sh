#!/usr/bin/env bash
# Post-reboot resume chain — runs Stage 3 (nsight w/ extended metrics) + Stage 4 + Stage 5+.
# Assumes:
#   - /mydata mount alive (fstab auto)
#   - docker daemon running (system service)
#   - NVreg_RestrictProfilingToAdminUsers=0 (after reboot with modprobe.d set)
#   - GPU 0 MIG mode enabled (user runs `sudo nvidia-smi -i 0 -mig 1` first)
#
# Output goes to results/$(date)/...

set -uo pipefail
LOG=/tmp/chain_post_reboot.log
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "===== POST-REBOOT RESUME CHAIN START ====="

# Sanity checks
log "Sanity checks:"
log "  NVreg perm: $(cat /sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers 2>&1)"
log "  GPU 0 MIG: $(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader)"
log "  /mydata: $(df -h /mydata | tail -1)"
log "  docker: $(systemctl is-active docker)"
log "  airan image: $(docker images | grep -c airan:25-3-final)"

# ----- Stage 3: nsight with extended metrics -----
log ""
log "===== STAGE 3: nsight_full_matrix (with 16 metrics) ====="
if bash ~/cloudlab_aerial/nsight_full_matrix.sh > /tmp/chain_nsight.log 2>&1; then
  log "  ✓ Stage 3 OK"
else
  log "  ✗ Stage 3 exited non-zero (continuing)"
fi

# ----- Stage 4: AI per-op latency -----
log ""
log "===== STAGE 4: ai_per_op_latency (16 cells × alone/with_l1) ====="
if bash ~/cloudlab_aerial/run_ai_per_op_latency_matrix.sh > /tmp/chain_stage4.log 2>&1; then
  log "  ✓ Stage 4 OK"
else
  log "  ✗ Stage 4 exited non-zero (continuing)"
fi

# ----- Stage 5: P3 partition sweep -----
log ""
log "===== STAGE 5: P3 partition sweep ====="
if bash ~/cloudlab_aerial/run_p3_partition_sweep.sh > /tmp/chain_p3.log 2>&1; then
  log "  ✓ P3 OK"
else
  log "  ✗ P3 failed"
fi

# ----- Stage 6: P4 timeseries -----
log ""
log "===== STAGE 6: P4 L1 timeseries ====="
if bash ~/cloudlab_aerial/run_p4_l1_timeseries.sh > /tmp/chain_p4.log 2>&1; then
  log "  ✓ P4 OK"
else
  log "  ✗ P4 failed"
fi

# ----- Stage 7: P5 sustained 5-min -----
log ""
log "===== STAGE 7: P5 sustained ====="
if bash ~/cloudlab_aerial/run_p5_sustained.sh > /tmp/chain_p5.log 2>&1; then
  log "  ✓ P5 OK"
else
  log "  ✗ P5 failed"
fi

# ----- Stage 8: P7 PDSCH TX -----
log ""
log "===== STAGE 8: P7 PDSCH TX ====="
if bash ~/cloudlab_aerial/run_p7_pdsch_tx.sh > /tmp/chain_p7.log 2>&1; then
  log "  ✓ P7 OK"
else
  log "  ✗ P7 failed"
fi

# ----- (Optional) M9 retry -----
log ""
log "===== STAGE 9: M9 retry (2g L1 multi-AI cells) ====="
log "  TODO: M9 retry script needed if M9 cells matter"

log ""
log "===== POST-REBOOT CHAIN COMPLETE ====="
log "Total stages: 3 (nsight) + 4 (per-op) + 5 (P3) + 6 (P4) + 7 (P5) + 8 (P7)"
log "Results dir: ~/cloudlab_aerial/results/$(date +%Y%m%d)/"
