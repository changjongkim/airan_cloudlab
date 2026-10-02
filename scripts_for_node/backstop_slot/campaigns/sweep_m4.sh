#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for rate in 4 8; do
  for pct in 20 30 50 70; do
    SLOT_TAG=m4r${rate}s$pct SLOT_PERIODS=2400 SLOT_AI_RATE=$rate SLOT_EXTRA="--static-mps-pct $pct" SLOT_MATRIX="c16:rescue:static" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  done
  for pct in 50 70 100; do
    SLOT_TAG=m4r${rate}c$pct SLOT_PERIODS=2400 SLOT_AI_RATE=$rate SLOT_EXTRA="--static-mps-pct $pct" SLOT_MATRIX="c16:rescue:backstop_corun" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  done
  for pct in 50 70; do
    SLOT_TAG=m4r${rate}q$pct SLOT_PERIODS=2400 SLOT_AI_RATE=$rate SLOT_EXTRA="--static-mps-pct $pct --nrx-bound-corun-ms 2.6" SLOT_MATRIX="c16:rescue:backstop_corun" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  done
done
echo SWEEP_DONE
