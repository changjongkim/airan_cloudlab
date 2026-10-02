#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
SLOT_TAG=m6r4n SLOT_PERIODS=2400 SLOT_AI_RATE=4 SLOT_MATRIX="c16:rescue_value:none" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
for rate in 4 8; do
  SLOT_TAG=m6r${rate}b SLOT_PERIODS=2400 SLOT_AI_RATE=$rate SLOT_MATRIX="c16:rescue_value:backstop" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  for pct in 50 70; do
    SLOT_TAG=m6r${rate}c$pct SLOT_PERIODS=2400 SLOT_AI_RATE=$rate SLOT_EXTRA="--static-mps-pct $pct" SLOT_MATRIX="c16:rescue_value:backstop_corun" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  done
  for pct in 30 50; do
    SLOT_TAG=m6r${rate}s$pct SLOT_PERIODS=2400 SLOT_AI_RATE=$rate SLOT_EXTRA="--static-mps-pct $pct" SLOT_MATRIX="c16:rescue_value:static" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  done
done
echo SWEEP_DONE
