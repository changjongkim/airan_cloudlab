#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
seedx() { echo "--seed $((81000 + 1000 * $1)) --ring-offset $((50 * ($1 - 1))) --ai-admission 1"; }
for seed in 1 2; do
  for rate in 8 12; do
    SLOT_TAG=d${seed}r${rate} SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$(seedx $seed) --static-mps-pct 50" \
      SLOT_MATRIX="c16:rescue_value:backstop_corun c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=d${seed}r${rate}s30 SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$(seedx $seed) --static-mps-pct 30" \
      SLOT_MATRIX="c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  done
  SLOT_TAG=d${seed}w075 SLOT_PERIODS=4000 SLOT_AI_RATE=8 SLOT_EXTRA="$(seedx $seed) --static-mps-pct 50 --weak-fraction 0.75" \
    SLOT_MATRIX="c16:rescue_value:backstop_corun c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
done
echo ADMISSION_DONE
