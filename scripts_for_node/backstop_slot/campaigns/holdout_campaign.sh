#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 3 4; do
  extra="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1"
  SLOT_TAG=h${seed}r8 SLOT_PERIODS=4000 SLOT_AI_RATE=8 SLOT_EXTRA="$extra --static-mps-pct 50" \
    SLOT_MATRIX="c16:rescue_value:backstop_corun c16:rescue_value:static c16:parallel_admit:static c16:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
  SLOT_TAG=h${seed}r8s30 SLOT_PERIODS=4000 SLOT_AI_RATE=8 SLOT_EXTRA="$extra --static-mps-pct 30" \
    SLOT_MATRIX="c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
done
echo HOLDOUT_DONE
