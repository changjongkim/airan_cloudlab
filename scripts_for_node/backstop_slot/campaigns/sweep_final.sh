#!/usr/bin/env bash
# Final slot-scale campaign: 2 seeds x {8,16} cells x AI load {4,8} req/s per GPU.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for seed in 1 2; do
  extra="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1)))"
  for cells in 8 16; do
    SLOT_TAG=f${seed}n SLOT_PERIODS=4000 SLOT_AI_RATE=4 SLOT_EXTRA="$extra" SLOT_MATRIX="c$cells:rescue_value:none" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
    for rate in 4 8; do
      SLOT_TAG=f${seed}r${rate} SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$extra --static-mps-pct 50" SLOT_MATRIX="c$cells:rescue_value:backstop_corun c$cells:rescue_value:static c$cells:parallel:static c$cells:off:static" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
    done
  done
done
echo SWEEP_DONE
