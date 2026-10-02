#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for seed in 1 2; do
  extra="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1)))"
  for rate in 4 8; do
    SLOT_TAG=f${seed}r${rate} SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$extra --static-mps-pct 50" SLOT_MATRIX="c8:parallel_admit:static c16:parallel_admit:static" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
  done
done
echo SWEEP_DONE
