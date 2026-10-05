#!/usr/bin/env bash
# 32 cells, 1 lane: Antiphase with co-run bounds raised by ~0.4 ms (closer to p99.9).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms 4.5 --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms 6.5"
  SLOT_TAG=y${seed}d65w SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating 128:4.5,512:5.1,1024:5.7/conv=128" \
    SLOT_MATRIX="c32:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
done
echo Y2_DONE
