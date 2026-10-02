#!/usr/bin/env bash
# Holdout: fresh seeds 3/4 for the 1-lane setting, 24 and 32 cells, rescue deadline 6.5 ms.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 3 4; do
  for cells in 24 32; do
    if [ $cells -eq 24 ]; then b="128:3.8,512:4.5,1024:5.2"; bc=3.8; else b="128:4.1,512:4.7,1024:5.3"; bc=4.1; fi
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms 6.5"
    SLOT_TAG=h${seed}d65n SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=h${seed}d65u SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $b/conv=128" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
    for pct in 50 70; do
      SLOT_TAG=h${seed}d65s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct $pct" \
        SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    done
    SLOT_TAG=h${seed}d65s50 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 50" \
      SLOT_MATRIX="c$cells:parallel_admit:static" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo YH_DONE
