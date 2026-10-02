#!/usr/bin/env bash
# Scheme v2 main campaign: rescue deadline 6.5 ms for every NeuralRx policy,
# AI admission for every AI policy, 16/24/32 cells, AI 8/12 req/s, 2 seeds.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  for cells in 16 24 32; do
    if [ $cells -eq 16 ]; then lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.5 --nrx-bound-corun-ms 2.8"
    else lanes="--lanes-per-gpu 2 --nrx-bound-ms 3.3 --nrx-bound-corun-ms 3.7"; fi
    base="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --rescue-deadline-ms 6.5 $lanes"
    SLOT_TAG=v${seed}n SLOT_PERIODS=4000 SLOT_AI_RATE=8 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
    for rate in 8 12; do
      SLOT_TAG=v${seed}r$rate SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$base --static-mps-pct 50" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_corun c$cells:rescue_value:static c$cells:parallel_admit:static c$cells:off:static c$cells:rescue_value:backstop" $M 2>&1 | grep -E "^===|FAILED"
      SLOT_TAG=v${seed}r${rate}s30 SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$base --static-mps-pct 30" \
        SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
      SLOT_TAG=v${seed}r${rate}d40 SLOT_PERIODS=4000 SLOT_AI_RATE=$rate \
        SLOT_EXTRA="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --static-mps-pct 50 --lanes-per-gpu 1 --nrx-bound-ms 2.5 --nrx-bound-corun-ms 2.8" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_corun" $M 2>&1 | grep -E "^===|FAILED"
    done
  done
done
echo V2_DONE
