#!/usr/bin/env bash
# Y: one NeuralRx lane per GPU at 24/32 cells (bounds from the 1-lane probe), rescue deadline
# 6.5 and 11.5 ms, fair AI units, token AI admission, AI 12 req/s, 2 seeds.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  for spec in 24:6.5 32:6.5 24:11.5 32:11.5; do
    IFS=: read -r cells d2 <<<"$spec"
    if [ $cells -eq 24 ]; then b="128:3.8,512:4.5,1024:5.2"; else b="128:4.1,512:4.7,1024:5.3"; fi
    lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms ${b%%,*}"
    lanes="${lanes/128:/}"
    tag=$(echo $d2 | tr -d .)
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $lanes --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms $d2"
    SLOT_TAG=y${seed}d${tag}n SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=y${seed}d${tag}u SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $b/conv=128" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=y${seed}d${tag}v SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $b/conv=" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
    for pct in 50 70 100; do
      SLOT_TAG=y${seed}d${tag}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct $pct" \
        SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    done
    SLOT_TAG=y${seed}d${tag}s50 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 50" \
      SLOT_MATRIX="c$cells:parallel_admit:static" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo Y_DONE
