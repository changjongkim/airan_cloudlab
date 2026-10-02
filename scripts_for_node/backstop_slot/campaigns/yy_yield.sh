#!/usr/bin/env bash
# Our Scheme with "yield to waiting NeuralRx": 32 cells (6.5 and 11.5 ms) and 24 cells (6.5 ms).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  for spec in 32:6.5 32:11.5 24:6.5; do
    IFS=: read -r cells d2 <<<"$spec"
    if [ $cells -eq 24 ]; then b="128:3.8,512:4.5,1024:5.2"; bc=3.8; else b="128:4.1,512:4.7,1024:5.3"; bc=4.1; fi
    tag=$(echo $d2 | tr -d .)
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms $d2"
    SLOT_TAG=y${seed}d${tag}y SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $b/conv=128+yield" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo YY_DONE
