#!/usr/bin/env bash
# One NeuralRx lane per GPU at 24/32 cells: NeuralRx and conventional under a 100% AI share
# with one AI chunk at a time (rescue deadline 41.5 ms so every NeuralRx runs to the end).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for cells in 32 24; do
  base="--seed 82000 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms 41.5 --ai-admission 0 --static-mps-pct 100 --ai-chunks 128,512,1024"
  SLOT_TAG=r1n SLOT_PERIODS=2400 SLOT_AI_RATE=12 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "FAILED"
  for c in 128 512; do
    SLOT_TAG=r1f$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --ai-force-chunk $c" SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "FAILED"
  done
done
echo L1B_DONE
