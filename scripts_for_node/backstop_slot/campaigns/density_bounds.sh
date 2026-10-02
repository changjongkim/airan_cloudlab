#!/usr/bin/env bash
# Density phase B1: NeuralRx and conventional times while AI runs one unit size continuously
# (100% share, no admission, rescue deadline 41.5 ms so every NeuralRx runs to the end).
# S1: 40 cells (10 per GPU), weak 50%, 1 or 2 lanes.  S2: 48 cells (12 per GPU), weak 25%, 1 lane.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
probe() {   # tag cells extra...
  local tag=$1 cells=$2; shift 2
  local base="--seed 82000 --rescue-deadline-ms 41.5 --ai-admission 0 --static-mps-pct 100 --ai-chunks 128,512,1024 $*"
  SLOT_TAG=${tag}n SLOT_PERIODS=2400 SLOT_AI_RATE=12 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
  for c in 128 512 1024; do
    SLOT_TAG=${tag}f$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --ai-force-chunk $c" SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  done
}
probe dbl1 40 --lanes-per-gpu 1 --nrx-bound-ms 2.8
probe dbl2 40 --lanes-per-gpu 2 --nrx-bound-ms 4.3
probe dbw25l1 48 --weak-fraction 0.25 --lanes-per-gpu 1 --nrx-bound-ms 2.8
echo DENSITY_B1_DONE
