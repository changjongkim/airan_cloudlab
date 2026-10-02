#!/usr/bin/env bash
# v3 step 2 calibration: how much one AI unit size delays the conventional receivers per ms
# of overlap.  AI is saturated (no admission) and may overlap the conventional phase of each
# period for X ms (X = 0, 0.5, 1, 2); outside it AI runs freely.  The rescue deadline is
# 41.5 ms so the NeuralRx rule never limits AI.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
cal() {   # cells radio-options
  local cells=$1 radio=$2
  local base="--seed 82000 $radio --rescue-deadline-ms 41.5 --ai-admission 0 --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --static-mps-pct 100 --ai-adaptive-bound 1"
  SLOT_TAG=kn SLOT_PERIODS=2400 SLOT_AI_RATE=4 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
  SLOT_TAG=kx0c128 SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --ai-force-chunk 128 --unit-gating 128:5,512:5,1024:5/conv= --conv-budget alone=2.8/margin=0.2/128:1000,512:1000,1024:1000" \
    SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
  for c in 128 512 1024; do
    for x in 0.5 1.0 2.0; do
      local a=$(python3 -c "print(1/$x)")
      SLOT_TAG=kx$(echo $x | tr -d .)c$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --ai-force-chunk $c --unit-gating 128:5,512:5,1024:5/conv= --conv-budget alone=2.8/margin=0.2/128:$a,512:$a,1024:$a" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
    done
  done
}
for spec in "$@"; do
  case $spec in
    16) cal 16 "--lanes-per-gpu 1 --nrx-bound-ms 2.5" ;;
    24) cal 24 "--lanes-per-gpu 1 --nrx-bound-ms 2.8" ;;
    32) cal 32 "--lanes-per-gpu 1 --nrx-bound-ms 2.8" ;;
    40) cal 40 "--lanes-per-gpu 2 --nrx-bound-ms 4.5" ;;
  esac
done
echo V3_CAL_DONE
