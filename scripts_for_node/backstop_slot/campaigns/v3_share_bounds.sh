#!/usr/bin/env bash
# NeuralRx co-run time when AI runs one unit size continuously under an MPS share cap.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
cells=$1; radio=$2; shift 2
for pct in "$@"; do
  for c in 128 512 1024; do
    SLOT_TAG=b${pct}f$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="--seed 82000 $radio --rescue-deadline-ms 41.5 --ai-admission 0 --static-mps-pct $pct --ai-chunks 128,512,1024 --ai-force-chunk $c" \
      SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo BOUNDS_DONE
