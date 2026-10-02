#!/usr/bin/env bash
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
base="--seed 82000 --ai-admission 1 --static-mps-pct 50 --rescue-deadline-ms 6.5"
for c in 24 32; do
  SLOT_TAG=s3l1 SLOT_PERIODS=2400 SLOT_AI_RATE=8 SLOT_EXTRA="$base" SLOT_MATRIX="c$c:rescue_value:none c$c:rescue_value:backstop_corun" $M 2>&1 | grep -E "^===|FAILED"
  SLOT_TAG=s3l2 SLOT_PERIODS=2400 SLOT_AI_RATE=8 SLOT_EXTRA="$base --lanes-per-gpu 2 --nrx-bound-ms 3.3 --nrx-bound-corun-ms 3.7" SLOT_MATRIX="c$c:rescue_value:backstop_corun" $M 2>&1 | grep -E "^===|FAILED"
done
echo SMOKE_DONE
