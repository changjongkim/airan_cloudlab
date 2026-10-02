#!/usr/bin/env bash
# Density phase B2: AI capacity under the radio targets at 10 and 12 cells per GPU.
# No AI unit is safe next to the conventional receiver at these densities (B1), so Our Scheme
# keeps every unit out of the conventional phase; fixed shares run as before.
#   a: 40 cells, weak 50%, 2 lanes/GPU, rescue deadline 11.5 ms
#   b: 48 cells, weak 25%, 1 lane/GPU,  rescue deadline 11.5 ms
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
run_setting() {   # key cells radio-options unit-gating corun128
  local key=$1 cells=$2 radio=$3 gating=$4 bc=$5
  for seed in 1 2; do
    local common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $radio --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms 11.5 --ai-dispatch global"
    for ref in n m; do
      SLOT_TAG=e${seed}${key}${ref} SLOT_PERIODS=4000 SLOT_AI_RATE=4 SLOT_EXTRA="$common" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
    done
    for rate in 2 4; do
      SLOT_TAG=e${seed}${key}r${rate}u SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $gating" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
      for pct in 30 50; do
        SLOT_TAG=e${seed}${key}r${rate}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct $pct" \
          SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
      done
    done
  done
}
run_setting a 40 "--lanes-per-gpu 2 --nrx-bound-ms 4.5" "128:5.3,512:5.7,1024:6.2/conv=+yield" 5.3
run_setting b 48 "--weak-fraction 0.25 --lanes-per-gpu 1 --nrx-bound-ms 4.3" "128:5.1,512:5.3,1024:5.7/conv=+yield" 5.1
echo DENSITY_B2_DONE
