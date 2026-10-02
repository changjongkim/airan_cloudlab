#!/usr/bin/env bash
# v3 step 4: partial load.  48 cells (12 per GPU, one process per cell); each cell carries a
# TB in an uplink period with probability p, independently (bernoulli) or in busy runs of
# mean 8 periods (bursty).  Rescue deadline 6.5 ms.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
cells=48; table=${TABLE:-"128:3.5,512:3.9,1024:4.1"}
for act in "$@"; do     # e.g. b50 (bernoulli 0.5), u50 (bursty 0.5), b75
  mode=bernoulli; [ ${act:0:1} = u ] && mode=bursty
  prob=0.${act:1}
  for seed in ${SEEDS:-1 2}; do
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --controller controller3.py --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms 3.5 --ai-chunks 128,512,1024 --rescue-deadline-ms 6.5 --ai-dispatch global --activity-prob $prob --activity-mode $mode"
    for ref in n m; do go y${seed}${act}${ref} 4 "$common" "c$cells:rescue_value:none"; done
    for rate in ${RATES:-16 32}; do
      go y${seed}${act}r${rate}u $rate "$common --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/conv=128,512,1024+queue" "c$cells:rescue_value:backstop_units"
      for pct in 30 50 70; do
        go y${seed}${act}r${rate}s$pct $rate "$common --static-mps-pct $pct" "c$cells:rescue_value:static"
      done
    done
  done
done
echo V3_PARTIAL_DONE
