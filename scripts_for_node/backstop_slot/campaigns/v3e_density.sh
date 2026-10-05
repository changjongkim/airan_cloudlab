#!/usr/bin/env bash
# v3e: Antiphase with the AI share capped (MPS) and co-run bounds measured under that cap.
# usage: v3e_density.sh cells d2 radio gating-suffix budget pct table rates...
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
cells=$1 d2=$2 radio=$3 gating=$4 budget=$5 pct=$6 table=$7; shift 7
bc=${table#128:}; bc=${bc%%,*}
for seed in ${SEEDS:-1 2}; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $radio --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global --static-mps-pct 100 --gated-mps-pct $pct --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/$gating $budget"
  for rate in "$@"; do
    go h${seed}r${rate}v3e$pct $rate "$common" "c$cells:rescue_value:backstop_units"
  done
done
echo V3E_DONE
