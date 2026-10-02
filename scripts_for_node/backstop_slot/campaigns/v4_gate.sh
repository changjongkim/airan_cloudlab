#!/usr/bin/env bash
# Which protection for waiting NeuralRx TBs: none, yield (no AI while any TB waits), or the
# queue feasibility check.  Our Scheme, k1, one seed.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
one() {  # cells d2 table conv rate budget
  local cells=$1 d2=$2 table=$3 conv=$4 rate=$5 budget=$6
  local bc=${table#128:}; bc=${bc%%,*}
  local base="--seed 82000 --ai-admission 1 --controller controller3.py --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global --nrx-max-cb-fail 1 --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained $budget"
  for g in "" "+yield" "+queue"; do
    go g$(echo "${g:-+none}" | tr -d +)u $rate "$base --unit-gating $table/conv=$conv$g" "c$cells:rescue_value:backstop_units"
  done
}
one 24 6.5 "128:3.5,512:3.9,1024:4.1" "128,512,1024" 32 ""
one 32 11.5 "128:4.0,512:4.5,1024:4.6" "128,512" 16 "--conv-budget alone=2.8/margin=0.2/1024:1.0"
echo V4_GATE_DONE
