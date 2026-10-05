#!/usr/bin/env bash
# v3c: Antiphase with the NeuralRx slack budget (delay per ms of AI overlap instead of the
# all-or-nothing co-run bound), calibrated conventional overlap and sustained chunk choice.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
setting() {   # cells d2 radio corun-table gating budget rates...
  local cells=$1 d2=$2 radio=$3 table=$4 gating=$5 budget=$6; shift 6
  local bc=${table#128:}; bc=${bc%%,*}
  for seed in ${SEEDS:-1 2}; do
    local common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $radio --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/$gating+nrxbudget $budget"
    for rate in "$@"; do
      go h${seed}r${rate}v3c $rate "$common" "c$cells:rescue_value:backstop_units"
    done
  done
}
for spec in "$@"; do
  case $spec in
    24) setting 24 6.5 "--lanes-per-gpu 1 --nrx-bound-ms 2.8" "128:3.8,512:4.5,1024:5.2" "conv=128,512,1024" "" 16 32 48 ;;
    32) setting 32 11.5 "--lanes-per-gpu 1 --nrx-bound-ms 2.8" "128:4.1,512:4.7,1024:5.3" "conv=128,512+yield" "--conv-budget alone=2.8/margin=0.2/1024:1.0" 8 16 32 ;;
    16) setting 16 6.5 "--lanes-per-gpu 1 --nrx-bound-ms 2.5" "128:3.1,512:3.9,1024:4.3" "conv=128,512,1024" "" 16 32 48 ;;
  esac
done
echo V3C_DONE
