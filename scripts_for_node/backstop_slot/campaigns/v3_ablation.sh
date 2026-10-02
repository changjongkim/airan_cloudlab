#!/usr/bin/env bash
# v3 step 7: remove one mechanism at a time from the final configuration (v3e70).
#   full      capped AI share (70%), sustained chunk choice, calibrated conventional overlap,
#             value rule, global AI dispatch (+ yield at 32 cells)
#   nocap     AI share 100% with the 100% co-run bounds
#   budget    chunk chosen by what is left of the piece (v2 rule)
#   convv2    conventional-safe sizes as in v2
#   novalue   NeuralRx for every conventional failure
#   nodisp    AI requests arrive per GPU instead of being placed by the controller
#   noyield   (32 cells) AI is granted even while NeuralRx TBs wait for a lane
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
setting() {   # cells d2 bound rate cap-table full-table conv-full conv-v2 budget yield
  local cells=$1 d2=$2 b=$3 rate=$4 t70=$5 t100=$6 cfull=$7 cv2=$8 budget=$9 y=${10}
  for seed in 1 2; do
    local base="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms $b --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --static-mps-pct 100 --ai-max-piece-ms 4.0"
    local disp="--ai-dispatch global"
    go a${seed}n 4 "$base $disp" "c$cells:rescue_value:none"
    go a${seed}full $rate "$base $disp --gated-mps-pct 70 --ai-chunk-choice sustained --unit-gating $t70/conv=$cfull$y $budget" "c$cells:rescue_value:backstop_units"
    go a${seed}nocap $rate "$base $disp --ai-chunk-choice sustained --unit-gating $t100/conv=$cfull$y $budget" "c$cells:rescue_value:backstop_units"
    go a${seed}budget $rate "$base $disp --gated-mps-pct 70 --ai-chunk-choice budget --unit-gating $t70/conv=$cfull$y $budget" "c$cells:rescue_value:backstop_units"
    go a${seed}convv2 $rate "$base $disp --gated-mps-pct 70 --ai-chunk-choice sustained --unit-gating $t70/conv=$cv2$y" "c$cells:rescue_value:backstop_units"
    go a${seed}novalue $rate "$base $disp --gated-mps-pct 70 --ai-chunk-choice sustained --unit-gating $t70/conv=$cfull$y $budget" "c$cells:rescue:backstop_units"
    go a${seed}nodisp $rate "$base --gated-mps-pct 70 --ai-chunk-choice sustained --unit-gating $t70/conv=$cfull$y $budget" "c$cells:rescue_value:backstop_units"
    [ -n "$y" ] && go a${seed}noyield $rate "$base $disp --gated-mps-pct 70 --ai-chunk-choice sustained --unit-gating $t70/conv=$cfull $budget" "c$cells:rescue_value:backstop_units"
  done
}
for spec in "$@"; do
  case $spec in
    24) setting 24 6.5 2.8 32 "128:3.5,512:3.9,1024:4.1" "128:3.8,512:4.5,1024:5.2" "128,512,1024" "128,512" "" "" ;;
    32) setting 32 11.5 2.8 16 "128:4.0,512:4.5,1024:4.6" "128:4.1,512:4.7,1024:5.3" "128,512" "128" "--conv-budget alone=2.8/margin=0.2/1024:1.0" "+yield" ;;
  esac
done
echo V3_ABLATION_DONE
