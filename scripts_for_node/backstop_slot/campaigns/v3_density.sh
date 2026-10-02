#!/usr/bin/env bash
# v3 step 3: AI capacity under the radio targets across cells per GPU (4, 6, 8, 10).
#   v3  conventional overlap by calibration (job 59151256) + adaptive AI unit bounds
#   v2  the earlier final configuration
#   fixed MPS shares; Orion-like approximations (o1: AI only while the GPU has no radio work,
#   o2: only the smallest AI unit, always allowed)
# Same NeuralRx rules, AI units, admission and global dispatch for every policy.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() {   # tag rate extra matrix
  SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"
}
setting() {   # cells d2 radio corun-table v2-gating v3-gating v3-budget shares rates...
  local cells=$1 d2=$2 radio=$3 table=$4 v2=$5 v3=$6 v3b=$7 shares=$8; shift 8
  local bc=${table#128:}; bc=${bc%%,*}
  for seed in 1 2; do
    local common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $radio --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global"
    for ref in n m; do go h${seed}${ref} 4 "$common" "c$cells:rescue_value:none"; done
    for rate in "$@"; do
      go h${seed}r${rate}v3 $rate "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-adaptive-bound 1 --unit-gating $table/$v3 $v3b" "c$cells:rescue_value:backstop_units"
      [ -n "$v2" ] && go h${seed}r${rate}v2 $rate "$common --static-mps-pct 100 --ai-max-piece-ms 3.0 --unit-gating $table/$v2" "c$cells:rescue_value:backstop_units"
      for pct in $shares; do
        go h${seed}r${rate}s$pct $rate "$common --static-mps-pct $pct" "c$cells:rescue_value:static"
      done
    done
    if [ $seed -eq 1 ] && [ $# -ge 2 ]; then
      go h1r${2}o1 $2 "$common --static-mps-pct 100 --ai-max-piece-ms 3.0" "c$cells:rescue_value:backstop"
      go h1r${2}o2 $2 "$common --static-mps-pct 100 --ai-force-chunk 128" "c$cells:rescue_value:static"
    fi
  done
}
for spec in "$@"; do
  case $spec in
    24) setting 24 6.5 "--lanes-per-gpu 1 --nrx-bound-ms 2.8" "128:3.8,512:4.5,1024:5.2" "conv=128,512" "conv=128,512,1024" "" "30 50 70" 16 32 48 ;;
    32) setting 32 11.5 "--lanes-per-gpu 1 --nrx-bound-ms 2.8" "128:4.1,512:4.7,1024:5.3" "conv=128+yield" "conv=128,512+yield" "--conv-budget alone=2.8/margin=0.2/1024:1.0" "30 50 70" 8 16 32 ;;
    16) setting 16 6.5 "--lanes-per-gpu 1 --nrx-bound-ms 2.5" "128:3.1,512:3.9,1024:4.3" "conv=128,512" "conv=128,512,1024" "" "50 70 100" 16 32 48 ;;
    40) setting 40 11.5 "--lanes-per-gpu 2 --nrx-bound-ms 4.5" "128:5.3,512:5.7,1024:6.2" "" "conv=+yield" "--conv-budget alone=2.8/margin=0.2/128:1.0" "30 50" 4 8 ;;
  esac
done
echo V3_DENSITY_DONE
