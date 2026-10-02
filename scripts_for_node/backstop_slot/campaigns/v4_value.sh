#!/usr/bin/env bash
# v4 (NeuralRx fed the NVlabs LS layout): which TBs to rescue.  No AI and AI at one load.
#   k1/k2/k3  NeuralRx for conventional failures with at most 1/2/3 failed code blocks
#   v         serve fewer-failed-code-block TBs first (value order); e = deadline order only
#   legacy    the LS layout used before 2026-10-01 (k1)
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
setting() {   # cells d2 table gating lanes
  local cells=$1 d2=$2 table=$3 gating=$4 lanes=$5 rate=$6
  local bc=${table#128:}; bc=${bc%%,*}
  local base="--seed 82000 --ai-admission 1 --controller controller3.py --lanes-per-gpu $lanes --nrx-bound-ms 2.8 --nrx-bound-busy 2.8,4.0 --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global"
  local ai="--static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/$gating"
  for variant in k1e k2v k3v k2e k3e legacy; do
    case $variant in
      k1e) nrx="--nrx-max-cb-fail 1 --nrx-flags second_lane=deadline" ;;
      k2v) nrx="--nrx-max-cb-fail 2 --nrx-flags second_lane=deadline,rank=value" ;;
      k3v) nrx="--nrx-max-cb-fail 3 --nrx-flags second_lane=deadline,rank=value" ;;
      k2e) nrx="--nrx-max-cb-fail 2 --nrx-flags second_lane=deadline" ;;
      k3e) nrx="--nrx-max-cb-fail 3 --nrx-flags second_lane=deadline" ;;
      legacy) nrx="--nrx-max-cb-fail 1 --nrx-flags second_lane=deadline --nrx-ls-input example" ;;
    esac
    go f${lanes}${variant}n 4 "$base $nrx" "c$cells:rescue_value:none"
    [ $variant = legacy ] && continue
    go f${lanes}${variant}u $rate "$base $nrx $ai" "c$cells:rescue_value:backstop_units"
  done
}
for spec in "$@"; do
  case $spec in
    24a) setting 24 6.5 "128:3.5,512:3.9,1024:4.1" "conv=128,512,1024+queue" 1 32 ;;
    24b) setting 24 6.5 "128:3.5,512:3.9,1024:4.1" "conv=128,512,1024+queue" 2 32 ;;
    32a) setting 32 11.5 "128:4.0,512:4.5,1024:4.6" "conv=128,512+queue" 1 16 ;;
    32b) setting 32 11.5 "128:4.0,512:4.5,1024:4.6" "conv=128,512+queue" 2 16 ;;
  esac
done
echo V4_VALUE_DONE
