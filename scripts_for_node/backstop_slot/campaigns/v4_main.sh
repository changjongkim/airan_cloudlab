#!/usr/bin/env bash
# v4 main comparison (NeuralRx fed the NVlabs LS layout; rescue = conventional failure with
# one failed code block).  Our Scheme (v3e70 + yield + stall fix) vs fixed GPU shares, same
# NeuralRx rules, AI units, admission and global dispatch; AI load up to saturation; 2 seeds.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
setting() {   # cells d2 bound table gating budget shares rates...
  local cells=$1 d2=$2 b=$3 table=$4 gating=$5 budget=$6 shares=$7; shift 7
  local bc=${table#128:}; bc=${bc%%,*}
  for seed in 1 2; do
    local common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --controller controller3.py --lanes-per-gpu 1 --nrx-bound-ms $b --nrx-bound-corun-ms $bc --nrx-max-cb-fail ${KFAIL:-1} ${NRXFLAGS:-} --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global"
    for ref in n m; do go ${TAG:-m}${seed}${ref} 4 "$common" "c$cells:rescue_value:none"; done
    for rate in "$@"; do
      go ${TAG:-m}${seed}r${rate}v4 $rate "$common --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/$gating $budget" "c$cells:rescue_value:backstop_units"
      for pct in $shares; do
        go ${TAG:-m}${seed}r${rate}s$pct $rate "$common --static-mps-pct $pct" "c$cells:rescue_value:static"
      done
    done
  done
}
for spec in "$@"; do
  case $spec in
    24) setting 24 6.5 2.8 "128:3.5,512:3.9,1024:4.1" "conv=128,512,1024+yield" "" "30 50 70" 16 32 48 ;;
    32) setting 32 11.5 2.8 "128:4.0,512:4.5,1024:4.6" "conv=128,512+yield" "--conv-budget alone=2.8/margin=0.2/1024:1.0" "30 50 70" 8 16 32 ;;
    16) setting 16 6.5 2.5 "128:2.9,512:3.4,1024:3.4" "conv=128,512,1024" "" "50 70 100" 16 32 48 ;;
  esac
done
echo V4_MAIN_DONE
