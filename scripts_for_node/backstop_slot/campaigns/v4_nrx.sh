#!/usr/bin/env bash
# v4 (corrected NeuralRx input): NeuralRx-primary cells.  P weak cells have every TB decoded by NeuralRx from
# arrival (deadline = rescue deadline); the other weak cells keep the rescue rule.  Two lanes
# per GPU, the second used only when both TBs still meet their deadlines.  Same NeuralRx
# rules for every policy; only the AI side differs.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
PERIODS=${PERIODS:-4000}
go() { SLOT_TAG=$1 SLOT_PERIODS=$PERIODS SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
cells=24; d2=6.5; table="128:3.5,512:3.9,1024:4.1"
for p in "$@"; do
  for seed in ${SEEDS:-1 2}; do
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --controller controller3.py --lanes-per-gpu 2 --nrx-bound-ms 2.8 --nrx-bound-busy 2.8,4.0 --nrx-flags second_lane=deadline --nrx-primary-cells $p --nrx-bound-corun-ms 3.5 --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --ai-dispatch global"
    go ${TAG:-p}${seed}p${p}n 4 "$common" "c$cells:rescue_value:none"
    for rate in ${RATES:-16 32}; do
      go ${TAG:-p}${seed}p${p}r${rate}u $rate "$common --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/conv=128,512,1024${GATE:-+queue}" "c$cells:rescue_value:backstop_units"
      for pct in 50 70; do
        go ${TAG:-p}${seed}p${p}r${rate}s$pct $rate "$common --static-mps-pct $pct" "c$cells:rescue_value:static"
      done
    done
  done
done
echo V4_NRX_DONE
