#!/usr/bin/env bash
# v3 step 5: several AI job classes on each GPU (24 cells, rescue deadline 6.5 ms).
#   chat  Qwen2.5-1.5B prefill, SLO 200 ms      small  Qwen2.5-0.5B prefill, SLO 100 ms
#   batch Qwen2.5-1.5B, always-full queue of 1,024-token prompts, no SLO
# Antiphase serves the most urgent class whose unit fits (urgency) or one line in arrival
# order (fifo); fixed shares use the same worker without grants.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
cells=24; table="128:3.5,512:3.9,1024:4.1"
for seed in ${SEEDS:-1 2}; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms 3.5 --ai-chunks 128,512,1024 --rescue-deadline-ms 6.5"
  for ref in n m; do go o${seed}${ref} 4 "$common" "c$cells:rescue_value:none"; done
  for rate in ${RATES:-6 12}; do
    for mix in ${MIXES:-cs csb}; do
      case $mix in
        cs) classes="chat:1.5B:$rate:200,small:0.5B:$rate:100" ;;
        csb) classes="chat:1.5B:$rate:200,small:0.5B:$rate:100,batch:1.5B" ;;
      esac
      ours="$common --ai-classes $classes --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/conv=128,512,1024"
      go o${seed}r${rate}${mix}u $rate "$ours" "c$cells:rescue_value:backstop_units"
      [ $mix = csb ] && go o${seed}r${rate}${mix}f $rate "$ours --ai-class-order fifo" "c$cells:rescue_value:backstop_units"
      for pct in 50 70; do
        go o${seed}r${rate}${mix}s$pct $rate "$common --ai-classes $classes --static-mps-pct $pct" "c$cells:rescue_value:static"
      done
    done
  done
done
echo V3_CLASSES_DONE
