#!/usr/bin/env bash
# Global AI dispatch (same dispatcher for every policy): 24 cells @6.5 ms and 32 cells @11.5 ms.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  for spec in 24:6.5 32:11.5; do
    IFS=: read -r cells d2 <<<"$spec"
    if [ $cells -eq 24 ]; then ours="128:3.8,512:4.5,1024:5.2/conv=128,512"; bc=3.8
    else ours="128:4.1,512:4.7,1024:5.3/conv=128+yield"; bc=4.1; fi
    tag=$(echo $d2 | tr -d .)
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms $d2 --ai-dispatch global"
    for rate in 12 16 20; do
      SLOT_TAG=g${seed}d${tag}r${rate}u SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $ours" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "FAILED"
      for pct in 50 70; do
        SLOT_TAG=g${seed}d${tag}r${rate}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct $pct" \
          SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "FAILED"
      done
    done
  done
done
echo G_DONE
