#!/usr/bin/env bash
# Larger AI shares: Our Scheme at 70/100% MPS vs a fixed 70/100% share, fair AI units
# (chunks 128/512/1024), AI 12 req/s, 2 seeds. 16 cells at rescue deadline 4.0/5.0 ms,
# 32 cells at 5.0/6.5 ms.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  for spec in 16:4.0 16:5.0 32:5.0 32:6.5; do
    IFS=: read -r cells d2 <<<"$spec"
    if [ $cells -eq 16 ]; then lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.5 --nrx-bound-corun-ms 2.8"
    else lanes="--lanes-per-gpu 2 --nrx-bound-ms 3.3 --nrx-bound-corun-ms 3.7"; fi
    tag=$(echo $d2 | tr -d .)
    for pct in 70 100; do
      common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $lanes --static-mps-pct $pct --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms $d2"
      SLOT_TAG=q${seed}d${tag}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_corun c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    done
  done
done
echo SHARE_DONE
