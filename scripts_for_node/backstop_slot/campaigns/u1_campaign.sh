#!/usr/bin/env bash
# U1: unit-aware Our Scheme (100% AI share, co-run bound per AI chunk from U0) vs fixed
# shares 30/50/70/100% and the prior-work combination; fair AI units (128/512/1024),
# token-based AI admission for every policy, AI 12 req/s, 2 seeds.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  for spec in 16:6.5 24:6.5 32:6.5 16:5.0; do
    IFS=: read -r cells d2 <<<"$spec"
    if [ $cells -eq 16 ]; then
      lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.5 --nrx-bound-corun-ms 3.0"
      gating="128:3.0,512:3.8,1024:4.2;conv=128,512"
    else
      lanes="--lanes-per-gpu 2 --nrx-bound-ms 3.3 --nrx-bound-corun-ms 4.7"
      gating="128:4.7,512:5.4,1024:5.8;conv=128"
    fi
    tag=$(echo $d2 | tr -d .)
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $lanes --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms $d2"
    SLOT_TAG=u${seed}d${tag} SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $gating" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "^===|FAILED"
    for pct in 30 50 70 100; do
      SLOT_TAG=u${seed}d${tag}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct $pct" \
        SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    done
    if [ "$d2" = "6.5" ]; then
      SLOT_TAG=u${seed}d${tag}s50 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --static-mps-pct 50" \
        SLOT_MATRIX="c$cells:parallel_admit:static" $M 2>&1 | grep -E "^===|FAILED"
    fi
    if [ "$d2" = "5.0" ]; then
      SLOT_TAG=u${seed}d${tag}n SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common" \
        SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
    fi
  done
done
echo U1_DONE
