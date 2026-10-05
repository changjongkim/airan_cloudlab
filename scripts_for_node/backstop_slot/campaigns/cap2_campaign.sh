#!/usr/bin/env bash
# Follow-up: (a) Antiphase at 24 cells allowing 512-token AI units during conventional
# (U0/1-lane probe: conventional p99.9 3.03 ms with 512 vs 3.06 with 128), AI 12/16/20;
# fixed 50/70% at AI 20; (b) a second no-AI run per seed at 24 and 32 cells (noise floor).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
for seed in 1 2; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms 3.8 --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms 6.5"
  for rate in 12 16 20; do
    SLOT_TAG=c${seed}d65r${rate}w SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating 128:3.8,512:4.5,1024:5.2/conv=128,512" \
      SLOT_MATRIX="c24:rescue_value:backstop_units" $M 2>&1 | grep -E "FAILED"
  done
  for pct in 50 70; do
    SLOT_TAG=c${seed}d65r20s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=20 SLOT_EXTRA="$common --static-mps-pct $pct" \
      SLOT_MATRIX="c24:rescue_value:static" $M 2>&1 | grep -E "FAILED"
  done
  SLOT_TAG=c${seed}d65m SLOT_PERIODS=4000 SLOT_AI_RATE=4 SLOT_EXTRA="$common" SLOT_MATRIX="c24:rescue_value:none" $M 2>&1 | grep -E "FAILED"
  common32="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms 11.5"
  SLOT_TAG=c${seed}d115m SLOT_PERIODS=4000 SLOT_AI_RATE=4 SLOT_EXTRA="$common32" SLOT_MATRIX="c32:rescue_value:none" $M 2>&1 | grep -E "FAILED"
done
echo CAP2_DONE
