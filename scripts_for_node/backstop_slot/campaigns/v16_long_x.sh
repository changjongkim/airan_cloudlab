#!/usr/bin/env bash
# v16: the server without a neural receiver and without AI in runs of 100 s (tag xa, policy code x): the first of
# the three levels of the layer-1 latency (conventional receiver alone / with the neural receiver / with AI).
# 16 cells on four GPUs at full load, as step long16 of v16_long.sh.  WITH_N=1 also runs the server with the
# neural receiver and without AI (code n) in the same job.
#   usage: bash v16_long_x.sh      (env: SEEDS, PERIODS, WITH_N)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines go() and BASE; no step of v14h.sh runs
export PERIODS=${PERIODS:-40000}
for seed in ${SEEDS:-1 2 3 4 5}; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) $BASE --weak-fraction 0.25 $RINGOPT"
  go xa${seed}x 4 "$common" "c16:off:none"
  [ -n "${WITH_N:-}" ] && go xa${seed}n 4 "$common" "c16:rescue_value:none"
done
echo V16_LONG_X_DONE
