#!/usr/bin/env bash
# v16, second part: the same 100-s runs for the policies that v16_long.sh leaves out at 16 cells (the two
# estimator policies and low priority without a cap), so that every policy of the headline has a steady-state
# L1 statistic from the same job.  Tag xa, seeds 1-5; the runs of v16_long.sh in this job are skipped.
# usage: bash v16_long_b.sh [extra16]      (env: SEEDS, POLS, PERIODS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export YYR_EST=$PWD/results/backstop_slot/yyr_estimator_fixed_j59210955.json
export YYP_EST=$PWD/results/backstop_slot/yyr_estimator_lowprio_j59210955.json
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
( export SEEDS="${SEEDS:-1 2 3 4 5}" PERIODS=${PERIODS:-40000}
  cond xa 16 "32" "--weak-fraction 0.25 $RINGOPT" "${POLS:-p100 yyp yyr}" )
echo V16_LONG_B_DONE
