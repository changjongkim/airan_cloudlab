#!/usr/bin/env bash
# v16: long runs (100 s, 40,000 uplink slots per cell) for the statistics of the L1 misses and of the latency
# tails.  The scheme and every option are those of v14h.sh (policy code wd = the final rule: the two smaller
# unit classes next to the conventional receiver for any number of active cells); only the run length changes.
#   long16  16 cells, four GPUs, full load              (tag xa; seeds 1-5; n wd s10 p30 p70)
#   long1   4 cells, one GPU, one two-user cell         (tag xs; seeds 1-5; n wd s10 p30)
#   long20  20 cells, four GPUs, NeuralRx bound 7.6 ms  (tag xv; seeds 1-3; n wd s10 p30)
#   long48  48 cells, four GPUs, one third load         (tag xn; seeds 1-3; n wd s10 p30)
# usage: bash v16_long.sh STEP...      (env: SEEDS, POLS, PERIODS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
for step in $steps; do
  case $step in
    long16) ( export SEEDS="${SEEDS:-1 2 3 4 5}" PERIODS=${PERIODS:-40000}
              cond xa 16 "32" "--weak-fraction 0.25 $RINGOPT" "${POLS:-n wd s10 p30 p70}" ) ;;
    long1)  ( export SEEDS="${SEEDS:-1 2 3 4 5}" PERIODS=${PERIODS:-40000}
              cond xs 4 "32" "--weak-fraction 0.25 --gpus 1 $RINGOPT" "${POLS:-n wd s10 p30}" ) ;;
    long20) ( export SEEDS="${SEEDS:-1 2 3}" PERIODS=${PERIODS:-40000}
              cond xv 20 "32" "--weak-fraction 0.2 --nrx-bound-ms 7.6 $RINGOPT" "${POLS:-n wd s10 p30}" ) ;;
    long48) ( export SEEDS="${SEEDS:-1 2 3}" PERIODS=${PERIODS:-40000}
              cond xn 48 "32" "--weak-fraction 0.25 --activity-prob 0.3333 $RINGOPT" "${POLS:-n wd s10 p30}" ) ;;
  esac
done
echo V16_LONG_DONE
