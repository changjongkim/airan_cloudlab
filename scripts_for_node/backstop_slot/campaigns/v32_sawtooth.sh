#!/usr/bin/env bash
# v32: the recovery deadline in steps of 0.25 ms from 11.0 to 14.0 ms (latest start 3.4 to 6.4 ms; the time limit of
# a neural receiver run stays 7.6 ms).  The default of the scheme stays 11.5 ms.  The loss model says that the lost
# candidates fall in steps as the deadline moves, at a deadline that differs by policy (where its runs go from three
# slots to two and from two to one).  The predictions are computed before the runs (predict_deadline_sweep.py;
# results/backstop_slot/predicted_deadline_sweep.json is committed before the steps start).
# Headline condition (16 cells, four GPUs, four two-user cells, full load, fixed MCS); 11.5, 12.0, 12.5 ms were run
# before (tags fa, fy, fz).
#   a  11.0 (ja) 11.25 (jb) 11.75 (jc) 12.25 (jd) 12.75 (je)       b  13.0 (jf) 13.25 (jg) 13.5 (jh) 13.75 (ji) 14.0 (jk)
# usage: bash v32_sawtooth.sh STEP...      (env: SEEDS, POLS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
dl() { cond $1 16 "32" "--weak-fraction 0.25 $RINGOPT --rescue-deadline-ms $2" "${POLS:-n wd p30 p70}"; }
export SEEDS="${SEEDS:-1 2}" PERIODS=4000
for step in $steps; do
  case $step in
    a) dl ja 11.0; dl jb 11.25; dl jc 11.75; dl jd 12.25; dl je 12.75 ;;
    b) dl jf 13.0; dl jg 13.25; dl jh 13.5; dl ji 13.75; dl jk 14.0 ;;
  esac
done
echo V32_SAWTOOTH_DONE
