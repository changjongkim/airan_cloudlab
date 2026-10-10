#!/usr/bin/env bash
# v24: the recovery deadline at 12.0 and 12.5 ms.  The default of the scheme stays 11.5 ms; with a MAC lead of three
# slots the decision for the retransmission slot could be as late as 12.0-12.5 ms (README 3.4), and every result so far
# is at one deadline.  The time limit of a neural receiver run stays 7.6 ms, so the latest start is 4.4 and 4.9 ms.
#   fixed  headline condition (16 cells, four GPUs, four two-user cells, full load, fixed MCS): tags fy (12.0), fz (12.5)
#          The lost candidates of every policy are computed before the run (predict_new_conditions.py deadline;
#          results/backstop_slot/predicted_deadline.json is committed before the step starts).
#   cl     eight two-user cells at the 1% target under rate control: tags lby (12.0), lbz (12.5)
# usage: bash v24_deadline.sh STEP...      (env: SEEDS, POLS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
C=run_state/backstop_slot/la_cl6.sh
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
for step in $steps; do
  case $step in
    fixed) ( export SEEDS="${SEEDS:-1 2 3}" PERIODS=4000
             cond fy 16 "32" "--weak-fraction 0.25 $RINGOPT --rescue-deadline-ms 12.0" "${POLS:-n wd p30 p70 p100}"
             cond fz 16 "32" "--weak-fraction 0.25 $RINGOPT --rescue-deadline-ms 12.5" "${POLS:-n wd p30 p70 p100}" ) ;;
    cl)    WEAK=0.5 EXTRA="--rescue-deadline-ms 12.0" bash $C lby 0.01 "${SEEDS:-1 2 3}" "${POLS:-n wm p30 p100}"
           WEAK=0.5 EXTRA="--rescue-deadline-ms 12.5" bash $C lbz 0.01 "${SEEDS:-1 2 3}" "${POLS:-n wm p30 p100}" ;;
  esac
done
echo V24_DEADLINE_DONE
