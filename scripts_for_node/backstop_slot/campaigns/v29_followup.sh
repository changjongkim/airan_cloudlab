#!/usr/bin/env bash
# v29: follow-ups of the measurements of 2026-10-10 (README 4.14, 4.16.6).  The scheme and its options are unchanged.
#   cap_a   the 70% cap under rate control with a later recovery deadline: eight two-user cells, target 1%,
#           deadline 12.0 ms (tag lcy) and 12.5 ms (tag lcz); n wm p70, seeds 1-3
#   cap_b   the same at 12.5 ms with four two-user cells: target 1% (tag ldb) and target 10% (tag lda)
#   long32  the rule on unit sizes in runs of 100 s with 32 cells at half load (tag xm): n wn wd, seeds 1-3
#   (20 cells, more seeds:  SEEDS="4 5 6" POLS="n wn wd" bash v16_long.sh long20)
# usage: bash v29_followup.sh STEP...      (env: SEEDS, POLS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
C=run_state/backstop_slot/la_cl6.sh
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
for step in $steps; do
  case $step in
    cap_a)  WEAK=0.5 EXTRA="--rescue-deadline-ms 12.0" bash $C lcy 0.01 "${SEEDS:-1 2 3}" "${POLS:-n wm p70}"
            WEAK=0.5 EXTRA="--rescue-deadline-ms 12.5" bash $C lcz 0.01 "${SEEDS:-1 2 3}" "${POLS:-n wm p70}" ;;
    cap_b)  EXTRA="--rescue-deadline-ms 12.5" bash $C ldb 0.01 "${SEEDS:-1 2 3}" "${POLS:-n wm p70}"
            EXTRA="--rescue-deadline-ms 12.5" bash $C lda 0.10 "${SEEDS:-1 2 3}" "${POLS:-n wm p70}" ;;
    long32) ( export SEEDS="${SEEDS:-1 2 3}" PERIODS=${PERIODS:-40000}
              cond xm 32 "32" "--weak-fraction 0.25 --activity-prob 0.5 $RINGOPT" "${POLS:-n wn wd}" ) ;;
  esac
done
echo V29_FOLLOWUP_DONE
