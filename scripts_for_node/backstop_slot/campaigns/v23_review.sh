#!/usr/bin/env bash
# v23: measurements that answer three questions about the claims (2026-10-10).  The scheme and its options are
# unchanged; every run is a run of la_cl6.sh (closed-loop link adaptation) or of cond() of v14h.sh (fixed MCS).
#
# 1. seeds     eight two-user cells at the 1% target, more seeds (tag led; la_cl4.sh, in the queue directly)
# 2. baselines Is stopping the AI when a neural receiver starts enough?  Is checking each neural receiver's own
#              deadline enough?  Two baselines next to the rule (wm / wd), each one rule away from it:
#                wn   stop on the neural receiver only (no unit-size rule next to the conventional receiver)
#                wr0  own deadline only (AI next to a neural receiver while that run still meets its own
#                     recovery deadline; unit-size rule kept)
#              and the references n (no AI) and p100 (low priority alone).
#                cl_a  four two-user cells, target 10% (tag lba)     cl_b  four two-user cells, target 1% (tag lbb)
#                cl_c  eight two-user cells, target 1% (tag lbc)     fixed full load, fixed MCS (tag fx; n wr0 wd)
# 3. predicted Conditions that were never run, with the lost share of every policy computed before the run
#              (predict_new_conditions.py; the file of predictions is committed before the step starts):
#                oa  12 cells, 3 GPUs, 2 two-user cells    ob  12 cells, 3 GPUs, 3 two-user cells
#                oc  16 cells, 4 GPUs, 3 two-user cells    od  16 cells, 4 GPUs, 5 two-user cells
#                oe  16 cells, 4 GPUs, 7 two-user cells    of   8 cells, 2 GPUs, 3 two-user cells
# usage: bash v23_review.sh STEP...      (STEP: cl_a cl_b cl_c fixed pre_a pre_b;  env: SEEDS, POLS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
C=run_state/backstop_slot/la_cl6.sh
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines cond(); no step of v14h.sh runs
for step in $steps; do
  case $step in
    cl_a)  bash $C lba 0.10 "${SEEDS:-1 2 3}" "${POLS:-n p100 wn wr0 wm}" ;;
    cl_b)  bash $C lbb 0.01 "${SEEDS:-1 2 3}" "${POLS:-n p100 wn wr0 wm}" ;;
    cl_c)  WEAK=0.5 bash $C lbc 0.01 "${SEEDS:-1 2 3}" "${POLS:-n p100 wn wr0 wm}" ;;
    fixed) ( export SEEDS="${SEEDS:-1 2 3 4 5}" PERIODS=4000 VF_CONV=128,512
             cond fx 16 "32" "--weak-fraction 0.25 $RINGOPT" "${POLS:-n wr0 wd}" ) ;;
    pre_a) ( export SEEDS="${SEEDS:-1 2 3}" PERIODS=4000
             cond oa 12 "32" "--weak-fraction 0.1667 --gpus 3 $RINGOPT" "${POLS:-n wd p30 p70}"
             cond ob 12 "32" "--weak-fraction 0.25 --gpus 3 $RINGOPT" "${POLS:-n wd p30 p70}"
             cond oc 16 "32" "--weak-fraction 0.1875 $RINGOPT" "${POLS:-n wd p30 p70}" ) ;;
    pre_b) ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
             cond od 16 "32" "--weak-fraction 0.3125 $RINGOPT" "${POLS:-n wd p30 p70}"
             cond oe 16 "32" "--weak-fraction 0.4375 $RINGOPT" "${POLS:-n wd p30 p70}"
             cond of 8 "32" "--weak-fraction 0.375 --gpus 2 $RINGOPT" "${POLS:-n wd p30 p70}" ) ;;
  esac
done
echo V23_REVIEW_DONE
