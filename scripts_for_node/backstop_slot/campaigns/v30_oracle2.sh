#!/usr/bin/env bash
# v30: a stronger variant of the clairvoyant controller of v25_oracle.sh (a measured reference, not a setting of the
# scheme).  wo1 allows or forbids AI for a whole neural receiver run.  wqM adds the reuse-time rule: next to a run
# that is needed two slots later, AI may still run while the predicted end of that run stays before its reuse time
# (the latest start of the candidates two slots later).  M = neural receivers kept free beyond the known demand.
# Scripts: run_state/backstop_slot/scripts_oracle (controller5.py with +oracleM+reuse).
#   smoke  seed 1, 3 s: n, wd, wq1 (tag fs)       full  headline condition, seeds 1-5: n wd wo1 wq1 wq0 (tag ft)
# usage: bash v30_oracle2.sh STEP...      (env: SEEDS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines BASE, RINGOPT, go(), cond(); no step of v14h.sh runs
M="bash $R/scripts_oracle/run_matrix.sh"
T=$R/oracle
oracle() {   # oracle TAG SEED CODE GATE
  local common="--seed $((81000 + 1000 * $2)) --ring-offset $((50 * ($2 - 1))) $BASE --weak-fraction 0.25 $RINGOPT"
  go ${1}${2}r32$3 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512+yield+$4 --oracle-candidates $T/fa$2.json" "c16:rescue_value:backstop_units"
}
for step in $steps; do
  case $step in
    smoke) ( export SEEDS=1 PERIODS=1200
             cond fs 16 "32" "--weak-fraction 0.25 $RINGOPT" "n wd"
             oracle fs 1 wq1 "oracle1+reuse" ) ;;
    full)  ( export PERIODS=4000
             for seed in ${SEEDS:-1 2 3 4 5}; do
               ( export SEEDS=$seed; cond ft 16 "32" "--weak-fraction 0.25 $RINGOPT" "n wd" )
               oracle ft $seed wo1 "oracle1"
               oracle ft $seed wq1 "oracle1+reuse"
               oracle ft $seed wq0 "oracle0+reuse"
             done ) ;;
  esac
done
echo V30_ORACLE2_DONE
