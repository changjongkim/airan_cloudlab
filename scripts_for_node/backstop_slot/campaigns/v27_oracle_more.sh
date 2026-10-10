#!/usr/bin/env bash
# v27: the clairvoyant controller of v25_oracle.sh in the other conditions for which the offline schedule was
# estimated (README 4.12): other demand for the neural receiver and a load that changes.  Same scripts
# (run_state/backstop_slot/scripts_oracle), same policies: n, wd (the rule), wo0, wo1.  The tables of candidates
# come from the runs without AI of the same condition and seed (oracle_table.py; tags sx sy sw fb fc).
#   demand  two / six / eight two-user cells of sixteen (tags ox oy ow)
#   load    load alternates full / half every 2 s (tag og, 20 s); load steps at random times (tag oh, 40 s)
# usage: bash v27_oracle_more.sh STEP...      (env: SEEDS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines BASE, RINGOPT, go(), cond(); no step of v14h.sh runs
M="bash $R/scripts_oracle/run_matrix.sh"
T=$R/oracle
oracle() {   # oracle TAG SEED MARGIN TABLE CELLS "EXTRA"
  local common="--seed $((81000 + 1000 * $2)) --ring-offset $((50 * ($2 - 1))) $BASE $6"
  go ${1}${2}r32wo$3 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512+yield+oracle$3 --oracle-candidates $T/$4$2.json" "c$5:rescue_value:backstop_units"
}
one() {      # one TAG TABLE CELLS "EXTRA": n, wd, wo0, wo1 of every seed
  local seed
  for seed in $SEEDS; do
    ( export SEEDS=$seed; cond $1 $3 "32" "$4" "n wd" )
    oracle $1 $seed 0 $2 $3 "$4"
    oracle $1 $seed 1 $2 $3 "$4"
  done
}
for step in $steps; do
  case $step in
    demand) ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
              one ox sx 16 "--weak-fraction 0.125 $RINGOPT"
              one oy sy 16 "--weak-fraction 0.375 $RINGOPT"
              one ow sw 16 "--weak-fraction 0.5 $RINGOPT" ) ;;
    load)   ( export SEEDS="${SEEDS:-1 2}" PERIODS=8000
              one og fb 16 "--weak-fraction 0.25 --activity-prob 0.5 --activity-mode phased --activity-phase 800 $RINGOPT" )
            ( export SEEDS="${SEEDS:-1 2}" PERIODS=16000
              one oh fc 16 "--weak-fraction 0.25 --activity-mode steps --activity-phase 400 $RINGOPT" ) ;;
  esac
done
echo V27_ORACLE_MORE_DONE
