#!/usr/bin/env bash
# v25: a clairvoyant controller as a measured reference (not a setting of the scheme).  It reads the number of
# candidates of every slot from a run without AI of the same seed (which slots are candidates is fixed by the data
# set, the cell and the seed) and lets the AI run next to the neural receiver runs whose receiver is not needed two
# slots later; next to the others the AI stops as in the rule.  The unit-size rule next to the conventional receiver
# is that of the rule.  wo0: no receiver kept free beyond the known demand; wo1: one.
# The runs use a copy of the script directory (run_state/backstop_slot/scripts_oracle: controller5.py and
# make_config.py with the option oracle_candidates), so the scripts of the other campaigns are untouched.
#   smoke  seed 1, 3 s: n, wd, wo0 (tag fq)       full  headline condition, seeds 1-5: n wd p100 wo0 wo1 (tag fo)
# Tables of candidates: python3 oracle_table.py run_state/backstop_slot/oracle fa 16 59313958 1 2 3 4 5
# usage: bash v25_oracle.sh STEP...      (env: SEEDS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
steps="$*"
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines BASE, RINGOPT, go(), cond(); no step of v14h.sh runs
M="bash $R/scripts_oracle/run_matrix.sh"
T=$R/oracle
oracle() {   # oracle TAG SEED MARGIN
  local common="--seed $((81000 + 1000 * $2)) --ring-offset $((50 * ($2 - 1))) $BASE --weak-fraction 0.25 $RINGOPT"
  go ${1}${2}r32wo$3 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512+yield+oracle$3 --oracle-candidates $T/fa$2.json" "c16:rescue_value:backstop_units"
}
for step in $steps; do
  case $step in
    smoke) ( export SEEDS=1 PERIODS=1200
             cond fq 16 "32" "--weak-fraction 0.25 $RINGOPT" "n wd"
             oracle fq 1 0 ) ;;
    full)  ( export PERIODS=4000
             for seed in ${SEEDS:-1 2 3 4 5}; do
               ( export SEEDS=$seed; cond fo 16 "32" "--weak-fraction 0.25 $RINGOPT" "n wd p100" )
               oracle fo $seed 0
               oracle fo $seed 1
             done ) ;;
  esac
done
echo V25_ORACLE_DONE
