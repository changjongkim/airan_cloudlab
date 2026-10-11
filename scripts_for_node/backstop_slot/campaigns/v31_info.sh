#!/usr/bin/env bash
# v31: the value of knowing the failures in advance.  The clairvoyant controller of v25_oracle.sh (one neural receiver
# kept free; a measured reference, not a setting of the scheme) reads a table of the candidates of every slot.  Here
# the table has errors (oracle_noise.py): misses (rNN: NN% of the true candidates are in the table) and false alarms
# (fNN: NN% of the cell-slots without a candidate are shown as one), and one table with both (m: r75 and f10).
# Headline condition (16 cells, four GPUs, full load), tag fk: n, wd (the rule), wo1 (the exact table), wk<variant>.
# usage: bash v31_info.sh SEED...
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
seeds="$*"
source run_state/backstop_slot/v14h.sh none >/dev/null      # defines BASE, RINGOPT, go(), cond(); no step of v14h.sh runs
T=$R/oracle
orc() {      # orc TAG SEED CODE TABLE
  local common="--seed $((81000 + 1000 * $2)) --ring-offset $((50 * ($2 - 1))) $BASE --weak-fraction 0.25 $RINGOPT"
  go ${1}${2}r32$3 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512+yield+oracle1 --oracle-candidates $T/$4.json" "c16:rescue_value:backstop_units"
}
export PERIODS=4000
for seed in $seeds; do
  ( export SEEDS=$seed; cond fk 16 "32" "--weak-fraction 0.25 $RINGOPT" "n wd" )
  orc fk $seed wo1 fa$seed
  for v in r90 r75 r50 r25 r00 f05 f10 f25 f50 m; do orc fk $seed wk$v fa${seed}_$v; done
done
echo V31_INFO_DONE
