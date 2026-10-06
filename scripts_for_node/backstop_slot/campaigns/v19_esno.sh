#!/usr/bin/env bash
# v19: closed-loop link adaptation beyond one channel at one Es/No.  Every run is a run of la_cl3.sh; the scheme
# and its options are unchanged.
#   static14  high-correlation channel at 14 dB: four two-user cells at the 10% and 1% targets (lha, lhb), eight at 1% (lhc)
#   static20  the same at 20 dB (lja, ljb, ljc)
#   low6      low-correlation channel at 6 dB (lwa, lwb, lwc)
#   vary      the two-user cells move between 14, 16 and 20 dB every 2 s: 10% and 1% targets (lva, lvb), runs of 40 s
#   vary8     the same with eight two-user cells at the 1% target (lvc)
# usage: bash v19_esno.sh STEP...      (env: SEEDS, POLS, PHASE)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl3.sh
P=${POLS:-"x n wm s10 p30 p70 p100"}
PV=${POLS:-"x n wm s10 p30 p50 p70 p100"}
three() {   # three DATA TAGS...   (four two-user cells at 10% and 1%, eight at 1%)
  local data=$1 a=$2 b=$3 c=$4 seed
  for seed in ${SEEDS:-1 2}; do
    DATA=$data bash $C $a 0.10 "$seed" "$P"
    DATA=$data bash $C $b 0.01 "$seed" "$P"
    DATA=$data WEAK=0.5 bash $C $c 0.01 "$seed" "$P"
  done
}
for step in "$@"; do
  case $step in
    static14) three high14 lha lhb lhc ;;
    static20) three high20 lja ljb ljc ;;
    low6)     three low6 lwa lwb lwc ;;
    vary)     for seed in ${SEEDS:-1 2 3}; do
                PERIODS=16000 STATES="high14 high16 high20" bash $C lva 0.10 "$seed" "$PV"
                PERIODS=16000 STATES="high14 high16 high20" bash $C lvb 0.01 "$seed" "$PV"
              done ;;
    vary8)    for seed in ${SEEDS:-1 2 3}; do
                PERIODS=16000 STATES="high14 high16 high20" WEAK=0.5 bash $C lvc 0.01 "$seed" "$PV"
              done ;;
  esac
done
echo V19_ESNO_DONE
