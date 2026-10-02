#!/usr/bin/env bash
# v13 calibration: receiver times next to AI that runs at a lower MPS priority.
# Every NeuralRx runs to the end (recovery deadline 41.5 ms).  AI of one unit size runs
# continuously: 70% share (the v7 setting), 70% share + low priority, no cap + low priority.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
cells=${CELLS:-16}
base="--seed 82000 --weak-profile nv_mu2 --dataset $R/dataset_mix --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --weak-fraction 0.25 ${EXTRA:-} --controller controller4.py --rescue-deadline-ms 41.5 --nrx-bound-ms 4.0 --nrx-max-cb-fail 9 --ai-admission 0 --ai-chunks 128,512,1024"
go() { SLOT_TAG=$1 SLOT_PERIODS=2400 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="c$cells:rescue_value:$4" $M 2>&1 | grep -E "^===|FAILED"; }
go ${TAG:-k}n 4 "$base" none
for c in ${CHUNKS:-128 512 1024}; do
  for v in ${VARIANTS:-s70 p70 p100}; do
    pct=${v#?}; prio=""; [ "${v:0:1}" = p ] && prio="--ai-mps-priority 1"
    go ${TAG:-k}${v}f$c 24 "$base --static-mps-pct $pct --ai-force-chunk $c $prio" static
  done
done
echo V13_CAL_DONE
