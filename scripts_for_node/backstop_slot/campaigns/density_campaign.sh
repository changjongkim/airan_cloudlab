#!/usr/bin/env bash
# Density phase A (radio only, no AI): how many cells one GPU can hold.
# 4 GPUs with 8-16 cells per GPU; NeuralRx lanes per GPU 1 or 2; rescue deadline 6.5/11.5/41.5 ms
# (41.5 ms lets every admitted NeuralRx finish: the reference for what can be rescued).
# Extras: weak-cell fraction 25%, and a single GPU alone (host load far below the 4-GPU runs).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
echo "cpus visible: $(nproc)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
one() {   # tag cells extra...
  local tag=$1 cells=$2; shift 2
  SLOT_TAG=$tag SLOT_PERIODS=2400 SLOT_AI_RATE=4 SLOT_EXTRA="--seed 82000 $*" \
    SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
}
for cells in 32 40 48 56 64; do
  for d2 in 6.5 11.5 41.5; do
    t=$(echo $d2 | tr -d .)
    one dnl1d$t $cells --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms $d2
    one dnl2d$t $cells --lanes-per-gpu 2 --nrx-bound-ms 4.3 --rescue-deadline-ms $d2
  done
done
for cells in 48 64; do
  for d2 in 6.5 11.5; do
    t=$(echo $d2 | tr -d .)
    one dnw25l1d$t $cells --weak-fraction 0.25 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms $d2
  done
done
for cells in 8 12 16; do
  one dng1l1d115 $cells --gpus 1 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms 11.5
  one dng1l2d115 $cells --gpus 1 --lanes-per-gpu 2 --nrx-bound-ms 4.3 --rescue-deadline-ms 11.5
done
echo DENSITY_A_DONE
