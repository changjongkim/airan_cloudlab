#!/usr/bin/env bash
# Closed-loop link adaptation, fourth set (2026-10-04): seed 3 of two GPUs at the 1% target; one GPU (four cells,
# one two-user cell) at the 1% target; two GPUs at the 3% target; eight two-user cells of sixteen at the 3% target.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl2.sh
P="x n wm s10 p30 p70 p100"
CELLS=8 EXTRA="--gpus 2" bash $C lad 0.01 "3" "$P"
CELLS=4 EXTRA="--gpus 1" bash $C lag 0.01 "1 2" "$P"
CELLS=8 EXTRA="--gpus 2" bash $C lah 0.03 "1 2" "$P"
WEAK=0.5 bash $C lai 0.03 "1 2" "$P"
echo LA_CL_CHAIN4_DONE
