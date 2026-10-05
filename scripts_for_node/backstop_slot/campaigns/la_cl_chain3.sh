#!/usr/bin/env bash
# Closed-loop link adaptation, third set (2026-10-04): two GPUs (eight cells, two two-user cells) at the 1% and
# 10% targets; eight two-user cells of sixteen at the 1% target; seed 3 of the 10% and 1% targets.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl2.sh
P="x n wm s10 p30 p70 p100"
CELLS=8 EXTRA="--gpus 2" bash $C lad 0.01 "1 2" "$P"
WEAK=0.5 bash $C laf 0.01 "1 2" "$P"
CELLS=8 EXTRA="--gpus 2" bash $C lae 0.10 "1 2" "$P"
bash $C laa 0.10 "3" "$P"
bash $C lab 0.01 "3" "$P"
echo LA_CL_CHAIN3_DONE
