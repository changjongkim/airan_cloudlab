#!/usr/bin/env bash
# Closed-loop link adaptation, runs left after job 59345188 ended (2026-10-05): the missing run of one GPU; seed 3 of
# eight two-user cells at the 1% target; the outer loop with a step of 0.5 levels per failed TB; the 3% target with
# eight two-user cells and with two GPUs.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl2.sh
P="x n wm s10 p30 p70 p100"
CELLS=4 EXTRA="--gpus 1" bash $C lag 0.01 "2" "p100"
WEAK=0.5 bash $C laf 0.01 "3" "$P"
WEAK=0.5 DOWN=0.5 bash $C lak 0.01 "1" "x n wm p30 p70 p100"
DOWN=0.5 bash $C lal 0.10 "1" "x n wm p70 p100"
WEAK=0.5 bash $C lai 0.03 "1 2" "$P"
CELLS=8 EXTRA="--gpus 2" bash $C lah 0.03 "1 2" "$P"
echo LA_CL_CHAIN6_DONE
