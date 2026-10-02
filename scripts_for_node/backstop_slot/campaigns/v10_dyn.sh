#!/usr/bin/env bash
# v10: Our Scheme against a dynamic-share baseline when the load changes within a run.
# 16 cells, nrx_large rescue (v7 setting), 20-second runs, 2-second phases alternating between
# full load and every cell active with probability 0.5.  The baseline keeps one pre-loaded AI
# worker per share on every GPU and activates the low share in full-load phases and the high
# share otherwise, seeing the load with no delay (lag 0) or one second late (lag 400 periods).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export DATA=$R/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt KFAIL=20 D2=11.5
export NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="conv=128+yield" PERIODS=${PERIODS:-8000} CELLS=16 CONTROLLER=controller4.py
export EXTRA="--weak-fraction 0.25 --activity-prob ${PROB:-0.5} --activity-mode phased --activity-phase ${PHASE:-800}"
TAG=${TAG:-dy} SHARES="${SHARES:-10 30}" RATES="${RATES:-32}" DYNS="${DYNS:-10,30:0 10,30:400 10,50:0 10,50:400 20,50:0}" bash run_state/backstop_slot/mu_main.sh
echo V10_DYN_DONE
