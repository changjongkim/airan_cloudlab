#!/usr/bin/env bash
# v7, 16 cells: smaller fixed shares, to see whether a low share reaches Antiphase's protection.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
export CELLS=16 EXTRA="--weak-fraction 0.25" KFAIL=20 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" D2=11.5 REFS=" "
TAG=w GATING="conv=128+yield" SHARES="10 20" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
echo V7_LOW_DONE
