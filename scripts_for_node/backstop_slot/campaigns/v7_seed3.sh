#!/usr/bin/env bash
# v7, 16 cells, rescue deadline 11.5 ms: a third seed for the main comparison.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
CELLS=16 TAG=w SEEDS=3 EXTRA="--weak-fraction 0.25" KFAIL=20 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" D2=11.5 GATING="conv=128+yield" SHARES="10 20 30 50 70" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
echo V7_SEED3_DONE
