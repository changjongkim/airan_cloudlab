#!/usr/bin/env bash
# v8 follow-ups: 20 cells with the 16.5 ms rescue deadline, and a third seed on the high-correlation channel.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt KFAIL=20
CELLS=20 EXTRA="--weak-fraction 0.2" DATA=$R/dataset_nv NRX_BOUND=8.6 TABLE="128:8.7,512:10.2,1024:10.3" D2=16.5 TAG=jq GATING="conv=128+yield" SHARES="10 20 30 50" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
CELLS=16 EXTRA="--weak-fraction 0.25" DATA=$R/dataset_v5 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" D2=11.5 TAG=e SEEDS=3 GATING="conv=128+yield" SHARES="10 20 30 50 70" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
echo V8_MORE_DONE
