#!/usr/bin/env bash
# v8, UMi: only TBs with at most nine failed code blocks (both UEs) go to NeuralRx.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt CELLS=16 EXTRA="--weak-fraction 0.25"
export NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" D2=11.5
DATA=$R/dataset_v8umi TAG=f9 KFAIL=9 GATING="conv=128+yield" SHARES="30 50 70" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
echo V8_UMI9_DONE
