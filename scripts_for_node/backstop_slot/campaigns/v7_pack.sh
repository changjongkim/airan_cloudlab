#!/usr/bin/env bash
# v7 with AI packing: requests fill the lowest-numbered GPUs first (controller4.py), so some
# GPUs keep their NeuralRx lane and conventional receivers free of AI.  References: tag w.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
export CELLS=16 KFAIL=20 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" D2=11.5 REFS=" " CONTROLLER=controller4.py
export EXTRA="--weak-fraction 0.25 --ai-extra $PWD/run_state/backstop_slot/configs/pack.json"
TAG=pw GATING="conv=128+yield" SHARES="30 50" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
echo V7_PACK_DONE
