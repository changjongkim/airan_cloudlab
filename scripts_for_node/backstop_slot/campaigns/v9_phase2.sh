#!/usr/bin/env bash
# v9: a second changing-load pattern (1-second phases, full load / probability 0.25) and a third
# seed of the first pattern.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export DATA=$R/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt KFAIL=20 D2=11.5
export NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="conv=128+yield" PERIODS=8000 CELLS=16 SHARES="10 20 30 50" RATES="32"
TAG=pi EXTRA="--weak-fraction 0.25 --activity-prob 0.25 --activity-mode phased --activity-phase 400" bash run_state/backstop_slot/mu_main.sh
TAG=ph SEEDS=3 EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode phased --activity-phase 800" bash run_state/backstop_slot/mu_main.sh
echo V9_PHASE2_DONE
