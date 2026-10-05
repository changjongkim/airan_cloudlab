#!/usr/bin/env bash
# Tight rescue deadline (6.5 ms), 16 cells, with NeuralRx admitted by its p99 run time (3.2 ms)
# instead of p99.9 (4.2 ms).  AI gating of Antiphase keeps the p99.9 co-run bounds.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=6.5
CELLS=16 TAG=u KFAIL=20 NRX_BOUND=3.2 TABLE="128:4.4,512:4.4,1024:4.6" GATING="conv=128+yield" SHARES="30 50 70" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
echo V6_P99_DONE
