#!/usr/bin/env bash
# Fill in the 16-cell v5 curve: fixed 30% share, and the protective variants at AI load 16.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=11.5
CELLS=16 TAG=x KFAIL=20 NRX_BOUND=5.2 TABLE="128:5.4,512:6.3,1024:6.3" GATING="conv=128,512+yield" SHARES="30" RATES="16 32" REFS=" " bash run_state/backstop_slot/mu_main.sh
RATES=16 bash run_state/backstop_slot/v5_tune.sh yb yd
echo V5_AFTER3_DONE
