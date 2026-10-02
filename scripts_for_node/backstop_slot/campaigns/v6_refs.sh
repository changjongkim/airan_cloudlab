#!/usr/bin/env bash
# No-AI references of the 24-cell v6 runs (the chain scripts left REFS empty after their variant block).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 CELLS=24 KFAIL=20 NRX_BOUND=4.1 TABLE="128:5.0,512:4.6,1024:4.8" REFS="n m" RATES=" " SHARES=" "
D2=11.5 TAG=x bash run_state/backstop_slot/mu_main.sh
D2=6.5 TAG=t bash run_state/backstop_slot/mu_main.sh
echo V6_REFS_DONE
