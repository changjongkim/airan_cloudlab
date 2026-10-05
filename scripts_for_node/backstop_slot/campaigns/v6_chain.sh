#!/usr/bin/env bash
# v6: v5 (correlated two-UE MU-MIMO weak cells, both receivers 20 LDPC iterations, NVlabs nrx_rt)
# with the fast NeuralRx path (2.4 ms alone instead of 3.1 ms).  Rescue deadline 11.5 ms.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=11.5
for spec in "$@"; do
  case $spec in
    16) CELLS=16 TAG=x KFAIL=20 NRX_BOUND=4.2 TABLE="128:4.4,512:4.4,1024:4.6" GATING="conv=128+yield" SHARES="30 50 70 100" RATES="16 32" bash run_state/backstop_slot/mu_main.sh ;;
    24) CELLS=24 TAG=x KFAIL=20 NRX_BOUND=4.1 TABLE="128:5.0,512:4.6,1024:4.8" GATING="conv=128+yield" SHARES="30 50 70" RATES="8 16" bash run_state/backstop_slot/mu_main.sh ;;
    16v) # variants of Antiphase at 16 cells (references: tag x)
      ( export CELLS=16 KFAIL=20 NRX_BOUND=4.2 TABLE="128:4.4,512:4.4,1024:4.6" REFS=" " SHARES=" " RATES="16 32"
      TAG=ya GATING="conv=128,512+yield" bash run_state/backstop_slot/mu_main.sh
      TAG=yb GATING="conv=128+yield" BUDGET="--conv-budget alone=2.6/margin=0.2/512:1.0,1024:1.0" bash run_state/backstop_slot/mu_main.sh
      TAG=yc GATING="conv=128+yield" PIECE=2.0 bash run_state/backstop_slot/mu_main.sh
      TAG=yd GATING="conv=128,512,1024+yield" bash run_state/backstop_slot/mu_main.sh ) ;;
  esac
done
echo V6_CHAIN_DONE
