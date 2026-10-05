#!/usr/bin/env bash
# After the main v6 runs: 24-cell variants of Antiphase, 32-cell receiver times, then the
# offline receiver-regime checks (one GPU, nothing else running).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20
( export CELLS=24 KFAIL=20 NRX_BOUND=4.1 TABLE="128:5.0,512:4.6,1024:4.8" REFS=" " SHARES=" " RATES="8 16" D2=11.5
  TAG=za GATING="conv=128,512+yield" bash run_state/backstop_slot/mu_main.sh
  TAG=zb GATING="conv=128+yield" BUDGET="--conv-budget alone=3.2/margin=0.2/512:1.0,1024:1.0" bash run_state/backstop_slot/mu_main.sh )
CELLS=32 TAG=v6l bash run_state/backstop_slot/mu_cal.sh times
bash run_state/backstop_slot/nv/regime_check.sh large med
echo V6_MORE_DONE
