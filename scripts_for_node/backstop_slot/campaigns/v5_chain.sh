#!/usr/bin/env bash
# v5: weak cells carry a correlated two-UE MU-MIMO pair (DoubleTDL high correlation, 13-16 dB),
# both receivers use 20 LDPC iterations, NeuralRx = NVlabs nrx_rt.  Rescue deadline 11.5 ms.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=11.5
CELLS=16 TAG=x KFAIL=20 NRX_BOUND=5.2 TABLE="128:5.4,512:6.3,1024:6.3" GATING="conv=128,512+yield" SHARES="50 70 100" RATES="16 32" \
  bash run_state/backstop_slot/mu_main.sh
CELLS=24 TAG=x KFAIL=8 NRX_BOUND=5.6 TABLE="128:6.4,512:6.6,1024:6.7" GATING="conv=128+yield" SHARES="30 50 70" RATES="8 16" \
  bash run_state/backstop_slot/mu_main.sh
echo V5_CHAIN_DONE
