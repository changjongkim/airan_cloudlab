#!/usr/bin/env bash
# v6 with the tight rescue deadline (6.5 ms: before the next-but-one uplink slot), which the
# fast NeuralRx path (2.7 ms in a loaded run) now fits.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=6.5
for spec in "$@"; do
  case $spec in
    16) CELLS=16 TAG=t KFAIL=20 NRX_BOUND=4.2 TABLE="128:4.4,512:4.4,1024:4.6" GATING="${GATE16:-conv=128,512,1024+yield}" SHARES="30 50 70 100" RATES="16 32" bash run_state/backstop_slot/mu_main.sh ;;
    24) CELLS=24 TAG=t KFAIL=20 NRX_BOUND=4.1 TABLE="128:5.0,512:4.6,1024:4.8" GATING="${GATE24:-conv=128+yield}" SHARES="30 50 70" RATES="8 16" bash run_state/backstop_slot/mu_main.sh ;;
    16v) ( export CELLS=16 KFAIL=20 NRX_BOUND=4.2 TABLE="128:4.4,512:4.4,1024:4.6" REFS=" " SHARES=" " RATES="16 32"
      TAG=ta GATING="conv=128+yield" bash run_state/backstop_slot/mu_main.sh
      TAG=tb GATING="conv=128,512+yield" bash run_state/backstop_slot/mu_main.sh ) ;;
  esac
done
echo V6_TIGHT_DONE
