#!/usr/bin/env bash
# v8: 20 cells (5 per GPU: one MU-MIMO cell with nrx_large rescue, four strong cells), low correlation.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt CELLS=20 EXTRA="--weak-fraction 0.2" DATA=$R/dataset_nv
for step in "$@"; do
  case $step in
    cal)  TAG=v8t bash run_state/backstop_slot/mu_cal.sh times ;;
    main) KFAIL=20 NRX_BOUND=${NRX_BOUND:?} TABLE=${TABLE:?} D2=11.5 TAG=j GATING="conv=128+yield" SHARES="10 20 30 50" RATES="16 32" bash run_state/backstop_slot/mu_main.sh ;;
  esac
done
echo V8_C20_DONE
