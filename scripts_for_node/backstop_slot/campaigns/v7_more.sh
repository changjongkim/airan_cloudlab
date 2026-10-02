#!/usr/bin/env bash
# v7 follow-ups (nrx_large rescue, 16 cells, a quarter of the cells weak).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
export CELLS=16 EXTRA="--weak-fraction 0.25" KFAIL=20 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3"
for step in "$@"; do
  case $step in
    variants)  # Our Scheme only, rescue deadline 11.5 ms (references: tag w)
      ( export D2=11.5 REFS=" " SHARES=" " RATES="16 32"
        TAG=wa GATING="conv=128,512+yield" bash run_state/backstop_slot/mu_main.sh
        TAG=wb GATING="conv=128+yield" BUDGET="--conv-budget alone=3.3/margin=0.2/512:1.0,1024:1.0" bash run_state/backstop_slot/mu_main.sh ) ;;
    d16)       # rescue deadline 16.5 ms
      D2=16.5 TAG=q GATING="conv=128+yield" SHARES="30 50 70" RATES="16 32" bash run_state/backstop_slot/mu_main.sh ;;
    c24)       # 24 cells (six weak cells)
      CELLS=24 TAG=v7 bash run_state/backstop_slot/mu_cal.sh times ;;
  esac
done
echo V7_MORE_DONE
