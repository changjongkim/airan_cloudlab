#!/usr/bin/env bash
# v5, 16 cells, AI load 32: variants of Our Scheme only (references: tag x of v5_chain.sh).
#   ya  shorter AI pieces (2 ms)            yb  only 128-token units may overlap conventional decode
#   yc  128 always, 512/1024 within a conventional delay budget
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=11.5 CELLS=16 KFAIL=20 NRX_BOUND=5.2 TABLE="128:5.4,512:6.3,1024:6.3" REFS=" " SHARES=" " RATES="${RATES:-32}"
for v in "$@"; do
  case $v in
    ya) TAG=ya PIECE=2.0 GATING="conv=128,512+yield" bash run_state/backstop_slot/mu_main.sh ;;
    yb) TAG=yb GATING="conv=128+yield" bash run_state/backstop_slot/mu_main.sh ;;
    yc) TAG=yc GATING="conv=128+yield" BUDGET="--conv-budget alone=3.0/margin=0.2/512:1.0,1024:1.0" bash run_state/backstop_slot/mu_main.sh ;;
    yd) TAG=yd PIECE=2.0 GATING="conv=128+yield" BUDGET="--conv-budget alone=3.0/margin=0.2/512:1.0,1024:1.0" bash run_state/backstop_slot/mu_main.sh ;;
  esac
done
echo V5_TUNE_DONE
