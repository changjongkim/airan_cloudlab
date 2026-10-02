#!/usr/bin/env bash
# v9: the v7 setting (nrx_large rescue, low-correlation MU-MIMO weak cells, both receivers 20 LDPC
# iterations, rescue deadline 11.5 ms) when cells do not carry a TB in every uplink slot.
#   b50  16 cells, each cell active in a slot with probability 0.5, independently
#   u50  16 cells, probability 0.5 in busy runs of mean 8 periods
#   c32  32 cells, probability 0.5, independently
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export DATA=$R/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt KFAIL=20 D2=${D2:-11.5}
export NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="${GATING:-conv=128+yield}" SHARES="${SHARES:-10 20 30 50 70}" RATES="${RATES:-16 32}"
for step in "$@"; do
  case $step in
    b50) CELLS=16 TAG=pa EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode bernoulli" bash run_state/backstop_slot/mu_main.sh ;;
    u50) CELLS=16 TAG=pb EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode bursty --activity-burst 8" bash run_state/backstop_slot/mu_main.sh ;;
    c32) CELLS=32 TAG=pc EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode bernoulli" bash run_state/backstop_slot/mu_main.sh ;;
  esac
done
echo V9_CHAIN_DONE
