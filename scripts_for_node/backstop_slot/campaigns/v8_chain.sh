#!/usr/bin/env bash
# v8: the v7 setting (nrx_large rescue, 16 cells, a quarter of the cells weak, both receivers
# 20 LDPC iterations, rescue deadline 11.5 ms) on more channels and with extra checks.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=$PWD/run_state/backstop_slot
export ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt CELLS=16 EXTRA="--weak-fraction 0.25"
export KFAIL=20 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" D2=11.5
main() {   # tag dataset
  DATA=$R/$2 TAG=${1}c bash run_state/backstop_slot/mu_cal.sh times_noai
  DATA=$R/$2 TAG=$1 GATING="conv=128+yield" SHARES="10 20 30 50 70" RATES="16 32" bash run_state/backstop_slot/mu_main.sh
}
for step in "$@"; do
  case $step in
    high) main e dataset_v5 ;;
    umi)  main f dataset_v8umi ;;
    corun128)  # low correlation: 128-token AI units may run next to NeuralRx (p99 co-run bound)
      ( export DATA=$R/dataset_nv REFS=" " SHARES=" " RATES="16 32"
        TAG=wc TABLE="128:7.8,512:9.8,1024:10.3" GATING="conv=128+yield" bash run_state/backstop_slot/mu_main.sh ) ;;
    long)      # low correlation, 40 s runs, one seed
      DATA=$R/dataset_nv TAG=g SEEDS=1 PERIODS=16000 GATING="conv=128+yield" SHARES="10 20 30" RATES="32" bash run_state/backstop_slot/mu_main.sh ;;
    primary)   # every slot of the weak cells goes to NeuralRx (no conventional-first rescue)
      ( export DATA=$R/dataset_nv REFS="n" SHARES=" " RATES=" " NRXFLAGS="--nrx-primary-cells 4"
        TAG=pr bash run_state/backstop_slot/mu_main.sh ) ;;
  esac
done
echo V8_CHAIN_DONE
