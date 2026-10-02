#!/usr/bin/env bash
# v12: more seeds for the headline setting, and the same comparison at 32 cells.
#   seeds  16 cells, mixed channels, load alternating full / half every 2 s: seeds 3, 4, 5
#   steady 16 cells, mixed channels, full load, AI load 32: seeds 3, 4, 5
#   c32    32 cells, mixed channels, load alternating half / quarter every 2 s: seeds 1, 2
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
export ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt D2=11.5 DATA=$R/dataset_mix KFAIL=9
export NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="conv=128+yield"
for step in "$@"; do
  case $step in
    seeds)
      ( export CELLS=16 PERIODS=8000 CONTROLLER=controller4.py REFS="n" SHARES="10 30" RATES="32" DYNS="10,30:0 10,50:0 10,30:400" SEEDS="3 4 5"
        TAG=my EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode phased --activity-phase 800" bash run_state/backstop_slot/mu_main.sh ) ;;
    steady)
      ( export CELLS=16 REFS="n" SHARES="10 20 30 50" RATES="32" SEEDS="3 4 5"
        TAG=mx EXTRA="--weak-fraction 0.25" bash run_state/backstop_slot/mu_main.sh ) ;;
    c32)
      ( export CELLS=32 PERIODS=8000 CONTROLLER=controller4.py REFS="n" SHARES="10 30" RATES="32" DYNS="10,30:0 10,50:0 10,30:400" SEEDS="1 2"
        TAG=mz EXTRA="--weak-fraction 0.25 --activity-prob 0.25 --activity-high 0.5 --activity-mode phased --activity-phase 800" bash run_state/backstop_slot/mu_main.sh ) ;;
  esac
done
echo V12_CHAIN_DONE
