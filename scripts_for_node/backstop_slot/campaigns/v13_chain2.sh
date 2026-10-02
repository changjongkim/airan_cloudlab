#!/usr/bin/env bash
# v13, second allocation: third seed of the AI-load sweep, then the sweeps over NeuralRx demand,
# cells, GPUs, cell bursts, AI bursts, the 20-cell case, and a third seed of the changing-load runs.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
V="bash run_state/backstop_slot/v13b.sh"
for step in "$@"; do
  case $step in
    seed3)   SEEDS=3 $V aiload ;;
    nrx)     $V nrx ;;
    cells)   $V cells ;;
    gpus)    $V gpus ;;
    burst)   BURSTS="ba:8 bc:800" $V burst ;;
    aiburst) $V aiburst ;;
    dense)   VF_TABLE="128:8.1,512:8.6,1024:8.9" VF_CONV=128 NRX_BOUND20=7.8 $V dense ;;
    change3) SEEDS=3 PHASES="hb:800" $V phase; SEEDS=3 $V steps ;;
  esac
done
echo V13_CHAIN2_DONE
