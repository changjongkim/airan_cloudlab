#!/usr/bin/env bash
# v14h, third part: more seeds of 20 cells (bound 7.6 ms), 32 cells and 48 cells with the final rule (wd) and the baselines.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for seed in ${SEEDS_C:-3 4}; do
  SEEDS="$seed" POLS="n wd s10 p30 p70" bash run_state/backstop_slot/v14h.sh dense76 cells
done
echo V14H_CHAIN_C_DONE
