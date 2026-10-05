#!/usr/bin/env bash
# v14g: re-measurement with the final rule of the conditions that had only v13 results.
#   a   AI load, AI bursts, one GPU (seeds 3-5), 32 and 48 cells          (about 85 min)
#   b   alternation every 0.2 s and 5 s, cell bursts                       (about 95 min)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export YYR_EST=$PWD/results/backstop_slot/yyr_estimator_fixed_j59210955.json
export YYP_EST=$PWD/results/backstop_slot/yyr_estimator_lowprio_j59210955.json
for part in "$@"; do
  case $part in
    a) bash run_state/backstop_slot/v14g.sh aiload aiburst
       SEEDS="3 4 5" POLS="n wm p30 p70 s10" bash run_state/backstop_slot/v14g.sh gpu1
       bash run_state/backstop_slot/v14g.sh cells ;;
    b) bash run_state/backstop_slot/v14g.sh phase burst ;;
  esac
done
echo V14G_CHAIN_DONE
