#!/usr/bin/env bash
# v14 headline, part 1 (ring 512, seeds from SEEDS): smoke run of the several kinds of AI work,
# steady full load, load alternating every 2 s.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export YYR_EST=$PWD/results/backstop_slot/yyr_estimator_fixed_j59210955.json
export YYP_EST=$PWD/results/backstop_slot/yyr_estimator_lowprio_j59210955.json
for step in "$@"; do
  case $step in
    smoke) bash run_state/backstop_slot/v15.sh smoke ;;
    full)  POLS="${FULL_POLS:-n wr3 wn wr1 vf s10 s30 p30 p50 p70 p100 yyr yyp}" bash run_state/backstop_slot/v14c.sh full ;;
    alt)   POLS="${ALT_POLS:-n wr3 s10 s30 p30 p50 p70 d10x50l0 e30x70l0 yyr yyp}" bash run_state/backstop_slot/v14c.sh alt ;;
    steps) POLS="${STEPS_POLS:-n wr3 s10 p30 p50 p70 d10x30x50l0 e30x50x70l0 yyr yyp}" bash run_state/backstop_slot/v14c.sh steps ;;
    chat|mix|mixnf) bash run_state/backstop_slot/v15.sh $step ;;
  esac
done
echo V14_HEADLINE_DONE
