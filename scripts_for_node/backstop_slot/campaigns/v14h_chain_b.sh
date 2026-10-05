#!/usr/bin/env bash
# v14h, second part: alternation every 0.2 s and 5 s (16 cells), cell bursts (32 and 16 cells).
# wd = the two smaller unit classes next to the conventional receiver for any number of active cells
# (equal to wm with 16 cells, where at most four cells of a GPU are active).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export YYR_EST=$PWD/results/backstop_slot/yyr_estimator_fixed_j59210955.json
export YYP_EST=$PWD/results/backstop_slot/yyr_estimator_lowprio_j59210955.json
POLS="n wm s10 p30 p70 d10x50l0 e30x70l0 yyr yyp" bash run_state/backstop_slot/v14h.sh phase
POLS="n wm wd s10 p30 p70 e30x70l0" bash run_state/backstop_slot/v14h.sh burst
echo V14H_CHAIN_B_DONE
