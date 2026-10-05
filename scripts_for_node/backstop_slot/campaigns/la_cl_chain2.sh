#!/usr/bin/env bash
# Closed-loop link adaptation, second set (2026-10-04): 3% target; no recovery path with low-priority AI;
# the rule with AI allowed next to a neural receiver while 3 / 1 neural receivers are free.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl2.sh
bash $C lac 0.03 "1 2" "x n wm s10 p30 p70 p100 xp100"
bash $C laa 0.10 "1 2" "xp100 wr3 wr1"
bash $C lab 0.01 "1 2" "xp100 wr3 wr1"
echo LA_CL_CHAIN2_DONE
