#!/usr/bin/env bash
# Closed-loop link adaptation, campaigns of 2026-10-04: 10% and 1% target of TBs needing a retransmission.
# Feedback after 6 periods (15 ms): every neural receiver result is in before the outer loop reads it.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DELAY=6
bash run_state/backstop_slot/la_cl.sh laa 0.10 "1 2" "x n wm s10 p30 p70 p100"
bash run_state/backstop_slot/la_cl.sh lab 0.01 "1 2" "x n wm s10 p30 p70 p100"
echo LA_CL_CHAIN_DONE
