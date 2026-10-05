#!/usr/bin/env bash
# Kinds of AI work: every policy in one job (three seeds of the mix, two of chat only).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
SEEDS="1 2 3" POLS="n wm wn s10 s30 p30 p70 p100" bash run_state/backstop_slot/v15.sh mix
SEEDS="1 2" POLS="n wm s10 s30 p30 p70 p100" bash run_state/backstop_slot/v15.sh chat
echo V15_SAMEJOB_DONE
