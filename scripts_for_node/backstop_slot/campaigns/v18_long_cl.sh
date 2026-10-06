#!/usr/bin/env bash
# v18: runs of 100 s with closed-loop link adaptation at the 10% target (tag lga), the condition in which the
# neural receivers run for most of the time and the room of the layer-1 deadline is smallest.  For the layer-1
# latency and the TBs past the deadline in steady state at three levels: conventional receiver alone (x), with
# the neural receiver (n), with AI under each policy.  Every run is a run of la_cl2.sh with PERIODS=40000.
#   usage: bash v18_long_cl.sh      (env: SEEDS, POLS, TARGET, TAG)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
PERIODS=${PERIODS:-40000} bash run_state/backstop_slot/la_cl2.sh ${TAG:-lga} ${TARGET:-0.10} "${SEEDS:-1 2 3}" "${POLS:-x n wm s10 p30 p70 p100}"
echo V18_LONG_CL_DONE
