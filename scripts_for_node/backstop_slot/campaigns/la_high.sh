#!/usr/bin/env bash
# Link adaptation on the second channel (two users with highly correlated channels): three streams on GPUs 0-2.
#   usage: bash la_high.sh [CHANNEL [ESNO,... [SLOTS]]]
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
channel=${1:-DoubleTDLhigh}; points=${2:-12,14,16,18,20,22,24}; slots=${3:-150}
L=run_state/backstop_slot/logs
GPU=0 MCS="10 13 16" bash run_state/backstop_slot/la3.sh $channel $points $slots > $L/la_${channel}_g0.log 2>&1 &
GPU=1 MCS="14 11" bash run_state/backstop_slot/la3.sh $channel $points $slots > $L/la_${channel}_g1.log 2>&1 &
GPU=2 MCS="15 12" bash run_state/backstop_slot/la3.sh $channel $points $slots > $L/la_${channel}_g2.log 2>&1 &
wait
echo LA_HIGH_DONE
