#!/usr/bin/env bash
# Link adaptation streams side by side: one la3.sh per "GPU:MCS MCS ..." argument.
#   usage: bash la_streams.sh CHANNEL ESNO,ESNO,... SLOTS "1:14 12 10" "2:15 13 11" ...
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
channel=$1; points=$2; slots=$3; shift 3
L=run_state/backstop_slot/logs
for spec in "$@"; do
  gpu=${spec%%:*}; list=${spec#*:}
  GPU=$gpu MCS="$list" bash run_state/backstop_slot/la3.sh $channel $points $slots > $L/la_${channel}_g${gpu}_$(date +%H%M).log 2>&1 &
done
wait
echo LA_STREAMS_DONE
