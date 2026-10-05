#!/usr/bin/env bash
# Link adaptation: the MCS that la.sh skipped and a finer Es/No grid (0.5 dB) around the
# points where the error rate of the conventional receiver crosses the 10% target.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
MCS=12 bash run_state/backstop_slot/la.sh DoubleTDLlow 5,6,7 150
SUFFIX=_b MCS="12 13 14 15" bash run_state/backstop_slot/la2.sh DoubleTDLlow 4.5,5.5,6.5 150
SUFFIX=_c MCS="11" bash run_state/backstop_slot/la2.sh DoubleTDLlow 4.5 150
SUFFIX=_c MCS="16" bash run_state/backstop_slot/la2.sh DoubleTDLlow 6.5 150
SUFFIX=_d MCS="15 16" bash run_state/backstop_slot/la2.sh DoubleTDLlow 7.5 150
echo LA_MORE_DONE
