#!/usr/bin/env bash
# v6 = v5 with the fast NeuralRx path.  Receiver times at 16 and 24 cells.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20
CELLS=16 TAG=v6s bash run_state/backstop_slot/mu_cal.sh times
CELLS=24 TAG=v6 bash run_state/backstop_slot/mu_cal.sh times
echo V6_CAL_DONE
