#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
TAG=dy PROB=0.5 PHASE=800 bash run_state/backstop_slot/v10_dyn.sh
TAG=dz PROB=0.25 PHASE=400 DYNS="10,30:0 10,30:400 10,50:0 10,50:400" bash run_state/backstop_slot/v10_dyn.sh
echo V10_CHAIN_DONE
