#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
bash run_state/backstop_slot/v3_contexts.sh 24 6.5 2.8 "128:3.5,512:3.9,1024:4.1" "conv=128,512,1024" 16 32
bash run_state/backstop_slot/v3_ablation.sh 24 32
echo CHAIN2_DONE
