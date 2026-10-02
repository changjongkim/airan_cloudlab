#!/usr/bin/env bash
# Steps 6, 5, 4 back to back on one allocation.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
bash run_state/backstop_slot/v3_nrx.sh 2 3
bash run_state/backstop_slot/v3_classes.sh
bash run_state/backstop_slot/v3_partial.sh b50 u50
echo CHAIN_DONE
