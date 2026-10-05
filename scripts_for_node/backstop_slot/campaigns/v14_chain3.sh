#!/usr/bin/env bash
# v14/v15 steps with the longer slot pool: v15 steps (several kinds of AI work) and v14c steps.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for step in "$@"; do
  case $step in
    bench|smoke|chat|mix|mixnf) bash run_state/backstop_slot/v15.sh $step ;;
    *) bash run_state/backstop_slot/v14c.sh $step ;;
  esac
done
echo V14_CHAIN3_DONE
