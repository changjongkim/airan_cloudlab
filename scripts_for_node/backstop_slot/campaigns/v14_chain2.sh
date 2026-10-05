#!/usr/bin/env bash
# v14, after the rule comparison: unit bounds and a smoke run of the multi-model AI worker, the
# longer slot pool, the calibration of the YinYangRAN-style estimator.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for step in "$@"; do
  case $step in
    bench|smoke|chat|mix|mixnf) bash run_state/backstop_slot/v15.sh $step ;;
    *) bash run_state/backstop_slot/v14b.sh $step ;;
  esac
done
echo V14_CHAIN2_DONE
