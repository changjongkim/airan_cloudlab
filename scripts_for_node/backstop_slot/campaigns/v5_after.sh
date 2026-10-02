#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_eq.py --dataset $SOFTWALL_ROOT/run_state/backstop_slot/dataset_v5 \
  --output $SLOT_RESULTS/raw/eq_check_v5_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|rror|Traceback' | tail -3
bash run_state/backstop_slot/v5_tune.sh ya yb yc yd
echo V5_AFTER_DONE
