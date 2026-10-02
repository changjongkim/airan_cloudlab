#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu_fast.py --dataset $SOFTWALL_ROOT/run_state/backstop_slot/dataset_v5 \
  --engine /softwall_runtime/engines/nv/nrx_rt_273prb_2ue.trt --output $SLOT_RESULTS/raw/mu_fast_check_j${SLURM_JOB_ID}.json "$@" 2>&1 | grep -E '^\{|rror|Traceback|  File|line ' | tail -12
