#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu_stages.py --dataset $SOFTWALL_ROOT/run_state/backstop_slot/dataset_v5 \
  --engine /softwall_runtime/engines/nv/nrx_rt_273prb_2ue.trt "$@" 2>&1 | grep -E '^\{|factor|rror|Traceback|  File|line ' | tail -12
