#!/usr/bin/env bash
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot/slot_env.sh
cd "$SOFTWALL_ROOT"
shifter_slot python3 /backstop_slot/probe_cellgroup.py --dataset "$SLOT_DATA" --gpu 0 \
  --output "$SLOT_RESULTS/raw/cellgroup_probe_j${SLURM_JOB_ID}.json" "$@"
