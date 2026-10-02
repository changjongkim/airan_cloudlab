#!/usr/bin/env bash
# Run one slot-scale configuration under MPS on all four GPUs.
#   bash run_slot.sh <config.json> <output.json>
set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot/slot_env.sh
require_allocation
config=$1
output=$2
export SOFTWALL_MPS_GPU=0,1,2,3
softwall_mps_start
trap softwall_mps_stop EXIT
shifter_slot python3 /backstop_slot/launch_slot.py --config "$config" --output "$output"
