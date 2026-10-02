#!/usr/bin/env bash
# Build the pre-generated uplink slot pools (weak rank-1 and strong rank-2).
set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot/slot_env.sh
require_allocation
mkdir -p "$SLOT_DATA"
weak=${SLOT_WEAK_COUNT:-1024}
strong=${SLOT_STRONG_COUNT:-512}
shifter_slot env CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/softwall_runtime/sionna_deps:/backstop_slot:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    python3 /backstop_slot/build_ul_dataset.py --profile weak_rank1 --count "$weak" \
    --payload-seed 71001 --channel-seed 72001 --out-dir "$SLOT_DATA" > "$SLOT_DATA/weak.log" 2>&1 &
weak_pid=$!
shifter_slot env CUDA_VISIBLE_DEVICES=1 \
    PYTHONPATH=/softwall_runtime/sionna_deps:/backstop_slot:/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
    python3 /backstop_slot/build_ul_dataset.py --profile strong_rank2 --count "$strong" \
    --payload-seed 71002 --channel-seed 72002 --out-dir "$SLOT_DATA" > "$SLOT_DATA/strong.log" 2>&1 &
strong_pid=$!
wait "$weak_pid"
wait "$strong_pid"
tail -n 2 "$SLOT_DATA/weak.log" "$SLOT_DATA/strong.log"
