#!/usr/bin/env bash

set -euo pipefail

root=/pscratch/sd/s/sgkim/kcj/airan_cloudlab
runner="$root/scripts_for_node/softwall_same_gpu/run_confirm105_nrx_count_interleaved.sh"
job=${SLURM_JOB_ID:?}

run_arm() {
    local name=$1 payload=$2 channel=$3
    env \
        SOFTWALL_GATE_PREFIX="confirm105_${name}_job${job}" \
        SOFTWALL_PAYLOAD_SEED="$payload" \
        SOFTWALL_CHANNEL_SEED_BASE="$channel" \
        SOFTWALL_GATE_ITERATIONS=2000 \
        SOFTWALL_CORRELATED_FAILURE_EVERY=0 \
        SOFTWALL_NRX_BOUND_MS=45 \
        SOFTWALL_CONV_BOUND_MS=12 \
        SOFTWALL_GC_MODE=off \
        bash "$runner"
}

run_arm run1 21005001 21030000
run_arm run2 21005002 21040000

echo "confirm105 two interleaved matched runs completed"
