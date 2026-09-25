#!/usr/bin/env bash

set -euo pipefail

root=/pscratch/sd/s/sgkim/kcj/airan_cloudlab
runner="$root/scripts_for_node/softwall_same_gpu/run_confirm103_nrx_count_probe.sh"
job=${SLURM_JOB_ID:?}

run_arm() {
    local name=$1 count=$2 payload=$3 channel=$4
    env \
        SOFTWALL_GATE_PREFIX="confirm103_${name}_job${job}" \
        SOFTWALL_MAX_NRX_ADMISSIONS="$count" \
        SOFTWALL_PAYLOAD_SEED="$payload" \
        SOFTWALL_CHANNEL_SEED_BASE="$channel" \
        SOFTWALL_GATE_ITERATIONS=300 \
        SOFTWALL_CORRELATED_FAILURE_EVERY=0 \
        SOFTWALL_NRX_BOUND_MS=45 \
        SOFTWALL_CONV_BOUND_MS=12 \
        SOFTWALL_GC_MODE=off \
        bash "$runner"
}

run_arm a_one 1 21003001 21004000
run_arm b_two 2 21003001 21004000
run_arm c_two 2 21003002 21005000
run_arm d_one 1 21003002 21005000

echo "confirm103 NeuralRx-count ABBA completed"
