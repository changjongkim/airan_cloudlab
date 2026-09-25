#!/usr/bin/env bash

# Frozen Confirm112 arm order.  Every arm restarts MPS and requalifies the
# conventional path through run_trace_baseline_arm.sh.

set -euo pipefail

cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab

run_arm() {
    local name=$1 system=$2 payload_seed=$3 channel_seed=$4 prefix=$5
    echo "confirm112 arm begin: $name ($system)"
    SOFTWALL_GATE_PREFIX="$prefix" \
    SOFTWALL_GATE_ITERATIONS=340 \
    SOFTWALL_PAYLOAD_SEED="$payload_seed" \
    SOFTWALL_CHANNEL_SEED_BASE="$channel_seed" \
    SOFTWALL_AI_MODEL=Qwen/Qwen2.5-1.5B \
    SOFTWALL_AI_BOUND_MAP=16:35,32:35,64:35,128:40,256:65,512:75 \
    SOFTWALL_TRACE_PATH=/softwall_burstgpt/softwall_burst60_prefill_trace.json \
    SOFTWALL_TRACE_SHA256=da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d \
        bash scripts_for_node/softwall_same_gpu/run_trace_baseline_arm.sh "$system"
    echo "confirm112 arm end: $name"
}

run_arm static_s1 static 20800051 20801000 c112_static_s1_j58815435
run_arm wc_s1_a work_conserving 20800051 20801000 c112_wc_s1a_j58815435
run_arm sw_s1_a softwall 20800051 20801000 c112_sw_s1a_j58815435
run_arm sw_s1_b softwall 20800051 20801000 c112_sw_s1b_j58815435
run_arm wc_s1_b work_conserving 20800051 20801000 c112_wc_s1b_j58815435

run_arm static_s2 static 20900132 21520000 c112_static_s2_j58815435
run_arm sw_s2_a softwall 20900132 21520000 c112_sw_s2a_j58815435
run_arm wc_s2_a work_conserving 20900132 21520000 c112_wc_s2a_j58815435
run_arm wc_s2_b work_conserving 20900132 21520000 c112_wc_s2b_j58815435
run_arm sw_s2_b softwall 20900132 21520000 c112_sw_s2b_j58815435

echo "confirm112 all frozen arms completed"
