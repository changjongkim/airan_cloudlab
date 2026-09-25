#!/usr/bin/env bash

# Frozen-arm driver for Confirm125. Arm parameters are duplicated in the
# prospective protocol so an interrupted campaign can be audited exactly.

set -euo pipefail

cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab

run_arm() {
    local label=$1
    local policy=$2
    local payload_seed0=$3
    local payload_seed1=$4
    local channel_seed0=$5
    local channel_seed1=$6

    echo "CONFIRM125_START ${label} ${policy} $(date -u +%FT%TZ)"
    SOFTWALL_SHARD_LABEL="$label" \
    SOFTWALL_GLOBAL_POLICY="$policy" \
    SOFTWALL_CONV_BOUND_MS=25 \
    SOFTWALL_SHARD_ITERATIONS=340 \
    SOFTWALL_SHARD_WARMUP=20 \
    SOFTWALL_SHARD_PAYLOAD_SEED0="$payload_seed0" \
    SOFTWALL_SHARD_PAYLOAD_SEED1="$payload_seed1" \
    SOFTWALL_SHARD_CHANNEL_SEED0="$channel_seed0" \
    SOFTWALL_SHARD_CHANNEL_SEED1="$channel_seed1" \
        bash scripts_for_node/softwall_same_gpu/run_global_ai_two_home_bound_arm.sh
    echo "CONFIRM125_END ${label} $(date -u +%FT%TZ)"
}

run_arm c125_s1_g1_job58819952 global           26700051 26710051 26701000 26711000
run_arm c125_s1_p1_job58819952 static_partition 26700051 26710051 26701000 26711000
run_arm c125_s1_p2_job58819952 static_partition 26700051 26710051 26701000 26711000
run_arm c125_s1_g2_job58819952 global           26700051 26710051 26701000 26711000
run_arm c125_s2_p1_job58819952 static_partition 26800061 26810061 26801000 26811000
run_arm c125_s2_g1_job58819952 global           26800061 26810061 26801000 26811000
run_arm c125_s2_g2_job58819952 global           26800061 26810061 26801000 26811000
run_arm c125_s2_p2_job58819952 static_partition 26800061 26810061 26801000 26811000

source scripts_for_node/softwall_same_gpu/common.sh
shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    python3 /softwall/analyze_confirm125_conv25_global_vs_static.py
