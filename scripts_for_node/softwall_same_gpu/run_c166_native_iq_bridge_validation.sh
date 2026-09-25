#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

build_dir="$SOFTWALL_ROOT/run_state/c166_native_iq_build"
fixture_dir="$SOFTWALL_ROOT/results/softwall_multigpu/c166_native_fixture_seed20359400"
output="$SOFTWALL_ROOT/results/softwall_multigpu/c166_native_iq_bridge_validation.json"
module_path="$(find "$build_dir" -maxdepth 1 -type f -name '_softwall_native_iq*.so' -print -quit)"
if [[ -z "$module_path" ]]; then
    echo "build native IQ bridge first" >&2
    exit 1
fi

shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$SOFTWALL_SCRIPTS:/softwall" \
    --volume="$build_dir:/softwall_native_build" \
    --volume="$fixture_dir:/fixture" \
    --volume="$(dirname "$output"):/results" \
    --env=PYTHONPATH=/softwall_native_build \
    --env=CUDA_VISIBLE_DEVICES=0 \
    python3 /softwall/validate_c166_native_iq_bridge.py \
        --fixture /fixture \
        --module "/softwall_native_build/$(basename "$module_path")" \
        --output "/results/$(basename "$output")"
