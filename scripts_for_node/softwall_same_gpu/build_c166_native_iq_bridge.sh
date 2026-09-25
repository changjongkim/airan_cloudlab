#!/usr/bin/env bash

set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

source_dir="$SOFTWALL_SCRIPTS/native_fast_path"
build_dir="$SOFTWALL_ROOT/run_state/c166_native_iq_build"
mkdir -p "$build_dir"

shifter --module=gpu --image="$AERIAL_IMAGE" \
    --volume="$source_dir:/softwall_native_src" \
    --volume="$build_dir:/softwall_native_build" \
    bash -lc '
        set -euo pipefail
        cmake -S /softwall_native_src -B /softwall_native_build \
            -DCMAKE_BUILD_TYPE=Release \
            -DSOFTWALL_BUILD_CUDA_BRIDGE=ON
        cmake --build /softwall_native_build --parallel 4 \
            --target _softwall_native_iq
    '

module_path="$(find "$build_dir" -maxdepth 1 -type f -name '_softwall_native_iq*.so' -print -quit)"
if [[ -z "$module_path" ]]; then
    echo "native IQ module was not produced" >&2
    exit 1
fi
echo "C166 native IQ bridge built: $module_path"
