#!/usr/bin/env bash
# usage: run_nv_dataset.sh <output.npz> [nv_dataset.py options]
set -euo pipefail
ROOT=/pscratch/sd/s/sgkim/kcj/airan_cloudlab
source $ROOT/scripts_for_node/softwall_same_gpu/common.sh
out=$1; shift
cd $ROOT/third_party/neural_rx/scripts
shifter --module=gpu --image="$AERIAL_IMAGE" \
  --env=PYTHONPATH=$ROOT/runtime/softwall_same_gpu/nrx_export_deps --env=PYTHONNOUSERSITE=1 \
  python3 $ROOT/scripts_for_node/backstop_slot/nv/nv_dataset.py --output "$out" "$@"
