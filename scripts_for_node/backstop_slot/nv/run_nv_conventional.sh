#!/usr/bin/env bash
# usage: run_nv_conventional.sh <input.npz> <output.json> [options]
set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot/slot_env.sh
in=$1; out=$2; shift 2
cd "$SOFTWALL_ROOT"
shifter_slot python3 /backstop_slot/nv/nv_conventional.py --input "$in" --output "$out" "$@"
