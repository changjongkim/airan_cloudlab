#!/usr/bin/env bash
# Persistent L1 wrapper — runs real_l1.py in continuous loop until killed.
# Required for valid AI throughput measurement (5/24 issue: L1 ended after ~10s
# while AI ran for 30s, making "with_l1" runs ~mostly_alone).
#
# Usage (inside container):
#   bash real_l1_loop.sh <label> <num_cells> <iters_per_call>
#
# Usage (as Docker entrypoint):
#   docker run -d --rm --gpus device=$L1_UUID \
#       -v $AERIAL_SDK:/opt/nvidia/cuBB \
#       -v $SCRIPT_DIR:/scripts \
#       $IMAGE bash /scripts/real_l1_loop.sh bg_l1 20 50
#
# Loop continues until SIGTERM (docker kill).

set -uo pipefail

LABEL="${1:-bg_l1}"
CELLS="${2:-20}"
ITERS="${3:-50}"
SCRIPT="${SCRIPT:-/scripts/real_l1.py}"

# Trap signals for clean exit
trap 'echo "[loop] SIGTERM received, exiting"; exit 0' SIGTERM SIGINT

echo "[loop] starting persistent L1: label=$LABEL cells=$CELLS iters=$ITERS"

i=0
while true; do
  i=$((i+1))
  echo "[loop] iter set $i ($(date +%H:%M:%S))"
  python3 "$SCRIPT" "${LABEL}_${i}" "$CELLS" "$ITERS" >/dev/null 2>&1 || \
    echo "[loop] real_l1 returned non-zero at iter set $i"
done
