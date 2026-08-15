#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${1:-/tmp/dart_rx_offline}
mkdir -p "$OUTPUT_DIR"

python3 -m py_compile \
    "$SCRIPT_DIR/dart_runtime.py" \
    "$SCRIPT_DIR/test_dart_runtime.py" \
    "$SCRIPT_DIR/dart_transaction_trace.py" \
    "$SCRIPT_DIR/build_dart_q_trace.py" \
    "$SCRIPT_DIR/test_build_dart_q_trace.py" \
    "$SCRIPT_DIR/dart_q_simulator.py" \
    "$SCRIPT_DIR/test_dart_q_simulator.py"

(
    cd "$SCRIPT_DIR"
    python3 test_dart_runtime.py \
        --fault-iterations 10000 \
        --output "$OUTPUT_DIR/G7_FAULT_CORRECTNESS.json"
    python3 test_build_dart_q_trace.py
    python3 test_dart_q_simulator.py
)

date -u +%FT%TZ >"$OUTPUT_DIR/OFFLINE_G7_G8_SCAFFOLD_PASS"
printf '[DART-OFFLINE] PASS output=%s\n' "$OUTPUT_DIR"

