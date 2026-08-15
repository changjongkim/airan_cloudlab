#!/usr/bin/env bash
set -Eeuo pipefail
REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/06_dart_scheme/COMPLETE}
for _ in $(seq 1 28800); do
    [[ -s "$DEPENDENCY" ]] && break
    [[ -s "${DEPENDENCY%/*}/FAILED" ]] && exit 20
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || exit 21
python3 "$REPO/pyaerial/isca_v2/analyze_dart_scheme.py" \
    --root "$RESULTS_ROOT" 2>&1 | tee "$RESULTS_ROOT/dart_analysis.log"
date -u +%FT%TZ >"$RESULTS_ROOT/DART_MECHANISM_GATES_COMPLETE"
