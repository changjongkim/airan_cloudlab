#!/usr/bin/env bash
set -Eeuo pipefail
REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/05_fiveway_compute/COMPLETE}
LOG="$RESULTS_ROOT/analysis_runner.log"
for _ in $(seq 1 21600); do
    [[ -s "$DEPENDENCY" ]] && break
    [[ -s "${DEPENDENCY%/*}/FAILED" ]] && {
        echo "[FULL-ANALYSIS] dependency failed" >&2
        exit 20
    }
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || exit 21
python3 "$REPO/pyaerial/isca_v2/analyze_full_campaign.py" \
    --root "$RESULTS_ROOT" 2>&1 | tee "$LOG"
date -u +%FT%TZ >"$RESULTS_ROOT/FULL_CORE_COMPLETE"
