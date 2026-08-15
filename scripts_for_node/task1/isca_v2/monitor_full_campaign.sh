#!/usr/bin/env bash
set -Eeuo pipefail
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
INTERVAL=${INTERVAL:-60}
STATUS="$RESULTS_ROOT/CAMPAIGN_STATUS.tsv"
ALERTS="$RESULTS_ROOT/ALERTS.log"
stages=(
    02_nrx_stack
    02b_nrx_profilers
    03_workload_qualification
    04_nrx_capacity
    05_fiveway_compute
    06_dart_scheme
)
state_of() {
    local directory=$1
    if [[ -s "$directory/COMPLETE" ]]; then echo COMPLETE
    elif [[ -s "$directory/FAILED" ]]; then echo FAILED
    elif [[ -e "$directory/RUNNING" ]]; then echo RUNNING
    elif [[ -d "$directory" ]]; then echo WAITING
    else echo NOT_CREATED
    fi
}
while true; do
    temporary="$STATUS.tmp"
    printf 'utc\tstage\tstate\n' >"$temporary"
    failed=0
    now=$(date -u +%FT%TZ)
    for stage in "${stages[@]}"; do
        state=$(state_of "$RESULTS_ROOT/$stage")
        printf '%s\t%s\t%s\n' "$now" "$stage" "$state" >>"$temporary"
        if [[ "$state" == FAILED ]]; then failed=1; fi
    done
    mv "$temporary" "$STATUS"
    if ((failed)); then
        printf '%s [ALERT] one or more full-day stages failed\n' "$now" \
            | tee -a "$ALERTS"
        exit 20
    fi
    if [[ -s "$RESULTS_ROOT/FULL_CORE_COMPLETE" \
          && -s "$RESULTS_ROOT/DART_MECHANISM_GATES_COMPLETE" ]]; then
        printf '%s [COMPLETE] core and DART mechanism gates validated\n' "$now" \
            | tee -a "$ALERTS"
        exit 0
    fi
    sleep "$INTERVAL"
done
