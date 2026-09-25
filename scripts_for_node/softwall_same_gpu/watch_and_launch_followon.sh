#!/usr/bin/env bash

# Keep the next interactive request out of Slurm's submit quota until the
# currently running same-node comparison has released its allocation.
set -uo pipefail

cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab || exit 2
previous_job=${1:?previous Slurm job ID required}
result="$PWD/results/softwall_same_gpu"
raw="$result/raw"
mkdir -p "$raw"
watch_lock="$raw/.lock_followon_after_${previous_job}"
mkdir "$watch_lock" 2>/dev/null || {
    echo "follow-on watcher already running for job $previous_job" >&2
    exit 3
}
trap 'rmdir "$watch_lock" 2>/dev/null || true' EXIT INT TERM

while true; do
    state=$(sacct -j "$previous_job" --format=State -P -n | head -n 1)
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            echo "prior job $previous_job ended: $state"
            break
            ;;
        RUNNING|PENDING|COMPLETING|CONFIGURING|SUSPENDED|REQUEUED|RESIZING)
            sleep 60
            ;;
        *)
            echo "unrecognized or missing state for job $previous_job: $state" >&2
            sleep 60
            ;;
    esac
done

attempt=0
while true; do
    attempt=$((attempt + 1))
    attempt_log="$result/followon_allocation_attempt_${attempt}.log"
    echo "follow-on allocation attempt $attempt"
    salloc --account=m5320_g --constraint=gpu --qos=interactive \
        --time=04:00:00 --nodes=1 --gpus=4 \
        srun --overlap bash \
        scripts_for_node/softwall_same_gpu/run_followon_confirm39_37_38.sh \
        2>&1 | tee "$attempt_log"
    allocation_status=${PIPESTATUS[0]}
    if rg -q 'Granted job allocation' "$attempt_log"; then
        echo "follow-on allocation finished with status $allocation_status"
        exit "$allocation_status"
    fi
    if rg -q 'QOSMaxSubmitJobPerUserLimit|Job violates accounting/QOS policy' "$attempt_log"; then
        echo 'Slurm submit quota remains full; retrying after 60 seconds'
        sleep 60
        continue
    fi
    echo "follow-on request failed for a reason other than submit quota: status $allocation_status" >&2
    exit "$allocation_status"
done
