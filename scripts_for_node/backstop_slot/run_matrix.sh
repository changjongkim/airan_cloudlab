#!/usr/bin/env bash
# Run a list of slot-scale configurations back to back.
#   SLOT_MATRIX="c8:rescue:backstop c16:parallel:static ..." SLOT_TAG=m1 bash run_matrix.sh
set -uo pipefail
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot/slot_env.sh
require_allocation
tag=${SLOT_TAG:?}
periods=${SLOT_PERIODS:-2400}
rate=${SLOT_AI_RATE:-4.0}
configs="$SOFTWALL_ROOT/run_state/backstop_slot/configs/$tag"
mkdir -p "$configs" "$SLOT_RESULTS/raw"
for entry in ${SLOT_MATRIX:?}; do
    IFS=: read -r cells nrx ai <<<"$entry"
    name="${tag}_${cells}_${nrx}_${ai}"
    config="$configs/$name.json"
    if [ -f "$SLOT_RESULTS/raw/${name}_j${SLURM_JOB_ID}.json" ]; then
        echo "=== $name (already done)"
        continue
    fi
    shifter --image="$AERIAL_IMAGE" bash -c "cd $SLOT_SCRIPTS && python3 make_config.py --cells ${cells#c} --nrx-policy $nrx --ai-policy $ai --periods $periods --ai-rate $rate ${SLOT_EXTRA:-} --output $config"
    echo "=== $name $(date +%T)"
    bash "$SLOT_SCRIPTS/run_slot.sh" "$config" "$SLOT_RESULTS/raw/${name}_j${SLURM_JOB_ID}.json" \
        > "$SLOT_RESULTS/raw/${name}_j${SLURM_JOB_ID}.log" 2>&1 || echo "FAILED $name"
    grep -A 30 '^{' "$SLOT_RESULTS/raw/${name}_j${SLURM_JOB_ID}.log" | tr -d '\n ' | cut -c1-600; echo
done
