#!/usr/bin/env bash

# Six independent process/seed arms requalify the complete V12 AI40 contract
# on one A100 node not used by C135 or the earlier independent-node campaign.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

campaign="$SOFTWALL_ROOT/results/softwall_multigpu/confirm136_v12_ai40_requalification_protocol.json"
result="$SOFTWALL_ROOT/results/softwall_multigpu/confirm136_v12_ai40_requalification_result.json"
node=$(hostname -s)
python3 - "$campaign" "$node" <<'PY'
import json,sys
from pathlib import Path
campaign=json.loads(Path(sys.argv[1]).read_text())
node=sys.argv[2]
if node in campaign["excluded_nodes"]:
    raise SystemExit("C136 requires a node outside the frozen exclusion set")
PY

arm_results=()
for arm in 1 2 3 4 5 6; do
    offset=$(( (arm - 1) * 20000 ))
    base=$((26000000 + offset))
    label="confirm136_s${arm}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label"
    export SOFTWALL_SHARD_ITERATIONS=160
    export SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base + 51))
    export SOFTWALL_SHARD_PAYLOAD_SEED1=$((base + 10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base + 1000))
    export SOFTWALL_SHARD_CHANNEL_SEED1=$((base + 11000))
    bash "$SOFTWALL_SCRIPTS/run_v11_ai40_candidate_arm.sh" \
        2>&1 | tee "$SOFTWALL_ROOT/results/softwall_multigpu/${label}.log"
    arm_results+=("$SOFTWALL_ROOT/results/softwall_multigpu/${label}_result.json")
done

python3 "$SOFTWALL_SCRIPTS/analyze_confirm136_v12_requalification.py" \
    --campaign "$campaign" --arms "${arm_results[@]}" --output "$result"
echo "C136 V12 cross-node requalification complete: $result"
