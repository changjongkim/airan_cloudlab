#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh

require_allocation

iterations=${SOFTWALL_P2P_ITERATIONS:-10000}
label=${SOFTWALL_P2P_LABEL:-g1_p2p_ipc_job${SLURM_JOB_ID}}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/${label}"
script="$SOFTWALL_SCRIPTS/multigpu_p2p_ipc_gate.py"
channel="$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py"
protocol="$result_root/${label}_protocol.json"
owner_output="$result_root/${label}_owner.json"
worker_output="$result_root/${label}_worker.json"
result="$result_root/${label}_result.json"
owner_log="$result_root/${label}_owner.log"
worker_log="$result_root/${label}_worker.log"
tag="${label}_${BASHPID}"

mkdir -p "$result_root" "$state"
rm -f "$state"/cuda_ipc_* "$owner_output" "$worker_output" "$result" \
    "$owner_log" "$worker_log"

python3 - "$protocol" "$script" "$channel" "$iterations" "$label" <<'PY'
import hashlib, json, sys
from pathlib import Path

out = Path(sys.argv[1])
script = Path(sys.argv[2])
channel = Path(sys.argv[3])
iterations = int(sys.argv[4])
label = sys.argv[5]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

value = {
    "schema": "softwall-multigpu-p2p-ipc-protocol-v1",
    "status": "frozen-before-run",
    "label": label,
    "iterations": iterations,
    "source_device": 0,
    "destination_device": 1,
    "mps": False,
    "ring_depth": 1,
    "timing_warmup_units": min(20, iterations),
    "forward_bytes": 1415232,
    "backward_bytes": 314496,
    "integrity": "GPU-side elementwise exact float32 pattern plus sequence check",
    "source_sha256": {
        "/softwall/multigpu_p2p_ipc_gate.py": sha256(script),
        "/softwall_task1/isca_v2/cuda_ipc_channel.py": sha256(channel),
    },
}
temporary = out.with_suffix(out.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2))
temporary.replace(out)
PY

shifter_p2p() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/softwall:/softwall_task1 \
        --env=CUDA_VISIBLE_DEVICES=0,1 \
        "$@"
}

owner_pid=""
worker_pid=""
cleanup() {
    [[ -z "$owner_pid" ]] || kill "$owner_pid" 2>/dev/null || true
    [[ -z "$worker_pid" ]] || kill "$worker_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

shifter_p2p python3 /softwall/multigpu_p2p_ipc_gate.py \
    --role owner --tag "$tag" --ipc-dir "$state" \
    --iterations "$iterations" --source-device 0 --destination-device 1 \
    --output "$owner_output" >"$owner_log" 2>&1 &
owner_pid=$!

info="$state/cuda_ipc_${tag}.info"
for _ in {1..2400}; do
    [[ -f "$info" ]] && break
    kill -0 "$owner_pid" 2>/dev/null || {
        cat "$owner_log" >&2
        echo "P2P owner exited before publishing IPC handles" >&2
        exit 1
    }
    sleep 0.05
done
[[ -f "$info" ]] || {
    echo "P2P owner IPC publication timeout" >&2
    exit 1
}

shifter_p2p python3 /softwall/multigpu_p2p_ipc_gate.py \
    --role worker --tag "$tag" --ipc-dir "$state" \
    --iterations "$iterations" --source-device 0 --destination-device 1 \
    --output "$worker_output" >"$worker_log" 2>&1 &
worker_pid=$!

wait "$owner_pid"
owner_pid=""
wait "$worker_pid"
worker_pid=""

shifter_p2p python3 /softwall/analyze_multigpu_p2p_ipc_gate.py \
    --protocol "$protocol" --owner "$owner_output" --worker "$worker_output" \
    --output "$result"

trap - EXIT INT TERM
echo "multi-GPU P2P IPC gate complete: $result"
