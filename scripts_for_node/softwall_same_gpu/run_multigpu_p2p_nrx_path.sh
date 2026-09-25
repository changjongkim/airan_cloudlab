#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh

require_allocation

iterations=${SOFTWALL_P2P_NRX_ITERATIONS:-1000}
warmup=${SOFTWALL_P2P_NRX_WARMUP:-20}
period_ms=${SOFTWALL_P2P_NRX_PERIOD_MS:-20}
deadline_ms=${SOFTWALL_P2P_NRX_DEADLINE_MS:-15}
label=${SOFTWALL_P2P_NRX_LABEL:-g2a_p2p_nrx_job${SLURM_JOB_ID}}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/${label}"
controller="$SOFTWALL_SCRIPTS/same_request_ipc_controller.py"
worker="$SOFTWALL_SCRIPTS/multigpu_p2p_nrx_worker.py"
engine="$SOFTWALL_RUNTIME/engines/neural_rx_fp16_full.trt"
protocol="$result_root/${label}_protocol.json"
controller_output="$result_root/${label}_controller.json"
worker_output="$result_root/${label}_worker.json"
controller_log="$result_root/${label}_controller.log"
worker_log="$result_root/${label}_worker.log"
tag="${label}_${BASHPID}"

mkdir -p "$result_root" "$state"
rm -f "$state"/cuda_ipc_* "$controller_output" "$worker_output" \
    "$controller_log" "$worker_log"

python3 - "$protocol" "$controller" "$worker" \
    "$SOFTWALL_SCRIPTS/multigpu_p2p_ipc_gate.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" \
    "$iterations" "$warmup" "$period_ms" "$deadline_ms" "$label" <<'PY'
import hashlib, json, sys
from pathlib import Path

out = Path(sys.argv[1])
sources = [Path(value) for value in sys.argv[2:6]]
iterations, warmup = int(sys.argv[6]), int(sys.argv[7])
period_ms, deadline_ms = float(sys.argv[8]), float(sys.argv[9])
label = sys.argv[10]
keys = [
    "/softwall/same_request_ipc_controller.py",
    "/softwall/multigpu_p2p_nrx_worker.py",
    "/softwall/multigpu_p2p_ipc_gate.py",
    "/softwall_task1/isca_v2/cuda_ipc_channel.py",
]
value = {
    "schema": "softwall-multigpu-p2p-nrx-protocol-v1",
    "status": "frozen-before-run",
    "label": label,
    "iterations": iterations,
    "warmup": warmup,
    "period_ms": period_ms,
    "deadline_ms": deadline_ms,
    "source_device": 0,
    "destination_device": 1,
    "mps": False,
    "channel": "clean deterministic valid PUSCH",
    "policy": "s2",
    "source_sha256": {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in zip(keys, sources)
    },
}
temporary = out.with_suffix(out.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2))
temporary.replace(out)
PY

shifter_p2p() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_VISIBLE_DEVICES=0,1 \
        "$@"
}

controller_pid=""
worker_pid=""
cleanup() {
    [[ -z "$controller_pid" ]] || kill "$controller_pid" 2>/dev/null || true
    [[ -z "$worker_pid" ]] || kill "$worker_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

shifter_p2p python3 /softwall/same_request_ipc_controller.py \
    --tag "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$controller_output" --iterations "$iterations" --warmup "$warmup" \
    --period-ms "$period_ms" --deadline-ms "$deadline_ms" \
    --endpoint-timeout-ms 100 --seed 22110001 --policy s2 \
    >"$controller_log" 2>&1 &
controller_pid=$!

shifter_p2p python3 /softwall/multigpu_p2p_nrx_worker.py \
    --tag "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$worker_output" --source-device 0 --destination-device 1 \
    >"$worker_log" 2>&1 &
worker_pid=$!

wait "$controller_pid"
controller_pid=""
wait "$worker_pid"
worker_pid=""
trap - EXIT INT TERM

python3 - "$protocol" "$controller_output" "$worker_output" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); w=json.load(open(sys.argv[3]))
expected=p["iterations"] + p["warmup"]
gates={
 "warmup_correct": c["warmup_correct"] == p["warmup"],
 "timed_correct": c["correct_releases"] == p["iterations"],
 "deadline": c["deadline_misses"] == 0,
 "timeout": c["endpoint_timeouts"] == 0,
 "worker_count": w["completed_units"] == expected,
 "worker_sequence": w["sequences_contiguous"],
 "finite": w["nonfinite_outputs"] == 0,
 "worker_error": w["error"] is None,
}
out={
 "schema":"softwall-multigpu-p2p-nrx-path-gate-v1",
 "protocol":sys.argv[1], "controller":sys.argv[2], "worker":sys.argv[3],
 "controller_response_ms":c["response_ms"],
 "front_gpu_ms":c["front_gpu_ms"], "post_gpu_ms":c["post_gpu_ms"],
 "forward_gpu_us":w["forward_gpu_us"], "nrx_gpu_ms":w["nrx_gpu_ms"],
 "backward_gpu_us":w["backward_gpu_us"], "worker_path_ms":w["worker_path_ms"],
 "gates":gates, "all_pass":all(gates.values()),
 "scope":"isolated clean-PUSCH remote P2P NeuralRx path; no MPS co-run or all-fail transaction claim",
}
path=sys.argv[2].replace('_controller.json','_result.json')
tmp=path+'.tmp'; open(tmp,'w').write(json.dumps(out,indent=2)); __import__('os').replace(tmp,path)
print(json.dumps(out,indent=2))
if not out['all_pass']: raise SystemExit('remote P2P NRx path gate failed')
PY

echo "multi-GPU P2P NRx path complete: ${controller_output%_controller.json}_result.json"

