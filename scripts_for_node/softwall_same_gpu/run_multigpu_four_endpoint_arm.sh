#!/usr/bin/env bash

# One local-IPC plus three remote-P2P SoftWall arm using four physical GPUs.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation

iterations=${SOFTWALL_MGPU_ITERATIONS:-340}
warmup=${SOFTWALL_MGPU_WARMUP:-20}
label=${SOFTWALL_MGPU_LABEL:-g5_four_endpoint_job${SLURM_JOB_ID}}
system=softwall
payload_seed=${SOFTWALL_MGPU_PAYLOAD_SEED:-22200051}
channel_seed=${SOFTWALL_MGPU_CHANNEL_SEED:-22201000}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state="$SOFTWALL_ROOT/run_state/softwall_multigpu/${label}"
protocol="$result_root/${label}_protocol.json"
controller_output="$raw/${label}_controller.json"
local_worker_output="$raw/${label}_local_worker.json"
remote1_worker_output="$raw/${label}_remote1_worker.json"
remote2_worker_output="$raw/${label}_remote2_worker.json"
remote3_worker_output="$raw/${label}_remote3_worker.json"
background_output="$raw/${label}_background.json"
requal_output="$raw/${label}_requalification.json"
result="$result_root/${label}_result.json"
tag="mg_${SLURM_JOB_ID}_${BASHPID}"
socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/mg_${BASHPID}.sock"

mkdir -p "$result_root" "$raw" "$state" "$(dirname "$socket")"
rm -f "$state"/cuda_ipc_* "$socket" "$controller_output" \
    "$local_worker_output" "$remote1_worker_output" "$remote2_worker_output" "$remote3_worker_output" "$background_output" \
    "$requal_output" "$result"

python3 - "$protocol" "$iterations" "$warmup" "$label" "$system" \
    "$SOFTWALL_SCRIPTS/four_cell_trace_baseline_controller.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/multigpu_p2p_nrx_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_SCRIPTS/multigpu_p2p_ipc_gate.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" <<'PY'
import hashlib, json, sys
from pathlib import Path

out=Path(sys.argv[1]); iterations=int(sys.argv[2]); warmup=int(sys.argv[3]); label=sys.argv[4]
system=sys.argv[5]; paths=[Path(value) for value in sys.argv[6:]]
keys=[
 "/softwall/four_cell_trace_baseline_controller.py",
 "/softwall/same_request_ipc_worker.py",
 "/softwall/multigpu_p2p_nrx_worker.py",
 "/softwall/trace_qwen_worker.py",
 "/softwall/multigpu_p2p_ipc_gate.py",
 "/softwall_task1/isca_v2/cuda_ipc_channel.py",
]
value={
 "schema":"softwall-multigpu-heterogeneous-arm-protocol-v1",
 "status":"frozen-before-run", "label":label,
 "system":system,
 "iterations":iterations, "warmup":warmup,
 "placement":{
   "gpu0":"4-cell cuPHY/Aerial + local NRx endpoint + Qwen2.5-1.5B",
   "gpu1":"remote resident NRx endpoint 1",
   "gpu2":"remote resident NRx endpoint 2",
   "gpu3":"remote resident NRx endpoint 3",
   "transport0":"same-device CUDA IPC",
   "transport1":"GPU0<->GPU1 cross-process CUDA IPC + bidirectional NVLink P2P",
   "transport2":"GPU0<->GPU2 cross-process CUDA IPC + bidirectional NVLink P2P",
   "transport3":"GPU0<->GPU3 cross-process CUDA IPC + bidirectional NVLink P2P",
 },
 "mps":True, "mps_devices":[0,1,2,3], "local_nrx_cap":40,
 "remote_nrx_cap_each":40, "qwen_cap":20, "nrx_endpoints":4,
 "period_ms":180, "deadline_ms":155, "nrx_bound_ms":45,
 "conv_bound_ms":12, "commit_guard_ms":2,
 "fault_every":5, "fault_pattern":"mixed",
 "trace_sha256":"da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d",
 "source_sha256":{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in zip(keys,paths)},
}
tmp=out.with_suffix(out.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(out)
PY

shifter_local_gpu() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --volume="$SOFTWALL_ROOT/data/public/burstgpt:/softwall_burstgpt" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=CUDA_MODULE_LOADING=LAZY \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_VISIBLE_DEVICES=0 "$@"
}

shifter_remote_gpu() {
    local visible=$1
    shift
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=CUDA_MODULE_LOADING=LAZY \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_VISIBLE_DEVICES="$visible" "$@"
}

shifter_local_qwen() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=CUDA_MODULE_LOADING=LAZY \
        --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/softwall \
        --env=HF_HOME=/softwall_runtime/cache/huggingface \
        --env=TMPDIR=/softwall_runtime/tmp --env=HF_HUB_DISABLE_XET=1 \
        --env=CUDA_VISIBLE_DEVICES=0 "$@"
}

local_pid=""; remote1_pid=""; remote2_pid=""; remote3_pid=""; background_pid=""
cleanup() {
    for pid in "$local_pid" "$remote1_pid" "$remote2_pid" "$remote3_pid" "$background_pid"; do
        [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
    done
    for _ in {1..30}; do
        alive=0
        for pid in "$local_pid" "$remote1_pid" "$remote2_pid" "$remote3_pid" "$background_pid"; do
            [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null || alive=1
        done
        [[ "$alive" -eq 0 ]] && break
        sleep 0.1
    done
    for pid in "$local_pid" "$remote1_pid" "$remote2_pid" "$remote3_pid" "$background_pid"; do
        [[ -z "$pid" ]] || kill -KILL "$pid" 2>/dev/null || true
        [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    done
    rm -f "$socket"
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
sleep 5
export SOFTWALL_MPS_GPU=0,1,2,3
softwall_mps_start
softwall_mps_assert

shifter_local_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal_output" --iterations 50 --warmup 10 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 50 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 22201955 --snr-db -8.5 --channel-seed-base 22209900

shifter_local_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/same_request_ipc_worker.py \
    --tag "${tag}_c0" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$local_worker_output" &
local_pid=$!

shifter_remote_gpu 0,1 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/multigpu_p2p_nrx_worker.py \
    --tag "${tag}_c1" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$remote1_worker_output" --source-device 0 --destination-device 1 &
remote1_pid=$!

shifter_remote_gpu 0,2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/multigpu_p2p_nrx_worker.py \
    --tag "${tag}_c2" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$remote2_worker_output" --source-device 0 --destination-device 1 &
remote2_pid=$!

shifter_remote_gpu 0,3 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/multigpu_p2p_nrx_worker.py \
    --tag "${tag}_c3" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$remote3_worker_output" --source-device 0 --destination-device 1 &
remote3_pid=$!

shifter_local_qwen env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/trace_qwen_worker.py \
    --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 16,32,64,128,256,512 \
    --batch-size 1 --warmup-per-length 3 --socket "$socket" \
    --output "$background_output" &
background_pid=$!

for _ in {1..3000}; do
    [[ -S "$socket" ]] && break
    kill -0 "$background_pid" 2>/dev/null || {
        echo "multi-GPU Qwen worker exited before ready" >&2; exit 1;
    }
    sleep 0.05
done
[[ -S "$socket" ]] || { echo "multi-GPU Qwen readiness timeout" >&2; exit 1; }

shifter_local_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/four_cell_trace_baseline_controller.py \
    --tag-prefix "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --socket "$socket" \
    --trace /softwall_burstgpt/softwall_burst60_prefill_trace.json \
    --trace-sha256 da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d \
    --system "$system" --ai-bound-map 16:35,32:35,64:35,128:40,256:65,512:75 \
    --output "$controller_output" --iterations "$iterations" --cells 4 --nrx-endpoints 4 \
    --warmup "$warmup" --period-ms 180 --deadline-ms 155 \
    --nrx-bound-ms 45 --conv-bound-ms 12 --commit-guard-ms 2 \
    --endpoint-timeout-ms 100 --ai-guard-ms 2 --ai-rpc-timeout-ms 120 \
    --seed "$payload_seed" --snr-db -8.5 --channel-seed-base "$channel_seed" \
    --gate-mode low_threshold --gate-threshold 1.9490545988082886 \
    --gc-mode off --alternate-admission-order \
    --inject-correlated-failure-every 5 --fault-pattern mixed

wait "$local_pid"; local_pid=""
wait "$remote1_pid"; remote1_pid=""
wait "$remote2_pid"; remote2_pid=""
wait "$remote3_pid"; remote3_pid=""
wait "$background_pid"; background_pid=""

python3 - "$protocol" "$controller_output" "$local_worker_output" \
    "$remote1_worker_output" "$remote2_worker_output" "$remote3_worker_output" "$background_output" "$requal_output" "$result" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

protocol,controller,local,remote1,remote2,remote3,background,requal,out=map(Path,sys.argv[1:])
p=json.loads(protocol.read_text()); c=json.loads(controller.read_text())
l=json.loads(local.read_text()); r1=json.loads(remote1.read_text()); r2=json.loads(remote2.read_text()); r3=json.loads(remote3.read_text())
b=json.loads(background.read_text()); q=json.loads(requal.read_text())
gates={
 "schema":c.get("schema")=="softwall-four-cell-trace-baseline-v1" and not c.get("failed",False),
 "requalification":q["deadline_misses"]==0,
 "radio_deadline":c["deadline_misses"]==0,
 "nrx_bound":c["nrx_bound_violations"]==0,
 "conv_bound":c["conv_bound_violations"]==0 and c["conv_path_bound_violations"]==0,
 "ai_bound":c["background_budget_violations"]==0 and c["background_horizon_violations"]==0,
 "pre_radio_guard":c["pre_radio_ai_guard_violations"]==0,
 "endpoint_faults":len(c["endpoint_faults"])==0,
 "background_faults":len(c["background_faults"])==0,
 "local_worker":l["completed_units"]>0,
 "remote1_worker":r1["completed_units"]>0 and r1["nonfinite_outputs"]==0 and r1["error"] is None,
 "remote2_worker":r2["completed_units"]>0 and r2["nonfinite_outputs"]==0 and r2["error"] is None,
 "remote3_worker":r3["completed_units"]>0 and r3["nonfinite_outputs"]==0 and r3["error"] is None,
 "remote_sequence":bool(r1["sequences_contiguous"]) and bool(r2["sequences_contiguous"]) and bool(r3["sequences_contiguous"]),
 "recovery_credit_empty":c["fallback_calendar_final"]["outstanding"]==0
     and c["fallback_calendar_final"]["joint_leases_outstanding"]==0,
 "endpoint_credit_restored":all(v["outstanding"]==0 for v in c["endpoint_final"].values()),
}
result={
 "schema":"softwall-multigpu-four-endpoint-arm-gate-v1",
 "protocol":str(protocol), "controller":str(controller),
 "local_worker":str(local), "remote1_worker":str(remote1), "remote2_worker":str(remote2), "remote3_worker":str(remote3),
 "background":str(background), "requalification":str(requal),
 "summary":{
   "system":c["system"],
   "iterations":c["iterations"], "radio_records":len(c["records"]),
   "correct_cells":c["correct_cells"], "nrx_commits":c["nrx_commits"],
   "conv_commits":c["conv_commits"], "fallbacks":c["fallbacks"],
   "recovery_retime_count":c["recovery_retime_count"],
   "joint_lease_retired_count":c["joint_lease_retired_count"],
   "ai_timely_units":c["ai_timely_units"],
   "ai_timely_value_tokens":c["ai_timely_value_tokens"],
   "nrx_response_ms":c["nrx_response_ms"],
   "remote1_worker_path_ms":r1["worker_path_ms"],
   "remote2_worker_path_ms":r2["worker_path_ms"],
   "remote3_worker_path_ms":r3["worker_path_ms"],
 },
 "gates":gates, "all_pass":all(gates.values()),
 "artifact_sha256":{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [protocol,controller,local,remote1,remote2,remote3,background,requal]},
 "scope":"one local IPC plus three remote P2P NRx endpoints on four physical GPUs in the four-cell SoftWall/Qwen/fault path",
}
tmp=out.with_suffix(out.suffix+'.tmp'); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(out)
print(json.dumps(result,indent=2))
if not result['all_pass']: raise SystemExit('four-endpoint SoftWall arm failed')
PY

cleanup
trap - EXIT INT TERM
echo "four-endpoint SoftWall arm complete: $result"
