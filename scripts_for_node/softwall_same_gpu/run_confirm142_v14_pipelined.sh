#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

result_dir="$SOFTWALL_ROOT/results/softwall_multigpu"
protocol="$result_dir/confirm142_v14_pipelined_protocol.json"
result="$result_dir/confirm142_v14_pipelined_result.json"
node=$(hostname -s)
excluded_nodes=(nid001245 nid002817 nid002688 nid001781 nid001049 nid001177 nid001144 nid001252 nid001372)
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || { echo "C142 requires a new node" >&2; exit 2; }
done

shifter_gpu env SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7 python3 -m unittest \
    test_pipelined_global_trace_client \
    test_global_trace_pipelined_control_controller \
    test_verify_pipelined_control_model

python3 "$SOFTWALL_SCRIPTS/verify_pipelined_control_model.py" \
    --output "$result_dir/softwall_pipelined_control_model_v1.json"

python3 - "$protocol" "$node" "$SLURM_JOB_ID" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_pipelined_control_controller.py" \
    "$SOFTWALL_SCRIPTS/pipelined_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/test_pipelined_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_TASK1/isca_v2/dart_runtime.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" \
    "$SOFTWALL_SCRIPTS/run_v14_ai45_pipelined_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_confirm142_v14_pipelined.py" \
    "$SOFTWALL_SCRIPTS/verify_pipelined_control_model.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);paths=[Path(x) for x in sys.argv[4:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab');rel=[str(p.relative_to(root)) for p in paths]
keys=['/softwall/global_trace_fully_budgeted_fault_contained_controller.py','/softwall/global_trace_pipelined_control_controller.py','/softwall/pipelined_global_trace_client.py','/softwall/test_pipelined_global_trace_client.py','/softwall/global_trace_lease_broker.py','/softwall/same_request_ipc_worker.py','/softwall/trace_qwen_worker.py','/softwall_task1/isca_v2/dart_runtime.py','/softwall_task1/isca_v2/cuda_ipc_channel.py']
arms=[]
for i in range(2):
 base=34200000+i*20000
 arms.append({'name':'a%d'%(i+1),'label':'confirm142_ai45_s%d_job%d'%(i+1,job),'iterations':160,'warmup':20,'payload_seeds':[base+51,base+10051],'channel_seeds':[base+1000,base+11000]})
value={'schema':'softwall-confirm142-v14-pipelined-protocol-v1','status':'frozen-after-canary-before-formal-arms','allocation_node':node,'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049','nid001177','nid001144','nid001252','nid001372'],'arms':arms,'trace':'data/current/softwall_context64_mechanism_trace_v1.json','trace_sha256':'b14388c7356878458031d4e881a0510f42f1b4414493f876cd13d2b881979277','raw_ai_bound_ms':45,'socket_timeout_ms':5,'commit_bound_ms':7,'critical_control_budget_ms':7,'ai_completion_guard_ms':2,'static_all_fail_slack_ms':53,'conditional_decision_window_ms':58,'effective_transaction_bound_ms':54,'finite_model_artifact':'results/softwall_multigpu/softwall_pipelined_control_model_v1.json','failure_rule':'stop on first formal arm failure; preserve failures; do not widen AI or commit bounds post hoc','container_source_map':dict(zip(keys,rel[:len(keys)])),'source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rel,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

arms=()
for arm in 1 2; do
    base=$((34200000 + (arm - 1) * 20000))
    label="confirm142_ai45_s${arm}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label" SOFTWALL_SHARD_ITERATIONS=160 SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base+51)) SOFTWALL_SHARD_PAYLOAD_SEED1=$((base+10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base+1000)) SOFTWALL_SHARD_CHANNEL_SEED1=$((base+11000))
    bash "$SOFTWALL_SCRIPTS/run_v14_ai45_pipelined_arm.sh" 2>&1 | tee "$result_dir/${label}.log"
    arms+=("$result_dir/${label}_result.json")
done

python3 "$SOFTWALL_SCRIPTS/analyze_confirm142_v14_pipelined.py" \
    --campaign "$protocol" --arms "${arms[@]}" --output "$result"
echo "C142 V14 pipelined qualification complete: $result"
