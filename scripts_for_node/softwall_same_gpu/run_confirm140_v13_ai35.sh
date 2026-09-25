#!/usr/bin/env bash

# Two prospective arms validate the corrected AI35 exchange-only mechanism.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

campaign="$SOFTWALL_ROOT/results/softwall_multigpu/confirm140_v13_ai35_protocol.json"
result="$SOFTWALL_ROOT/results/softwall_multigpu/confirm140_v13_ai35_result.json"
node=$(hostname -s)
excluded_nodes=(nid001245 nid002817 nid002688 nid001781 nid001049 nid001177 nid001144)
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || { echo "C140 node is excluded" >&2; exit 2; }
done

shifter_gpu env SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7 python3 -m unittest \
    test_deadline_global_trace_client \
    test_instrumented_deadline_global_trace_client_v2 \
    test_global_trace_qualified_control_bound_controller

python3 - "$campaign" "$node" "$SLURM_JOB_ID" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_qualified_control_bound_controller.py" \
    "$SOFTWALL_SCRIPTS/deadline_fault_contained_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/instrumented_deadline_global_trace_client_v2.py" \
    "$SOFTWALL_SCRIPTS/global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_TASK1/isca_v2/dart_runtime.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" \
    "$SOFTWALL_SCRIPTS/run_v13_ai35_candidate_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_confirm140_v13_ai35.py" \
    "$SOFTWALL_SCRIPTS/analyze_confirm135_static_counterfactual.py" \
    "$SOFTWALL_SCRIPTS/build_softwall_context64_mechanism_trace.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);paths=[Path(value) for value in sys.argv[4:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
relative=[str(path.relative_to(root)) for path in paths]
keys=["/softwall/global_trace_fully_budgeted_fault_contained_controller.py","/softwall/global_trace_qualified_control_bound_controller.py","/softwall/deadline_fault_contained_global_trace_client.py","/softwall/instrumented_deadline_global_trace_client_v2.py","/softwall/global_trace_lease_broker.py","/softwall/same_request_ipc_worker.py","/softwall/trace_qwen_worker.py","/softwall_task1/isca_v2/dart_runtime.py","/softwall_task1/isca_v2/cuda_ipc_channel.py"]
container_map=dict(zip(keys,relative[:len(keys)]))
arms=[]
for index in range(2):
 base=31000000+index*20000
 arms.append({'name':'s%d'%(index+1),'label':'confirm140_s%d_job%d'%(index+1,job),'iterations':160,'warmup':20,
              'payload_seeds':[base+51,base+10051],'channel_seeds':[base+1000,base+11000]})
value={'schema':'softwall-confirm140-v13-ai35-protocol-v1','status':'frozen-after-canary-before-formal-arms','allocation_node':node,
       'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049','nid001177','nid001144'],
       'arms':arms,'trace':'data/current/softwall_context64_mechanism_trace_v1.json','trace_sha256':'b14388c7356878458031d4e881a0510f42f1b4414493f876cd13d2b881979277',
       'raw_ai_bound_ms':35,'socket_timeout_ms':5,'rpc_admission_bound_ms':7,'control_transaction_budget_ms':21,'ai_completion_guard_ms':2,
       'static_all_fail_slack_ms':53,'conditional_decision_window_ms':58,'effective_transaction_bound_ms':58,
       'failure_rule':'stop on first formal arm failure; do not change the AI or control bound post hoc',
       'container_source_map':container_map,
       'source_sha256':{rel:hashlib.sha256(path.read_bytes()).hexdigest() for rel,path in zip(relative,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

arm_results=()
for arm in 1 2; do
    base=$((31000000 + (arm - 1) * 20000))
    label="confirm140_s${arm}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label"
    export SOFTWALL_SHARD_ITERATIONS=160
    export SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base + 51))
    export SOFTWALL_SHARD_PAYLOAD_SEED1=$((base + 10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base + 1000))
    export SOFTWALL_SHARD_CHANNEL_SEED1=$((base + 11000))
    bash "$SOFTWALL_SCRIPTS/run_v13_ai35_candidate_arm.sh" \
        2>&1 | tee "$SOFTWALL_ROOT/results/softwall_multigpu/${label}.log"
    arm_results+=("$SOFTWALL_ROOT/results/softwall_multigpu/${label}_result.json")
done

python3 "$SOFTWALL_SCRIPTS/analyze_confirm140_v13_ai35.py" \
    --campaign "$campaign" --arms "${arm_results[@]}" --output "$result"
echo "C140 V13 AI35 conditional exchange complete: $result"
