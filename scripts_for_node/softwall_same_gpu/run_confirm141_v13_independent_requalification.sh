#!/usr/bin/env bash

# Requalify both corrected V13 components on a node independent of C139/C140.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

result_dir="$SOFTWALL_ROOT/results/softwall_multigpu"
raw_dir="$result_dir/raw"
control_protocol="$result_dir/confirm141_control_protocol.json"
control_result="$result_dir/confirm141_control_result.json"
ai_protocol="$result_dir/confirm141_ai35_protocol.json"
ai_result="$result_dir/confirm141_ai35_result.json"
combined_result="$result_dir/confirm141_v13_independent_requalification_result.json"
environment="$raw_dir/confirm141_environment_job${SLURM_JOB_ID}.json"
node=$(hostname -s)
excluded_nodes=(nid001245 nid002817 nid002688 nid001781 nid001049 nid001177 nid001144 nid001252)
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || { echo "C141 requires an independent node" >&2; exit 2; }
done

shifter_gpu env SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7 python3 -m unittest \
    test_deadline_global_trace_client \
    test_instrumented_deadline_global_trace_client_v2 \
    test_global_trace_qualified_control_bound_controller

python3 - "$environment" "$node" <<'PY'
import json,os,platform,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2]
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader'],universal_newlines=True)
value={'schema':'softwall-confirm141-allocation-environment-v1','job_id':int(os.environ['SLURM_JOB_ID']),
       'node':node,'platform':platform.platform(),'gpu_inventory':gpu.strip().splitlines(),
       'formal_control_arms_completed':0,'formal_ai35_arms_completed':0}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

python3 - "$control_protocol" "$node" "$SLURM_JOB_ID" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_qualified_control_bound_controller.py" \
    "$SOFTWALL_SCRIPTS/deadline_fault_contained_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/instrumented_deadline_global_trace_client_v2.py" \
    "$SOFTWALL_SCRIPTS/test_instrumented_deadline_global_trace_client_v2.py" \
    "$SOFTWALL_SCRIPTS/test_global_trace_qualified_control_bound_controller.py" \
    "$SOFTWALL_SCRIPTS/run_qualified_control_bound_fault_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_confirm139_qualified_control_bound.py" \
    "$SOFTWALL_SCRIPTS/control_point_crash_global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_TASK1/isca_v2/dart_runtime.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);paths=[Path(x) for x in sys.argv[4:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab'); rel=[str(p.relative_to(root)) for p in paths]
ops=['prepare','prepare','commit','commit','complete','complete'];arms=[]
for i,op in enumerate(ops):
 base=32000000+i*20000
 arms.append({'name':'c%d'%(i+1),'label':'confirm141_control_%s%d_job%d'%(op,(i%2)+1,job),
              'operation':op,'iterations':340,'warmup':20,
              'payload_seeds':[base+51,base+10051],'channel_seeds':[base+1000,base+11000]})
value={'schema':'softwall-confirm141-control-protocol-v1','status':'frozen-after-canary-before-formal-arms',
 'allocation_node':node,'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049','nid001177','nid001144','nid001252'],
 'arms':arms,'period_ms':180,'deadline_ms':155,'nrx_bound_ms':45,'conv_bound_ms':25,
 'global_broker_rpc_timeout_ms':5,'global_broker_rpc_admission_bound_ms':7,
 'global_broker_transaction_budget_ms':21,'admission_ai_guard_ms':23,'crash_after_operation_number':30,
 'fault_model':'broker process exits with status 86 immediately after applying the selected home0 operation and before replying',
 'failure_rule':'stop on first formal arm failure; preserve the failure; never widen the bound post hoc',
 'source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rel,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

control_arms=()
operations=(prepare prepare commit commit complete complete)
for index in 0 1 2 3 4 5; do
    arm=$((index + 1)); operation=${operations[$index]}; repetition=$((index % 2 + 1))
    base=$((32000000 + index * 20000)); label="confirm141_control_${operation}${repetition}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label" SOFTWALL_SHARD_ITERATIONS=340 SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_BROKER_CRASH_OPERATION="$operation" SOFTWALL_BROKER_CRASH_NUMBER=30
    export SOFTWALL_BROKER_RPC_SOCKET_TIMEOUT_MS=5 SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base+51)) SOFTWALL_SHARD_PAYLOAD_SEED1=$((base+10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base+1000)) SOFTWALL_SHARD_CHANNEL_SEED1=$((base+11000))
    bash "$SOFTWALL_SCRIPTS/run_qualified_control_bound_fault_arm.sh" 2>&1 | tee "$result_dir/${label}.log"
    control_arms+=("$result_dir/${label}_result.json")
    python3 - "$environment" "$arm" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads(p.read_text());v['formal_control_arms_completed']=int(sys.argv[2]);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
PY
done
python3 "$SOFTWALL_SCRIPTS/analyze_confirm139_qualified_control_bound.py" \
    --campaign "$control_protocol" --arms "${control_arms[@]}" --output "$control_result"
python3 "$SOFTWALL_SCRIPTS/normalize_confirm141_subcampaign.py" \
    --input "$control_result" --kind control \
    --output "$result_dir/confirm141_control_requalification_result.json"

python3 - "$ai_protocol" "$node" "$SLURM_JOB_ID" \
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
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);paths=[Path(x) for x in sys.argv[4:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab');rel=[str(p.relative_to(root)) for p in paths]
keys=['/softwall/global_trace_fully_budgeted_fault_contained_controller.py','/softwall/global_trace_qualified_control_bound_controller.py','/softwall/deadline_fault_contained_global_trace_client.py','/softwall/instrumented_deadline_global_trace_client_v2.py','/softwall/global_trace_lease_broker.py','/softwall/same_request_ipc_worker.py','/softwall/trace_qwen_worker.py','/softwall_task1/isca_v2/dart_runtime.py','/softwall_task1/isca_v2/cuda_ipc_channel.py']
arms=[]
for i in range(2):
 base=33000000+i*20000
 arms.append({'name':'a%d'%(i+1),'label':'confirm141_ai35_s%d_job%d'%(i+1,job),'iterations':160,'warmup':20,
              'payload_seeds':[base+51,base+10051],'channel_seeds':[base+1000,base+11000]})
value={'schema':'softwall-confirm141-ai35-protocol-v1','status':'frozen-after-control-pass-before-ai-arms','allocation_node':node,
 'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049','nid001177','nid001144','nid001252'],
 'arms':arms,'trace':'data/current/softwall_context64_mechanism_trace_v1.json','trace_sha256':'b14388c7356878458031d4e881a0510f42f1b4414493f876cd13d2b881979277',
 'raw_ai_bound_ms':35,'socket_timeout_ms':5,'rpc_admission_bound_ms':7,'control_transaction_budget_ms':21,
 'ai_completion_guard_ms':2,'static_all_fail_slack_ms':53,'conditional_decision_window_ms':58,'effective_transaction_bound_ms':58,
 'failure_rule':'stop on first formal arm failure; do not change the AI or control bound post hoc',
 'container_source_map':dict(zip(keys,rel[:len(keys)])),
 'source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rel,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

ai_arms=()
for arm in 1 2; do
    base=$((33000000 + (arm - 1) * 20000)); label="confirm141_ai35_s${arm}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label" SOFTWALL_SHARD_ITERATIONS=160 SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base+51)) SOFTWALL_SHARD_PAYLOAD_SEED1=$((base+10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base+1000)) SOFTWALL_SHARD_CHANNEL_SEED1=$((base+11000))
    bash "$SOFTWALL_SCRIPTS/run_v13_ai35_candidate_arm.sh" 2>&1 | tee "$result_dir/${label}.log"
    ai_arms+=("$result_dir/${label}_result.json")
    python3 - "$environment" "$arm" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads(p.read_text());v['formal_ai35_arms_completed']=int(sys.argv[2]);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
PY
done
python3 "$SOFTWALL_SCRIPTS/analyze_confirm140_v13_ai35.py" \
    --campaign "$ai_protocol" --arms "${ai_arms[@]}" --output "$ai_result"
python3 "$SOFTWALL_SCRIPTS/normalize_confirm141_subcampaign.py" \
    --input "$ai_result" --kind ai35 \
    --output "$result_dir/confirm141_ai35_requalification_result.json"
python3 "$SOFTWALL_SCRIPTS/analyze_confirm141_v13_independent_requalification.py" \
    --control "$result_dir/confirm141_control_requalification_result.json" \
    --ai35 "$result_dir/confirm141_ai35_requalification_result.json" \
    --output "$combined_result"
echo "C141 V13 independent-node requalification complete: $combined_result"
