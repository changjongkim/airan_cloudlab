#!/usr/bin/env bash

# Six prospective fail-stop arms qualify a 7 ms wall-time admission charge
# while retaining the 5 ms socket timeout that C138 showed was not itself a bound.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

campaign="$SOFTWALL_ROOT/results/softwall_multigpu/confirm139_qualified_control_bound_protocol.json"
result="$SOFTWALL_ROOT/results/softwall_multigpu/confirm139_qualified_control_bound_result.json"
environment="$SOFTWALL_ROOT/results/softwall_multigpu/raw/confirm139_environment_job${SLURM_JOB_ID}.json"
node=$(hostname -s)
excluded_nodes=(nid001245 nid002817 nid002688 nid001781 nid001049 nid001177 nid001144)
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || {
        echo "C139 requires a node outside the frozen exclusion set" >&2
        exit 2
    }
done

# The formal campaign is not frozen until both base containment and v2
# attempted-call instrumentation pass in the exact container interpreter.
shifter_gpu env SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7 python3 -m unittest \
    test_deadline_global_trace_client \
    test_instrumented_deadline_global_trace_client_v2 \
    test_global_trace_qualified_control_bound_controller

python3 - "$campaign" "$node" "$SLURM_JOB_ID" \
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
out=Path(sys.argv[1]);node=sys.argv[2];job_id=int(sys.argv[3]);paths=[Path(value) for value in sys.argv[4:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
relative=[str(path.relative_to(root)) for path in paths]
operations=['prepare','prepare','commit','commit','complete','complete']
arms=[]
for index,operation in enumerate(operations):
 base=30000000+index*20000
 arms.append({'name':'s%d'%(index+1),'label':'confirm139_%s%d_job%d'%(operation,(index%2)+1,job_id),
              'operation':operation,'iterations':340,'warmup':20,
              'payload_seeds':[base+51,base+10051],
              'channel_seeds':[base+1000,base+11000]})
value={
 'schema':'softwall-confirm139-qualified-control-bound-protocol-v1',
 'status':'frozen-after-canary-before-formal-arms',
 'allocation_node':node,
 'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049','nid001177','nid001144'],
 'arms':arms,
 'period_ms':180,'deadline_ms':155,'nrx_bound_ms':45,'conv_bound_ms':25,
 'global_broker_rpc_timeout_ms':5,'global_broker_rpc_admission_bound_ms':7,
 'global_broker_transaction_budget_ms':21,'admission_ai_guard_ms':23,
 'crash_after_operation_number':30,
 'fault_model':'broker process exits with status 86 immediately after applying the selected home0 operation and before replying',
 'telemetry_semantics':'record every socket-attempted RPC, including the detecting call; exclude calls after client quarantine; socket timeout exceedance is diagnostic while elapsed time above 7 ms fails qualification',
 'candidate_rationale':'prospectively rounded 7 ms admission charge after C138 rejected equality between a 5 ms socket timeout and a wall-time bound at observed maxima 5.205092 and 5.830741 ms',
 'failure_rule':'stop on first formal arm failure; preserve failure; do not widen the 7 ms admission bound post hoc',
 'source_sha256':{rel:hashlib.sha256(path.read_bytes()).hexdigest()
                  for rel,path in zip(relative,paths)},
}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

python3 - "$environment" "$node" <<'PY'
import json,os,platform,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2]
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader'],universal_newlines=True)
value={'schema':'softwall-confirm139-allocation-environment-v1','job_id':int(os.environ['SLURM_JOB_ID']),
       'node':node,'platform':platform.platform(),'gpu_inventory':gpu.strip().splitlines(),
       'formal_arms_started':True,'formal_arms_completed':0}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

operations=(prepare prepare commit commit complete complete)
arm_results=()
for index in 0 1 2 3 4 5; do
    arm=$((index + 1))
    operation=${operations[$index]}
    repetition=$((index % 2 + 1))
    base=$((30000000 + index * 20000))
    label="confirm139_${operation}${repetition}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label"
    export SOFTWALL_SHARD_ITERATIONS=340
    export SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_BROKER_CRASH_OPERATION="$operation"
    export SOFTWALL_BROKER_CRASH_NUMBER=30
    export SOFTWALL_BROKER_RPC_SOCKET_TIMEOUT_MS=5
    export SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base + 51))
    export SOFTWALL_SHARD_PAYLOAD_SEED1=$((base + 10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base + 1000))
    export SOFTWALL_SHARD_CHANNEL_SEED1=$((base + 11000))
    bash "$SOFTWALL_SCRIPTS/run_qualified_control_bound_fault_arm.sh" \
        2>&1 | tee "$SOFTWALL_ROOT/results/softwall_multigpu/${label}.log"
    arm_results+=("$SOFTWALL_ROOT/results/softwall_multigpu/${label}_result.json")
    python3 - "$environment" "$arm" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);value=json.loads(path.read_text());value['formal_arms_completed']=int(sys.argv[2])
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
PY
done

python3 "$SOFTWALL_SCRIPTS/analyze_confirm139_qualified_control_bound.py" \
    --campaign "$campaign" --arms "${arm_results[@]}" --output "$result"

echo "C139 qualified control-bound campaign complete: $result"
