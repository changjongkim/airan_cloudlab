#!/usr/bin/env bash

# Six frozen arms on a new A100 node add on-path wall-clock telemetry for the
# prepare/commit/complete broker budget without changing the C136 sources.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

campaign="$SOFTWALL_ROOT/results/softwall_multigpu/confirm137_service_bound_telemetry_protocol.json"
result="$SOFTWALL_ROOT/results/softwall_multigpu/confirm137_service_bound_telemetry_result.json"
environment="$SOFTWALL_ROOT/results/softwall_multigpu/raw/confirm137_environment_job${SLURM_JOB_ID}.json"
node=$(hostname -s)
excluded_nodes=(nid001245 nid002817 nid002688 nid001781 nid001049)
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || {
        echo "C137 requires a node outside the frozen exclusion set" >&2
        exit 2
    }
done

# Interpreter compatibility and instrumentation canary before any formal arm.
shifter_gpu python3 -m unittest \
    test_deadline_global_trace_client \
    test_instrumented_deadline_global_trace_client

python3 - "$campaign" "$node" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_service_bound_controller.py" \
    "$SOFTWALL_SCRIPTS/deadline_fault_contained_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/instrumented_deadline_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/run_service_bound_telemetry_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_confirm137_service_bound_telemetry.py" \
    "$SOFTWALL_SCRIPTS/analyze_confirm136_v12_requalification.py" \
    "$SOFTWALL_SCRIPTS/analyze_confirm135_static_counterfactual.py" \
    "$SOFTWALL_SCRIPTS/global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_TASK1/isca_v2/dart_runtime.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];paths=[Path(value) for value in sys.argv[3:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab')
relative=[str(path.relative_to(root)) for path in paths]
arms=[]
for index in range(6):
 base=28000000+index*20000
 arms.append({'name':'s%d'%(index+1),'iterations':160,'warmup':20,
              'payload_seeds':[base+51,base+10051],
              'channel_seeds':[base+1000,base+11000]})
value={
 'schema':'softwall-confirm137-service-bound-telemetry-protocol-v1',
 'status':'frozen-after-canary-before-formal-arms',
 'allocation_node':node,
 'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049'],
 'arms':arms,
 'period_ms':180,'deadline_ms':155,'nrx_bound_ms':45,'conv_bound_ms':25,
 'target_ai_bound_ms':40,'global_broker_rpc_timeout_ms':5,
 'global_broker_transaction_budget_ms':15,'admission_ai_guard_ms':17,
 'required_rpc_operations':['prepare','commit','complete'],
 'failure_rule':'stop on first formal arm failure; do not widen a bound post hoc',
 'source_sha256':{rel:hashlib.sha256(path.read_bytes()).hexdigest()
                  for rel,path in zip(relative,paths)},
}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

python3 - "$environment" "$node" <<'PY'
import json,os,platform,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2]
# Python 3.6 does not support text=True; use universal_newlines explicitly.
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader'],universal_newlines=True)
value={'schema':'softwall-confirm137-allocation-environment-v1','job_id':int(os.environ['SLURM_JOB_ID']),
       'node':node,'platform':platform.platform(),'gpu_inventory':gpu.strip().splitlines(),
       'formal_arms_started':True,'formal_arms_completed':0}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

arm_results=()
for arm in 1 2 3 4 5 6; do
    offset=$(( (arm - 1) * 20000 ))
    base=$((28000000 + offset))
    label="confirm137_s${arm}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label"
    export SOFTWALL_SHARD_ITERATIONS=160
    export SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base + 51))
    export SOFTWALL_SHARD_PAYLOAD_SEED1=$((base + 10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base + 1000))
    export SOFTWALL_SHARD_CHANNEL_SEED1=$((base + 11000))
    bash "$SOFTWALL_SCRIPTS/run_service_bound_telemetry_arm.sh" \
        2>&1 | tee "$SOFTWALL_ROOT/results/softwall_multigpu/${label}.log"
    arm_results+=("$SOFTWALL_ROOT/results/softwall_multigpu/${label}_result.json")
done

python3 "$SOFTWALL_SCRIPTS/analyze_confirm137_service_bound_telemetry.py" \
    --campaign "$campaign" --arms "${arm_results[@]}" --output "$result"

python3 - "$environment" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);value=json.loads(path.read_text());value['formal_arms_completed']=6
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
PY

echo "C137 service-bound telemetry complete: $result"
