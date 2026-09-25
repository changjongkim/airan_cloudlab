#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

result_dir="$SOFTWALL_ROOT/results/softwall_multigpu"
protocol="$result_dir/confirm144_v14_independent_node_protocol.json"
result="$result_dir/confirm144_v14_independent_node_result.json"
node=$(hostname -s)
excluded_nodes=(nid001245 nid002817 nid002688 nid001781 nid001049 nid001177 nid001144 nid001252 nid001372 nid001361)
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || { echo "C144 requires an independent node" >&2; exit 2; }
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
    "$SOFTWALL_SCRIPTS/run_v14_ai45_pipelined_arm.sh" \
    "$SOFTWALL_SCRIPTS/run_v14_pipelined_control_fault_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_confirm144_v14_independent_node.py" \
    "$SOFTWALL_SCRIPTS/verify_pipelined_control_model.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);paths=[Path(x) for x in sys.argv[4:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab');rel=[str(p.relative_to(root)) for p in paths]
value={'schema':'softwall-confirm144-v14-independent-node-protocol-v1','status':'frozen-before-arms','allocation_node':node,'excluded_nodes':['nid001245','nid002817','nid002688','nid001781','nid001049','nid001177','nid001144','nid001252','nid001372','nid001361'],'ai_arm':{'label':'confirm144_ai45_job%d'%job,'iterations':160,'warmup':20,'payload_seeds':[36000051,36010051],'channel_seeds':[36001000,36011000]},'fault_arms':[{'operation':op,'label':'confirm144_%s_job%d'%(op,job),'iterations':200,'warmup':20,'payload_seeds':[36100051+i*20000,36110051+i*20000],'channel_seeds':[36101000+i*20000,36111000+i*20000]} for i,op in enumerate(('prepare','commit','complete'))],'ai_contract':{'raw_ms':45,'commit_ms':7,'guard_ms':2,'effective_ms':54,'static_slack_ms':53,'conditional_window_ms':58},'failure_rule':'stop on first failure; preserve failure; no post-hoc bound widening','source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rel,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

export SOFTWALL_SHARD_LABEL="confirm144_ai45_job${SLURM_JOB_ID}" SOFTWALL_SHARD_ITERATIONS=160 SOFTWALL_SHARD_WARMUP=20
export SOFTWALL_SHARD_PAYLOAD_SEED0=36000051 SOFTWALL_SHARD_PAYLOAD_SEED1=36010051
export SOFTWALL_SHARD_CHANNEL_SEED0=36001000 SOFTWALL_SHARD_CHANNEL_SEED1=36011000
bash "$SOFTWALL_SCRIPTS/run_v14_ai45_pipelined_arm.sh" 2>&1 | tee "$result_dir/${SOFTWALL_SHARD_LABEL}.log"
ai_result="$result_dir/${SOFTWALL_SHARD_LABEL}_result.json"

fault_results=()
operations=(prepare commit complete)
for index in 0 1 2; do
    operation=${operations[$index]}; base=$((36100000 + index * 20000))
    label="confirm144_${operation}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label" SOFTWALL_SHARD_ITERATIONS=200 SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_BROKER_CRASH_OPERATION="$operation" SOFTWALL_BROKER_CRASH_NUMBER=30
    export SOFTWALL_BROKER_RPC_SOCKET_TIMEOUT_MS=5 SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base+51)) SOFTWALL_SHARD_PAYLOAD_SEED1=$((base+10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base+1000)) SOFTWALL_SHARD_CHANNEL_SEED1=$((base+11000))
    bash "$SOFTWALL_SCRIPTS/run_v14_pipelined_control_fault_arm.sh" 2>&1 | tee "$result_dir/${label}.log"
    fault_results+=("$result_dir/${label}_result.json")
done

python3 "$SOFTWALL_SCRIPTS/analyze_confirm144_v14_independent_node.py" \
    --protocol "$protocol" --ai "$ai_result" --faults "${fault_results[@]}" \
    --output "$result"
echo "C144 V14 independent-node requalification complete: $result"
