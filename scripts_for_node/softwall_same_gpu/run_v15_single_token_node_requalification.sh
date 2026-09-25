#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

campaign=${SOFTWALL_V15_CAMPAIGN:?set campaign name, for example confirm145}
seed_base=${SOFTWALL_V15_SEED_BASE:?set integer seed base}
excluded_csv=${SOFTWALL_V15_EXCLUDED_NODES:-nid001245,nid002817,nid002688,nid001781,nid001049,nid001177,nid001144,nid001252,nid001372,nid001361,nid002049}
node=$(hostname -s)
IFS=',' read -r -a excluded_nodes <<< "$excluded_csv"
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || { echo "$campaign requires a new node" >&2; exit 2; }
done

result_dir="$SOFTWALL_ROOT/results/softwall_multigpu"
protocol="$result_dir/${campaign}_v15_single_token_protocol.json"
result="$result_dir/${campaign}_v15_single_token_result.json"

shifter_gpu env SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7 python3 -m unittest \
    test_single_token_pipelined_global_trace_client \
    test_audit_v14_v15_staged_ownership \
    test_verify_pipelined_control_model_v2 \
    test_global_trace_single_token_pipelined_controller

shifter_gpu python3 /softwall/verify_pipelined_control_model_v2.py \
    --output "$result_dir/softwall_pipelined_control_model_v2.json"
shifter_gpu python3 /softwall/audit_v14_v15_staged_ownership.py \
    --output "$result_dir/softwall_v14_v15_staged_ownership_regression_v1.json"

python3 - "$protocol" "$node" "$SLURM_JOB_ID" "$campaign" "$seed_base" "$excluded_csv" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_single_token_pipelined_controller.py" \
    "$SOFTWALL_SCRIPTS/single_token_pipelined_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/pipelined_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/run_v15_ai45_single_token_arm.sh" \
    "$SOFTWALL_SCRIPTS/run_v15_single_token_control_fault_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_v15_single_token_node.py" \
    "$SOFTWALL_SCRIPTS/verify_pipelined_control_model_v2.py" \
    "$SOFTWALL_SCRIPTS/audit_v14_v15_staged_ownership.py" \
    "$SOFTWALL_SCRIPTS/test_single_token_pipelined_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/test_global_trace_single_token_pipelined_controller.py" \
    "$SOFTWALL_SCRIPTS/run_v15_single_token_node_requalification.sh" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);campaign=sys.argv[4];base=int(sys.argv[5]);excluded=[x for x in sys.argv[6].split(',') if x];paths=[Path(x) for x in sys.argv[7:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab');rel=[str(p.relative_to(root)) for p in paths]
value={'schema':'softwall-v15-single-token-node-protocol-v1','status':'frozen-before-arms','campaign':campaign,'allocation_node':node,'job_id':job,'excluded_nodes':excluded,'ownership_contract':'at-most-one-staged-or-offered-token-per-home','ai_arm':{'label':'%s_ai45_job%d'%(campaign,job),'iterations':160,'warmup':20,'payload_seeds':[base+51,base+10051],'channel_seeds':[base+1000,base+11000]},'fault_arms':[{'operation':op,'label':'%s_%s_job%d'%(campaign,op,job),'iterations':200,'warmup':20,'payload_seeds':[base+100000+i*20000+51,base+100000+i*20000+10051],'channel_seeds':[base+100000+i*20000+1000,base+100000+i*20000+11000]} for i,op in enumerate(('prepare','commit','complete'))],'ai_contract':{'raw_ms':45,'commit_ms':7,'guard_ms':2,'effective_ms':54,'static_slack_ms':53,'conditional_window_ms':58},'required_physical_gates':['single_unlaunched_token','retained_token_branch_exercised','broker_drained','system_safety','commit_admission_wall_bound'],'failure_rule':'stop on first failure; preserve failure; no post-hoc bound widening','source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rel,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

export SOFTWALL_SHARD_LABEL="${campaign}_ai45_job${SLURM_JOB_ID}" \
    SOFTWALL_SHARD_ITERATIONS=160 SOFTWALL_SHARD_WARMUP=20
export SOFTWALL_SHARD_PAYLOAD_SEED0=$((seed_base+51)) \
    SOFTWALL_SHARD_PAYLOAD_SEED1=$((seed_base+10051))
export SOFTWALL_SHARD_CHANNEL_SEED0=$((seed_base+1000)) \
    SOFTWALL_SHARD_CHANNEL_SEED1=$((seed_base+11000))
bash "$SOFTWALL_SCRIPTS/run_v15_ai45_single_token_arm.sh" 2>&1 | \
    tee "$result_dir/${SOFTWALL_SHARD_LABEL}.log"
ai_result="$result_dir/${SOFTWALL_SHARD_LABEL}_result.json"

fault_results=()
operations=(prepare commit complete)
for index in 0 1 2; do
    operation=${operations[$index]}
    base=$((seed_base + 100000 + index * 20000))
    label="${campaign}_${operation}_job${SLURM_JOB_ID}"
    export SOFTWALL_SHARD_LABEL="$label" SOFTWALL_SHARD_ITERATIONS=200 \
        SOFTWALL_SHARD_WARMUP=20
    export SOFTWALL_BROKER_CRASH_OPERATION="$operation" \
        SOFTWALL_BROKER_CRASH_NUMBER=30
    export SOFTWALL_BROKER_RPC_SOCKET_TIMEOUT_MS=5 \
        SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7
    export SOFTWALL_SHARD_PAYLOAD_SEED0=$((base+51)) \
        SOFTWALL_SHARD_PAYLOAD_SEED1=$((base+10051))
    export SOFTWALL_SHARD_CHANNEL_SEED0=$((base+1000)) \
        SOFTWALL_SHARD_CHANNEL_SEED1=$((base+11000))
    bash "$SOFTWALL_SCRIPTS/run_v15_single_token_control_fault_arm.sh" 2>&1 | \
        tee "$result_dir/${label}.log"
    fault_results+=("$result_dir/${label}_result.json")
done

python3 "$SOFTWALL_SCRIPTS/analyze_v15_single_token_node.py" \
    --protocol "$protocol" --ai "$ai_result" --faults "${fault_results[@]}" \
    --output "$result"
echo "$campaign V15 single-token requalification complete: $result"
