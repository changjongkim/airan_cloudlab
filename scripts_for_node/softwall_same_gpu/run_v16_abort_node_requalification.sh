#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
require_allocation

campaign=${SOFTWALL_V16_CAMPAIGN:?set campaign name, for example confirm147}
seed_base=${SOFTWALL_V16_SEED_BASE:?set integer seed base}
excluded_csv=${SOFTWALL_V16_EXCLUDED_NODES:-nid001245,nid002817,nid002688,nid001781,nid001049,nid001177,nid001144,nid001252,nid001372,nid001361,nid002049,nid001244,nid003417}
node=$(hostname -s)
IFS=',' read -r -a excluded_nodes <<< "$excluded_csv"
for excluded in "${excluded_nodes[@]}"; do
    [[ "$node" != "$excluded" ]] || {
        echo "$campaign requires a new node" >&2
        exit 2
    }
done

result_dir="$SOFTWALL_ROOT/results/softwall_multigpu"
protocol="$result_dir/${campaign}_v16_abort_protocol.json"
result="$result_dir/${campaign}_v16_abort_result.json"

shifter_gpu env SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7 python3 -m unittest \
    test_four_point_crash_broker \
    test_single_token_pipelined_global_trace_client \
    test_global_trace_single_token_pipelined_controller

python3 - "$protocol" "$node" "$SLURM_JOB_ID" "$campaign" "$seed_base" \
    "$excluded_csv" \
    "$SOFTWALL_SCRIPTS/control_point_crash_global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/four_point_crash_global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/global_trace_single_token_pipelined_controller.py" \
    "$SOFTWALL_SCRIPTS/single_token_pipelined_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/run_v16_abort_fault_arm.sh" \
    "$SOFTWALL_SCRIPTS/analyze_v16_abort_node.py" \
    "$SOFTWALL_SCRIPTS/test_four_point_crash_broker.py" \
    "$SOFTWALL_SCRIPTS/run_v16_abort_node_requalification.sh" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);node=sys.argv[2];job=int(sys.argv[3]);campaign=sys.argv[4];base=int(sys.argv[5]);excluded=[x for x in sys.argv[6].split(',') if x];paths=[Path(x) for x in sys.argv[7:]]
root=Path('/pscratch/sd/s/sgkim/kcj/airan_cloudlab');rel=[str(p.relative_to(root)) for p in paths]
value={'schema':'softwall-v16-abort-node-protocol-v1','status':'frozen-before-arm','campaign':campaign,'allocation_node':node,'job_id':job,'excluded_nodes':excluded,'ownership_contract':'at-most-one-staged-or-offered-token-per-home','fault_arm':{'operation':'abort','operation_number':3,'label':'%s_abort_job%d'%(campaign,job),'iterations':200,'warmup':20,'payload_seeds':[base+51,base+10051],'channel_seeds':[base+1000,base+11000]},'required_physical_gates':['abort_arm_pass','target_not_executed','both_homes_radio_continue','all_four_operations_observed','single_unlaunched_token','retained_token_branch_exercised'],'failure_rule':'stop on failure; preserve failure; no post-hoc bound widening','source_sha256':{r:hashlib.sha256(p.read_bytes()).hexdigest() for r,p in zip(rel,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

label="${campaign}_abort_job${SLURM_JOB_ID}"
export SOFTWALL_SHARD_LABEL="$label" SOFTWALL_SHARD_ITERATIONS=200 \
    SOFTWALL_SHARD_WARMUP=20
export SOFTWALL_BROKER_CRASH_OPERATION=abort \
    SOFTWALL_BROKER_CRASH_NUMBER=3
export SOFTWALL_BROKER_RPC_SOCKET_TIMEOUT_MS=5 \
    SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS=7
export SOFTWALL_SHARD_PAYLOAD_SEED0=$((seed_base+51)) \
    SOFTWALL_SHARD_PAYLOAD_SEED1=$((seed_base+10051))
export SOFTWALL_SHARD_CHANNEL_SEED0=$((seed_base+1000)) \
    SOFTWALL_SHARD_CHANNEL_SEED1=$((seed_base+11000))
bash "$SOFTWALL_SCRIPTS/run_v16_abort_fault_arm.sh" 2>&1 | \
    tee "$result_dir/${label}.log"
arm_result="$result_dir/${label}_result.json"

python3 "$SOFTWALL_SCRIPTS/analyze_v16_abort_node.py" \
    --protocol "$protocol" --arm "$arm_result" --output "$result"
echo "$campaign V16 abort requalification complete: $result"
