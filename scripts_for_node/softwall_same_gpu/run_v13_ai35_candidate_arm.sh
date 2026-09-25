#!/usr/bin/env bash

# V13 physical mechanism arm: two four-cell homes consume only context-64
# requests under the corrected 7 ms-per-RPC admission wall bound.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

iterations=${SOFTWALL_SHARD_ITERATIONS:-160}
warmup=${SOFTWALL_SHARD_WARMUP:-20}
label=${SOFTWALL_SHARD_LABEL:-v13_ai35_candidate_job${SLURM_JOB_ID}}
global_policy=${SOFTWALL_GLOBAL_POLICY:-global}
conv_bound_ms=${SOFTWALL_CONV_BOUND_MS:-25}
payload_seed0=${SOFTWALL_SHARD_PAYLOAD_SEED0:-24000051}
payload_seed1=${SOFTWALL_SHARD_PAYLOAD_SEED1:-24010051}
channel_seed0=${SOFTWALL_SHARD_CHANNEL_SEED0:-24001000}
channel_seed1=${SOFTWALL_SHARD_CHANNEL_SEED1:-24011000}
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state_root="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
barrier="$state_root/barrier"
protocol="$result_root/${label}_protocol.json"
result="$result_root/${label}_result.json"
broker_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/gb_${BASHPID}.sock"
broker_output="$raw/${label}_broker.json"
mkdir -p "$raw" "$state_root/h0" "$state_root/h1" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"
rm -rf "$barrier"

for home in 0 1; do
    rm -f "$state_root/h${home}"/cuda_ipc_* \
        "$raw/${label}_h${home}_controller.json" \
        "$raw/${label}_h${home}_worker0.json" \
        "$raw/${label}_h${home}_worker1.json" \
        "$raw/${label}_h${home}_background.json" \
        "$raw/${label}_h${home}_requalification.json" \
        "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/sh${home}_${BASHPID}.sock"
done
rm -f "$result" "$broker_socket" "$broker_output"

python3 - "$protocol" "$iterations" "$warmup" "$label" "$global_policy" "$conv_bound_ms" \
    "$payload_seed0" "$payload_seed1" "$channel_seed0" "$channel_seed1" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_qualified_control_bound_controller.py" \
    "$SOFTWALL_SCRIPTS/deadline_fault_contained_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/instrumented_deadline_global_trace_client_v2.py" \
    "$SOFTWALL_SCRIPTS/global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_TASK1/isca_v2/dart_runtime.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);iterations=int(sys.argv[2]);warmup=int(sys.argv[3]);label=sys.argv[4];policy=sys.argv[5];conv_bound_ms=float(sys.argv[6]);payload_seeds=[int(sys.argv[7]),int(sys.argv[8])];channel_seeds=[int(sys.argv[9]),int(sys.argv[10])]
keys=["/softwall/global_trace_fully_budgeted_fault_contained_controller.py","/softwall/global_trace_qualified_control_bound_controller.py","/softwall/deadline_fault_contained_global_trace_client.py","/softwall/instrumented_deadline_global_trace_client_v2.py","/softwall/global_trace_lease_broker.py","/softwall/same_request_ipc_worker.py","/softwall/trace_qwen_worker.py","/softwall_task1/isca_v2/dart_runtime.py","/softwall_task1/isca_v2/cuda_ipc_channel.py"]
paths=[Path(value) for value in sys.argv[11:]]
value={"schema":"softwall-v13-ai35-candidate-arm-protocol-v1","status":"frozen-before-run","label":label,"iterations":iterations,"warmup":warmup,"global_policy":policy,"payload_seeds":payload_seeds,"channel_seeds":channel_seeds,"trace":"/softwall_data/softwall_context64_mechanism_trace_v1.json","trace_sha256":"b14388c7356878458031d4e881a0510f42f1b4414493f876cd13d2b881979277","placement":{"gpu0":"four-cell home + two local NRx + Qwen","gpu1":"four-cell home + two local NRx + Qwen"},"shared_resources":"one serializable synthetic context64 request queue; recovery, NRx endpoint/IPC, Qwen execution and physical credits remain per-home","transaction_order":"observe NRx -> merge all ready recoveries in live-certificate order -> global hold -> local atomic recovery replan+AI lease -> global commit -> physical completion -> local retire+global complete","cuda_module_loading":"LAZY","period_ms":180,"deadline_ms":155,"nrx_bound_ms":45,"conv_bound_ms":conv_bound_ms,"guard_ms":2,"global_broker_rpc_timeout_ms":5,"global_broker_rpc_admission_bound_ms":7,"global_broker_transaction_budget_ms":21,"admission_ai_guard_ms":23,"target_ai_bound_ms":35,"effective_transaction_bound_ms":58,"endpoint_ring_depth":1,"rejected_recovery_before_exchange":False,"fault_every":5,"rpc_telemetry_required":True,"source_sha256":{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in zip(keys,paths)}}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out)
PY

shifter_home_aerial() {
    local gpu=$1; shift
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --volume="$SOFTWALL_ROOT/data/public/burstgpt:/softwall_burstgpt" \
        --volume="$SOFTWALL_ROOT/data/current:/softwall_data" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

shifter_home_qwen() {
    local gpu=$1; shift
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/softwall \
        --env=HF_HOME=/softwall_runtime/cache/huggingface \
        --env=TMPDIR=/softwall_runtime/tmp --env=HF_HUB_DISABLE_XET=1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

pids=""
worker_pids=""
controller_pids=""
broker_pid=""
cleanup() {
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    for _ in {1..30}; do
        alive=0
        for pid in $pids; do ! kill -0 "$pid" 2>/dev/null || alive=1; done
        [[ "$alive" -eq 0 ]] && break
        sleep 0.1
    done
    for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
    rm -f "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/sh0_${BASHPID}.sock" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/sh1_${BASHPID}.sock" "$broker_socket"
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

softwall_mps_configure
softwall_mps_stop
sleep 5
export SOFTWALL_MPS_GPU=0,1
softwall_mps_start
softwall_mps_assert

shifter_home_aerial 0 python3 /softwall/global_trace_lease_broker.py \
    --socket "$broker_socket" --trace /softwall_data/softwall_context64_mechanism_trace_v1.json \
    --trace-sha256 b14388c7356878458031d4e881a0510f42f1b4414493f876cd13d2b881979277 \
    --participants 2 --policy "$global_policy" --output "$broker_output" &
broker_pid=$!
pids="$pids $broker_pid"
for _ in {1..1000}; do [[ -S "$broker_socket" ]] && break; sleep 0.02; done
[[ -S "$broker_socket" ]] || { echo "global broker readiness timeout" >&2; exit 1; }

for home in 0 1; do
    shifter_home_aerial "$home" env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/s2_runner.py --policy conventional_only \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$raw/${label}_h${home}_requalification.json" \
        --iterations 30 --warmup 10 --cells 1 --period-ms 90 --deadline-ms 80 \
        --nrx-bound-ms 50 --conv-bound-ms 25 --commit-guard-ms 2 \
        --inject-failure-every 0 --seed "$((24001955 + home))" --snr-db -8.5 \
        --channel-seed-base "$((24009900 + home * 1000))"
done

tag="sh_${SLURM_JOB_ID}_${BASHPID}"
for home in 0 1; do
    for endpoint in 0 1; do
        shifter_home_aerial "$home" env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
            python3 /softwall/same_request_ipc_worker.py \
            --tag "${tag}_h${home}_c${endpoint}" --ipc-dir "$state_root/h${home}" \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --output "$raw/${label}_h${home}_worker${endpoint}.json" &
        pids="$pids $!"
        worker_pids="$worker_pids $!"
    done
    socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/sh${home}_${BASHPID}.sock"
    shifter_home_qwen "$home" env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/trace_qwen_worker.py --model Qwen/Qwen2.5-1.5B \
        --allowed-context-lengths 64 --batch-size 1 --warmup-per-length 3 \
        --socket "$socket" --output "$raw/${label}_h${home}_background.json" &
    pids="$pids $!"
    worker_pids="$worker_pids $!"
done

for home in 0 1; do
    socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/sh${home}_${BASHPID}.sock"
    for _ in {1..3000}; do [[ -S "$socket" ]] && break; sleep 0.05; done
    [[ -S "$socket" ]] || { echo "home $home Qwen readiness timeout" >&2; exit 1; }
done

for home in 0 1; do
    socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/sh${home}_${BASHPID}.sock"
    if [[ "$home" -eq 0 ]]; then payload=$payload_seed0; channel=$channel_seed0; else payload=$payload_seed1; channel=$channel_seed1; fi
    shifter_home_aerial "$home" env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/global_trace_qualified_control_bound_controller.py \
        --tag-prefix "${tag}_h${home}" --ipc-dir "$state_root/h${home}" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --socket "$socket" --trace /softwall_data/softwall_context64_mechanism_trace_v1.json \
        --trace-sha256 b14388c7356878458031d4e881a0510f42f1b4414493f876cd13d2b881979277 \
        --system softwall --ai-bound-map 64:35 \
        --output "$raw/${label}_h${home}_controller.json" --iterations "$iterations" \
        --cells 4 --nrx-endpoints 2 --warmup "$warmup" --period-ms 180 --deadline-ms 155 \
        --nrx-bound-ms 45 --conv-bound-ms "$conv_bound_ms" --commit-guard-ms 2 \
        --endpoint-timeout-ms 100 --ai-guard-ms 2 --ai-rpc-timeout-ms 120 \
        --seed "$payload" --snr-db -8.5 --channel-seed-base "$channel" \
        --gate-mode low_threshold --gate-threshold 1.9490545988082886 --gc-mode off \
        --alternate-admission-order --inject-correlated-failure-every 5 --fault-pattern mixed \
        --start-barrier-dir "$barrier" --start-barrier-member "$home" --start-barrier-participants 2 \
        --global-trace-broker "$broker_socket" --global-home-id "$home" \
        --global-broker-rpc-timeout-ms 5 --global-broker-rpc-bound-ms 7 &
    pids="$pids $!"
    controller_pids="$controller_pids $!"
done

failure=0
for pid in $controller_pids; do wait "$pid" || failure=1; done
[[ "$failure" -eq 0 ]] || { echo "a sharded-home process failed" >&2; exit 1; }
for pid in $worker_pids; do wait "$pid" || failure=1; done
[[ "$failure" -eq 0 ]] || { echo "a sharded-home worker failed" >&2; exit 1; }
shifter_home_aerial 0 python3 /softwall/global_trace_lease_broker.py \
    --socket "$broker_socket" --stop-only
wait "$broker_pid" || failure=1
[[ "$failure" -eq 0 ]] || { echo "global broker failed" >&2; exit 1; }
pids=""

python3 - "$protocol" "$raw" "$label" "$warmup" "$broker_output" "$result" <<'PY'
import hashlib,json,sys
from collections import Counter
from pathlib import Path
protocol,raw,label,warmup,broker_path,out=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),Path(sys.argv[5]),Path(sys.argv[6])
homes=[];artifacts=[protocol,broker_path];request_ids=[];controllers=[];rpc_by_home=[]
for home in (0,1):
 cpath=raw/f'{label}_h{home}_controller.json';c=json.loads(cpath.read_text());artifacts.append(cpath);controllers.append(c)
 telemetry=c.get('global_queue_status',{}).get('rpc_telemetry');rpc_by_home.append(telemetry)
 request_ids.extend(row['request_id'] for row in c['background_records'])
 admissions=Counter(row['endpoint_id'] for row in c['records'] if row['endpoint_id'] is not None)
 workers=[]
 for endpoint in (0,1):
  path=raw/f'{label}_h{home}_worker{endpoint}.json';w=json.loads(path.read_text());artifacts.append(path)
  expected=admissions[f'nrx{endpoint}']+(warmup+1)*2
  workers.append({'endpoint':endpoint,'completed_units':w['completed_units'],'expected_units':expected,'count_match':w['completed_units']==expected})
 bpath=raw/f'{label}_h{home}_background.json';rpath=raw/f'{label}_h{home}_requalification.json';artifacts.extend([bpath,rpath])
 homes.append({'home':home,'controller':str(cpath),'first_release_ns':c['first_release_ns'],'start_barrier':c['start_barrier'],'workers':workers,'rpc_telemetry':telemetry,'summary':{'records':len(c['records']),'correct_cells':c['correct_cells'],'nrx_commits':c['nrx_commits'],'conv_commits':c['conv_commits'],'atomic_exchange':c['joint_lease_retired_count'],'ai_timely_value_tokens':c['ai_timely_value_tokens']},'safety':{'deadline':c['deadline_misses']==0,'nrx_bound':c['nrx_bound_violations']==0,'conv_bound':c['conv_bound_violations']==0 and c['conv_path_bound_violations']==0,'ai_bound':c['background_budget_violations']==0 and c['background_horizon_violations']==0,'faults':not c['endpoint_faults'] and not c['background_faults'] and not c.get('global_queue_status',{}).get('faults',[]),'credits':c['fallback_calendar_final']['outstanding']==0 and c['fallback_calendar_final']['joint_leases_outstanding']==0 and all(v['outstanding']==0 for v in c['endpoint_final'].values())}})
candidate_branches=0;candidate_exchanges=0;candidate_margins=[];max_admitted=0;p=json.loads(protocol.read_text())
for c in controllers:
 byindex={i:[] for i in range(p['iterations'])}
 for row in c['records']:byindex[row['index']].append(row)
 bg={row['request_id']:row for row in c['background_records']}
 for decision in c['recovery_decisions']:
  rows=byindex[decision['release_index']];admitted=[row for row in rows if row['admitted']];success=[row for row in admitted if row['nrx_commit']];rejected=[row for row in rows if not row['admitted']]
  max_admitted=max(max_admitted,len(admitted));branch=len(admitted)==2 and len(success)==2 and len(rejected)==2;candidate_branches+=int(branch)
  record=bg.get(decision.get('selected_request_id'))
  if branch and decision.get('ai_lease') and record is not None and record['bound_ms']==35.0:
   candidate_exchanges+=1;candidate_margins.append((record['horizon_ns']-record['returned_ns']-round(c['admission_ai_guard_ms']*1e6))/1e6)
broker=json.loads(broker_path.read_text());summary=broker['summary']
rpc_records=[row for telemetry in rpc_by_home if telemetry is not None for row in telemetry.get('records',[])]
rpc_max=max((row['elapsed_ms'] for row in rpc_records),default=None)
rpc_control={'records':len(rpc_records),'by_operation':dict(Counter(row['operation'] for row in rpc_records)),'max_elapsed_ms':rpc_max,'faulted':sum(row['faulted'] for row in rpc_records),'admission_bound_exceeded':sum(row['elapsed_ms']>p['global_broker_rpc_admission_bound_ms'] for row in rpc_records)}
gates={'two_homes':len(homes)==2,'same_first_release':homes[0]['first_release_ns']==homes[1]['first_release_ns'],'eight_cells':sum(x['summary']['records'] for x in homes)==8*p['iterations'],'worker_count_match':all(w['count_match'] for h in homes for w in h['workers']),'system_safety':all(all(h['safety'].values()) for h in homes),'broker_homes':summary['registered_homes']==[0,1] and summary['finalized_homes']==[0,1],'global_unique_requests':len(request_ids)==len(set(request_ids))==summary['timely_requests'],'broker_drained':summary['outstanding_tokens']==0 and summary['duplicate_commit_count']==0,'both_homes_ai':all(h['summary']['ai_timely_value_tokens']>0 for h in homes),'physical_ring_depth_conformance':max_admitted<=2*p['endpoint_ring_depth'],'candidate_branch_coverage':candidate_branches>0,'ai35_candidate_exchange_observed':candidate_exchanges>0,'candidate_horizon_margin_nonnegative':bool(candidate_margins) and min(candidate_margins)>=0,'control_budget_accounted':all(c['global_broker_transaction_budget_ms']==21 and c['admission_ai_guard_ms']==23 for c in controllers),'rpc_telemetry_present':all(value is not None and value.get('schema')=='softwall-broker-rpc-telemetry-v2' for value in rpc_by_home),'rpc_prepare_commit_complete_coverage':{'prepare','commit','complete'}<=set(row['operation'] for row in rpc_records),'rpc_admission_wall_bound':bool(rpc_records) and rpc_control['faulted']==0 and rpc_control['admission_bound_exceeded']==0 and rpc_max<=p['global_broker_rpc_admission_bound_ms']}
value={'schema':'softwall-v13-ai35-candidate-arm-result-v1','protocol':str(protocol),'broker':str(broker_path),'broker_summary':summary,'homes':homes,'candidate_evidence':{'branches':candidate_branches,'exchanges':candidate_exchanges,'minimum_reserved_horizon_margin_ms':min(candidate_margins) if candidate_margins else None,'maximum_admitted_nrx_per_home_release':max_admitted},'control_evidence':rpc_control,'gates':gates,'all_pass':all(gates.values()),'artifact_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},'scope':'synthetic context64-only mechanism qualification for the exchange-only AI35 plus corrected 21 ms control budget and 2 ms completion guard across two disjoint four-cell recovery homes'}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out);print(json.dumps(gates,indent=2))
if not value['all_pass']:raise SystemExit('v13 AI35 candidate arm failed')
PY

cleanup
trap - EXIT INT TERM
echo "v13 AI35 candidate arm complete: $result"
