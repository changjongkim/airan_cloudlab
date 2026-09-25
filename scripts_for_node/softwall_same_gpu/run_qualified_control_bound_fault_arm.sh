#!/usr/bin/env bash

# C139 arm: separate the 5 ms socket timeout from a 7 ms wall-time charge.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

iterations=${SOFTWALL_SHARD_ITERATIONS:-340}
warmup=${SOFTWALL_SHARD_WARMUP:-20}
label=${SOFTWALL_SHARD_LABEL:-qualified_control_bound_fault_job${SLURM_JOB_ID}}
global_policy=${SOFTWALL_GLOBAL_POLICY:-global}
conv_bound_ms=${SOFTWALL_CONV_BOUND_MS:-25}
rpc_socket_timeout_ms=${SOFTWALL_BROKER_RPC_SOCKET_TIMEOUT_MS:-5}
rpc_admission_bound_ms=${SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS:-7}
crash_operation=${SOFTWALL_BROKER_CRASH_OPERATION:?set prepare, commit, or complete}
crash_number=${SOFTWALL_BROKER_CRASH_NUMBER:-30}
case "$crash_operation" in prepare|commit|complete) ;; *) echo "invalid crash operation" >&2; exit 2;; esac
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
broker_output="$raw/${label}_broker_unexpected.json"
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

python3 - "$protocol" "$iterations" "$warmup" "$label" "$global_policy" "$conv_bound_ms" "$crash_operation" "$crash_number" "$rpc_socket_timeout_ms" "$rpc_admission_bound_ms" \
    "$payload_seed0" "$payload_seed1" "$channel_seed0" "$channel_seed1" \
    "$SOFTWALL_SCRIPTS/global_trace_fully_budgeted_fault_contained_controller.py" \
    "$SOFTWALL_SCRIPTS/global_trace_qualified_control_bound_controller.py" \
    "$SOFTWALL_SCRIPTS/deadline_fault_contained_global_trace_client.py" \
    "$SOFTWALL_SCRIPTS/instrumented_deadline_global_trace_client_v2.py" \
    "$SOFTWALL_SCRIPTS/control_point_crash_global_trace_lease_broker.py" \
    "$SOFTWALL_SCRIPTS/same_request_ipc_worker.py" \
    "$SOFTWALL_SCRIPTS/trace_qwen_worker.py" \
    "$SOFTWALL_TASK1/isca_v2/dart_runtime.py" \
    "$SOFTWALL_TASK1/isca_v2/cuda_ipc_channel.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);iterations=int(sys.argv[2]);warmup=int(sys.argv[3]);label=sys.argv[4];policy=sys.argv[5];conv_bound_ms=float(sys.argv[6]);operation=sys.argv[7];number=int(sys.argv[8]);socket_timeout_ms=float(sys.argv[9]);rpc_bound_ms=float(sys.argv[10]);payload_seeds=[int(sys.argv[11]),int(sys.argv[12])];channel_seeds=[int(sys.argv[13]),int(sys.argv[14])]
keys=["/softwall/global_trace_fully_budgeted_fault_contained_controller.py","/softwall/global_trace_qualified_control_bound_controller.py","/softwall/deadline_fault_contained_global_trace_client.py","/softwall/instrumented_deadline_global_trace_client_v2.py","/softwall/control_point_crash_global_trace_lease_broker.py","/softwall/same_request_ipc_worker.py","/softwall/trace_qwen_worker.py","/softwall_task1/isca_v2/dart_runtime.py","/softwall_task1/isca_v2/cuda_ipc_channel.py"]
paths=[Path(value) for value in sys.argv[15:]]
value={"schema":"softwall-qualified-control-bound-fault-arm-protocol-v1","status":"frozen-before-run","label":label,"iterations":iterations,"warmup":warmup,"global_policy":policy,"payload_seeds":payload_seeds,"channel_seeds":channel_seeds,"placement":{"gpu0":"four-cell home + two local NRx + Qwen","gpu1":"four-cell home + two local NRx + Qwen"},"shared_resources":"one serializable BurstGPT request queue; recovery, NRx endpoint/IPC, Qwen execution and physical credits remain per-home","transaction_order":"observe NRx -> certificate order -> bounded global prepare/hold -> local atomic replan+AI lease -> bounded global commit -> physical AI -> bounded global complete; broker fail-stop disables global AI clients without retry; local RAN certificates continue","fault_injection":{"home":0,"operation":operation,"operation_number":number,"point":"process calls os._exit(86) immediately after the configured state transition, before reply; attempted-RPC wall time includes the detecting call and excludes post-quarantine no-ops","expected_exit_code":86,"broker_rpc_socket_timeout_ms":socket_timeout_ms},"cuda_module_loading":"LAZY","period_ms":180,"deadline_ms":155,"nrx_bound_ms":45,"conv_bound_ms":conv_bound_ms,"guard_ms":2,"global_broker_rpc_timeout_ms":socket_timeout_ms,"global_broker_rpc_admission_bound_ms":rpc_bound_ms,"global_broker_transaction_budget_ms":3*rpc_bound_ms,"admission_ai_guard_ms":3*rpc_bound_ms+2,"fault_every":5,"rpc_telemetry_required":True,"source_sha256":{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in zip(keys,paths)}}
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

shifter_home_aerial 0 python3 /softwall/control_point_crash_global_trace_lease_broker.py \
    --socket "$broker_socket" --trace /softwall_burstgpt/softwall_burst60_prefill_trace.json \
    --trace-sha256 da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d \
    --participants 2 --policy "$global_policy" --crash-after-operation "$crash_operation" \
    --crash-after-home 0 --crash-after-number "$crash_number" --output "$broker_output" &
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
        --allowed-context-lengths 16,32,64,128,256,512 --batch-size 1 --warmup-per-length 3 \
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
        --socket "$socket" --trace /softwall_burstgpt/softwall_burst60_prefill_trace.json \
        --trace-sha256 da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d \
        --system softwall --ai-bound-map 16:35,32:35,64:35,128:40,256:65,512:75 \
        --output "$raw/${label}_h${home}_controller.json" --iterations "$iterations" \
        --cells 4 --nrx-endpoints 2 --warmup "$warmup" --period-ms 180 --deadline-ms 155 \
        --nrx-bound-ms 45 --conv-bound-ms "$conv_bound_ms" --commit-guard-ms 2 \
        --endpoint-timeout-ms 100 --ai-guard-ms 2 --ai-rpc-timeout-ms 120 \
        --seed "$payload" --snr-db -8.5 --channel-seed-base "$channel" \
        --gate-mode low_threshold --gate-threshold 1.9490545988082886 --gc-mode off \
        --alternate-admission-order --inject-correlated-failure-every 5 --fault-pattern mixed \
        --start-barrier-dir "$barrier" --start-barrier-member "$home" --start-barrier-participants 2 \
        --global-trace-broker "$broker_socket" --global-home-id "$home" \
        --global-broker-rpc-timeout-ms "$rpc_socket_timeout_ms" \
        --global-broker-rpc-bound-ms "$rpc_admission_bound_ms" &
    pids="$pids $!"
    controller_pids="$controller_pids $!"
done

failure=0
for pid in $controller_pids; do wait "$pid" || failure=1; done
[[ "$failure" -eq 0 ]] || { echo "a sharded-home process failed" >&2; exit 1; }
for pid in $worker_pids; do wait "$pid" || failure=1; done
[[ "$failure" -eq 0 ]] || { echo "a sharded-home worker failed" >&2; exit 1; }
broker_status=0
wait "$broker_pid" || broker_status=$?
[[ "$broker_status" -eq 86 ]] || { echo "broker did not exit at the injected fail-stop point: $broker_status" >&2; exit 1; }
[[ ! -f "$broker_output" ]] || { echo "broker unexpectedly reached graceful output" >&2; exit 1; }
pids=""

python3 - "$protocol" "$raw" "$label" "$warmup" "$broker_status" "$result" "$crash_operation" <<'PY'
import hashlib,json,sys
from collections import Counter
from pathlib import Path
protocol,raw,label,warmup,broker_status,out,operation=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),Path(sys.argv[6]),sys.argv[7]
homes=[];artifacts=[protocol];request_ids=[];controllers=[];rpc_by_home=[]
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
 homes.append({'home':home,'controller':str(cpath),'first_release_ns':c['first_release_ns'],'start_barrier':c['start_barrier'],'workers':workers,'global_queue_status':c['global_queue_status'],'rpc_telemetry':telemetry,'summary':{'records':len(c['records']),'correct_cells':c['correct_cells'],'nrx_commits':c['nrx_commits'],'conv_commits':c['conv_commits'],'atomic_exchange':c['joint_lease_retired_count'],'ai_timely_value_tokens':c['ai_timely_value_tokens'],'last_release_index':max(row['index'] for row in c['records'])},'safety':{'deadline':c['deadline_misses']==0,'nrx_bound':c['nrx_bound_violations']==0,'conv_bound':c['conv_bound_violations']==0 and c['conv_path_bound_violations']==0,'ai_bound':c['background_budget_violations']==0 and c['background_horizon_violations']==0,'physical_faults':not c['endpoint_faults'] and not c['background_faults'],'credits':c['fallback_calendar_final']['outstanding']==0 and c['fallback_calendar_final']['joint_leases_outstanding']==0 and all(v['outstanding']==0 for v in c['endpoint_final'].values())}})
p=json.loads(protocol.read_text())
operational_faults=[]
for c in controllers:
 operational_faults.append([row for row in c['global_queue_status']['faults'] if not row['operation'].startswith('metadata_')])
target_faults=[row for row in operational_faults[0] if row['operation']==operation]
target_request_id=target_faults[0]['request_id'] if len(target_faults)==1 else None
fault_detected_ns=min(row['detected_ns'] for faults in operational_faults for row in faults)
radio_after=[sum(row['commit_return_ns']>fault_detected_ns for row in c['records']) for c in controllers]
rpc_records=[row for telemetry in rpc_by_home if telemetry is not None for row in telemetry.get('records',[])]
rpc_ops_by_home=[set(row['operation'] for row in telemetry.get('records',[])) if telemetry is not None else set() for telemetry in rpc_by_home]
rpc_max=max((row['elapsed_ms'] for row in rpc_records),default=None)
faulted_records=[row for row in rpc_records if row['faulted']]
target_fault_records=[row for row in rpc_by_home[0].get('records',[]) if row['operation']==operation and row['faulted']] if rpc_by_home[0] is not None else []
rpc_control={'records':len(rpc_records),'by_operation':dict(Counter(row['operation'] for row in rpc_records)),'max_elapsed_ms':rpc_max,'socket_timeout_elapsed_exceeded':sum(row['wall_timeout_exceeded'] for row in rpc_records),'admission_bound_exceeded':sum(row['elapsed_ms']>p['global_broker_rpc_admission_bound_ms'] for row in rpc_records),'faulted':len(faulted_records),'faulted_by_operation':dict(Counter(row['operation'] for row in faulted_records)),'target_fault_records':len(target_fault_records)}
gates={
 'two_homes':len(homes)==2,
 'same_first_release':homes[0]['first_release_ns']==homes[1]['first_release_ns'],
 'full_radio_continuity':all(h['summary']['records']==4*p['iterations'] and h['summary']['last_release_index']==p['iterations']-1 for h in homes),
 'worker_count_match':all(w['count_match'] for h in homes for w in h['workers']),
 'system_safety':all(all(h['safety'].values()) for h in homes),
 'control_budget_accounted':all(c['global_broker_transaction_budget_ms']==p['global_broker_transaction_budget_ms'] and c['admission_ai_guard_ms']==p['admission_ai_guard_ms'] for c in controllers),
 'expected_broker_exit':broker_status==86,
 'home0_target_operation_fault':len(target_faults)==1,
 'both_homes_fail_closed':all(not c['global_queue_status']['enabled'] and operational_faults[index] for index,c in enumerate(controllers)),
 'both_homes_radio_after_detection':all(value>0 for value in radio_after),
 'target_execution_semantics':(
     target_request_id is None if operation=='prepare'
     else request_ids.count(target_request_id)==0 if operation=='commit'
     else request_ids.count(target_request_id)==1
 ),
 'no_duplicate_execution':len(request_ids)==len(set(request_ids)),
 'both_homes_executed_ai':all(h['summary']['ai_timely_value_tokens']>0 for h in homes),
 'rpc_telemetry_present':len(rpc_by_home)==2 and all(value is not None and value.get('schema')=='softwall-broker-rpc-telemetry-v2' for value in rpc_by_home),
 'socket_and_admission_bounds_distinct':p['global_broker_rpc_timeout_ms']<p['global_broker_rpc_admission_bound_ms'] and all(c['global_queue_status']['broker_rpc_socket_timeout_ms']==p['global_broker_rpc_timeout_ms'] and c['global_queue_status']['broker_rpc_admission_bound_ms']==p['global_broker_rpc_admission_bound_ms'] for c in controllers),
 'rpc_prepare_commit_complete_each_home':all({'prepare','commit','complete'}<=ops for ops in rpc_ops_by_home),
 'target_fault_call_recorded_once':len(target_fault_records)==1,
 'fault_records_reconcile':len(faulted_records)==sum(len(value) for value in operational_faults),
 'rpc_admission_wall_bound':bool(rpc_records) and rpc_control['admission_bound_exceeded']==0 and rpc_max<=p['global_broker_rpc_admission_bound_ms'],
}
value={'schema':'softwall-qualified-control-bound-fault-arm-result-v1','protocol':str(protocol),'crash_operation':operation,'broker_exit_status':broker_status,'fault_detected_ns':fault_detected_ns,'target_request_id':target_request_id,'operational_global_queue_faults':operational_faults,'radio_records_after_fault_detection':radio_after,'homes':homes,'control_evidence':rpc_control,'gates':gates,'all_pass':all(gates.values()),'artifact_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},'scope':f'5 ms socket timeout separated from a prospective 7 ms admission wall bound, charged three times, for the post-apply home0 {operation}30 fail-stop; all global AI clients fail closed while local RAN certificates continue'}
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(out);print(json.dumps(gates,indent=2))
if not value['all_pass']:raise SystemExit('qualified control-bound fault arm failed')
PY
cleanup
trap - EXIT INT TERM
echo "qualified control-bound fault arm complete: $result"
