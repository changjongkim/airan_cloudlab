#!/usr/bin/env bash

# C161 phase-2: four independent physical event/channel-fault arms.

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

label=${SOFTWALL_C161_LABEL:-c161_phase2_job${SLURM_JOB_ID}}
campaign=${SOFTWALL_C161_CAMPAIGN:-development}
seed_base=${SOFTWALL_C161_SEED_BASE:-38000000}
excluded_nodes=${SOFTWALL_C161_EXCLUDED_NODES:-nid002288,nid002100,nid001109,nid001085,nid001064,nid001124,nid001177,nid001308,nid001204,nid001632,nid001824,nid001025}
arm_order=${SOFTWALL_C161_ARM_ORDER:-stale_duplicate_nrx,post_fence_reply_delay,pre_fence_channel_loss,stale_duplicate_recovery}
period_ms=${SOFTWALL_C161_PERIOD_MS:-180}
warmup=${SOFTWALL_C161_WARMUP:-10}
release_lead_ms=${SOFTWALL_C161_RELEASE_LEAD_MS:-3000}
prespec="$SOFTWALL_ROOT/results/softwall_multigpu/confirm159_experiment_prespec_v1.json"
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
raw="$result_root/raw"
state_root="$SOFTWALL_ROOT/run_state/softwall_multigpu"
protocol="$result_root/${label}_protocol.json"
campaign_result="$result_root/${label}_result.json"
arm_table="$state_root/${label}_arms.tsv"

rm -f "$protocol" "$campaign_result" "$arm_table"
mkdir -p "$raw" "$state_root" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"

python3.11 "$SOFTWALL_SCRIPTS/build_c161_phase2_protocol.py" \
    --output "$protocol" --label "$label" --campaign "$campaign" \
    --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
    --result-root "$result_root" --raw-dir "$raw" --state-root "$state_root" \
    --prespec "$prespec" --seed-base "$seed_base" \
    --excluded-nodes "$excluded_nodes" --arm-order "$arm_order" \
    --period-ms "$period_ms" --warmup "$warmup" \
    --release-lead-ms "$release_lead_ms"

current_node=$(hostname)
case ",$excluded_nodes," in
    *",$current_node,"*)
        echo "C161 phase-2 node $current_node is in the frozen exclusion set" >&2
        exit 3
        ;;
esac

python3.11 - "$protocol" "$arm_table" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
with open(sys.argv[2],'w',encoding='utf-8') as f:
    for a in p['arms']:
        f.write('\t'.join(map(str,(
            a['index'],a['arm'],a['label'],a['iterations'],a['state_dir'],
            a['peer_spec'],a['peer_tsv'],a['schedule_file'],a['completion_dir'],
            a['qwen_socket_name'],a['qwen_output'],a['nrx_output'],
            a['coordinator_output'],a['inventory'],a['result'],
        )))+'\n')
PY

shifter_all() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_TASK1:/softwall_task1" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/softwall:/softwall_task1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=0,1,2,3 "$@"
}

shifter_qwen_gpu2() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$SOFTWALL_SCRIPTS:/softwall" \
        --volume="$SOFTWALL_RUNTIME:/softwall_runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" \
        --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/softwall \
        --env=HF_HOME=/softwall_runtime/cache/huggingface \
        --env=TMPDIR=/softwall_runtime/tmp --env=HF_HUB_DISABLE_XET=1 \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES=2 "$@"
}

pids=""
cleanup() {
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    for pid in $pids; do wait "$pid" 2>/dev/null || true; done
    pids=""
    softwall_mps_stop || true
}
trap cleanup EXIT INT TERM

run_arm() {
    local arm_index=$1 arm=$2 arm_label=$3 iterations=$4 state=$5
    local peer_spec=$6 peer_tsv=$7 schedule_file=$8 completion_dir=$9
    shift 9
    local qwen_socket_name=$1 qwen_output=$2 nrx_output=$3 coordinator_output=$4
    local inventory=$5 result=$6
    local qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/$qwen_socket_name"
    mkdir -p "$state" "$completion_dir"
    rm -f "$qwen_socket" "$qwen_output" "$nrx_output" "$coordinator_output" "$result"
    rm -f "$completion_dir"/*.json
    nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$inventory"

    softwall_mps_configure
    softwall_mps_stop
    export SOFTWALL_MPS_GPU=0,1,2,3
    softwall_mps_start
    softwall_mps_assert

    shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/c161_phase2_qwen_worker.py \
        --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 16,32,64,128,256,512 \
        --batch-size 1 --warmup-per-length 3 --socket "$qwen_socket" \
        --completion-dir "$completion_dir" --fault-response-delay-ms 100 \
        --output "$qwen_output" >"$raw/${arm_label}_qwen.log" 2>&1 &
    local qwen_pid=$!
    pids="$qwen_pid"
    for _ in {1..3600}; do
        [[ -S "$qwen_socket" ]] && break
        if ! kill -0 "$qwen_pid" 2>/dev/null; then
            wait "$qwen_pid" || true
            echo "C161 phase-2 Qwen worker exited before readiness: $arm" >&2
            return 1
        fi
        sleep 0.05
    done
    [[ -S "$qwen_socket" ]] || { echo "C161 phase-2 Qwen readiness timeout" >&2; return 1; }

    while IFS=$'\t' read -r home request source receiver_seed channel_seed_base snr \
            nrx_tag recovery_tag ready_file decision_control owner_output; do
        [[ -n "$nrx_tag" ]] || continue
        shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/c159_prestaged_owner.py \
            --home-id "$home" --request-id "$request" --source-device "$source" \
            --nrx-tag "$nrx_tag" --recovery-tag "$recovery_tag" \
            --ipc-dir "$state" --schedule-file "$schedule_file" \
            --decision-control "$decision_control" --ready-file "$ready_file" \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --output "$owner_output" --iterations "$iterations" \
            --receiver-seed "$receiver_seed" --channel-seed-base "$channel_seed_base" \
            --snr-db "$snr" >"${owner_output%.json}.log" 2>&1 &
        pids="$pids $!"
    done < "$peer_tsv"

    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/c159_persistent_nrx_worker.py \
        --peer-spec "$peer_spec" --ipc-dir "$state" --schedule-file "$schedule_file" \
        --iterations "$iterations" --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$nrx_output" --destination-device 3 \
        >"$raw/${arm_label}_nrx_worker.log" 2>&1 &
    pids="$pids $!"

    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/c161_phase2_coordinator.py \
        --peer-spec "$peer_spec" --ipc-dir "$state" --schedule-file "$schedule_file" \
        --qwen-socket "$qwen_socket" --completion-dir "$completion_dir" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$coordinator_output" --iterations "$iterations" \
        --period-ms "$period_ms" --destination-device 2 --warmup "$warmup" \
        --release-lead-ms "$release_lead_ms" --fault-arm "$arm" \
        --event-fault-interval 6 --event-fault-target 10 \
        --terminal-fault-after-epoch 5 --qwen-response-timeout-margin-ms 8 \
        >"$raw/${arm_label}_coordinator.log" 2>&1 &
    pids="$pids $!"

    local failure=0
    for pid in $pids; do wait "$pid" || failure=1; done
    pids=""
    softwall_mps_stop || true
    [[ "$failure" -eq 0 ]] || {
        echo "C161 phase-2 process failure: $arm; artifacts preserved" >&2
        return 1
    }

    python3.11 "$SOFTWALL_SCRIPTS/analyze_c161_phase2_arm.py" \
        --protocol "$protocol" --arm-index "$arm_index" \
        --coordinator "$coordinator_output" --nrx-worker "$nrx_output" \
        --qwen "$qwen_output" --inventory "$inventory" \
        --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
        --output "$result"
    python3.11 - "$result" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['all_pass'] else 1)
PY
}

while IFS=$'\t' read -r arm_index arm arm_label iterations state peer_spec peer_tsv \
        schedule_file completion_dir qwen_socket_name qwen_output nrx_output \
        coordinator_output inventory result; do
    run_arm "$arm_index" "$arm" "$arm_label" "$iterations" "$state" \
        "$peer_spec" "$peer_tsv" "$schedule_file" "$completion_dir" \
        "$qwen_socket_name" "$qwen_output" "$nrx_output" \
        "$coordinator_output" "$inventory" "$result"
done < "$arm_table"

python3.11 "$SOFTWALL_SCRIPTS/analyze_c161_phase2_campaign.py" \
    --protocol "$protocol" --scripts-root "$SOFTWALL_SCRIPTS" \
    --task1-root "$SOFTWALL_TASK1" --output "$campaign_result"
python3.11 - "$campaign_result" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['all_pass'] else 1)
PY

cleanup
trap - EXIT INT TERM
echo "C161 phase-2 campaign complete: $campaign_result"
