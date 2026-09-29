#!/usr/bin/env bash

# C176 bursty-arrival campaign on the C159-Q2 P180/D155 radio path.
#
# Each run pairs one arrival pattern with one admission policy and one
# execution mode (natural or bound-padded). Every run uses the same radio seeds,
# so the NeuralRx outcome sequence is shared across policies.
#
#   SOFTWALL_C176_RUNS="gamma_cv1:backstop:natural gamma_cv1:idle_time:padded"

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh
require_allocation

runs=${SOFTWALL_C176_RUNS:?set SOFTWALL_C176_RUNS to pattern:policy:mode entries}
campaign=${SOFTWALL_C176_CAMPAIGN:-development}
seed_base=${SOFTWALL_C176_SEED_BASE:-36000000}
excluded_nodes=${SOFTWALL_C176_EXCLUDED_NODES:-nid002288,nid002100,nid001109,nid001085,nid001064,nid001124,nid001177,nid001308}
iterations=${SOFTWALL_C176_ITERATIONS:-600}
slo_ms=${SOFTWALL_C176_SLO_MS:-200}
max_leases=${SOFTWALL_C176_MAX_LEASES:-4}
period_ms=${SOFTWALL_C176_PERIOD_MS:-180}
warmup=${SOFTWALL_C176_WARMUP:-10}
release_lead_ms=${SOFTWALL_C176_RELEASE_LEAD_MS:-3000}
prefix=${SOFTWALL_C176_PREFIX:-c176}
prespec="$SOFTWALL_ROOT/results/softwall_multigpu/confirm159_experiment_prespec_v1.json"
result_root="$SOFTWALL_ROOT/results/softwall_multigpu"
trace_dir="$result_root/c176_traces"
raw="$result_root/raw"
mkdir -p "$raw" "$SOFTWALL_ROOT/mps/$SLURM_JOB_ID"

current_node=$(hostname)
case ",$excluded_nodes," in
    *",$current_node,"*)
        echo "C176 node $current_node is in the frozen exclusion set" >&2
        exit 3
        ;;
esac

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

run_one() {
    local pattern=$1 policy=$2 mode=$3
    local label="${prefix}_${pattern}_${policy}_${mode}_j${SLURM_JOB_ID}"
    local state="$SOFTWALL_ROOT/run_state/softwall_multigpu/$label"
    local protocol="$result_root/${label}_radio_protocol.json"
    local peer_spec="$result_root/${label}_peer_spec.json"
    local peer_tsv="$state/peers.tsv"
    local schedule_file="$state/schedule.json"
    local qwen_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c176.sock"
    local trace="$trace_dir/c176_trace_${pattern}.json"
    local padded_arg=()
    [[ -f "$trace" ]] || { echo "missing trace $trace" >&2; return 2; }
    [[ "$mode" == natural || "$mode" == padded ]] || { echo "bad mode $mode" >&2; return 2; }
    [[ "$mode" == padded ]] && padded_arg+=(--padded)
    [[ ! -e "$raw/${label}_coordinator.json" ]] || { echo "refusing to overwrite $label" >&2; return 2; }

    rm -rf "$state"
    mkdir -p "$state"
    rm -f "$qwen_socket"
    nvidia-smi --query-gpu=index,name,uuid,mig.mode.current --format=csv > "$raw/${label}_gpu_inventory.csv"
    python3.11 "$SOFTWALL_SCRIPTS/build_confirm159_q2_protocol.py" \
        --output "$protocol" --peer-spec "$peer_spec" --peer-tsv "$peer_tsv" \
        --label "$label" --campaign "$campaign" \
        --scripts-root "$SOFTWALL_SCRIPTS" --task1-root "$SOFTWALL_TASK1" \
        --raw-dir "$raw" --state-dir "$state" --prespec "$prespec" \
        --seed-base "$seed_base" --excluded-nodes "$excluded_nodes" \
        --iterations "$iterations" --period-ms "$period_ms" \
        --warmup "$warmup" --release-lead-ms "$release_lead_ms" >/dev/null

    softwall_mps_configure
    softwall_mps_stop
    export SOFTWALL_MPS_GPU=0,1,2,3
    softwall_mps_start
    softwall_mps_assert

    shifter_qwen_gpu2 env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/c159_q2_qwen_worker.py \
        --model Qwen/Qwen2.5-1.5B --allowed-context-lengths 16,32,64,128,256,512 \
        --batch-size 1 --warmup-per-length 3 --socket "$qwen_socket" \
        --output "$raw/${label}_qwen.json" >"$raw/${label}_qwen.log" 2>&1 &
    local qwen_pid=$!
    pids="$pids $qwen_pid"
    for _ in {1..3600}; do
        [[ -S "$qwen_socket" ]] && break
        if ! kill -0 "$qwen_pid" 2>/dev/null; then
            wait "$qwen_pid" || true
            echo "C176 Qwen worker exited before readiness ($label)" >&2
            return 1
        fi
        sleep 0.05
    done
    [[ -S "$qwen_socket" ]] || { echo "C176 Qwen readiness timeout ($label)" >&2; return 1; }

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
            --receiver-seed "$receiver_seed" \
            --channel-seed-base "$channel_seed_base" --snr-db "$snr" \
            >"${owner_output%.json}.log" 2>&1 &
        pids="$pids $!"
    done < "$peer_tsv"

    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/c159_persistent_nrx_worker.py \
        --peer-spec "$peer_spec" --ipc-dir "$state" \
        --schedule-file "$schedule_file" --iterations "$iterations" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$raw/${label}_nrx_worker.json" --destination-device 3 \
        >"$raw/${label}_nrx_worker.log" 2>&1 &
    pids="$pids $!"

    shifter_all env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/c176_burst_coordinator.py \
        --peer-spec "$peer_spec" --ipc-dir "$state" \
        --schedule-file "$schedule_file" --qwen-socket "$qwen_socket" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$raw/${label}_coordinator.json" --iterations "$iterations" \
        --ai-trace "$trace" --policy "$policy" --slo-ms "$slo_ms" \
        --max-leases "$max_leases" "${padded_arg[@]}" \
        --period-ms "$period_ms" --destination-device 2 \
        --warmup "$warmup" --release-lead-ms "$release_lead_ms" \
        >"$raw/${label}_coordinator.log" 2>&1 &
    pids="$pids $!"

    local failure=0
    for pid in $pids; do wait "$pid" || failure=1; done
    pids=""
    softwall_mps_stop || true
    if [[ "$failure" -ne 0 ]]; then
        echo "C176 process failure in $label; artifacts preserved" >&2
        return 1
    fi
    echo "C176 run complete: $label"
}

status=0
for entry in $runs; do
    IFS=: read -r pattern policy mode <<<"$entry"
    run_one "$pattern" "$policy" "$mode" || status=1
done
cleanup
trap - EXIT INT TERM
exit "$status"
