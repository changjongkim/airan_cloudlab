#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

campaign=${1:-smoke_same_request_transaction}
iterations=${2:-100}
cap=${ENDPOINT_CAP:-80}
period_ms=${PERIOD_MS:-60}
deadline_ms=${DEADLINE_MS:-35}
timeout_ms=${ENDPOINT_TIMEOUT_MS:-30}
warmup=${WARMUP:-20}
seed=${SEED:-20321001}
snr_db=${SNR_DB:-}
channel_seed_base=${CHANNEL_SEED_BASE:-20322000}
policy=${POLICY:-s2}
nrx_bound_ms=${NRX_BOUND_MS:-30}
conv_bound_ms=${CONV_BOUND_MS:-25}
commit_guard_ms=${COMMIT_GUARD_MS:-2}
recovery_gap_ai=${RECOVERY_GAP_AI:-0}
background_kind=${BACKGROUND_KIND:-}
background_cap=${BACKGROUND_CAP:-20}
background_repeats=${BACKGROUND_REPEATS:-1}
ai_budget_ms=${AI_BUDGET_MS:-6}
ai_guard_ms=${AI_GUARD_MS:-1}
ai_rpc_timeout_ms=${AI_RPC_TIMEOUT_MS:-7}
gc_mode=${GC_MODE:-on}
[[ "$gc_mode" == on || "$gc_mode" == off ]] || {
    echo 'GC_MODE must be on or off' >&2
    exit 2
}
[[ "$cap" =~ ^[0-9]+$ ]] && ((cap >= 1 && cap <= 100)) || {
    echo "invalid endpoint cap: $cap" >&2
    exit 2
}
[[ "$policy" == s2 ]] || { echo 'transactional controller supports s2 only' >&2; exit 2; }
[[ "$recovery_gap_ai" == 0 || "$recovery_gap_ai" == 1 ]] || {
    echo 'RECOVERY_GAP_AI must be 0 or 1' >&2
    exit 2
}
if [[ -n "$background_kind" ]]; then
    [[ "$background_kind" == nrx || "$background_kind" == gemm || "$background_kind" == hbm ]] || {
        echo "invalid background kind: $background_kind" >&2
        exit 2
    }
    [[ "$background_cap" =~ ^[0-9]+$ ]] && ((background_cap >= 1 && background_cap <= 100)) || {
        echo "invalid background cap: $background_cap" >&2
        exit 2
    }
    [[ "$background_repeats" =~ ^[0-9]+$ ]] && ((background_repeats >= 1)) || {
        echo "invalid background repeats: $background_repeats" >&2
        exit 2
    }
fi
if [[ "$recovery_gap_ai" == 1 && -z "$background_kind" ]]; then
    echo 'RECOVERY_GAP_AI requires BACKGROUND_KIND' >&2
    exit 2
fi

raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
state="$SOFTWALL_ROOT/run_state/softwall_same_gpu/job${SLURM_JOB_ID}"
mkdir -p "$raw" "$state"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
owns_gpu_lock=0
if [[ "${SOFTWALL_GPU_LOCK_HELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || {
        echo "external GPU0 lock was declared but is absent" >&2
        exit 4
    }
else
    if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    fi
    owns_gpu_lock=1
fi
lock_dir="$raw/.lock_${campaign}"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "campaign already running: $campaign" >&2
    ((owns_gpu_lock == 0)) || rmdir "$gpu_lock_dir" 2>/dev/null || true
    exit 3
fi

tag="sr_${SLURM_JOB_ID}_${BASHPID}"
controller_output="$raw/${campaign}_cap${cap}_job${SLURM_JOB_ID}_controller.json"
worker_output="$raw/${campaign}_cap${cap}_job${SLURM_JOB_ID}_worker.json"
background_output="$raw/${campaign}_cap${cap}_job${SLURM_JOB_ID}_background.json"
background_socket="$state/bg_${tag}.sock"
worker_pid=""
background_pid=""
cleanup() {
    if [[ -n "$worker_pid" ]]; then
        kill "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "$background_pid" ]]; then
        kill "$background_pid" 2>/dev/null || true
    fi
    rm -f "$background_socket"
    rmdir "$lock_dir" 2>/dev/null || true
    ((owns_gpu_lock == 0)) || rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

shifter_gpu env \
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$cap" \
    CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/same_request_ipc_worker.py \
    --tag "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$worker_output" &
worker_pid=$!

controller_args=()
if [[ -n "$snr_db" ]]; then
    controller_args+=(--snr-db "$snr_db" --channel-seed-base "$channel_seed_base")
fi
if [[ -n "$background_kind" ]]; then
    shifter_gpu env \
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$background_cap" \
        CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/ai_worker.py \
        --kind "$background_kind" --mode rpc \
        --repeats "$background_repeats" --socket "$background_socket" \
        --output "$background_output" &
    background_pid=$!
    for _ in {1..1200}; do
        [[ -S "$background_socket" ]] && break
        kill -0 "$background_pid" 2>/dev/null || {
            echo "background worker exited before ready" >&2
            exit 1
        }
        sleep 0.05
    done
    [[ -S "$background_socket" ]] || {
        echo "background worker readiness timeout" >&2
        exit 1
    }
    controller_args+=(
        --socket "$background_socket"
        --ai-budget-ms "$ai_budget_ms"
        --ai-guard-ms "$ai_guard_ms"
        --ai-rpc-timeout-ms "$ai_rpc_timeout_ms"
    )
fi
if [[ "$recovery_gap_ai" == 1 ]]; then
    controller_args+=(--recovery-gap-ai)
fi
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/same_request_transaction_controller_gc_probe.py \
    --tag "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$controller_output" \
    --iterations "$iterations" --warmup "$warmup" \
    --period-ms "$period_ms" --deadline-ms "$deadline_ms" \
    --endpoint-timeout-ms "$timeout_ms" --seed "$seed" --policy "$policy" \
    --nrx-bound-ms "$nrx_bound_ms" --conv-bound-ms "$conv_bound_ms" \
    --commit-guard-ms "$commit_guard_ms" \
    --gc-mode "$gc_mode" \
    "${controller_args[@]}"

wait "$worker_pid"
worker_pid=""
if [[ -n "$background_pid" ]]; then
    wait "$background_pid"
    background_pid=""
fi
cleanup
trap - EXIT INT TERM
echo "same-request IPC campaign completed: $campaign"
