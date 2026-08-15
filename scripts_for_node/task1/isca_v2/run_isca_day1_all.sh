#!/usr/bin/env bash
set -Eeuo pipefail

# DART-Rx Day-1 unattended campaign.  The topology invariant is fixed GPU0
# 4g+3g.  No command in this runner creates/destroys MIG instances.
REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RDMA_IMG=${RDMA_IMG:-airan:25-3-rdma}
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/isca_v2/day1_$(date -u +%Y%m%dT%H%M%SZ)}
MIG_4G=${MIG_4G:-}
MIG_3G=${MIG_3G:-}
GPU1=${GPU1:-GPU-86123961-c0e3-6d2b-4aeb-b08504fe6647}
GPU2=${GPU2:-GPU-7cfa813b-475f-c71a-5b15-7e352d2ded67}
GPU3=${GPU3:-GPU-ee09d6cb-2e38-e9e1-9b76-bbc0f8f79b1a}
RUN_TOKEN=$(basename "$RESULTS_ROOT" | tr -cd 'A-Za-z0-9_.-')
STATUS_TSV="$RESULTS_ROOT/STATUS.tsv"
FAILURES=0

sudo mkdir -p "$RESULTS_ROOT"
sudo chmod 0777 "$RESULTS_ROOT"
exec 9>"$RESULTS_ROOT/.lock"
if ! flock -n 9; then
    echo "[DART-DAY1] another runner owns $RESULTS_ROOT" >&2
    exit 3
fi

timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [DART-DAY1] $*"; }

discover_topology() {
    local topology
    topology=$(nvidia-smi -L)
    [[ $(grep -c 'MIG 4g.20gb' <<<"$topology") -eq 1 ]]
    [[ $(grep -c 'MIG 3g.20gb' <<<"$topology") -eq 1 ]]
    MIG_4G=${MIG_4G:-$(awk '/MIG 4g/{print $6}' <<<"$topology" | tr -d ')' | head -1)}
    MIG_3G=${MIG_3G:-$(awk '/MIG 3g/{print $6}' <<<"$topology" | tr -d ')' | head -1)}
    grep -F "$MIG_4G" <<<"$topology" | grep -Fq 'MIG 4g.20gb'
    grep -F "$MIG_3G" <<<"$topology" | grep -Fq 'MIG 3g.20gb'
    export MIG_4G MIG_3G
}

cleanup_scoped() {
    mapfile -t scoped < <(
        docker ps -a --format '{{.Names}}' | grep '^dart-day1-' || true
    )
    if ((${#scoped[@]})); then
        docker rm -f "${scoped[@]}" >/dev/null 2>&1 || true
    fi
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup_scoped
    rm -f "$RESULTS_ROOT/RUNNING"
    if ((rc == 0 && FAILURES == 0)); then
        timestamp >"$RESULTS_ROOT/COMPLETE"
        log "ALL JOBS COMPLETE results=$RESULTS_ROOT"
    elif ((rc == 0)); then
        timestamp >"$RESULTS_ROOT/COMPLETE_WITH_FAILURES"
        log "campaign finished with $FAILURES failed optional jobs"
    else
        timestamp >"$RESULTS_ROOT/INTERRUPTED"
        log "campaign interrupted rc=$rc; rerun with the same RESULTS_ROOT to resume"
    fi
    exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_job() {
    local id=$1 importance=$2 function=$3
    JOB_DIR="$RESULTS_ROOT/$id"
    export JOB_DIR
    sudo mkdir -p "$JOB_DIR"
    sudo chmod 0777 "$JOB_DIR"
    if [[ -s "$JOB_DIR/COMPLETE" ]]; then
        log "SKIP $id (complete)"
        return 0
    fi
    rm -f "$JOB_DIR/FAILED"
    local started rc
    started=$(timestamp)
    printf '%s\t%s\tRUNNING\t%s\n' "$started" "$id" "$importance" \
        >>"$STATUS_TSV"
    log "START $id"
    set +e
    (set -Eeuo pipefail; "$function") > >(tee "$JOB_DIR/job.log") 2>&1
    rc=$?
    set -e
    if ((rc == 0)); then
        timestamp >"$JOB_DIR/COMPLETE"
        printf '%s\t%s\tCOMPLETE\t%s\n' "$(timestamp)" "$id" "$importance" \
            >>"$STATUS_TSV"
        log "DONE $id"
        return 0
    fi
    printf '%s\n' "$rc" >"$JOB_DIR/FAILED"
    printf '%s\t%s\tFAILED(rc=%s)\t%s\n' "$(timestamp)" "$id" "$rc" \
        "$importance" >>"$STATUS_TSV"
    log "FAILED $id rc=$rc importance=$importance"
    if [[ "$importance" == required ]]; then
        return "$rc"
    fi
    FAILURES=$((FAILURES + 1))
    return 0
}

docker_gpu() {
    local name=$1 devices=$2 image=$3
    shift 3
    docker run --rm --name "$name" --runtime=nvidia --ipc=host \
        -e NVIDIA_VISIBLE_DEVICES="$devices" \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" \
        -v "$ENGINE_HOST:/engines:ro" \
        -v "$JOB_DIR:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$image" "$@"
}

job_preflight() {
    discover_topology
    if ! grep -q ACTIVE /sys/class/infiniband/mlx5_0/ports/1/state; then
        log "restoring scoped NIC PHY loopback"
        sudo /mydata/nic_loopback_restore.sh
    fi
    python3 "$PYAERIAL/isca_v2/preflight.py" \
        --output "$JOB_DIR/preflight.json" --source-root "$PYAERIAL"
}

job_scheme_correctness() {
    cd "$PYAERIAL/isca_v2"
    python3 -m unittest -v test_dart_runtime.py
    python3 test_dart_runtime.py --fault-iterations 10000 \
        >"$JOB_DIR/fault_campaign.json"
    python3 - <<'PY'
import json, os
path = os.path.join(os.environ["JOB_DIR"], "fault_campaign.json")
text = open(path, encoding="utf-8").read()
value = json.loads(text[text.index("{"):])
assert value["pass"] is True
assert value["wrong_commit"] == 0
print("[DART-CORRECTNESS] 10k fault campaign PASS")
PY
}

job_nrx_profile() {
    docker_gpu dart-day1-nrx-profile "$MIG_4G" "$IMG" \
        python3 nrx_trt_direct.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --warmup 100 --iterations 3000 --cuda-graph --compare-wrapper \
            --output /results/direct_graph.json
    docker_gpu dart-day1-wrapper-profile "$MIG_4G" "$IMG" \
        python3 nrx_deep_profile.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --warmup 30 --iterations 300 --output-bits 2 \
            --output /results/wrapper_breakdown.json
    test -s "$JOB_DIR/direct_graph.json"
    test -s "$JOB_DIR/wrapper_breakdown.json"
}

job_nsys_profile() {
    docker_gpu dart-day1-nsys "$MIG_4G" "$IMG" \
        nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
            --force-overwrite=true -o /results/nrx_wrapper_profile \
            python3 nrx_deep_profile.py \
                --engine /engines/neural_rx_fp16_4g.trt \
                --warmup 10 --iterations 30 --output-bits 2 \
                --output /results/nsys_wrapper_breakdown.json
    test -s "$JOB_DIR/nrx_wrapper_profile.nsys-rep"
    docker_gpu dart-day1-nsys-stats "$MIG_4G" "$IMG" \
        nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
            --format csv /results/nrx_wrapper_profile.nsys-rep \
        >"$JOB_DIR/nsys_stats.csv"
}

job_fixed_placement() {
    local trials=${PLACEMENT_TRIALS:-3}
    local iterations=${PLACEMENT_ITERS:-1000}
    for trial in $(seq 1 "$trials"); do
        docker_gpu "dart-day1-place-standalone-$trial" "$MIG_4G" "$IMG" \
            env NRX_ENGINE=/engines/neural_rx_fp16_4g.trt RESULTS_DIR=/results \
            python3 p2p_overlap_bench.py standalone "fixed43_standalone_t$trial" \
                --iterations "$iterations" --warmup 50 --ring-depth 2
        docker_gpu "dart-day1-place-same-$trial" "$MIG_4G" "$IMG" \
            env NRX_ENGINE=/engines/neural_rx_fp16_4g.trt RESULTS_DIR=/results \
            python3 p2p_overlap_bench.py same "fixed43_same_t$trial" \
                --iterations "$iterations" --warmup 50 --ring-depth 2
        docker_gpu "dart-day1-place-p2p-$trial" "$MIG_4G,$MIG_3G" "$IMG" \
            env NRX_ENGINE=/engines/neural_rx_fp16_4g.trt RESULTS_DIR=/results \
            python3 p2p_overlap_bench.py p2p "fixed43_p2p_t$trial" \
                --iterations "$iterations" --warmup 50 --ring-depth 2
    done
    [[ $(find "$JOB_DIR" -name 'p2p_overlap_*.json' | wc -l) -eq $((trials * 3)) ]]
}

remove_gdr_tag() {
    local tag=$1
    sudo find /dev/shm -maxdepth 1 -type f \
        -name "gdr_rdma_${tag}_*" -delete 2>/dev/null || true
}

job_fixed_gdr() {
    local trials=${GDR_TRIALS:-3}
    local iterations=${GDR_ITERS:-1000}
    local rdma_args=(--network=host --cap-add=IPC_LOCK \
        --device=/dev/infiniband --ipc=host --ulimit memlock=-1)
    for trial in $(seq 1 "$trials"); do
        local tag="dart_${RUN_TOKEN}_gdr_t${trial}"
        local consumer="dart-day1-gdr-cons-$trial"
        local producer="dart-day1-gdr-prod-$trial"
        remove_gdr_tag "$tag"
        docker rm -f "$consumer" "$producer" >/dev/null 2>&1 || true
        docker run -d --name "$consumer" --runtime=nvidia \
            -e NVIDIA_VISIBLE_DEVICES="$MIG_3G" \
            -e NRX_ENGINE=/engines/neural_rx_fp16_4g.trt \
            -e PYTHONDONTWRITEBYTECODE=1 \
            "${rdma_args[@]}" -v "$REPO:/opt/nvidia/cuBB" \
            -v "$ENGINE_HOST:/engines:ro" -w /opt/nvidia/cuBB/pyaerial \
            "$RDMA_IMG" python3 nrx_consumer_gdr.py "$tag" >/dev/null
        local info="/dev/shm/gdr_rdma_${tag}_fwd_cons.info"
        for _ in $(seq 1 100); do
            [[ -s "$info" ]] && break
            docker inspect -f '{{.State.Running}}' "$consumer" | grep -qx true
            sleep 0.2
        done
        [[ -s "$info" ]]
        docker run --rm --name "$producer" --runtime=nvidia \
            -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" \
            -e RESULTS_DIR=/results -e PYTHONDONTWRITEBYTECODE=1 \
            "${rdma_args[@]}" -v "$REPO:/opt/nvidia/cuBB" \
            -v "$JOB_DIR:/results" -w /opt/nvidia/cuBB/pyaerial \
            "$RDMA_IMG" python3 l1_producer_gdr.py \
                "fixed43_gdr_t$trial" "$iterations" "$tag" \
            >"$JOB_DIR/l1_t${trial}.log" 2>&1
        local consumer_rc
        consumer_rc=$(timeout 30 docker wait "$consumer")
        docker logs "$consumer" >"$JOB_DIR/nrx_t${trial}.log" 2>&1 || true
        docker rm -f "$consumer" >/dev/null 2>&1 || true
        remove_gdr_tag "$tag"
        [[ "$consumer_rc" == 0 ]]
        grep -q '\[NRx-GDR\] ready' "$JOB_DIR/nrx_t${trial}.log"
        grep -q '\[NRx-GDR\] shutdown signal' "$JOB_DIR/nrx_t${trial}.log"
        grep -q '\[L1-GDR\].*mean=' "$JOB_DIR/l1_t${trial}.log"
    done
    [[ $(find "$JOB_DIR" -name 'l1prod_*.json' | wc -l) -eq "$trials" ]]
}

job_compile_profiles() {
    python3 "$PYAERIAL/isca_v2/compile_dart_profiles.py" \
        --results-root "$RESULTS_ROOT" --endpoint-count 2 \
        --error-guard-pct 10 --output "$JOB_DIR/dart_profile.json"
}

job_control_replay() {
    docker run --rm --name dart-day1-replay \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" \
        -v "$RESULTS_ROOT:/campaign:ro" -v "$JOB_DIR:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/dart_policy_replay.py \
            --profile /campaign/06_compile_profiles/dart_profile.json \
            --trace 'low:500:2,burst:1400:3,low:500:2' \
            --policies S0,S1,S2,S3,S6,S7,S8 \
            --deadline-ms 5 --conventional-ms 1 --qmax-us 500 \
            --output /results/control_replay.json
}

job_hardware_policy() {
    docker_gpu dart-day1-policy "$MIG_4G,$GPU1,$GPU2" "$IMG" \
        python3 isca_v2/nrx_dart_policy_sweep.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --policies round_robin,shortest_queue,predicted_finish \
            --rates 1000,2000,2800,3400 --duration 2 \
            --warmup 30 --calibration-requests 500 \
            --output /results/hardware_policy.json
}

job_replica_sweep() {
    docker_gpu dart-day1-replicas "$MIG_4G" "$IMG" \
        python3 nrx_replica_sweep.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --replicas 1,2,4,8 --rates 400,600,800,1000,1200 \
            --closed-loop-requests 4000 --open-loop-duration 2 \
            --warmup-rounds 20 --output /results/replica_sweep.json
}

job_background_suite() {
    RESULTS_ROOT="$JOB_DIR" PRIMARY_MIG="$MIG_4G" SPARE_MIG="$MIG_3G" \
        WORKLOADS="resnet50 bert_base whisper_base qwen_decode" \
        POLICIES="naive_share adaptive_reclaim" \
        TRACE='low:500:2.01,burst:1100:3,low:500:3' \
        bash "$PYAERIAL/isca_v2/run_fixed_mig_background_suite.sh"
}

job_mps_quick() {
    RESULTS_ROOT="$JOB_DIR" GPU_UUID="$GPU3" \
        MPS_PCTS="30 70 100" TRIALS=1 ITERS=500 WARMUP=50 \
        QWEN_DUR=150 QWEN_WARMUP=30 \
        bash "$PYAERIAL/run_mps_direct_sweep.sh"
}

job_mig_mps_quick() {
    RESULTS_ROOT="$JOB_DIR" MIG_PCTS="30 70 100" \
        TRIALS=1 ITERS=500 WARMUP=50 \
        bash "$PYAERIAL/run_mig_mps_direct.sh"
}

main() {
    timestamp >"$RESULTS_ROOT/RUNNING"
    rm -f "$RESULTS_ROOT/INTERRUPTED" "$RESULTS_ROOT/COMPLETE_WITH_FAILURES"
    cleanup_scoped
    # run_job executes each body in an isolation subshell.  Discover the
    # immutable endpoint UUIDs once in the parent so all later jobs inherit
    # them; preflight independently re-validates the same topology.
    discover_topology
    run_job 00_preflight required job_preflight
    run_job 01_scheme_correctness required job_scheme_correctness
    run_job 02_nrx_profile optional job_nrx_profile
    run_job 03_nsys_profile optional job_nsys_profile
    run_job 04_fixed_placement optional job_fixed_placement
    run_job 05_fixed_gdr optional job_fixed_gdr
    run_job 06_compile_profiles optional job_compile_profiles
    run_job 07_control_replay optional job_control_replay
    run_job 08_hardware_policy optional job_hardware_policy
    run_job 09_replica_sweep optional job_replica_sweep
    run_job 10_background_suite optional job_background_suite
    run_job 11_mps_quick optional job_mps_quick
    run_job 12_mig_mps_quick optional job_mig_mps_quick
    find "$RESULTS_ROOT" -maxdepth 2 -type f -printf '%P\t%s\n' \
        | sort >"$RESULTS_ROOT/ARTIFACTS.tsv"
}

main "$@"
