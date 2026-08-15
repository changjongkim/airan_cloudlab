#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/02_nrx_stack/COMPLETE}
JOB_DIR="$RESULTS_ROOT/02b_nrx_profilers"
MIG_4G=${MIG_4G:-MIG-dae3f173-7b15-594b-bc80-6cef80687a56}

sudo mkdir -p "$JOB_DIR"
sudo chmod 0777 "$RESULTS_ROOT" "$JOB_DIR"
exec 9>"$JOB_DIR/.lock"
flock -n 9 || exit 3
timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [NRX-PROFILERS] $*"; }
cleanup() { docker rm -f dart-full-profiler >/dev/null 2>&1 || true; }
on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup
    if ((rc != 0)); then
        rm -f "$JOB_DIR/RUNNING"
        printf '%s\n' "$rc" >"$JOB_DIR/FAILED"
        log "FAILED rc=$rc"
    fi
    exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -f "$JOB_DIR/FAILED"
log "waiting for dependency $DEPENDENCY"
for _ in $(seq 1 2880); do
    [[ -s "$DEPENDENCY" ]] && break
    [[ -s "${DEPENDENCY%/*}/FAILED" ]] && exit 20
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || exit 21
touch "$JOB_DIR/RUNNING"
rm -f "$JOB_DIR/COMPLETE"
cp "$0" "$JOB_DIR/runner.captured.sh"
sha256sum "$PYAERIAL/isca_v2/nrx_profiler_target.py" \
    "$ENGINE_HOST/neural_rx_fp16_4g.trt" >"$JOB_DIR/MANIFEST.sha256"

docker_base=(docker run --rm --name dart-full-profiler --runtime=nvidia \
    --ipc=host -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" \
    -e PYTHONDONTWRITEBYTECODE=1 -v "$REPO:/opt/nvidia/cuBB" \
    -v "$ENGINE_HOST:/engines:ro" -v "$JOB_DIR:/results" \
    -w /opt/nvidia/cuBB/pyaerial "$IMG")

if [[ ! -s "$JOB_DIR/direct_graph.nsys-rep" ]]; then
    log "START Nsight Systems direct graph"
    "${docker_base[@]}" nsys profile --trace=cuda,nvtx,osrt \
        --sample=none --cpuctxsw=none --force-overwrite=true \
        -o /results/direct_graph \
        python3 isca_v2/nrx_profiler_target.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --warmup 20 --iterations 100 --output /results/nsys_target.json \
        2>&1 | tee "$JOB_DIR/nsys.log"
fi
"${docker_base[@]}" nsys stats \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    --format csv /results/direct_graph.nsys-rep \
    >"$JOB_DIR/nsys_stats.csv"

NCU_FAILURES=0
for trial in 1 2 3; do
    report="$JOB_DIR/ncu_t${trial}.ncu-rep"
    if [[ -s "$report" ]]; then continue; fi
    log "START Nsight Compute trial=$trial"
    set +e
    timeout 2400 "${docker_base[@]}" ncu --target-processes all \
        --set full --replay-mode kernel --force-overwrite \
        --nvtx --nvtx-include 'nrx_inference/' \
        -o "/results/ncu_t${trial}" \
        python3 isca_v2/nrx_profiler_target.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --warmup 5 --iterations 1 --output "/results/ncu_target_t${trial}.json" \
        >"$JOB_DIR/ncu_t${trial}.log" 2>&1
    rc=$?
    set -e
    if ((rc != 0)); then
        NCU_FAILURES=$((NCU_FAILURES + 1))
        log "NCU trial=$trial failed rc=$rc; preserving log and continuing"
    fi
done
for report in "$JOB_DIR"/*.ncu-rep; do
    [[ -e "$report" ]] || continue
    stem=$(basename "$report" .ncu-rep)
    "${docker_base[@]}" ncu --import "/results/$(basename "$report")" \
        --csv --page details >"$JOB_DIR/${stem}_details.csv" 2>&1 || true
done
printf '%s\n' "$NCU_FAILURES" >"$JOB_DIR/NCU_FAILURES"
test -s "$JOB_DIR/direct_graph.nsys-rep"
test -s "$JOB_DIR/nsys_stats.csv"
rm -f "$JOB_DIR/RUNNING"
timestamp >"$JOB_DIR/COMPLETE"
log "PROFILERS COMPLETE ncu_failures=$NCU_FAILURES"
