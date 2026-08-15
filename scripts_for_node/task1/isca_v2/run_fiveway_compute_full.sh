#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RDMA_IMG=${RDMA_IMG:-airan:25-3-rdma}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/04_nrx_capacity/COMPLETE}
JOB_DIR="$RESULTS_ROOT/05_fiveway_compute"
MIG_4G=${MIG_4G:-MIG-dae3f173-7b15-594b-bc80-6cef80687a56}
MIG_3G=${MIG_3G:-MIG-80a4659b-f06f-540b-9f4b-1c91f78aaaf3}
FULL_GPU=${FULL_GPU:-GPU-ee09d6cb-2e38-e9e1-9b76-bbc0f8f79b1a}
TRIALS=${TRIALS:-5}
DURATION=${DURATION:-60}
LOADS=${LOADS:-"0.50 0.85 0.95 1.05"}
MPS_PIPE=/tmp/nvidia-mps-dart-full-fiveway
MPS_LOG=/tmp/nvidia-mps-dart-full-fiveway-log
ACTIVE_CONSUMER=""
MPS_RUNNING=0

sudo mkdir -p "$JOB_DIR"
sudo chmod 0777 "$RESULTS_ROOT" "$JOB_DIR"
exec 9>"$JOB_DIR/.lock"
flock -n 9 || exit 3
timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [FIVEWAY-FULL] $*"; }

clean_cuda_tag() {
    local tag=$1
    sudo rm -f "/dev/shm/cuda_ipc_${tag}.info" \
        "/dev/shm/cuda_ipc_${tag}.ctrl"
}
clean_gdr_tag() {
    local tag=$1
    sudo find /dev/shm -maxdepth 1 -type f \
        -name "gdr_rdma_${tag}_*" -delete 2>/dev/null || true
}
stop_mps() {
    if ((MPS_RUNNING)) && [[ -S "$MPS_PIPE/control" ]]; then
        echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    MPS_RUNNING=0
}
cleanup() {
    [[ -z "$ACTIVE_CONSUMER" ]] || \
        docker rm -f "$ACTIVE_CONSUMER" >/dev/null 2>&1 || true
    docker rm -f dart-full-fiveway-prod dart-full-fiveway-serial \
        >/dev/null 2>&1 || true
    stop_mps
}
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

start_mps() {
    local device=$1
    stop_mps
    sudo rm -rf "$MPS_PIPE" "$MPS_LOG"
    sudo mkdir -p "$MPS_PIPE" "$MPS_LOG"
    sudo chmod 0777 "$MPS_PIPE" "$MPS_LOG"
    sudo env CUDA_VISIBLE_DEVICES="$device" \
        CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
        nvidia-cuda-mps-control -d </dev/null \
        >"$JOB_DIR/mps_daemon_$(basename "$device").log" 2>&1
    for _ in $(seq 1 100); do
        [[ -S "$MPS_PIPE/control" ]] && break
        sleep 0.1
    done
    [[ -S "$MPS_PIPE/control" ]]
    MPS_RUNNING=1
}

rm -f "$JOB_DIR/FAILED"
log "waiting for dependency $DEPENDENCY"
for _ in $(seq 1 17280); do
    [[ -s "$DEPENDENCY" ]] && break
    [[ -s "${DEPENDENCY%/*}/FAILED" ]] && exit 20
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || exit 21

REFERENCE_CAPACITY=$(python3 - "$RESULTS_ROOT/04_nrx_capacity" <<'PY'
import glob, json, statistics, sys
values=[]
for path in glob.glob(sys.argv[1]+"/capacity_4g_t*.json"):
    value=json.load(open(path))
    one=next(x for x in value["configurations"] if x["replicas"] == 1)
    values.append(float(one["saturation"]["throughput_slots_per_s"]))
if len(values) < 3: raise SystemExit("insufficient 4g capacity trials")
print(f"{statistics.median(values):.9f}")
PY
)
log "common reference capacity=$REFERENCE_CAPACITY slots/s"

touch "$JOB_DIR/RUNNING"
rm -f "$JOB_DIR/COMPLETE"
cp "$0" "$JOB_DIR/runner.captured.sh"
{
    echo "started_utc=$(timestamp)"
    echo "trials=$TRIALS"
    echo "duration_s=$DURATION"
    echo "load_fractions=$LOADS"
    echo "reference_capacity_slots_s=$REFERENCE_CAPACITY"
    echo "common_trace=true"
    echo "timing_boundary=l1_front_start_to_ldpc_crc_complete"
    docker image inspect --format 'image_id={{.Id}}' "$IMG"
    docker image inspect --format 'rdma_image_id={{.Id}}' "$RDMA_IMG"
    sha256sum "$PYAERIAL/isca_v2/placement_serial_bench.py" \
        "$PYAERIAL/isca_v2/cuda_ipc_channel.py" \
        "$PYAERIAL/isca_v2/cuda_ipc_l1_producer.py" \
        "$PYAERIAL/isca_v2/cuda_ipc_nrx_consumer.py" \
        "$PYAERIAL/l1_producer_gdr.py" "$PYAERIAL/nrx_consumer_gdr.py"
} >"$JOB_DIR/MANIFEST.txt"

valid_output() {
    local path=$1 approach=$2 rate=$3
    python3 - "$path" "$approach" "$rate" <<'PY'
import json, math, os, sys
path, approach, rate = sys.argv[1], sys.argv[2], float(sys.argv[3])
if not os.path.isfile(path): raise SystemExit(1)
v=json.load(open(path)); assert v.get("pass", True) is True
assert math.isclose(v["arrival_rate_slots_per_s"], rate, rel_tol=1e-8)
assert v["iterations"] > 0
if approach == "gdr": assert "raw_sojourn_ms" in v
else: assert "sojourn_ms" in v["metrics"]
PY
}

run_serial() {
    local approach=$1 mode=$2 devices=$3 engine=$4 load=$5 rate=$6 trial=$7
    local outdir="$JOB_DIR/$approach/load_${load}/trial_${trial}"
    local output="$outdir/result.json"
    mkdir -p "$outdir"; chmod 0777 "$outdir"
    if valid_output "$output" "$approach" "$rate"; then log "SKIP $approach $load t$trial"; return; fi
    rm -f "$output" "$output.tmp"
    log "START $approach load=$load rate=$rate trial=$trial"
    timeout 300 docker run --rm --name dart-full-fiveway-serial --runtime=nvidia \
        --ipc=host -e NVIDIA_VISIBLE_DEVICES="$devices" \
        -e PYTHONDONTWRITEBYTECODE=1 -v "$REPO:/opt/nvidia/cuBB" \
        -v "$ENGINE_HOST:/engines:ro" -v "$outdir:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/placement_serial_bench.py --mode "$mode" \
            --engine "/engines/$engine" --warmup 100 \
            --arrival-rate "$rate" --duration "$DURATION" --trial "$trial" \
            --output /results/result.json \
        >"$outdir/run.log" 2>&1
    valid_output "$output" "$approach" "$rate"
}

run_cuda_ipc() {
    local approach=$1 device=$2 engine=$3 load=$4 rate=$5 trial=$6
    local outdir="$JOB_DIR/$approach/load_${load}/trial_${trial}"
    local output="$outdir/result.json"
    mkdir -p "$outdir"; chmod 0777 "$outdir"
    if valid_output "$output" "$approach" "$rate"; then log "SKIP $approach $load t$trial"; return; fi
    local tag="five_${approach}_${load//./}_${trial}_$$"
    clean_cuda_tag "$tag"
    docker rm -f dart-full-fiveway-prod dart-full-fiveway-cons >/dev/null 2>&1 || true
    ACTIVE_CONSUMER=dart-full-fiveway-cons
    log "START $approach load=$load rate=$rate trial=$trial"
    docker run -d --name dart-full-fiveway-prod --runtime=nvidia \
        --ipc=host --pid=host -e NVIDIA_VISIBLE_DEVICES="$device" \
        -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" -v "$MPS_PIPE:$MPS_PIPE" \
        -v "$REPO:/opt/nvidia/cuBB" -v "$outdir:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/cuda_ipc_l1_producer.py --tag "$tag" \
            --warmup 100 --arrival-rate "$rate" --duration "$DURATION" \
            --trial "$trial" --variant "$approach" --output /results/result.json \
        >/dev/null
    for _ in $(seq 1 100); do
        [[ -s "/dev/shm/cuda_ipc_${tag}.info" ]] && break
        docker inspect -f '{{.State.Running}}' dart-full-fiveway-prod | grep -qx true
        sleep 0.1
    done
    [[ -s "/dev/shm/cuda_ipc_${tag}.info" ]]
    docker run -d --name "$ACTIVE_CONSUMER" --runtime=nvidia \
        --ipc=host --pid=host -e NVIDIA_VISIBLE_DEVICES="$device" \
        -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" -v "$MPS_PIPE:$MPS_PIPE" \
        -v "$REPO:/opt/nvidia/cuBB" -v "$ENGINE_HOST:/engines:ro" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/cuda_ipc_nrx_consumer.py --tag "$tag" \
            --engine "/engines/$engine" >/dev/null
    prc=$(timeout 300 docker wait dart-full-fiveway-prod)
    crc=$(timeout 60 docker wait "$ACTIVE_CONSUMER")
    docker logs dart-full-fiveway-prod >"$outdir/l1.log" 2>&1 || true
    docker logs "$ACTIVE_CONSUMER" >"$outdir/nrx.log" 2>&1 || true
    docker rm -f dart-full-fiveway-prod "$ACTIVE_CONSUMER" >/dev/null 2>&1 || true
    ACTIVE_CONSUMER=""
    clean_cuda_tag "$tag"
    [[ "$prc" == 0 && "$crc" == 0 ]]
    grep -q '\[CUDA-IPC-NRX\] ready' "$outdir/nrx.log"
    grep -q '\[CUDA-IPC-NRX\] shutdown' "$outdir/nrx.log"
    valid_output "$output" "$approach" "$rate"
}

run_gdr() {
    local load=$1 rate=$2 trial=$3
    local approach=gdr
    local outdir="$JOB_DIR/$approach/load_${load}/trial_${trial}"
    mkdir -p "$outdir"; chmod 0777 "$outdir"
    local existing
    existing=$(find "$outdir" -maxdepth 1 -name 'l1prod_*.json' -print -quit)
    if [[ -n "$existing" ]] && valid_output "$existing" gdr "$rate"; then log "SKIP gdr $load t$trial"; return; fi
    rm -f "$outdir"/l1prod_*.json
    local tag="five_gdr_${load//./}_${trial}_$$"
    clean_gdr_tag "$tag"
    docker rm -f dart-full-fiveway-prod dart-full-fiveway-cons >/dev/null 2>&1 || true
    ACTIVE_CONSUMER=dart-full-fiveway-cons
    local rdma_args=(--network=host --cap-add=IPC_LOCK \
        --device=/dev/infiniband --ipc=host --ulimit memlock=-1)
    log "START gdr load=$load rate=$rate trial=$trial"
    docker run -d --name "$ACTIVE_CONSUMER" --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$MIG_3G" \
        -e NRX_ENGINE=/engines/neural_rx_fp16_4g.trt \
        "${rdma_args[@]}" -v "$REPO:/opt/nvidia/cuBB" \
        -v "$ENGINE_HOST:/engines:ro" -w /opt/nvidia/cuBB/pyaerial \
        "$RDMA_IMG" python3 nrx_consumer_gdr.py "$tag" >/dev/null
    for _ in $(seq 1 100); do
        [[ -s "/dev/shm/gdr_rdma_${tag}_fwd_cons.info" ]] && break
        docker inspect -f '{{.State.Running}}' "$ACTIVE_CONSUMER" | grep -qx true
        sleep 0.2
    done
    [[ -s "/dev/shm/gdr_rdma_${tag}_fwd_cons.info" ]]
    timeout 300 docker run --rm --name dart-full-fiveway-prod --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" -e RESULTS_DIR=/results \
        -e ARRIVAL_RATE_SLOTS_S="$rate" -e ARRIVAL_DURATION_S="$DURATION" \
        -e EXPERIMENT_TRIAL="$trial" \
        "${rdma_args[@]}" -v "$REPO:/opt/nvidia/cuBB" \
        -v "$outdir:/results" -w /opt/nvidia/cuBB/pyaerial "$RDMA_IMG" \
        python3 l1_producer_gdr.py "gdr_l${load}_t${trial}" 1 "$tag" \
        >"$outdir/l1.log" 2>&1
    crc=$(timeout 60 docker wait "$ACTIVE_CONSUMER")
    docker logs "$ACTIVE_CONSUMER" >"$outdir/nrx.log" 2>&1 || true
    docker rm -f "$ACTIVE_CONSUMER" >/dev/null 2>&1 || true
    ACTIVE_CONSUMER=""
    clean_gdr_tag "$tag"
    [[ "$crc" == 0 ]]
    existing=$(find "$outdir" -maxdepth 1 -name 'l1prod_*.json' -print -quit)
    [[ -n "$existing" ]]
    valid_output "$existing" gdr "$rate"
}

# The exact same absolute arrival rates are replayed for all five approaches.
for approach in mig_local p2p; do
    for load in $LOADS; do
        rate=$(python3 -c "print($REFERENCE_CAPACITY * $load)")
        for trial in $(seq 1 "$TRIALS"); do
            if [[ "$approach" == mig_local ]]; then
                run_serial "$approach" mig_local "$MIG_4G" neural_rx_fp16_4g.trt "$load" "$rate" "$trial"
            else
                run_serial "$approach" p2p "$MIG_4G,$MIG_3G" neural_rx_fp16_4g.trt "$load" "$rate" "$trial"
            fi
        done
    done
done

start_mps "$FULL_GPU"
for load in $LOADS; do
    rate=$(python3 -c "print($REFERENCE_CAPACITY * $load)")
    for trial in $(seq 1 "$TRIALS"); do run_cuda_ipc mps "$FULL_GPU" neural_rx_fp16_full.trt "$load" "$rate" "$trial"; done
done
stop_mps

start_mps "$MIG_4G"
for load in $LOADS; do
    rate=$(python3 -c "print($REFERENCE_CAPACITY * $load)")
    for trial in $(seq 1 "$TRIALS"); do run_cuda_ipc mig_mps "$MIG_4G" neural_rx_fp16_4g.trt "$load" "$rate" "$trial"; done
done
stop_mps

for load in $LOADS; do
    rate=$(python3 -c "print($REFERENCE_CAPACITY * $load)")
    for trial in $(seq 1 "$TRIALS"); do run_gdr "$load" "$rate" "$trial"; done
done

python3 - "$JOB_DIR" "$TRIALS" <<'PY'
import glob, json, os, sys
root, trials = sys.argv[1], int(sys.argv[2])
for approach in ("mps","mig_local","mig_mps","p2p","gdr"):
    paths=glob.glob(os.path.join(root,approach,"load_*","trial_*","*.json"))
    assert len(paths) == trials*4, (approach,len(paths),trials*4)
    for path in paths:
        value=json.load(open(path)); assert value.get("pass",True) is True
        assert value["iterations"] > 0
print("[FIVEWAY-FULL] validated 5 approaches x 4 loads")
PY
rm -f "$JOB_DIR/RUNNING"
timestamp >"$JOB_DIR/COMPLETE"
log "ALL FIVE-WAY COMPUTE TRIALS COMPLETE"
