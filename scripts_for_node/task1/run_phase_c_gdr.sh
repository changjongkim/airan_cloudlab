#!/bin/bash
# Run ONLY Phase C GPUDirect RDMA staging variants (Configs 4 and 7) with
# Topology B (3g+2g+2g).
# Appends results to /mydata/results/chain/SUMMARY.txt.
set -u

SDK_HOST=/mydata/aerial-cuda-accelerated-ran
SDK_MNT=/opt/nvidia/cuBB
RESULTS_ROOT=/mydata/results/chain
RDMA_IMG=airan:25-3-rdma
QWEN_IMG=airan:25-3-final
L1_ITERS="${L1_ITERS:-30}"
QWEN_DUR="${QWEN_DUR:-300}"
SUMMARY_FILE="$RESULTS_ROOT/SUMMARY.txt"
STATUS_FILE="$RESULTS_ROOT/STATUS.txt"
RDMA_ARGS=(--network=host --cap-add=IPC_LOCK --device=/dev/infiniband --ipc=host --ulimit memlock=-1)
TOPOLOGY_NEEDS_RESTORE=0

log()   { echo "[$(date -u +%FT%TZ)] $*"; }
alert() { echo "[$(date -u +%FT%TZ)] [ALERT] $*"; echo "[ALERT] $*" >> "$STATUS_FILE"; }
step()  { echo ""; log "=== $* ==="; echo "$*" >> "$STATUS_FILE"; }

parse_and_record() {
    local cfg="$1" pct="$2" log_file="$3" qwen_log="${4:-}"
    local line mean p95 p99 qiters qitps qwen_line
    line=$(grep -E 'mean=[0-9.]+ms +p95=' "$log_file" | tail -1)
    mean=$(echo "$line" | grep -oE 'mean=[0-9.]+' | head -1 | cut -d= -f2)
    p95=$(echo "$line" | grep -oE 'p95=[0-9.]+' | head -1 | cut -d= -f2)
    p99=$(echo "$line" | grep -oE 'p99=[0-9.]+' | head -1 | cut -d= -f2)
    qiters=""; qitps=""
    if [ -n "$qwen_log" ] && [ -f "$qwen_log" ]; then
        qwen_line=$(grep -E 'done: [0-9]+ iters' "$qwen_log" | tail -1)
        [ -z "$qwen_line" ] && qwen_line=$(grep -E 'progress: [0-9]+ iters' "$qwen_log" | tail -1)
        qiters=$(echo "$qwen_line" | grep -oE '[0-9]+ iters' | head -1 | awk '{print $1}')
        qitps=$(echo "$qwen_line" | grep -oE '[0-9.]+ it/s' | head -1 | awk '{print $1}')
    fi
    echo "$cfg,$pct,$mean,$p95,$p99,$qiters,$qitps," >> "$SUMMARY_FILE"
    if [ -n "$mean" ] && awk -v m="$mean" 'BEGIN{exit !(m>250)}'; then
        alert "$cfg pct=$pct: L1 mean=${mean}ms exceeds 250ms threshold"
    fi
}

cleanup_gdr_tag() {
    local tag="$1"
    local pattern="/dev/shm/gdr_rdma_${tag}_*"
    sudo find /dev/shm -maxdepth 1 -type f \
        -path "$pattern" -delete 2>/dev/null || true
}

cleanup_containers() {
    docker rm -f \
        airan_c4_gdr_l1 airan_c4_gdr_nrx airan_c4_gdr_qwen \
        airan_c7_gdr_l1 airan_c7_gdr_nrx airan_c7_gdr_qwen \
        > /dev/null 2>&1 || true
    cleanup_gdr_tag config4_gdr
    cleanup_gdr_tag config7_gdr
}

destroy_mig() {
    log "MIG teardown on GPU 0 (kill MPS first)"
    sudo pkill -9 -f '[n]vidia-cuda-mps' > /dev/null 2>&1 || true
    sleep 2
    sudo nvidia-smi mig -dci -i 0 > /dev/null 2>&1 || true
    sudo nvidia-smi mig -dgi -i 0 > /dev/null 2>&1 || true
    sleep 1
}

setup_topology_B() {
    log "MIG topology B: 3g + 2g + 2g on GPU 0"
    destroy_mig
    sudo nvidia-smi -i 0 -mig 1 > /dev/null 2>&1 || return 1
    sleep 2
    sudo nvidia-smi mig -cgi 9,14,14 -C -i 0 > /tmp/mig_B.log 2>&1
    if [ $? -ne 0 ]; then
        alert "Topology B failed; see /tmp/mig_B.log"; return 1
    fi
    MIG_AI=$(nvidia-smi -L | awk '/MIG 3g/{print $6}' | tr -d ')' | head -1)
    MIGS_2g=($(nvidia-smi -L | awk '/MIG 2g/{print $6}' | tr -d ')'))
    MIG_L1=${MIGS_2g[0]}
    MIG_NRX=${MIGS_2g[1]}
    log "MIG_L1(2g)=$MIG_L1  MIG_NRX(2g)=$MIG_NRX  MIG_AI(3g)=$MIG_AI"
    export MIG_L1 MIG_AI MIG_NRX
}

restore_topology_A() {
    log "restoring Topology A (4g + 3g)"
    destroy_mig
    if ! sudo nvidia-smi -i 0 -mig 1 > /dev/null 2>&1; then
        alert "restore Topology A failed while enabling MIG"
        return 1
    fi
    sleep 2
    if ! sudo nvidia-smi mig -cgi 5,9 -C -i 0 > /dev/null 2>&1; then
        alert "restore Topology A failed while creating 4g+3g"
        return 1
    fi
}

wait_for_info_file() {
    local path="$1" container="$2" deadline=$((SECONDS + 15))
    while [ ! -s "$path" ]; do
        if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -qx true; then
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            return 1
        fi
        sleep 0.2
    done
}

collect_logs() {
    local outdir="$1" nrx_name="$2" qwen_name="$3"
    docker logs "$nrx_name" > "$outdir/nrx.log" 2>&1 || true
    docker logs "$qwen_name" > "$outdir/qwen.log" 2>&1 || true
}

run_gdr_config() {
    local cfg="$1" label="$2" tag="$3" outdir_name="$4" prefix="$5" description="$6"
    local outdir="$RESULTS_ROOT/$outdir_name"
    local l1_name="airan_${prefix}_gdr_l1"
    local nrx_name="airan_${prefix}_gdr_nrx"
    local qwen_name="airan_${prefix}_gdr_qwen"
    local info_file="/dev/shm/gdr_rdma_${tag}_fwd_cons.info"
    local l1_rc nrx_wait_rc nrx_rc

    step "$description"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    cleanup_gdr_tag "$tag"
    docker rm -f "$l1_name" "$nrx_name" "$qwen_name" > /dev/null 2>&1 || true

    # Warm the AI workload before establishing the two GDR QPs. Starting NRx
    # first and waiting for its ready line would deadlock: GdrRdmaEndpoint
    # publishes its info file and then waits for the L1 peer in __init__.
    if ! docker run -d --name "$qwen_name" --gpus "\"device=$MIG_AI\"" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
        -v "$SDK_HOST/pyaerial:/work" -w /work "$QWEN_IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" > /dev/null 2>&1; then
        alert "$cfg: failed to start Qwen"
        return 1
    fi
    sleep 60
    if ! docker inspect -f '{{.State.Running}}' "$qwen_name" 2>/dev/null | grep -qx true; then
        collect_logs "$outdir" "$nrx_name" "$qwen_name"
        alert "$cfg: Qwen died during warmup"
        docker rm -f "$qwen_name" > /dev/null 2>&1 || true
        return 1
    fi

    if ! docker run -d --name "$nrx_name" --gpus "\"device=$MIG_NRX\"" \
        "${RDMA_ARGS[@]}" \
        -v "$SDK_HOST:$SDK_MNT" -w "$SDK_MNT/pyaerial" "$RDMA_IMG" \
        python3 nrx_consumer_gdr.py "$tag" > /dev/null 2>&1; then
        collect_logs "$outdir" "$nrx_name" "$qwen_name"
        alert "$cfg: failed to start NRx"
        docker rm -f "$nrx_name" "$qwen_name" > /dev/null 2>&1 || true
        cleanup_gdr_tag "$tag"
        return 1
    fi
    if ! wait_for_info_file "$info_file" "$nrx_name"; then
        collect_logs "$outdir" "$nrx_name" "$qwen_name"
        alert "$cfg: NRx did not publish $info_file within 15s"
        docker rm -f "$nrx_name" "$qwen_name" > /dev/null 2>&1 || true
        cleanup_gdr_tag "$tag"
        return 1
    fi

    docker run --rm --name "$l1_name" --gpus "\"device=$MIG_L1\"" \
        "${RDMA_ARGS[@]}" \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$RDMA_IMG" \
        python3 l1_producer_gdr.py "$label" "$L1_ITERS" "$tag" > "$outdir/l1.log" 2>&1
    l1_rc=$?

    nrx_rc=""
    if nrx_rc=$(timeout 30 docker wait "$nrx_name" 2>/dev/null); then
        nrx_wait_rc=0
    else
        nrx_wait_rc=$?
    fi
    collect_logs "$outdir" "$nrx_name" "$qwen_name"
    docker rm -f "$nrx_name" "$qwen_name" > /dev/null 2>&1 || true
    cleanup_gdr_tag "$tag"

    if [ "$l1_rc" -ne 0 ]; then
        alert "$cfg: L1 failed rc=$l1_rc"
        return 1
    fi
    if [ "$nrx_wait_rc" -ne 0 ]; then
        alert "$cfg: timed out waiting for NRx exit (wait rc=$nrx_wait_rc)"
        return 1
    fi
    if [ "$nrx_rc" != "0" ]; then
        alert "$cfg: NRx exited rc=${nrx_rc:-unknown}"
        return 1
    fi
    if ! grep -q '\[NRx-GDR\] ready' "$outdir/nrx.log"; then
        alert "$cfg: NRx-GDR ready marker missing"
        return 1
    fi
    if ! grep -q '\[NRx-GDR\] shutdown signal' "$outdir/nrx.log"; then
        alert "$cfg: NRx-GDR shutdown marker missing"
        return 1
    fi

    local stats
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' \
        "$outdir/l1.log" | tail -1)
    if [ -z "$stats" ]; then
        alert "$cfg: L1 summary line missing"
        return 1
    fi
    log "$cfg: L1 $stats"
    parse_and_record "$cfg" "-" "$outdir/l1.log" "$outdir/qwen.log"
}

# ============================================================== main
on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup_containers
    if [ "$TOPOLOGY_NEEDS_RESTORE" -eq 1 ]; then
        if restore_topology_A; then
            TOPOLOGY_NEEDS_RESTORE=0
        else
            alert "PHASE C GDR exit cleanup could not restore Topology A"
            rc=1
        fi
    fi
    exit "$rc"
}

trap on_exit EXIT
trap 'alert "PHASE C GDR stopped by SIGINT"; exit 130' INT
trap 'alert "PHASE C GDR stopped by SIGTERM"; exit 143' TERM

log "==== PHASE C GPUDirect RDMA staging start (Topology B: 3g+2g+2g) ===="
for cfg in config4_gdr config7_gdr; do
    if grep -q "^${cfg}," "$SUMMARY_FILE"; then
        alert "$cfg already exists in SUMMARY.txt; refusing duplicate append"
        exit 1
    fi
done
cleanup_containers
TOPOLOGY_NEEDS_RESTORE=1
setup_topology_B || { alert "Topology B setup failed, aborting"; exit 1; }
if [ -n "${MIG_L1:-}" ] && [ -n "${MIG_NRX:-}" ] && [ -n "${MIG_AI:-}" ]; then
    run_gdr_config \
        config4_gdr config4_gdr_l1 config4_gdr config4_cross_mig_gdr c4 \
        "Config 4 GPUDirect RDMA staging: L1(2g)<->NRx(2g) + Qwen(3g)"
    rc4=$?
    run_gdr_config \
        config7_gdr config7_gdr_l1 config7_gdr config7_cross_repeat_gdr c7 \
        "Config 7 GPUDirect RDMA staging: cross-partition repeat + Qwen(3g)"
    rc7=$?
    if [ "$rc4" -ne 0 ] || [ "$rc7" -ne 0 ]; then
        alert "PHASE C GDR failed (config4_rc=$rc4 config7_rc=$rc7)"
        exit 1
    fi
else
    alert "Topology B UUID discovery failed"
    exit 1
fi
step "restore Topology A"
if restore_topology_A; then
    TOPOLOGY_NEEDS_RESTORE=0
else
    alert "normal completion could not restore Topology A"
    exit 1
fi
log "==== PHASE C GPUDirect RDMA staging done ===="
tail -3 "$SUMMARY_FILE"
