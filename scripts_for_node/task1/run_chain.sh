#!/bin/bash
# Full experiment chain — runs all 7 configs unattended.
# Survives SSH disconnect (started via nohup). Emits [ALERT] to log on any
# anomaly the orchestrator can detect (crashed container, missing output,
# L1 latency > 250ms, no Qwen progress in 60s).
#
# Usage: nohup ./run_chain.sh > /mydata/results/chain/orchestrator.log 2>&1 &

set -u
export SDK_HOST=/mydata/aerial-cuda-accelerated-ran
export SDK_MNT=/opt/nvidia/cuBB
export RESULTS_ROOT=/mydata/results/chain
export IMG=airan:25-3-final
export L1_ITERS="${L1_ITERS:-30}"
export QWEN_DUR="${QWEN_DUR:-300}"    # each L1 run finishes in <60s incl init
export MPS_PCTS="${MPS_PCTS:-30 50 70 100}"

mkdir -p "$RESULTS_ROOT"
STATUS_FILE="$RESULTS_ROOT/STATUS.txt"
SUMMARY_FILE="$RESULTS_ROOT/SUMMARY.txt"
echo "" > "$STATUS_FILE"
echo "config,pct,mean_ms,p95_ms,p99_ms,qwen_iters,qwen_itps,notes" > "$SUMMARY_FILE"

log()   { echo "[$(date -u +%FT%TZ)] $*"; }
alert() { echo "[$(date -u +%FT%TZ)] [ALERT] $*"; echo "[ALERT] $*" >> "$STATUS_FILE"; }
step()  { echo ""; log "=== $* ==="; echo "$*" >> "$STATUS_FILE"; }

# ---------------------------------------------------------------- utilities
cleanup_containers() {
    docker ps -aq --filter "name=airan_" | xargs -r docker rm -f > /dev/null 2>&1 || true
    sudo rm -f /dev/shm/airan_* 2>/dev/null || true
}

destroy_mig() {
    log "MIG teardown on GPU 0 (kill MPS first — it holds the GPU handle)"
    sudo pkill -9 -f nvidia-cuda-mps > /dev/null 2>&1 || true
    sleep 2
    sudo nvidia-smi mig -dci -i 0 > /dev/null 2>&1 || true
    sudo nvidia-smi mig -dgi -i 0 > /dev/null 2>&1 || true
    sudo nvidia-smi -i 0 -mig 0    > /dev/null 2>&1 || true
    sleep 1
}

setup_topology_A() {
    # 4g + 3g on GPU 0
    log "MIG topology A: 4g.20gb + 3g.20gb on GPU 0"
    destroy_mig
    sudo nvidia-smi -i 0 -mig 1 > /dev/null 2>&1
    sleep 2
    sudo nvidia-smi mig -cgi 5,9 -C -i 0 > /tmp/mig_A.log 2>&1
    if [ $? -ne 0 ]; then
        alert "Topology A creation failed; see /tmp/mig_A.log"
        return 1
    fi
    MIG_L1=$(nvidia-smi -L | awk '/MIG 4g/{print $6}' | tr -d ')')
    MIG_AI=$(nvidia-smi -L | awk '/MIG 3g/{print $6}' | tr -d ')')
    log "MIG_L1(4g)=$MIG_L1  MIG_AI(3g)=$MIG_AI"
    export MIG_L1 MIG_AI
}

setup_topology_B() {
    # 3g + 2g + 2g on GPU 0 (only 3-partition split fitting A100 40GB memory
    # slice budget of 8; 3g uses 4 memory slices even though it has 3 compute).
    # Layout: Qwen on 3g (20GB HBM needed for 7B fp16), L1 on 2g[0], NRx on 2g[1].
    log "MIG topology B: 3g + 2g + 2g on GPU 0"
    destroy_mig
    sudo nvidia-smi -i 0 -mig 1 > /dev/null 2>&1
    sleep 2
    sudo nvidia-smi mig -cgi 9,14,14 -C -i 0 > /tmp/mig_B.log 2>&1
    if [ $? -ne 0 ]; then
        alert "Topology B creation failed; see /tmp/mig_B.log"
        return 1
    fi
    MIG_AI=$(nvidia-smi -L | awk '/MIG 3g/{print $6}' | tr -d ')' | head -1)
    MIGS_2g=($(nvidia-smi -L | awk '/MIG 2g/{print $6}' | tr -d ')'))
    MIG_L1=${MIGS_2g[0]}
    MIG_NRX=${MIGS_2g[1]}
    log "MIG_L1(2g)=$MIG_L1  MIG_AI(3g)=$MIG_AI  MIG_NRX(2g)=$MIG_NRX"
    log "  note: L1 and NRx have SM 28 each (down from SM 56/42 in Topology A)"
    export MIG_L1 MIG_AI MIG_NRX
}

start_mps_daemon() {
    # MPS scoped to a specific GPU (default: GPU 3, which is MIG-free).
    local gpu_uuid="${1:-GPU-ee09d6cb-2e38-e9e1-9b76-bbc0f8f79b1a}"  # GPU 3
    log "starting MPS control daemon (scoped to $gpu_uuid)"
    sudo pkill -9 -f nvidia-cuda-mps 2>/dev/null || true
    sleep 2
    sudo env CUDA_VISIBLE_DEVICES="$gpu_uuid" nvidia-cuda-mps-control -d \
        < /dev/null > /tmp/mps_daemon.log 2>&1 &
    disown $! 2>/dev/null || true
    sleep 3
}

stop_mps_daemon() {
    log "stopping MPS control daemon"
    echo quit | sudo nvidia-cuda-mps-control 2>/dev/null || true
    sudo pkill -f nvidia-cuda-mps 2>/dev/null || true
    sleep 1
}

# ------------------------------------------------------ per-config runners
run_config5_cuphy_baseline() {
    step "Config 5: cuPHY baseline (GPU 3, no other workload)"
    outdir="$RESULTS_ROOT/config5_cuphy"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    docker run --rm --name airan_c5 \
        --gpus "\"device=3\"" \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 real_l1.py config5_baseline 20 "$L1_ITERS" > "$outdir/run.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        alert "Config 5 failed rc=$rc"; return 1
    fi
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/run.log" | tail -1)
    log "Config 5: $stats"
    parse_and_record "config5" "-" "$outdir/run.log"
}

run_config6_cuphy_nrx_baseline() {
    step "Config 6: cuPHY+NRx baseline (GPU 3, no other workload)"
    outdir="$RESULTS_ROOT/config6_cuphy_nrx"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    docker run --rm --name airan_c6 \
        --gpus "\"device=3\"" \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 real_l1_nrx.py config6_baseline 1 "$L1_ITERS" > "$outdir/run.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        alert "Config 6 failed rc=$rc"; return 1
    fi
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/run.log" | tail -1)
    log "Config 6: $stats"
    parse_and_record "config6" "-" "$outdir/run.log"
}

run_config1_colocate_mig() {
    step "Config 1: L1+NRx co-located on 4g, Qwen on 3g, MIG only"
    outdir="$RESULTS_ROOT/config1_colocate_mig"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    # Start Qwen on 3g partition (background)
    docker run -d --name airan_c1_qwen --gpus "\"device=$MIG_AI\"" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -v "$SDK_HOST/pyaerial:/work" -w /work -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 "$IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" > "$outdir/qwen.log" 2>&1
    sleep 60  # wait for Qwen to load and start looping
    if ! docker ps --filter name=airan_c1_qwen --format '{{.Names}}' | grep -q airan_c1_qwen; then
        alert "Config 1: Qwen container died during load"
        docker logs airan_c1_qwen > "$outdir/qwen_dead.log" 2>&1 || true
        return 1
    fi
    # Now run L1+NRx pipeline on 4g partition (foreground)
    docker run --rm --name airan_c1_l1 --gpus "\"device=$MIG_L1\"" \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 real_l1_nrx.py config1_l1nrx 1 "$L1_ITERS" > "$outdir/l1.log" 2>&1
    rc=$?
    docker logs airan_c1_qwen > "$outdir/qwen.log" 2>&1 || true
    docker rm -f airan_c1_qwen > /dev/null 2>&1
    if [ $rc -ne 0 ]; then
        alert "Config 1: L1 failed rc=$rc"; return 1
    fi
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/l1.log" | tail -1)
    qiters=$(grep -oE 'done: [0-9]+ iters' "$outdir/qwen.log" | tail -1 | awk '{print $2}')
    log "Config 1: L1 $stats  Qwen iters=$qiters"
    parse_and_record "config1" "-" "$outdir/l1.log" "$outdir/qwen.log"
}

run_config3_colocate_mig_mps() {
    # SIMPLIFIED: On A100 MIG mode, host MPS daemon on one partition blocks
    # non-MPS clients on sibling partitions (CUDA err 100 confirmed empirically).
    # Since each MIG partition here holds a single workload, MPS pct sweep is
    # not meaningful — MIG already isolates SMs. Config 3 = Config 1 + note.
    step "Config 3: MIG+MPS collapses to Config 1 (single workload / partition; MPS pct N/A)"
    outdir="$RESULTS_ROOT/config3_mig_only_repeat"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    docker run -d --name airan_c3_qwen --gpus "\"device=$MIG_AI\"" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
        -v "$SDK_HOST/pyaerial:/work" -w /work "$IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" > "$outdir/qwen.log" 2>&1
    sleep 60
    if ! docker ps --filter name=airan_c3_qwen --format '{{.Names}}' | grep -q airan_c3_qwen; then
        alert "Config 3: Qwen died"
        docker logs airan_c3_qwen > "$outdir/qwen_dead.log" 2>&1 || true; return 1
    fi
    docker run --rm --name airan_c3_l1 --gpus "\"device=$MIG_L1\"" \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 real_l1_nrx.py "config3_repeat" 1 "$L1_ITERS" > "$outdir/l1.log" 2>&1
    rc=$?
    docker logs airan_c3_qwen > "$outdir/qwen.log" 2>&1 || true
    docker rm -f airan_c3_qwen > /dev/null 2>&1
    if [ $rc -ne 0 ]; then
        alert "Config 3: L1 failed rc=$rc"; return 1
    fi
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/l1.log" | tail -1)
    qiters=$(grep -oE 'done: [0-9]+ iters' "$outdir/qwen.log" | tail -1 | awk '{print $2}')
    log "Config 3: L1 $stats  Qwen=$qiters"
    parse_and_record "config3" "-" "$outdir/l1.log" "$outdir/qwen.log"
}

run_config2_full_share_mps() {
    # NOTE: Uses GPU 3 (MIG-free) instead of destroying MIG on GPU 0.
    # Rationale: GPU 0 MIG-disable can require reboot after MPS use, and any
    # A100 chip works for the "full GPU + MPS" story equally well.
    step "Config 2: GPU 3 (MIG-free), full GPU via MPS"
    start_mps_daemon
    for pct in $MPS_PCTS; do
        step "Config 2: MPS pct=$pct (Qwen cap)"
        outdir="$RESULTS_ROOT/config2_mps_${pct}"
        mkdir -p "$outdir"; chmod 777 "$outdir"
        docker run -d --name airan_c2_qwen --gpus "\"device=3\"" --ipc=host \
            -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
            -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
            -v /tmp/nvidia-log:/tmp/nvidia-log \
            -v /mydata/hf_cache:/mydata/hf_cache \
            -e HF_HOME=/mydata/hf_cache \
            -v "$SDK_HOST/pyaerial:/work" -w /work "$IMG" \
            python3 qwen7b_stress.py "$QWEN_DUR" > "$outdir/qwen.log" 2>&1
        sleep 60
        if ! docker ps --filter name=airan_c2_qwen --format '{{.Names}}' | grep -q airan_c2_qwen; then
            alert "Config 2 pct=$pct: Qwen died"
            docker logs airan_c2_qwen > "$outdir/qwen_dead.log" 2>&1 || true
            continue
        fi
        docker run --rm --name airan_c2_l1 --gpus "\"device=3\"" --ipc=host \
            -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
            -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
            -v /tmp/nvidia-mps:/tmp/nvidia-mps \
            -v /tmp/nvidia-log:/tmp/nvidia-log \
            -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
            -w "$SDK_MNT/pyaerial" "$IMG" \
            python3 real_l1_nrx.py "config2_p${pct}" 1 "$L1_ITERS" > "$outdir/l1.log" 2>&1
        rc=$?
        docker logs airan_c2_qwen > "$outdir/qwen.log" 2>&1 || true
        docker rm -f airan_c2_qwen > /dev/null 2>&1
        if [ $rc -ne 0 ]; then
            alert "Config 2 pct=$pct: L1 failed rc=$rc"; continue
        fi
        stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/l1.log" | tail -1)
        qiters=$(grep -oE 'done: [0-9]+ iters' "$outdir/qwen.log" | tail -1 | awk '{print $2}')
        log "Config 2 pct=$pct: L1 $stats  Qwen=$qiters"
        parse_and_record "config2" "$pct" "$outdir/l1.log" "$outdir/qwen.log"
    done
    stop_mps_daemon
}

run_config4_cross_partition() {
    # L1 on 3g[0], NRx on 1g via shm IPC, Qwen on 3g[1]. Topology B.
    step "Config 4: Cross-partition L1<->NRx (shm) + Qwen on 3rd partition, MIG only"
    outdir="$RESULTS_ROOT/config4_cross_mig"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    # Start NRx consumer on 1g partition
    docker run -d --name airan_c4_nrx --gpus "\"device=$MIG_NRX\"" --ipc=host \
        -v "$SDK_HOST:$SDK_MNT" \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 nrx_consumer.py config4 > "$outdir/nrx.log" 2>&1
    # Start Qwen on 2nd 3g partition
    docker run -d --name airan_c4_qwen --gpus "\"device=$MIG_AI\"" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -v "$SDK_HOST/pyaerial:/work" -w /work -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 "$IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" > "$outdir/qwen.log" 2>&1
    sleep 60
    if ! docker ps --filter name=airan_c4_qwen --format '{{.Names}}' | grep -q airan_c4_qwen; then
        alert "Config 4: Qwen died"
    fi
    if ! docker logs airan_c4_nrx 2>&1 | grep -q '\[NRx\] ready'; then
        alert "Config 4: NRx consumer not ready after 60s"
    fi
    # Run L1 producer on first 3g partition
    docker run --rm --name airan_c4_l1 --gpus "\"device=$MIG_L1\"" --ipc=host \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 l1_producer.py config4_l1 "$L1_ITERS" config4 > "$outdir/l1.log" 2>&1
    rc=$?
    docker logs airan_c4_qwen > "$outdir/qwen.log" 2>&1 || true
    docker logs airan_c4_nrx > "$outdir/nrx.log" 2>&1 || true
    docker rm -f airan_c4_nrx airan_c4_qwen > /dev/null 2>&1
    sudo rm -f /dev/shm/airan_config4_* 2>/dev/null
    if [ $rc -ne 0 ]; then
        alert "Config 4: L1 failed rc=$rc"; return 1
    fi
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/l1.log" | tail -1)
    qiters=$(grep -oE 'done: [0-9]+ iters' "$outdir/qwen.log" | tail -1 | awk '{print $2}')
    log "Config 4: L1 $stats  Qwen=$qiters"
    parse_and_record "config4" "-" "$outdir/l1.log" "$outdir/qwen.log"
}

run_config7_cross_partition_mps() {
    # SIMPLIFIED same as Config 3: MIG+MPS with single workload/partition is
    # a no-op. Config 7 collapses to Config 4.
    step "Config 7: Cross-partition MIG+MPS collapses to Config 4"
    outdir="$RESULTS_ROOT/config7_cross_repeat"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    docker run -d --name airan_c7_nrx --gpus "\"device=$MIG_NRX\"" --ipc=host \
        -v "$SDK_HOST:$SDK_MNT" -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 nrx_consumer.py "config7_repeat" > "$outdir/nrx.log" 2>&1
    docker run -d --name airan_c7_qwen --gpus "\"device=$MIG_AI\"" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
        -v "$SDK_HOST/pyaerial:/work" -w /work "$IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" > "$outdir/qwen.log" 2>&1
    sleep 60
    docker run --rm --name airan_c7_l1 --gpus "\"device=$MIG_L1\"" --ipc=host \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" -e RESULTS_DIR=/results \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 l1_producer.py "config7_repeat" "$L1_ITERS" "config7_repeat" > "$outdir/l1.log" 2>&1
    rc=$?
    docker logs airan_c7_qwen > "$outdir/qwen.log" 2>&1 || true
    docker logs airan_c7_nrx > "$outdir/nrx.log" 2>&1 || true
    docker rm -f airan_c7_nrx airan_c7_qwen > /dev/null 2>&1
    sudo rm -f /dev/shm/airan_config7_repeat_* 2>/dev/null
    if [ $rc -ne 0 ]; then
        alert "Config 7: L1 failed rc=$rc"; return 1
    fi
    stats=$(grep -oE 'mean=[0-9.]+ms p95=[0-9.]+ms p99=[0-9.]+ms' "$outdir/l1.log" | tail -1)
    qiters=$(grep -oE 'done: [0-9]+ iters' "$outdir/qwen.log" | tail -1 | awk '{print $2}')
    log "Config 7: L1 $stats  Qwen=$qiters"
    parse_and_record "config7" "-" "$outdir/l1.log" "$outdir/qwen.log"
}

parse_and_record() {
    local cfg="$1" pct="$2" log_file="$3" qwen_log="${4:-}"
    local line mean p95 p99 qiters qitps qwen_line
    line=$(grep -E 'mean=[0-9.]+ms +p95=' "$log_file" | tail -1)
    mean=$(echo "$line" | grep -oE 'mean=[0-9.]+' | head -1 | cut -d= -f2)
    p95=$(echo "$line" | grep -oE 'p95=[0-9.]+' | head -1 | cut -d= -f2)
    p99=$(echo "$line" | grep -oE 'p99=[0-9.]+' | head -1 | cut -d= -f2)
    qiters=""
    qitps=""
    if [ -n "$qwen_log" ] && [ -f "$qwen_log" ]; then
        # Prefer final "done:" line; fall back to last "progress:" line (Qwen
        # container is killed at L1 completion, so "done" often not printed).
        qwen_line=$(grep -E 'done: [0-9]+ iters' "$qwen_log" | tail -1)
        if [ -z "$qwen_line" ]; then
            qwen_line=$(grep -E 'progress: [0-9]+ iters' "$qwen_log" | tail -1)
        fi
        qiters=$(echo "$qwen_line" | grep -oE '[0-9]+ iters' | head -1 | awk '{print $1}')
        qitps=$(echo "$qwen_line" | grep -oE '[0-9.]+ it/s' | head -1 | awk '{print $1}')
    fi
    echo "$cfg,$pct,$mean,$p95,$p99,$qiters,$qitps," >> "$SUMMARY_FILE"
    # anomaly: L1 latency > 250ms mean is suspicious
    if [ -n "$mean" ] && awk -v m="$mean" 'BEGIN{exit !(m>250)}'; then
        alert "$cfg pct=$pct: L1 mean=${mean}ms exceeds 250ms threshold"
    fi
}

# ============================================================== MAIN CHAIN
trap 'log "CHAIN STOPPED (signal); cleaning up"; cleanup_containers; stop_mps_daemon; exit 1' INT TERM

log "==== AI-RAN experiment chain start ===="
log "L1_ITERS=$L1_ITERS  QWEN_DUR=$QWEN_DUR  MPS_PCTS='$MPS_PCTS'"
cleanup_containers

# Phase A: baselines + Topology A
step "PHASE A: Topology A (4g + 3g)"
setup_topology_A || { alert "Topology A setup failed, chain abort"; exit 1; }
run_config5_cuphy_baseline
run_config6_cuphy_nrx_baseline
run_config1_colocate_mig
run_config3_colocate_mig_mps

# Phase B: Full share
step "PHASE B: Topology C (no MIG) + MPS"
run_config2_full_share_mps

# Phase C: Cross-partition
step "PHASE C: Topology B (3g+3g+1g)"
setup_topology_B || { alert "Topology B setup failed, skipping Configs 4/7"; }
if [ -n "${MIG_L1:-}" ] && [ -n "${MIG_NRX:-}" ]; then
    run_config4_cross_partition
    run_config7_cross_partition_mps
fi

# Restore Topology A (steady state after chain).
step "CLEANUP: restore Topology A"
setup_topology_A || alert "post-chain topology restore failed"

log "==== chain complete ===="
log "summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
