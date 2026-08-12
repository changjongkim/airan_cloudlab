#!/bin/bash
# Fair overlap experiment:
#   Topology A: 4g L1-only baseline, then L1+NRx concurrent streams on 4g.
#   Topology B: 2g L1-only baseline, then L1(2g)<->NRx(2g) direct CUDA P2P.
# Qwen runs on the separate 3g MIG in both topologies.
set -u

SDK_HOST=/mydata/aerial-cuda-accelerated-ran
SDK_MNT=/opt/nvidia/cuBB
RESULTS_ROOT="${RESULTS_ROOT:-/mydata/results/p2p_fair/direct_trt}"
ENGINE_HOST=/mydata/results/nrx_deep_profile/engines
IMG=airan:25-3-final
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-100}"
RING_DEPTH="${RING_DEPTH:-2}"
TRIALS="${TRIALS:-3}"
QWEN_DUR="${QWEN_DUR:-300}"
QWEN_WARMUP="${QWEN_WARMUP:-60}"
TOPOLOGY_NEEDS_RESTORE=0

log() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { log "[ALERT] $*"; exit 1; }

cleanup_containers() {
    docker rm -f p2p_fair_qwen_a p2p_fair_qwen_b p2p_fair_bench \
        > /dev/null 2>&1 || true
}

destroy_mig() {
    log "MIG teardown GPU0"
    sudo pkill -9 -f '[n]vidia-cuda-mps' > /dev/null 2>&1 || true
    sleep 2
    sudo nvidia-smi mig -dci -i 0 > /dev/null 2>&1 || true
    sudo nvidia-smi mig -dgi -i 0 > /dev/null 2>&1 || true
    sleep 1
}

setup_topology_b() {
    log "create Topology B: 3g + 2g + 2g"
    destroy_mig
    sudo nvidia-smi -i 0 -mig 1 > /dev/null 2>&1 || return 1
    sleep 2
    sudo nvidia-smi mig -cgi 9,14,14 -C -i 0 > /tmp/p2p_fair_mig_b.log 2>&1 \
        || return 1
}

restore_topology_a() {
    log "restore Topology A: 4g + 3g"
    destroy_mig
    sudo nvidia-smi -i 0 -mig 1 > /dev/null 2>&1 || return 1
    sleep 2
    sudo nvidia-smi mig -cgi 5,9 -C -i 0 > /tmp/p2p_fair_mig_a.log 2>&1 \
        || return 1
}

discover_topology_a() {
    MIG_4G=$(nvidia-smi -L | awk '/MIG 4g/{print $6}' | tr -d ')' | head -1)
    MIG_AI=$(nvidia-smi -L | awk '/MIG 3g/{print $6}' | tr -d ')' | head -1)
    [ -n "$MIG_4G" ] && [ -n "$MIG_AI" ]
}

discover_topology_b() {
    MIG_AI=$(nvidia-smi -L | awk '/MIG 3g/{print $6}' | tr -d ')' | head -1)
    MIGS_2G=($(nvidia-smi -L | awk '/MIG 2g/{print $6}' | tr -d ')'))
    [ -n "$MIG_AI" ] && [ "${#MIGS_2G[@]}" -eq 2 ] || return 1
    MIG_L1=${MIGS_2G[0]}
    MIG_NRX=${MIGS_2G[1]}
}

start_qwen() {
    local name="$1" mig="$2" outdir="$3"
    docker rm -f "$name" > /dev/null 2>&1 || true
    docker run -d --name "$name" --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$mig" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
        -v "$SDK_HOST/pyaerial:/work" -w /work "$IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" > /dev/null
    sleep "$QWEN_WARMUP"
    if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true; then
        docker logs "$name" > "$outdir/qwen.log" 2>&1 || true
        return 1
    fi
}

stop_qwen() {
    local name="$1" outdir="$2"
    docker logs "$name" > "$outdir/qwen.log" 2>&1 || true
    docker rm -f "$name" > /dev/null 2>&1 || true
    grep -qE '\[Qwen\] (progress|done): [0-9]+ iters' "$outdir/qwen.log"
}

run_bench() {
    local mode="$1" label="$2" visible="$3" outdir="$4"
    mkdir -p "$outdir"; chmod 777 "$outdir"
    if find "$outdir" -maxdepth 1 -name 'p2p_overlap_*.json' | grep -q .; then
        log "[ALERT] refusing stale JSON in $outdir"
        return 1
    fi
    docker rm -f p2p_fair_bench > /dev/null 2>&1 || true
    log "run $label mode=$mode visible=$visible"
    docker run --rm --name p2p_fair_bench --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$visible" \
        -e NRX_ENGINE=/engines/neural_rx_fp16_4g.trt \
        -e RESULTS_DIR=/results \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" \
        -v "$ENGINE_HOST:/engines:ro" \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 p2p_overlap_bench.py "$mode" "$label" \
        --iterations "$ITERS" --warmup "$WARMUP" --ring-depth "$RING_DEPTH" \
        > "$outdir/bench.log" 2>&1 || return 1
    [ "$(find "$outdir" -maxdepth 1 -name 'p2p_overlap_*.json' | wc -l)" -eq 1 ] \
        || return 1
    grep -q '\[P2P-BENCH\] RESULT' "$outdir/bench.log" || return 1
}

validate_pair() {
    local baseline_dir="$1" overlap_dir="$2"
    python3 - "$baseline_dir" "$overlap_dir" <<'PY'
import glob, json, math, sys
baseline = json.load(open(glob.glob(sys.argv[1] + "/p2p_overlap_*.json")[0]))
overlap = json.load(open(glob.glob(sys.argv[2] + "/p2p_overlap_*.json")[0]))
assert baseline["mode"] == "standalone"
assert overlap["mode"] in ("same", "p2p")
for result in (baseline, overlap):
    assert result["iterations"] > 0
    assert len(result["raw"]) == result["iterations"]
    assert math.isfinite(result["metrics"]["l1_active_ms"]["mean"])
assert math.isfinite(overlap["metrics"]["nrx_ms"]["mean"])
print(
    overlap["label"],
    "l1_slowdown=",
    overlap["metrics"]["l1_active_ms"]["mean"]
    / baseline["metrics"]["l1_active_ms"]["mean"],
)
PY
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup_containers
    if [ "$TOPOLOGY_NEEDS_RESTORE" -eq 1 ]; then
        restore_topology_a || rc=1
    fi
    exit "$rc"
}

trap on_exit EXIT
trap 'log "[ALERT] interrupted"; exit 130' INT
trap 'log "[ALERT] terminated"; exit 143' TERM

if [ -e "$RESULTS_ROOT/COMPLETE" ]; then
    fail "$RESULTS_ROOT already complete"
fi
sudo mkdir -p "$RESULTS_ROOT"
sudo chmod -R 777 "$RESULTS_ROOT"
cleanup_containers

log "==== P2P fair overlap experiment start ===="
discover_topology_a || fail "Topology A (4g+3g) not present at start"
log "Topology A 4g=$MIG_4G AI3g=$MIG_AI"
mkdir -p "$RESULTS_ROOT/topology_a"; chmod 777 "$RESULTS_ROOT/topology_a"
start_qwen p2p_fair_qwen_a "$MIG_AI" "$RESULTS_ROOT/topology_a" \
    || fail "Topology A Qwen warm-up failed"
for trial in $(seq 1 "$TRIALS"); do
    run_bench standalone "l1_only_4g_t${trial}" "$MIG_4G" \
        "$RESULTS_ROOT/topology_a/trial${trial}/l1_only_4g" \
        || fail "4g L1-only trial $trial failed"
    run_bench same "same_overlap_4g_t${trial}" "$MIG_4G" \
        "$RESULTS_ROOT/topology_a/trial${trial}/same_overlap_4g" \
        || fail "4g same overlap trial $trial failed"
    validate_pair "$RESULTS_ROOT/topology_a/trial${trial}/l1_only_4g" \
        "$RESULTS_ROOT/topology_a/trial${trial}/same_overlap_4g" \
        | tee "$RESULTS_ROOT/topology_a/trial${trial}/validation.txt"
done
stop_qwen p2p_fair_qwen_a "$RESULTS_ROOT/topology_a" \
    || fail "Topology A Qwen progress missing"

TOPOLOGY_NEEDS_RESTORE=1
setup_topology_b || fail "Topology B setup failed"
discover_topology_b || fail "Topology B UUID discovery failed"
log "Topology B L1=$MIG_L1 NRx=$MIG_NRX AI=$MIG_AI"
mkdir -p "$RESULTS_ROOT/topology_b"; chmod 777 "$RESULTS_ROOT/topology_b"
start_qwen p2p_fair_qwen_b "$MIG_AI" "$RESULTS_ROOT/topology_b" \
    || fail "Topology B Qwen warm-up failed"
for trial in $(seq 1 "$TRIALS"); do
    run_bench standalone "l1_only_2g_t${trial}" "$MIG_L1" \
        "$RESULTS_ROOT/topology_b/trial${trial}/l1_only_2g" \
        || fail "2g L1-only trial $trial failed"
    run_bench p2p "cross_p2p_2g2g_t${trial}" "$MIG_L1,$MIG_NRX" \
        "$RESULTS_ROOT/topology_b/trial${trial}/cross_p2p_2g2g" \
        || fail "2g+2g P2P trial $trial failed"
    validate_pair "$RESULTS_ROOT/topology_b/trial${trial}/l1_only_2g" \
        "$RESULTS_ROOT/topology_b/trial${trial}/cross_p2p_2g2g" \
        | tee "$RESULTS_ROOT/topology_b/trial${trial}/validation.txt"
done
stop_qwen p2p_fair_qwen_b "$RESULTS_ROOT/topology_b" \
    || fail "Topology B Qwen progress missing"

restore_topology_a || fail "normal Topology A restore failed"
TOPOLOGY_NEEDS_RESTORE=0
date -u +%FT%TZ > "$RESULTS_ROOT/COMPLETE"
log "==== P2P fair overlap experiment complete ===="
