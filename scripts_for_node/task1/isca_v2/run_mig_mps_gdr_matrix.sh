#!/usr/bin/env bash
set -Eeuo pipefail

# Proper two-process MIG+MPS gate: L1 and NRx are distinct MPS clients on the
# same fixed 4g MIG and communicate through the same GDR pipeline used by the
# cross-MIG baseline.  The sibling 3g runs an isolated Qwen kernel workload.
REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RDMA_IMG=${RDMA_IMG:-airan:25-3-rdma}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
MIG_4G=${MIG_4G:?set MIG_4G}
MIG_3G=${MIG_3G:?set MIG_3G}
ALLOCATIONS=${ALLOCATIONS:-"30:70 50:50 70:30"}
TRIALS=${TRIALS:-3}
ITERATIONS=${ITERATIONS:-1000}
QWEN_DURATION=${QWEN_DURATION:-300}
QWEN_WARMUP=${QWEN_WARMUP:-30}
MPS_PIPE=${MPS_PIPE:-/tmp/nvidia-mps-dart-migmps}
MPS_LOG=${MPS_LOG:-/tmp/nvidia-mps-dart-migmps-log}
PREFIX=${PREFIX:-dart-migmps-gdr}
RDMA_ARGS=(--network=host --cap-add=IPC_LOCK --device=/dev/infiniband \
    --ipc=host --ulimit memlock=-1)
qwen_name="${PREFIX}-qwen"
active_consumer=""

clean_tag() {
    local tag=$1
    sudo find /dev/shm -maxdepth 1 -type f \
        -name "gdr_rdma_${tag}_*" -delete 2>/dev/null || true
}

cleanup() {
    if [[ -n "$active_consumer" ]]; then
        docker rm -f "$active_consumer" >/dev/null 2>&1 || true
    fi
    docker stop -t 20 "$qwen_name" >/dev/null 2>&1 || true
    docker logs "$qwen_name" >"$RESULTS_ROOT/qwen.log" 2>&1 || true
    docker rm -f "$qwen_name" >/dev/null 2>&1 || true
    if [[ -S "$MPS_PIPE/control" ]]; then
        echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

topology=$(nvidia-smi -L)
grep -F "$MIG_4G" <<<"$topology" | grep -Fq 'MIG 4g.20gb'
grep -F "$MIG_3G" <<<"$topology" | grep -Fq 'MIG 3g.20gb'
sudo mkdir -p "$RESULTS_ROOT" "$MPS_PIPE" "$MPS_LOG"
sudo chmod 0777 "$RESULTS_ROOT" "$MPS_PIPE" "$MPS_LOG"
cleanup
sudo mkdir -p "$MPS_PIPE" "$MPS_LOG"
sudo chmod 0777 "$MPS_PIPE" "$MPS_LOG"
sudo env CUDA_VISIBLE_DEVICES="$MIG_4G" \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
    nvidia-cuda-mps-control -d </dev/null \
    >"$RESULTS_ROOT/mps_daemon.log" 2>&1 &
for _ in $(seq 1 100); do
    [[ -S "$MPS_PIPE/control" ]] && break
    sleep 0.1
done
[[ -S "$MPS_PIPE/control" ]]

docker run -d --name "$qwen_name" --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$MIG_3G" \
    -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
    -v /mydata/hf_cache:/mydata/hf_cache \
    -v "$REPO/pyaerial:/work" -w /work "$IMG" \
    python3 qwen7b_stress.py "$QWEN_DURATION" >/dev/null
sleep "$QWEN_WARMUP"
docker inspect -f '{{.State.Running}}' "$qwen_name" | grep -qx true

for allocation in $ALLOCATIONS; do
    IFS=: read -r l1_pct nrx_pct <<<"$allocation"
    [[ "$l1_pct" =~ ^[0-9]+$ && "$nrx_pct" =~ ^[0-9]+$ ]]
    ((l1_pct > 0 && nrx_pct > 0 && l1_pct + nrx_pct == 100))
    for trial in $(seq 1 "$TRIALS"); do
        outdir="$RESULTS_ROOT/l1_${l1_pct}_nrx_${nrx_pct}/trial${trial}"
        mkdir -p "$outdir"; chmod 0777 "$outdir"
        tag="migmps_${l1_pct}_${nrx_pct}_t${trial}_$$"
        active_consumer="${PREFIX}-cons-${l1_pct}-${nrx_pct}-${trial}"
        producer="${PREFIX}-prod-${l1_pct}-${nrx_pct}-${trial}"
        clean_tag "$tag"
        docker rm -f "$active_consumer" "$producer" >/dev/null 2>&1 || true

        docker run -d --name "$active_consumer" --runtime=nvidia \
            -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" \
            -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$nrx_pct" \
            -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            -e NRX_ENGINE=/engines/neural_rx_fp16_4g.trt \
            "${RDMA_ARGS[@]}" -v "$MPS_PIPE:$MPS_PIPE" \
            -v "$MPS_LOG:$MPS_LOG" -v "$REPO:/opt/nvidia/cuBB" \
            -v "$ENGINE_HOST:/engines:ro" -w /opt/nvidia/cuBB/pyaerial \
            "$RDMA_IMG" python3 nrx_consumer_gdr.py "$tag" >/dev/null
        info="/dev/shm/gdr_rdma_${tag}_fwd_cons.info"
        for _ in $(seq 1 100); do
            [[ -s "$info" ]] && break
            docker inspect -f '{{.State.Running}}' "$active_consumer" | grep -qx true
            sleep 0.2
        done
        [[ -s "$info" ]]

        docker run --rm --name "$producer" --runtime=nvidia \
            -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" \
            -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$l1_pct" \
            -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            -e RESULTS_DIR=/results "${RDMA_ARGS[@]}" \
            -v "$MPS_PIPE:$MPS_PIPE" -v "$MPS_LOG:$MPS_LOG" \
            -v "$REPO:/opt/nvidia/cuBB" -v "$outdir:/results" \
            -w /opt/nvidia/cuBB/pyaerial "$RDMA_IMG" \
            python3 l1_producer_gdr.py \
                "migmps_l1_${l1_pct}_nrx_${nrx_pct}_t${trial}" \
                "$ITERATIONS" "$tag" >"$outdir/l1.log" 2>&1
        consumer_rc=$(timeout 30 docker wait "$active_consumer")
        docker logs "$active_consumer" >"$outdir/nrx.log" 2>&1 || true
        docker rm -f "$active_consumer" >/dev/null 2>&1 || true
        active_consumer=""
        clean_tag "$tag"
        [[ "$consumer_rc" == 0 ]]
        grep -q '\[NRx-GDR\] ready' "$outdir/nrx.log"
        grep -q '\[NRx-GDR\] shutdown signal' "$outdir/nrx.log"
        grep -q '\[L1-GDR\].*mean=' "$outdir/l1.log"
        date -u +%FT%TZ >"$outdir/COMPLETE"
        echo "[MIG-MPS-GDR] OK l1=$l1_pct nrx=$nrx_pct trial=$trial"
    done
done

cleanup
trap - EXIT INT TERM
date -u +%FT%TZ >"$RESULTS_ROOT/COMPLETE"
echo "[MIG-MPS-GDR] ALL OK"
