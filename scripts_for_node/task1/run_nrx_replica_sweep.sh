#!/usr/bin/env bash
set -Eeuo pipefail

IMG=${IMG:-airan:25-3-final}
ROOT=${ROOT:-/mydata/aerial-cuda-accelerated-ran}
RESULTS=${RESULTS:-/mydata/results/nrx_deep_profile}
ENGINE=${ENGINE:-/results/engines/neural_rx_fp16_4g.trt}
CONTAINER=nrx_replica_sweep_2g
RESTORE_NEEDED=0

destroy_mig() {
    sudo pkill -f nvidia-cuda-mps-control 2>/dev/null || true
    sudo nvidia-smi mig -dci -i 0 >/dev/null 2>&1 || true
    sudo nvidia-smi mig -dgi -i 0 >/dev/null 2>&1 || true
}

restore_topology_a() {
    echo "[NRX-SWEEP-RUNNER] restoring Topology A (4g+3g)" >&2
    destroy_mig
    sudo nvidia-smi -i 0 -mig 1 >/dev/null
    sudo nvidia-smi mig -cgi 5,9 -C -i 0 >/dev/null
    RESTORE_NEEDED=0
}

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    if [[ "$RESTORE_NEEDED" == 1 ]]; then
        restore_topology_a || true
    fi
}
trap cleanup EXIT INT TERM

if docker ps -q | grep -q .; then
    echo "[NRX-SWEEP-RUNNER] refusing MIG reconfiguration: running containers exist" >&2
    docker ps --format '{{.Names}} {{.Status}}' >&2
    exit 1
fi

RESTORE_NEEDED=1
destroy_mig
sudo nvidia-smi -i 0 -mig 1 >/dev/null
sudo nvidia-smi mig -cgi 9,14,14 -C -i 0 >/tmp/nrx_sweep_mig_b.log 2>&1
mapfile -t MIGS_2G < <(nvidia-smi -L | awk '/MIG 2g/{print $6}' | tr -d ')')
if [[ ${#MIGS_2G[@]} -ne 2 ]]; then
    echo "[NRX-SWEEP-RUNNER] expected two 2g instances" >&2
    nvidia-smi -L >&2
    exit 1
fi
MIG_NRX=${MIGS_2G[1]}
nvidia-smi -L | tee "$RESULTS/topology_b_for_nrx_sweep.txt"
echo "[NRX-SWEEP-RUNNER] NRx MIG=$MIG_NRX"

docker run --rm --name "$CONTAINER" --gpus "device=$MIG_NRX" \
    -v "$ROOT":/opt/nvidia/cuBB \
    -v "$RESULTS":/results \
    -w /opt/nvidia/cuBB/pyaerial \
    "$IMG" python3 nrx_replica_sweep.py \
        --engine "$ENGINE" \
        --replicas 1,2,4,8,16 \
        --rates 250,400,500,750,1000 \
        --closed-loop-requests 4000 \
        --open-loop-duration 3 \
        --warmup-rounds 20 \
        --output /results/nrx_replica_sweep_2g.json

restore_topology_a
nvidia-smi -L
echo "[NRX-SWEEP-RUNNER] OK"
