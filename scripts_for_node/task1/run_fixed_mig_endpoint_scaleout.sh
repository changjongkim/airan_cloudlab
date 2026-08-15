#!/usr/bin/env bash
set -euo pipefail

IMG=${IMG:-airan:25-3-final}
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/drain_free/fixed_mig_endpoint_scaleout}
REPO=/mydata/aerial-cuda-accelerated-ran
ENGINE_HOST=/mydata/results/nrx_deep_profile
QWEN_NAME=fixedmig_sibling_qwen
GPU1_CHANGED=0
GPU2_CHANGED=0

cleanup() {
    docker rm -f "$QWEN_NAME" >/dev/null 2>&1 || true
    for gpu in 1 2; do
        sudo nvidia-smi mig -dci -i "$gpu" >/dev/null 2>&1 || true
        sudo nvidia-smi mig -dgi -i "$gpu" >/dev/null 2>&1 || true
        sudo nvidia-smi -i "$gpu" -mig 0 >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT INT TERM

mig_uuid() {
    local gpu=$1 profile=$2
    nvidia-smi -L | awk -v target="GPU ${gpu}:" -v profile="$profile" '
        /^GPU [0-9]+:/ {inside = index($0, target) == 1}
        inside && index($0, "MIG " profile) {
            sub(/^.*UUID: /, ""); sub(/\).*$/, ""); print; exit
        }'
}

sudo mkdir -p "$RESULTS_ROOT"
sudo chmod 0777 "$RESULTS_ROOT"
docker rm -f "$QWEN_NAME" >/dev/null 2>&1 || true
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1,2 | sed '/^$/d')"
nvidia-smi -L >"$RESULTS_ROOT/topology_before.txt"

for gpu in 1 2; do
    sudo nvidia-smi -i "$gpu" -mig 1 >/dev/null
    sudo nvidia-smi mig -cgi 5,9 -C -i "$gpu" >/dev/null
done
GPU1_CHANGED=1
GPU2_CHANGED=1
nvidia-smi -L >"$RESULTS_ROOT/topology_mig.txt"

G0_4G=$(mig_uuid 0 4g.20gb)
G1_4G=$(mig_uuid 1 4g.20gb)
G1_3G=$(mig_uuid 1 3g.20gb)
G2_4G=$(mig_uuid 2 4g.20gb)
for value in "$G0_4G" "$G1_4G" "$G1_3G" "$G2_4G"; do
    [[ -n "$value" ]]
done

run_sweep() {
    local visible=$1 counts=$2 rates=$3 output=$4 log=$5
    docker run --rm --runtime=nvidia --ipc=host \
        -e NVIDIA_VISIBLE_DEVICES="$visible" -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" \
        -v "$ENGINE_HOST:/nrx_results:ro" \
        -v "$RESULTS_ROOT:/results" -w /opt/nvidia/cuBB/pyaerial \
        "$IMG" python3 -B nrx_independent_endpoint_sweep.py \
            --engine /nrx_results/engines/neural_rx_fp16_4g.trt \
            --endpoint-counts "$counts" --rates "$rates" \
            --closed-loop-requests 3000 --open-loop-duration 2 \
            --output "/results/$output" | tee "$RESULTS_ROOT/$log"
}

# Exact single-4g baseline on GPU1 before its isolated sibling is loaded.
run_sweep "$G1_4G" 1 "500,700,800" \
    baseline_gpu1_4g.json baseline_gpu1_4g.log

docker run -d --name "$QWEN_NAME" --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$G1_3G" \
    -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/mydata/hf_cache \
    -v /mydata/hf_cache:/mydata/hf_cache \
    -v "$REPO/pyaerial:/work" -w /work "$IMG" \
    python3 qwen7b_stress.py 240 >/dev/null
for _ in $(seq 1 600); do
    docker logs "$QWEN_NAME" 2>&1 | grep -q '\[Qwen\] HBM=' && break
    if ! docker inspect -f '{{.State.Running}}' "$QWEN_NAME" 2>/dev/null | grep -qx true; then
        docker logs "$QWEN_NAME" >&2
        exit 1
    fi
    sleep 0.25
done
docker logs "$QWEN_NAME" 2>&1 | grep -q '\[Qwen\] HBM='

# GPU1 is listed first, so the one-endpoint configuration is the exact sibling
# test; additional isolated 4g endpoints are then added without MIG changes.
run_sweep "$G1_4G,$G0_4G,$G2_4G" "1,2,3" \
    "500,700,800,1000,1400,1500,1900,2100,2300" \
    sibling_qwen_scaleout.json sibling_qwen_scaleout.log

docker stop -t 20 "$QWEN_NAME" >/dev/null
docker logs "$QWEN_NAME" >"$RESULTS_ROOT/qwen.log" 2>&1
docker rm -f "$QWEN_NAME" >/dev/null

cleanup
trap - EXIT INT TERM
nvidia-smi -L >"$RESULTS_ROOT/topology_after.txt"
# GPU0 must remain 4g+3g and GPUs 1/2 must return to full-GPU mode.
grep -q 'MIG 4g.20gb' "$RESULTS_ROOT/topology_after.txt"
[[ $(grep -c '^GPU [12]:' "$RESULTS_ROOT/topology_after.txt") -eq 2 ]]
echo "[FIXED-MIG-SCALEOUT] OK results=$RESULTS_ROOT"
