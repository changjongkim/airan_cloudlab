#!/usr/bin/env bash
set -euo pipefail

# Drain-free resident NRx-pool experiment on the already configured 4g+3g
# instances of one physical A100.  This runner never changes MIG geometry.

IMG=${IMG:-airan:25-3-final}
PRIMARY_MIG=${PRIMARY_MIG:?set PRIMARY_MIG to the fixed 4g MIG UUID}
SPARE_MIG=${SPARE_MIG:?set SPARE_MIG to the fixed 3g MIG UUID}
POLICY=${POLICY:-adaptive_reclaim}
TRACE=${TRACE:-low:800:2.01,burst:1100:3,low:800:3}
QWEN_MODE=${QWEN_MODE:-decode}
QWEN_SEQUENCE_LENGTH=${QWEN_SEQUENCE_LENGTH:-512}
QWEN_DECODE_STEPS_BEFORE_RESET=${QWEN_DECODE_STEPS_BEFORE_RESET:-128}
PRIMARY_DEVICE=${PRIMARY_DEVICE:-1}
SPARE_DEVICE=${SPARE_DEVICE:-0}
ADAPTIVE_HIGH_RATE=${ADAPTIVE_HIGH_RATE:-900}
ADAPTIVE_LOW_RATE=${ADAPTIVE_LOW_RATE:-700}
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/drain_free/fixed_mig_${POLICY}}
REPO=/mydata/aerial-cuda-accelerated-ran
QWEN_NAME=fixed_mig_elastic_qwen

cleanup() {
    docker stop -t 20 "$QWEN_NAME" >/dev/null 2>&1 || true
    docker logs "$QWEN_NAME" >"$RESULTS_ROOT/qwen.log" 2>&1 || true
    docker rm -f "$QWEN_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sudo mkdir -p "$RESULTS_ROOT" "$RESULTS_ROOT/control"
sudo chmod 0777 "$RESULTS_ROOT" "$RESULTS_ROOT/control"
rm -f "$RESULTS_ROOT/control/gate" "$RESULTS_ROOT/control/qwen.ready" \
    "$RESULTS_ROOT/control/qwen.state"
printf '0\n' >"$RESULTS_ROOT/control/gate"
chmod 0666 "$RESULTS_ROOT/control/gate"

topology=$(nvidia-smi -L)
grep -Fq "$PRIMARY_MIG" <<<"$topology"
grep -Fq "$SPARE_MIG" <<<"$topology"
grep -F "$PRIMARY_MIG" <<<"$topology" | grep -Fq 'MIG 4g.20gb'
grep -F "$SPARE_MIG" <<<"$topology" | grep -Fq 'MIG 3g.20gb'

docker rm -f "$QWEN_NAME" >/dev/null 2>&1 || true
docker run -d --name "$QWEN_NAME" --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$SPARE_MIG" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$REPO:/opt/nvidia/cuBB" -v /mydata/hf_cache:/mydata/hf_cache \
    -v "$RESULTS_ROOT:/results" -w /opt/nvidia/cuBB/pyaerial \
    "$IMG" python3 -B qwen7b_gated.py \
        --gate-file /results/control/gate \
        --state-file /results/control/qwen.state \
        --ready-file /results/control/qwen.ready \
        --output /results/qwen_timeline.json --duration 300 \
        --sequence-length "$QWEN_SEQUENCE_LENGTH" --mode "$QWEN_MODE" \
        --decode-steps-before-reset "$QWEN_DECODE_STEPS_BEFORE_RESET" >/dev/null

for _ in $(seq 1 600); do
    [[ -f "$RESULTS_ROOT/control/qwen.ready" ]] && break
    if ! docker inspect -f '{{.State.Running}}' "$QWEN_NAME" 2>/dev/null \
        | grep -qx true; then
        docker logs "$QWEN_NAME" >&2
        exit 1
    fi
    sleep 0.25
done
[[ -f "$RESULTS_ROOT/control/qwen.ready" ]]

docker run --rm --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$PRIMARY_MIG,$SPARE_MIG" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$REPO:/opt/nvidia/cuBB" \
    -v /mydata/results/nrx_deep_profile:/nrx_results:ro \
    -v "$RESULTS_ROOT:/results" -w /opt/nvidia/cuBB/pyaerial \
    "$IMG" python3 -B nrx_elastic_pool_timeline.py \
        --engine /nrx_results/engines/neural_rx_fp16_4g.trt \
        --gate-file /results/control/gate \
        --qwen-state-file /results/control/qwen.state \
        --policy "$POLICY" --trace "$TRACE" \
        --primary-device "$PRIMARY_DEVICE" --spare-device "$SPARE_DEVICE" \
        --adaptive-high-rate "$ADAPTIVE_HIGH_RATE" \
        --adaptive-low-rate "$ADAPTIVE_LOW_RATE" \
        --output /results/nrx_timeline.json | tee "$RESULTS_ROOT/nrx.log"

cleanup
trap - EXIT INT TERM
echo "[FIXED-MIG-ELASTIC] OK policy=$POLICY results=$RESULTS_ROOT"
