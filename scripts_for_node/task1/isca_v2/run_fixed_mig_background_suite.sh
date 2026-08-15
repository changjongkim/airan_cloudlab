#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed 4g+3g only.  This runner never destroys or creates a MIG instance.
IMG=${IMG:-airan:25-3-final}
REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
PRIMARY_MIG=${PRIMARY_MIG:?set PRIMARY_MIG to the fixed 4g UUID}
SPARE_MIG=${SPARE_MIG:?set SPARE_MIG to the fixed 3g UUID}
WORKLOADS=${WORKLOADS:-"resnet50 bert_base whisper_base qwen_decode"}
POLICIES=${POLICIES:-"naive_share adaptive_reclaim"}
TRACE=${TRACE:-low:500:2.01,burst:1100:3,low:500:3}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-dart_bg}
# CUDA 13/R580 enumerates the two visible MIG instances by their internal
# device/PCI order, not by the NVIDIA_VISIBLE_DEVICES string order: 3g is 0
# and 4g is 1 on this fixed topology.  Keep this explicit and auditable.
PRIMARY_DEVICE=${PRIMARY_DEVICE:-1}
SPARE_DEVICE=${SPARE_DEVICE:-0}

current_background=""
cleanup() {
    if [[ -n "$current_background" ]]; then
        docker stop -t 20 "$current_background" >/dev/null 2>&1 || true
        docker rm -f "$current_background" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

topology=$(nvidia-smi -L)
grep -F "$PRIMARY_MIG" <<<"$topology" | grep -Fq 'MIG 4g.20gb'
grep -F "$SPARE_MIG" <<<"$topology" | grep -Fq 'MIG 3g.20gb'
sudo mkdir -p "$RESULTS_ROOT"
sudo chmod 0777 "$RESULTS_ROOT"
mapping=$(docker run --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="$PRIMARY_MIG,$SPARE_MIG" \
    --entrypoint python3 "$IMG" -c \
    'import cupy as c; print(*(c.cuda.runtime.getDeviceProperties(i)["multiProcessorCount"] for i in range(c.cuda.runtime.getDeviceCount())))')
read -r -a sms <<<"$mapping"
[[ ${#sms[@]} -eq 2 ]]
((sms[PRIMARY_DEVICE] > sms[SPARE_DEVICE]))
printf 'logical0_sm=%s logical1_sm=%s primary_device=%s spare_device=%s\n' \
    "${sms[0]}" "${sms[1]}" "$PRIMARY_DEVICE" "$SPARE_DEVICE" \
    >"$RESULTS_ROOT/DEVICE_MAPPING.txt"

for workload in $WORKLOADS; do
    for policy in $POLICIES; do
        outdir="$RESULTS_ROOT/${workload}/${policy}"
        if [[ -s "$outdir/COMPLETE" ]]; then
            echo "[BACKGROUND-SUITE] skip complete $workload/$policy"
            continue
        fi
        sudo mkdir -p "$outdir/control"
        sudo chmod 0777 "$outdir" "$outdir/control"
        rm -f "$outdir/control/gate" "$outdir/control/background.ready" \
            "$outdir/control/background.state"
        printf '0\n' >"$outdir/control/gate"
        chmod 0666 "$outdir/control/gate"
        safe_workload=${workload//_/-}
        safe_policy=${policy//_/-}
        current_background="${CONTAINER_PREFIX}-${safe_workload}-${safe_policy}"
        docker rm -f "$current_background" >/dev/null 2>&1 || true

        if [[ "$workload" == qwen_decode ]]; then
            docker run -d --name "$current_background" --runtime=nvidia --ipc=host \
                -e NVIDIA_VISIBLE_DEVICES="$SPARE_MIG" \
                -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
                -e PYTHONDONTWRITEBYTECODE=1 \
                -v "$REPO:/opt/nvidia/cuBB" \
                -v /mydata/hf_cache:/mydata/hf_cache \
                -v "$outdir:/results" -w /opt/nvidia/cuBB/pyaerial \
                "$IMG" python3 -B qwen7b_gated.py \
                    --gate-file /results/control/gate \
                    --state-file /results/control/background.state \
                    --ready-file /results/control/background.ready \
                    --output /results/background_timeline.json \
                    --duration 300 --sequence-length 512 --mode decode \
                    --decode-steps-before-reset 128 >/dev/null
        else
            docker run -d --name "$current_background" --runtime=nvidia --ipc=host \
                -e NVIDIA_VISIBLE_DEVICES="$SPARE_MIG" \
                -e PYTHONDONTWRITEBYTECODE=1 \
                -v "$REPO:/opt/nvidia/cuBB" \
                -v "$outdir:/results" -w /opt/nvidia/cuBB/pyaerial \
                "$IMG" python3 -B isca_v2/background_gated.py \
                    --workload "$workload" \
                    --gate-file /results/control/gate \
                    --state-file /results/control/background.state \
                    --ready-file /results/control/background.ready \
                    --output /results/background_timeline.json \
                    --duration 300 >/dev/null
        fi

        for _ in $(seq 1 1200); do
            [[ -s "$outdir/control/background.ready" ]] && break
            if ! docker inspect -f '{{.State.Running}}' "$current_background" \
                    2>/dev/null | grep -qx true; then
                docker logs "$current_background" >"$outdir/background.log" 2>&1 || true
                echo "[BACKGROUND-SUITE] background died $workload/$policy" >&2
                exit 1
            fi
            sleep 0.25
        done
        [[ -s "$outdir/control/background.ready" ]]

        docker run --rm --runtime=nvidia --ipc=host \
            -e NVIDIA_VISIBLE_DEVICES="$PRIMARY_MIG,$SPARE_MIG" \
            -e PYTHONDONTWRITEBYTECODE=1 \
            -v "$REPO:/opt/nvidia/cuBB" \
            -v "$ENGINE_HOST:/nrx_results:ro" \
            -v "$outdir:/results" -w /opt/nvidia/cuBB/pyaerial \
            "$IMG" python3 -B nrx_elastic_pool_timeline.py \
                --engine /nrx_results/engines/neural_rx_fp16_4g.trt \
                --gate-file /results/control/gate \
                --qwen-state-file /results/control/background.state \
                --policy "$policy" --trace "$TRACE" \
                --primary-device "$PRIMARY_DEVICE" \
                --spare-device "$SPARE_DEVICE" \
                --adaptive-high-rate 900 --adaptive-low-rate 650 \
                --output /results/nrx_timeline.json \
            >"$outdir/nrx.log" 2>&1

        docker stop -t 20 "$current_background" >/dev/null 2>&1 || true
        docker logs "$current_background" >"$outdir/background.log" 2>&1 || true
        docker rm -f "$current_background" >/dev/null 2>&1 || true
        current_background=""
        test -s "$outdir/nrx_timeline.json"
        test -s "$outdir/background_timeline.json"
        date -u +%FT%TZ >"$outdir/COMPLETE"
        echo "[BACKGROUND-SUITE] OK $workload/$policy"
    done
done

date -u +%FT%TZ >"$RESULTS_ROOT/COMPLETE"
echo "[BACKGROUND-SUITE] ALL OK"
