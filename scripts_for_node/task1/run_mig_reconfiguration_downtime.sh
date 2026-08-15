#!/usr/bin/env bash
set -euo pipefail

IMG=${IMG:-airan:25-3-final}
TRIALS=${TRIALS:-3}
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/drain_free/mig_reconfiguration_downtime}
REPO=/mydata/aerial-cuda-accelerated-ran
ENGINE=/mydata/results/nrx_deep_profile/engines/neural_rx_fp16_full.trt
CONTAINER=drainfree_nrx_resident
CURRENT_TOPOLOGY=A

now_ns() { date +%s%N; }

destroy_instances() {
    sudo nvidia-smi mig -dci -i 0 >/dev/null
    sudo nvidia-smi mig -dgi -i 0 >/dev/null
}

create_a() {
    destroy_instances
    sudo nvidia-smi mig -cgi 5,9 -C -i 0 >/dev/null
    CURRENT_TOPOLOGY=A
}

create_b() {
    destroy_instances
    sudo nvidia-smi mig -cgi 9,14,14 -C -i 0 >/dev/null
    CURRENT_TOPOLOGY=B
}

find_mig_uuid() {
    local profile=$1
    nvidia-smi -L | awk -v p="$profile" '
        index($0, "MIG " p) {sub(/^.*UUID: /, ""); sub(/\).*$/, ""); print; exit}'
}

stop_service() {
    docker stop -t 2 "$CONTAINER" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

restore() {
    stop_service
    if [[ "$CURRENT_TOPOLOGY" != A ]]; then
        create_a || true
    fi
}
trap restore EXIT INT TERM

start_service() {
    local uuid=$1 label=$2
    local ready="$RESULTS_ROOT/${label}.ready"
    rm -f "$ready"
    docker run -d --name "$CONTAINER" --runtime=nvidia --ipc=host \
        -e NVIDIA_VISIBLE_DEVICES="$uuid" -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" \
        -v /mydata/results/nrx_deep_profile:/nrx_results:ro \
        -v "$RESULTS_ROOT:/results" -w /opt/nvidia/cuBB/pyaerial \
        "$IMG" python3 -B nrx_resident_service.py \
            --engine /nrx_results/engines/neural_rx_fp16_full.trt \
            --ready-file "/results/${label}.ready" \
            --output "/results/${label}_startup.json" >/dev/null
    for _ in $(seq 1 600); do
        [[ -f "$ready" ]] && return 0
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
            docker logs "$CONTAINER" >&2
            return 1
        fi
        sleep 0.05
    done
    return 1
}

sudo mkdir -p "$RESULTS_ROOT"
sudo chmod 0777 "$RESULTS_ROOT"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader | grep -q Enabled
nvidia-smi -L >"$RESULTS_ROOT/topology_before.txt"

create_a
A_UUID=$(find_mig_uuid 4g.20gb)
[[ -n "$A_UUID" ]]
start_service "$A_UUID" initial_a

printf 'trial,direction,stop_ms,destroy_ms,create_ms,start_to_ready_ms,total_outage_ms\n' \
    >"$RESULTS_ROOT/reconfiguration.csv"

for trial in $(seq 1 "$TRIALS"); do
    total_start=$(now_ns)
    step_start=$total_start
    stop_service
    stop_end=$(now_ns)
    destroy_instances
    destroy_end=$(now_ns)
    sudo nvidia-smi mig -cgi 9,14,14 -C -i 0 >/dev/null
    CURRENT_TOPOLOGY=B
    create_end=$(now_ns)
    B_UUID=$(find_mig_uuid 2g.10gb)
    [[ -n "$B_UUID" ]]
    start_service "$B_UUID" "trial${trial}_b"
    ready_end=$(now_ns)
    printf '%s,A_to_B,%.3f,%.3f,%.3f,%.3f,%.3f\n' \
        "$trial" "$((stop_end-step_start))e-6" \
        "$((destroy_end-stop_end))e-6" "$((create_end-destroy_end))e-6" \
        "$((ready_end-create_end))e-6" "$((ready_end-total_start))e-6" \
        >>"$RESULTS_ROOT/reconfiguration.csv"

    total_start=$(now_ns)
    step_start=$total_start
    stop_service
    stop_end=$(now_ns)
    destroy_instances
    destroy_end=$(now_ns)
    sudo nvidia-smi mig -cgi 5,9 -C -i 0 >/dev/null
    CURRENT_TOPOLOGY=A
    create_end=$(now_ns)
    A_UUID=$(find_mig_uuid 4g.20gb)
    [[ -n "$A_UUID" ]]
    start_service "$A_UUID" "trial${trial}_a"
    ready_end=$(now_ns)
    printf '%s,B_to_A,%.3f,%.3f,%.3f,%.3f,%.3f\n' \
        "$trial" "$((stop_end-step_start))e-6" \
        "$((destroy_end-stop_end))e-6" "$((create_end-destroy_end))e-6" \
        "$((ready_end-create_end))e-6" "$((ready_end-total_start))e-6" \
        >>"$RESULTS_ROOT/reconfiguration.csv"
done

stop_service
nvidia-smi -L >"$RESULTS_ROOT/topology_after.txt"
trap - EXIT INT TERM
echo "[MIG-RECONFIG] OK results=$RESULTS_ROOT"
