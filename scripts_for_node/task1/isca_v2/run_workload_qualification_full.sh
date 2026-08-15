#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
IMG=${IMG:-airan:25-3-final}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT to the full-day campaign directory}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/02_nrx_stack/COMPLETE}
JOB_DIR="$RESULTS_ROOT/03_workload_qualification"
MIG_3G=${MIG_3G:-MIG-80a4659b-f06f-540b-9f4b-1c91f78aaaf3}
FULL_GPU=${FULL_GPU:-GPU-ee09d6cb-2e38-e9e1-9b76-bbc0f8f79b1a}
TRIALS=${TRIALS:-3}
DURATION=${DURATION:-90}
ALPACA_HOST=${ALPACA_HOST:-/mydata/datasets/alpaca/alpaca_data.json}
CIFAR_HOST=${CIFAR_HOST:-/mydata/datasets/cifar10}
TORCH_CACHE=${TORCH_CACHE:-/mydata/torch_cache}
HF_CACHE=${HF_CACHE:-/mydata/hf_cache}

sudo mkdir -p "$JOB_DIR" "$CIFAR_HOST" "$TORCH_CACHE"
sudo chmod 0777 "$RESULTS_ROOT" "$JOB_DIR" "$CIFAR_HOST" "$TORCH_CACHE"
exec 9>"$JOB_DIR/.lock"
flock -n 9 || { echo "[WORKLOAD-FULL] another runner owns $JOB_DIR" >&2; exit 3; }

timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [WORKLOAD-FULL] $*"; }
cleanup() {
    docker rm -f dart-full-workload >/dev/null 2>&1 || true
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

rm -f "$JOB_DIR/FAILED"
log "waiting for dependency $DEPENDENCY"
for _ in $(seq 1 2880); do
    [[ -s "$DEPENDENCY" ]] && break
    if [[ -s "${DEPENDENCY%/*}/FAILED" ]]; then
        echo "[WORKLOAD-FULL] upstream NRx stack failed" >&2
        exit 20
    fi
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || { echo "[WORKLOAD-FULL] dependency timeout" >&2; exit 21; }

touch "$JOB_DIR/RUNNING"
rm -f "$JOB_DIR/COMPLETE" "$JOB_DIR/FAILED"
cp "$0" "$JOB_DIR/runner.captured.sh"
{
    echo "started_utc=$(timestamp)"
    echo "trials=$TRIALS"
    echo "duration=$DURATION"
    echo "mig_3g=$MIG_3G"
    echo "full_gpu=$FULL_GPU"
    docker image inspect --format 'image_id={{.Id}}' "$IMG"
    sha256sum "$PYAERIAL/isca_v2/qwen_prompt_qualification.py" \
        "$PYAERIAL/isca_v2/training_qualification.py" \
        "$PYAERIAL/isca_v2/prepare_workload_data.py"
} >"$JOB_DIR/MANIFEST.txt"

if [[ ! -s "$JOB_DIR/DATA_MANIFEST.json" ]]; then
    log "preparing CIFAR-10 and pretrained ResNet-50 assets"
    docker run --rm --name dart-full-workload \
        -e TORCH_HOME=/torch-cache -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" -v /mydata/datasets:/datasets \
        -v "$TORCH_CACHE:/torch-cache" -v "$JOB_DIR:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/prepare_workload_data.py \
            --dataset-root /datasets/cifar10 \
            --alpaca /datasets/alpaca/alpaca_data.json \
            --output /results/DATA_MANIFEST.json
fi

run_qwen() {
    local topology=$1 device=$2 mode=$3 duty=$4 trial=$5
    local stem="qwen_${topology}_${mode}_d${duty/./}_t${trial}"
    local output="$JOB_DIR/$stem.json"
    if python3 - "$output" "$DURATION" <<'PY'
import json, os, sys
path, duration = sys.argv[1], float(sys.argv[2])
if not os.path.isfile(path): raise SystemExit(1)
v=json.load(open(path)); assert v.get("pass") is True
assert v["duration_actual_s"] >= duration * .99 and v["service_gpu_ms"]["n"] > 0
PY
    then log "SKIP $stem"; return; fi
    rm -f "$output" "$output.tmp"
    log "START $stem"
    timeout 600 docker run --rm --name dart-full-workload --runtime=nvidia \
        --ipc=host -e NVIDIA_VISIBLE_DEVICES="$device" \
        -e HF_HOME=/hf-cache -e TRANSFORMERS_OFFLINE=1 \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" -v "$HF_CACHE:/hf-cache" \
        -v /mydata/datasets:/datasets:ro -v "$JOB_DIR:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/qwen_prompt_qualification.py \
            --dataset /datasets/alpaca/alpaca_data.json \
            --mode "$mode" --duty-cycle "$duty" --duration "$DURATION" \
            --target-tokens 512 --trial "$trial" --output "/results/$stem.json" \
        2>&1 | tee "$JOB_DIR/$stem.log"
}

run_training() {
    local topology=$1 device=$2 batch=$3 duty=$4 trial=$5
    local stem="training_${topology}_mb${batch}_d${duty/./}_t${trial}"
    local output="$JOB_DIR/$stem.json"
    if python3 - "$output" "$DURATION" <<'PY'
import json, os, sys
path, duration = sys.argv[1], float(sys.argv[2])
if not os.path.isfile(path): raise SystemExit(1)
v=json.load(open(path)); assert v.get("pass") is True
assert v["duration_actual_s"] >= duration * .99 and v["unit_gpu_ms"]["n"] > 0
PY
    then log "SKIP $stem"; return; fi
    rm -f "$output" "$output.tmp"
    log "START $stem"
    timeout 600 docker run --rm --name dart-full-workload --runtime=nvidia \
        --ipc=host -e NVIDIA_VISIBLE_DEVICES="$device" \
        -e TORCH_HOME=/torch-cache -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" -v "$TORCH_CACHE:/torch-cache:ro" \
        -v "$CIFAR_HOST:/datasets/cifar10:ro" -v "$JOB_DIR:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/training_qualification.py \
            --dataset-root /datasets/cifar10 --microbatch "$batch" \
            --duty-cycle "$duty" --duration "$DURATION" \
            --trial "$trial" --output "/results/$stem.json" \
        2>&1 | tee "$JOB_DIR/$stem.log"
}

for topology in full 3g; do
    if [[ "$topology" == full ]]; then device=$FULL_GPU; else device=$MIG_3G; fi
    for trial in $(seq 1 "$TRIALS"); do
        for mode in prefill decode; do
            for duty in 0.5 0.9; do run_qwen "$topology" "$device" "$mode" "$duty" "$trial"; done
        done
        for batch in 1 4; do
            for duty in 0.5 0.9; do run_training "$topology" "$device" "$batch" "$duty" "$trial"; done
        done
    done
done

python3 - "$JOB_DIR" "$TRIALS" <<'PY'
import json, os, sys
root, trials = sys.argv[1], int(sys.argv[2])
expected = trials * 2 * 4
qwen=[]; training=[]
for name in os.listdir(root):
    if not name.endswith(".json") or name == "DATA_MANIFEST.json": continue
    value=json.load(open(os.path.join(root,name)))
    if name.startswith("qwen_"): qwen.append(value)
    elif name.startswith("training_"): training.append(value)
assert len(qwen) == expected, (len(qwen), expected)
assert len(training) == expected, (len(training), expected)
assert all(v.get("pass") is True for v in qwen+training)
print(f"[WORKLOAD-FULL] validated qwen={len(qwen)} training={len(training)}")
PY

rm -f "$JOB_DIR/RUNNING"
timestamp >"$JOB_DIR/COMPLETE"
log "ALL WORKLOAD QUALIFICATION TRIALS COMPLETE"
