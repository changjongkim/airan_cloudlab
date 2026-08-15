#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT to the full-day campaign directory}
JOB_DIR="$RESULTS_ROOT/02_nrx_stack"
MIG_4G=${MIG_4G:-MIG-dae3f173-7b15-594b-bc80-6cef80687a56}
TRIALS=${TRIALS:-5}
ITERATIONS=${ITERATIONS:-10000}
WARMUP=${WARMUP:-100}

sudo mkdir -p "$JOB_DIR"
sudo chmod 0777 "$RESULTS_ROOT" "$JOB_DIR"
exec 9>"$JOB_DIR/.lock"
flock -n 9 || { echo "[NRX-FULL] another runner owns $JOB_DIR" >&2; exit 3; }

timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [NRX-FULL] $*"; }

cleanup() {
    docker rm -f dart-full-nrx >/dev/null 2>&1 || true
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

topology=$(nvidia-smi -L)
grep -F "$MIG_4G" <<<"$topology" | grep -Fq 'MIG 4g.20gb'
if docker ps --format '{{.Names}}' | grep -qx dart-full-nrx; then
    echo "[NRX-FULL] scoped container is already running" >&2
    exit 4
fi

{
    echo "started_utc=$(timestamp)"
    echo "trials=$TRIALS"
    echo "iterations=$ITERATIONS"
    echo "warmup=$WARMUP"
    echo "mig_4g=$MIG_4G"
    echo "image=$IMG"
    docker image inspect --format 'image_id={{.Id}}' "$IMG"
    sha256sum "$PYAERIAL/isca_v2/nrx_stack_full.py" \
        "$PYAERIAL/nrx_trt_direct.py" \
        "$ENGINE_HOST/neural_rx_fp16_4g.trt"
    nvidia-smi -L
} >"$JOB_DIR/MANIFEST.txt"
cp "$0" "$JOB_DIR/runner.captured.sh"

touch "$JOB_DIR/RUNNING"
rm -f "$JOB_DIR/COMPLETE" "$JOB_DIR/FAILED"
for trial in $(seq 1 "$TRIALS"); do
    output="$JOB_DIR/trial_${trial}.json"
    if python3 - "$output" "$ITERATIONS" <<'PY'
import json, os, sys
path, iterations = sys.argv[1], int(sys.argv[2])
if not os.path.isfile(path):
    raise SystemExit(1)
value = json.load(open(path, encoding="utf-8"))
assert value.get("pass") is True
assert value["iterations"] == iterations
assert len(value["variants"]) == 5
for variant in value["variants"].values():
    assert variant["metrics"]["gpu_ms"]["n"] == iterations
PY
    then
        log "SKIP trial=$trial already valid"
        continue
    fi
    rm -f "$output" "$output.tmp"
    log "START trial=$trial/$TRIALS iterations=$ITERATIONS"
    timeout 7200 docker run --rm --name dart-full-nrx --runtime=nvidia \
        --ipc=host -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" \
        -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$REPO:/opt/nvidia/cuBB" \
        -v "$ENGINE_HOST:/engines:ro" \
        -v "$JOB_DIR:/results" \
        -w /opt/nvidia/cuBB/pyaerial "$IMG" \
        python3 isca_v2/nrx_stack_full.py \
            --engine /engines/neural_rx_fp16_4g.trt \
            --warmup "$WARMUP" --iterations "$ITERATIONS" \
            --trial "$trial" --output "/results/trial_${trial}.json" \
        2>&1 | tee "$JOB_DIR/trial_${trial}.log"
    log "DONE trial=$trial/$TRIALS"
done

python3 - "$JOB_DIR" "$TRIALS" "$ITERATIONS" <<'PY'
import csv, json, os, sys
root, trials, iterations = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
for trial in range(1, trials + 1):
    path = os.path.join(root, f"trial_{trial}.json")
    value = json.load(open(path, encoding="utf-8"))
    assert value.get("pass") is True and value["iterations"] == iterations
    for name, variant in value["variants"].items():
        metric = variant["metrics"]["gpu_ms"]
        assert metric["n"] == iterations
        rows.append({"trial": trial, "variant": name, **metric})
temporary = os.path.join(root, "NRX_STACK.csv.tmp")
with open(temporary, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, os.path.join(root, "NRX_STACK.csv"))
PY

rm -f "$JOB_DIR/RUNNING"
timestamp >"$JOB_DIR/COMPLETE"
log "ALL TRIALS COMPLETE"
