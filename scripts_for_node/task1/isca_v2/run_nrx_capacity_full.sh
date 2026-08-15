#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT to the full-day campaign directory}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/03_workload_qualification/COMPLETE}
JOB_DIR="$RESULTS_ROOT/04_nrx_capacity"
MIG_4G=${MIG_4G:-MIG-dae3f173-7b15-594b-bc80-6cef80687a56}
MIG_3G=${MIG_3G:-MIG-80a4659b-f06f-540b-9f4b-1c91f78aaaf3}
FULL_GPU=${FULL_GPU:-GPU-ee09d6cb-2e38-e9e1-9b76-bbc0f8f79b1a}
TRIALS=${TRIALS:-5}
DURATION=${DURATION:-60}

sudo mkdir -p "$JOB_DIR"
sudo chmod 0777 "$RESULTS_ROOT" "$JOB_DIR"
exec 9>"$JOB_DIR/.lock"
flock -n 9 || { echo "[CAPACITY-FULL] another runner owns $JOB_DIR" >&2; exit 3; }
timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [CAPACITY-FULL] $*"; }
cleanup() { docker rm -f dart-full-capacity >/dev/null 2>&1 || true; }
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
for _ in $(seq 1 5760); do
    [[ -s "$DEPENDENCY" ]] && break
    if [[ -s "${DEPENDENCY%/*}/FAILED" ]]; then exit 20; fi
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || exit 21

touch "$JOB_DIR/RUNNING"
rm -f "$JOB_DIR/COMPLETE" "$JOB_DIR/FAILED"
cp "$0" "$JOB_DIR/runner.captured.sh"
{
    echo "started_utc=$(timestamp)"
    echo "trials=$TRIALS"
    echo "duration_per_window_s=$DURATION"
    echo "replicas=1,2,4,8"
    echo "load_fractions=0.50,0.85,0.95,1.05"
    docker image inspect --format 'image_id={{.Id}}' "$IMG"
    sha256sum "$PYAERIAL/isca_v2/nrx_capacity_full.py" \
        "$PYAERIAL/nrx_trt_direct.py" "$ENGINE_HOST"/*.trt
} >"$JOB_DIR/MANIFEST.txt"

for topology in 4g 3g full; do
    case "$topology" in
        4g) device=$MIG_4G; engine=neural_rx_fp16_4g.trt ;;
        3g) device=$MIG_3G; engine=neural_rx_fp16_4g.trt ;;
        full) device=$FULL_GPU; engine=neural_rx_fp16_full.trt ;;
    esac
    for trial in $(seq 1 "$TRIALS"); do
        stem="capacity_${topology}_t${trial}"
        output="$JOB_DIR/$stem.json"
        if python3 - "$output" "$topology" <<'PY'
import json, os, sys
path, topology = sys.argv[1:]
if not os.path.isfile(path): raise SystemExit(1)
v=json.load(open(path)); assert v.get("pass") is True and v["topology"] == topology
assert [x["replicas"] for x in v["configurations"]] == [1,2,4,8]
assert all(len(x["open_loop"]) == 4 for x in v["configurations"])
PY
        then log "SKIP $stem"; continue; fi
        rm -f "$output" "$output.tmp"
        log "START $stem"
        timeout 2400 docker run --rm --name dart-full-capacity --runtime=nvidia \
            --ipc=host -e NVIDIA_VISIBLE_DEVICES="$device" \
            -e PYTHONDONTWRITEBYTECODE=1 \
            -v "$REPO:/opt/nvidia/cuBB" -v "$ENGINE_HOST:/engines:ro" \
            -v "$JOB_DIR:/results" -w /opt/nvidia/cuBB/pyaerial "$IMG" \
            python3 isca_v2/nrx_capacity_full.py \
                --engine "/engines/$engine" --topology "$topology" \
                --replicas 1,2,4,8 --load-fractions 0.50,0.85,0.95,1.05 \
                --duration "$DURATION" --warmup 50 --trial "$trial" \
                --output "/results/$stem.json" \
            2>&1 | tee "$JOB_DIR/$stem.log"
    done
done

python3 - "$JOB_DIR" "$TRIALS" <<'PY'
import json, os, sys
root, trials = sys.argv[1], int(sys.argv[2])
paths=[x for x in os.listdir(root) if x.startswith("capacity_") and x.endswith(".json")]
assert len(paths) == trials*3, (len(paths), trials*3)
for path in paths:
    value=json.load(open(os.path.join(root,path)))
    assert value.get("pass") is True and len(value["configurations"]) == 4
print(f"[CAPACITY-FULL] validated files={len(paths)}")
PY
rm -f "$JOB_DIR/RUNNING"
timestamp >"$JOB_DIR/COMPLETE"
log "ALL CAPACITY TRIALS COMPLETE"
