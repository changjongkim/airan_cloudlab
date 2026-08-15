#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/mydata/aerial-cuda-accelerated-ran}
PYAERIAL="$REPO/pyaerial"
ENGINE_HOST=${ENGINE_HOST:-/mydata/results/nrx_deep_profile/engines}
IMG=${IMG:-airan:25-3-final}
RESULTS_ROOT=${RESULTS_ROOT:?set RESULTS_ROOT}
DEPENDENCY=${DEPENDENCY:-$RESULTS_ROOT/05_fiveway_compute/COMPLETE}
JOB_DIR="$RESULTS_ROOT/06_dart_scheme"
MIG_3G=${MIG_3G:-MIG-80a4659b-f06f-540b-9f4b-1c91f78aaaf3}
GPU1=${GPU1:-GPU-86123961-c0e3-6d2b-4aeb-b08504fe6647}
GPU2=${GPU2:-GPU-7cfa813b-475f-c71a-5b15-7e352d2ded67}
TRIALS=${TRIALS:-5}
DURATION=${DURATION:-60}

sudo mkdir -p "$JOB_DIR"
sudo chmod 0777 "$RESULTS_ROOT" "$JOB_DIR"
exec 9>"$JOB_DIR/.lock"
flock -n 9 || exit 3
timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] [DART-SCHEME-FULL] $*"; }
cleanup() { docker rm -f dart-full-scheme >/dev/null 2>&1 || true; }
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
for _ in $(seq 1 21600); do
    [[ -s "$DEPENDENCY" ]] && break
    [[ -s "${DEPENDENCY%/*}/FAILED" ]] && exit 20
    sleep 5
done
[[ -s "$DEPENDENCY" ]] || exit 21
touch "$JOB_DIR/RUNNING"
rm -f "$JOB_DIR/COMPLETE"
cp "$0" "$JOB_DIR/runner.captured.sh"

AGGREGATE_CAPACITY=$(python3 - "$RESULTS_ROOT/04_nrx_capacity" <<'PY'
import glob,json,statistics,sys
values={"3g":[],"full":[]}
for topology in values:
 for path in glob.glob(sys.argv[1]+f"/capacity_{topology}_t*.json"):
  value=json.load(open(path)); one=next(x for x in value["configurations"] if x["replicas"]==1)
  values[topology].append(one["saturation"]["throughput_slots_per_s"])
assert all(len(x)==5 for x in values.values())
print(statistics.median(values["3g"])+2*statistics.median(values["full"]))
PY
)
RATES=$(python3 - "$AGGREGATE_CAPACITY" <<'PY'
import sys
c=float(sys.argv[1]); print(",".join(str(c*x) for x in (.50,.85,.95,1.05,1.20)))
PY
)
log "aggregate compute capacity=$AGGREGATE_CAPACITY rates=$RATES"

{
    echo "started_utc=$(timestamp)"; echo "trials=$TRIALS"; echo "duration=$DURATION"
    echo "aggregate_capacity=$AGGREGATE_CAPACITY"; echo "rates=$RATES"
    echo "hardware_scope=actual_nrx_compute_and_queue_no_l1_or_transport"
    sha256sum "$PYAERIAL/isca_v2/dart_runtime.py" \
        "$PYAERIAL/isca_v2/test_dart_runtime.py" \
        "$PYAERIAL/isca_v2/nrx_dart_policy_sweep.py" \
        "$PYAERIAL/isca_v2/compile_full_dart_profile.py"
} >"$JOB_DIR/MANIFEST.txt"

mkdir -p "$JOB_DIR/hardware_routing" "$JOB_DIR/faults" \
    "$JOB_DIR/profiles" "$JOB_DIR/control_replay"
chmod 0777 "$JOB_DIR"/{hardware_routing,faults,profiles,control_replay}
for trial in $(seq 1 "$TRIALS"); do
    output="$JOB_DIR/hardware_routing/trial_${trial}.json"
    if python3 - "$output" <<'PY'
import json,os,sys
if not os.path.isfile(sys.argv[1]): raise SystemExit(1)
v=json.load(open(sys.argv[1])); assert v["outputs_finite"] and len(v["runs"])==20
assert all(len(x["raw"]["latency_ms"]) == x["requests"] for x in v["runs"])
PY
    then log "SKIP hardware routing trial=$trial"; else
        rm -f "$output" "$output.tmp"
        log "START hardware routing trial=$trial"
        timeout 9000 docker run --rm --name dart-full-scheme --runtime=nvidia \
            --ipc=host -e NVIDIA_VISIBLE_DEVICES="$MIG_3G,$GPU1,$GPU2" \
            -e PYTHONDONTWRITEBYTECODE=1 -v "$REPO:/opt/nvidia/cuBB" \
            -v "$ENGINE_HOST:/engines:ro" -v "$JOB_DIR:/results" \
            -w /opt/nvidia/cuBB/pyaerial "$IMG" \
            python3 isca_v2/nrx_dart_policy_sweep.py \
                --engine /engines/neural_rx_fp16_4g.trt \
                --policies round_robin,shortest_queue,predicted_finish,hybrid_finish \
                --rates "$RATES" --duration "$DURATION" \
                --warmup 50 --calibration-requests 1000 \
                --output "/results/hardware_routing/trial_${trial}.json" \
            >"$JOB_DIR/hardware_routing/trial_${trial}.log" 2>&1
    fi
done

for trial in $(seq 1 "$TRIALS"); do
    seed=$((20260813 + trial))
    python3 "$PYAERIAL/isca_v2/test_dart_runtime.py" \
        --fault-iterations 10000 --seed "$seed" \
        --output "$JOB_DIR/faults/trial_${trial}.json" \
        >"$JOB_DIR/faults/trial_${trial}.log" 2>&1
done

LOW_RATE=$(python3 -c "print($AGGREGATE_CAPACITY * .5)")
BURST_RATE=$(python3 -c "print($AGGREGATE_CAPACITY * 1.2)")
TRACE="low:${LOW_RATE}:100,burst:${BURST_RATE}:50,low:${LOW_RATE}:100"
for guard in 0 5 10 20; do
    profile="$JOB_DIR/profiles/guard_${guard}.json"
    python3 "$PYAERIAL/isca_v2/compile_full_dart_profile.py" \
        --root "$RESULTS_ROOT" --guard-pct "$guard" --endpoints 3 \
        --output "$profile"
    for trial in $(seq 1 "$TRIALS"); do
        seed=$((20260813 + trial))
        python3 "$PYAERIAL/isca_v2/dart_policy_replay.py" \
            --profile "$profile" --trace "$TRACE" \
            --policies S0,S1,S2,S3,S6,S7,S8 \
            --deadline-ms 5 --conventional-ms 1 --qmax-us 500 \
            --seed "$seed" \
            --output "$JOB_DIR/control_replay/guard_${guard}_t${trial}.json" \
            >"$JOB_DIR/control_replay/guard_${guard}_t${trial}.log" 2>&1
    done
done

python3 - "$JOB_DIR" "$TRIALS" <<'PY'
import glob,json,os,sys
root,trials=sys.argv[1],int(sys.argv[2])
hardware=glob.glob(root+"/hardware_routing/*.json")
faults=glob.glob(root+"/faults/*.json")
replay=glob.glob(root+"/control_replay/*.json")
assert len(hardware)==trials and len(faults)==trials and len(replay)==trials*4
assert all(json.load(open(x))["outputs_finite"] for x in hardware)
for path in faults:
    value=json.load(open(path)); storm=value.get("fallback_storm", {})
    assert value["pass"] and storm.get("pass")
    assert storm.get("over_admitted")==0 and storm.get("reservation_leaks")==0
print(f"[DART-SCHEME-FULL] validated hardware={len(hardware)} faults={len(faults)} replay={len(replay)}")
PY
rm -f "$JOB_DIR/RUNNING"
timestamp >"$JOB_DIR/COMPLETE"
log "DART SCHEME GATES COMPLETE"
