#!/usr/bin/env bash
# Stage 4b supplement: per-op latency for workloads missing in Stage 4 baseline.
# Adds xapp + Forecaster (per-op latency for AI workloads NOT yet covered).
# sat_compute/sat_hbm don't have meaningful per-op (continuous loops); skipped.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-5}"
DURATION="${DURATION:-30}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_per_op_latency_b"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

cgi_for_ai() {
  case "$1" in
    1g) echo "9,14,19" ;; 2g) echo "9,14" ;; 3g) echo "9,9" ;; 4g) echo "9,5" ;;
  esac
}
ai_profile_for() {
  case "$1" in
    1g) echo "1g.5gb" ;; 2g) echo "2g.10gb" ;; 3g) echo "3g.20gb" ;; 4g) echo "4g.20gb" ;;
  esac
}
reconfigure_mig() {
  sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -cgi "$1" -C >/dev/null 2>&1
  sleep 2
}
uuid_for_profile() {
  nvidia-smi -L | grep -E "MIG[[:space:]]+$1[[:space:]]" | grep -oE "MIG-[a-f0-9-]{36}" | head -1
}
uuid_for_profile_nth() {
  nvidia-smi -L | grep -E "MIG[[:space:]]+$1[[:space:]]" | grep -oE "MIG-[a-f0-9-]{36}" | sed -n "${2}p"
}

start_persistent_l1() {
  docker run -d --rm --gpus "\"device=$1\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" \
    --name "l1_bg_b_$$_$(date +%s)_$RANDOM" \
    "$IMAGE" bash /scripts/real_l1_loop.sh bg_l1 20 50
}

script_args_for() {
  local wl=$1 a=$2 b=$3
  case "$wl" in
    xapp)        echo "/workspace/AIRAN_Changjong/experiments/run_xapp_latency.py 0 $DURATION $a" ;;
    forecaster)  echo "/workspace/AIRAN_Changjong/experiments/run_forecaster_latency.py 0 $DURATION $a $b" ;;
  esac
}

run_ai_once() {
  docker run --rm --gpus "\"device=$1\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" python3 $3 > "$2" 2>&1
}

CELLS=(
  # xapp on each partition
  "xapp       1g 16  0"
  "xapp       2g 16  0"
  "xapp       3g 16  0"
  "xapp       4g 16  0"
  # forecaster on each partition
  "forecaster 1g 32  256"
  "forecaster 2g 64  384"
  "forecaster 3g 128 512"
  "forecaster 4g 128 512"
)

run_cell() {
  local wl=$1 part=$2 a=$3 b=$4
  local tag="${wl}_${part}"
  log "===== $tag (arg2=$a arg3=$b) ====="
  reconfigure_mig "$(cgi_for_ai "$part")"
  local l1_uuid; l1_uuid=$(uuid_for_profile "3g.20gb")
  local ai_uuid
  if [[ "$part" == "3g" ]]; then
    ai_uuid=$(uuid_for_profile_nth "3g.20gb" 2)
  else
    ai_uuid=$(uuid_for_profile "$(ai_profile_for "$part")")
  fi
  [[ -z "$l1_uuid" || -z "$ai_uuid" ]] && { log "  UUID missing"; return 1; }
  log "  L1=$l1_uuid AI=$ai_uuid"
  local args; args=$(script_args_for "$wl" "$a" "$b")
  local alone_dir="$OUT_ROOT/${tag}/alone"
  local with_dir="$OUT_ROOT/${tag}/with_l1"
  mkdir -p "$alone_dir" "$with_dir" && chmod 777 "$alone_dir" "$with_dir"
  for i in $(seq 1 $N); do
    log "  alone $i/$N"
    run_ai_once "$ai_uuid" "$alone_dir/run_${i}.log" "$args"
    grep -E "done|json" "$alone_dir/run_${i}.log" | tail -2
  done
  for i in $(seq 1 $N); do
    log "  with_l1 $i/$N"
    local cid; cid=$(start_persistent_l1 "$l1_uuid")
    sleep 8
    run_ai_once "$ai_uuid" "$with_dir/run_${i}.log" "$args"
    grep -E "done|json" "$with_dir/run_${i}.log" | tail -2
    docker kill "$cid" >/dev/null 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    sleep 3
  done
}

START_TS=$(date +%s)
for entry in "${CELLS[@]}"; do
  read -r wl part a b <<< "$entry"
  run_cell "$wl" "$part" "$a" "$b" || log "  cell FAILED: $wl on $part"
done
END_TS=$(date +%s)
log "ai_per_op_latency_b DONE elapsed=$((END_TS - START_TS))s = $(( (END_TS - START_TS)/60 ))m"
