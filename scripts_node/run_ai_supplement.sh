#!/usr/bin/env bash
# Supplement matrix v2: ResNet-50 (fp16) + Traffic forecaster (Informer-lite)
# Each partition × 2 batch-tier (light, saturating) × alone/with_l1.
# 16 cells total.
#
# Output: results/$DATE_DIR/ai_supplement/<wl>_<part>_<tier>/<setup>/run_<i>.log

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

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_supplement"
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
    1g) echo "1g.5gb"  ;; 2g) echo "2g.10gb" ;; 3g) echo "3g.20gb" ;; 4g) echo "4g.20gb" ;;
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

# (workload, partition, tier, arg2, arg3)
#   ResNet: arg2 = batch_size, arg3 = precision (fp16/fp32)
#   Forecaster: arg2 = batch_size, arg3 = d_model
CELLS=(
  # ResNet-50 fp16 — batch sweep per partition (small/med/large)
  "resnet     1g bs16  16   fp16"
  "resnet     1g bs32  32   fp16"
  "resnet     1g bs64  64   fp16"
  "resnet     2g bs32  32   fp16"
  "resnet     2g bs64  64   fp16"
  "resnet     2g bs128 128  fp16"
  "resnet     3g bs64  64   fp16"
  "resnet     3g bs128 128  fp16"
  "resnet     3g bs256 256  fp16"
  "resnet     4g bs64  64   fp16"
  "resnet     4g bs128 128  fp16"
  "resnet     4g bs256 256  fp16"
  # Forecaster — batch sweep per partition (small/med/large)
  "forecaster 1g bs16  16   384"
  "forecaster 1g bs32  32   384"
  "forecaster 1g bs64  64   384"
  "forecaster 2g bs32  32   512"
  "forecaster 2g bs64  64   512"
  "forecaster 2g bs128 128  512"
  "forecaster 3g bs64  64   512"
  "forecaster 3g bs128 128  512"
  "forecaster 3g bs256 256  512"
  "forecaster 4g bs64  64   512"
  "forecaster 4g bs128 128  512"
  "forecaster 4g bs256 256  512"
)

start_persistent_l1() {
  docker run -d --rm --gpus "\"device=$1\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" \
    --name "l1_bg_sup_$$_$(date +%s)_$RANDOM" \
    "$IMAGE" bash /scripts/real_l1_loop.sh bg_l1 20 50
}

script_args_for() {
  local wl=$1 a=$2 b=$3
  case "$wl" in
    resnet)     echo "/workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 $DURATION $a $b" ;;
    forecaster) echo "/workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 $DURATION $a $b" ;;
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

run_cell() {
  local wl=$1 part=$2 tier=$3 a=$4 b=$5
  local tag="${wl}_${part}_${tier}"
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
    grep -E "done|it/s|batch/s|cells_forecast/s" "$alone_dir/run_${i}.log" | tail -1
  done
  for i in $(seq 1 $N); do
    log "  with_l1 $i/$N"
    local cid; cid=$(start_persistent_l1 "$l1_uuid")
    sleep 8
    run_ai_once "$ai_uuid" "$with_dir/run_${i}.log" "$args"
    grep -E "done|it/s|batch/s|cells_forecast/s" "$with_dir/run_${i}.log" | tail -1
    docker kill "$cid" >/dev/null 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    sleep 3
  done
}

START_TS=$(date +%s)
for entry in "${CELLS[@]}"; do
  read -r wl part tier a b <<< "$entry"
  run_cell "$wl" "$part" "$tier" "$a" "$b" || log "  cell FAILED: $wl on $part $tier"
done
END_TS=$(date +%s)
log "ai_supplement DONE elapsed=$((END_TS - START_TS))s = $(( (END_TS - START_TS)/60 ))m"
