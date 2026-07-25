#!/usr/bin/env bash
# AI per-op latency matrix (Priority 1, new framing).
# Goal: show AI workloads ALSO have inflated tail latency under L1 co-tenant,
# refuting the misleading "AI is well-isolated" finding from throughput-only measurements.
#
# 4 workloads × 4 partitions × {alone, with_l1} × N=5 = 160 runs
# Each run = 30s of per-op CUDA event timing; outputs JSON with p50/p95/p99/p999.

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

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_per_op_latency"
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
    --name "l1_bg_lat_$$_$(date +%s)_$RANDOM" \
    "$IMAGE" bash /scripts/real_l1_loop.sh bg_l1 20 50
}

script_args_for() {
  local wl=$1 a=$2 b=$3
  case "$wl" in
    resnet)    echo "/workspace/AIRAN_Changjong/experiments/run_resnet_per_op_latency.py 0 $DURATION $a $b" ;;
    qwen)      echo "/workspace/AIRAN_Changjong/experiments/run_qwen_decode_latency.py 0 $DURATION $a" ;;
    chanpred)  echo "/workspace/AIRAN_Changjong/experiments/run_chanpred_latency.py 0 $DURATION $a" ;;
    neuralrx)  echo "/workspace/AIRAN_Changjong/experiments/run_neuralrx_latency.py 0 $DURATION" ;;
  esac
}

run_ai_once() {
  local ai_uuid=$1 outfile=$2 script_args=$3
  docker run --rm --gpus "\"device=$ai_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" python3 $script_args > "$outfile" 2>&1
}

# (workload, partition, arg2, arg3)
#   resnet: arg2=batch, arg3=precision
#   qwen:   arg2=model_id, arg3=unused
#   chanpred: arg2=batch
#   neuralrx: unused
CELLS=(
  # ResNet per-op latency
  "resnet   1g 32  fp16"
  "resnet   2g 64  fp16"
  "resnet   3g 128 fp16"
  "resnet   4g 128 fp16"
  # Qwen-1.5B decode latency (fits 1g+)
  "qwen     1g Qwen/Qwen2.5-1.5B 0"
  "qwen     2g Qwen/Qwen2.5-1.5B 0"
  "qwen     3g Qwen/Qwen2.5-1.5B 0"
  "qwen     4g Qwen/Qwen2.5-1.5B 0"
  # chanpred LSTM
  "chanpred 1g 16 0"
  "chanpred 2g 16 0"
  "chanpred 3g 16 0"
  "chanpred 4g 16 0"
  # NeuralRx
  "neuralrx 1g 0 0"
  "neuralrx 2g 0 0"
  "neuralrx 3g 0 0"
  "neuralrx 4g 0 0"
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
log "ai_per_op_latency DONE elapsed=$((END_TS - START_TS))s = $(( (END_TS - START_TS)/60 ))m"
log "Results: $OUT_ROOT"
