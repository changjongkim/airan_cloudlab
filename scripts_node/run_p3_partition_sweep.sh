#!/usr/bin/env bash
# P3: AI partition size sweep — fix L1 on 3g, vary AI partition (1g/2g/3g/4g).
# For each AI partition: measure L1 latency alone (no AI) + with various AI workloads.
# Workloads: sat_compute (control - never disturbs), qwen_small, chanpred, xapp.
#
# Output: results/$DATE_DIR/p3_partition_sweep/<ai_part>/<workload>/{alone,with_ai}/run_<i>_l1.log

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-5}"
DURATION="${DURATION:-30}"
CELLS_L1="${CELLS_L1:-20}"
ITERS_L1="${ITERS_L1:-30}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"
HOST_UID=$(id -u); HOST_GID=$(id -g)

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/p3_partition_sweep"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

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

start_ai_bg() {
  local ai_uuid=$1 wl=$2 alloc=$3 gemm=$4
  local script_args
  case "$wl" in
    qwen_small)  script_args="/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py 0 $DURATION" ;;
    chanpred)    script_args="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py 0 $DURATION" ;;
    xapp)        script_args="/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py 0 $DURATION" ;;
    neuralrx)    script_args="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DURATION" ;;
    sat_compute) script_args="/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py 0 $DURATION $alloc $gemm" ;;
    sat_hbm)     script_args="/workspace/AIRAN_Changjong/experiments/run_hbm_saturated.py 0 $DURATION $alloc" ;;
    resnet)      script_args="/workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 $DURATION $alloc $gemm" ;;
    forecaster)  script_args="/workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 $DURATION $alloc $gemm" ;;
  esac
  docker run -d --rm --gpus "\"device=$ai_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    --name "p3_ai_$$_$(date +%s)_$RANDOM" \
    "$IMAGE" bash -c "python3 $script_args > /tmp/ai.log 2>&1"
}

run_l1_once() {
  local l1_uuid=$1 outfile=$2 tag=$3
  docker run --rm --user "$HOST_UID:$HOST_GID" --gpus "\"device=$l1_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -w /scripts "$IMAGE" python3 real_l1.py "$tag" "$CELLS_L1" "$ITERS_L1" > "$outfile" 2>&1
}

# (ai_partition, cgi, ai_profile, l1_uuid_index_offset)
LAYOUTS=(
  "1g 9,14,19 1g.5gb"
  "2g 9,14    2g.10gb"
  "3g 9,9     3g.20gb"
  "4g 5,9     4g.20gb"
)
# (workload, alloc, gemm)
WORKLOADS=(
  "sat_compute 8.5 8192"    # compute saturation control
  "sat_hbm     8.5 0"        # HBM bw saturation control
  "qwen_small  0   0"        # LLM chaotic memory
  "chanpred    0   0"        # LSTM
  "xapp        0   0"        # autoencoder
  "neuralrx    0   0"        # PHY-NN outlier ⭐
  "resnet      64  fp16"     # CNN fp16 Tensor Core
  "forecaster  64  384"      # Time-series transformer (Informer-lite)
)

START_TS=$(date +%s)
for layout_entry in "${LAYOUTS[@]}"; do
  read -r ai_part cgi ai_profile <<< "$layout_entry"
  log "===== LAYOUT: L1=3g + AI partition=$ai_part (cgi=$cgi) ====="
  reconfigure_mig "$cgi"
  # L1 UUID — always 3g.20gb. For 3g+3g layout, L1=first 3g.
  local_l1_uuid=$(uuid_for_profile "3g.20gb")
  # AI UUID — use 2nd of same profile if AI=3g (since L1 took first 3g)
  if [[ "$ai_part" == "3g" ]]; then
    ai_uuid=$(uuid_for_profile_nth "3g.20gb" 2)
  else
    ai_uuid=$(uuid_for_profile "$ai_profile")
  fi
  if [[ -z "$local_l1_uuid" || -z "$ai_uuid" ]]; then
    log "  UUID missing (l1=$local_l1_uuid ai=$ai_uuid), skip"
    continue
  fi
  log "  L1 UUID=$local_l1_uuid  AI UUID=$ai_uuid"

  # ALONE pass: L1 alone in this layout (no AI co-tenant)
  alone_dir="$OUT_ROOT/AI=${ai_part}/alone"
  mkdir -p "$alone_dir" && chmod 777 "$alone_dir"
  for i in $(seq 1 $N); do
    log "  [AI=$ai_part] alone $i/$N"
    run_l1_once "$local_l1_uuid" "$alone_dir/run_${i}.log" "P3_${ai_part}_alone_run${i}"
    grep "mean=" "$alone_dir/run_${i}.log" | tail -1
  done

  # WITH-AI pass: each AI workload
  for wl_entry in "${WORKLOADS[@]}"; do
    read -r wl alloc gemm <<< "$wl_entry"
    log "  --- AI=$ai_part + $wl ---"
    with_dir="$OUT_ROOT/AI=${ai_part}/${wl}"
    mkdir -p "$with_dir" && chmod 777 "$with_dir"
    for i in $(seq 1 $N); do
      log "    $wl run $i/$N"
      cid=$(start_ai_bg "$ai_uuid" "$wl" "$alloc" "$gemm")
      sleep 12
      run_l1_once "$local_l1_uuid" "$with_dir/run_${i}_l1.log" "P3_${ai_part}_${wl}_run${i}"
      grep "mean=" "$with_dir/run_${i}_l1.log" | tail -1
      docker kill "$cid" >/dev/null 2>&1 || true
      docker rm -f "$cid" >/dev/null 2>&1 || true
      sleep 2
    done
  done
done
END_TS=$(date +%s)
log "p3 DONE elapsed=$((END_TS - START_TS))s = $(( (END_TS - START_TS)/60 ))m"
