#!/usr/bin/env bash
# Full AI throughput matrix: existing workloads × all partitions + new saturating workloads × all partitions.
# Closes coverage gaps (1g, 4g) and adds partition-saturating co-tenants.
#
# Each entry runs: N=5 alone + N=5 with persistent L1 on adjacent 3g.
# L1 always pinned to 3g (saturation L1 baseline = ~41 ms).
#
# AI partitions tested: 1g, 2g, 3g, 4g
# Workloads (per partition size cap):
#   - existing: qwen_small, chanpred, xapp, neuralrx
#   - new:      sat_compute, sat_hbm
#   - existing 3g/4g only: qwen7b_prefill, qwen7b_decode, qwen7b_stress
#
# Output: results/$DATE_DIR/ai_full_matrix/<workload>_<part>/<setup>/run_<i>.log

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

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_full_matrix"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ----------------------------------------------------------------------------
# MIG layout per AI partition (L1 always on 3g)
# ----------------------------------------------------------------------------
cgi_for_ai() {
  case "$1" in
    1g) echo "9,14,19" ;;   # 3g + 2g + 1g  → L1 on 3g, AI on 1g
    2g) echo "9,14"    ;;   # 3g + 2g       → L1 on 3g, AI on 2g
    3g) echo "9,9"     ;;   # 3g + 3g       → L1 on first 3g, AI on second 3g
    4g) echo "9,5"     ;;   # 3g + 4g       → L1 on 3g, AI on 4g
  esac
}
ai_profile_for() {
  case "$1" in
    1g) echo "1g.5gb"  ;;
    2g) echo "2g.10gb" ;;
    3g) echo "3g.20gb" ;;
    4g) echo "4g.20gb" ;;
  esac
}

reconfigure_mig() {
  local cgi=$1
  sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -cgi "$cgi" -C >/dev/null 2>&1
  sleep 2
}
uuid_for_profile() {
  local target=$1
  nvidia-smi -L | grep -E "MIG[[:space:]]+${target}[[:space:]]" \
    | grep -oE "MIG-[a-f0-9-]{36}" | head -1
}
uuid_for_profile_nth() {
  local target=$1 n=$2
  nvidia-smi -L | grep -E "MIG[[:space:]]+${target}[[:space:]]" \
    | grep -oE "MIG-[a-f0-9-]{36}" | sed -n "${n}p"
}

start_persistent_l1() {
  local l1_uuid=$1
  docker run -d --rm --gpus "\"device=$l1_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    --name "l1_bg_$$_$(date +%s)_$RANDOM" \
    "$IMAGE" bash /scripts/real_l1_loop.sh bg_l1 20 50
}

# Workload -> docker run argument (Python script + args)
script_for_workload() {
  local wl=$1 alloc=$2 gemm_dim=$3
  case "$wl" in
    qwen_small)     echo "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py 0 $DURATION" ;;
    chanpred)       echo "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py 0 $DURATION" ;;
    xapp)           echo "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py 0 $DURATION" ;;
    neuralrx)       echo "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DURATION" ;;
    qwen7b_prefill) echo "/workspace/AIRAN_Changjong/experiments/run_qwen7b_prefill.py 0 $DURATION" ;;
    qwen7b_decode)  echo "/workspace/AIRAN_Changjong/experiments/run_qwen7b_decode.py 0 $DURATION" ;;
    qwen7b_stress)  echo "/workspace/AIRAN_Changjong/experiments/run_qwen7b_stress.py 0 $DURATION" ;;
    sat_compute)    echo "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py 0 $DURATION $alloc $gemm_dim" ;;
    sat_hbm)        echo "/workspace/AIRAN_Changjong/experiments/run_hbm_saturated.py 0 $DURATION $alloc" ;;
    *) echo ""; return 1 ;;
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

run_workload_matrix_cell() {
  local wl=$1 part=$2 alloc=$3 gemm_dim=$4
  local cgi profile ai_uuid l1_uuid
  cgi=$(cgi_for_ai "$part")
  profile=$(ai_profile_for "$part")
  log "===== $wl on AI=$part (alloc=$alloc gemm=$gemm_dim) ====="
  reconfigure_mig "$cgi"
  l1_uuid=$(uuid_for_profile "3g.20gb")
  if [[ "$part" == "3g" ]]; then
    ai_uuid=$(uuid_for_profile_nth "3g.20gb" 2)
  else
    ai_uuid=$(uuid_for_profile "$profile")
  fi
  if [[ -z "$l1_uuid" ]] || [[ -z "$ai_uuid" ]]; then
    log "  ERROR: UUID not found (l1=$l1_uuid ai=$ai_uuid)"
    return 1
  fi
  log "  L1 UUID=$l1_uuid AI UUID=$ai_uuid"

  local script_args
  script_args=$(script_for_workload "$wl" "$alloc" "$gemm_dim")
  local alone_dir="$OUT_ROOT/${wl}_${part}/alone"
  local with_dir="$OUT_ROOT/${wl}_${part}/with_l1"
  mkdir -p "$alone_dir" "$with_dir" && chmod 777 "$alone_dir" "$with_dir"

  for i in $(seq 1 $N); do
    log "  alone $i/$N"
    run_ai_once "$ai_uuid" "$alone_dir/run_${i}.log" "$script_args"
    grep -E "done:|tflops=|bw=|inferences|it/s|inf/s|pred/s" "$alone_dir/run_${i}.log" | tail -2
  done

  for i in $(seq 1 $N); do
    log "  with_l1 $i/$N"
    local cid; cid=$(start_persistent_l1 "$l1_uuid")
    sleep 8
    run_ai_once "$ai_uuid" "$with_dir/run_${i}.log" "$script_args"
    grep -E "done:|tflops=|bw=|inferences|it/s|inf/s|pred/s" "$with_dir/run_${i}.log" | tail -2
    docker kill "$cid" >/dev/null 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    sleep 3
  done
}

# ----------------------------------------------------------------------------
# Cells: (workload, AI partition, alloc_gb, gemm_dim)
# alloc_gb and gemm_dim only used by sat_compute/sat_hbm
# ----------------------------------------------------------------------------
CELLS=(
  # Fill coverage gaps for existing workloads on 1g and 4g
  "qwen_small     1g 0 0"
  "qwen_small     4g 0 0"
  "chanpred       1g 0 0"
  "chanpred       4g 0 0"
  "xapp           1g 0 0"
  "xapp           4g 0 0"
  "neuralrx       1g 0 0"
  "neuralrx       4g 0 0"
  # Qwen-7B variants on 4g (model 14GB fits; can't fit 1g/2g)
  "qwen7b_prefill 4g 0 0"
  "qwen7b_decode  4g 0 0"
  "qwen7b_stress  4g 0 0"
  # New saturating workloads on every partition
  "sat_compute    1g 4.0 4096"
  "sat_compute    2g 8.5 8192"
  "sat_compute    3g 17  8192"
  "sat_compute    4g 17  8192"
  "sat_hbm        1g 4.0 0"
  "sat_hbm        2g 8.5 0"
  "sat_hbm        3g 17  0"
  "sat_hbm        4g 17  0"
)

START_TS=$(date +%s)
for entry in "${CELLS[@]}"; do
  read -r wl part alloc gemm_dim <<< "$entry"
  if ! run_workload_matrix_cell "$wl" "$part" "$alloc" "$gemm_dim"; then
    log "  cell FAILED: $wl on $part (continuing)"
  fi
done
END_TS=$(date +%s)
log "ai_full_matrix DONE elapsed=$((END_TS - START_TS))s = $(( (END_TS - START_TS)/60 ))m"
log "Results: $OUT_ROOT"
