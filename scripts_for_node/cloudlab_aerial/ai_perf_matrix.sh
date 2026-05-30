#!/usr/bin/env bash
# AI workload performance matrix — measures throughput across partitions × with/without L1.
#
# Why this matters:
#   Symmetric to L1 latency measurement. Paper's "MIG isolation fails" claim needs
#   bidirectional evidence — both L1 AND AI degrade under co-residency.
#
# Comparison axes:
#   A. AI partition size (AI alone):       1g vs 2g vs 3g vs 4g
#   B. L1 co-tenant impact:                AI alone vs AI + L1 on neighbor
#   C. AI workload type:                   Qwen small vs NeuralRx vs ChanPred vs xApp
#
# Output:
#   results/$DATE_DIR/ai_perf/<workload>_<partition>_<setup>/run_<i>.log
#   results/$DATE_DIR/ai_perf/<workload>_<partition>_<setup>/throughput.txt
#
# Each AI workload prints "done: ... it/s" or "... inf/s" — parsed for throughput.

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

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_perf"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ============================================================
# AI workload definitions (script path + memory requirement)
# ============================================================
declare -A AI_SCRIPT
declare -A AI_MIN_MEM_GB
AI_SCRIPT[qwen_small]="/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py"
AI_MIN_MEM_GB[qwen_small]=4

AI_SCRIPT[neuralrx]="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py"
AI_MIN_MEM_GB[neuralrx]=1

AI_SCRIPT[chanpred]="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py"
AI_MIN_MEM_GB[chanpred]=1

AI_SCRIPT[xapp]="/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py"
AI_MIN_MEM_GB[xapp]=1

# ============================================================
# MIG helpers
# ============================================================
mig_create() {
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi "$1" -C >/dev/null 2>&1
  sleep 2
}

get_uuid() {
  local size=$1
  local idx=${2:-1}
  nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$size" | head -n "$idx" | tail -1 | grep -oE 'MIG-[0-9a-f-]+'
}

# ============================================================
# Measurement function
# ============================================================
measure_ai() {
  local ai_name=$1     # workload name (key in AI_SCRIPT)
  local partition=$2   # partition size string e.g. "2g.10gb"
  local setup=$3       # "alone" or "with_l1"
  local ai_uuid=$4     # MIG UUID for AI
  local l1_uuid=$5     # MIG UUID for L1 (empty if alone)

  local dir="$OUT_ROOT/${ai_name}_${partition//./_}_${setup}"
  mkdir -p "$dir" && chmod 777 "$dir"

  for i in $(seq 1 "$N"); do
    log "  [$ai_name | $partition | $setup] run $i/$N"

    # If with_l1, start persistent L1 background first
    local l1_cid=""
    if [[ "$setup" == "with_l1" && -n "$l1_uuid" ]]; then
      l1_cid=$(docker run -d --rm --gpus "\"device=$l1_uuid\"" \
        -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
        -v "$SCRIPT_DIR:/scripts" \
        "$IMAGE" bash /scripts/real_l1_loop.sh bg_l1 20 50)
      sleep 8  # let L1 reach steady state
    fi

    # Run AI workload, measure throughput
    docker run --rm --gpus "\"device=$ai_uuid\"" \
      -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
      -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
      -v "$SCRIPT_DIR:/scripts" \
      -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
      -e HF_HOME="/mnt/dockerdata/hf_cache" \
      "$IMAGE" python3 "${AI_SCRIPT[$ai_name]}" 0 "$DURATION" \
      > "$dir/run_${i}.log" 2>&1

    # Kill persistent L1
    if [[ -n "$l1_cid" ]]; then
      docker kill "$l1_cid" >/dev/null 2>&1 || true
      sleep 2
    fi

    # Extract throughput line
    grep -E "done:|it/s|inf/s|pred/s|inferences in" "$dir/run_${i}.log" | tail -1 \
      | tee -a "$dir/throughput.txt"
  done
}

# ============================================================
# AXIS A: AI alone, across partition sizes (baseline)
# ============================================================
log "===== AXIS A: AI throughput baseline across partitions ====="

# A1: AI on 1g.5gb (max 7 instances, but we just need 1)
log "--- Setup: 7-way 1g layout ---"
mig_create "19,19,19,19,19,19,19"
UUID_1G=$(get_uuid "1g.5gb")
if [[ -n "$UUID_1G" ]]; then
  for ai in neuralrx chanpred xapp; do
    measure_ai "$ai" "1g.5gb" "alone" "$UUID_1G" ""
  done
fi

# A2: AI on 2g.10gb
log "--- Setup: split-60-40 with 3g idle ---"
mig_create "9,14,14"
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_2G" ]]; then
  for ai in qwen_small neuralrx chanpred xapp; do
    measure_ai "$ai" "2g.10gb" "alone" "$UUID_2G" ""
  done
fi

# A3: AI on 3g.20gb
log "--- Setup: split-60-40 with 2g idle ---"
mig_create "9,14"
UUID_3G=$(get_uuid "3g.20gb")
if [[ -n "$UUID_3G" ]]; then
  for ai in qwen_small neuralrx chanpred xapp; do
    measure_ai "$ai" "3g.20gb" "alone" "$UUID_3G" ""
  done
fi

# A4: AI on 4g.20gb
log "--- Setup: 3way-asym with 2g+1g idle ---"
mig_create "5,14,19"
UUID_4G=$(get_uuid "4g.20gb")
if [[ -n "$UUID_4G" ]]; then
  for ai in qwen_small neuralrx chanpred xapp; do
    measure_ai "$ai" "4g.20gb" "alone" "$UUID_4G" ""
  done
fi

# ============================================================
# AXIS B: AI + L1 co-located (impact of L1 on AI throughput)
# ============================================================
log "===== AXIS B: AI throughput with L1 co-tenant ====="

# B1: L1 on 3g + AI on 2g (typical AI-RAN, split-60-40)
log "--- Layout: 3g L1 + 2g AI (split-60-40) ---"
mig_create "9,14"
UUID_3G_L1=$(get_uuid "3g.20gb")
UUID_2G_AI=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G_L1" && -n "$UUID_2G_AI" ]]; then
  for ai in qwen_small neuralrx chanpred xapp; do
    measure_ai "$ai" "2g.10gb" "with_l1_3g" "$UUID_2G_AI" "$UUID_3G_L1"
  done
fi

# B2: L1 on 3g + AI on 1g (5-way layout)
log "--- Layout: 3g L1 + 1g AI (5-way) ---"
mig_create "9,19,19,19,19"
UUID_3G_L1=$(get_uuid "3g.20gb")
UUID_1G_AI=$(get_uuid "1g.5gb")
if [[ -n "$UUID_3G_L1" && -n "$UUID_1G_AI" ]]; then
  for ai in neuralrx chanpred xapp; do
    measure_ai "$ai" "1g.5gb" "with_l1_3g" "$UUID_1G_AI" "$UUID_3G_L1"
  done
fi

# B3: L1 on 4g + AI on 2g (3way-asym)
log "--- Layout: 4g L1 + 2g AI (3way-asym) ---"
mig_create "5,14,19"
UUID_4G_L1=$(get_uuid "4g.20gb")
UUID_2G_AI=$(get_uuid "2g.10gb")
if [[ -n "$UUID_4G_L1" && -n "$UUID_2G_AI" ]]; then
  for ai in qwen_small neuralrx chanpred xapp; do
    measure_ai "$ai" "2g.10gb" "with_l1_4g" "$UUID_2G_AI" "$UUID_4G_L1"
  done
fi

# B4: L1 on 4g + AI on 1g (M4 layout)
log "--- Layout: 4g L1 + 1g AI (4-way) ---"
mig_create "5,19,19,19"
UUID_4G_L1=$(get_uuid "4g.20gb")
UUID_1G_AI=$(get_uuid "1g.5gb")
if [[ -n "$UUID_4G_L1" && -n "$UUID_1G_AI" ]]; then
  for ai in neuralrx chanpred xapp; do
    measure_ai "$ai" "1g.5gb" "with_l1_4g" "$UUID_1G_AI" "$UUID_4G_L1"
  done
fi

# B5: L1 on 2g + AI on 3g (D1a-like, L1 small)
log "--- Layout: 2g L1 + 3g AI (split-40-60) ---"
mig_create "14,9"
UUID_2G_L1=$(get_uuid "2g.10gb")
UUID_3G_AI=$(get_uuid "3g.20gb")
if [[ -n "$UUID_2G_L1" && -n "$UUID_3G_AI" ]]; then
  for ai in qwen_small neuralrx chanpred xapp; do
    measure_ai "$ai" "3g.20gb" "with_l1_2g" "$UUID_3G_AI" "$UUID_2G_L1"
  done
fi

# ============================================================
# Analysis / summary
# ============================================================
log "===== ALL AI PERF MEASUREMENTS DONE ====="
log "Output: $OUT_ROOT"
log ""
log "Summary script:"
log "  for d in $OUT_ROOT/*/; do"
log "    echo \"--- \$(basename \$d) ---\""
log "    cat \"\$d/throughput.txt\" | tail -3"
log "  done"
log ""
log "Comparison axes:"
log "  A (partition size, AI alone):  *_1g.5gb_alone / *_2g.10gb_alone / *_3g.20gb_alone / *_4g.20gb_alone"
log "  B (L1 co-tenant impact):       compare *_alone vs *_with_l1_*"
log "  C (workload type):             across qwen_small / neuralrx / chanpred / xapp"
log ""
log "Expected output table:"
log "  Throughput(workload, partition, setup) for all combinations"
log "  ΔThroughput = (alone) - (with_l1) = L1 disruption on AI"
