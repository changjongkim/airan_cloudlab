#!/usr/bin/env bash
# Nsight Systems profiling — captures timeline + GPU metrics across 4 scenarios.
#
# Goal: prove the MIG architectural overhead mechanism by directly observing
#   - SM utilization (compute-bound or not?)
#   - DRAM throughput (HBM bw saturation?)
#   - Kernel launch gaps (launch path overhead?)
#   - L2 cache behavior (slice fragmentation?)
#   - GPU clock (DVFS difference?)
#
# Outputs:
#   results/$DATE_DIR/nsys/<scenario>.nsys-rep    (binary, view with nsys-ui)
#   results/$DATE_DIR/nsys/<scenario>.sqlite       (queryable)
#   results/$DATE_DIR/nsys/<scenario>_summary.txt  (text summary)
#
# Requires: nsys installed in container (Aerial 25-3 ships with it at /opt/nsight-systems-cli)

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
ITERS_PROFILE="${ITERS_PROFILE:-20}"  # shorter iters for profile (avoid huge trace files)
CELLS="${CELLS:-20}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
SCRIPT_DIR="$HOME/cloudlab_aerial"
REPO_DIR="$HOME/AIRAN_Changjong"

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/nsys"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ----------------------------------------------------------------------------
# Helper: get MIG UUID by size
# ----------------------------------------------------------------------------
get_mig_uuid() {
  local size=$1
  local gpu=$2
  nvidia-smi -L | awk -v g=$gpu 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$size" | head -1 | grep -oE 'MIG-[0-9a-f-]+'
}

# ----------------------------------------------------------------------------
# Common nsys profile invocation
# ----------------------------------------------------------------------------
run_nsys() {
  local scenario=$1
  local gpu_arg=$2   # "all" or "device=MIG-uuid"
  local extra_env=$3 # extra env vars for docker

  local out_base="$OUT_ROOT/$scenario"
  log "[$scenario] nsys profile starting..."

  docker run --rm --gpus "$gpu_arg" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$OUT_ROOT:/nsys_out" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    $extra_env \
    "$IMAGE" bash -c "
      cd /scripts && \
      nsys profile \
        --trace=cuda,nvtx,osrt \
        --gpu-metrics-device=all \
        --gpu-metrics-frequency=10000 \
        --output=/nsys_out/${scenario} \
        --stats=true \
        --force-overwrite=true \
        python3 /scripts/real_l1.py ${scenario} ${CELLS} ${ITERS_PROFILE}
    " 2>&1 | tee "$out_base.log"

  # Generate text summary from sqlite
  if [[ -f "$out_base.sqlite" ]]; then
    docker run --rm \
      -v "$OUT_ROOT:/nsys_out" \
      "$IMAGE" bash -c "
        nsys stats --format csv,column \
          --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,gpu_metrics \
          /nsys_out/${scenario}.sqlite > /nsys_out/${scenario}_summary.txt 2>&1
      "
  fi

  log "[$scenario] done. Files: ${out_base}.nsys-rep .sqlite _summary.txt"
}

# ============================================================
# Scenario A: Full GPU baseline (no MIG)
# ============================================================
log "===== Scenario A: Full GPU baseline ====="
log "Note: requires MIG disabled on GPU $GPU. Set up before calling this script."
mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i $GPU)
if [[ "$mig_state" == "Enabled" ]]; then
  log "WARNING: MIG enabled on GPU $GPU — skipping Scenario A. Run with MIG off for fullgpu profile."
else
  run_nsys "A_fullgpu" "all" ""
fi

# ============================================================
# Scenario B: 7g MIG single instance
# ============================================================
log "===== Scenario B: 7g.40gb MIG single instance ====="
log "Requires MIG enabled + 7g instance on GPU $GPU"
UUID_7G=$(get_mig_uuid "7g.40gb" $GPU)
if [[ -n "$UUID_7G" ]]; then
  run_nsys "B_7g_mig" "\"device=$UUID_7G\"" ""
else
  log "Skipping B: no 7g instance. Run with: sudo nvidia-smi mig -i $GPU -cgi 0 -C"
fi

# ============================================================
# Scenario C: 3g L1 alone (split-60-40, AI partition idle)
# ============================================================
log "===== Scenario C: 3g.20gb L1 alone ====="
log "Requires split-60-40 (3g + 2g) on GPU $GPU"
UUID_3G=$(get_mig_uuid "3g.20gb" $GPU)
if [[ -n "$UUID_3G" ]]; then
  run_nsys "C_3g_alone" "\"device=$UUID_3G\"" ""
else
  log "Skipping C: no 3g instance. Run with: sudo nvidia-smi mig -i $GPU -cgi 9,14 -C"
fi

# ============================================================
# Scenario D: 2g L1 alone (split-40-60)
# ============================================================
log "===== Scenario D: 2g.10gb L1 alone ====="
log "Requires split-40-60 (2g + 3g) on GPU $GPU"
UUID_2G=$(get_mig_uuid "2g.10gb" $GPU)
if [[ -n "$UUID_2G" ]]; then
  run_nsys "D_2g_alone" "\"device=$UUID_2G\"" ""
else
  log "Skipping D: no 2g instance."
fi

# ============================================================
# Scenario E: 3g + Qwen co-located (concurrent)
# ============================================================
log "===== Scenario E: 3g L1 + Qwen on 2g (concurrent) ====="
UUID_2G_AI=$(get_mig_uuid "2g.10gb" $GPU)
UUID_3G_L1=$(get_mig_uuid "3g.20gb" $GPU)
if [[ -n "$UUID_2G_AI" ]] && [[ -n "$UUID_3G_L1" ]]; then
  # Start Qwen in background
  QWEN_CID=$(docker run -d --rm --gpus "\"device=$UUID_2G_AI\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "/mydata/hf_cache:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_qwen7b_prefill.py 0 120)
  log "Qwen background started: $QWEN_CID"
  sleep 15  # let Qwen reach steady state

  run_nsys "E_3g_qwen_concurrent" "\"device=$UUID_3G_L1\"" ""

  docker kill "$QWEN_CID" >/dev/null 2>&1 || true
fi

# ============================================================
# Summary extraction
# ============================================================
log "===== Extracting key metrics from all scenarios ====="
for scenario in A_fullgpu B_7g_mig C_3g_alone D_2g_alone E_3g_qwen_concurrent; do
  rep="$OUT_ROOT/${scenario}.nsys-rep"
  summary="$OUT_ROOT/${scenario}_summary.txt"
  if [[ -f "$summary" ]]; then
    echo ""
    echo "=== $scenario ==="
    head -50 "$summary"
  fi
done

log "DONE — nsys profiles in $OUT_ROOT"
log "View with: nsys-ui <file>.nsys-rep, or query .sqlite directly"
