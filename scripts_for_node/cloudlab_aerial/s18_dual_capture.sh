#!/usr/bin/env bash
# §18 dual-process NSYS capture — profiles BOTH L1 and AI simultaneously.
#
# Purpose: §17 only captured cuPHY L1 process. §18 captures the AI process
#   itself, so we can apply the same kernel/memcpy/memset/sync/idle + host
#   runtime API decomposition from the AI side and answer:
#     - Does the inflated AI (NeuralRx) show the same idle-gap-dominated
#       and cudaFree-heavy decomposition as L1?
#     - Does the stable AI (chanpred) NOT show it (asymmetric proof)?
#
# Each scenario produces TWO sqlite files in the same window:
#     ${label}_AI.{nsys-rep,sqlite}    — AI-process view
#     ${label}_L1.{nsys-rep,sqlite}    — cuPHY L1 view  (empty for AI-alone cases)
#
# Both nsys runs use --delay=15 --duration=30 → identical 30s capture window
# starting after a 15s warmup. AI total runtime 60s, L1 60s (cuPHY measures
# whatever NSYS_ITERS produces, typically ~30s of measurement traffic).
#
# Total scenarios: 6 (NeuralRx × 3 placements + chanpred × 3 placements)
# Expected wall time: ~50 minutes including MIG reconfig.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
CELLS="${CELLS:-20}"
NSYS_ITERS="${NSYS_ITERS:-20}"
WARMUP_S="${WARMUP_S:-15}"        # AI background warmup before capture window
CAPTURE_S="${CAPTURE_S:-30}"      # nsys window length (both AI and L1)
AI_TOTAL_S=$(( WARMUP_S + CAPTURE_S + 15 ))   # safety margin past nsys window
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
SCRIPT_DIR="$HOME/cloudlab_aerial"
REPO_DIR="$HOME/AIRAN_Changjong"
HF_CACHE="/mydata/hf_cache"

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/s18_ai_nsys"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ----------------------------------------------------------------------------
# MIG helpers (reuse nsight_full_matrix.sh patterns)
# ----------------------------------------------------------------------------
mig_create() {
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi "$1" -C >/dev/null 2>&1
  sleep 2
}

get_uuid() {
  # arg1: size string (e.g. "3g.20gb"); arg2: optional 1-based index for multi
  nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$1" | head -n "${2:-1}" | tail -1 | grep -oE 'MIG-[0-9a-f-]+'
}

# ----------------------------------------------------------------------------
# Dual capture primitive
#   $1 label  $2 ai_script  $3 ai_uuid  $4 l1_uuid (empty for AI-alone)
# ----------------------------------------------------------------------------
dual_capture() {
  local label=$1
  local ai_script=$2
  local ai_uuid=$3
  local l1_uuid=${4:-}

  log "===== ${label} ====="
  log "  AI script : ${ai_script}"
  log "  AI UUID   : ${ai_uuid}"
  log "  L1 UUID   : ${l1_uuid:-<none, AI alone>}"

  # Launch AI in background, wrapped in nsys (delay 15s, capture 30s)
  local AI_CID
  AI_CID=$(docker run -d --rm --gpus "\"device=$ai_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$OUT_ROOT:/nsys_out" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" bash -c "
      nsys profile --trace=cuda,nvtx,osrt \
        --delay=${WARMUP_S} --duration=${CAPTURE_S} \
        --output=/nsys_out/${label}_AI --force-overwrite=true \
        --stats=false \
        python3 ${ai_script} 0 ${AI_TOTAL_S}
    ")
  log "  AI container: $AI_CID"

  if [[ -n "$l1_uuid" ]]; then
    # Wait for AI warmup, then launch L1 nsys in foreground.
    # L1 has its own 30s nsys window with --delay=0 because we already waited.
    log "  Waiting ${WARMUP_S}s for AI warmup before launching L1..."
    sleep "$WARMUP_S"
    log "  Launching L1 nsys (capture ${CAPTURE_S}s)..."
    docker run --rm --gpus "\"device=$l1_uuid\"" \
      -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
      -v "$SCRIPT_DIR:/scripts" \
      -v "$OUT_ROOT:/nsys_out" \
      -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
      "$IMAGE" bash -c "
        nsys profile --trace=cuda,nvtx,osrt \
          --delay=0 --duration=${CAPTURE_S} \
          --output=/nsys_out/${label}_L1 --force-overwrite=true \
          --stats=false \
          python3 /scripts/real_l1.py ${label}_L1 ${CELLS} ${NSYS_ITERS}
      " 2>&1 | tail -5 | tee "$OUT_ROOT/${label}_L1.log"
  fi

  # Wait for AI container to finish — bounded by AI_TOTAL_S; force kill if stuck
  local DEADLINE=$(( $(date +%s) + AI_TOTAL_S + 30 ))
  while docker ps -q --no-trunc | grep -q "$AI_CID"; do
    if (( $(date +%s) > DEADLINE )); then
      log "  AI container exceeded deadline — killing"
      docker kill "$AI_CID" >/dev/null 2>&1 || true
      break
    fi
    sleep 2
  done
  log "  AI container done."

  # Generate sqlite from rep (sqlite needs explicit conversion in some nsys versions)
  for view in AI L1; do
    local rep="$OUT_ROOT/${label}_${view}.nsys-rep"
    local db="$OUT_ROOT/${label}_${view}.sqlite"
    if [[ -f "$rep" && ! -f "$db" ]]; then
      docker run --rm -v "$OUT_ROOT:/nsys_out" "$IMAGE" bash -c "
        nsys export --type=sqlite --output=/nsys_out/${label}_${view}.sqlite \
          /nsys_out/${label}_${view}.nsys-rep --force-overwrite=true
      " >/dev/null 2>&1
    fi
    if [[ -f "$db" ]]; then
      local sz=$(du -h "$db" | cut -f1)
      log "  → ${label}_${view}.sqlite (${sz})"
    fi
  done
  log "  ${label} complete."
  echo ""
}

# ----------------------------------------------------------------------------
# Scenario matrix — 6 conditions
#
#   AI workload   |  AI partition | L1 partition  |  Hypothesis (from §2.2, §10, §G coloc)
#   ─────────────────────────────────────────────────────────────────────────
#   NeuralRx       |  3g (alone)   |   —           |  AI baseline
#   NeuralRx       |  2g           |  3g           |  L1 +376% (PHY-AI cross-part inflation)
#   NeuralRx       |  3g (same)    |  3g (same)    |  L1 +537% (8× coloc collapse)
#   chanpred       |  3g (alone)   |   —           |  AI baseline
#   chanpred       |  2g           |  3g           |  L1 +60–72% (stable cross-part)
#   chanpred       |  3g (same)    |  3g (same)    |  L1 ~8× (coloc collapse, workload-invariant)
#
# For each scenario, AI nsys is the primary new data. L1 nsys validates that
# the L1 still shows §17-consistent behavior in this run.
# ----------------------------------------------------------------------------

NRX="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py"
CHP="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py"

# --- Stage 1: split-60-40 (3g + 2g) for alone + cross-partition ---
log "Configuring MIG split-60-40 (3g + 2g) for stages 1/2/4/5..."
mig_create "9,14"
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
log "  UUID_3G=$UUID_3G"
log "  UUID_2G=$UUID_2G"

# X1: NeuralRx alone (3g)
dual_capture "X1_neuralrx_alone_3g"          "$NRX" "$UUID_3G" ""

# X2: NeuralRx cross-partition (NeuralRx 2g, L1 3g) — §2.2 +376% condition
dual_capture "X2_neuralrx_2g_L1_3g_crosspart" "$NRX" "$UUID_2G" "$UUID_3G"

# X4: chanpred alone (3g)
dual_capture "X4_chanpred_alone_3g"          "$CHP" "$UUID_3G" ""

# X5: chanpred cross-partition (chanpred 2g, L1 3g)
dual_capture "X5_chanpred_2g_L1_3g_crosspart" "$CHP" "$UUID_2G" "$UUID_3G"

# --- Stage 2: 3g only (for same-partition coloc; both processes share UUID_3G) ---
log "Configuring MIG single 3g for stages 3/6 (same-partition coloc)..."
mig_create "9"
UUID_3G=$(get_uuid "3g.20gb")
log "  UUID_3G=$UUID_3G  (both AI and L1 will share this)"

# X3: NeuralRx + L1 same-partition coloc (3g)
dual_capture "X3_neuralrx_L1_coloc_3g"        "$NRX" "$UUID_3G" "$UUID_3G"

# X6: chanpred + L1 same-partition coloc (3g)
dual_capture "X6_chanpred_L1_coloc_3g"        "$CHP" "$UUID_3G" "$UUID_3G"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
log "====================================="
log "All scenarios complete. Outputs in: $OUT_ROOT"
ls -lh "$OUT_ROOT"/*.sqlite 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

log ""
log "Quick sanity (number of AI-side kernels per scenario):"
for f in "$OUT_ROOT"/*_AI.sqlite; do
  [[ -f "$f" ]] || continue
  local n=$(sqlite3 "$f" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL" 2>/dev/null || echo "?")
  echo "  $(basename "$f"): $n kernels"
done

log ""
log "Next step: rsync $OUT_ROOT to laptop and run:"
log "  python3 results/visual_evidence/build_time_breakdown_ai.py"
