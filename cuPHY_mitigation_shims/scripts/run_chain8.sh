#!/usr/bin/env bash
# Chain 8 — cudaFreeAsync + cudaMemPool mitigation shim evaluation
#
# Per cell size ∈ {4, 10, 40, 60} on 3g+2g layout:
#   A_c{N}_alone            — L1 alone, no shim
#   A_c{N}_nrx_baseline     — L1 + NRx coloc, no shim (Chain 7 X3 재현)
#   A_c{N}_nrx_freeasync    — L1 + NRx coloc + Option A shim (cudaFreeAsync)
#   A_c{N}_nrx_pool         — L1 + NRx coloc + Option B shim (memPool)
#
# Total: 4 conditions × 4 cells = 16 conditions, dual capture.
# Expected runtime: ~30-45 min.

set -uo pipefail
cd "$HOME"

DATE_DIR=20260701
RESULTS=$HOME/cloudlab_aerial/results/$DATE_DIR
OUT=$RESULTS/chain8
LOG_DIR=$RESULTS/logs
mkdir -p "$OUT" "$LOG_DIR"
chmod 777 "$OUT"

LOG=$LOG_DIR/chain8_$(date +%H%M).log
exec > >(tee -a "$LOG") 2>&1

GPU=0
IMAGE=airan:25-3-final
AERIAL_SDK=/mydata/aerial-cuda-accelerated-ran
SCRIPT_DIR=$HOME/cloudlab_aerial
REPO_DIR=$HOME/AIRAN_Changjong
HF_CACHE=/mydata/hf_cache
SHIMS=$HOME/shims

NRX=/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py

WARMUP_S=15
CAPTURE_S=30
AI_TOTAL_S=60
ITERS=20

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "===== CHAIN 8 START — cudaFreeAsync + cudaMemPool mitigation ====="

mig_reset() {
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sleep 1
}
mig_create() {
  sudo nvidia-smi mig -i $GPU -cgi "$1" -C >/dev/null
  sleep 3
}
get_uuid() {
  nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$1" | head -n "${2:-1}" | tail -1 | grep -oE 'MIG-[0-9a-f-]+'
}

# Run a condition
#   $1 label   $2 l1_uuid   $3 ai_specs   $4 cells   $5 shim_path (container-side, empty=no shim)
run_cond() {
  local label=$1; local l1_uuid=$2; local ai_specs=$3; local cells=$4; local shim=$5
  log "===== $label  cells=$cells shim=${shim:-NONE} ====="

  local ai_cids=(); local ai_names=()
  if [[ -n "$ai_specs" ]]; then
    IFS=',' read -ra specs <<< "$ai_specs"
    for spec in "${specs[@]}"; do
      IFS=':' read -ra parts <<< "$spec"
      local script=${parts[0]}; local uuid=${parts[1]}; local name=${parts[2]}
      local cid
      cid=$(docker run -d --rm --gpus "\"device=$uuid\"" \
        -v $AERIAL_SDK:/opt/nvidia/cuBB \
        -v $REPO_DIR:/workspace/AIRAN_Changjong \
        -v $OUT:/out \
        -v $HF_CACHE:/cache \
        -e HF_HOME=/cache \
        -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        -e LD_LIBRARY_PATH=/opt/nvidia/cuBB/pyaerial/src/aerial:/opt/nvidia/cuBB/pyaerial/src/aerial/pycuphy \
        $IMAGE bash -c "
          nsys profile --trace=cuda,nvtx,osrt \
            --delay=$WARMUP_S --duration=$CAPTURE_S \
            --output=/out/${label}_AI_${name} --force-overwrite=true --stats=false \
            python3 $script 0 $AI_TOTAL_S
        ")
      ai_cids+=("$cid"); ai_names+=("$name")
      log "  AI $name → ${cid:0:12}"
    done
    sleep $WARMUP_S
  fi

  # Build L1 command — with optional LD_PRELOAD for shim
  local SHIM_ENV=""
  local SHIM_LOG_ENV=""
  if [[ -n "$shim" ]]; then
    SHIM_ENV="-e LD_PRELOAD=$shim -v $SHIMS:/shims"
    # Enable shim's own logging based on which shim
    if [[ "$shim" == *"cudaFreeAsync"* ]]; then
      SHIM_LOG_ENV="-e CUFREE_ASYNC_LOG=1"
    elif [[ "$shim" == *"cudaMemPool"* ]]; then
      SHIM_LOG_ENV="-e CUPOOL_LOG=1"
    fi
  fi

  docker run --rm --gpus "\"device=$l1_uuid\"" \
    -v $AERIAL_SDK:/opt/nvidia/cuBB \
    -v $SCRIPT_DIR:/scripts \
    -v $OUT:/out \
    -v $REPO_DIR:/workspace/AIRAN_Changjong \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e LD_LIBRARY_PATH=/opt/nvidia/cuBB/pyaerial/src/aerial:/opt/nvidia/cuBB/pyaerial/src/aerial/pycuphy \
    $SHIM_ENV $SHIM_LOG_ENV \
    $IMAGE bash -c "
      nsys profile --trace=cuda,nvtx,osrt \
        --delay=0 --duration=$CAPTURE_S \
        --output=/out/${label}_L1 --force-overwrite=true --stats=false \
        python3 /scripts/real_l1.py ${label}_L1 $cells $ITERS 2>&1
    " > $LOG_DIR/${label}_L1.log 2>&1

  for i in "${!ai_cids[@]}"; do
    local cid=${ai_cids[$i]}; local name=${ai_names[$i]}
    for w in {1..90}; do
      docker ps -q --no-trunc | grep -q "$cid" || break
      sleep 2
    done
    docker logs "$cid" > $LOG_DIR/${label}_AI_${name}.log 2>&1 || true
    docker kill "$cid" >/dev/null 2>&1 || true
  done

  for view in L1 ${ai_names[@]+"${ai_names[@]/#/AI_}"}; do
    [[ -n "$view" ]] || continue
    rep=$OUT/${label}_${view}.nsys-rep
    db=$OUT/${label}_${view}.sqlite
    if [[ -f "$rep" && ! -f "$db" ]]; then
      docker run --rm -v $OUT:/out $IMAGE bash -c \
        "nsys export --type=sqlite --output=/out/${label}_${view}.sqlite \
          /out/${label}_${view}.nsys-rep --force-overwrite=true" >/dev/null 2>&1
    fi
  done

  grep -E '\[realL1\] ' $LOG_DIR/${label}_L1.log 2>/dev/null | tail -1 | sed 's/^/  /'
  # Also print shim report if any
  grep -E '\[cuda(FreeAsync|MemPool)_shim\]' $LOG_DIR/${label}_L1.log 2>/dev/null | tail -2 | sed 's/^/  /'
  sleep 2
}

# Setup 3g+2g
log "### Setup 3g+2g ###"
mig_reset; mig_create "9,14"
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
log "  UUID_3G=$UUID_3G  UUID_2G=$UUID_2G"

for CELLS in 4 10 40 60; do
  log "##### cells=$CELLS #####"
  # A: alone baseline (no shim, no AI)
  run_cond "A_c${CELLS}_alone"          "$UUID_3G" ""                                "$CELLS" ""
  # B: NRx coloc baseline (Chain 7 X3 재현, no shim)
  run_cond "A_c${CELLS}_nrx_baseline"   "$UUID_3G" "$NRX:$UUID_3G:nrx"               "$CELLS" ""
  # C: Option A — cudaFreeAsync shim
  run_cond "A_c${CELLS}_nrx_freeasync"  "$UUID_3G" "$NRX:$UUID_3G:nrx"               "$CELLS" "/shims/cudaFreeAsync.so"
  # D: Option B — cudaMemPool shim
  run_cond "A_c${CELLS}_nrx_pool"       "$UUID_3G" "$NRX:$UUID_3G:nrx"               "$CELLS" "/shims/cudaMemPool.so"
done

log "===== CHAIN 8 SUMMARY ====="
SUM=$RESULTS/SUMMARY_chain8.md
{
  echo "# Chain 8 — cudaFreeAsync + cudaMemPool mitigation"
  echo ""
  echo "Generated: $(ts)"
  echo ""
  echo "## L1 cudaFree by shim × cells"
  echo "| cells | alone | NRx coloc | + freeasync (A) | + memPool (B) |"
  echo "|---|---|---|---|---|"
  for CELLS in 4 10 40 60; do
    row="| $CELLS |"
    for lbl in "A_c${CELLS}_alone" "A_c${CELLS}_nrx_baseline" "A_c${CELLS}_nrx_freeasync" "A_c${CELLS}_nrx_pool"; do
      line=$(grep -E "^\\[realL1\\] ${lbl}_L1:" $LOG_DIR/${lbl}_L1.log 2>/dev/null | head -1)
      if [[ -n "$line" ]]; then
        mean=$(echo "$line" | grep -oE 'mean=[0-9.]+' | cut -d= -f2)
        row="$row $mean |"
      else
        row="$row ? |"
      fi
    done
    echo "$row"
  done
  echo ""
  echo "## DONE marker"
  echo "  All 16 conditions attempted."
} > $SUM

# DONE marker for auto-rsync
{
  echo "Chain 8 completed"
  echo "Timestamp: $(ts)"
  echo "  chain8: $(ls $OUT/*_L1.sqlite 2>/dev/null | wc -l) L1 sqlite"
} > $OUT/DONE_CHAIN8

log "===== CHAIN 8 DONE ====="
