#!/usr/bin/env bash
# Multi-AI count scaling: 3g L1 + 0/1/2/3/4 AI workloads
# Uses 5-way MIG profile (3g + 4× 1g) — 3+1+1+1+1=7 GPCs.
#
# Already have:
#   3g + 0 AI = L1_alone_3g20gb (52.54)
#   3g + 1 AI Qwen = A0 (55.73)
#   3g + 2 AI = M1 3way-balanced (73.94) - though it's 2g AI not 1g
#
# This script adds:
#   3g + 3 AI on 1g.5gb each
#   3g + 4 AI on 1g.5gb each
#
# AI workloads on 1g.5gb (5GB memory): only small models fit
#   - chanpred (5MB) ✓
#   - xapp_anomaly (5MB) ✓
#   - gpt2 (~500MB) ✓
#   - qwen_small (~3.6GB) - tight, may fail on 1g.5gb

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-5}"
CELLS=20
ITERS=50
DURATION=20
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"

LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/multi_ai_count.log"
mkdir -p "$(dirname $LOG)"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a $LOG; }

# Reconfigure GPU $GPU: 3g + 4× 1g
log "=== Setup GPU $GPU: 3g + 4× 1g ==="
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -cgi 9,19,19,19,19 -C >/dev/null 2>&1
sleep 2

# Get MIG UUIDs from GPU $GPU
mapfile -t MIG_UUIDS < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/ { match($0, /MIG-[0-9a-f-]+/); print substr($0, RSTART, RLENGTH) }')
mapfile -t MIG_SIZES < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/ { print $2 }')

L1_UUID=""
AI_UUIDS=()
for i in "${!MIG_SIZES[@]}"; do
  if [[ "${MIG_SIZES[$i]}" == "3g.20gb" ]] && [[ -z "$L1_UUID" ]]; then
    L1_UUID="${MIG_UUIDS[$i]}"
  fi
  if [[ "${MIG_SIZES[$i]}" == "1g.5gb" ]]; then
    AI_UUIDS+=("${MIG_UUIDS[$i]}")
  fi
done
log "L1 (3g.20gb): $L1_UUID"
log "AI partitions (1g.5gb): ${#AI_UUIDS[@]} - ${AI_UUIDS[*]}"

if [[ -z "$L1_UUID" ]] || [[ ${#AI_UUIDS[@]} -lt 4 ]]; then
  log "ERROR: MIG setup failed"
  exit 1
fi

# AI workloads to use (small ones for 1g.5gb)
# format: name:script_path
AI_SCRIPTS=(
  "chanpred:/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py"
  "xapp:/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py"
  "gpt2:/workspace/AIRAN_Changjong/experiments/run_gpt2_stress.py"
  "chanpred2:/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py"
)

# Run for k = 3 and 4 AI workloads
for K in 3 4; do
  TAG="L1_3g_${K}AI_1g"
  OUT_DIR="$HOME/cloudlab_aerial/results/$DATE_DIR/n${N}_$TAG"
  mkdir -p "$OUT_DIR" && chmod 777 "$OUT_DIR"

  log "============================================"
  log "=== K=$K AI on 1g.5gb + L1 on 3g.20gb, N=$N ==="
  log "============================================"

  for i in $(seq 1 $N); do
    log "------ run $i/$N ------"

    # Start K AI containers (background)
    CIDS=()
    for k in $(seq 0 $((K-1))); do
      ai_def="${AI_SCRIPTS[$k]}"
      ai_name="${ai_def%%:*}"
      ai_script="${ai_def##*:}"
      ai_uuid="${AI_UUIDS[$k]}"
      cid=$(docker run -d --rm --gpus "\"device=$ai_uuid\"" \
        -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
        -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
        "$IMAGE" python3 $ai_script 0 $DURATION)
      CIDS+=("$cid")
      log "  AI[$k] $ai_name -> $cid"
    done

    sleep 5  # let AI warm up

    # Run L1 (single 50-iter run)
    RES=$OUT_DIR/run_${i}_tmp
    mkdir -p $RES && chmod 777 $RES
    docker run --rm --gpus "\"device=$L1_UUID\"" \
      -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
      -v "$SCRIPT_DIR:/scripts" \
      -v "$RES:/results_out" \
      -e RESULTS_DIR=/results_out \
      "$IMAGE" python3 /scripts/real_l1.py "${K}AI_$i" "$CELLS" "$ITERS" \
      > $OUT_DIR/run_${i}.log 2>&1 || log "  L1 failed run $i"

    latest=$(ls -t $RES/*.json 2>/dev/null | head -1)
    if [[ -f "$latest" ]]; then
      cp "$latest" $OUT_DIR/run_${i}.json
      rm -rf $RES
      mean=$(python3 -c "import json; print(f\"{json.load(open('$latest'))['mean_ms']:.3f}\")")
      p99=$(python3 -c "import json; print(f\"{json.load(open('$latest'))['p99_ms']:.3f}\")")
      printf "  run %d  mean=%s ms  p99=%s ms\n" $i $mean $p99 | tee -a $OUT_DIR/summary.txt
    else
      log "  run $i NO JSON"
    fi

    # Kill AI containers
    for cid in "${CIDS[@]}"; do
      docker kill "$cid" >/dev/null 2>&1 || true
    done
    sleep 2
  done

  log "K=$K done. Output: $OUT_DIR"
done

log "DONE — multi-AI count scaling"
