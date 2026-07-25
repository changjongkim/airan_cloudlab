#!/usr/bin/env bash
# nsys matrix v3 — adds missing AI workloads (chanpred, ResNet, Forecaster, xapp + hybrid).
# Uses GPU 1 (idle, MIG 3g+3g already configured from earlier).
# Tier1-matched: CELLS=20 ITERS=30 N=3 per scenario.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-1}"
CELLS="${CELLS:-20}"
ITERS="${ITERS:-30}"
N_RUNS="${N_RUNS:-3}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"
HOST_UID=$(id -u); HOST_GID=$(id -g)

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/nsys_full"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

mig_create() {
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi "$1" -C >/dev/null 2>&1
  sleep 2
}
get_uuid() {
  local profile=$1 nth=${2:-1}
  nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$profile" | grep -oE "MIG-[a-f0-9-]+" | sed -n "${nth}p"
}

start_ai_bg() {
  local script=$1 ai_uuid=$2
  shift 2
  docker run -d --rm --gpus "\"device=$ai_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    --name "nsys_ai_v3_$(date +%s)_$RANDOM" \
    "$IMAGE" python3 "$script" 0 90 "$@"
}

kill_all_ai() {
  docker ps --filter "name=nsys_ai_v3_" -q | xargs -r docker kill 2>/dev/null
  sleep 2
}

profile_l1() {
  local scenario=$1 l1_uuid=$2
  for i in $(seq 1 $N_RUNS); do
    log "[$scenario] run $i/$N_RUNS..."
    docker run --rm --user "$HOST_UID:$HOST_GID" --gpus "\"device=$l1_uuid\"" \
      -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" -v "$OUT_ROOT:/out" \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
      -w /scripts "$IMAGE" \
      bash -c "nsys profile --trace=cuda --output=/out/${scenario}_run${i} --force-overwrite=true --stats=false \
        python3 real_l1.py ${scenario}_run${i} $CELLS $ITERS" 2>&1 | tail -1
  done
}

# ============================================================
# NEW SCENARIOS (extending S2-S26)
# ============================================================

# S27: 3g L1 + chanpred on 2g
log "===== S27: 3g L1 + chanpred ====="
mig_create "9,14"
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "$UUID_2G" >/dev/null
  sleep 10
  profile_l1 "S27_3g_chanpred" "$UUID_3G"
  kill_all_ai
fi

# S28: 3g L1 + ResNet (fp16) on 2g
log "===== S28: 3g L1 + ResNet fp16 ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  docker run -d --rm --gpus "\"device=$UUID_2G\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_resnet_$(date +%s)" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 90 64 fp16 >/dev/null
  sleep 10
  profile_l1 "S28_3g_resnet" "$UUID_3G"
  kill_all_ai
fi

# S29: 3g L1 + Forecaster on 2g
log "===== S29: 3g L1 + Forecaster ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  docker run -d --rm --gpus "\"device=$UUID_2G\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_fore_$(date +%s)" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 90 64 384 >/dev/null
  sleep 10
  profile_l1 "S29_3g_forecaster" "$UUID_3G"
  kill_all_ai
fi

# S30: 3g L1 + xapp on 2g
log "===== S30: 3g L1 + xapp ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py" "$UUID_2G" >/dev/null
  sleep 10
  profile_l1 "S30_3g_xapp" "$UUID_3G"
  kill_all_ai
fi

# S31: 3g L1 + ResNet + chanpred on 2g+2g (M5c equivalent)
log "===== S31: 3g L1 + ResNet + chanpred (3way-bal) ====="
mig_create "9,14,14"
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G_A=$(get_uuid "2g.10gb" 1)
UUID_2G_B=$(get_uuid "2g.10gb" 2)
if [[ -n "$UUID_3G" && -n "$UUID_2G_A" && -n "$UUID_2G_B" ]]; then
  docker run -d --rm --gpus "\"device=$UUID_2G_A\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_resnet_$(date +%s)_a" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 90 64 fp16 >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "$UUID_2G_B" >/dev/null
  sleep 12
  profile_l1 "S31_3g_resnet_chanpred" "$UUID_3G"
  kill_all_ai
fi

# S32: 3g L1 + ResNet + Forecaster on 2g+2g (M8a equivalent)
log "===== S32: 3g L1 + ResNet + Forecaster (M8a) ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G_A=$(get_uuid "2g.10gb" 1)
UUID_2G_B=$(get_uuid "2g.10gb" 2)
if [[ -n "$UUID_3G" && -n "$UUID_2G_A" && -n "$UUID_2G_B" ]]; then
  docker run -d --rm --gpus "\"device=$UUID_2G_A\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_resnet_$(date +%s)" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 90 64 fp16 >/dev/null
  docker run -d --rm --gpus "\"device=$UUID_2G_B\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_fore_$(date +%s)" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 90 64 384 >/dev/null
  sleep 12
  profile_l1 "S32_3g_resnet_forecaster" "$UUID_3G"
  kill_all_ai
fi

# S33: 4g L1 + chanpred on 2g (3way-asym)
log "===== S33: 4g L1 + chanpred ====="
mig_create "5,14,19"
UUID_4G=$(get_uuid "4g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_4G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "$UUID_2G" >/dev/null
  sleep 10
  profile_l1 "S33_4g_chanpred" "$UUID_4G"
  kill_all_ai
fi

# S34: 4g L1 + ResNet on 2g
log "===== S34: 4g L1 + ResNet ====="
UUID_4G=$(get_uuid "4g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_4G" && -n "$UUID_2G" ]]; then
  docker run -d --rm --gpus "\"device=$UUID_2G\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_resnet_$(date +%s)" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 90 64 fp16 >/dev/null
  sleep 10
  profile_l1 "S34_4g_resnet" "$UUID_4G"
  kill_all_ai
fi

# S35: 2g L1 + chanpred on 3g
log "===== S35: 2g L1 + chanpred ====="
mig_create "14,9"
UUID_2G=$(get_uuid "2g.10gb")
UUID_3G=$(get_uuid "3g.20gb")
if [[ -n "$UUID_2G" && -n "$UUID_3G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "$UUID_3G" >/dev/null
  sleep 10
  profile_l1 "S35_2g_chanpred" "$UUID_2G"
  kill_all_ai
fi

# S36: 4g L1 + Forecaster on 2g
log "===== S36: 4g L1 + Forecaster ====="
mig_create "5,14,19"
UUID_4G=$(get_uuid "4g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_4G" && -n "$UUID_2G" ]]; then
  docker run -d --rm --gpus "\"device=$UUID_2G\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" --name "nsys_ai_v3_fore_$(date +%s)" \
    "$IMAGE" python3 /workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 90 64 384 >/dev/null
  sleep 10
  profile_l1 "S36_4g_forecaster" "$UUID_4G"
  kill_all_ai
fi

log "===== ALL 10 NEW SCENARIOS DONE ====="
ls "$OUT_ROOT" | grep -E "S2[789]_|S3[0-6]_" | head
