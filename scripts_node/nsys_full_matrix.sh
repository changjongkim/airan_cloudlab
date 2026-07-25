#!/usr/bin/env bash
# Comprehensive nsys timeline capture for all 16 ncu-equivalent scenarios.
# Runs on GPU 1 (idle) to not conflict with chain running Stage 4 on GPU 0.
# Each scenario: setup MIG, optionally launch AI bg, run real_l1 under nsys.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-1}"
CELLS="${CELLS:-20}"   # Match Tier1 main: 20 cells per iteration
ITERS="${ITERS:-30}"   # Match Tier1 main: 30 iterations per run
N_RUNS="${N_RUNS:-3}"   # 3 runs per scenario for statistical power
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
  local cgi=$1
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi "$cgi" -C >/dev/null 2>&1
  sleep 2
}
get_uuid() {
  local profile=$1
  local nth=${2:-1}
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
    --name "nsys_ai_$(date +%s)_$RANDOM" \
    "$IMAGE" python3 "$script" 0 90 "$@"
}

kill_all_ai() {
  docker ps --filter "name=nsys_ai_" -q | xargs -r docker kill 2>/dev/null
  sleep 2
}

profile_l1() {
  local scenario=$1 l1_uuid=$2
  for i in $(seq 1 $N_RUNS); do
    log "[$scenario] nsys profile run $i/$N_RUNS..."
    docker run --rm --user "$HOST_UID:$HOST_GID" --gpus "\"device=$l1_uuid\"" \
      -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" -v "$OUT_ROOT:/out" \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
      -w /scripts "$IMAGE" \
      bash -c "nsys profile --trace=cuda --output=/out/${scenario}_run${i} --force-overwrite=true --stats=false \
        python3 real_l1.py ${scenario}_run${i} $CELLS $ITERS" 2>&1 | tail -2
  done
}

# ============================================================
# Scenarios — match ncu set
# ============================================================

# S2: 7g MIG single (Full GPU equivalent with MIG)
log "===== S2: 7g MIG single ====="
mig_create "0"
UUID=$(get_uuid "7g.40gb")
[[ -n "$UUID" ]] && profile_l1 "S2_7g_mig" "$UUID"

# S5: 3g L1 alone (split-60-40)
log "===== S5: 3g L1 alone ====="
mig_create "9,14"
UUID=$(get_uuid "3g.20gb")
[[ -n "$UUID" ]] && profile_l1 "S5_3g_alone" "$UUID"

# S6: 3g L1 + Qwen on 2g
log "===== S6: 3g L1 + Qwen on 2g ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py" "$UUID_2G")
  sleep 10
  profile_l1 "S6_3g_qwen" "$UUID_3G"
  kill_all_ai
fi

# S7: 3g L1 + NeuralRx on 2g
log "===== S7: 3g L1 + NeuralRx on 2g ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "$UUID_2G")
  sleep 10
  profile_l1 "S7_3g_neuralrx" "$UUID_3G"
  kill_all_ai
fi

# S9: 3g L1 + 3 small AI on 1g×3 (need 3g+1g+1g+1g layout)
log "===== S9: 3g L1 + 3 AI on 1g ====="
mig_create "9,19,19,19"
UUID_3G=$(get_uuid "3g.20gb")
mapfile -t U1G < <(get_uuid "1g.5gb"; get_uuid "1g.5gb" 2; get_uuid "1g.5gb" 3)
if [[ -n "$UUID_3G" && ${#U1G[@]} -ge 3 ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "${U1G[0]}" >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py" "${U1G[1]}" >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py" "${U1G[2]}" >/dev/null
  sleep 12
  profile_l1 "S9_3g_3AI_1g" "$UUID_3G"
  kill_all_ai
fi

# S10: 2g L1 alone (split-40-60)
log "===== S10: 2g L1 alone ====="
mig_create "14,9"
UUID=$(get_uuid "2g.10gb")
[[ -n "$UUID" ]] && profile_l1 "S10_2g_alone" "$UUID"

# S12: 2g L1 + 2 AI on 3g+2g (cgi 14,9,14)
log "===== S12: 2g L1 + 2 AI ====="
mig_create "14,9,14"
UUID_2G_L1=$(get_uuid "2g.10gb" 1)
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G_AI=$(get_uuid "2g.10gb" 2)
if [[ -n "$UUID_2G_L1" && -n "$UUID_3G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "$UUID_3G" >/dev/null
  [[ -n "$UUID_2G_AI" ]] && start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py" "$UUID_2G_AI" >/dev/null
  sleep 10
  profile_l1 "S12_2g_2AI" "$UUID_2G_L1"
  kill_all_ai
fi

# S13: 3g L1 + sat_compute on 2g
log "===== S13: 3g L1 + sat_compute ====="
mig_create "9,14"
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_2G" "8.5" "8192" >/dev/null
  sleep 10
  profile_l1 "S13_3g_sat_compute" "$UUID_3G"
  kill_all_ai
fi

# S14: 3g L1 + sat_hbm on 2g
log "===== S14: 3g L1 + sat_hbm ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_hbm_saturated.py" "$UUID_2G" "8.5" >/dev/null
  sleep 10
  profile_l1 "S14_3g_sat_hbm" "$UUID_3G"
  kill_all_ai
fi

# S15: 4g L1 + sat_compute on 2g (3way-asym)
log "===== S15: 4g L1 + sat_compute ====="
mig_create "5,14,19"
UUID_4G=$(get_uuid "4g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_4G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_2G" "8.5" "8192" >/dev/null
  sleep 10
  profile_l1 "S15_4g_sat_compute" "$UUID_4G"
  kill_all_ai
fi

# S17: 2g L1 + sat_compute on 3g
log "===== S17: 2g L1 + sat_compute on 3g ====="
mig_create "14,9"
UUID_2G=$(get_uuid "2g.10gb")
UUID_3G=$(get_uuid "3g.20gb")
if [[ -n "$UUID_2G" && -n "$UUID_3G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_3G" "17" "8192" >/dev/null
  sleep 10
  profile_l1 "S17_2g_sat_compute" "$UUID_2G"
  kill_all_ai
fi

# S18: 4g L1 + NeuralRx on 2g
log "===== S18: 4g L1 + NeuralRx ====="
mig_create "5,14,19"
UUID_4G=$(get_uuid "4g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_4G" && -n "$UUID_2G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "$UUID_2G" >/dev/null
  sleep 10
  profile_l1 "S18_4g_neuralrx" "$UUID_4G"
  kill_all_ai
fi

# S21: 4g L1 + 2 sat (2g+1g)
log "===== S21: 4g L1 + 2 sat ====="
UUID_4G=$(get_uuid "4g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
UUID_1G=$(get_uuid "1g.5gb")
if [[ -n "$UUID_4G" && -n "$UUID_2G" && -n "$UUID_1G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_2G" "8.5" "8192" >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_1G" "4.0" "4096" >/dev/null
  sleep 12
  profile_l1 "S21_4g_2sat" "$UUID_4G"
  kill_all_ai
fi

# S22: 2g L1 + NeuralRx on 3g
log "===== S22: 2g L1 + NeuralRx ====="
mig_create "14,9"
UUID_2G=$(get_uuid "2g.10gb")
UUID_3G=$(get_uuid "3g.20gb")
if [[ -n "$UUID_2G" && -n "$UUID_3G" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "$UUID_3G" >/dev/null
  sleep 10
  profile_l1 "S22_2g_neuralrx" "$UUID_2G"
  kill_all_ai
fi

# S24: 3g L1 + 2 sat on 2g+2g (M5a)
log "===== S24: 3g L1 + 2 sat ====="
mig_create "9,14,14"
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G_A=$(get_uuid "2g.10gb" 1)
UUID_2G_B=$(get_uuid "2g.10gb" 2)
if [[ -n "$UUID_3G" && -n "$UUID_2G_A" && -n "$UUID_2G_B" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_2G_A" "8.5" "8192" >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$UUID_2G_B" "8.5" "8192" >/dev/null
  sleep 12
  profile_l1 "S24_3g_2sat" "$UUID_3G"
  kill_all_ai
fi

# S26: 4g L1 + 3 sat on 1g×3 (M7a worst)
log "===== S26: 4g L1 + 3 sat ====="
mig_create "5,19,19,19"
UUID_4G=$(get_uuid "4g.20gb")
U1=$(get_uuid "1g.5gb" 1)
U2=$(get_uuid "1g.5gb" 2)
U3=$(get_uuid "1g.5gb" 3)
if [[ -n "$UUID_4G" && -n "$U1" && -n "$U2" && -n "$U3" ]]; then
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$U1" "4.0" "4096" >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$U2" "4.0" "4096" >/dev/null
  start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" "$U3" "4.0" "4096" >/dev/null
  sleep 12
  profile_l1 "S26_4g_3sat" "$UUID_4G"
  kill_all_ai
fi

log "ALL DONE — outputs in $OUT_ROOT"
ls -la "$OUT_ROOT" | head -20
