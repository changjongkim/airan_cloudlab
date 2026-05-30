#!/usr/bin/env bash
# Comprehensive Nsight profiling — 12-scenario matrix.
#
# 4 axes of comparison:
#   A. Partition size effect (L1 alone): S1 → S2 → S3 → S5 → S10
#   B. AI workload type effect (3g L1): S5 → S6 → S7 → S8 → S9
#   C. L1 partition effect (same AI): S6 vs S11, S8 vs S12
#   D. Layout effect (multi-AI 2g vs 1g): S8 vs S9
#
# For each scenario:
#   nsys: timeline + GPU metrics (10 min)
#   ncu: per-kernel metrics (30 min, only priority 8)
#
# Total time: ~6 hours (2h nsys + 4h ncu)
#
# Usage:
#   bash nsight_full_matrix.sh nsys     # only Nsight Systems (12 scenarios)
#   bash nsight_full_matrix.sh ncu      # only Nsight Compute (8 priority)
#   bash nsight_full_matrix.sh all      # both

set -uo pipefail
cd "$HOME/cloudlab_aerial"

MODE="${1:-all}"     # nsys / ncu / all
GPU="${GPU:-0}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
CELLS="${CELLS:-20}"
NSYS_ITERS="${NSYS_ITERS:-20}"
NCU_ITERS="${NCU_ITERS:-5}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
SCRIPT_DIR="$HOME/cloudlab_aerial"
REPO_DIR="$HOME/AIRAN_Changjong"
HF_CACHE="/mydata/hf_cache"

NSYS_OUT="$HOME/cloudlab_aerial/results/$DATE_DIR/nsys"
NCU_OUT="$HOME/cloudlab_aerial/results/$DATE_DIR/ncu"
mkdir -p "$NSYS_OUT" "$NCU_OUT" && chmod 777 "$NSYS_OUT" "$NCU_OUT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# ============================================================
# MIG configuration helpers
# ============================================================
mig_off() {
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi -i $GPU -mig 0 2>&1 | tail -1
}

mig_on() {
  sudo nvidia-smi -i $GPU -mig 1 2>&1 | tail -1
}

mig_create() {
  # arg: profile spec (e.g., "0" for 7g, "9,14" for split-60-40)
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi "$1" -C >/dev/null 2>&1
  sleep 2
}

get_uuid() {
  # arg: size (e.g., "3g.20gb")
  nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$1" | head -n "${2:-1}" | tail -1 | grep -oE 'MIG-[0-9a-f-]+'
}

# ============================================================
# Profiler invocations
# ============================================================
run_nsys() {
  local scenario=$1
  local gpu_arg=$2
  log "[$scenario] nsys profile..."
  docker run --rm --gpus "$gpu_arg" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$NSYS_OUT:/nsys_out" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    "$IMAGE" bash -c "
      nsys profile --trace=cuda,nvtx,osrt \
        --gpu-metrics-device=all --gpu-metrics-frequency=10000 \
        --output=/nsys_out/${scenario} --stats=true --force-overwrite=true \
        python3 /scripts/real_l1.py ${scenario} ${CELLS} ${NSYS_ITERS}
    " 2>&1 | tail -5 | tee "$NSYS_OUT/${scenario}.log"
}

run_ncu() {
  local scenario=$1
  local gpu_arg=$2
  log "[$scenario] ncu profile..."
  docker run --rm --gpus "$gpu_arg" --cap-add=SYS_ADMIN \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$NCU_OUT:/ncu_out" \
    "$IMAGE" bash -c "
      ncu --target-processes all --replay-mode kernel \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_hit_rate.pct,\
lts__t_sectors_hit_rate.pct,\
smsp__average_warps_active_per_cycle_pct,\
gpu__time_duration.sum \
        --csv --log-file /ncu_out/${scenario}_metrics.csv \
        -o /ncu_out/${scenario} --force-overwrite \
        python3 /scripts/real_l1.py ${scenario} ${CELLS} ${NCU_ITERS}
    " 2>&1 | tail -5 | tee "$NCU_OUT/${scenario}.log"
}

# Background AI starter
start_ai_bg() {
  local script=$1
  local uuid=$2
  local cid=$(docker run -d --rm --gpus "\"device=$uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" python3 "$script" 0 600)
  echo "$cid"
}

kill_all_bg() {
  docker ps -q --filter "name=ai_bg_" | xargs -r docker kill 2>/dev/null
  docker ps -q --filter "ancestor=$IMAGE" | xargs -r docker kill 2>/dev/null
  sleep 2
}

# ============================================================
# Scenario execution
# ============================================================
profile_scenario() {
  local scenario=$1
  local gpu_arg=$2
  local do_ncu=${3:-0}

  [[ "$MODE" == "nsys" || "$MODE" == "all" ]] && run_nsys "$scenario" "$gpu_arg"
  if [[ "$do_ncu" -eq 1 && ( "$MODE" == "ncu" || "$MODE" == "all" ) ]]; then
    run_ncu "$scenario" "$gpu_arg"
  fi
}

# ============================================================
# S1: Full GPU (no MIG)  — NSYS+NCU priority
# ============================================================
log "===== S1: Full GPU baseline (no MIG) ====="
mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i $GPU)
if [[ "$mig_state" == "Disabled" ]]; then
  profile_scenario "S1_fullgpu" "all" 1
else
  log "MIG enabled — skip S1 (run separately with MIG off, or reboot)"
fi

# ============================================================
# S2: 7g MIG single  — NSYS+NCU priority
# ============================================================
log "===== S2: 7g MIG single ====="
mig_on; sleep 2
mig_create "0"
UUID=$(get_uuid "7g.40gb")
if [[ -n "$UUID" ]]; then
  profile_scenario "S2_7g_mig" "\"device=$UUID\"" 1
fi

# ============================================================
# S3: 4g L1 alone (3way-asym, 2g+1g idle)  — NSYS only
# ============================================================
log "===== S3: 4g L1 alone (3way-asym) ====="
mig_create "5,14,19"
UUID=$(get_uuid "4g.20gb")
if [[ -n "$UUID" ]]; then
  profile_scenario "S3_4g_alone" "\"device=$UUID\"" 0
fi

# ============================================================
# S4: 4g L1 + 3 small AI (M4 style)  — NSYS only
# ============================================================
log "===== S4: 4g L1 + 3 small AI on 1g×3 (M4) ====="
mig_create "5,19,19,19"
UUID_4G=$(get_uuid "4g.20gb")
mapfile -t UUID_1Gs < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
  | grep "1g.5gb" | grep -oE 'MIG-[0-9a-f-]+')
if [[ -n "$UUID_4G" && ${#UUID_1Gs[@]} -ge 3 ]]; then
  CID1=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "${UUID_1Gs[0]}")
  CID2=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py"        "${UUID_1Gs[1]}")
  CID3=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_gpt2_stress.py"         "${UUID_1Gs[2]}")
  sleep 15
  profile_scenario "S4_4g_3AI_1g" "\"device=$UUID_4G\"" 0
  kill_all_bg
fi

# ============================================================
# S5: 3g L1 alone (split-60-40, 2g idle)  — NSYS+NCU priority
# ============================================================
log "===== S5: 3g L1 alone (split-60-40) ====="
mig_create "9,14"
UUID_3G=$(get_uuid "3g.20gb")
if [[ -n "$UUID_3G" ]]; then
  profile_scenario "S5_3g_alone" "\"device=$UUID_3G\"" 1
fi

# ============================================================
# S6: 3g + Qwen (A0)  — NSYS+NCU priority
# ============================================================
log "===== S6: 3g L1 + Qwen-7B on 2g (A0) ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen7b_prefill.py" "$UUID_2G")
  sleep 15
  profile_scenario "S6_3g_qwen" "\"device=$UUID_3G\"" 1
  kill_all_bg
fi

# ============================================================
# S7: 3g + NeuralRx (AR1)  — NSYS+NCU priority
# ============================================================
log "===== S7: 3g L1 + NeuralRx on 2g (AR1) ====="
UUID_3G=$(get_uuid "3g.20gb")
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_3G" && -n "$UUID_2G" ]]; then
  CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "$UUID_2G")
  sleep 15
  profile_scenario "S7_3g_neuralrx" "\"device=$UUID_3G\"" 1
  kill_all_bg
fi

# ============================================================
# S8: 3g + 2 AI on 2g+2g (M1)  — NSYS only
# ============================================================
log "===== S8: 3g L1 + 2 AI on 2g+2g (M1) ====="
mig_create "9,14,14"
UUID_3G=$(get_uuid "3g.20gb")
mapfile -t UUID_2Gs < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
  | grep "2g.10gb" | grep -oE 'MIG-[0-9a-f-]+')
if [[ -n "$UUID_3G" && ${#UUID_2Gs[@]} -ge 2 ]]; then
  CID1=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "${UUID_2Gs[0]}")
  CID2=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py"     "${UUID_2Gs[1]}")
  sleep 15
  profile_scenario "S8_3g_2AI_2g" "\"device=$UUID_3G\"" 0
  kill_all_bg
fi

# ============================================================
# S9: 3g + 3 AI on 1g×3  — NSYS+NCU priority (5/24 anomaly verify)
# ============================================================
log "===== S9: 3g L1 + 3 AI on 1g×3 ====="
mig_create "9,19,19,19,19"
UUID_3G=$(get_uuid "3g.20gb")
mapfile -t UUID_1Gs < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
  | grep "1g.5gb" | grep -oE 'MIG-[0-9a-f-]+')
if [[ -n "$UUID_3G" && ${#UUID_1Gs[@]} -ge 3 ]]; then
  CID1=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "${UUID_1Gs[0]}")
  CID2=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py"        "${UUID_1Gs[1]}")
  CID3=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_gpt2_stress.py"         "${UUID_1Gs[2]}")
  sleep 15
  profile_scenario "S9_3g_3AI_1g" "\"device=$UUID_3G\"" 1
  kill_all_bg
fi

# ============================================================
# S10: 2g L1 alone (split-40-60, 3g idle)  — NSYS+NCU priority
# ============================================================
log "===== S10: 2g L1 alone (split-40-60) ====="
mig_create "14,9"
UUID_2G=$(get_uuid "2g.10gb")
if [[ -n "$UUID_2G" ]]; then
  profile_scenario "S10_2g_alone" "\"device=$UUID_2G\"" 1
fi

# ============================================================
# S11: 2g L1 + Qwen on 3g (D1a)  — NSYS only
# ============================================================
log "===== S11: 2g L1 + Qwen on 3g (D1a) ====="
UUID_2G=$(get_uuid "2g.10gb")
UUID_3G=$(get_uuid "3g.20gb")
if [[ -n "$UUID_2G" && -n "$UUID_3G" ]]; then
  CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen7b_prefill.py" "$UUID_3G")
  sleep 15
  profile_scenario "S11_2g_qwen" "\"device=$UUID_2G\"" 0
  kill_all_bg
fi

# ============================================================
# S12: 2g L1 + 2 AI (M2)  — NSYS+NCU priority (catastrophic)
# ============================================================
log "===== S12: 2g L1 + 2 AI on 3g+2g (M2) ====="
mig_create "14,9,14"
UUID_2G_L1=$(get_uuid "2g.10gb" 1)   # first 2g for L1
UUID_3G_AI=$(get_uuid "3g.20gb")
UUID_2G_AI=$(get_uuid "2g.10gb" 2)   # second 2g for AI
if [[ -n "$UUID_2G_L1" && -n "$UUID_3G_AI" ]]; then
  CID1=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" "$UUID_3G_AI")
  [[ -n "$UUID_2G_AI" ]] && CID2=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py" "$UUID_2G_AI")
  sleep 15
  profile_scenario "S12_2g_2AI" "\"device=$UUID_2G_L1\"" 1
  kill_all_bg
fi

log ""
log "===== ALL 12 SCENARIOS DONE ====="
log "nsys outputs: $NSYS_OUT"
log "ncu outputs:  $NCU_OUT"
log ""
log "Comparison axes:"
log "  A (Partition size, L1 alone):       S1 vs S2 vs S3 vs S5 vs S10"
log "  B (AI workload, same 3g L1):        S5 vs S6 vs S7 vs S8 vs S9"
log "  C (L1 partition, same AI):          S6 vs S11   /   S8 vs S12"
log "  D (Multi-AI layout, 2g vs 1g):      S8 vs S9"
