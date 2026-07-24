#!/usr/bin/env bash
# Chain 14 — MPS breakdown threshold curve via HBM saturation gradient
# ─────────────────────────────────────────────────────────────────────
# 3 partition configs × 6 workloads × MPS on/off × 3 trials.
# BOTH L1-side AND AI-side profiled with nsys (dual profile).
#
# Configs:
#   A) 4g.20gb + 3g.20gb          (Chain 13 extension)
#   B) Full GPU 0 (no MIG)        (20260708 baseline reproduction)
#   C) 3g.20gb + 2g.10gb + 2g.10gb (small-slice isolation test — valid 40GB combo)
#
# Workloads:
#   nrx         (compute)
#   chanpred    (compute)
#   qwen_rag    (Qwen-3B n=64 continuous — real HBM stress from LLM prod)
#   whisper     (batch=4 audio streams)
#   qwen_vl     (VL-2B batch=4 images)
#   hbm_stress  (STREAM triad, 20260708 reference)
#
# Cross-partition workload: Qwen-2.5-3B always (fits in 1g.5gb via low mem util)
# ─────────────────────────────────────────────────────────────────────

set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DUR=${DUR:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
CONFIGS=${CONFIGS:-A B C}     # override for partial runs, e.g., CONFIGS='A' bash run_chain14.sh

IMAGE=airan:25-3-final
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3

SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain14
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1=""      # L1 partition
UUID_CROSS=""   # cross-partition primary (Qwen)
UUID_ALT=""     # optional alternate cross (for 3g+3g+1g variant)
MPS_ENVS=""

# ─────────────────────────────────────────────────────────────────────
# MIG / GPU state helpers
# ─────────────────────────────────────────────────────────────────────
mig_off() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 3
}

setup_config_A() {  # 4g + 3g
  log "== config A: 4g.20gb + 3g.20gb =="
  mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C 2>&1 | tail -3 >>"$OUT/run.log"
  sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_CROSS=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_ALT=""
  log "  UUID_L1(4g)=$UUID_L1  UUID_CROSS(3g)=$UUID_CROSS"
  [[ -n $UUID_L1 && -n $UUID_CROSS ]] || return 1
}

setup_config_B() {  # Full GPU 0 (no MIG)
  log "== config B: Full GPU 0 (no MIG) =="
  mig_off
  UUID_L1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | head -1)
  UUID_CROSS=""
  UUID_ALT=""
  log "  UUID_L1(full)=$UUID_L1"
  [[ -n $UUID_L1 ]] || return 1
}

setup_config_C() {  # 3g + 2g + 2g  (valid on A100-40GB: 20+10+10 = 40GB, 3+2+2 = 7/8 SMs)
  log "== config C: 3g.20gb + 2g.10gb + 2g.10gb =="
  mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 3g.20gb,2g.10gb,2g.10gb -C 2>&1 | tail -3 >>"$OUT/run.log"
  sleep 2
  UUID_L1=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)   # 3g = L1 home
  local two_g_uuids
  two_g_uuids=$(nvidia-smi -L | grep '2g.10gb' | grep -oE 'MIG-[0-9a-f-]+')
  UUID_CROSS=$(echo "$two_g_uuids" | sed -n '1p')                                    # first 2g = primary cross
  UUID_ALT=$(echo "$two_g_uuids" | sed -n '2p')                                      # second 2g = alt cross
  log "  UUID_L1(3g)=$UUID_L1"
  log "  UUID_CROSS(2g#1)=$UUID_CROSS"
  log "  UUID_ALT(2g#2)=$UUID_ALT"
  [[ -n $UUID_L1 && -n $UUID_CROSS && -n $UUID_ALT ]] || return 1
}

# ─────────────────────────────────────────────────────────────────────
# MPS server management (per-partition)
# ─────────────────────────────────────────────────────────────────────
mps_start() {
  local target_uuid=$1
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo mkdir -p /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo chmod 777 /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  docker rm -f mps_srv 2>/dev/null || true
  docker run -d --gpus "device=$target_uuid" --ipc=host --pid=host --user 0:0 \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU \
    -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU \
    --name mps_srv "$IMAGE" \
    bash -c "nvidia-cuda-mps-control -d && sleep infinity" >/dev/null
  sleep 5
  MPS_ENVS="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
}

mps_stop() {
  docker rm -f mps_srv 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  MPS_ENVS=""
}

# ─────────────────────────────────────────────────────────────────────
# Workload background launcher — includes nsys wrapping for AI-side
#   $1=workload_name  $2=target_uuid  $3=tag  $4=nsys_wrap(0/1)
# ─────────────────────────────────────────────────────────────────────
start_workload_bg() {
  local wl=$1 target_uuid=$2 tag=$3 nsys_wrap=${4:-1}
  local nsys_prefix=""
  if [[ $nsys_wrap == 1 ]]; then
    nsys_prefix="nsys profile --trace=cuda --duration=$DUR --output=/aiout/${tag}_ai --force-overwrite=true --stats=false"
  fi
  case "$wl" in
    nrx)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
        -e cuBB_SDK=/opt/nvidia/cuBB \
        -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "$nsys_prefix python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR + 60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    chanpred)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "$nsys_prefix python3 run_chanpred.py ${tag} $((DUR + 60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    qwen_rag)
      # vLLM image lacks nsys — skip AI-side profile
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        --entrypoint bash \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$VLLM_IMAGE" \
        -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR + 60)) --rag > /aiout/${tag}.log 2>&1"
      ;;
    qwen_llm_cross)
      # Cross-partition LLM baseline; vLLM lacks nsys — skip AI-side profile
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        --entrypoint bash \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$VLLM_IMAGE" \
        -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR + 60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 32 --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
      ;;
    whisper)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "pip install -q 'transformers==4.44.2' accelerate==0.34.2 datasets soundfile 2>&1 > /aiout/${tag}_pip.log && $nsys_prefix python3 run_whisper_hbm.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    qwen_vl)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "pip install -q 'transformers==4.45.2' accelerate==0.34.2 qwen-vl-utils Pillow datasets 2>&1 > /aiout/${tag}_pip.log && $nsys_prefix python3 run_qwen_vl_hbm.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    hbm_stress)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$IMAGE" \
        bash -c "$nsys_prefix python3 run_hbm_stress.py $tag $((DUR + 60)) --gb 8 > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    qwen_chat_b1)
      # Truly memory-bound: Qwen-3B batch=1 latency-critical decode (eager mode → many kernel launches)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        --entrypoint bash \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$VLLM_IMAGE" \
        -c "pip install -q datasets 2>/dev/null; python3 run_qwen_chat_b1.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    whisper_stream_b1)
      # Realistic streaming ASR: batch=1, 5s audio chunks
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "pip install -q 'transformers==4.44.2' accelerate==0.34.2 datasets soundfile 2>&1 > /aiout/${tag}_pip.log && $nsys_prefix python3 run_whisper_stream_b1.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    bert_b1)
      # BERT-large batch=1 continuous — realistic NLP + latency-critical
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "pip install -q 'transformers==4.44.2' accelerate==0.34.2 datasets 2>&1 > /aiout/${tag}_pip.log && $nsys_prefix python3 run_bert_b1.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    embed_lookup)
      # DLRM-style embedding random-access lookup — TRUE memory-bound, cache-hostile
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$IMAGE" \
        bash -c "$nsys_prefix python3 run_embed_lookup.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    memcpy_loop)
      # High-rate small-copy loop — extreme launch rate + moderate BW
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$IMAGE" \
        bash -c "$nsys_prefix python3 run_memcpy_loop.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    idle) : ;;
    *) log "unknown workload $wl"; return 1;;
  esac
}

kill_all_workloads() {
  docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null
  sleep 2
}

# ─────────────────────────────────────────────────────────────────────
# L1 profiling (30s nsys)
# ─────────────────────────────────────────────────────────────────────
profile_l1() {
  local label=$1 target_uuid=$2
  docker run --rm --user 0:0 --gpus "device=$target_uuid" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=$DUR --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py $label $CELLS $L1_ITERS" >/dev/null 2>&1
}

# ─────────────────────────────────────────────────────────────────────
# Matrix runs
# ─────────────────────────────────────────────────────────────────────
run_matrix_same_part() {
  local cfg=$1
  log "===== SAME-PARTITION cfg=$cfg (L1 + co-tenant in $UUID_L1) ====="

  # SP baseline (L1 alone, cross-Qwen if applicable)
  for t in $(seq 1 $N_TRIALS); do
    log "--- SP-0 baseline cfg=$cfg t=$t ---"
    if [[ -n $UUID_CROSS ]]; then
      start_workload_bg qwen_llm_cross "$UUID_CROSS" "cfg${cfg}_SP0_baseline_cross_t${t}"
      sleep 20
    fi
    profile_l1 "cfg${cfg}_SP0_baseline_t${t}" "$UUID_L1"
    kill_all_workloads
  done

  # SP main
  for WL in nrx chanpred qwen_rag whisper qwen_vl hbm_stress qwen_chat_b1 whisper_stream_b1 bert_b1 embed_lookup memcpy_loop; do
    for MPS in off on; do
      [[ $MPS == on ]] && mps_start "$UUID_L1"
      LABEL="cfg${cfg}_SP_${WL}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        if [[ -n $UUID_CROSS ]]; then
          start_workload_bg qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"
          sleep 20
        fi
        start_workload_bg "$WL" "$UUID_L1" "${LABEL}_same_t${t}"
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"
        kill_all_workloads
      done
      [[ $MPS == on ]] && mps_stop
    done
  done
}

run_matrix_cross_part() {
  local cfg=$1
  [[ -z $UUID_CROSS ]] && { log "no CROSS partition for cfg=$cfg, skip CP"; return; }
  log "===== CROSS-PARTITION cfg=$cfg (L1 alone in $UUID_L1) ====="
  for WL in idle nrx chanpred qwen_rag whisper qwen_vl hbm_stress qwen_chat_b1 whisper_stream_b1 bert_b1 embed_lookup memcpy_loop qwen_llm_cross; do
    LABEL="cfg${cfg}_CP_${WL}"
    log "--- $LABEL ---"
    for t in $(seq 1 $N_TRIALS); do
      if [[ $WL != idle ]]; then
        start_workload_bg "$WL" "$UUID_CROSS" "${LABEL}_t${t}"
        sleep 20
      fi
      profile_l1 "${LABEL}_t${t}" "$UUID_L1"
      kill_all_workloads
    done
  done
  # Extra: for config C, also test workload in 1g slot
  if [[ -n $UUID_ALT ]]; then
    log "===== CROSS-PARTITION cfg=$cfg 1g slot variant ====="
    for WL in idle nrx chanpred whisper qwen_llm_cross hbm_stress; do
      LABEL="cfg${cfg}_CP1g_${WL}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        if [[ $WL != idle ]]; then
          start_workload_bg "$WL" "$UUID_ALT" "${LABEL}_t${t}"
          sleep 20
        fi
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"
        kill_all_workloads
      done
    done
  fi
}

# ─────────────────────────────────────────────────────────────────────
# Main loop over configs
# ─────────────────────────────────────────────────────────────────────
log "=== chain14 START configs=$CONFIGS ==="

for cfg in $CONFIGS; do
  case $cfg in
    A) setup_config_A || { log "cfg A setup failed"; continue; } ;;
    B) setup_config_B || { log "cfg B setup failed"; continue; } ;;
    C) setup_config_C || { log "cfg C setup failed"; continue; } ;;
    *) log "unknown cfg=$cfg"; continue;;
  esac
  run_matrix_same_part "$cfg"
  run_matrix_cross_part "$cfg"
done

mig_off
log "=== chain14 DONE ==="
echo "captures: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)"
