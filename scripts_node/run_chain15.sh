#!/usr/bin/env bash
# Chain 15 — Batch-size sweep on realistic workloads
# ─────────────────────────────────────────────────────────────────────
# Purpose: hold workload identity constant, vary batch → sweep arithmetic
# intensity from memory-bound (small batch) to compute-bound (large batch).
# Measure MPS on/off cudaFree penalty at each point.
#
# Configs: A (4g+3g), B (Full GPU), C (3g+2g+2g)
#
# Workload × batch sweep:
#   qwen_chat_bN   for N in 1, 2, 4, 8, 16, 32
#   bert_bN        for N in 1, 4, 16, 64
#   whisper_bN     for N in 1, 2, 4, 8
#   qwen_vl_bN     for N in 1, 2, 4
#
# Reference (no sweep, same as chain14):
#   nrx, chanpred, hbm_stress
# ─────────────────────────────────────────────────────────────────────
set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DUR=${DUR:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
CONFIGS=${CONFIGS:-A B C}

IMAGE=airan:25-3-final
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3

SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain15
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1="" UUID_CROSS="" UUID_ALT="" MPS_ENVS=""

# ─── MIG / GPU state helpers (identical to chain14) ────────────────
mig_off() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 3
}

setup_config_A() {
  log "== config A: 4g.20gb + 3g.20gb =="; mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C 2>&1 | tail -3 >>"$OUT/run.log"; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_CROSS=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_ALT=""
  log "  UUID_L1=$UUID_L1  UUID_CROSS=$UUID_CROSS"
  [[ -n $UUID_L1 && -n $UUID_CROSS ]] || return 1; return 0
}

setup_config_B() {
  log "== config B: Full GPU 0 =="; mig_off
  UUID_L1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | head -1)
  UUID_CROSS=""; UUID_ALT=""
  log "  UUID_L1=$UUID_L1"; [[ -n $UUID_L1 ]] || return 1; return 0
}

setup_config_C() {
  log "== config C: 3g.20gb + 2g.10gb + 2g.10gb =="; mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 3g.20gb,2g.10gb,2g.10gb -C 2>&1 | tail -3 >>"$OUT/run.log"; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  local two; two=$(nvidia-smi -L | grep '2g.10gb' | grep -oE 'MIG-[0-9a-f-]+')
  UUID_CROSS=$(echo "$two" | sed -n '1p'); UUID_ALT=$(echo "$two" | sed -n '2p')
  log "  UUID_L1=$UUID_L1  UUID_CROSS=$UUID_CROSS  UUID_ALT=$UUID_ALT"
  [[ -n $UUID_L1 && -n $UUID_CROSS && -n $UUID_ALT ]] || return 1; return 0
}

mps_start() {
  local u=$1
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo mkdir -p /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo chmod 777 /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  docker rm -f mps_srv 2>/dev/null || true
  docker run -d --gpus "device=$u" --ipc=host --pid=host --user 0:0 \
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

kill_all_workloads() { docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 2; }

# ─── Workload starters — parameterized by batch ───────────────────
start_qwen_chat() {  # $1=uuid $2=tag $3=batch
  local u=$1 tag=$2 batch=$3
  local ncat=$batch
  [[ $batch -eq 1 ]] && ncat=1
  # Note: qwen_chat_b1 script hardcodes batch=1; use qwen_rag_hbm for batch>1 via n_concurrent
  if [[ $batch -eq 1 ]]; then
    docker run -d --rm --init --user 0:0 --gpus "device=$u" \
      --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS --entrypoint bash \
      -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
      -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
      --name "ai_${tag}" "$VLLM_IMAGE" \
      -c "pip install -q datasets 2>/dev/null; python3 run_qwen_chat_b1.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
  else
    docker run -d --rm --init --user 0:0 --gpus "device=$u" \
      --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS --entrypoint bash \
      -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
      -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
      --name "ai_${tag}" "$VLLM_IMAGE" \
      -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR + 60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent $ncat --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
  fi
}

start_bert() {  # $1=uuid $2=tag $3=batch
  local u=$1 tag=$2 batch=$3 nsys_p=$4
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "pip install -q 'transformers==4.44.2' accelerate==0.34.2 datasets 2>&1 > /aiout/${tag}_pip.log && $nsys_p python3 run_bert_batch.py $tag $((DUR + 60)) --batch $batch > /aiout/${tag}.log 2>&1"
}

start_whisper() {  # $1=uuid $2=tag $3=batch
  local u=$1 tag=$2 batch=$3 nsys_p=$4
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "pip install -q 'transformers==4.44.2' accelerate==0.34.2 datasets soundfile 2>&1 > /aiout/${tag}_pip.log && $nsys_p python3 run_whisper_hbm.py $tag $((DUR + 60)) --batch $batch > /aiout/${tag}.log 2>&1"
}

start_vl() {  # $1=uuid $2=tag $3=batch
  local u=$1 tag=$2 batch=$3 nsys_p=$4
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "pip install -q 'transformers==4.45.2' accelerate==0.34.2 qwen-vl-utils Pillow datasets 2>&1 > /aiout/${tag}_pip.log && $nsys_p python3 run_qwen_vl_hbm.py $tag $((DUR + 60)) --batch $batch > /aiout/${tag}.log 2>&1"
}

start_qwen_llm_cross() {  # $1=uuid $2=tag
  local u=$1 tag=$2
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS --entrypoint bash \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$VLLM_IMAGE" \
    -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR + 60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 32 --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
}

profile_l1() {
  local label=$1 u=$2
  docker run --rm --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=$DUR --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py $label $CELLS $L1_ITERS" >/dev/null 2>&1
}

# ─── Batch sweep matrix (SP) ─────────────────────────────────────
QWEN_CHAT_BS="1 2 4 8 16 32"
BERT_BS="1 4 16 64"
WHISPER_BS="1 2 4 8"
VL_BS="1 2 4"

run_sp() {
  local cfg=$1
  log "===== SAME-PARTITION cfg=$cfg (L1 in $UUID_L1) ====="

  # baseline
  for t in $(seq 1 $N_TRIALS); do
    log "--- SP-0 baseline cfg=$cfg t=$t ---"
    [[ -n $UUID_CROSS ]] && { start_qwen_llm_cross "$UUID_CROSS" "cfg${cfg}_SP0_baseline_cross_t${t}"; sleep 20; }
    profile_l1 "cfg${cfg}_SP0_baseline_t${t}" "$UUID_L1"; kill_all_workloads
  done

  for MPS in off on; do
    [[ $MPS == on ]] && mps_start "$UUID_L1"
    local NSYS_P="nsys profile --trace=cuda --duration=$DUR --output=/aiout/AI_ai --force-overwrite=true --stats=false"

    # Qwen-chat batch sweep
    for BS in $QWEN_CHAT_BS; do
      LABEL="cfg${cfg}_SP_qwen_chat_b${BS}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        start_qwen_chat "$UUID_L1" "${LABEL}_same_t${t}" $BS
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
    done

    # BERT batch sweep
    for BS in $BERT_BS; do
      LABEL="cfg${cfg}_SP_bert_b${BS}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        local nsp="nsys profile --trace=cuda --duration=$DUR --output=/aiout/${LABEL}_same_t${t}_ai --force-overwrite=true --stats=false"
        start_bert "$UUID_L1" "${LABEL}_same_t${t}" $BS "$nsp"
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
    done

    # Whisper batch sweep
    for BS in $WHISPER_BS; do
      LABEL="cfg${cfg}_SP_whisper_b${BS}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        local nsp="nsys profile --trace=cuda --duration=$DUR --output=/aiout/${LABEL}_same_t${t}_ai --force-overwrite=true --stats=false"
        start_whisper "$UUID_L1" "${LABEL}_same_t${t}" $BS "$nsp"
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
    done

    # VL batch sweep
    for BS in $VL_BS; do
      LABEL="cfg${cfg}_SP_vl_b${BS}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        local nsp="nsys profile --trace=cuda --duration=$DUR --output=/aiout/${LABEL}_same_t${t}_ai --force-overwrite=true --stats=false"
        start_vl "$UUID_L1" "${LABEL}_same_t${t}" $BS "$nsp"
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
    done

    [[ $MPS == on ]] && mps_stop
  done
}

log "=== chain15 START configs=$CONFIGS ==="
for cfg in $CONFIGS; do
  case $cfg in
    A) setup_config_A || { log "cfg A fail"; continue; } ;;
    B) setup_config_B || { log "cfg B fail"; continue; } ;;
    C) setup_config_C || { log "cfg C fail"; continue; } ;;
  esac
  run_sp "$cfg"
done
mig_off
log "=== chain15 DONE ==="
echo "captures: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)"
