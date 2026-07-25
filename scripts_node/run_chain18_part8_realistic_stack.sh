#!/usr/bin/env bash
# Chain 18 Part 8 — Realistic AI-RAN diverse workload stack
#
# Missing gap in Chain 9-17: we tested L1 + N identical NRx, but real deployment
# has 5-8 DIVERSE workloads (Qwen chat + Whisper + BERT + NRx + CSI + Beam + …).
# Part 8 quantifies L1 sync stability under a realistic diverse stack.
#
# Scenarios (3 trials each):
#   CP-realistic: L1 on 4g.20gb, 5-workload diverse stack on 3g.20gb
#   SP-realistic: L1 on 4g.20gb, 5-workload diverse stack on SAME 4g.20gb
#   CP-uniform  : L1 on 4g.20gb, 5× NRx on 3g.20gb   (control: uniform vs diverse)
#   SP-uniform  : L1 on 4g.20gb, 5× NRx on SAME 4g.20gb
#   Baseline    : L1 alone on 4g.20gb
set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DUR=${DUR:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}

IMAGE=airan:25-3-final
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain18_p8
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1="" UUID_AI="" MPS_ENVS_L1="" MPS_ENVS_AI=""

setup_configA() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n systemctl stop nvidia-dcgm 2>/dev/null || true
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 3
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C 2>&1 | tail -3 >>"$OUT/run.log"
  sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_AI=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  log "  UUID_L1=$UUID_L1 UUID_AI=$UUID_AI"
}

mps_start_uuid() {
  local u=$1 tag=$2
  local pipe=/tmp/mps_pipe_${tag}
  local logd=/tmp/mps_log_${tag}
  sudo rm -rf $pipe $logd
  sudo mkdir -p $pipe $logd
  sudo chmod 777 $pipe $logd
  docker rm -f mps_srv_${tag} 2>/dev/null || true
  docker run -d --gpus "device=$u" --ipc=host --pid=host --user 0:0 \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=$pipe \
    -e CUDA_MPS_LOG_DIRECTORY=$logd \
    --name mps_srv_${tag} "$IMAGE" \
    bash -c "nvidia-cuda-mps-control -d && sleep infinity" >/dev/null
  sleep 5
  eval "MPS_ENVS_${tag^^}=\"-e CUDA_MPS_PIPE_DIRECTORY=$pipe -e CUDA_MPS_LOG_DIRECTORY=$logd\""
}
mps_stop_uuid() {
  local tag=$1
  docker rm -f mps_srv_${tag} 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_${tag} /tmp/mps_log_${tag}
  eval "MPS_ENVS_${tag^^}=\"\""
}
kill_all_workloads() { docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 2; }

# ─── Realistic diverse AI stack launchers ─────────────────────
start_qwen() {
  local u=$1 tag=$2 mps_envs=$3
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    --entrypoint bash \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$VLLM_IMAGE" \
    -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR+60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 16 --gpu_mem_util 0.35 > /aiout/${tag}.log 2>&1"
}
start_whisper() {
  local u=$1 tag=$2 mps_envs=$3
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "python3 run_whisper.py $tag $((DUR+60)) > /aiout/${tag}.log 2>&1"
}
start_bert() {
  local u=$1 tag=$2 mps_envs=$3
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "python3 run_bert_batch.py $tag $((DUR+60)) --batch 16 > /aiout/${tag}.log 2>&1"
}
start_nrx() {
  local u=$1 tag=$2 mps_envs=$3
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
    -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    --name "ai_${tag}" "$IMAGE" \
    bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR+60)) > /aiout/${tag}.log 2>&1"
}
start_csinet() {
  local u=$1 tag=$2 mps_envs=$3
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "python3 run_ranai_mix.py $tag $((DUR+60)) --n_nrx 0 --n_csinet 1 --n_beampred 0 > /aiout/${tag}.log 2>&1"
}
start_beampred() {
  local u=$1 tag=$2 mps_envs=$3
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "python3 run_ranai_mix.py $tag $((DUR+60)) --n_nrx 0 --n_csinet 0 --n_beampred 1 > /aiout/${tag}.log 2>&1"
}

start_diverse_stack() {
  local u=$1 label=$2 mps_envs=$3
  start_qwen     "$u" "${label}_qwen"     "$mps_envs"
  start_whisper  "$u" "${label}_whisper"  "$mps_envs"
  start_bert     "$u" "${label}_bert"     "$mps_envs"
  start_nrx      "$u" "${label}_nrx"      "$mps_envs"
  start_csinet   "$u" "${label}_csinet"   "$mps_envs"
  start_beampred "$u" "${label}_beampred" "$mps_envs"
}
start_uniform_stack() {
  local u=$1 label=$2 mps_envs=$3 N=$4
  for i in $(seq 1 $N); do start_nrx "$u" "${label}_nrx${i}" "$mps_envs"; done
}

# ─── L1 profile ───────────────────────────────────────────────
profile_l1() {
  local label=$1 mps_envs=$2
  docker run --rm --user 0:0 --gpus "device=$UUID_L1" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=$DUR --output=/out/${label} --force-overwrite=true python3 real_l1.py $label $CELLS $L1_ITERS" > "$OUT/${label}.stdout" 2>&1
}

log "=== chain18 Part 8 (Realistic Stack) START ==="
setup_configA

# --- Baseline: L1 alone ---
for t in $(seq 1 $N_TRIALS); do
  LABEL="p8_baseline_t${t}"; log "--- $LABEL ---"
  profile_l1 "$LABEL" ""
done

# --- CP diverse: L1 on 4g, 6 diverse workloads on 3g ---
for t in $(seq 1 $N_TRIALS); do
  LABEL="p8_CPdiverse_t${t}"; log "--- $LABEL ---"
  mps_start_uuid "$UUID_AI" "ai"
  start_diverse_stack "$UUID_AI" "$LABEL" "$MPS_ENVS_AI"
  sleep 30
  profile_l1 "$LABEL" ""
  kill_all_workloads
  mps_stop_uuid "ai"
done

# --- CP uniform: L1 on 4g, 6× NRx on 3g ---
for t in $(seq 1 $N_TRIALS); do
  LABEL="p8_CPuniform_t${t}"; log "--- $LABEL ---"
  mps_start_uuid "$UUID_AI" "ai"
  start_uniform_stack "$UUID_AI" "$LABEL" "$MPS_ENVS_AI" 6
  sleep 20
  profile_l1 "$LABEL" ""
  kill_all_workloads
  mps_stop_uuid "ai"
done

# --- SP diverse: L1 + 6 diverse all on 4g.20gb ---
for t in $(seq 1 $N_TRIALS); do
  LABEL="p8_SPdiverse_t${t}"; log "--- $LABEL ---"
  mps_start_uuid "$UUID_L1" "l1"
  start_diverse_stack "$UUID_L1" "$LABEL" "$MPS_ENVS_L1"
  sleep 30
  profile_l1 "$LABEL" "$MPS_ENVS_L1"
  kill_all_workloads
  mps_stop_uuid "l1"
done

# --- SP uniform: L1 + 6× NRx all on 4g.20gb ---
for t in $(seq 1 $N_TRIALS); do
  LABEL="p8_SPuniform_t${t}"; log "--- $LABEL ---"
  mps_start_uuid "$UUID_L1" "l1"
  start_uniform_stack "$UUID_L1" "$LABEL" "$MPS_ENVS_L1" 6
  sleep 20
  profile_l1 "$LABEL" "$MPS_ENVS_L1"
  kill_all_workloads
  mps_stop_uuid "l1"
done

log "=== chain18 Part 8 DONE ==="
echo "captures: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)"
touch /users/sgkim/CHAIN18_PART8_DONE
