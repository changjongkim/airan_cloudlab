#!/usr/bin/env bash
# Part 5 only: multi-thread vs multi-process controlled experiment
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
OUT=/mydata/results/$DATE_DIR/chain18_p5
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }
UUID_L1="" UUID_CROSS="" MPS_ENVS=""

mig_off() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n systemctl stop nvidia-dcgm 2>/dev/null || true
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 3
}
setup_A() {
  mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C >>"$OUT/run.log" 2>&1; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_CROSS=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  log "  UUID_L1=$UUID_L1 UUID_CROSS=$UUID_CROSS"
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
mps_stop() { docker rm -f mps_srv 2>/dev/null || true; sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU; MPS_ENVS=""; }
kill_all() { docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 2; }

start_qwen_cross() {
  local u=$1 tag=$2
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    --entrypoint bash \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$VLLM_IMAGE" \
    -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR+60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 32 --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
}
start_ranai_mix() {
  local u=$1 tag=$2 nrx=$3 csi=$4 beam=$5
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "python3 run_ranai_mix.py $tag $((DUR+60)) --n_nrx $nrx --n_csinet $csi --n_beampred $beam > /aiout/${tag}.log 2>&1" >/dev/null
}
profile_l1() {
  local label=$1 u=$2
  docker run --rm --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=$DUR --output=/out/${label} --force-overwrite=true python3 real_l1.py $label $CELLS $L1_ITERS" >/dev/null 2>&1
}

log "=== Part 5: Multi-thread vs multi-process ==="
setup_A
# Thread-only (1 process, varying beam thread count)
mps_start "$UUID_L1"
for BEAM in 4 8 16 32; do
  LABEL="p5_thrOnly_beam${BEAM}"
  log "--- $LABEL ---"
  for t in $(seq 1 $N_TRIALS); do
    start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
    start_ranai_mix "$UUID_L1" "${LABEL}_same_t${t}" 2 4 $BEAM
    sleep 15
    profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all
  done
done
mps_stop
# Process-only (N processes, each ranai_mix=14 threads)
for N in 1 2 4 8; do
  mps_start "$UUID_L1"
  LABEL="p5_procOnly_N${N}"
  log "--- $LABEL ---"
  for t in $(seq 1 $N_TRIALS); do
    start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
    for i in $(seq 1 $N); do start_ranai_mix "$UUID_L1" "${LABEL}_same_t${t}_p${i}" 2 4 8; done
    sleep 15
    profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all
  done
  mps_stop
done
mig_off
log "=== Part 5 DONE ==="
touch /users/sgkim/CHAIN18_PART5_DONE
