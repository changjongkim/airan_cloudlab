#!/usr/bin/env bash
# Chain 16 — Multi-instance realistic RAN AI mix as same-partition co-tenant
# ─────────────────────────────────────────────────────────────────────
# Purpose: test whether MIG+MPS handles high concurrent load from realistic
# per-UE / per-cell RAN AI workloads (not synthetic HBM stress).
#
# Same-partition co-tenant candidates (all realistic AI-RAN):
#   ranai_mix        2× NRx + 4× CsiNet + 8× BeamPred (14 workers)
#   ranai_mix_heavy  4× NRx + 8× CsiNet + 16× BeamPred (28 workers, high pressure)
#   nrx_multi4       4 concurrent NRx (multi-cell scenario)
#
# Reference: nrx (single instance from chain14), memcpy_loop (control)
#
# Cross-partition (always): Qwen-3B (edge AI baseline)
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
OUT=/mydata/results/$DATE_DIR/chain16
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1="" UUID_CROSS="" UUID_ALT="" MPS_ENVS=""

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
  log "  UUID_L1=$UUID_L1 UUID_CROSS=$UUID_CROSS"
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
  log "  UUID_L1=$UUID_L1 UUID_CROSS=$UUID_CROSS UUID_ALT=$UUID_ALT"
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
mps_stop() { docker rm -f mps_srv 2>/dev/null || true; sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU; MPS_ENVS=""; }
kill_all_workloads() { docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 2; }

start_workload_bg() {
  local wl=$1 u=$2 tag=$3
  local nsp="nsys profile --trace=cuda --duration=$DUR --output=/aiout/${tag}_ai --force-overwrite=true --stats=false"
  case "$wl" in
    ranai_mix)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "$nsp python3 run_ranai_mix.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1"
      ;;
    ranai_mix_heavy)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "$nsp python3 run_ranai_mix.py $tag $((DUR + 60)) --n_nrx 4 --n_csinet 8 --n_beampred 16 > /aiout/${tag}.log 2>&1"
      ;;
    nrx_multi4)
      # 4 concurrent NRx instances (multi-cell scenario)
      for i in 1 2 3 4; do
        docker run -d --rm --init --user 0:0 --gpus "device=$u" \
          --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
          -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
          -e cuBB_SDK=/opt/nvidia/cuBB \
          -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
          -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
          --name "ai_${tag}_c${i}" "$IMAGE" \
          bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR + 60)) > /aiout/${tag}_c${i}.log 2>&1"
      done
      ;;
    qwen_llm_cross)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        --entrypoint bash \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$VLLM_IMAGE" \
        -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR + 60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 32 --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
      ;;
    idle) : ;;
    *) log "unknown $wl"; return 1;;
  esac
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

run_sp() {
  local cfg=$1
  log "===== SAME-PART cfg=$cfg (L1 in $UUID_L1) ====="
  # baseline
  for t in $(seq 1 $N_TRIALS); do
    log "--- SP-0 baseline cfg=$cfg t=$t ---"
    [[ -n $UUID_CROSS ]] && { start_workload_bg qwen_llm_cross "$UUID_CROSS" "cfg${cfg}_SP0_baseline_cross_t${t}"; sleep 20; }
    profile_l1 "cfg${cfg}_SP0_baseline_t${t}" "$UUID_L1"; kill_all_workloads
  done
  # 3 new workloads × MPS on/off
  for WL in ranai_mix ranai_mix_heavy nrx_multi4; do
    for MPS in off on; do
      [[ $MPS == on ]] && mps_start "$UUID_L1"
      LABEL="cfg${cfg}_SP_${WL}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_workload_bg qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        start_workload_bg "$WL" "$UUID_L1" "${LABEL}_same_t${t}"
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
      [[ $MPS == on ]] && mps_stop
    done
  done
}

log "=== chain16 START configs=$CONFIGS ==="
for cfg in $CONFIGS; do
  case $cfg in
    A) setup_config_A || continue ;;
    B) setup_config_B || continue ;;
    C) setup_config_C || continue ;;
  esac
  run_sp "$cfg"
done
mig_off
log "=== chain16 DONE ==="
echo "captures: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)"
