#!/usr/bin/env bash
# Chain 13 — MIG same-partition (co-tenant in 4g) vs cross-partition (co-tenant in 3g)
# with 5 co-tenant workloads (NRx, ChanPred, Qwen-RAG, Whisper, Qwen2-VL).
# Cross-partition은 항상 Qwen-2.5-7B LLM이 3g에 상주 (실전 배포 조건).

set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DUR=${DUR:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain13
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_4G=""
UUID_3G=""
MPS_ENVS=""

# ===========================================================================
# MIG 파티션 관리
# ===========================================================================
mig_setup() {
  log "== enabling MIG on GPU $GPU =="
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sleep 2

  log "== creating 4g.20gb + 3g.20gb =="
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C >/dev/null 2>&1 || {
    log "MIG creation FAILED"; return 1
  }
  sleep 2

  # UUID 캡처: 프로파일명으로 명시적 매칭 (nvidia-smi -L은 GPU instance ID 순으로 뽑아서 4g/3g 순서가 뒤바뀔 수 있음)
  UUID_4G=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_3G=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  log "  UUID_4G=$UUID_4G"
  log "  UUID_3G=$UUID_3G"
  if [[ -z "$UUID_4G" || -z "$UUID_3G" ]]; then
    log "no UUIDs found"; return 1
  fi
  return 0
}

mig_teardown() {
  log "== disabling MIG =="
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 2
}

# ===========================================================================
# MPS 관리 (4g partition에 대해서만)
# ===========================================================================
start_mps_on_4g() {
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

stop_mps() {
  docker rm -f mps_srv 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  MPS_ENVS=""
}

# ===========================================================================
# 워크로드 실행 (background container)
# ===========================================================================
start_workload_bg() {
  local wl=$1 target_uuid=$2 tag=$3
  case "$wl" in
    nrx)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
        -e cuBB_SDK=/opt/nvidia/cuBB \
        -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR + 60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    chanpred)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" \
        -w /scripts --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "python3 run_chanpred.py ${tag} $((DUR + 60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    qwen_llm|qwen_rag)
      local rag_flag=""; [[ "$wl" == "qwen_rag" ]] && rag_flag="--rag"
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        --entrypoint python3 \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$VLLM_IMAGE" \
        run_qwen_llm.py "$tag" $((DUR + 60)) $rag_flag >/dev/null
      ;;
    whisper)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "pip install -q 'transformers==4.44.2' accelerate==0.34.2 && python3 run_whisper.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    qwen_vl)
      docker run -d --rm --init --user 0:0 --gpus "device=$target_uuid" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "pip install -q 'transformers==4.45.2' accelerate==0.34.2 qwen-vl-utils Pillow && python3 run_qwen_vl.py $tag $((DUR + 60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    idle) : ;;  # nothing to start
    *) log "unknown workload $wl"; return 1;;
  esac
}

kill_all_workloads() {
  docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null
  sleep 2
}

# ===========================================================================
# L1 프로파일링
# ===========================================================================
profile_l1() {
  local label=$1 target_uuid=$2
  docker run --rm --user 0:0 --gpus "device=$target_uuid" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=$DUR --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py $label $CELLS $L1_ITERS" >/dev/null 2>&1
}

# ===========================================================================
# Same-partition 매트릭스
# ===========================================================================
run_same_partition() {
  log "===== SAME-PARTITION (4g에 L1 + co-tenant, 3g에 Qwen 상주) ====="

  for WL in nrx chanpred qwen_rag whisper qwen_vl; do
    for MPS in off on; do
      # SP-0: baseline은 처음 한 번만 (WL=nrx, MPS=off일 때)
      if [[ "$WL" == "nrx" && "$MPS" == "off" ]]; then
        log "--- SP-0 baseline: L1 alone in 4g, Qwen in 3g ---"
        for t in $(seq 1 $N_TRIALS); do
          start_workload_bg qwen_llm "$UUID_3G" "sp0_cross_t${t}"
          sleep 20   # Qwen warmup
          profile_l1 "SP0_baseline_t${t}" "$UUID_4G"
          kill_all_workloads
        done
      fi

      # SP main runs
      if [[ "$MPS" == "on" ]]; then start_mps_on_4g "$UUID_4G"; fi

      LABEL="SP_${WL}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        start_workload_bg qwen_llm "$UUID_3G" "${LABEL}_cross_t${t}"
        sleep 20  # Qwen warmup on 3g
        start_workload_bg "$WL"    "$UUID_4G" "${LABEL}_same_t${t}"
        sleep 15  # same-part warmup
        profile_l1 "${LABEL}_t${t}" "$UUID_4G"
        kill_all_workloads
      done

      if [[ "$MPS" == "on" ]]; then stop_mps; fi
    done
  done
}

# ===========================================================================
# Cross-partition 매트릭스
# ===========================================================================
run_cross_partition() {
  log "===== CROSS-PARTITION (4g에 L1 alone, 3g에 각 워크로드) ====="

  for WL in idle nrx chanpred qwen_rag whisper qwen_vl qwen_llm; do
    LABEL="CP_${WL}"
    log "--- $LABEL ---"
    for t in $(seq 1 $N_TRIALS); do
      if [[ "$WL" != "idle" ]]; then
        start_workload_bg "$WL" "$UUID_3G" "${LABEL}_t${t}"
        sleep 20
      fi
      profile_l1 "${LABEL}_t${t}" "$UUID_4G"
      kill_all_workloads
    done
  done
}

# ===========================================================================
# Main
# ===========================================================================
log "=== chain13 START ==="
mig_setup || { log "MIG setup failed, abort"; exit 1; }

run_same_partition
run_cross_partition

mig_teardown
log "=== chain13 DONE ==="
ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l
