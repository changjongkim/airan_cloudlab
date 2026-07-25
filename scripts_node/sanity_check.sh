#!/usr/bin/env bash
# Chain 13 sanity check — each workload alone
# 사용법: bash sanity_check.sh [qwen_llm|qwen_rag|whisper|qwen_vl|chanpred|nrx|all]

set -uo pipefail

WORKLOAD=${1:-all}
DURATION=${DURATION:-20}
HF_CACHE=${HF_CACHE:-/mydata/hf_cache}
SDK=/mydata/aerial-cuda-accelerated-ran
SCRIPT_DIR=/users/sgkim/cloudlab_aerial
REPO=/mydata/AIRAN_Changjong
GPU=${GPU:-0}
OUT=/tmp/chain13_sanity
mkdir -p "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }

run_qwen_llm() {
  log "=== qwen_llm sanity ($DURATION s) ==="
  docker run --rm --user 0:0 --gpus "device=$GPU" --ipc=host --pid=host \
    --entrypoint python3 \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT_DIR:/scripts" -w /scripts \
    vllm/vllm-openai:v0.6.6 \
    run_qwen_llm.py sanity $DURATION 2>&1 | tee "$OUT/qwen_llm.log" | tail -20
}

run_qwen_rag() {
  log "=== qwen_rag sanity ($DURATION s) ==="
  docker run --rm --user 0:0 --gpus "device=$GPU" --ipc=host --pid=host \
    --entrypoint python3 \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT_DIR:/scripts" -w /scripts \
    vllm/vllm-openai:v0.6.6 \
    run_qwen_llm.py sanity_rag $DURATION --rag 2>&1 | tee "$OUT/qwen_rag.log" | tail -20
}

run_whisper() {
  log "=== whisper sanity ($DURATION s) ==="
  docker run --rm --user 0:0 --gpus "device=$GPU" --ipc=host --pid=host \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
    -v "$SCRIPT_DIR:/scripts" -w /scripts \
    nvcr.io/nvidia/pytorch:24.10-py3 bash -c \
    "pip install -q 'transformers==4.44.2' accelerate==0.34.2 && python3 run_whisper.py sanity $DURATION" \
    2>&1 | tee "$OUT/whisper.log" | tail -20
}

run_qwen_vl() {
  log "=== qwen_vl sanity ($DURATION s) ==="
  docker run --rm --user 0:0 --gpus "device=$GPU" --ipc=host --pid=host \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e TRANSFORMERS_CACHE=/hf \
    -v "$SCRIPT_DIR:/scripts" -w /scripts \
    nvcr.io/nvidia/pytorch:24.10-py3 bash -c \
    "pip install -q 'transformers==4.45.2' accelerate==0.34.2 qwen-vl-utils Pillow && python3 run_qwen_vl.py sanity $DURATION" \
    2>&1 | tee "$OUT/qwen_vl.log" | tail -20
}

run_chanpred() {
  log "=== chanpred sanity ($DURATION s) — PyTorch mini-transformer CSI predictor ==="
  docker run --rm --user 0:0 --gpus "device=$GPU" --ipc=host --pid=host \
    -v "$SCRIPT_DIR:/scripts" -w /scripts \
    nvcr.io/nvidia/pytorch:24.10-py3 \
    python3 run_chanpred.py sanity $DURATION 2>&1 | tee "$OUT/chanpred.log" | tail -20
}

run_nrx() {
  log "=== nrx sanity ($DURATION s) — 기존 스크립트 재사용 ==="
  docker run --rm --user 0:0 --gpus "device=$GPU" --ipc=host --pid=host \
    -v "$SDK:/opt/nvidia/cuBB" \
    -v "$REPO:/workspace/AIRAN_Changjong" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    airan:25-3-final \
    python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DURATION 2>&1 | tee "$OUT/nrx.log" | tail -20
}

case "$WORKLOAD" in
  qwen_llm) run_qwen_llm ;;
  qwen_rag) run_qwen_rag ;;
  whisper)  run_whisper ;;
  qwen_vl)  run_qwen_vl ;;
  chanpred) run_chanpred ;;
  nrx)      run_nrx ;;
  all)
    run_nrx
    run_chanpred
    run_qwen_llm
    run_qwen_rag
    run_whisper
    run_qwen_vl
    ;;
  *) echo "unknown workload: $WORKLOAD"; exit 1;;
esac

log "=== all logs saved to $OUT/ ==="
