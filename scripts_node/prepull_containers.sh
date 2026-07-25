#!/usr/bin/env bash
# Chain 13 사전 준비: 컨테이너 이미지 pull
# 실행: bash prepull_containers.sh
set -euo pipefail

echo "=== [$(date +%H:%M:%S)] pulling vllm container ==="
docker pull vllm/vllm-openai:v0.6.6

echo "=== [$(date +%H:%M:%S)] pulling nvcr pytorch container (Whisper/VLM) ==="
docker pull nvcr.io/nvidia/pytorch:24.10-py3

echo "=== [$(date +%H:%M:%S)] verifying existing airan:25-3-final ==="
docker image inspect airan:25-3-final >/dev/null && echo "airan:25-3-final OK" || {
  echo "ERROR: airan:25-3-final not found. Rebuild required."
  exit 1
}

echo "=== [$(date +%H:%M:%S)] disk usage ==="
docker system df

echo "=== DONE ==="
