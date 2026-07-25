#!/usr/bin/env bash
# Chain 13 사전 준비: HuggingFace 모델 사전 다운로드
# 실행: bash prepull_models.sh
set -euo pipefail

HF_CACHE=${HF_CACHE:-/mydata/hf_cache}
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

echo "=== [$(date +%H:%M:%S)] cache dir: $HF_CACHE ==="
echo "=== disk before ==="
df -h /mydata | tail -1

# 모든 모델을 pytorch 컨테이너 안에서 다운로드 (huggingface_hub 없어도 됨)
docker run --rm --user 0:0 \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
  nvcr.io/nvidia/pytorch:24.10-py3 bash -c '
pip install -q --no-cache-dir "huggingface_hub[cli]"

echo "[Qwen2.5-7B-Instruct]"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir-use-symlinks False

echo "[Qwen2-VL-7B-Instruct]"
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir-use-symlinks False

echo "[Whisper-large-v3]"
huggingface-cli download openai/whisper-large-v3 --local-dir-use-symlinks False

echo "DONE"
'

echo "=== disk after ==="
df -h /mydata | tail -1
du -sh "$HF_CACHE"
