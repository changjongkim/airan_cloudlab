#!/usr/bin/env bash
# Quick setup — restore Aerial + cuPHY + HF cache when /mydata was reset.
#
# Use after recovery.sh diagnoses PARTIAL or WORST.
# Most operations run in parallel where possible.
#
# Total time: 60-90 min if NGC + HF have good bandwidth.

set -uo pipefail
SETUP_LOG=/tmp/recovery/setup_$(date +%H%M).log
mkdir -p /tmp/recovery
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$SETUP_LOG"; }

log "===== QUICK SETUP START ====="

# ----------------------------------------------------------------------------
# Step 1: /mydata directory structure
# ----------------------------------------------------------------------------
log ""
log "--- Step 1: /mydata structure ---"
sudo mkdir -p /mydata/{docker,hf_cache} 2>/dev/null
sudo chown -R sgkim:sgkim /mydata 2>/dev/null || true
mkdir -p /mnt/dockerdata/hf_cache 2>/dev/null || true
df -h /mydata /

# ----------------------------------------------------------------------------
# Step 2: Move docker root to /mydata (NVMe) — if not already
# ----------------------------------------------------------------------------
log ""
log "--- Step 2: Docker root on NVMe ---"
docker_root=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')
log "Current docker root: $docker_root"
if [[ "$docker_root" == "/var/lib/docker" ]]; then
  log "WARNING: docker root on small / partition. Aerial pull (35GB) may fail."
  log "Consider: edit /etc/docker/daemon.json data-root → /mydata/docker, then restart docker"
fi

# ----------------------------------------------------------------------------
# Step 3: Pull Aerial container in background (~30 min for 35GB)
# ----------------------------------------------------------------------------
log ""
log "--- Step 3: Aerial container pull (background) ---"
AERIAL_IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
if docker images | grep -q "aerial-cuda-accelerated-ran.*25-3-cubb"; then
  log "Aerial image already present"
else
  log "Starting docker pull $AERIAL_IMG (background)"
  nohup docker pull "$AERIAL_IMG" > /tmp/recovery/aerial_pull.log 2>&1 &
  echo $! > /tmp/recovery/aerial_pull.pid
  log "  pid: $(cat /tmp/recovery/aerial_pull.pid)"
fi

# ----------------------------------------------------------------------------
# Step 4: Clone Aerial repo to /mydata (if needed for cuPHY build)
# ----------------------------------------------------------------------------
log ""
log "--- Step 4: Clone Aerial repo ---"
AERIAL_DIR=/mydata/aerial-cuda-accelerated-ran
if [[ -d "$AERIAL_DIR" ]] && [[ -f "$AERIAL_DIR/CMakeLists.txt" ]]; then
  log "Aerial repo present at $AERIAL_DIR"
else
  log "Cloning aerial-cuda-accelerated-ran (sparse, no large LFS files)"
  cd /mydata && git clone --depth 1 --branch 25.3 \
    https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git 2>&1 | tail -5 | tee -a "$SETUP_LOG"
fi

# ----------------------------------------------------------------------------
# Step 5: Pre-download Qwen weights in background (~10-15 min for 14GB)
# ----------------------------------------------------------------------------
log ""
log "--- Step 5: Qwen-7B weights (background) ---"
HF_CACHE=/mnt/dockerdata/hf_cache
mkdir -p "$HF_CACHE"
if [[ -d "$HF_CACHE/hub/models--Qwen--Qwen2.5-7B" ]] || [[ -d "$HF_CACHE/models--Qwen--Qwen2.5-7B" ]]; then
  log "Qwen-7B weights present in $HF_CACHE"
else
  log "Pre-downloading Qwen2.5-7B (background, ~14GB)"
  pip install -q huggingface_hub 2>&1 | tail -2 | tee -a "$SETUP_LOG" || true
  nohup env HF_HOME="$HF_CACHE" huggingface-cli download Qwen/Qwen2.5-7B \
    > /tmp/recovery/qwen_dl.log 2>&1 &
  echo $! > /tmp/recovery/qwen_dl.pid
  log "  pid: $(cat /tmp/recovery/qwen_dl.pid)"
fi

# ----------------------------------------------------------------------------
# Step 6: Enable MIG (after docker pull starts so we don't wait)
# ----------------------------------------------------------------------------
log ""
log "--- Step 6: MIG enable ---"
mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0)
if [[ "$mig_state" != "Enabled" ]]; then
  log "Enabling MIG (driver_reset if needed)"
  bash "$HOME/cloudlab_aerial/driver_reset.sh" 2>&1 | tail -5 | tee -a "$SETUP_LOG"
  sudo nvidia-smi -i 0 -mig 1 2>&1 | tee -a "$SETUP_LOG"
  sleep 3
fi
nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader

# ----------------------------------------------------------------------------
# Step 7: Wait for Aerial pull
# ----------------------------------------------------------------------------
log ""
log "--- Step 7: Wait for Aerial pull ---"
if [[ -f /tmp/recovery/aerial_pull.pid ]]; then
  pid=$(cat /tmp/recovery/aerial_pull.pid)
  log "Waiting for Aerial pull pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    log "  still pulling... $(du -sh $(docker info | grep 'Docker Root' | awk '{print $4}') 2>/dev/null | awk '{print $1}') used"
  done
  log "Aerial pull complete"
  docker images | grep aerial | tee -a "$SETUP_LOG"
fi

# ----------------------------------------------------------------------------
# Step 8: Build airan:25-3 image (Dockerfile.airan adds PyTorch + transformers)
# ----------------------------------------------------------------------------
log ""
log "--- Step 8: Build airan:25-3 image ---"
if docker images | grep -q "airan:25-3"; then
  log "airan:25-3 already built"
else
  if [[ -f "$HOME/cloudlab_aerial/Dockerfile.airan" ]]; then
    log "Building airan:25-3 (PyTorch + transformers on Aerial base)"
    cd "$HOME/cloudlab_aerial"
    docker build -t airan:25-3-final -f Dockerfile.airan . 2>&1 | tail -10 | tee -a "$SETUP_LOG"
  else
    log "ERROR: Dockerfile.airan missing"
  fi
fi

# ----------------------------------------------------------------------------
# Step 9: cuPHY build (only if needed — most experiments work without it)
# ----------------------------------------------------------------------------
log ""
log "--- Step 9: cuPHY build (skip — Aerial image has pre-built libs) ---"
log "Note: real_l1.py uses component API from aerial.phy5g — pre-built in container"
log "Skip external cuPHY build unless explicitly needed"

# ----------------------------------------------------------------------------
# Step 10: Sanity test
# ----------------------------------------------------------------------------
log ""
log "--- Step 10: Sanity test (real_l1.py 1 run on split-50-50) ---"
cd "$HOME/cloudlab_aerial"
# Quick test: N=2 iterations on split-50-50 alone
nvidia-smi mig -i 0 -dci >/dev/null 2>&1 || true
nvidia-smi mig -i 0 -dgi >/dev/null 2>&1 || true
log "Running sanity sweep..."
N=2 PRESET=split-50-50 AI=none TAG=sanity DURATION=10 \
  bash ./run_n20.sh 2>&1 | tail -15 | tee -a "$SETUP_LOG"

sanity_json=$(ls $HOME/cloudlab_aerial/results/$(date +%Y%m%d)/n2_sanity/run_*.json 2>/dev/null | head -1)
if [[ -f "$sanity_json" ]]; then
  mean=$(python3 -c "import json; print(json.load(open('$sanity_json'))['mean_ms'])")
  log "Sanity OK: mean = ${mean} ms (expected ~46ms for 3g.20gb baseline)"
else
  log "ERROR: sanity failed (no JSON output). Check $SETUP_LOG and container logs."
fi

log ""
log "===== QUICK SETUP DONE ====="
log "Log: $SETUP_LOG"
log ""
log "Next: bash master_5h_sweep.sh (or run individual phases manually if low on time)"
