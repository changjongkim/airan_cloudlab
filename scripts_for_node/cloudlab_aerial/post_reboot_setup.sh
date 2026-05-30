#!/usr/bin/env bash
# Post-reboot fast setup — incorporates ALL 5/24 lessons learned.
#
# Use when:
#   1. Saved image is present (root only) → /mydata reset on new reservation
#   2. Need to restore: NVMe mount, docker root, containerd, Aerial repo, containers
#
# Total time: 45-60 min (parallelized)
#
# Usage:
#   bash post_reboot_setup.sh

set -uo pipefail
SETUP_LOG=/tmp/post_reboot_$(date +%H%M).log
mkdir -p /tmp/recovery
ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$SETUP_LOG"; }

log "===== POST-REBOOT SETUP START ====="

# ----------------------------------------------------------------------------
# Step 1: NVMe + /mydata mount (5/24 lesson: was empty)
# ----------------------------------------------------------------------------
log "Step 1: /mydata NVMe mount"
if ! mountpoint -q /mydata; then
  if [[ -b /dev/nvme0n1 ]]; then
    log "Formatting and mounting /dev/nvme0n1"
    sudo mkfs.ext4 -F /dev/nvme0n1 >/dev/null 2>&1 || true
    sudo mkdir -p /mydata
    sudo mount /dev/nvme0n1 /mydata
    grep -q '/mydata' /etc/fstab || \
      echo "/dev/nvme0n1 /mydata ext4 defaults 0 0" | sudo tee -a /etc/fstab
  else
    log "ERROR: /dev/nvme0n1 not found, check lsblk"
    exit 1
  fi
fi
sudo chown -R sgkim:airanslicing-PG0 /mydata
mkdir -p /mydata/{docker,hf_cache,containerd}
df -h /mydata

# ----------------------------------------------------------------------------
# Step 2: Docker + containerd both on /mydata (5/24 lesson)
# ----------------------------------------------------------------------------
log "Step 2: Docker data-root + containerd location"
sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
sleep 2

# Docker daemon.json
sudo mkdir -p /etc/docker
echo '{"data-root":"/mydata/docker"}' | sudo tee /etc/docker/daemon.json

# Containerd symlink (5/24 issue: extraction filled / partition)
if [[ ! -L /var/lib/containerd ]]; then
  sudo rm -rf /var/lib/containerd
  sudo ln -s /mydata/containerd /var/lib/containerd
fi

sudo systemctl start containerd && sleep 2 && sudo systemctl start docker && sleep 3
docker info 2>&1 | grep -E "Server Version|Docker Root|Snapshotter" | tee -a "$SETUP_LOG"

# ----------------------------------------------------------------------------
# Step 3: HF cache permissions (5/24 lesson: container UID 1000 can't write)
# ----------------------------------------------------------------------------
log "Step 3: HF cache permissions for container"
sudo chmod -R 777 /mydata/hf_cache

# ----------------------------------------------------------------------------
# Step 4: Aerial container pull (background, ~25-30 min)
# ----------------------------------------------------------------------------
log "Step 4: Aerial container pull (background)"
AERIAL_IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
if docker images | grep -q "aerial-cuda-accelerated-ran.*25-3-cubb"; then
  log "Aerial image already present"
else
  nohup docker pull "$AERIAL_IMG" > /tmp/recovery/aerial_pull.log 2>&1 &
  echo $! > /tmp/recovery/aerial_pull.pid
  log "Pull pid: $(cat /tmp/recovery/aerial_pull.pid)"
fi

# ----------------------------------------------------------------------------
# Step 5: Qwen weights download (background, ~10-15 min)
# ----------------------------------------------------------------------------
log "Step 5: Qwen-7B weights download (background)"
HF_CACHE=/mydata/hf_cache
if [[ -d "$HF_CACHE/hub/models--Qwen--Qwen2.5-7B/snapshots" ]] && \
   ls "$HF_CACHE/hub/models--Qwen--Qwen2.5-7B/snapshots"/*/model-00001*.safetensors 2>/dev/null >/dev/null; then
  log "Qwen-7B already present"
else
  python3 -m pip install --user --quiet huggingface_hub 2>&1 | tail -2 | tee -a "$SETUP_LOG"
  nohup env HF_HOME="$HF_CACHE" ~/.local/bin/hf download Qwen/Qwen2.5-7B \
    > /tmp/recovery/qwen_dl.log 2>&1 &
  echo $! > /tmp/recovery/qwen_dl.pid
  log "DL pid: $(cat /tmp/recovery/qwen_dl.pid)"
fi

# ----------------------------------------------------------------------------
# Step 6: Aerial repo clone (5/24 lesson: needed for pyaerial)
# ----------------------------------------------------------------------------
log "Step 6: Aerial repo clone with chown for container UID"
AERIAL_DIR=/mydata/aerial-cuda-accelerated-ran
if [[ -d "$AERIAL_DIR/.git" ]]; then
  log "Aerial repo present"
else
  cd /mydata && git clone --depth 1 --branch 25.3.2 \
    https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git 2>&1 | tail -3 | tee -a "$SETUP_LOG"
fi
sudo chown -R 1000:1000 "$AERIAL_DIR"
log "Aerial repo chowned to 1000:1000 (container UID)"

# ----------------------------------------------------------------------------
# Step 7: MIG enable on GPUs 0/1/2 (5/31 plan: GPU 3 stays no-MIG)
# ----------------------------------------------------------------------------
log "Step 7: MIG enable on GPUs"
nvidia-smi --query-gpu=index,mig.mode.current --format=csv | tee -a "$SETUP_LOG"
for g in 0 1 2; do
  sudo nvidia-smi -i $g -mig 1 2>&1 | tail -1 | tee -a "$SETUP_LOG"
done
log "Note: if 'pending enable', reboot required afterward"

# ----------------------------------------------------------------------------
# Step 8: Wait for Aerial pull, build airan:25-3-final
# ----------------------------------------------------------------------------
log "Step 8: Waiting for Aerial pull, then building airan:25-3-final"
if [[ -f /tmp/recovery/aerial_pull.pid ]]; then
  pid=$(cat /tmp/recovery/aerial_pull.pid)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    log "  Aerial pull still running... (/mydata: $(du -sh /mydata 2>/dev/null | awk '{print $1}'))"
  done
fi

if docker images | grep -q "airan:25-3-final"; then
  log "airan:25-3-final already built"
else
  log "Building airan:25-3-final from Dockerfile.airan"
  cd "$HOME/cloudlab_aerial"
  docker build -t airan:25-3-final -f Dockerfile.airan . 2>&1 | tail -10 | tee -a "$SETUP_LOG"
fi

# ----------------------------------------------------------------------------
# Step 9: Build pyaerial inside container (5/24 lesson: required for real_l1.py)
# ----------------------------------------------------------------------------
log "Step 9: Build pyaerial bindings inside container"
if docker run --rm --gpus all -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
     airan:25-3-final python3 -c "import aerial" 2>/dev/null; then
  log "pyaerial already importable"
else
  log "Building pyaerial (cmake + ninja, ~15-20 min)"
  docker run -d --gpus all --name aerial-build \
    -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
    -w /opt/nvidia/cuBB airan:25-3-final bash -c "
      set -e
      cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
        -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
      cmake --build build -t _pycuphy pycuphycpp -j16
      bash /opt/nvidia/cuBB/pyaerial/scripts/install_dev_pkg.sh
      python3 -c 'import aerial; print(\"OK\", aerial.__file__)'
    " 2>&1 | tee -a "$SETUP_LOG"
  # Wait for build
  while [[ "$(docker inspect -f '{{.State.Status}}' aerial-build 2>/dev/null)" == "running" ]]; do
    sleep 30
    log "  pyaerial build still running..."
  done
  log "Build container exit: $(docker inspect -f '{{.State.ExitCode}}' aerial-build)"
  # Commit pyaerial PYTHONPATH to image
  docker commit -c "ENV PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:\${PYTHONPATH}" \
    aerial-build airan:25-3-final
  docker rm -f aerial-build
fi

# Verify pyaerial
docker run --rm --gpus all -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  airan:25-3-final python3 -c \
  "import aerial; from aerial.phy5g.algorithms import ChannelEstimator; print('pyaerial OK:', aerial.__file__)" \
  2>&1 | tail -3 | tee -a "$SETUP_LOG"

# ----------------------------------------------------------------------------
# Step 10: Wait for Qwen download
# ----------------------------------------------------------------------------
log "Step 10: Wait for Qwen download to complete"
if [[ -f /tmp/recovery/qwen_dl.pid ]]; then
  pid=$(cat /tmp/recovery/qwen_dl.pid)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    log "  Qwen DL ongoing... ($(du -sh /mydata/hf_cache 2>/dev/null | awk '{print $1}'))"
  done
fi
log "Qwen DL final size: $(du -sh /mydata/hf_cache 2>/dev/null)"

# ----------------------------------------------------------------------------
# Step 11: Sanity test
# ----------------------------------------------------------------------------
log "Step 11: Sanity test (N=2 on split-50-50 if MIG)"
sudo nvidia-smi -i 0 -mig 1 2>&1 | tail -1 || true
sleep 2
mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0)
if [[ "$mig_state" == "Enabled" ]]; then
  cd "$HOME/cloudlab_aerial"
  N=2 PRESET=split-50-50 AI=none TAG=sanity_post_reboot DURATION=10 \
    bash ./run_n20.sh 2>&1 | tail -10 | tee -a "$SETUP_LOG"
  log "Sanity output check above"
else
  log "MIG not enabled — sanity test skipped. Reboot may be required."
fi

log ""
log "===== POST-REBOOT SETUP DONE ====="
log "Log: $SETUP_LOG"
log ""
log "Next steps:"
log "  1. Verify MIG state: nvidia-smi --query-gpu=mig.mode.current --format=csv"
log "  2. Disable MIG on GPU 3 (for no-MIG baseline): sudo nvidia-smi -i 3 -mig 0; reboot if pending"
log "  3. Run experiments: phase1_sweep.sh / nsys_profile_runner.sh / ai_throughput_v2.sh"
