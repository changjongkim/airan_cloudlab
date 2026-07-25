#!/usr/bin/env bash
# Bootstrap a CloudLab d8545 node (Ubuntu 22.04, A100 x4) for NVIDIA Aerial.
# Idempotent — safe to re-run. Tested for: driver 550 + CUDA 12.4 + Docker + nvidia-container-toolkit.
#
# Usage on the CloudLab node:
#   scp 00_bootstrap.sh <user>@<node>.cloudlab.us:~/
#   ssh <user>@<node>.cloudlab.us
#   chmod +x 00_bootstrap.sh && sudo ./00_bootstrap.sh 2>&1 | tee bootstrap.log
#   sudo reboot          # required after driver install
#   # after reboot: nvidia-smi  -> should list 4x A100

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log()  { printf '\n=== %s ===\n' "$*"; }
need_root() { [[ $EUID -eq 0 ]] || { echo "run as root (sudo)"; exit 1; }; }
need_root

log "1/7 apt update + base packages"
apt-get update -y
apt-get install -y --no-install-recommends --allow-change-held-packages \
  build-essential dkms ca-certificates curl wget gnupg lsb-release \
  pkg-config software-properties-common \
  linux-headers-$(uname -r) \
  git git-lfs tmux htop jq unzip

log "2/7 mount /mydata extra space (CloudLab convention)"
if [[ -d /mydata ]] && ! mountpoint -q /mydata; then
  echo "/mydata exists but not mounted — CloudLab usually auto-mounts; skipping"
fi
# Use /mydata if present, else /tmp for big downloads
WORKDIR=/mydata
[[ -w /mydata ]] || WORKDIR=/var/tmp
echo "WORKDIR=$WORKDIR"

log "3/7 NVIDIA driver (570-server) via cuda-keyring — required by cuPHY 25-3 ABI"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  cd "$WORKDIR"
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -y
  apt-get install -y nvidia-driver-570-server nvidia-utils-570-server
  echo "Driver 570 installed — REBOOT REQUIRED before nvidia-smi works."
else
  CURR_DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "nvidia-smi already present: driver $CURR_DRV"
  if [[ "$CURR_DRV" == 5[0-6]* ]]; then
    echo "Driver $CURR_DRV < 570 — upgrading to 570-server"
    apt-get update -y
    apt-get install -y --allow-change-held-packages nvidia-driver-570-server nvidia-utils-570-server
    echo "Driver 570 installed — REBOOT REQUIRED."
  fi
fi

log "4/7 CUDA toolkit 12.8 (matches Aerial 25-3 requirement)"
if ! command -v nvcc >/dev/null 2>&1 || [[ ! -d /usr/local/cuda-12.8 ]]; then
  apt-get install -y cuda-toolkit-12-8
  ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
  cat >/etc/profile.d/cuda.sh <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
EOF
fi

log "5/7 Docker CE"
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
  # let the original cloudlab user run docker without sudo
  if [[ -n "${SUDO_USER:-}" ]]; then usermod -aG docker "$SUDO_USER"; fi
fi

log "6/7 nvidia-container-toolkit"
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

log "7/7 NGC CLI"
if ! command -v ngc >/dev/null 2>&1; then
  cd "$WORKDIR"
  wget -q -O ngccli.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip
  unzip -o ngccli.zip -d /opt/ngc
  ln -sf /opt/ngc/ngc-cli/ngc /usr/local/bin/ngc
fi

log "DONE. Next steps:"
cat <<'EOF'
  1) sudo reboot   (required if driver was newly installed)
  2) After reboot: nvidia-smi      -> verify 4x A100
  3) docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  4) ngc config set                -> paste API key from https://ngc.nvidia.com/setup/api-key
  5) Run ./01_aerial.sh
EOF
