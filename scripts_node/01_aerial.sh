#!/usr/bin/env bash
# Clone NVIDIA Aerial (open-source) and run its dev container.
# Repo: https://github.com/NVIDIA/aerial-cuda-accelerated-ran  (Apache 2.0)
#
# The repo is open source, but its dev container is hosted on NGC at:
#   nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:26-1-cubb
# Some NGC images are public (pull works with REGISTRY READ); others are gated.
# Strategy: try pulling unauthenticated first; fall back to NGC login if it fails.
#
# Run AFTER 00_bootstrap.sh + reboot. As regular user (not root); script uses sudo where needed.

set -euo pipefail

WORKDIR="${WORKDIR:-/mydata}"
[[ -w "$WORKDIR" ]] || WORKDIR="$HOME"
REPO_DIR="$WORKDIR/aerial-cuda-accelerated-ran"
# Match Perlmutter (25-3-cubb) for direct result comparability.
# Override with: AERIAL_IMAGE=...:26-1-cubb ./01_aerial.sh
AERIAL_IMAGE="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"

log() { printf '\n=== %s ===\n' "$*"; }

log "prereq check"
command -v docker     >/dev/null || { echo "docker missing — run 00_bootstrap.sh first"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing — reboot after 00_bootstrap.sh"; exit 1; }
command -v git        >/dev/null || sudo apt-get install -y git
nvidia-smi -L

log "git-lfs"
if ! command -v git-lfs >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y git-lfs
  git lfs install
fi

log "clone Aerial repo into $REPO_DIR"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --recurse-submodules https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git "$REPO_DIR"
  ( cd "$REPO_DIR" && git lfs pull )
else
  echo "repo already exists, pulling latest"
  ( cd "$REPO_DIR" && git pull && git submodule update --init --recursive && git lfs pull )
fi

log "try pulling Aerial container (public attempt)"
if docker pull "$AERIAL_IMAGE"; then
  echo "image pulled without authentication."
else
  echo
  echo "public pull failed — image is gated. Logging into NGC."
  if ! command -v ngc >/dev/null 2>&1; then
    echo "ngc CLI missing — re-run 00_bootstrap.sh"; exit 1
  fi
  if ! ngc config current >/dev/null 2>&1; then
    echo "Configure NGC first:"
    echo "  ngc config set    # paste API key from https://ngc.nvidia.com/setup/api-key"
    exit 1
  fi
  NGC_KEY=$(ngc config current 2>/dev/null | awk -F= '/apikey/ {gsub(/ /,"",$2); print $2}')
  if [[ -z "$NGC_KEY" ]]; then
    echo "could not read NGC API key from 'ngc config current'"; exit 1
  fi
  echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
  docker pull "$AERIAL_IMAGE"
fi

log "smoke-test image — list GPUs from inside container"
docker run --rm --gpus all "$AERIAL_IMAGE" nvidia-smi -L || true

log "next: launch the dev container"
cat <<EOF
Image ready: $AERIAL_IMAGE
Repo cloned: $REPO_DIR

To launch the Aerial development container (interactive shell):
    cd $REPO_DIR
    ./cuPHY-CP/container/run_aerial.sh

Inside the container, build the SDK with:
    ./testBenches/phase4_test_scripts/build_aerial_sdk.sh

Run-mode reminder:
  * RFsim / test-vector  -> recommended on CloudLab (no real RU/PTP)
  * OTA                  -> needs ConnectX-6 Dx + PTP grandmaster (likely not feasible here)

Docs: https://docs.nvidia.com/aerial/cuda-accelerated-ran/
EOF
