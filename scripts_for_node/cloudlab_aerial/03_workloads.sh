#!/usr/bin/env bash
# Build the airan:25-3 image (Aerial base + PyTorch + transformers).
# Idempotent — does nothing if image already exists.
# Run AFTER 01_aerial.sh (which pulls the Aerial base).

set -euo pipefail
log() { printf '\n=== %s ===\n' "$*"; }

IMAGE_TAG="${IMAGE_TAG:-airan:25-3}"
DOCKERFILE="$(dirname "$0")/Dockerfile.airan"

[[ -f "$DOCKERFILE" ]] || { echo "missing $DOCKERFILE"; exit 1; }

if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  log "image $IMAGE_TAG already built — skipping. (delete with: docker rmi $IMAGE_TAG)"
else
  log "building $IMAGE_TAG (~5-15 min, depends on network)"
  docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" "$(dirname "$0")"
fi

log "smoke test: torch + GPU"
docker run --rm --gpus all "$IMAGE_TAG" python3 -c "
import torch, transformers, torchvision
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('cuda devices:', torch.cuda.device_count())
print('transformers:', transformers.__version__)
print('torchvision:', torchvision.__version__)
"

log "ready. Use IMAGE=$IMAGE_TAG in run_experiment.sh"
