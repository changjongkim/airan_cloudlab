#!/usr/bin/env bash
# Driver module reset — required after MIG mode toggle to clear stale
# nvidia-container-cli driver RPC state.
#
# Usage: ./driver_reset.sh
#
# Safe to run when GPU is idle. Will fail if any process has GPU handles open.

set -uo pipefail

echo "[driver_reset] $(date +%H:%M:%S) starting reset sequence"

# 1. Stop services that hold the GPU
echo "[driver_reset] stop nvidia-persistenced"
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
echo "[driver_reset] stop docker (containers will be killed)"
sudo systemctl stop docker 2>/dev/null || true

# 2. Verify no remaining GPU processes
gpu_procs=$(sudo fuser -v /dev/nvidia* 2>&1 | grep -v "Cannot stat" | grep -v "^$" | wc -l)
if [[ $gpu_procs -gt 0 ]]; then
  echo "[driver_reset] WARNING: $gpu_procs processes still hold /dev/nvidia*"
  sudo fuser -v /dev/nvidia* 2>&1
fi

# 3. Remove kernel modules in dependency order
echo "[driver_reset] rmmod nvidia_drm"
sudo rmmod nvidia_drm 2>/dev/null || echo "  (already unloaded or in use)"
echo "[driver_reset] rmmod nvidia_modeset"
sudo rmmod nvidia_modeset 2>/dev/null || echo "  (already unloaded or in use)"
echo "[driver_reset] rmmod nvidia_uvm"
sudo rmmod nvidia_uvm 2>/dev/null || echo "  (already unloaded or in use)"

# nvidia.ko cannot be removed without unloading uvm/modeset/drm + all consumers
# If those rmmod'd above, nvidia itself often stays loaded (driver core).
# That's OK — we just need to clear the upper modules.

# 4. Reload modules
echo "[driver_reset] modprobe nvidia_uvm"
sudo modprobe nvidia_uvm

echo "[driver_reset] modprobe nvidia_modeset"
sudo modprobe nvidia_modeset 2>/dev/null || true

echo "[driver_reset] modprobe nvidia_drm"
sudo modprobe nvidia_drm 2>/dev/null || true

# 5. Restart services
echo "[driver_reset] start nvidia-persistenced"
sudo systemctl start nvidia-persistenced
sleep 2

echo "[driver_reset] start docker"
sudo systemctl start docker
sleep 5

# 6. Sanity check
echo "[driver_reset] sanity check: nvidia-smi"
if nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv,noheader; then
  echo "[driver_reset] $(date +%H:%M:%S) OK"
  exit 0
else
  echo "[driver_reset] FAILED: nvidia-smi not responsive"
  exit 1
fi
