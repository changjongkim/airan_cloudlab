#!/usr/bin/env bash
# Perlmutter no-MIG 실험 — 환경 점검
# 실행: bash 00_env_check.sh 2>&1 | tee env_check.log

set -uo pipefail
echo "=== Perlmutter Environment Check ==="
date; echo "host: $(hostname)"; echo "user: $(whoami)"
echo

echo "--- 1) Node + GPU info ---"
nvidia-smi --query-gpu=index,name,driver_version,mig.mode.current --format=csv 2>&1 || echo "nvidia-smi not on login node (need salloc)"

echo
echo "--- 2) SLURM partitions ---"
sinfo -p gpu --Format=PartitionName,Available,Cpus,Gres 2>&1 | head -10
echo "QoS available:"
sacctmgr -P show qos format=name 2>&1 | head -20

echo
echo "--- 3) Container runtime ---"
echo "shifter: $(which shifter 2>/dev/null || echo MISSING)"
echo "podman-hpc: $(which podman-hpc 2>/dev/null || echo MISSING)"
echo "docker: $(which docker 2>/dev/null || echo MISSING)"
shifterimg images 2>&1 | grep -i aerial | head -5

echo
echo "--- 4) Modules (cuda, nsight) ---"
module avail cuda 2>&1 | head -10
module avail nsight 2>&1 | head -5

echo
echo "--- 5) Storage paths ---"
echo "SCRATCH=$SCRATCH (size: $(df -h $SCRATCH 2>/dev/null | tail -1 | awk '{print $4}'))"
echo "HOME=$HOME"
echo "CFS=${CFS:-not set}"

echo
echo "--- 6) Repo location ---"
find $SCRATCH $HOME -maxdepth 3 -name "airan_cloudlab" -type d 2>/dev/null | head -3

echo
echo "--- 7) Aerial repo (open source) ---"
find $SCRATCH $HOME -maxdepth 3 -name "aerial-cuda-accelerated-ran" -type d 2>/dev/null | head -3

echo
echo "=== Check complete ==="
