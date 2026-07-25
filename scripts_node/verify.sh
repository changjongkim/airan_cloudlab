#!/usr/bin/env bash
# Sanity check after 00_bootstrap.sh + reboot. Run as regular user (not root).
set -uo pipefail

pass=0; fail=0
check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf '  [OK]   %s\n' "$name"; pass=$((pass+1))
  else
    printf '  [FAIL] %s\n' "$name"; fail=$((fail+1))
  fi
}

echo "== node info =="
hostname; uname -r; lsb_release -d 2>/dev/null || true

echo "== driver / GPU =="
nvidia-smi -L 2>/dev/null || echo "  nvidia-smi not working — reboot needed?"
check "4x A100 visible"  bash -c "nvidia-smi -L | grep -c A100 | grep -q 4"
check "CUDA toolkit"     bash -c "nvcc --version"

echo "== container stack =="
check "docker daemon"    docker info
check "docker non-root"  bash -c "docker ps"
check "GPU in container" bash -c "docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L | grep -q A100"

echo "== networking (Aerial fronthaul prereq) =="
echo "  NICs:"
ip -br link | awk '{print "    " $0}'
echo "  Looking for ConnectX (Mellanox):"
lspci | grep -i mellanox || echo "    none — OTA mode will not be possible"

echo "== NGC =="
check "ngc CLI"          command -v ngc
check "ngc configured"   ngc config current

echo
echo "result: $pass passed, $fail failed"
[[ $fail -eq 0 ]] && echo "ready to run ./01_aerial.sh" || echo "fix failures above before running 01_aerial.sh"
