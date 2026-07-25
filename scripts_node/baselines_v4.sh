#!/usr/bin/env bash
# B2/B4 baselines: L1 alone on different partition sizes
# Determines pure partition-size effect (no AI contention)
set -uo pipefail
cd "$HOME/cloudlab_aerial"
SCRIPT_DIR="$HOME/cloudlab_aerial"
GPU=0
mark() { printf '\n========== %s : %s ==========\n' "$(date +%H:%M:%S)" "$*"; }

# Re-enable MIG (after v3 disabled it for A)
state=$(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader)
echo "MIG mode = $state"
if [[ "$state" != "Enabled" ]]; then
  sudo systemctl stop nvidia-persistenced 2>&1 || true
  sudo rmmod nvidia_drm 2>&1 || true
  sudo rmmod nvidia_modeset 2>&1 || true
  sudo rmmod nvidia_uvm 2>&1 || true
  sudo nvidia-smi -i $GPU -mig 1
  sudo modprobe nvidia_uvm
  sudo systemctl start nvidia-persistenced
  sudo systemctl restart docker
  sleep 5
fi
nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader

run_l1_solo() {
  local label=$1
  local profile=$2  # cgi profile id (e.g., 14=2g.10gb, 9=3g.20gb, 5=4g.20gb)
  mark "$label : L1 alone on profile $profile"
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -cgi $profile -C
  sleep 2
  UUID=$(nvidia-smi -L | awk '/MIG/ {for(i=1;i<=NF;i++) if($i ~ /^MIG-/) {gsub(/[()]/,"",$i); print $i}}' | head -1)
  echo "  L1 UUID=$UUID"
  outdir="$SCRIPT_DIR/results/v4_${label}_$(date +%H%M%S)"
  mkdir -p "$outdir/results"
  docker run --rm --gpus "\"device=$UUID\"" --user $(id -u):$(id -g) \
    -v /users/sgkim/AIRAN_Changjong:/workspace/AIRAN_Changjong \
    -v /mydata/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$outdir/results:/results_out" \
    -w /workspace/AIRAN_Changjong \
    -e RESULTS_DIR=/results_out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/home/aerial/.local/lib/python3.10/site-packages \
    -e LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:/opt/nvidia/cuBB/pyaerial/src/aerial/pycuphy:/opt/nvidia/cuBB/build.x86_64/cuPHY/src/cuphy:/opt/nvidia/cuBB/build.x86_64/cuPHY/util/aerial_utils:/usr/local/lib/python3.10/dist-packages/torch/lib \
    -e LD_PRELOAD=/usr/local/cuda-12.9/lib64/libcupti.so.12 \
    -e HOME=/tmp \
    -e NUM_RX_ANT=8 -e NUM_TX_ANT=8 \
    airan:25-3-final \
    python3 /scripts/real_l1.py "$label" 20 50 > "$outdir/l1.log" 2>&1
  json=$(ls "$outdir/results"/*.json 2>/dev/null | head -1)
  if [ -n "$json" ]; then
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(f'  RESULT: mean={d[\"mean_ms\"]:.2f} p99={d[\"p99_ms\"]:.2f}')" "$json"
  else
    echo "  FAIL"
    tail -3 "$outdir/l1.log"
  fi
}

run_l1_solo "B2_2g10gb_alone" 14
run_l1_solo "B4_4g20gb_alone" 5

mark "v4 DONE"
