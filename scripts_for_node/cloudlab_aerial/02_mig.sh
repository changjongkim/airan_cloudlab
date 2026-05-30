#!/usr/bin/env bash
# Hardware-MIG configurator for A100 40GB on CloudLab d8545.
#
# IMPORTANT: A100 MIG only supports DISCRETE profile combinations, not arbitrary ratios.
# You CANNOT make an exact 10:90 or 30:70 split. The closest 2-instance configs are:
#
#   profile combo            L1 slices : AI slices    L1 HBM : AI HBM    "approx ratio"
#   --------------------------------------------------------------------------------
#   1g.5gb  +  4g.20gb       1 : 4   (2 slices wasted)    5GB : 20GB        ~20:80
#   2g.10gb +  3g.20gb       2 : 3   (2 slices wasted)   10GB : 20GB        ~40:60 (small L1)
#   3g.20gb +  3g.20gb       3 : 3   (1 slice  wasted)   20GB : 20GB        50:50
#   3g.20gb +  4g.20gb       3 : 4   (full GPU used)     20GB : 20GB        ~40:60
#   4g.20gb +  3g.20gb       4 : 3   (full GPU used)     20GB : 20GB        ~60:40
#
# For 4-way (L1, AI1, AI2, AI3) the cleanest options are:
#   4g.20gb + 1g.5gb x3      4 : 1 : 1 : 1  (full GPU used)
#   3g.20gb + 1g.5gb x3      3 : 1 : 1 : 1  (1 slice wasted)
#   2g.10gb + 2g.10gb + 2g.10gb + 1g.5gb   2 : 2 : 2 : 1   (full GPU used)
#
# Each reconfiguration kills CUDA contexts on that GPU. Stop containers/processes first.
# Profile IDs (A100 40GB):  0=7g.40gb  5=4g.20gb  9=3g.20gb  14=2g.10gb  19=1g.5gb

set -uo pipefail
GPU="${GPU:-0}"
need_root() { [[ $EUID -eq 0 ]] || { echo "run as root (sudo)"; exit 1; }; }

usage() {
  sed -n '2,40p' "$0"
  cat <<EOF

Commands:
  enable                          turn MIG mode ON for GPU \$GPU (default 0)
  disable                         turn MIG mode OFF
  reset                           destroy all instances on GPU \$GPU (keeps MIG mode on)
  hard-reset                      reset + disable MIG mode (full clean state)
  status                          show MIG mode + instance list + UUIDs

  config <preset>                 apply a named preset (resets first)
    presets:
      no-mig            disable MIG, full GPU shared (use MPS for soft split)
      split-20-80       1g.5gb + 4g.20gb         (closest to 20:80)
      split-40-60       2g.10gb + 3g.20gb        (small-L1 variant)
      split-50-50       3g.20gb x 2
      split-60-40       4g.20gb + 3g.20gb        (full GPU, larger L1)
      four-way-eq       2g + 2g + 2g + 1g        (3 AI workloads + small L1)
      four-way-bigL1    4g + 1g + 1g + 1g        (1 large L1 + 3 small AI)
      seven-1g          1g.5gb x 7               (max partitioning)

  uuids                           print MIG instance UUIDs (for CUDA_VISIBLE_DEVICES)

Env: GPU=<index>   (default 0)
EOF
  exit 1
}

show_uuids() {
  nvidia-smi -L | awk '/MIG/ {gsub(/[()]/,""); print}'
}

reset_gpu() {
  nvidia-smi mig -i "$GPU" -dci  >/dev/null 2>&1 || true
  nvidia-smi mig -i "$GPU" -dgi  >/dev/null 2>&1 || true
}

ensure_mig_on() {
  local mode
  mode=$(nvidia-smi -i "$GPU" --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null)
  if [[ "$mode" == "Enabled" ]]; then
    return 0
  fi
  echo "enabling MIG mode on GPU $GPU"
  # Retry loop — CUDA contexts may need time to fully release after prior workloads.
  for i in 1 2 3 4 5; do
    if nvidia-smi -i "$GPU" -mig 1 2>&1 | grep -q "All done"; then
      mode=$(nvidia-smi -i "$GPU" --query-gpu=mig.mode.current --format=csv,noheader)
      if [[ "$mode" == "Enabled" ]]; then return 0; fi
    fi
    echo "  attempt $i failed (mode=$mode); waiting + retry"
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
    echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
    sudo killall -9 nvidia-cuda-mps-server nvidia-cuda-mps-control 2>/dev/null || true
    sleep 3
  done
  echo "FAILED to enable MIG after 5 attempts."
  exit 1
}

cmd="${1:-}"; [[ -z "$cmd" ]] && usage

case "$cmd" in
  enable)
    need_root; ensure_mig_on
    nvidia-smi -i "$GPU" --query-gpu=mig.mode.current --format=csv,noheader
    ;;

  disable)
    need_root; reset_gpu
    nvidia-smi -i "$GPU" -mig 0
    ;;

  reset)
    need_root; reset_gpu
    echo "instances destroyed on GPU $GPU (MIG mode still on)"
    ;;

  hard-reset)
    need_root; reset_gpu
    nvidia-smi -i "$GPU" -mig 0
    echo "GPU $GPU: instances destroyed + MIG disabled"
    ;;

  status)
    nvidia-smi --query-gpu=index,mig.mode.current --format=csv
    echo "---"; nvidia-smi mig -lgi 2>/dev/null || true
    echo "---"; nvidia-smi -L
    ;;

  uuids)
    show_uuids
    ;;

  config)
    need_root
    preset="${2:-}"; [[ -z "$preset" ]] && usage

    if [[ "$preset" == "no-mig" ]]; then
      reset_gpu
      nvidia-smi -i "$GPU" -mig 0
      echo "GPU $GPU: MIG disabled, full GPU available"
      exit 0
    fi

    ensure_mig_on
    reset_gpu

    case "$preset" in
      split-20-80)    spec="19,5"   ;;   # 1g + 4g
      split-40-60)    spec="14,9"   ;;   # 2g + 3g
      split-50-50)    spec="9,9"    ;;   # 3g + 3g
      split-60-40)    spec="5,9"    ;;   # 4g + 3g
      four-way-eq)    spec="14,14,14,19" ;;   # 2g+2g+2g+1g
      four-way-bigL1) spec="5,19,19,19"  ;;   # 4g+1g+1g+1g
      seven-1g)       spec="19,19,19,19,19,19,19" ;;
      *) echo "unknown preset: $preset"; usage ;;
    esac

    echo "applying preset '$preset' on GPU $GPU (profile spec: $spec)"
    nvidia-smi mig -i "$GPU" -cgi "$spec" -C
    echo "---"
    show_uuids
    ;;

  *) usage ;;
esac
