#!/usr/bin/env bash
# Sweep partitioning configurations and run L1+AI interference experiments.
#
# Two families of partitioning, both swept here for direct comparison:
#
#   HARDWARE MIG  (real isolation: SM + L2 + HBM bandwidth)
#     no-mig            full GPU, no isolation                  baseline
#     split-20-80       1g.5gb + 4g.20gb                        ~20:80
#     split-40-60       2g.10gb + 3g.20gb                       ~40:60
#     split-50-50       3g.20gb + 3g.20gb                       50:50
#     split-60-40       4g.20gb + 3g.20gb                       ~60:40
#
#   MPS  (soft SM partitioning only — Perlmutter style, no HBM isolation)
#     mps-10-90 / 20-80 / 30-70 / 40-60 / 50-50      (L1%-AI% thread percentage)
#
# AI workload python scripts in AIRAN_Changjong/experiments/ have Perlmutter-hardcoded paths:
#   - run_l1_graph.py     writes JSON to /pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results
#                         args: <label> <num_cells> <graph_mode 0/1>     iterations hardcoded
#   - run_gpt2_stress.py  HF cache /pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models
#                         args: <gpu_id> <duration_sec>
#   - run_hbm_stress.py   no Perlmutter paths
#                         args: <gpu_id> <duration_sec> [alloc_gb]
# We bind-mount writeable dirs over those Perlmutter paths so the scripts work unchanged.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/AIRAN_Changjong}"
IMAGE="${IMAGE:-airan:25-3-final}"
AERIAL_SDK="${AERIAL_SDK:-/mydata/aerial-cuda-accelerated-ran}"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
GPU="${GPU:-0}"
AI="${AI:-gpt2}"                           # gpt2 | hbm | neuralrx | none
CELLS="${CELLS:-20}"
DURATION="${DURATION:-60}"
GRAPH_MODE="${GRAPH_MODE:-1}"              # 1 = CUDA graph (matches Perlmutter results)

MIG_PRESETS="no-mig split-20-80 split-40-60 split-50-50 split-60-40"
MPS_PRESETS="mps-10-90 mps-20-80 mps-30-70 mps-40-60 mps-50-50"
PRESETS="${PRESETS:-$MIG_PRESETS $MPS_PRESETS}"

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$SCRIPT_DIR/results/$TS"
mkdir -p "$OUTDIR"

# Perlmutter-style paths the scripts write to; we bind-mount these in.
PERL_RESULTS=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results
PERL_HF=/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models
HOST_HF_CACHE="$HOME/.cache/hf_cloudlab"
mkdir -p "$HOST_HF_CACHE"

echo "results -> $OUTDIR"
[[ -d "$REPO_DIR" ]] || { echo "AIRAN_Changjong repo not found at $REPO_DIR"; exit 1; }

mig() { sudo "$SCRIPT_DIR/02_mig.sh" "$@"; }

# AI can be a comma-separated list of workload names; cycled across AI MIG instances.
# Single name keeps backward-compat (all AI containers use same workload).
# Supported: gpt2, hbm, resnet, neuralrx, none.
IFS=',' read -ra AI_LIST <<< "$AI"
echo "AI list: ${AI_LIST[*]}"

ai_script_for() {
  case "$1" in
    gpt2)     echo "experiments/run_gpt2_stress.py" ;;
    hbm)      echo "experiments/run_hbm_stress.py" ;;
    resnet)   echo "experiments/run_resnet_stress.py" ;;
    neuralrx) echo "experiments/run_neural_rx_stress.py" ;;
    none)     echo "" ;;
    *) echo "unknown AI workload: $1" >&2; return 1 ;;
  esac
}

ai_args_for() {
  case "$1" in
    gpt2)     echo "0 $DURATION" ;;
    hbm)      echo "0 $DURATION ${HBM_ALLOC:-1.0}" ;;
    resnet)   echo "0 $DURATION ${RESNET_BS:-16}" ;;
    neuralrx) echo "0 $DURATION" ;;
    none)     echo "" ;;
  esac
}

# Backward-compat: first item in list selects primary AI_SCRIPT for single-AI presets.
AI_SCRIPT=$(ai_script_for "${AI_LIST[0]}")
AI_ARGS_FN() { ai_args_for "${AI_LIST[0]}"; }

# Build the common -v / -e args for a container.
common_mounts() {
  local results_host="$1"
  echo "--user $HOST_UID:$HOST_GID"
  echo "-v $REPO_DIR:/workspace/AIRAN_Changjong"
  echo "-v $AERIAL_SDK:/opt/nvidia/cuBB"
  echo "-v $SCRIPT_DIR:/scripts"
  echo "-v $results_host:$PERL_RESULTS"
  echo "-v $results_host:/results_out"
  echo "-v $HOST_HF_CACHE:$PERL_HF"
  echo "-w /workspace/AIRAN_Changjong"
  echo "-e RESULTS_DIR=/results_out"
  echo "-e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/home/aerial/.local/lib/python3.10/site-packages"
  echo "-e LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:/opt/nvidia/cuBB/pyaerial/src/aerial/pycuphy:/opt/nvidia/cuBB/build.x86_64/cuPHY/src/cuphy:/opt/nvidia/cuBB/build.x86_64/cuPHY/util/aerial_utils:/usr/local/lib/python3.10/dist-packages/torch/lib"
  echo "-e LD_PRELOAD=/usr/local/cuda-12.9/lib64/libcupti.so.12"
  echo "-e HF_HOME=$PERL_HF"
  echo "-e HOME=/tmp"
  # L1 (real_l1.py) parameter overrides passed via env.
  [[ -n "${L1_NUM_PRBS:-}" ]]   && echo "-e NUM_PRBS=$L1_NUM_PRBS"
  [[ -n "${L1_NUM_RX_ANT:-}" ]] && echo "-e NUM_RX_ANT=$L1_NUM_RX_ANT"
  [[ -n "${L1_NUM_TX_ANT:-}" ]] && echo "-e NUM_TX_ANT=$L1_NUM_TX_ANT"
  [[ -n "${L1_MCS_INDEX:-}" ]]  && echo "-e MCS_INDEX=$L1_MCS_INDEX"
}

# --- MPS daemon helpers ----------------------------------------------------
MPS_PIPE_DIR=/tmp/nvidia-mps
MPS_LOG_DIR=/tmp/nvidia-log

start_mps() {
  echo "starting MPS daemon (host)"
  sudo mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
  echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sleep 1
  sudo CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" \
       nvidia-cuda-mps-control -d
  sleep 2
}
stop_mps() {
  echo "stopping MPS daemon"
  echo quit | sudo CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" nvidia-cuda-mps-control >/dev/null 2>&1 || true
}

cleanup_containers() {
  docker ps -q --filter ancestor="$IMAGE" | xargs -r docker rm -f >/dev/null 2>&1 || true
  docker ps -q --filter ancestor=airan:25-3 | xargs -r docker rm -f >/dev/null 2>&1 || true
  docker ps -q --filter ancestor=airan:25-3-build | xargs -r docker rm -f >/dev/null 2>&1 || true
  docker ps -q --filter ancestor=airan:25-3-final | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
  echo quit | sudo CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sudo killall -9 nvidia-cuda-mps-server nvidia-cuda-mps-control 2>/dev/null || true
  # Stale MPS pipe dirs cause CUDA error 805 on MIG instances. Remove them before MIG runs.
  sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log
  for i in 1 2 3 4 5; do
    if ! sudo fuser /dev/nvidia0 2>/dev/null | grep -q .; then break; fi
    sleep 1
  done
}

# --- core runner -----------------------------------------------------------
run_mig_preset() {
  local preset="$1" pdir="$2"
  mkdir -p "$pdir/results"

  cleanup_containers
  mig hard-reset

  local L1_DEV="" AI_DEVS=()
  local GPU_ARG_L1
  if [[ "$preset" == "no-mig" ]]; then
    GPU_ARG_L1="all"
    L1_DEV="all-GPU-0"; AI_DEVS=("all")
  else
    mig config "$preset"
    sleep 2
    mapfile -t UUIDS < <(nvidia-smi -L | awk '/MIG/ {gsub(/[()]/,""); for(i=1;i<=NF;i++) if($i ~ /^MIG-/) print $i}')
    if [[ ${#UUIDS[@]} -lt 2 ]]; then
      echo "expected >=2 MIG instances, got ${#UUIDS[@]}"; return 1
    fi
    # Per-preset L1 assignment:
    #   split-X-Y           : L1 = smallest (last)               — semantics "L1 gets X% (small)"
    #   four-way-bigL1      : L1 = largest (first)               — semantics "big L1, 3 small AIs"
    #   four-way-eq         : L1 = smallest (last = 1g)          — semantics "small L1, 3 equal AIs"
    #   seven-1g            : L1 = first 1g, rest AI             — all equal partitions
    local n=${#UUIDS[@]}
    case "$preset" in
      four-way-bigL1)
        L1_DEV="${UUIDS[0]}"
        AI_DEVS=("${UUIDS[@]:1}")
        ;;
      *)
        L1_DEV="${UUIDS[$((n-1))]}"
        AI_DEVS=("${UUIDS[@]:0:$((n-1))}")
        ;;
    esac
    GPU_ARG_L1="\"device=$L1_DEV\""
  fi
  nvidia-smi -L > "$pdir/mig.txt"
  echo "  L1 device: $L1_DEV"
  echo "  AI devices: ${AI_DEVS[*]}"

  local mounts; mounts=$(common_mounts "$pdir/results")

  # Launch one AI container per AI instance, cycling through AI_LIST.
  local AI_CIDS=()
  local AI_NAMES=()
  if [[ -n "$AI_SCRIPT" ]]; then
    local idx=0
    for d in "${AI_DEVS[@]}"; do
      local gpu_arg="\"device=$d\""
      [[ "$d" == "all" ]] && gpu_arg="all"
      local ai_name="${AI_LIST[$((idx % ${#AI_LIST[@]}))]}"
      local ai_script_path; ai_script_path=$(ai_script_for "$ai_name")
      local ai_args; ai_args=$(ai_args_for "$ai_name")
      cid=$(docker run -d --rm --gpus "$gpu_arg" \
        $mounts \
        "$IMAGE" \
        python3 "$ai_script_path" $ai_args)
      AI_CIDS+=("$cid")
      AI_NAMES+=("$ai_name")
      echo "  AI[$idx] $ai_name on $d -> $cid"
      idx=$((idx+1))
    done
    sleep 12
  fi

  echo "  running L1 -> $pdir/l1.log"
  docker run --rm --gpus "$GPU_ARG_L1" \
    $mounts \
    "$IMAGE" \
    python3 /scripts/real_l1.py "$preset" "$CELLS" "${L1_ITERATIONS:-50}" \
    > "$pdir/l1.log" 2>&1 || true

  for i in "${!AI_CIDS[@]}"; do
    docker logs "${AI_CIDS[$i]}" > "$pdir/ai_${i}_${AI_NAMES[$i]}.log" 2>&1 || true
    docker rm -f "${AI_CIDS[$i]}" >/dev/null 2>&1 || true
  done
}

run_mps_preset() {
  local preset="$1" pdir="$2"
  mkdir -p "$pdir/results"
  local rest="${preset#mps-}"
  local L1_PCT="${rest%%-*}"
  local AI_PCT="${rest##*-}"
  echo "  MPS percentages — L1: $L1_PCT%   AI: $AI_PCT%"

  cleanup_containers
  mig hard-reset
  start_mps

  local mounts; mounts=$(common_mounts "$pdir/results")

  local AI_CID=""
  if [[ -n "$AI_SCRIPT" ]]; then
    AI_CID=$(docker run -d --rm --gpus all --ipc=host \
      -v "$MPS_PIPE_DIR:$MPS_PIPE_DIR" -v "$MPS_LOG_DIR:$MPS_LOG_DIR" \
      -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" \
      -e CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" \
      -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$AI_PCT" \
      $mounts \
      "$IMAGE" \
      python3 "$AI_SCRIPT" $(AI_ARGS_FN))
    echo "  AI container: $AI_CID"
    sleep 8
  fi

  echo "  running L1 -> $pdir/l1.log"
  docker run --rm --gpus all --ipc=host \
    -v "$MPS_PIPE_DIR:$MPS_PIPE_DIR" -v "$MPS_LOG_DIR:$MPS_LOG_DIR" \
    -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" \
    -e CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR" \
    -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$L1_PCT" \
    $mounts \
    "$IMAGE" \
    python3 /scripts/synthetic_l1.py "$preset" "$CELLS" 100 \
    > "$pdir/l1.log" 2>&1 || true

  if [[ -n "$AI_CID" ]]; then
    docker logs "$AI_CID" > "$pdir/ai.log" 2>&1 || true
    docker rm -f "$AI_CID" >/dev/null 2>&1 || true
  fi

  stop_mps
}

run_one() {
  local preset="$1"
  local pdir="$OUTDIR/$preset"
  mkdir -p "$pdir"
  echo
  echo "============================================================"
  echo "config: $preset   AI: $AI   cells: $CELLS   graph_mode: $GRAPH_MODE"
  echo "============================================================"
  case "$preset" in
    mps-*)  run_mps_preset "$preset" "$pdir" ;;
    *)      run_mig_preset "$preset" "$pdir" ;;
  esac
  echo "  done: $preset"
}

trap 'stop_mps; mig hard-reset 2>/dev/null || true' EXIT

for p in $PRESETS; do
  run_one "$p" || echo "preset $p failed, continuing"
done

mig hard-reset

echo
echo "sweep complete. results: $OUTDIR"
ls -la "$OUTDIR"
