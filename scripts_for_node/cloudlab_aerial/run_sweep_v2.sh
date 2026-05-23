#!/usr/bin/env bash
# run_sweep_v2.sh — robust per-preset sweep, NEVER toggles MIG mode mid-sweep.
#
# Key invariant: MIG mode is enabled ONCE at sweep start, and only DESTROYED+RECREATED
# between presets. mig mode is disabled only at the very end (or by master orchestrator).
#
# This is the workaround for the kernel-level "in use by another client" lock that
# follows any CUDA workload — toggling MIG off→on after CUDA fails reliably.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/AIRAN_Changjong}"
IMAGE="${IMAGE:-airan:25-3-final}"
AERIAL_SDK="${AERIAL_SDK:-/mydata/aerial-cuda-accelerated-ran}"
HOST_UID=$(id -u); HOST_GID=$(id -g)
GPU="${GPU:-0}"
AI="${AI:-gpt2}"
CELLS="${CELLS:-20}"
DURATION="${DURATION:-90}"

PRESETS="${PRESETS:-split-50-50}"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="$SCRIPT_DIR/results/$DATE_DIR/$TS"
mkdir -p "$OUTDIR"
PERL_RESULTS=/pscratch/sd/s/sgkim/kcj/AI-RAN/experiments/results
PERL_HF=/pscratch/sd/s/sgkim/kcj/AI-RAN/datasets/models
HOST_HF_CACHE="$HOME/.cache/hf_cloudlab"
mkdir -p "$HOST_HF_CACHE"

# AI workload table
IFS=',' read -ra AI_LIST <<< "$AI"
ai_script_for() {
  case "$1" in
    gpt2)           echo "experiments/run_gpt2_stress.py" ;;
    hbm)            echo "experiments/run_hbm_stress.py" ;;
    hbm_2g)         echo "experiments/run_hbm_stress.py" ;;    # 2g.10gb cap-safe (alloc=6GB)
    hbm_1g)         echo "experiments/run_hbm_stress.py" ;;    # 1g.5gb cap-safe (alloc=3GB)
    resnet)         echo "experiments/run_resnet_stress.py" ;;
    qwen7b)         echo "experiments/run_qwen7b_stress.py" ;;
    qwen7b_prefill) echo "experiments/run_qwen7b_prefill.py" ;;
    qwen7b_decode)  echo "experiments/run_qwen7b_decode.py" ;;
    qwen_small)     echo "experiments/run_qwen_small_stress.py" ;;   # Qwen2.5-1.5B (~3.6GB), 2g.10gb fit
    neuralrx)       echo "experiments/run_neural_rx_stress.py" ;;
    none)           echo "" ;;
  esac
}
ai_args_for() {
  case "$1" in
    gpt2)           echo "0 $DURATION" ;;
    hbm)            echo "0 $DURATION ${HBM_ALLOC:-1.0}" ;;
    hbm_2g)         echo "0 $DURATION ${HBM_ALLOC_2G:-6.0}" ;;   # 6GB for 2g.10gb cap
    hbm_1g)         echo "0 $DURATION ${HBM_ALLOC_1G:-3.0}" ;;   # 3GB for 1g.5gb cap
    resnet)         echo "0 $DURATION ${RESNET_BS:-16}" ;;
    qwen7b)         echo "0 $DURATION" ;;
    qwen7b_prefill) echo "0 $DURATION" ;;
    qwen7b_decode)  echo "0 $DURATION" ;;
    qwen_small)     echo "0 $DURATION" ;;
    neuralrx)       echo "0 $DURATION" ;;
    none)           echo "" ;;
  esac
}

common_mounts() {
  local results_host="$1"
  echo "--user $HOST_UID:$HOST_GID"
  echo "-v $REPO_DIR:/workspace/AIRAN_Changjong"
  echo "-v $AERIAL_SDK:/opt/nvidia/cuBB"
  echo "-v $SCRIPT_DIR:/scripts"
  echo "-v $results_host:$PERL_RESULTS"
  echo "-v $results_host:/results_out"
  echo "-v $HOST_HF_CACHE:$PERL_HF"
  echo "-v /mnt/dockerdata/hf_cache:/pscratch/sd/s/sgkim/kcj/hf_cache"
  echo "-w /workspace/AIRAN_Changjong"
  echo "-e RESULTS_DIR=/results_out"
  echo "-e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src:/home/aerial/.local/lib/python3.10/site-packages"
  echo "-e LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:/opt/nvidia/cuBB/pyaerial/src/aerial/pycuphy:/opt/nvidia/cuBB/build.x86_64/cuPHY/src/cuphy:/opt/nvidia/cuBB/build.x86_64/cuPHY/util/aerial_utils:/usr/local/lib/python3.10/dist-packages/torch/lib"
  echo "-e LD_PRELOAD=/usr/local/cuda-12.9/lib64/libcupti.so.12"
  echo "-e HF_HOME=$PERL_HF"
  echo "-e HOME=/tmp"
  [[ -n "${L1_NUM_PRBS:-}" ]]   && echo "-e NUM_PRBS=$L1_NUM_PRBS"
  [[ -n "${L1_NUM_RX_ANT:-}" ]] && echo "-e NUM_RX_ANT=$L1_NUM_RX_ANT"
  [[ -n "${L1_NUM_TX_ANT:-}" ]] && echo "-e NUM_TX_ANT=$L1_NUM_TX_ANT"
  [[ -n "${L1_MCS_INDEX:-}" ]]  && echo "-e MCS_INDEX=$L1_MCS_INDEX"
}

cleanup_workload_state() {
  # Kill any leftover containers (but DO NOT disturb MIG mode).
  docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sudo killall -9 nvidia-cuda-mps-server nvidia-cuda-mps-control 2>/dev/null || true
  sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log
  sleep 2
}

create_mig_instances() {
  # Destroy existing instances and create new ones. MIG mode stays ON.
  # MIG profile IDs on A100 40GB:
  #   0=7g.40gb  5=4g.20gb  9=3g.20gb  14=2g.10gb  19=1g.5gb
  local preset="$1"
  sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
  local spec
  case "$preset" in
    # 2-partition presets (existing)
    split-20-80)    spec="19,5"   ;;    # 1g + 4g
    split-40-60)    spec="14,9"   ;;    # 2g + 3g
    split-50-50)    spec="9,9"    ;;    # 3g + 3g (symmetric)
    split-60-40)    spec="5,9"    ;;    # 4g + 3g (asymmetric, bimodal)
    # 3+ partition presets (existing)
    four-way-eq)    spec="14,14,14,19" ;;   # 3× 2g + 1g
    four-way-bigL1) spec="5,19,19,19"  ;;   # 4g + 3× 1g
    seven-1g)       spec="19,19,19,19,19,19,19" ;;
    # NEW multi-AI partition presets (5/24 onwards)
    3way-balanced)  spec="9,14,14"    ;;    # 3g + 2g + 2g  → L1=3g, AI×2 on 2g
    3way-L1small)   spec="9,14,14"    ;;    # 3g + 2g + 2g  → L1=2g, AI on 3g + 2g
    3way-asym)      spec="5,14,19"    ;;    # 4g + 2g + 1g  → L1=4g, AI on 2g + 1g
    4way-1L1+3AI)   spec="5,19,19,19" ;;    # 4g + 3× 1g    → L1=4g, AI×3 on 1g (same alloc as four-way-bigL1)
    *) echo "unknown preset: $preset"; return 1 ;;
  esac
  sudo nvidia-smi mig -i "$GPU" -cgi "$spec" -C >/dev/null 2>&1
}

get_mig_uuids() {
  nvidia-smi -L | awk '/MIG/ {for(i=1;i<=NF;i++) if($i ~ /^MIG-/) {gsub(/[()]/,"",$i); print $i}}'
}

# Returns UUIDs for instances matching a given profile name (e.g., "3g.20gb").
# Used for explicit L1/AI assignment in multi-partition presets where positional
# UUID indexing is ambiguous (instances are listed in GPC order, not spec order).
uuids_for_profile() {
  local target="$1"   # e.g., "3g.20gb", "2g.10gb", "1g.5gb", "4g.20gb"
  nvidia-smi -L | grep -E "MIG[[:space:]]+${target}[[:space:]]" \
    | grep -oE "MIG-[a-f0-9-]{36}"
}

run_preset() {
  local preset="$1"
  local pdir="$OUTDIR/$preset"
  mkdir -p "$pdir/results"

  echo "============================================================"
  echo "config: $preset   AI: $AI   cells: $CELLS   $(date +%H:%M:%S)"
  echo "============================================================"

  cleanup_workload_state

  local L1_DEV="" GPU_ARG_L1 AI_DEVS=()
  if [[ "$preset" == "no-mig" ]]; then
    L1_DEV="all-GPU-0"
    GPU_ARG_L1="all"
    AI_DEVS=("all")
  else
    create_mig_instances "$preset"
    sleep 2
    mapfile -t UUIDS < <(get_mig_uuids)
    if [[ ${#UUIDS[@]} -lt 2 ]]; then
      echo "ERROR: expected >=2 MIG instances, got ${#UUIDS[@]}"
      return 1
    fi
    local n=${#UUIDS[@]}
    case "$preset" in
      # Multi-partition presets — use profile-based explicit assignment
      3way-balanced)
        # 3g + 2g + 2g. L1=3g. AI×2 on the two 2g instances.
        L1_DEV=$(uuids_for_profile "3g.20gb" | head -1)
        mapfile -t AI_DEVS < <(uuids_for_profile "2g.10gb")
        ;;
      3way-L1small)
        # 3g + 2g + 2g. L1=2g (one of). AI on the OTHER 2g + the 3g.
        mapfile -t TWO_G < <(uuids_for_profile "2g.10gb")
        local three_g; three_g=$(uuids_for_profile "3g.20gb" | head -1)
        L1_DEV="${TWO_G[0]}"
        AI_DEVS=("${TWO_G[1]}" "$three_g")
        ;;
      3way-asym)
        # 4g + 2g + 1g. L1=4g (biggest). AI on 2g + 1g.
        L1_DEV=$(uuids_for_profile "4g.20gb" | head -1)
        local two_g; two_g=$(uuids_for_profile "2g.10gb" | head -1)
        local one_g; one_g=$(uuids_for_profile "1g.5gb" | head -1)
        AI_DEVS=("$two_g" "$one_g")
        ;;
      4way-1L1+3AI)
        # 4g + 1g + 1g + 1g. L1=4g. AI×3 on the three 1g.
        L1_DEV=$(uuids_for_profile "4g.20gb" | head -1)
        mapfile -t AI_DEVS < <(uuids_for_profile "1g.5gb")
        ;;
      # Legacy: four-way-bigL1 — L1 first (4g.20gb), rest are 1g
      four-way-bigL1)
        L1_DEV="${UUIDS[0]}"
        AI_DEVS=("${UUIDS[@]:1}")
        ;;
      # Default: L1 on last UUID (works for 2-partition splits)
      *)
        L1_DEV="${UUIDS[$((n-1))]}"
        AI_DEVS=("${UUIDS[@]:0:$((n-1))}")
        ;;
    esac
    # Verify assignment succeeded
    if [[ -z "$L1_DEV" ]] || [[ ${#AI_DEVS[@]} -eq 0 ]]; then
      echo "ERROR: L1_DEV or AI_DEVS empty for preset $preset"
      echo "  UUIDS: ${UUIDS[*]}"
      echo "  L1_DEV: $L1_DEV"
      echo "  AI_DEVS: ${AI_DEVS[*]}"
      return 1
    fi
    GPU_ARG_L1="\"device=$L1_DEV\""
  fi
  nvidia-smi -L > "$pdir/mig.txt"
  echo "  L1 device: $L1_DEV"
  echo "  AI devices: ${AI_DEVS[*]}"

  local mounts; mounts=$(common_mounts "$pdir/results")

  local AI_CIDS=() AI_NAMES=()
  local first_ai="${AI_LIST[0]}"
  if [[ "$first_ai" != "none" && -n "$first_ai" ]]; then
    local idx=0
    for d in "${AI_DEVS[@]}"; do
      local gpu_arg="\"device=$d\""
      [[ "$d" == "all" ]] && gpu_arg="all"
      local ai_name="${AI_LIST[$((idx % ${#AI_LIST[@]}))]}"
      local ai_script_path; ai_script_path=$(ai_script_for "$ai_name")
      local ai_args; ai_args=$(ai_args_for "$ai_name")
      cid=$(docker run -d --rm --gpus "$gpu_arg" $mounts "$IMAGE" \
        python3 "$ai_script_path" $ai_args)
      AI_CIDS+=("$cid")
      AI_NAMES+=("$ai_name")
      echo "  AI[$idx] $ai_name on $d -> $cid"
      idx=$((idx+1))
    done
    sleep 12
  fi

  echo "  running L1 -> $pdir/l1.log"
  docker run --rm --gpus "$GPU_ARG_L1" $mounts "$IMAGE" \
    python3 /scripts/real_l1.py "$preset" "$CELLS" "${L1_ITERATIONS:-50}" \
    > "$pdir/l1.log" 2>&1 || echo "  L1 returned non-zero"

  for i in "${!AI_CIDS[@]}"; do
    docker logs "${AI_CIDS[$i]}" > "$pdir/ai_${i}_${AI_NAMES[$i]}.log" 2>&1 || true
    docker rm -f "${AI_CIDS[$i]}" >/dev/null 2>&1 || true
  done

  # Verify result file produced.
  if ls "$pdir/results"/*.json >/dev/null 2>&1; then
    echo "  ✓ $preset OK"
    return 0
  else
    echo "  ✗ $preset FAILED (no JSON)"
    return 1
  fi
}

echo "results dir: $OUTDIR"
for p in $PRESETS; do
  run_preset "$p"
done

echo
echo "sweep complete: $OUTDIR"
ls -la "$OUTDIR"
