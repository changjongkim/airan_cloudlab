#!/usr/bin/env bash
# Chain 17 Part C — NCU low-level counters on L1 kernels
# ─────────────────────────────────────────────────────────────────────
# For each workload + MPS state, capture NCU metrics on L1 kernels:
#   - dram__bytes.sum (actual HBM bytes)
#   - dram__throughput.avg.pct_of_peak_sustained_elapsed (HBM %)
#   - smsp__cycles_active.avg.pct_of_peak_sustained_elapsed (SM %)
#   - l2_tex__t_sectors.sum (L2 traffic)
#   - lts__t_sectors_op_read.sum, lts__t_sectors_op_write.sum
#
# NCU is 10-50x slower than nsys; capture only a few kernels each run.
# Config A only (MIG 4g); other configs skipped for time.
# ─────────────────────────────────────────────────────────────────────
set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-3}     # reduced for NCU (fewer kernels per iter)
L1_ITERS=${L1_ITERS:-5}
N_TRIALS=${N_TRIALS:-1}  # NCU is slow, 1 trial
DUR=${DUR:-60}         # longer window since NCU slows L1
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
NCU_KERNELS=${NCU_KERNELS:-20}   # top-N kernels to capture

IMAGE=airan:25-3-final
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain17_ncu
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1="" UUID_CROSS="" MPS_ENVS=""

mig_off() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n systemctl stop nvidia-dcgm 2>/dev/null || true
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 3
}
setup_config_A() {
  log "== MIG A: 4g+3g =="; mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C 2>&1 | tail -3 >>"$OUT/run.log"; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_CROSS=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  log "  L1=$UUID_L1 CROSS=$UUID_CROSS"
  [[ -n $UUID_L1 ]] || return 1
}
mps_start() {
  local u=$1
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo mkdir -p /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo chmod 777 /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  docker rm -f mps_srv 2>/dev/null || true
  docker run -d --gpus "device=$u" --ipc=host --pid=host --user 0:0 \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU \
    -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU \
    --name mps_srv "$IMAGE" \
    bash -c "nvidia-cuda-mps-control -d && sleep infinity" >/dev/null
  sleep 5
  MPS_ENVS="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
}
mps_stop() { docker rm -f mps_srv 2>/dev/null || true; sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU; MPS_ENVS=""; }
kill_all_workloads() { docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 2; }

# NCU-profiled L1 run
profile_l1_ncu() {
  local label=$1 u=$2
  # NCU metrics — memory subsystem focus
  local METRICS="dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
l2_tex__t_sectors.sum,\
lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
smsp__inst_executed.sum,\
gpu__time_active.sum"
  docker run --rm --user 0:0 --gpus "device=$u" \
    --cap-add SYS_ADMIN \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "ncu --target-processes all --launch-count $NCU_KERNELS \
      --metrics $METRICS \
      --csv --log-file /out/${label}.ncu.csv \
      python3 real_l1.py $label $CELLS $L1_ITERS 2>&1 | tail -5" > "$OUT/${label}.ncu.stdout" 2>&1
}

start_workload_bg() {
  local wl=$1 u=$2 tag=$3
  case "$wl" in
    idle) : ;;
    nrx)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
        -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DUR > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    memcpy_loop)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 run_memcpy_loop.py $tag $DUR > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    embed_lookup)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 run_embed_lookup.py $tag $DUR > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    ranai_mix)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "python3 run_ranai_mix.py $tag $DUR > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    nrx_multi4)
      for i in 1 2 3 4; do
        docker run -d --rm --init --user 0:0 --gpus "device=$u" \
          --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
          -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
          -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
          -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
          --name "ai_${tag}_c${i}" "$IMAGE" \
          bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DUR > /aiout/${tag}_c${i}.log 2>&1" >/dev/null
      done
      ;;
  esac
}

# ─── Main NCU pass ────────────────────────────────────────────
log "=== chain17_ncu START ==="
setup_config_A || { log "MIG fail"; exit 1; }

# NCU workloads: baseline + key sync/memory-sensitive
for WL in idle nrx memcpy_loop embed_lookup ranai_mix nrx_multi4; do
  for MPS in off on; do
    [[ $MPS == on ]] && mps_start "$UUID_L1"
    LABEL="cfgA_NCU_${WL}_MPS${MPS}"
    log "--- $LABEL ---"
    [[ $WL != idle ]] && { start_workload_bg "$WL" "$UUID_L1" "${LABEL}_same"; sleep 15; }
    profile_l1_ncu "$LABEL" "$UUID_L1"
    kill_all_workloads
    [[ $MPS == on ]] && mps_stop
  done
done

mig_off
log "=== chain17_ncu DONE ==="
echo "ncu csv: $(ls "$OUT"/*.ncu.csv 2>/dev/null | wc -l)"
