#!/usr/bin/env bash
# Chain 18 Part 2b — NCU on Full GPU MPS-on using ncu --mps client
# Redo of Part 2 MPSon which failed due to missing --mps client flag
set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-3}
L1_ITERS=${L1_ITERS:-5}
DUR=${DUR:-60}
NCU_KERNELS=${NCU_KERNELS:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}

IMAGE=airan:25-3-final
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/chain18_p2b_ncu_mps
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1=""
MPS_ENVS=""

mig_off_fullgpu() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n systemctl stop nvidia-dcgm 2>/dev/null
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -mig 0 2>/dev/null || true
  sleep 3
  UUID_L1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | head -1)
  log "  UUID_L1(Full GPU)=$UUID_L1"
}
mps_start() {
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo mkdir -p /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo chmod 777 /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  docker rm -f mps_srv 2>/dev/null || true
  docker run -d --gpus "device=$UUID_L1" --ipc=host --pid=host --user 0:0 \
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

profile_l1_ncu_mps() {
  local label=$1
  local METRICS="dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
l2_tex__t_sectors.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
smsp__inst_executed.sum,gpu__time_active.sum"
  docker run --rm --user 0:0 --gpus "device=$UUID_L1" --cap-add SYS_ADMIN \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "ncu --target-processes all --launch-count $NCU_KERNELS \
      --mps client \
      --metrics $METRICS \
      --csv --log-file /out/${label}.ncu.csv \
      python3 real_l1.py $label $CELLS $L1_ITERS 2>&1 | tail -5" > "$OUT/${label}.ncu.stdout" 2>&1
}

start_workload_bg() {
  local wl=$1 tag=$2
  case "$wl" in
    idle) : ;;
    nrx)
      docker run -d --rm --init --user 0:0 --gpus "device=$UUID_L1" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
        -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DUR > /aiout/${tag}.log 2>&1" >/dev/null ;;
    memcpy_loop)
      docker run -d --rm --init --user 0:0 --gpus "device=$UUID_L1" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 run_memcpy_loop.py $tag $DUR > /aiout/${tag}.log 2>&1" >/dev/null ;;
    embed_lookup)
      docker run -d --rm --init --user 0:0 --gpus "device=$UUID_L1" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 run_embed_lookup.py $tag $DUR > /aiout/${tag}.log 2>&1" >/dev/null ;;
    ranai_mix)
      docker run -d --rm --init --user 0:0 --gpus "device=$UUID_L1" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "python3 run_ranai_mix.py $tag $DUR > /aiout/${tag}.log 2>&1" >/dev/null ;;
    nrx_multi4)
      for i in 1 2 3 4; do
        docker run -d --rm --init --user 0:0 --gpus "device=$UUID_L1" \
          --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
          -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
          -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
          -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
          --name "ai_${tag}_c${i}" "$IMAGE" \
          bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DUR > /aiout/${tag}_c${i}.log 2>&1" >/dev/null
      done ;;
  esac
}

log "=== chain18 Part 2b (NCU --mps client) START ==="
mig_off_fullgpu

for WL in idle nrx memcpy_loop embed_lookup ranai_mix nrx_multi4; do
  mps_start
  LABEL="p2b_ncu_${WL}_MPSon"
  log "--- $LABEL ---"
  [[ $WL != idle ]] && { start_workload_bg "$WL" "${LABEL}_same"; sleep 15; }
  profile_l1_ncu_mps "$LABEL"
  kill_all_workloads
  mps_stop
done

log "=== chain18 Part 2b DONE ==="
echo "ncu csv: $(ls "$OUT"/*.ncu.csv 2>/dev/null | wc -l)"
touch /users/sgkim/CHAIN18_PART2B_DONE
