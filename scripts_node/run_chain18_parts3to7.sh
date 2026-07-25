#!/usr/bin/env bash
# Chain 18 Parts 3-7 combined runner (sequential after Part 2)
# Part 3: N-sweep extension (nrx: N=5,7,10,12,16; memcpy/embed N=1,2,4,6,8)
# Part 4: Fine MPS thread% (10 pct × 4 workloads on Config A)
# Part 5: Multi-thread vs multi-process control (ranai_mix variants)
# Part 6: Multi-GPU baseline (L1 GPU0, AI GPU1)
# Part 7: Long-window (300s) + statistical (10 trials on breakpoint)
set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DUR=${DUR:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
CONFIGS=${CONFIGS:-A}   # Part 3 uses A only for time; can be A B C

IMAGE=airan:25-3-final
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain18
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
  log "== config A: 4g.20gb + 3g.20gb =="; mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C 2>&1 | tail -3 >>"$OUT/run.log"; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_CROSS=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  log "  UUID_L1=$UUID_L1 UUID_CROSS=$UUID_CROSS"
  [[ -n $UUID_L1 && -n $UUID_CROSS ]] || return 1
  return 0
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

start_qwen_cross() {
  local u=$1 tag=$2
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    --entrypoint bash \
    -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$VLLM_IMAGE" \
    -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR+60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 32 --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
}

start_nrx_multiN() {
  local u=$1 tag=$2 N=$3 pct=${4:-0}
  local env_pct=""; [[ $pct -gt 0 && $pct -lt 100 ]] && env_pct="-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct"
  for i in $(seq 1 $N); do
    docker run -d --rm --init --user 0:0 --gpus "device=$u" \
      --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
      -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
      -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
      -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
      --name "ai_${tag}_c${i}" "$IMAGE" \
      bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR+60)) > /aiout/${tag}_c${i}.log 2>&1" >/dev/null
  done
}
start_memcpy_multiN() {
  local u=$1 tag=$2 N=$3 pct=${4:-0}
  local env_pct=""; [[ $pct -gt 0 && $pct -lt 100 ]] && env_pct="-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct"
  for i in $(seq 1 $N); do
    docker run -d --rm --init --user 0:0 --gpus "device=$u" \
      --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
      -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
      --name "ai_${tag}_c${i}" "$IMAGE" \
      bash -c "python3 run_memcpy_loop.py ${tag}_c${i} $((DUR+60)) > /aiout/${tag}_c${i}.log 2>&1" >/dev/null
  done
}
start_embed_multiN() {
  local u=$1 tag=$2 N=$3 pct=${4:-0}
  local env_pct=""; [[ $pct -gt 0 && $pct -lt 100 ]] && env_pct="-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct"
  for i in $(seq 1 $N); do
    docker run -d --rm --init --user 0:0 --gpus "device=$u" \
      --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
      -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
      --name "ai_${tag}_c${i}" "$IMAGE" \
      bash -c "python3 run_embed_lookup.py ${tag}_c${i} $((DUR+60)) > /aiout/${tag}_c${i}.log 2>&1" >/dev/null
  done
}
start_ranai_mix() {
  local u=$1 tag=$2 nrx=$3 csi=$4 beam=$5 pct=${6:-0}
  local env_pct=""; [[ $pct -gt 0 && $pct -lt 100 ]] && env_pct="-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct"
  docker run -d --rm --init --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
    -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
    --name "ai_${tag}" "$HF_IMAGE" \
    bash -c "python3 run_ranai_mix.py $tag $((DUR+60)) --n_nrx $nrx --n_csinet $csi --n_beampred $beam > /aiout/${tag}.log 2>&1" >/dev/null
}

profile_l1() {
  local label=$1 u=$2
  docker run --rm --user 0:0 --gpus "device=$u" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=$DUR --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py $label $CELLS $L1_ITERS" >/dev/null 2>&1
}

# ─── Part 3: N-sweep extension ────────────────────────────────
run_part3() {
  log "===== PART 3: N-sweep extension ====="
  setup_config_A || return
  # NRx: extend to N=5,7,10,12,16
  for N in 5 7 10 12 16; do
    for MPS in off on; do
      [[ $MPS == on ]] && mps_start "$UUID_L1"
      LABEL="p3_nrxN${N}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
        start_nrx_multiN "$UUID_L1" "${LABEL}_same_t${t}" $N
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
      [[ $MPS == on ]] && mps_stop
    done
  done
  # Memcpy: N=1,2,4,6,8
  for N in 1 2 4 6 8; do
    for MPS in off on; do
      [[ $MPS == on ]] && mps_start "$UUID_L1"
      LABEL="p3_memcpyN${N}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
        start_memcpy_multiN "$UUID_L1" "${LABEL}_same_t${t}" $N
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
      [[ $MPS == on ]] && mps_stop
    done
  done
  # Embed lookup: N=1,2,4,6,8
  for N in 1 2 4 6 8; do
    for MPS in off on; do
      [[ $MPS == on ]] && mps_start "$UUID_L1"
      LABEL="p3_embedN${N}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
        start_embed_multiN "$UUID_L1" "${LABEL}_same_t${t}" $N
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
      [[ $MPS == on ]] && mps_stop
    done
  done
}

# ─── Part 4: Fine MPS thread% sweep ────────────────────────────
run_part4() {
  log "===== PART 4: Fine MPS thread% sweep ====="
  setup_config_A || return
  # workloads to sweep:
  # nrx_multi4, ranai_mix (14 threads), embed_multi4, memcpy_multi4
  for PCT in 100 90 80 70 60 50 40 30 20 10; do
    for WLNAME in nrx4 ranai memcpy4 embed4; do
      mps_start "$UUID_L1"
      LABEL="p4_${WLNAME}_pct${PCT}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
        case $WLNAME in
          nrx4)    start_nrx_multiN   "$UUID_L1" "${LABEL}_same_t${t}" 4 $PCT ;;
          ranai)   start_ranai_mix    "$UUID_L1" "${LABEL}_same_t${t}" 2 4 8 $PCT ;;
          memcpy4) start_memcpy_multiN "$UUID_L1" "${LABEL}_same_t${t}" 4 $PCT ;;
          embed4)  start_embed_multiN  "$UUID_L1" "${LABEL}_same_t${t}" 4 $PCT ;;
        esac
        sleep 15
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
      done
      mps_stop
    done
  done
}

# ─── Part 5: Multi-thread vs multi-process control ────────────
run_part5() {
  log "===== PART 5: Multi-thread vs multi-process control ====="
  setup_config_A || return
  # Multi-thread sweep (1 process, varying threads via ranai_mix scaling)
  # nrx=2, csi=4, beam=X for total = 6+X threads
  mps_start "$UUID_L1"
  for BEAM in 4 8 16 32; do
    LABEL="p5_thrOnly_beam${BEAM}"
    log "--- $LABEL (single proc, ${BEAM}+6 threads) ---"
    for t in $(seq 1 $N_TRIALS); do
      start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
      start_ranai_mix "$UUID_L1" "${LABEL}_same_t${t}" 2 4 $BEAM
      sleep 15
      profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
    done
  done
  mps_stop
  # Multi-process sweep (N processes each with default ranai_mix 14 threads)
  for N in 1 2 4 8; do
    mps_start "$UUID_L1"
    LABEL="p5_procOnly_N${N}"
    log "--- $LABEL ($N procs, each 14 threads) ---"
    for t in $(seq 1 $N_TRIALS); do
      start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
      for i in $(seq 1 $N); do
        start_ranai_mix "$UUID_L1" "${LABEL}_same_t${t}_p${i}" 2 4 8
      done
      sleep 15
      profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
    done
    mps_stop
  done
}

# ─── Part 6: Multi-GPU baseline ────────────────────────────────
run_part6() {
  log "===== PART 6: Multi-GPU baseline (L1 on GPU0, AI on GPU1) ====="
  mig_off   # disable MIG on GPU 0
  UUID_L1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | head -1)
  UUID_G1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1 | head -1)
  log "  UUID_L1(GPU0)=$UUID_L1 UUID_G1=$UUID_G1"
  # Workloads on GPU 1
  for WL in nrx ranai_mix nrx_multi4 memcpy_loop embed_lookup; do
    LABEL="p6_multiGPU_${WL}"
    log "--- $LABEL ---"
    for t in $(seq 1 $N_TRIALS); do
      case $WL in
        nrx)          start_nrx_multiN "$UUID_G1" "${LABEL}_g1_t${t}" 1 ;;
        ranai_mix)    start_ranai_mix  "$UUID_G1" "${LABEL}_g1_t${t}" 2 4 8 ;;
        nrx_multi4)   start_nrx_multiN "$UUID_G1" "${LABEL}_g1_t${t}" 4 ;;
        memcpy_loop)  start_memcpy_multiN "$UUID_G1" "${LABEL}_g1_t${t}" 1 ;;
        embed_lookup) start_embed_multiN  "$UUID_G1" "${LABEL}_g1_t${t}" 1 ;;
      esac
      sleep 15
      profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
    done
  done
}

# ─── Part 7: Long-window + statistical ────────────────────────
run_part7() {
  log "===== PART 7: Long-window (300s) + statistical (10 trials) ====="
  setup_config_A || return
  # Long window on baseline + N=4 (safe) + N=6 (breakdown)
  DUR_ORIG=$DUR
  DUR=300
  for KEY in "baseline" "nrxN4_MPSon" "nrxN6_MPSon"; do
    LABEL="p7_long_${KEY}"
    log "--- $LABEL (300s) ---"
    if [[ $KEY != "baseline" ]]; then
      N=${KEY:4:1}; MPS=${KEY##*_MPS}
      [[ $MPS == on ]] && mps_start "$UUID_L1"
    fi
    for t in 1 2; do   # 2 trials at long duration
      start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
      case $KEY in
        baseline)    : ;;
        nrxN4_MPSon) start_nrx_multiN "$UUID_L1" "${LABEL}_same_t${t}" 4 ;;
        nrxN6_MPSon) start_nrx_multiN "$UUID_L1" "${LABEL}_same_t${t}" 6 ;;
      esac
      sleep 15
      profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
    done
    [[ ${KEY} == *"MPSon" ]] && mps_stop || true
  done
  DUR=$DUR_ORIG
  # 10-trial statistical at breakpoint (N=5, 6, 7 with MPS on)
  for N in 5 6 7; do
    mps_start "$UUID_L1"
    LABEL="p7_stat_nrxN${N}_MPSon"
    log "--- $LABEL (10 trials) ---"
    for t in $(seq 1 10); do
      start_qwen_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20
      start_nrx_multiN "$UUID_L1" "${LABEL}_same_t${t}" $N
      sleep 15
      profile_l1 "${LABEL}_t${t}" "$UUID_L1"; kill_all_workloads
    done
    mps_stop
  done
}

log "=== chain18 parts 3-7 START ==="
run_part3
run_part4
run_part5
run_part6
run_part7
mig_off
log "=== chain18 parts 3-7 DONE ==="
echo "captures: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)"
touch /users/sgkim/CHAIN18_DONE
