#!/usr/bin/env bash
# Chain 17 — Sensitivity sweeps + Low-level measurements
# ─────────────────────────────────────────────────────────────────────
# Part A: N-process sensitivity (1,2,3,4,6,8 concurrent NRx / ranai_mix)
# Part B: MPS thread% cap (100/70/50/30 for AI clients)
# Part D: DCGM real-time monitoring (parallel with all runs)
# Part C: NCU low-level counters (separate runner: run_chain17_ncu.sh)
# ─────────────────────────────────────────────────────────────────────
set -uo pipefail

GPU=${GPU:-0}
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DUR=${DUR:-30}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
CONFIGS=${CONFIGS:-A B C}
PARTS=${PARTS:-A B}   # A=N-sweep, B=MPS-cap

IMAGE=airan:25-3-final
VLLM_IMAGE=vllm/vllm-openai:v0.6.6
HF_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3

SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/mydata/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
HF_CACHE=/mydata/hf_cache
OUT=/mydata/results/$DATE_DIR/chain17
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

UUID_L1="" UUID_CROSS="" UUID_ALT="" MPS_ENVS=""
DCGM_PID=""

# ─── MIG helpers (unchanged from chain14/16) ─────────────────────
mig_off() {
  docker ps -aq | xargs -r docker rm -f 2>/dev/null
  sudo -n systemctl stop nvidia-dcgm 2>/dev/null || true    # release GPU 0
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo -n nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 3
}
dcgm_daemon_start() {
  sudo -n systemctl start nvidia-dcgm 2>/dev/null || true
  sleep 3
}
setup_config_A() {
  log "== config A: 4g.20gb + 3g.20gb =="; mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 4g.20gb,3g.20gb -C 2>&1 | tail -3 >>"$OUT/run.log"; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '4g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_CROSS=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  UUID_ALT=""
  log "  UUID_L1=$UUID_L1 UUID_CROSS=$UUID_CROSS"
  [[ -n $UUID_L1 && -n $UUID_CROSS ]] || return 1
  dcgm_daemon_start
  return 0
}
setup_config_B() {
  log "== config B: Full GPU 0 =="; mig_off
  UUID_L1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | head -1)
  UUID_CROSS=""; UUID_ALT=""
  log "  UUID_L1=$UUID_L1"
  [[ -n $UUID_L1 ]] || return 1
  dcgm_daemon_start
  return 0
}
setup_config_C() {
  log "== config C: 3g.20gb + 2g.10gb + 2g.10gb =="; mig_off
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1; sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 3g.20gb,2g.10gb,2g.10gb -C 2>&1 | tail -3 >>"$OUT/run.log"; sleep 2
  UUID_L1=$(nvidia-smi -L | grep '3g.20gb' | grep -oE 'MIG-[0-9a-f-]+' | head -1)
  local two; two=$(nvidia-smi -L | grep '2g.10gb' | grep -oE 'MIG-[0-9a-f-]+')
  UUID_CROSS=$(echo "$two" | sed -n '1p'); UUID_ALT=$(echo "$two" | sed -n '2p')
  log "  UUID_L1=$UUID_L1 UUID_CROSS=$UUID_CROSS"
  [[ -n $UUID_L1 && -n $UUID_CROSS && -n $UUID_ALT ]] || return 1; return 0
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

# ─── DCGM parallel monitor (Part D) ──────────────────────────────
dcgm_start() {
  local label=$1
  if command -v dcgmi >/dev/null 2>&1; then
    dcgmi dmon -e 1001,1002,1003,1004,1005,1006,1007,1008 -c $((DUR+5)) -d 100 \
      > "$OUT/${label}_dcgm.tsv" 2>&1 &
    DCGM_PID=$!
  else
    DCGM_PID=""
  fi
}
dcgm_stop() {
  if [[ -n ${DCGM_PID:-} ]]; then
    kill $DCGM_PID 2>/dev/null || true
    wait $DCGM_PID 2>/dev/null || true
    DCGM_PID=""
  fi
}

# ─── Workload starters (with optional MPS thread% cap) ──────────
# $4 = mps_pct (0 or 100 = no cap)
start_workload_bg() {
  local wl=$1 u=$2 tag=$3 mps_pct=${4:-0}
  local env_pct=""
  [[ $mps_pct -gt 0 && $mps_pct -lt 100 ]] && env_pct="-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mps_pct"
  case "$wl" in
    nrx)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
        -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR+60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    chanpred)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "python3 run_chanpred.py $tag $((DUR+60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    memcpy_loop)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 run_memcpy_loop.py $tag $((DUR+60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    embed_lookup)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$IMAGE" \
        bash -c "python3 run_embed_lookup.py $tag $((DUR+60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    ranai_mix)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "python3 run_ranai_mix.py $tag $((DUR+60)) > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    ranai_mix_heavy)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$HF_IMAGE" \
        bash -c "python3 run_ranai_mix.py $tag $((DUR+60)) --n_nrx 4 --n_csinet 8 --n_beampred 16 > /aiout/${tag}.log 2>&1" >/dev/null
      ;;
    nrx_multi4)
      for i in 1 2 3 4; do
        docker run -d --rm --init --user 0:0 --gpus "device=$u" \
          --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
          -v "$SDK:/opt/nvidia/cuBB" -v "$REPO:/workspace/AIRAN_Changjong" -v "$OUT:/aiout" \
          -e cuBB_SDK=/opt/nvidia/cuBB -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
          -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
          --name "ai_${tag}_c${i}" "$IMAGE" \
          bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $((DUR+60)) > /aiout/${tag}_c${i}.log 2>&1" >/dev/null
      done
      ;;
    qwen_llm_cross)
      docker run -d --rm --init --user 0:0 --gpus "device=$u" \
        --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $env_pct \
        --entrypoint bash \
        -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
        -v "$SCRIPT:/scripts" -v "$OUT:/aiout" -w /scripts \
        --name "ai_${tag}" "$VLLM_IMAGE" \
        -c "pip install -q datasets 2>/dev/null; python3 run_qwen_rag_hbm.py $tag $((DUR+60)) --model Qwen/Qwen2.5-3B-Instruct --n_concurrent 32 --gpu_mem_util 0.55 > /aiout/${tag}.log 2>&1"
      ;;
    idle) : ;;
    *) log "unknown $wl"; return 1;;
  esac
}

# Part A: N-process NRx (variable N)
start_nrx_multiN() {
  local u=$1 tag=$2 N=$3 mps_pct=${4:-0}
  local env_pct=""
  [[ $mps_pct -gt 0 && $mps_pct -lt 100 ]] && env_pct="-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mps_pct"
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

# ─── Part A: N-process sweep ──────────────────────────────────
run_partA() {
  local cfg=$1
  log "===== Part A (N-process sweep) cfg=$cfg ====="
  for N in 1 2 3 4 6 8; do
    for MPS in off on; do
      [[ $MPS == on ]] && mps_start "$UUID_L1"
      LABEL="cfg${cfg}_A_nrxN${N}_MPS${MPS}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_workload_bg qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        start_nrx_multiN "$UUID_L1" "${LABEL}_same_t${t}" $N
        sleep 15
        dcgm_start "${LABEL}_t${t}"
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"
        dcgm_stop
        kill_all_workloads
      done
      [[ $MPS == on ]] && mps_stop
    done
  done
}

# ─── Part B: MPS thread% cap sweep ────────────────────────────
run_partB() {
  local cfg=$1
  log "===== Part B (MPS thread% cap) cfg=$cfg ====="
  # MPS ON only, sweep AI thread%
  for WL in nrx chanpred memcpy_loop embed_lookup ranai_mix ranai_mix_heavy nrx_multi4; do
    for PCT in 100 70 50 30; do
      mps_start "$UUID_L1"
      LABEL="cfg${cfg}_B_${WL}_pct${PCT}"
      log "--- $LABEL ---"
      for t in $(seq 1 $N_TRIALS); do
        [[ -n $UUID_CROSS ]] && { start_workload_bg qwen_llm_cross "$UUID_CROSS" "${LABEL}_cross_t${t}"; sleep 20; }
        start_workload_bg "$WL" "$UUID_L1" "${LABEL}_same_t${t}" $PCT
        sleep 15
        dcgm_start "${LABEL}_t${t}"
        profile_l1 "${LABEL}_t${t}" "$UUID_L1"
        dcgm_stop
        kill_all_workloads
      done
      mps_stop
    done
  done
}

# ─── Main ─────────────────────────────────────────────────────
log "=== chain17 START configs=$CONFIGS parts=$PARTS ==="
for cfg in $CONFIGS; do
  case $cfg in
    A) setup_config_A || continue ;;
    B) setup_config_B || continue ;;
    C) setup_config_C || continue ;;
  esac
  for part in $PARTS; do
    case $part in
      A) run_partA "$cfg" ;;
      B) run_partB "$cfg" ;;
    esac
  done
done
mig_off
log "=== chain17 DONE ==="
echo "captures: $(ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l)"
