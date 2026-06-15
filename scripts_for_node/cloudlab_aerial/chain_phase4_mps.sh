#!/usr/bin/env bash
# Phase 4 on d8545 — MIG off + true no-MIG baseline + MPS comparison.
# Run AFTER chain_all_tiers.sh completes and after a reboot if MIG-disable is pending.
#
# Sequence:
#   1) Disable MIG on GPU 0 (assumes reboot already done if needed)
#   2) D2-D: no-MIG default time-slice (no MPS) — L1 alone, L1+chanpred, L1+NeuralRx coloc
#   3) D2-M: MPS on, same conditions — quantifies MPS recovery vs default
#   4) Restore split-60-40 (4g+3g) on exit
#
# Compare with E0/E1/E2/E3 (MIG cross-partition) and E6 (MIG same-partition coloc)
# to fill the 3-way matrix MIG / no-MIG default / no-MIG MPS that the paper needs.

set -uo pipefail
RESULTS=/mydata/results/20260614
SCRIPTS=/mydata/work/airan_cloudlab/scripts_for_node/cloudlab_aerial
EXPERIMENTS=/mydata/work/airan_cloudlab/scripts_for_node/experiments
AERIAL=/mydata/work/aerial-cuda-accelerated-ran
IMG=airan:25-3
GPU=0
LOG=$RESULTS/logs/phase4_$(date +%H%M).log
mkdir -p "$RESULTS"/D2_nomig/{default,mps}/{alone,chanpred_coloc,neuralrx_coloc} "$RESULTS/logs"
sudo chmod -R 777 "$RESULTS"

ts(){ date +%H:%M:%S; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

skip_if_done(){
  local outdir="$1" expect="$2"
  local got=$(ls "$outdir"/realL1_*.json 2>/dev/null | wc -l)
  if [[ $got -ge $expect ]]; then return 0; else return 1; fi
}

###############################################################
# Step 1 — Disable MIG on GPU 0
###############################################################
log "===== Phase 4 START — MIG off + MPS ====="
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sleep 2
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sudo nvidia-smi -i $GPU -mig 0 2>&1 | tee -a "$LOG"
sudo systemctl start nvidia-persistenced 2>/dev/null || true
sleep 3
MIG_STATE=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i $GPU)
log "MIG state after disable attempt: $MIG_STATE"
if [[ "$MIG_STATE" == "Enabled" ]]; then
  log "ERROR: MIG still enabled (pending disable — needs reboot). Aborting Phase 4."
  log "  After reboot, re-run this script."
  exit 1
fi
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
log "no-MIG GPU 0 UUID: $GPU_UUID"

###############################################################
# Helper: L1 run under given MPS context (or none)
###############################################################
run_l1(){
  local label=$1 outdir=$2 extra_env=${3:-}
  docker run --rm --gpus "\"device=$GPU_UUID\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v "$outdir":/out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
    $extra_env \
    -w /scripts $IMG python3 real_l1.py "$label" 20 100 2>&1 | tail -1 | tee -a "$LOG"
}

ai_bg(){
  local name=$1 extra_env=$2; shift 2
  docker rm -f "$name" 2>/dev/null || true
  docker run -d --name "$name" --gpus "\"device=$GPU_UUID\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $EXPERIMENTS:/experiments -v /mydata/hf_cache:/hf_cache \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e cuBB_SDK=/opt/nvidia/cuBB \
    -e HF_HOME=/hf_cache $extra_env \
    $IMG python3 "$@" >> "$LOG" 2>&1
}

kill_all_ai(){
  for c in $(docker ps -q --filter "name=ai_"); do docker rm -f "$c" >/dev/null 2>&1; done
  sleep 3
}

run_3conditions(){
  local kind=$1 extra_env=$2

  outdir=$RESULTS/D2_nomig/$kind/alone
  if skip_if_done "$outdir" 10; then log "  skip $kind-alone"; else
    log "  $kind-alone (n=10)"
    for i in $(seq 1 10); do
      run_l1 "${kind}_alone_run${i}" "$outdir" "$extra_env"
    done
  fi

  outdir=$RESULTS/D2_nomig/$kind/chanpred_coloc
  if skip_if_done "$outdir" 10; then log "  skip $kind-chanpred"; else
    log "  $kind-chanpred coloc (n=10)"
    ai_bg "ai_p4_cp" "$extra_env" /experiments/run_channel_prediction.py 0 1200
    sleep 12
    for i in $(seq 1 10); do
      run_l1 "${kind}_chanpred_coloc_run${i}" "$outdir" "$extra_env"
    done
    kill_all_ai
  fi

  outdir=$RESULTS/D2_nomig/$kind/neuralrx_coloc
  if skip_if_done "$outdir" 10; then log "  skip $kind-neuralrx"; else
    log "  $kind-NeuralRx coloc (n=10, 90s TRT warmup)"
    ai_bg "ai_p4_nrx" "$extra_env" /experiments/run_neural_rx_stress.py 0 1200
    sleep 90
    if [[ "$(docker inspect -f '{{.State.Status}}' ai_p4_nrx 2>/dev/null)" != "running" ]]; then
      log "    NeuralRx died — skipping"
      kill_all_ai
    else
      for i in $(seq 1 10); do
        run_l1 "${kind}_neuralrx_coloc_run${i}" "$outdir" "$extra_env"
      done
      kill_all_ai
    fi
  fi
}

###############################################################
# Step 2 — no-MIG default time-slice (no MPS)
###############################################################
log "=== D2-default (no MPS) ==="
run_3conditions default ""

###############################################################
# Step 3 — no-MIG with MPS enabled
###############################################################
log "=== D2-MPS (MPS enabled) ==="
# MPS dirs under /mydata so they survive across container mounts
export CUDA_MPS_PIPE_DIRECTORY=/mydata/mps_pipe_$$
export CUDA_MPS_LOG_DIRECTORY=/mydata/mps_log_$$
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
sudo nvidia-cuda-mps-control -d 2>&1 | tee -a "$LOG"
sleep 3
echo "get_default_active_thread_percentage" | sudo nvidia-cuda-mps-control 2>&1 | head -2 | tee -a "$LOG"

# Propagate MPS env into all containers
MPS_ENV="-e CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY -v $CUDA_MPS_PIPE_DIRECTORY:$CUDA_MPS_PIPE_DIRECTORY"
run_3conditions mps "$MPS_ENV"

# Tear down MPS
echo quit | sudo nvidia-cuda-mps-control 2>/dev/null || true
sleep 2
sudo rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

###############################################################
# Step 4 — restore MIG split-60-40 (may need reboot to take effect)
###############################################################
log "===== restoring MIG (this is pending if MIG was off — reboot to fully apply) ====="
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sleep 2
sudo nvidia-smi -i $GPU -mig 1 2>&1 | tee -a "$LOG"
MIG_STATE=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i $GPU)
log "  MIG mode after enable: $MIG_STATE (may be pending; reboot to finalize)"
if [[ "$MIG_STATE" == "Enabled" ]]; then
  sudo nvidia-smi mig -i $GPU -cgi 5,9 -C 2>&1 | tee -a "$LOG"
fi
sudo systemctl start nvidia-persistenced 2>/dev/null || true

###############################################################
# Final summary table
###############################################################
log "===== Phase 4 summary ====="
python3 - <<PYEOF | tee -a "$LOG"
import json, glob, statistics
R = "$RESULTS"
def stats(p):
    ms = []
    for f in sorted(glob.glob(p)):
        ms += json.load(open(f)).get("raw_ms", [])
    if not ms: return None
    n = len(ms); s = sorted(ms)
    p99 = s[int(n*0.99)] if n >= 100 else max(ms)
    return n, statistics.median(ms), statistics.mean(ms), p99, max(ms)

print("{:42s} | {:>5s} | {:>6s} | {:>6s} | {:>6s} | {:>6s}".format(
    "condition","n","p50","mean","p99","max"))
print("-"*95)
for kind in ["default", "mps"]:
    for cond in ["alone", "chanpred_coloc", "neuralrx_coloc"]:
        s = stats(f"{R}/D2_nomig/{kind}/{cond}/realL1_*.json")
        if s: print("{:42s} | {:5d} | {:6.1f} | {:6.1f} | {:6.1f} | {:6.1f}".format(f"D2 {kind} {cond}", *s))
        else: print("{:42s} | NO DATA".format(f"D2 {kind} {cond}"))
PYEOF

log "===== Phase 4 DONE ====="
