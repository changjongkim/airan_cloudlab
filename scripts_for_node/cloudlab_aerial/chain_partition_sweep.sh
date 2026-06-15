#!/usr/bin/env bash
# Partition-size sweep on d8545 — all four MIG profiles × {alone, NeuralRx coloc}.
# Idempotent: skips conditions whose result dir already has enough JSON files.
# Designed to run under nohup so it survives SSH disconnect.

set -uo pipefail
RESULTS=/mydata/results/20260614
SCRIPTS=/mydata/work/airan_cloudlab/scripts_for_node/cloudlab_aerial
EXPERIMENTS=/mydata/work/airan_cloudlab/scripts_for_node/experiments
AERIAL=/mydata/work/aerial-cuda-accelerated-ran
IMG=airan:25-3
GPU=0
LOG=$RESULTS/logs/chain_$(date +%H%M).log
mkdir -p "$RESULTS"/{E5_alone_partition,E6_coloc_neuralrx}/{2g,3g,4g,7g} "$RESULTS"/logs
sudo chmod -R 777 "$RESULTS"

ts(){ date +%H:%M:%S; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

# Profile IDs (A100 40GB):  0=7g.40gb  5=4g.20gb  9=3g.20gb  14=2g.10gb
profile_for(){
  case "$1" in
    7g) echo 0  ;;
    4g) echo 5  ;;
    3g) echo 9  ;;
    2g) echo 14 ;;
    *) echo "unknown profile $1"; exit 1 ;;
  esac
}

reconfig_mig(){
  local size=$1
  local profile_id=$(profile_for "$size")
  log "MIG reconfig to $size (profile_id=$profile_id) on GPU $GPU"
  docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
  sudo systemctl stop nvidia-persistenced 2>/dev/null || true
  sleep 2
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sleep 2
  sudo nvidia-smi mig -i $GPU -cgi "$profile_id" -C 2>&1 | tee -a "$LOG"
  sudo systemctl start nvidia-persistenced 2>/dev/null || true
  sleep 3
  L1_UUID=$(nvidia-smi -L | grep MIG | head -1 | grep -oE "MIG-[a-f0-9-]+")
  log "  $size UUID: $L1_UUID"
}

# Run L1 alone on the current L1_UUID — n runs of 100 iters
run_alone(){
  local size=$1 outdir=$2 n=${3:-5}
  local got=$(ls "$outdir"/realL1_*.json 2>/dev/null | wc -l)
  if [[ $got -ge $n ]]; then
    log "  skip alone-$size: already have $got JSON files"
    return
  fi
  log "  alone-$size n=$n on UUID=$L1_UUID"
  for i in $(seq 1 "$n"); do
    docker run --rm --gpus "\"device=$L1_UUID\"" \
      -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v "$outdir":/out \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
      -w /scripts $IMG \
      python3 real_l1.py alone_${size}_run${i} 20 100 2>&1 | tail -1 | tee -a "$LOG"
  done
}

# Run L1 + NeuralRx coloc on the current L1_UUID — both share same MIG instance
run_coloc_nrx(){
  local size=$1 outdir=$2 n=${3:-10}
  local got=$(ls "$outdir"/realL1_*.json 2>/dev/null | wc -l)
  if [[ $got -ge $n ]]; then
    log "  skip coloc-NeuralRx-$size: already have $got JSON files"
    return
  fi
  log "  coloc-NeuralRx-$size n=$n on UUID=$L1_UUID"
  docker rm -f nrx_coloc 2>/dev/null || true
  docker run -d --name nrx_coloc --gpus "\"device=$L1_UUID\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $EXPERIMENTS:/experiments \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e cuBB_SDK=/opt/nvidia/cuBB \
    $IMG python3 /experiments/run_neural_rx_stress.py 0 3600 >> "$LOG" 2>&1
  log "    NeuralRx warmup 90s..."
  sleep 90
  local state=$(docker inspect -f '{{.State.Status}}' nrx_coloc 2>/dev/null)
  if [[ "$state" != "running" ]]; then
    log "    NeuralRx died ($state) — skipping coloc-$size"
    docker logs nrx_coloc 2>&1 | tail -10 | tee -a "$LOG"
    docker rm -f nrx_coloc 2>&1 | tee -a "$LOG"
    return
  fi
  for i in $(seq 1 "$n"); do
    docker run --rm --gpus "\"device=$L1_UUID\"" \
      -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v "$outdir":/out \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
      -w /scripts $IMG \
      python3 real_l1.py coloc_nrx_${size}_run${i} 20 100 2>&1 | tail -1 | tee -a "$LOG"
  done
  docker rm -f nrx_coloc 2>&1 | tail -1 | tee -a "$LOG"
}

# Cycle by partition size — process them in order that minimises reconfig pain
# For each size: reconfig, alone, coloc-NeuralRx.
log "===== CHAIN START — partition sweep ====="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)"

for size in 3g 4g 7g 2g; do
  reconfig_mig "$size"
  if [[ -z "${L1_UUID:-}" ]]; then
    log "  no UUID after reconfig — aborting size=$size"; continue
  fi
  run_alone "$size" "$RESULTS/E5_alone_partition/$size" 5
  run_coloc_nrx "$size" "$RESULTS/E6_coloc_neuralrx/$size" 10
done

# Restore split-60-40 (4g + 3g) so the node is left in the per-paper config
log "===== restoring MIG split-60-40 (4g+3g) ====="
docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sleep 2
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sleep 2
sudo nvidia-smi mig -i $GPU -cgi 5,9 -C 2>&1 | tee -a "$LOG"
sudo systemctl start nvidia-persistenced 2>/dev/null || true
nvidia-smi -L | grep MIG | tee -a "$LOG"

# Final summary
log "===== writing summary ====="
python3 - <<PYEOF | tee -a "$LOG"
import json, glob, statistics, os
RESULTS = "$RESULTS"
print(f"{'condition':28s} | {'n':>5s} | {'p50':>6s} | {'mean':>6s} | {'p99':>6s} | {'max':>6s}")
print("-" * 80)
for kind in ["E5_alone_partition", "E6_coloc_neuralrx"]:
    for size in ["2g", "3g", "4g", "7g"]:
        files = sorted(glob.glob(os.path.join(RESULTS, kind, size, "realL1_*.json")))
        if not files:
            print(f"{kind+'/'+size:28s} | NO DATA"); continue
        ms = []
        for f in files: ms += json.load(open(f)).get("raw_ms", [])
        if not ms: continue
        n = len(ms); s = sorted(ms)
        p99 = s[int(n*0.99)] if n >= 100 else max(ms)
        print(f"{kind+'/'+size:28s} | {n:>5d} | {statistics.median(ms):6.1f} | {statistics.mean(ms):6.1f} | {p99:6.1f} | {max(ms):6.1f}")
PYEOF

log "===== CHAIN DONE ====="
