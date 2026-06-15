#!/usr/bin/env bash
# All-tier chain on d8545 — Phase 1-3 (everything that can run without MIG-off).
# Phase 4 (MPS, true no-MIG) needs reboot and is run separately.
#
# Phase 1: current MIG split-60-40 (L1 on 3g, AI on 4g)
#   A1: multi-AI cross-partition stacking (chanpred×4, ResNet×2, kitchen)
#   A2: mixed coloc (L1 + chanpred + ResNet all in 3g)
#   B3: ResNet batch sweep in coloc on 3g (b16, b64, b256)
#   C1: NCU on L1 alone + L1+NeuralRx coloc (3g)
#   C2: NSYS on L1 alone + L1+NeuralRx coloc (3g)
#
# Phase 2: chanpred-coloc partition sweep (2g/4g/7g)
#   A3-2g, A3-4g, A3-7g  (we already have 3g via E3)
#
# Phase 3: four-way bigL1 (4g L1 + 1g+1g+1g AI)
#   B1: alone + 3-tenant cross-partition stacking
#
# Phase 5: 5-min sustained on restored split-60-40
#   C3: L1 alone, L1+chanpred coloc, L1+NeuralRx coloc, L1+NeuralRx cross
#
# Idempotent (skips if N files already present). nohup-safe.

set -uo pipefail
RESULTS=/mydata/results/20260614
SCRIPTS=/mydata/work/airan_cloudlab/scripts_for_node/cloudlab_aerial
EXPERIMENTS=/mydata/work/airan_cloudlab/scripts_for_node/experiments
AERIAL=/mydata/work/aerial-cuda-accelerated-ran
IMG=airan:25-3
GPU=0
LOG=$RESULTS/logs/chain_all_$(date +%H%M).log
mkdir -p "$RESULTS"/{A1_stacking,A2_mixed_coloc,A3_chanpred_coloc,B1_four_way,B3_resnet_batch,C1_ncu,C2_nsys,C3_sustained,logs}
sudo chmod -R 777 "$RESULTS"

ts(){ date +%H:%M:%S; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

profile_for(){
  case "$1" in
    7g) echo 0 ;; 4g) echo 5 ;; 3g) echo 9 ;; 2g) echo 14 ;; 1g) echo 19 ;;
    *) echo "??"; return 1 ;;
  esac
}

mig_reconfig(){
  local spec="$1" desc="$2"
  log "MIG reconfig: $desc (spec=$spec)"
  docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
  sudo systemctl stop nvidia-persistenced 2>/dev/null || true
  sleep 2
  sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
  sleep 2
  sudo nvidia-smi mig -i $GPU -cgi "$spec" -C 2>&1 | tee -a "$LOG"
  sudo systemctl start nvidia-persistenced 2>/dev/null || true
  sleep 3
  nvidia-smi -L | grep MIG | tee -a "$LOG"
}

mig_uuids(){
  nvidia-smi -L | awk -F'UUID: ' '/MIG/ {gsub(/\)/,"",$2); print $2}'
}

skip_if_done(){
  local outdir="$1" expect="$2"
  local got=$(ls "$outdir"/realL1_*.json 2>/dev/null | wc -l)
  if [[ $got -ge $expect ]]; then return 0; else return 1; fi
}

run_l1_once(){
  local label=$1 uuid=$2 outdir=$3
  docker run --rm --gpus "\"device=$uuid\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v "$outdir":/out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
    -w /scripts $IMG python3 real_l1.py "$label" 20 100 2>&1 | tail -1 | tee -a "$LOG"
}

ai_bg(){
  local name=$1 uuid=$2; shift 2
  docker rm -f "$name" 2>/dev/null || true
  docker run -d --name "$name" --gpus "\"device=$uuid\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $EXPERIMENTS:/experiments -v /mydata/hf_cache:/hf_cache \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e cuBB_SDK=/opt/nvidia/cuBB \
    -e HF_HOME=/hf_cache \
    $IMG python3 "$@" >> "$LOG" 2>&1
}

kill_all_ai(){
  # remove any ai_* and stacking containers, ignore L1 which uses --rm
  for c in $(docker ps -q --filter "name=ai_" --filter "name=stack_"); do docker rm -f "$c" >/dev/null 2>&1; done
  sleep 3
}

log "===== CHAIN-ALL START ====="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)"
log "Initial MIG: $(nvidia-smi -L | grep MIG | head -2 | tr '\n' '|')"

###############################################################
# Phase 1 — current split-60-40 (L1 on 3g, AI on 4g)
###############################################################
mig_reconfig "5,9" "Phase1 split-60-40 (4g + 3g)"
L1_UUID=$(mig_uuids | grep -A0 '' | sed -n '2p')  # 3g is second instance created
AI_UUID=$(mig_uuids | sed -n '1p')                # 4g is first
log "Phase1 L1=$L1_UUID (3g) AI=$AI_UUID (4g)"

###### A1: multi-AI cross-partition stacking ######
log "=== A1 stacking on cross-partition ==="

# A1-a: chanpred x4 on 4g
outdir=$RESULTS/A1_stacking/chanpred_x4
mkdir -p "$outdir"
if skip_if_done "$outdir" 5; then log "  skip A1-chanpred_x4 (already have ≥5)"; else
  for k in 1 2 3 4; do
    ai_bg "stack_cp_$k" "$AI_UUID" /experiments/run_channel_prediction.py 0 1200
  done
  log "  4 chanpred started on 4g, warming 12s"
  sleep 12
  for i in 1 2 3 4 5; do
    run_l1_once "A1_chanpred_x4_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
fi

# A1-b: resnet x2 on 4g
outdir=$RESULTS/A1_stacking/resnet_x2
mkdir -p "$outdir"
if skip_if_done "$outdir" 5; then log "  skip A1-resnet_x2"; else
  for k in 1 2; do
    ai_bg "stack_rn_$k" "$AI_UUID" /experiments/run_resnet_stress.py 0 1200 64 fp16
  done
  log "  2 resnet started on 4g, warming 12s"
  sleep 12
  for i in 1 2 3 4 5; do
    run_l1_once "A1_resnet_x2_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
fi

# A1-c: kitchen sink (chanpred + memcpy + gemm)
outdir=$RESULTS/A1_stacking/kitchen
mkdir -p "$outdir"
if skip_if_done "$outdir" 5; then log "  skip A1-kitchen"; else
  ai_bg "stack_kitch_cp" "$AI_UUID" /experiments/run_channel_prediction.py 0 1200
  ai_bg "stack_kitch_mc" "$AI_UUID" /experiments/run_memcpy_massive.py 0 1200
  ai_bg "stack_kitch_gm" "$AI_UUID" /experiments/run_gemm_massive.py 0 1200
  log "  kitchen sink (chanpred+memcpy+gemm) started, warming 12s"
  sleep 12
  for i in 1 2 3 4 5; do
    run_l1_once "A1_kitchen_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
fi

###### A2: mixed coloc (L1 + chanpred + ResNet all on 3g) ######
log "=== A2 mixed coloc on 3g (L1 + chanpred + ResNet) ==="
outdir=$RESULTS/A2_mixed_coloc/chanpred_resnet
mkdir -p "$outdir"
if skip_if_done "$outdir" 10; then log "  skip A2"; else
  ai_bg "ai_mix_cp" "$L1_UUID" /experiments/run_channel_prediction.py 0 1200
  ai_bg "ai_mix_rn" "$L1_UUID" /experiments/run_resnet_stress.py 0 1200 64 fp16
  log "  chanpred+resnet coloc on 3g, warming 12s"
  sleep 12
  for i in $(seq 1 10); do
    run_l1_once "A2_mixed_coloc_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
fi

###### B3: ResNet batch sweep in 3g coloc ######
log "=== B3 ResNet batch sweep on 3g coloc ==="
for b in 16 64 256; do
  outdir=$RESULTS/B3_resnet_batch/b${b}
  mkdir -p "$outdir"
  if skip_if_done "$outdir" 5; then log "  skip B3-b${b}"; continue; fi
  ai_bg "ai_rn_b${b}" "$L1_UUID" /experiments/run_resnet_stress.py 0 1200 ${b} fp16
  log "  ResNet b=${b} on 3g coloc, warming 10s"
  sleep 10
  for i in 1 2 3 4 5; do
    run_l1_once "B3_resnet_b${b}_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
done

###### C1: NCU on alone + coloc ######
log "=== C1 NCU (kernel replay, slow) ==="
NCU_METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor"

# alone
if [[ -f "$RESULTS/C1_ncu/alone.csv" ]]; then log "  skip C1-alone (csv exists)"; else
  log "  C1-alone (ITERS=3, replay-mode)"
  docker run --rm --gpus "\"device=$L1_UUID\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v $RESULTS/C1_ncu:/out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
    -w /scripts $IMG bash -c "ncu --target-processes all --replay-mode kernel --clock-control none \
        --metrics $NCU_METRICS --csv --log-file /out/alone.csv \
        python3 real_l1.py alone 20 3" 2>&1 | tail -3 | tee -a "$LOG" || true
fi

# coloc with NeuralRx
if [[ -f "$RESULTS/C1_ncu/coloc_nrx.csv" ]]; then log "  skip C1-coloc_nrx"; else
  log "  C1-coloc with NeuralRx (warmup 90s)"
  ai_bg "ai_ncu_nrx" "$L1_UUID" /experiments/run_neural_rx_stress.py 0 1200
  sleep 90
  if [[ "$(docker inspect -f '{{.State.Status}}' ai_ncu_nrx 2>/dev/null)" == "running" ]]; then
    docker run --rm --gpus "\"device=$L1_UUID\"" \
      -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v $RESULTS/C1_ncu:/out \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
      -w /scripts $IMG bash -c "ncu --target-processes all --replay-mode kernel --clock-control none \
          --metrics $NCU_METRICS --csv --log-file /out/coloc_nrx.csv \
          python3 real_l1.py coloc_nrx 20 3" 2>&1 | tail -3 | tee -a "$LOG" || true
  else
    log "  NeuralRx died — skip C1-coloc"
  fi
  kill_all_ai
fi

###### C2: NSYS alone + coloc ######
log "=== C2 NSYS ==="
if [[ -f "$RESULTS/C2_nsys/alone.nsys-rep" ]]; then log "  skip C2-alone"; else
  log "  C2-alone (ITERS=30 for NSYS)"
  docker run --rm --gpus "\"device=$L1_UUID\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v $RESULTS/C2_nsys:/out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
    -w /scripts $IMG bash -c "nsys profile --trace=cuda --output=/out/alone --force-overwrite=true --stats=false \
        python3 real_l1.py alone 20 30" 2>&1 | tail -3 | tee -a "$LOG" || true
fi

if [[ -f "$RESULTS/C2_nsys/coloc_nrx.nsys-rep" ]]; then log "  skip C2-coloc_nrx"; else
  log "  C2-coloc NeuralRx (warmup 90s)"
  ai_bg "ai_nsys_nrx" "$L1_UUID" /experiments/run_neural_rx_stress.py 0 1200
  sleep 90
  if [[ "$(docker inspect -f '{{.State.Status}}' ai_nsys_nrx 2>/dev/null)" == "running" ]]; then
    docker run --rm --gpus "\"device=$L1_UUID\"" \
      -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v $RESULTS/C2_nsys:/out \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
      -w /scripts $IMG bash -c "nsys profile --trace=cuda --output=/out/coloc_nrx --force-overwrite=true --stats=false \
          python3 real_l1.py coloc_nrx 20 30" 2>&1 | tail -3 | tee -a "$LOG" || true
  fi
  kill_all_ai
fi

###############################################################
# Phase 2 — chanpred coloc on 2g, 4g, 7g
###############################################################
for size in 2g 4g 7g; do
  outdir=$RESULTS/A3_chanpred_coloc/$size
  mkdir -p "$outdir"
  if skip_if_done "$outdir" 10; then log "skip A3-$size (already have ≥10)"; continue; fi
  mig_reconfig "$(profile_for $size)" "Phase2 single $size for A3 chanpred coloc"
  UUID=$(mig_uuids | head -1)
  log "  $size UUID=$UUID"
  ai_bg "ai_a3_cp" "$UUID" /experiments/run_channel_prediction.py 0 1200
  log "  chanpred coloc on $size, warming 12s"
  sleep 12
  for i in $(seq 1 10); do
    run_l1_once "A3_${size}_coloc_run${i}" "$UUID" "$outdir"
  done
  kill_all_ai
done

###############################################################
# Phase 3 — four-way bigL1: 4g L1 + 1g + 1g + 1g
###############################################################
log "=== B1 four-way bigL1 (4g + 1g+1g+1g) ==="
mig_reconfig "5,19,19,19" "Phase3 four-way-bigL1"
L1_UUID=$(mig_uuids | head -1)        # 4g first
AI1_UUID=$(mig_uuids | sed -n '2p')
AI2_UUID=$(mig_uuids | sed -n '3p')
AI3_UUID=$(mig_uuids | sed -n '4p')
log "  L1(4g)=$L1_UUID  AI1(1g)=$AI1_UUID  AI2(1g)=$AI2_UUID  AI3(1g)=$AI3_UUID"

# B1-a: 4g L1 alone (just to compare with E5-4g)
outdir=$RESULTS/B1_four_way/alone_4g
mkdir -p "$outdir"
if ! skip_if_done "$outdir" 5; then
  for i in 1 2 3 4 5; do
    run_l1_once "B1_alone_4g_run${i}" "$L1_UUID" "$outdir"
  done
fi

# B1-b: 4g L1 + 3 chanpred (each on its own 1g)
outdir=$RESULTS/B1_four_way/three_chanpred
mkdir -p "$outdir"
if ! skip_if_done "$outdir" 5; then
  ai_bg "ai_b1_cp1" "$AI1_UUID" /experiments/run_channel_prediction.py 0 1200
  ai_bg "ai_b1_cp2" "$AI2_UUID" /experiments/run_channel_prediction.py 0 1200
  ai_bg "ai_b1_cp3" "$AI3_UUID" /experiments/run_channel_prediction.py 0 1200
  log "  3 chanpred on 1g×3, warming 15s"
  sleep 15
  for i in 1 2 3 4 5; do
    run_l1_once "B1_3chanpred_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
fi

# B1-c: 4g L1 + heterogeneous (chanpred + resnet + neuralrx, each on a 1g)
# NOTE: NeuralRx needs ≥ ~2GB; 1g.5gb may be too small. Try, accept failure.
outdir=$RESULTS/B1_four_way/het_cp_rn_nrx
mkdir -p "$outdir"
if ! skip_if_done "$outdir" 5; then
  ai_bg "ai_b1_cp"   "$AI1_UUID" /experiments/run_channel_prediction.py 0 1200
  ai_bg "ai_b1_rn"   "$AI2_UUID" /experiments/run_resnet_stress.py 0 1200 32 fp16
  ai_bg "ai_b1_nrx"  "$AI3_UUID" /experiments/run_neural_rx_stress.py 0 1200
  log "  het: chanpred + resnet + neuralrx on 1g×3 — warming 90s for NeuralRx"
  sleep 90
  if [[ "$(docker inspect -f '{{.State.Status}}' ai_b1_nrx 2>/dev/null)" != "running" ]]; then
    log "  WARNING: NeuralRx died on 1g (likely 5GB too tight) — continuing without it"
  fi
  for i in 1 2 3 4 5; do
    run_l1_once "B1_het_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
fi

###############################################################
# Phase 5 — 5-min sustained on restored split-60-40
###############################################################
mig_reconfig "5,9" "restore split-60-40 for sustained"
L1_UUID=$(mig_uuids | sed -n '2p')   # 3g
AI_UUID=$(mig_uuids | sed -n '1p')   # 4g
log "  Restored L1(3g)=$L1_UUID  AI(4g)=$AI_UUID"

# helper: long L1 run bounded by MAX_SECONDS=300
run_l1_long(){
  local label=$1 uuid=$2 outdir=$3
  docker run --rm --gpus "\"device=$uuid\"" \
    -v $AERIAL:/opt/nvidia/cuBB -v $SCRIPTS:/scripts -v "$outdir":/out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e RESULTS_DIR=/out \
    -e MAX_SECONDS=300 \
    -w /scripts $IMG python3 real_l1.py "$label" 20 7500 2>&1 | tail -1 | tee -a "$LOG"
}

declare -a SUSTAINED_SCENARIOS=(
  "alone:none"
  "chanpred_cross:run_channel_prediction.py"
  "neuralrx_cross:run_neural_rx_stress.py"
  "chanpred_coloc:run_channel_prediction.py"
  "neuralrx_coloc:run_neural_rx_stress.py"
)

for scen in "${SUSTAINED_SCENARIOS[@]}"; do
  IFS=":" read -r name ai_script <<< "$scen"
  outdir=$RESULTS/C3_sustained/$name
  mkdir -p "$outdir"
  if skip_if_done "$outdir" 2; then log "  skip C3-$name"; continue; fi
  log "=== C3 sustained: $name (n=2 × 5min each) ==="
  if [[ "$ai_script" != "none" ]]; then
    case "$name" in
      *_coloc) ai_uuid="$L1_UUID" ;;
      *_cross) ai_uuid="$AI_UUID" ;;
    esac
    ai_bg "ai_sustained" "$ai_uuid" "/experiments/$ai_script" 0 3600
    sleep 12
    if [[ "$ai_script" == "run_neural_rx_stress.py" ]]; then
      sleep 78  # extra 78s for NeuralRx TRT engine build (total ~90s)
    fi
  fi
  for i in 1 2; do
    run_l1_long "C3_${name}_run${i}" "$L1_UUID" "$outdir"
  done
  kill_all_ai
done

###############################################################
# Final summary
###############################################################
log "===== writing combined summary ====="
python3 - <<PYEOF | tee -a "$LOG"
import json, glob, statistics, os
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
labels = [
    ("E0 baseline 3g cross",      f"{R}/E0_baseline_3g/realL1_*.json"),
    ("E1 cross + NeuralRx",       f"{R}/E1_neuralrx/realL1_*.json"),
    ("E2 cross + chanpred",       f"{R}/E2_chanpred/realL1_*.json"),
    ("E3 coloc 3g + chanpred",    f"{R}/E3_coloc/realL1_*.json"),
    ("A1 cross + chanpred x4",    f"{R}/A1_stacking/chanpred_x4/realL1_*.json"),
    ("A1 cross + resnet x2",      f"{R}/A1_stacking/resnet_x2/realL1_*.json"),
    ("A1 cross + kitchen",        f"{R}/A1_stacking/kitchen/realL1_*.json"),
    ("A2 mixed coloc (cp+rn)",    f"{R}/A2_mixed_coloc/chanpred_resnet/realL1_*.json"),
    ("B3 ResNet b16 coloc",       f"{R}/B3_resnet_batch/b16/realL1_*.json"),
    ("B3 ResNet b64 coloc",       f"{R}/B3_resnet_batch/b64/realL1_*.json"),
    ("B3 ResNet b256 coloc",      f"{R}/B3_resnet_batch/b256/realL1_*.json"),
    ("A3 chanpred coloc 2g",      f"{R}/A3_chanpred_coloc/2g/realL1_*.json"),
    ("A3 chanpred coloc 4g",      f"{R}/A3_chanpred_coloc/4g/realL1_*.json"),
    ("A3 chanpred coloc 7g",      f"{R}/A3_chanpred_coloc/7g/realL1_*.json"),
    ("B1 four-way alone 4g",      f"{R}/B1_four_way/alone_4g/realL1_*.json"),
    ("B1 four-way 3 chanpred",    f"{R}/B1_four_way/three_chanpred/realL1_*.json"),
    ("B1 four-way het cp+rn+nrx", f"{R}/B1_four_way/het_cp_rn_nrx/realL1_*.json"),
    ("C3 sustained alone",        f"{R}/C3_sustained/alone/realL1_*.json"),
    ("C3 sustained cross chanpred",f"{R}/C3_sustained/chanpred_cross/realL1_*.json"),
    ("C3 sustained cross NeuralRx",f"{R}/C3_sustained/neuralrx_cross/realL1_*.json"),
    ("C3 sustained coloc chanpred",f"{R}/C3_sustained/chanpred_coloc/realL1_*.json"),
    ("C3 sustained coloc NeuralRx",f"{R}/C3_sustained/neuralrx_coloc/realL1_*.json"),
]
for desc, pat in labels:
    s = stats(pat)
    if s: print("{:42s} | {:5d} | {:6.1f} | {:6.1f} | {:6.1f} | {:6.1f}".format(desc, *s))
    else: print("{:42s} | NO DATA".format(desc))
PYEOF

log "===== CHAIN-ALL DONE ====="
