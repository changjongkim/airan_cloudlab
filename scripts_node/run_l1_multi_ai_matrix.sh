#!/usr/bin/env bash
# Multi-AI scenarios: L1 latency under simultaneous AI co-tenants.
# Covers the gap in M1-M4 (which used qwen_small only — launch-overhead bound so
# masks contention). Here we use sat_compute / sat_hbm / heterogeneous combos.
#
# Layout = MIG cgi spec.
# L1 always on the largest partition.
# AI workloads pinned to remaining partitions and run concurrently.
#
# Per cell: N iterations of {launch AIs in bg, sleep warmup, run L1, wait AIs}.
# Output:
#   results/$DATE_DIR/l1_multi_ai/<tag>/run_<i>_l1.log + ai_<k>.log
#   results/$DATE_DIR/l1_multi_ai/<tag>/summary.txt

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-5}"
DURATION="${DURATION:-40}"   # AI runs slightly longer than L1 measurement window
CELLS_L1="${CELLS_L1:-20}"
ITERS_L1="${ITERS_L1:-30}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"
HOST_UID=$(id -u); HOST_GID=$(id -g)

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/l1_multi_ai"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# AI workload → docker run argument (Python script + args)
ai_script_args() {
  local wl=$1 alloc=$2 gemm=$3
  case "$wl" in
    qwen_small)   echo "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py 0 $DURATION" ;;
    chanpred)     echo "/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py 0 $DURATION" ;;
    xapp)         echo "/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py 0 $DURATION" ;;
    resnet)       echo "/workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 $DURATION ${alloc:-32}" ;;
    forecaster)   echo "/workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 $DURATION ${alloc:-64} ${gemm:-384}" ;;
    sat_compute)  echo "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py 0 $DURATION $alloc $gemm" ;;
    sat_hbm)      echo "/workspace/AIRAN_Changjong/experiments/run_hbm_saturated.py 0 $DURATION $alloc" ;;
  esac
}

reconfigure_mig() {
  local cgi=$1
  sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -cgi "$cgi" -C >/dev/null 2>&1
  sleep 2
}

uuids_by_profile() {
  # Print all MIG UUIDs matching profile, one per line.
  local target=$1
  nvidia-smi -L | grep -E "MIG[[:space:]]+${target}[[:space:]]" \
    | grep -oE "MIG-[a-f0-9-]{36}"
}

# Spawn one AI container (detached). Echo CID.
launch_ai() {
  local ai_uuid=$1 ai_logfile=$2 script_args=$3
  docker run -d \
    --gpus "\"device=$ai_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    --name "ai_$$_$(date +%s)_$RANDOM" \
    "$IMAGE" bash -c "python3 $script_args > /tmp/ai.log 2>&1; tail -20 /tmp/ai.log"
  # caller saves docker logs $cid > $ai_logfile after wait
}

run_l1_once() {
  local l1_uuid=$1 outfile=$2 tag=$3
  docker run --rm \
    --user "$HOST_UID:$HOST_GID" \
    --gpus "\"device=$l1_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp \
    -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -w /scripts \
    "$IMAGE" \
    python3 real_l1.py "$tag" "$CELLS_L1" "$ITERS_L1" > "$outfile" 2>&1
}

# ----------------------------------------------------------------------------
# Cell definition format:
#   tag | cgi | l1_profile | ai_def[;ai_def;...]
# ai_def format: profile:workload[:alloc[:gemm]]
# ----------------------------------------------------------------------------
CELLS=(
  # M5: 3g+2g+2g layout, L1 on 3g, AI on both 2g
  "M5a_3wbal_sat_compute | 9,14,14  | 3g.20gb | 2g.10gb:sat_compute:8.5:8192;2g.10gb:sat_compute:8.5:8192"
  "M5b_3wbal_sat_hbm     | 9,14,14  | 3g.20gb | 2g.10gb:sat_hbm:8.5:0;2g.10gb:sat_hbm:8.5:0"
  "M5c_3wbal_het_resnet_cp | 9,14,14 | 3g.20gb | 2g.10gb:resnet:64:0;2g.10gb:chanpred:0:0"
  # M6: 4g+2g+1g layout, L1 on 4g
  "M6a_3wasym_sat        | 5,14,19  | 4g.20gb | 2g.10gb:sat_compute:8.5:8192;1g.5gb:sat_compute:4.0:4096"
  "M6b_3wasym_het        | 5,14,19  | 4g.20gb | 2g.10gb:chanpred:0:0;1g.5gb:xapp:0:0"
  # M7: 4g+1g+1g+1g layout, L1 on 4g
  "M7a_4w_sat_compute    | 5,19,19,19 | 4g.20gb | 1g.5gb:sat_compute:4.0:4096;1g.5gb:sat_compute:4.0:4096;1g.5gb:sat_compute:4.0:4096"
  "M7b_4w_het_3AI        | 5,19,19,19 | 4g.20gb | 1g.5gb:chanpred:0:0;1g.5gb:xapp:0:0;1g.5gb:sat_compute:4.0:4096"
  # Diverse AI-RAN realistic combos (with resnet + forecaster)
  "M8a_3wbal_resnet_fore | 9,14,14    | 3g.20gb | 2g.10gb:resnet:64:0;2g.10gb:forecaster:64:384"
  "M8b_3wasym_resnet_fore| 5,14,19    | 4g.20gb | 2g.10gb:resnet:64:0;1g.5gb:forecaster:32:256"
  "M8c_4w_diverse_AIRAN  | 5,19,19,19 | 4g.20gb | 1g.5gb:resnet:32:0;1g.5gb:forecaster:32:256;1g.5gb:chanpred:0:0"
)

run_cell() {
  local line=$1
  IFS='|' read -r tag cgi l1_profile ai_defs <<< "$line"
  tag=$(echo $tag | xargs); cgi=$(echo $cgi | xargs); l1_profile=$(echo $l1_profile | xargs); ai_defs=$(echo $ai_defs | xargs)

  log "===== CELL $tag (cgi=$cgi L1=$l1_profile) ====="
  local outdir="$OUT_ROOT/$tag"; mkdir -p "$outdir/alone" "$outdir/multi" && chmod 777 "$outdir" "$outdir/alone" "$outdir/multi"

  reconfigure_mig "$cgi"
  local l1_uuid; l1_uuid=$(uuids_by_profile "$l1_profile" | head -1)
  if [[ -z "$l1_uuid" ]]; then log "  ERROR: L1 UUID not found for $l1_profile"; return 1; fi
  log "  L1 UUID: $l1_uuid"

  # Parse AI defs and claim UUIDs, accounting for duplicate profiles.
  IFS=';' read -ra ai_entries <<< "$ai_defs"
  declare -A profile_used
  declare -a ai_uuids ai_wls ai_allocs ai_gemms
  local idx=0
  for entry in "${ai_entries[@]}"; do
    IFS=':' read -r profile wl alloc gemm <<< "$entry"
    profile=$(echo $profile | xargs)
    used_n=${profile_used[$profile]:-0}
    new_n=$((used_n + 1))
    profile_used[$profile]=$new_n
    uuid=$(uuids_by_profile "$profile" | sed -n "${new_n}p")
    if [[ -z "$uuid" ]]; then log "  ERROR: no $profile #$new_n for AI $wl"; return 1; fi
    ai_uuids[$idx]=$uuid
    ai_wls[$idx]=$wl
    ai_allocs[$idx]=${alloc:-0}
    ai_gemms[$idx]=${gemm:-0}
    log "  AI[$idx] = $wl on $profile uuid=$uuid (alloc=${alloc:-0} gemm=${gemm:-0})"
    idx=$((idx + 1))
  done
  local n_ai=$idx

  # ALONE pass: L1 alone, no AI co-tenant
  log "  --- ALONE pass (N=$N) ---"
  for i in $(seq 1 $N); do
    run_l1_once "$l1_uuid" "$outdir/alone/run_${i}.log" "${tag}_alone_run${i}"
    grep -E "mean=|p99=" "$outdir/alone/run_${i}.log" | tail -1
  done

  # MULTI pass: L1 + all AIs concurrent
  log "  --- MULTI pass (N=$N) ---"
  for i in $(seq 1 $N); do
    # Launch AIs in background, capture CIDs
    declare -a cids
    for k in $(seq 0 $((n_ai - 1))); do
      args=$(ai_script_args "${ai_wls[$k]}" "${ai_allocs[$k]}" "${ai_gemms[$k]}")
      cid=$(launch_ai "${ai_uuids[$k]}" "$outdir/multi/run_${i}_ai${k}.log" "$args")
      cids[$k]=$cid
    done
    # Warmup: let AI containers reach steady state (model load + kernel cache)
    sleep 12
    # L1 measurement
    run_l1_once "$l1_uuid" "$outdir/multi/run_${i}_l1.log" "${tag}_multi_run${i}"
    grep -E "mean=|p99=" "$outdir/multi/run_${i}_l1.log" | tail -1
    # Wait for AI containers to finish their DURATION
    for cid in "${cids[@]}"; do
      docker wait "$cid" >/dev/null 2>&1 || true
      docker logs "$cid" > "$outdir/multi/run_${i}_ai_${cid:0:12}.log" 2>&1 || true
      docker rm -f "$cid" >/dev/null 2>&1 || true
    done
    sleep 3
  done

  log "  cell done: $tag"
}

START_TS=$(date +%s)
for cell in "${CELLS[@]}"; do
  run_cell "$cell" 2>&1 | tee -a /tmp/l1_multi_ai.log
done
END_TS=$(date +%s)
log "l1_multi_ai DONE elapsed=$((END_TS - START_TS))s = $(( (END_TS - START_TS)/60 ))m"
log "Results: $OUT_ROOT"
