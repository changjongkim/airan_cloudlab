#!/usr/bin/env bash
# Nsight Compute per-kernel metrics — detailed L2 cache, DRAM, SM utilization.
#
# Goal: prove F8 (MIG architectural overhead) by quantifying:
#   - L2 cache hit rate (slice fragmentation hypothesis)
#   - DRAM throughput (HBM bw saturation)
#   - SM throughput (compute-bound or not)
#   - Warp occupancy
#
# ncu inside container typically needs:
#   --cap-add=SYS_ADMIN  (for some metrics)
#   or compiled CUPTI access. Aerial container has it but check.
#
# Outputs:
#   results/$DATE_DIR/ncu/<scenario>.ncu-rep
#   results/$DATE_DIR/ncu/<scenario>_metrics.csv

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
ITERS_PROFILE="${ITERS_PROFILE:-5}"   # ncu is much slower per kernel, use few iters
CELLS="${CELLS:-20}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
SCRIPT_DIR="$HOME/cloudlab_aerial"

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ncu"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Key metrics for F8 hypothesis testing
METRICS=(
  "sm__throughput.avg.pct_of_peak_sustained_elapsed"         # SM saturation
  "dram__throughput.avg.pct_of_peak_sustained_elapsed"       # HBM bw saturation
  "l1tex__t_sectors_hit_rate.pct"                            # L1 cache hit
  "lts__t_sectors_hit_rate.pct"                              # L2 cache hit (slice frag test)
  "smsp__average_warps_active_per_cycle_pct"                 # Warp occupancy
  "smsp__inst_executed.sum"                                  # Total instructions
  "gpu__time_duration.sum"                                   # Kernel duration
)
METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

get_mig_uuid() {
  local size=$1
  local gpu=$2
  nvidia-smi -L | awk -v g=$gpu 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/' \
    | grep "$size" | head -1 | grep -oE 'MIG-[0-9a-f-]+'
}

run_ncu() {
  local scenario=$1
  local gpu_arg=$2

  log "[$scenario] ncu profile starting..."

  # ncu needs replay mode for multi-pass metrics. Use "kernel" replay (one kernel at a time).
  docker run --rm --gpus "$gpu_arg" \
    --cap-add=SYS_ADMIN \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$OUT_ROOT:/ncu_out" \
    "$IMAGE" bash -c "
      ncu \
        --target-processes all \
        --replay-mode kernel \
        --metrics ${METRIC_LIST} \
        --csv \
        --log-file /ncu_out/${scenario}_metrics.csv \
        -o /ncu_out/${scenario} \
        --force-overwrite \
        python3 /scripts/real_l1.py ${scenario} ${CELLS} ${ITERS_PROFILE}
    " 2>&1 | tee "$OUT_ROOT/${scenario}_run.log"

  log "[$scenario] done"
}

# Summary extraction for our F8 mechanism question
summarize_metrics() {
  local scenario=$1
  local csv="$OUT_ROOT/${scenario}_metrics.csv"
  if [[ ! -f "$csv" ]]; then return; fi

  echo ""
  echo "=== $scenario — key metrics summary ==="
  # Top 5 kernels by time
  python3 - <<PY 2>&1
import csv
import statistics
try:
  with open("$csv") as f:
    rows = list(csv.DictReader(f))
  if not rows: print("no rows"); exit()
  metric_keys = [k for k in rows[0].keys() if any(m in k for m in
    ["sm__throughput", "dram__throughput", "l1tex__t_sectors_hit",
     "lts__t_sectors_hit", "smsp__average_warps", "gpu__time_duration"])]
  print(f"Total kernels profiled: {len(rows)}")
  for mk in metric_keys[:10]:
    try:
      vals = [float(r[mk]) for r in rows if r[mk]]
      if vals:
        print(f"  {mk}: median={statistics.median(vals):.2f} mean={statistics.mean(vals):.2f}")
    except Exception as e:
      print(f"  {mk}: parse error")
except Exception as e:
  print(f"summarize error: {e}")
PY
}

# ============================================================
# Run scenarios (assumes appropriate MIG state already set)
# ============================================================
# Run before each scenario from outside:
#   - Scenario A (fullgpu): MIG off on GPU $GPU
#   - Scenario B (7g): MIG on, 7g instance
#   - Scenario C (3g): MIG on, split-60-40
#   - Scenario D (2g): MIG on, split-40-60

mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i $GPU)

if [[ "$mig_state" == "Disabled" ]]; then
  log "===== Scenario A (Full GPU, no MIG) ====="
  run_ncu "A_fullgpu" "all"
  summarize_metrics "A_fullgpu"
fi

UUID_7G=$(get_mig_uuid "7g.40gb" $GPU)
if [[ -n "$UUID_7G" ]]; then
  log "===== Scenario B (7g MIG single) ====="
  run_ncu "B_7g_mig" "\"device=$UUID_7G\""
  summarize_metrics "B_7g_mig"
fi

UUID_3G=$(get_mig_uuid "3g.20gb" $GPU)
if [[ -n "$UUID_3G" ]]; then
  log "===== Scenario C (3g L1 alone) ====="
  run_ncu "C_3g_alone" "\"device=$UUID_3G\""
  summarize_metrics "C_3g_alone"
fi

UUID_2G=$(get_mig_uuid "2g.10gb" $GPU)
if [[ -n "$UUID_2G" ]]; then
  log "===== Scenario D (2g L1 alone) ====="
  run_ncu "D_2g_alone" "\"device=$UUID_2G\""
  summarize_metrics "D_2g_alone"
fi

log "DONE — ncu profiles in $OUT_ROOT"
log "Compare scenarios — F8 hypotheses:"
log "  H_L2: lts__t_sectors_hit_rate.pct decreases with smaller partitions → L2 fragmentation confirmed"
log "  H_BW: dram__throughput < 100% in all → not HBM bw saturated → architectural cost"
log "  H_SM: sm__throughput < 100% in 3g/4g → not compute-bound → architectural cost"
