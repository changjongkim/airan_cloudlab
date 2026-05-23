#!/usr/bin/env bash
# Phase 3 — D1 partition cap separation + A baseline attempt.
#
# Addresses SENSITIVITY_EXPERIMENTS §D1 (split-40-60 leakage decomposition)
# and §L1 (full-GPU L1 alone baseline) from MASTER_SUMMARY.
#
# D1: split-40-60 + Qwen N=10 + split-40-60 alone N=10 → leakage = with_AI - alone
#     This separates "partition cap" (2g.10gb vs 3g.20gb difference)
#     from "AI leakage" (cross-partition interference).
#
# A baseline: L1 alone on full GPU (no-mig). Requires driver reset because
#     MIG mode toggle leaves driver in bad state.
#
# Total ~30-45 min.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR
N="${N:-10}"
DURATION="${DURATION:-30}"

LOG="$HOME/cloudlab_aerial/results/$DATE_DIR/phase3_master.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%H:%M:%S'; }
sec() { printf '\n========== [%s] %s ==========\n' "$(ts)" "$*" | tee -a "$LOG"; }

sec "PHASE 3 extras — DATE_DIR=$DATE_DIR, N=$N"

# ----------------------------------------------------------------------------
# D1a — split-40-60 + Qwen N=10 (with AI)
# ----------------------------------------------------------------------------
sec "D1a: split-40-60 + qwen7b N=$N"
N=$N PRESET=split-40-60 AI=qwen7b TAG=D1a_4060_qwen DMON=1 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# D1b — split-40-60 + none N=10 (L1 alone on 2g.10gb, neighbor idle)
# ----------------------------------------------------------------------------
sec "D1b: split-40-60 alone (no AI, just L1 on 2g.10gb) N=$N"
N=$N PRESET=split-40-60 AI=none TAG=D1b_4060_alone DMON=0 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# A baseline — Full-GPU L1 alone (no MIG)
# Requires MIG mode disable + driver reset for clean state.
# ----------------------------------------------------------------------------
sec "A baseline: disabling MIG and resetting driver"
sudo nvidia-smi mig -i 0 -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i 0 -dgi >/dev/null 2>&1 || true
sudo nvidia-smi -i 0 -mig 0 2>&1 | tee -a "$LOG"
sleep 2
bash ./driver_reset.sh 2>&1 | tee -a "$LOG"
sleep 5

sec "A baseline: L1 alone on full GPU N=$N"
N=$N PRESET=no-mig AI=none TAG=A_baseline_fullGPU DMON=0 DURATION=$DURATION \
  ./run_n20.sh 2>&1 | tee -a "$LOG"

# Re-enable MIG (so other experiments can continue if any)
sec "Re-enabling MIG mode (for any followup)"
sudo nvidia-smi -i 0 -mig 1 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
sec "Running bimodal_detect on phase 3 results"
for tag in D1a_4060_qwen D1b_4060_alone A_baseline_fullGPU; do
  dir="results/$DATE_DIR/n${N}_$tag"
  if [[ -d "$dir" ]]; then
    echo ""
    echo "----- $tag -----" | tee -a "$LOG"
    python3 "$HOME/bimodal_detect.py" "$dir" 2>&1 | tee -a "$LOG"
  fi
done

sec "PHASE 3 DONE"
grep "VERDICT:" "$LOG" | tail -3

# Compute D1 leakage decomposition
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import os, json, glob, statistics
date_dir = os.environ.get("DATE_DIR")
base = f"{os.path.expanduser('~')}/cloudlab_aerial/results/{date_dir}"
def mean_of(tag):
    files = sorted(glob.glob(f"{base}/n*_{tag}/run_*.json"))
    means = []
    for f in files:
        try:
            d = json.load(open(f))
            if d.get("mean_ms"): means.append(d["mean_ms"])
        except: pass
    return statistics.mean(means) if means else None

q = mean_of("D1a_4060_qwen")
a = mean_of("D1b_4060_alone")
b = mean_of("A_baseline_fullGPU")

print()
print("=== D1 Decomposition (partition cap vs AI leakage) ===")
if a and q:
    print(f"split-40-60 alone (2g.10gb L1):  {a:.2f} ms")
    print(f"split-40-60 + Qwen:              {q:.2f} ms")
    print(f"AI leakage (cross-partition):    {q-a:.2f} ms ({100*(q-a)/a:.1f}%)")
if b:
    print(f"A baseline (L1 alone full GPU):  {b:.2f} ms")
    if a: print(f"Partition cap (2g vs full):      {a-b:.2f} ms ({100*(a-b)/b:.1f}%)")
PY
