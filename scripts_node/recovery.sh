#!/usr/bin/env bash
# Morning recovery — diagnose state + run bootstrap if needed.
#
# Designed for 5/24 morning when 5/12 image snapshot is likely incomplete
# (only root preserved, /mydata reset).
#
# Outputs a clear branch decision: SKIP_SETUP, PARTIAL_SETUP, or FULL_BOOTSTRAP.

set -uo pipefail
mkdir -p /tmp/recovery
LOG=/tmp/recovery/diagnose_$(date +%H%M).log

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "===== MORNING RECOVERY DIAGNOSE ====="

# ----------------------------------------------------------------------------
# Diagnose
# ----------------------------------------------------------------------------
log ""
log "--- A: /mydata ---"
if [[ -d /mydata ]]; then
  df -h /mydata 2>&1 | tee -a "$LOG"
  ls /mydata/ 2>&1 | head -10 | tee -a "$LOG"
  mydata_size=$(du -sm /mydata 2>/dev/null | awk '{print $1}')
  log "/mydata size: ${mydata_size:-0} MB"
else
  log "/mydata MISSING"
  mydata_size=0
fi

log ""
log "--- B: Docker images ---"
docker images 2>&1 | tee -a "$LOG"
has_aerial=$(docker images | grep -c "aerial-cuda-accelerated-ran.*25-3-cubb" || echo 0)
has_airan=$(docker images | grep -c "airan:25-3" || echo 0)
log "aerial 25-3-cubb image: $has_aerial, airan:25-3 image: $has_airan"

log ""
log "--- C: cuPHY build ---"
has_cuphy_build=0
for p in /mydata/aerial-cuda-accelerated-ran/build.x86_64 \
         /mydata/aerial-cuda-accelerated-ran/build*; do
  if [[ -d "$p" ]] && [[ -f "$p/cuPHY/src/cuphy/libcuphy.so" ]]; then
    log "cuPHY build found at $p"
    has_cuphy_build=1
    break
  fi
done
[[ "$has_cuphy_build" -eq 0 ]] && log "cuPHY build MISSING"

log ""
log "--- D: HF cache ---"
has_qwen=0
for p in /mnt/dockerdata/hf_cache /mydata/hf_cache ~/.cache/huggingface ~/.cache/hf_cloudlab; do
  if [[ -d "$p" ]]; then
    sz=$(du -sm "$p" 2>/dev/null | awk '{print $1}')
    log "$p exists: ${sz} MB"
    if [[ "${sz:-0}" -gt 5000 ]]; then has_qwen=1; fi
  fi
done

log ""
log "--- E: Our scripts ---"
has_sweep=0
if [[ -f $HOME/cloudlab_aerial/master_5h_sweep.sh ]]; then has_sweep=1; fi
log "cloudlab_aerial/master_5h_sweep.sh: $has_sweep"
ls $HOME/cloudlab_aerial/*.sh 2>&1 | wc -l | xargs -I {} log "cloudlab_aerial/ shell count: {}"
ls $HOME/AIRAN_Changjong/experiments/run_*.py 2>&1 | wc -l | xargs -I {} log "experiments/ python count: {}"

log ""
log "--- F: MIG mode ---"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current --format=csv,noheader 2>&1 | tee -a "$LOG"
mig_state=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader -i 0)
log "MIG state: $mig_state"

# ----------------------------------------------------------------------------
# Branch decision
# ----------------------------------------------------------------------------
log ""
log "===== DECISION ====="

if [[ "$has_aerial" -gt 0 ]] && [[ "$has_airan" -gt 0 ]] && [[ "$has_cuphy_build" -eq 1 ]] && [[ "$has_qwen" -eq 1 ]]; then
  log "🟢 BEST — all artifacts present"
  log "ACTION: skip setup, run master_5h_sweep.sh immediately"
  echo "BEST"
elif [[ "$has_sweep" -eq 1 ]]; then
  log "🟡 PARTIAL — scripts OK but artifacts missing"
  log "ACTION: re-pull Aerial, rebuild cuPHY, then run master sweep"
  log "Estimated setup time: 60-90 min, then phase 1 (~2h) fits in window"
  echo "PARTIAL"
else
  log "🔴 WORST — even scripts missing"
  log "ACTION: full bootstrap from scratch"
  log "Estimated setup time: 2-3h, only phase 1 likely to fit"
  echo "WORST"
fi

log ""
log "Log saved: $LOG"
