#!/usr/bin/env bash
# Local finalizer for chain17 — polls AUTO_PIPELINE_DONE, syncs, figures, commits.
set -uo pipefail

NODE=sgkim@d8545-10s10505.wisc.cloudlab.us
DATE_DIR=${DATE_DIR:-20260724}
LOCAL_RESULTS=/Users/changjongkim/New_research/cloudlab_results/results/$DATE_DIR
LOCAL_SCRIPTS=/Users/changjongkim/New_research/cloudlab_aerial
LOCAL_REPO=/Users/changjongkim/New_research/cloudlab_results
LOG=/tmp/local_finalize_chain17.log

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

log "== local_finalize_chain17 START =="

# Poll for AUTO_PIPELINE_DONE (max 24h)
DEADLINE=$(($(date +%s) + 86400))
while true; do
  if ssh -o ConnectTimeout=10 -o BatchMode=yes $NODE "test -f /users/sgkim/AUTO_PIPELINE_DONE" 2>/dev/null; then
    log "AUTO_PIPELINE_DONE detected"; break
  fi
  [[ $(date +%s) -gt $DEADLINE ]] && { log "24h timeout"; exit 1; }
  sleep 300
done

# Sync results (chain17 files ended up in 20260725 due to midnight boundary)
log "syncing results (both 20260724 and 20260725 date dirs)"
mkdir -p "$LOCAL_RESULTS"
rsync -a "$NODE:/mydata/results/20260724/" "$LOCAL_RESULTS/" >>"$LOG" 2>&1
# Merge 20260725 chain17 into local 20260724 for unified analysis
rsync -a "$NODE:/mydata/results/20260725/" "$LOCAL_RESULTS/" >>"$LOG" 2>&1 || true
log "results synced ($(du -sh $LOCAL_RESULTS | cut -f1))"

# Sync scripts
log "syncing scripts"
rsync -a "$NODE:/users/sgkim/cloudlab_aerial/" "$LOCAL_SCRIPTS/" >>"$LOG" 2>&1

# Sync logs
scp $NODE:/users/sgkim/chain17.log $NODE:/users/sgkim/chain17_ncu.log $NODE:/users/sgkim/chain17_wrapper.log "$LOCAL_RESULTS/" >>"$LOG" 2>&1 || true

# Convert chain17 nsys → sqlite (on node)
log "converting nsys → sqlite"
for d in chain17 chain17_ncu; do
  for date_d in 20260724 20260725; do
    ssh $NODE "if [ -d /mydata/results/$date_d/$d ]; then docker run --rm -v /mydata/results/$date_d/$d:/data -w /data airan:25-3-final bash -c 'for f in *.nsys-rep; do [ -f \${f%.nsys-rep}.sqlite ] || nsys export -t sqlite -o \${f%.nsys-rep}.sqlite --force-overwrite=true \"\$f\" >/dev/null 2>&1; done; ls *.sqlite 2>/dev/null | wc -l'; fi" >>"$LOG" 2>&1 || true
  done
done
# Re-sync after conversion (both date dirs)
rsync -a "$NODE:/mydata/results/20260724/" "$LOCAL_RESULTS/" >>"$LOG" 2>&1
rsync -a "$NODE:/mydata/results/20260725/" "$LOCAL_RESULTS/" >>"$LOG" 2>&1 || true

# Aggregate summary
log "generating summary JSONs"
for CH in chain17 chain17_ncu; do
  [[ -d "$LOCAL_RESULTS/$CH" ]] || continue
  python3 "$LOCAL_SCRIPTS/aggregate_summary.py" \
    --chain-dir "$LOCAL_RESULTS/$CH" \
    --output "$LOCAL_RESULTS/${CH}_summary.json" >>"$LOG" 2>&1 || true
done

# Copy scripts into results/scripts
log "copying scripts to results/scripts"
mkdir -p "$LOCAL_RESULTS/scripts"
cp "$LOCAL_SCRIPTS"/run_chain17*.sh "$LOCAL_SCRIPTS"/run_ranai_mix.py "$LOCAL_SCRIPTS"/local_finalize_chain17.sh "$LOCAL_RESULTS/scripts/" 2>/dev/null || true

# Git commit + push
log "committing"
cd "$LOCAL_REPO"
git add "results/$DATE_DIR/" >>"$LOG" 2>&1
git commit -m "Chain 17: sensitivity + low-level measurements (N-sweep + MPS thread% + NCU + DCGM)

Part A: N-process sensitivity sweep (nrx_multi{1,2,3,4,6,8}) × 3 configs × MPS on/off × 3 trials
Part B: MPS thread% cap sweep (100/70/50/30) × 7 sync-causing workloads × 3 configs × 3 trials
Part C: NCU low-level counters (DRAM/L2/SM utilization) on key workloads × MPS off/on
Part D: DCGM real-time monitoring (parallel with Part A + B)

Full auto-collected via chain17.sh → chain17_ncu.sh → AUTO_PIPELINE_DONE marker,
picked up by local_finalize_chain17.sh polling on mac.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>" >>"$LOG" 2>&1 || log "commit failed"
git push >>"$LOG" 2>&1 && log "pushed" || log "push failed"

log "== local_finalize_chain17 DONE =="
osascript -e 'display notification "Chain 17 done, pushed" with title "AI-RAN"' 2>/dev/null || true
