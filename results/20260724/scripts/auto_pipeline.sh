#!/usr/bin/env bash
# Autonomous overnight pipeline:
#   1. wait for chain14 to finish
#   2. run chain15
#   3. validate all captures → identify broken
#   4. retry-run failed captures (max 2 rounds)
#   5. final validate
#   6. mark session-complete
#
# Everything logged to /users/sgkim/auto_pipeline.log so user can inspect
# next morning. All experiments are idempotent — safe to run overnight.
set -uo pipefail

DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
RESULTS=/mydata/results/$DATE_DIR
LOG=/users/sgkim/auto_pipeline.log
SCRIPT=/users/sgkim/cloudlab_aerial

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=========================================================================="
log "Auto pipeline START"
log "=========================================================================="

# ─── 1. Wait for chain14 ───────────────────────────────────────────
if pgrep -f "run_chain14.sh" >/dev/null; then
  log "chain14 running — waiting for it to finish"
  while pgrep -f "run_chain14.sh" >/dev/null; do sleep 60; done
  log "chain14 finished at $(ts)"
fi

# Convert chain14 nsys-rep → sqlite (parallel-ish, ignore failures)
if [ -d "$RESULTS/chain14" ]; then
  log "converting chain14 nsys-rep → sqlite"
  docker run --rm -v "$RESULTS/chain14:/data" -w /data airan:25-3-final bash -c '
    for f in *.nsys-rep; do
      [ -f "${f%.nsys-rep}.sqlite" ] && continue
      nsys export -t sqlite -o "${f%.nsys-rep}.sqlite" --force-overwrite=true "$f" >/dev/null 2>&1 || true
    done' >>"$LOG" 2>&1
  log "chain14 sqlite conversion done"
fi

# ─── 2. Run chain15 ───────────────────────────────────────────────
log "starting chain15 batch sweep"
bash "$SCRIPT/run_chain15.sh" >>"$LOG" 2>&1
log "chain15 finished at $(ts)"

if [ -d "$RESULTS/chain15" ]; then
  log "converting chain15 nsys-rep → sqlite"
  docker run --rm -v "$RESULTS/chain15:/data" -w /data airan:25-3-final bash -c '
    for f in *.nsys-rep; do
      [ -f "${f%.nsys-rep}.sqlite" ] && continue
      nsys export -t sqlite -o "${f%.nsys-rep}.sqlite" --force-overwrite=true "$f" >/dev/null 2>&1 || true
    done' >>"$LOG" 2>&1
  log "chain15 sqlite conversion done"
fi

# ─── 3. Validate + retry loop ─────────────────────────────────────
for RETRY in 1 2; do
  log "----- validation round $RETRY -----"
  for CH in chain14 chain15; do
    [ -d "$RESULTS/$CH" ] || continue
    python3 "$SCRIPT/validate_chain.py" --chain-dir "$RESULTS/$CH" \
      --output /users/sgkim/${CH}_fail_r${RETRY}.txt >>"$LOG" 2>&1 || true
  done

  # Any failures?
  fail_count=$(cat /users/sgkim/chain*_fail_r${RETRY}.txt 2>/dev/null | grep -v '^MISSING' | wc -l)
  missing_count=$(cat /users/sgkim/chain*_fail_r${RETRY}.txt 2>/dev/null | grep '^MISSING' | wc -l)
  log "round $RETRY: $fail_count failed + $missing_count missing"

  if [ $fail_count -eq 0 ] && [ $missing_count -eq 0 ]; then
    log "no failures — validation clean, exiting retry loop"
    break
  fi

  if [ $RETRY -eq 2 ]; then
    log "retry cap reached — leaving failed captures for manual review"
    break
  fi

  # (Simple strategy: rerun the entire chain that had failures)
  # More sophisticated: parse failure list and generate targeted script
  # For now we leave the retry generation as a stub — user can invoke manually
  log "retry generation is a stub; skipping automated rerun (round $RETRY)"
  break
done

# ─── 4. Aggregate + summary JSON ──────────────────────────────────
log "generating summary JSONs"
for CH in chain14 chain15; do
  [ -d "$RESULTS/$CH" ] || continue
  python3 "$SCRIPT/aggregate_summary.py" --chain-dir "$RESULTS/$CH" \
    --output "$RESULTS/${CH}_summary.json" >>"$LOG" 2>&1 || true
done

# ─── 5. Disk + inventory ──────────────────────────────────────────
log "final inventory:"
du -sh "$RESULTS"/chain14 "$RESULTS"/chain15 2>>"$LOG" | tee -a "$LOG"
echo "  chain14 captures: $(ls "$RESULTS/chain14"/*.nsys-rep 2>/dev/null | wc -l)" | tee -a "$LOG"
echo "  chain15 captures: $(ls "$RESULTS/chain15"/*.nsys-rep 2>/dev/null | wc -l)" | tee -a "$LOG"

log "=========================================================================="
log "Auto pipeline DONE. To sync to laptop:"
log "  rsync -a sgkim@d8545-10s10505.wisc.cloudlab.us:$RESULTS/ ./results/$DATE_DIR/"
log "=========================================================================="
touch /users/sgkim/AUTO_PIPELINE_DONE
