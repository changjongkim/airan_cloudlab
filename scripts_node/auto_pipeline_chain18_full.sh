#!/usr/bin/env bash
# Auto-pipeline: wait for Part 2 (currently running) then run Parts 3-7 sequentially
set -uo pipefail
LOG=/users/sgkim/auto_pipeline_chain18.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] auto pipeline chain18 START"

# Wait until run_chain18_part2 process no longer exists (or marker appears).
DEADLINE=$(( $(date +%s) + 4*3600 ))
while (( $(date +%s) < DEADLINE )); do
  if [[ -f /users/sgkim/CHAIN18_PART2_DONE ]]; then break; fi
  if ! pgrep -f run_chain18_part2_ncu_fullgpu.sh >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] Part 2 process not found; assuming complete"
    touch /users/sgkim/CHAIN18_PART2_DONE
    break
  fi
  sleep 60
done

echo "[$(date '+%F %T')] launching Parts 3-7"
cd /users/sgkim/cloudlab_aerial
bash ./run_chain18_parts3to7.sh 2>&1

echo "[$(date '+%F %T')] Parts 3-7 finished; touching CHAIN18_DONE marker"
touch /users/sgkim/CHAIN18_DONE
