#!/usr/bin/env bash
# Wait for Parts 3-7 done, run Part 2b, then Part 8, then final marker
set -uo pipefail
LOG=/users/sgkim/auto_pipeline_chain18_part2b.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] auto pipeline Part 2b+8 watcher START"
DEADLINE=$(( $(date +%s) + 24*3600 ))
while (( $(date +%s) < DEADLINE )); do
  [[ -f /users/sgkim/CHAIN18_DONE ]] && break
  sleep 120
done

echo "[$(date '+%F %T')] Parts 3-7 done; launching Part 2b"
cd /users/sgkim/cloudlab_aerial
bash ./run_chain18_part2b_ncu_mpsclient.sh 2>&1

echo "[$(date '+%F %T')] Part 2b done; launching Part 8"
bash ./run_chain18_part8_realistic_stack.sh 2>&1

echo "[$(date '+%F %T')] Part 8 done; touching CHAIN18_ALL_DONE"
touch /users/sgkim/CHAIN18_ALL_DONE
