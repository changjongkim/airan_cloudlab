#!/usr/bin/env bash
# Chain 18 PRIORITY 2h — kill current queue, run only Part 8+5+7stat+2b in order
set -uo pipefail

LOG=/users/sgkim/chain18_priority.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] ============ PRIORITY 2H CHAIN START ============"

# 1) Stop current chain gracefully
echo "[$(date '+%F %T')] Stopping current auto pipelines and running experiments"
pkill -f auto_pipeline_chain18_full.sh 2>/dev/null || true
pkill -f run_chain18_parts3to7.sh 2>/dev/null || true
pkill -f auto_pipeline_chain18_part2b.sh 2>/dev/null || true
# Give running containers time to finalize current trial
sleep 10
docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 5 2>/dev/null
sleep 3

# 2) Part 8 first (most important)
echo "[$(date '+%F %T')] === launching Part 8 (realistic stack) ==="
cd /users/sgkim/cloudlab_aerial
bash ./run_chain18_part8_realistic_stack.sh 2>&1

# 3) Part 5 (thread vs process) — extract from parts3to7 script
echo "[$(date '+%F %T')] === launching Part 5 only ==="
env PART_ONLY=5 bash ./run_chain18_parts5only.sh 2>&1 || echo "(Part 5 optional)"

# 4) Part 7 statistical only (10 trials at N=5,6,7)
echo "[$(date '+%F %T')] === launching Part 7 statistical only ==="
env STAT_ONLY=1 bash ./run_chain18_part7_stat_only.sh 2>&1 || echo "(Part 7 stat optional)"

# 5) Part 2b (NCU MPSon)
echo "[$(date '+%F %T')] === launching Part 2b ==="
bash ./run_chain18_part2b_ncu_mpsclient.sh 2>&1

# Signal complete
touch /users/sgkim/CHAIN18_ALL_DONE
echo "[$(date '+%F %T')] ============ PRIORITY 2H CHAIN DONE ============"
