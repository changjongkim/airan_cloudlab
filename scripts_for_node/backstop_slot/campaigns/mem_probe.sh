#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
( for i in $(seq 1 24); do sleep 5; echo "t=$((i*5)) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"; if [ $i -eq 13 ]; then nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sort -t, -k3 -n | awk -F, '{print $2, $3}' | sort | uniq -c | sort -rn | head -12; fi; done ) &
SLOT_TAG=memq SLOT_PERIODS=6000 SLOT_AI_RATE=8 SLOT_EXTRA="--seed 82000 --lanes-per-gpu 2 --nrx-bound-ms 4.3 --rescue-deadline-ms 11.5 --ai-admission 1 --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --static-mps-pct 100 --ai-dispatch global" \
  SLOT_MATRIX="c40:rescue_value:static" bash scripts_for_node/backstop_slot/run_matrix.sh 2>&1 | grep -E "^===|FAILED"
wait
