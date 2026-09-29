#!/usr/bin/env bash
# C177 memory probe (diagnostic; job 59066149 ran this script from run_state/c177_memprobe).
# Samples per-GPU memory while running: (1) the multi-GPU C176 pipeline for
# 600 periods, (2) the single-GPU pipeline for 60 periods, and (3) the
# single-GPU pipeline for 600 periods under a 6-minute limit.
set -u
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
OUT=$1
mkdir -p "$OUT"
S=scripts_for_node/softwall_same_gpu
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/mem_gpu.csv" &
S1=$!
( while true; do echo "== $(date +%s)"; nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader; sleep 2; done ) > "$OUT/mem_apps.log" 2>&1 &
S2=$!
phase() { echo "$1 $(date '+%Y/%m/%d %H:%M:%S')" >> "$OUT/phases.log"; }
phase "multi600 start"
SOFTWALL_C176_PREFIX=c177memmulti SOFTWALL_C176_RUNS="steady:backstop:natural" bash $S/run_c176_burst.sh
phase "multi600 end exit=$?"
phase "single60 start"
SOFTWALL_C177_ITERATIONS=60 SOFTWALL_C177_PREFIX=c177memsingle SOFTWALL_C177_RUNS="steady:backstop:natural" bash $S/run_c177_single_gpu.sh
phase "single60 end exit=$?"
phase "single600 start"
SOFTWALL_C177_PREFIX=c177memsingle600 SOFTWALL_C177_RUNS="steady:backstop:natural" timeout 360 bash $S/run_c177_single_gpu.sh
phase "single600 end exit=$?"
kill $S1 $S2
