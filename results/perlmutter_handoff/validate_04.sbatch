#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:25:00
#SBATCH -A m1248_g
#SBATCH -J nomig_val04
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Validate the RISKY part of priority 3: does ncu have GPU perf-counter
# permission on Perlmutter? (often admin-restricted -> ERR_NVGPUCTRPERM)
# Minimal: real_l1 alone, CELLS=2 ITERS=1, just confirm a CSV with DRAM metrics.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=2 ITERS=1
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/NCU_validate"
mkdir -p "$RESULTS_DIR"

IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
SCRIPTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/cloudlab_aerial"
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

date; nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1
echo "==== ncu permission test ===="
shifter --image="$IMG" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$RESULTS_DIR:/out" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
    --workdir=/scripts \
    bash -c "ncu --target-processes all --replay-mode kernel --clock-control none \
                 --metrics $METRICS --csv --log-file /out/NCU_perm_test.csv \
                 python3 real_l1.py NCU_perm_test $CELLS $ITERS" 2>&1 | tail -20

echo "==== CSV produced? ===="
ls -la "$RESULTS_DIR"/NCU_perm_test.csv 2>/dev/null && echo "--- first lines ---" && head -5 "$RESULTS_DIR"/NCU_perm_test.csv
echo "==== permission errors? ===="
grep -iE "ERR_NVGPUCTRPERM|permission|not permitted|--target" "$RESULTS_DIR"/../../*val04*.log 2>/dev/null | head || true
date
