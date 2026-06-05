#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH -q debug
#SBATCH -t 0:20:00
#SBATCH -A m1248_g
#SBATCH -J nomig_val04b
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# ncu blocked by DCGM holding the perf counters. Test the standard workaround on
# an EXCLUSIVE node: pause DCGM (host-level) -> run ncu -> resume.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=2 ITERS=1
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/NCU_validate"
mkdir -p "$RESULTS_DIR"
IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
SCRIPTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/cloudlab_aerial"
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

date; hostname; nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1

echo "==== dcgmi present? ===="
which dcgmi 2>&1 || echo "dcgmi NOT in PATH"

echo "==== try: dcgmi profile --pause (host) ===="
dcgmi profile --pause 2>&1; echo "pause exit=$?"

echo "==== ncu test (counters should be free now) ===="
shifter --image="$IMG" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$RESULTS_DIR:/out" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
    --workdir=/scripts \
    bash -c "ncu --target-processes all --replay-mode kernel --clock-control none \
                 --metrics $METRICS --csv --log-file /out/NCU_perm_test2.csv \
                 python3 real_l1.py NCU_perm_test2 $CELLS $ITERS" 2>&1 | tail -15

echo "==== dcgmi profile --resume ===="
dcgmi profile --resume 2>&1; echo "resume exit=$?"

echo "==== CSV with real metric values? ===="
grep -iE "dram__throughput|sm__warps" "$RESULTS_DIR"/NCU_perm_test2.csv 2>/dev/null | head -3 || echo "no metric rows"
date
