#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:20:00
#SBATCH -A m1248_g
#SBATCH -J nomig_val06
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Validate nsys execution: tiny alone + neuralrx, confirm .nsys-rep produced.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=4 ITERS=10 AI_DUR=200 NEURALRX_WAIT=60
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/nsys_validate"

# Run only the first two stages by trimming: reuse 06 but with tiny params.
# (06 runs alone, neuralrx, resnet; tiny params keep it well under 20 min.)
bash 06_nsys_percall_nomig.sh

echo "==================== NSYS VALIDATION SUMMARY ===================="
ls -la "$RESULTS_DIR"/nsys_*.nsys-rep 2>/dev/null
echo "count: $(ls -1 "$RESULTS_DIR"/nsys_*.nsys-rep 2>/dev/null | wc -l)"
echo "nsys errors:"; grep -iE "error|fail" "$RESULTS_DIR"/nsys.log 2>/dev/null | head || echo "  (none)"
