#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:28:00
#SBATCH -A m1248_g
#SBATCH -J nomig_val02
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Quick validation: run the FULL 02 matrix once with tiny params to confirm
# every condition / workload executes end-to-end on the no-MIG node.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=4 ITERS=10 N=1 AI_DUR=25 NEURALRX_WAIT=70
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/F_nomig_validate"

bash 02_F_saturation_nomig.sh

echo "==================== VALIDATION SUMMARY ===================="
echo "JSON files produced:"
ls -1 "$RESULTS_DIR"/realL1_*.json 2>/dev/null | wc -l
echo "Conditions covered:"
ls -1 "$RESULTS_DIR"/realL1_*.json 2>/dev/null | sed -E 's#.*/realL1_##; s#_run[0-9]+_[0-9]+_[0-9]+\.json##' | sort -u
echo "AI workload logs with errors (Traceback/FAIL):"
grep -lE "Traceback|FAIL|Error" "$RESULTS_DIR"/*_ai.log 2>/dev/null || echo "  (none)"
