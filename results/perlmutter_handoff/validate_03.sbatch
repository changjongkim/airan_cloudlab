#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:28:00
#SBATCH -A m1248_g
#SBATCH -J nomig_val03
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Quick validation: MPS path works on Perlmutter (daemon + L1/AI connect) and
# all 7 MPS conditions produce JSON. Tiny params.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=4 ITERS=10 N=1 AI_DUR=120 NEURALRX_WAIT=40
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/F_nomig_mps_validate"

bash 03_mps_compare.sh

echo "==================== MPS VALIDATION SUMMARY ===================="
echo "JSON files:"; ls -1 "$RESULTS_DIR"/realL1_MPS_*.json 2>/dev/null | wc -l
echo "Conditions:"; ls -1 "$RESULTS_DIR"/realL1_MPS_*.json 2>/dev/null | sed -E 's#.*/realL1_MPS_##; s#_run[0-9]+_[0-9]+_[0-9]+\.json##' | sort -u
echo "AI logs with errors:"; grep -lE "Traceback|FAIL|Error" "$RESULTS_DIR"/*_ai.log 2>/dev/null || echo "  (none)"
echo "MPS pipe errors in L1/AI logs:"; grep -riE "MPS|pipe|cannot connect" "$RESULTS_DIR"/*_ai.log 2>/dev/null | head || echo "  (none)"
