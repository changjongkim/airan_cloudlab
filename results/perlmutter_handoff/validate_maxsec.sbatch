#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:10:00
#SBATCH -A m1248_g
#SBATCH -J nomig_maxsec
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Validate the real_l1.py MAX_SECONDS change: (1) cap works, (2) default path
# (no cap) is unchanged so the other 4 queued jobs are unaffected.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff
RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/maxsec_validate"
mkdir -p "$RESULTS_DIR"
IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
SCRIPTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/cloudlab_aerial"

runL1() { # extra-env  label  cells  iters
  local env=$1 label=$2 cells=$3 iters=$4
  shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SCRIPTS_DIR:/scripts" --volume="$RESULTS_DIR:/out" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out $env \
    --workdir=/scripts python3 real_l1.py "$label" "$cells" "$iters" 2>&1 | tail -2
}

date
echo "==== [1] CAPPED: MAX_SECONDS=8, ITERS=100000 (should stop ~8s, iters<<100000) ===="
SECONDS=0; runL1 "--env=MAX_SECONDS=8" maxsec_capped 4 100000; echo "wall=${SECONDS}s"

echo "==== [2] DEFAULT (no cap), ITERS=20 (regression: must run full 20) ===="
runL1 "" maxsec_default 4 20

echo "==== iterations recorded ===="
for f in "$RESULTS_DIR"/realL1_maxsec_*.json; do
  python3 -c "import json;d=json.load(open('$f'));print('  ',d['label'],'iterations=',d['iterations'])"
done
date
