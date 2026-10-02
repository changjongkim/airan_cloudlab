#!/usr/bin/env bash
# MU-MIMO profile (two UEs, 16QAM, 273 PRBs; NVlabs neural receiver as NeuralRx).
#   probe  both receivers on the whole slot pool, one GPU, alone
#   times  timed runs with every NeuralRx running to the end (rescue deadline 41.5 ms):
#          no AI, then AI of one unit size running continuously under a 70% share
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
ENGINE=${ENGINE:-/softwall_runtime/engines/nv/nrx_rt_273prb_2ue.trt}
DATA=${DATA:-$SOFTWALL_ROOT/run_state/backstop_slot/dataset_nv}
ITER=${ITER:-20}        # LDPC iterations of both receivers
MU="--weak-profile nv_mu2 --dataset $DATA --engine $ENGINE --conv-ldpc-iterations $ITER --nrx-ldpc-iterations $ITER"
cells=${CELLS:-24}
for step in "$@"; do
  case $step in
    probe)
      CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu.py --dataset $DATA --engine $ENGINE \
        --conv-iterations $ITER --ldpc-iterations $ITER \
        --output $SLOT_RESULTS/raw/mu_probe_${TAG:-v5}_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|rror|Traceback' | tail -5 ;;
    times_noai)
      base="--seed 82000 $MU ${EXTRA:-} --controller controller3.py --rescue-deadline-ms 41.5 --nrx-bound-ms 4.0 --nrx-max-cb-fail 40 --ai-admission 0 --ai-chunks 128,512,1024"
      SLOT_TAG=${TAG:-v5}n SLOT_PERIODS=2400 SLOT_AI_RATE=4 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED" ;;
    times)
      base="--seed 82000 $MU ${EXTRA:-} --controller controller3.py --rescue-deadline-ms 41.5 --nrx-bound-ms 4.0 --nrx-max-cb-fail 40 --ai-admission 0 --ai-chunks 128,512,1024"
      SLOT_TAG=${TAG:-v5}cn SLOT_PERIODS=2400 SLOT_AI_RATE=4 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
      for c in 128 512 1024; do
        SLOT_TAG=${TAG:-v5}c70f$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --static-mps-pct 70 --ai-force-chunk $c" SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
      done ;;
  esac
done
echo MU_CAL_DONE
