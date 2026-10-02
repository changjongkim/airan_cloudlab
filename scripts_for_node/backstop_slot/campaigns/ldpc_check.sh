#!/usr/bin/env bash
# Rescue by the neural receiver versus rescue by more LDPC iterations on the conventional LLRs.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
for spec in "$@"; do
  case $spec in
    mu)   data=$SOFTWALL_ROOT/run_state/backstop_slot/dataset_nv; profile=nv_mu2; engine=/softwall_runtime/engines/nv/nrx_rt_273prb_2ue.trt; count=0 ;;
    weak) data=$SLOT_DATA; profile=weak_rank1; engine=/softwall_runtime/engines/neural_rx_fp16_full.trt; count=${COUNT:-512} ;;
    *)    data=$SLOT_DATA; profile=$spec; engine=/softwall_runtime/engines/neural_rx_fp16_full.trt; count=${COUNT:-512} ;;
  esac
  echo "== $profile"
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_ldpc.py --dataset $data --profile $profile --engine $engine --count $count \
    --iterations ${ITERS:-10,20,40} --output $SLOT_RESULTS/raw/ldpc_check_${profile}_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\||rror|Traceback|  File' | tail -40
done
echo LDPC_CHECK_DONE
