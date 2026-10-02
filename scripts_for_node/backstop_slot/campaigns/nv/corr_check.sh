#!/usr/bin/env bash
# NVlabs neural receiver vs cuPHY at equal LDPC iterations when the two UEs' channels are
# spatially correlated (DoubleTDL medium / high), where linear MMSE detection is weakest.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot/nv
S=scripts_for_node/backstop_slot/nv
one() {   # name channel ebno-list
  local name=$1 ch=$2 ebno=$3
  if [ ! -f $R/$name.npz ]; then
    echo "=== generate $name $(date +%T)"
    bash $S/run_nv_dataset.sh $R/$name.npz --base-config nrx_rt.cfg --prbs 273 --num-tx 2 --channel $ch --ebno $ebno --slots ${SLOTS:-50} --batch 10 --gpu 0 2>&1 | grep -E 'ebno [-0-9]|rror'
  fi
  shifter --image="$AERIAL_IMAGE" python3 $S/nv_build_dataset.py --input $R/$name.npz --ebno $ebno --output $SOFTWALL_ROOT/run_state/backstop_slot/dataset_$name 2>&1 | tail -1
  echo "=== decode $name $(date +%T)"
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_ldpc.py --dataset $SOFTWALL_ROOT/run_state/backstop_slot/dataset_$name --profile nv_mu2 \
    --engine /softwall_runtime/engines/nv/nrx_rt_273prb_2ue.trt --iterations ${ITERS:-10,20,40} \
    --output $SLOT_RESULTS/raw/ldpc_check_${name}_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\||rror|Traceback|  File' | tail -40
}
for spec in "$@"; do
  case $spec in
    high)   one rt_2ue_high DoubleTDLhigh 2,3,4,5,6,7,8,9,10 ;;
    medium) one rt_2ue_med  DoubleTDLmedium 2,3,4,5,6,7 ;;
    umi)    one rt_2ue_umi  UMi 1,2,3,4,5,6 ;;
  esac
done
echo CORR_CHECK_DONE
