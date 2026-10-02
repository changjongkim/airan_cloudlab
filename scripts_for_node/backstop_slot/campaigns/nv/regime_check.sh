#!/usr/bin/env bash
# Wider search for conditions where a public neural receiver beats cuPHY at equal LDPC iterations.
#   large   nrx_large (the bigger NVlabs model) exported at 273 PRBs, 2 UEs, then decoded on the
#           low-correlation, UMi and high-correlation pools
#   med     medium correlation at higher Eb/No (nrx_rt)
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
ROOT=$SOFTWALL_ROOT
E=$SOFTWALL_RUNTIME/engines/nv
decode() {   # name dataset engine-stem
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_ldpc.py --dataset $ROOT/run_state/backstop_slot/$2 --profile nv_mu2 \
    --engine /softwall_runtime/engines/nv/$3.trt --iterations ${ITERS:-10,20,40} \
    --output $SLOT_RESULTS/raw/ldpc_check_$1_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\| (conv|nrx|receiver)|rror|Traceback' | tail -12
}
for step in "$@"; do
  case $step in
    large)
      if [ ! -f $E/nrx_large_273prb_2ue.trt ]; then
        echo "=== export nrx_large $(date +%T)"
        ( cd $ROOT/third_party/neural_rx/scripts && shifter --module=gpu --image="$AERIAL_IMAGE" \
            --env=PYTHONPATH=$ROOT/runtime/softwall_same_gpu/nrx_export_deps --env=PYTHONNOUSERSITE=1 \
            python3 $ROOT/scripts_for_node/backstop_slot/nv/nv_export.py --base-config nrx_large.cfg --prbs 273 --num-tx 2 \
            --output-dir $E 2>&1 | grep -E '^\{|rror' | cut -c1-300 )
        bash scripts_for_node/backstop_slot/nv/build_nv_engine.sh nrx_large_273prb_2ue
      fi
      for pool in "low dataset_nv" "umi dataset_rt_2ue_umi" "high dataset_rt_2ue_high2"; do
        set -- $pool; echo "=== nrx_large on $1 $(date +%T)"; decode large_$1 $2 nrx_large_273prb_2ue
      done ;;
    med)
      SLOTS=${SLOTS:-50} EBNO=8,10,12,14,16,18 bash run_state/backstop_slot/nv/corr_check2.sh med2 2>&1 | grep -E '^\||===|rror' ;;
  esac
done
echo REGIME_CHECK_DONE
