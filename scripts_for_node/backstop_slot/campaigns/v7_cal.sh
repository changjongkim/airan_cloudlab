#!/usr/bin/env bash
# v7: the larger NVlabs model (nrx_large, 4.3 ms engine) as the rescue receiver, weak cells =
# two-UE MU-MIMO with low correlation (Eb/No 2-3 dB), both receivers 20 LDPC iterations.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_nv ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
for step in "$@"; do
  case $step in
    probe) TAG=v7 bash run_state/backstop_slot/mu_cal.sh probe ;;
    16) CELLS=16 TAG=v7s bash run_state/backstop_slot/mu_cal.sh times ;;
    24) CELLS=24 TAG=v7 bash run_state/backstop_slot/mu_cal.sh times ;;
  esac
done
echo V7_CAL_DONE
