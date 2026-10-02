#!/usr/bin/env bash
# v7: nrx_large (4.3 ms engine, 6.3 ms per rescue in a loaded run) rescues the failures of
# low-correlation two-UE MU-MIMO cells; both receivers 20 LDPC iterations; rescue deadline 11.5 ms.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_nv ITER=20 D2=${D2:-11.5} ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
for spec in "$@"; do
  case $spec in
    16q)  # a quarter of the cells weak (one MU-MIMO cell per GPU)
      CELLS=16 TAG=w EXTRA="--weak-fraction 0.25" KFAIL=20 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="conv=128+yield" SHARES="30 50 70 100" RATES="16 32" bash run_state/backstop_slot/mu_main.sh ;;
    16h)  # half of the cells weak: more failures than the lanes can take; which ones to rescue?
      ( export CELLS=16 NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="conv=128+yield" RATES="32" SHARES="50"
        TAG=hall KFAIL=20 REFS="n" bash run_state/backstop_slot/mu_main.sh
        TAG=hval KFAIL=20 REFS="n" NRXFLAGS="--nrx-flags rank=value" bash run_state/backstop_slot/mu_main.sh
        TAG=hk3 KFAIL=3 REFS="n" NRXFLAGS="--nrx-flags rank=value" bash run_state/backstop_slot/mu_main.sh
        TAG=hk9 KFAIL=9 REFS="n" NRXFLAGS="--nrx-flags rank=value" bash run_state/backstop_slot/mu_main.sh ) ;;
  esac
done
echo V7_CHAIN_DONE
