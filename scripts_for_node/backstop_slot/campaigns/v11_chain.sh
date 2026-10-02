#!/usr/bin/env bash
# v11: (a) the dynamic-share baseline with queue hand-over, (b) weak cells whose slots mix the
# three channels (low correlation 2-3 dB, high correlation 13-16 dB, UMi 4-6 dB), steady and
# changing load.  nrx_large rescue, both receivers 20 LDPC iterations, 16 cells, deadline 11.5 ms.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
export ITER=20 ENGINE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt D2=11.5 CELLS=16
export NRX_BOUND=7.6 TABLE="128:10.1,512:9.8,1024:10.3" GATING="conv=128+yield"
for step in "$@"; do
  case $step in
    probe)
      CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu.py --dataset $R/dataset_mix --engine $ENGINE --conv-iterations 20 --ldpc-iterations 20 \
        --output $SLOT_RESULTS/raw/mu_probe_mix_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|rror' | python3 -c "
import sys, json
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: print(line.strip()[:300]); continue
    print({k: d[k] for k in ('tbs', 'conventional', 'neural', 'conventional_only', 'neural_only', 'neither')})
    print('by slot failed CBs (failed TBs, rescued)', {k: (v['failed_tbs'], v['rescued_tbs']) for k, v in d['rescue_by_failed_code_blocks_of_slot'].items()})
" ;;
    dyn)   # dynamic share with hand-over, low-correlation pool, two load patterns
      ( export DATA=$R/dataset_nv KFAIL=20 PERIODS=8000 CONTROLLER=controller4.py REFS="n" SHARES=" " RATES="32" DYNS="10,30:0 10,50:0 10,30:400"
        TAG=ey EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode phased --activity-phase 800" bash run_state/backstop_slot/mu_main.sh
        TAG=ez EXTRA="--weak-fraction 0.25 --activity-prob 0.25 --activity-mode phased --activity-phase 400" bash run_state/backstop_slot/mu_main.sh ) ;;
    mix)   # mixed channels, full load
      DATA=$R/dataset_mix KFAIL=9 EXTRA="--weak-fraction 0.25" TAG=mx SHARES="10 20 30 50" RATES="16 32" bash run_state/backstop_slot/mu_main.sh ;;
    mixphase)  # mixed channels, load alternating between full and half every 2 s
      ( export DATA=$R/dataset_mix KFAIL=9 PERIODS=8000 CONTROLLER=controller4.py REFS="n" SHARES="10 30" RATES="32" DYNS="10,30:0 10,50:0 10,30:400"
        TAG=my EXTRA="--weak-fraction 0.25 --activity-prob 0.5 --activity-mode phased --activity-phase 800" bash run_state/backstop_slot/mu_main.sh ) ;;
  esac
done
echo V11_CHAIN_DONE
