#!/usr/bin/env bash
# v8 = v7 on two more channels.  Preparation on one GPU: a UMi slot pool, the larger model on
# the high-correlation and UMi pools, and the cost of LDPC iterations in the conventional receiver.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
LARGE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
if [ ! -f $R/nv/rt_2ue_umi2.npz ]; then
  echo "=== generate UMi 4-6 dB $(date +%T)"
  bash scripts_for_node/backstop_slot/nv/run_nv_dataset.sh $R/nv/rt_2ue_umi2.npz --base-config nrx_rt.cfg --prbs 273 --num-tx 2 --channel UMi --ebno 4,5,6 --slots 100 --batch 10 --gpu 0 2>&1 | grep -E 'ebno [-0-9]|rror'
fi
shifter --image="$AERIAL_IMAGE" python3 scripts_for_node/backstop_slot/nv/nv_build_dataset.py --input $R/nv/rt_2ue_umi2.npz --ebno 4,5,6 --link $R/dataset_v1 --output $R/dataset_v8umi 2>&1 | tail -1
for pool in "high dataset_v5" "umi dataset_v8umi"; do
  set -- $pool
  echo "=== nrx_large on $1 $(date +%T)"
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu.py --dataset $R/$2 --engine $LARGE --conv-iterations 20 --ldpc-iterations 20 \
    --output $SLOT_RESULTS/raw/mu_probe_v8$1_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|rror' | python3 -c "
import sys, json
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: print(line.strip()[:300]); continue
    print({k: d[k] for k in ('tbs', 'conventional', 'neural', 'conventional_only', 'neural_only', 'neither', 'nrx_ms')})
    print('by slot failed CBs', {k: (v['failed_tbs'], v['rescued_tbs']) for k, v in d['rescue_by_failed_code_blocks_of_slot'].items()})
"
done
for spec in "strong_rank2 dataset_v1" "nv_mu2 dataset_nv"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_conv_cost.py --dataset $R/$2 --profile $1 \
    --output $SLOT_RESULTS/raw/conv_cost_$1_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|rror' | cut -c1-400
done
echo V8_PREP_DONE
