#!/usr/bin/env bash
# Link adaptation, further Es/No points: as la.sh, with a suffix in the names so that the points
# of la.sh are kept.  Per MCS: slots from the NVlabs link simulation at every Es/No point, then
# both receivers on GPU 0 alone (20 LDPC iterations each, larger neural receiver).
#   usage: SUFFIX=_b MCS="12 13 14 15" bash la2.sh CHANNEL ESNO,ESNO,... [SLOTS per point]
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
LARGE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
channel=$1; points=$2; slots=${3:-150}; suffix=${SUFFIX:-}
mkdir -p $R/la $SLOT_RESULTS/raw
for mcs in ${MCS:-10 11 12 13 14 15 16}; do
  name=${channel}_m${mcs}${suffix}
  [ -f $SLOT_RESULTS/raw/la_${name}.json ] && { echo "=== $name (done)"; continue; }
  echo "=== $name $(date +%T)"
  if [ ! -f $R/la/$name.npz ]; then
    bash scripts_for_node/backstop_slot/nv/run_nv_dataset.sh $R/la/$name.npz --base-config nrx_large.cfg --prbs 273 --num-tx 2 \
      --channel $channel --mcs $mcs --esno $points --slots $slots --batch 10 --gpu 0 2>&1 | grep -E 'ebno [-0-9]|rror' | tail -4
  fi
  shifter --image="$AERIAL_IMAGE" python3 scripts_for_node/backstop_slot/nv/nv_build_dataset.py --input $R/la/$name.npz --ebno all \
    --profile nv_mu2_m$mcs --output $R/la/ds_$name 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_mu.py --dataset $R/la/ds_$name --profile nv_mu2_m$mcs --engine $LARGE \
    --conv-iterations 20 --ldpc-iterations 20 --passes 2 --output $SLOT_RESULTS/raw/la_${name}.json 2>&1 | grep -E 'rror|Traceback' | head -3
done
echo LA_DONE
