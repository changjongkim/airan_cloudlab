#!/usr/bin/env bash
# Slots for the closed-loop link adaptation runs: one channel at one Es/No, one ring per MCS level.
# Output: one dataset directory (profiles nv_mu2_m<MCS>, single-user cells linked in) and, per level, the
# receiver comparison on those slots (results/backstop_slot/raw/lacl_<TAG>_m<MCS>.json and _slots.npz).
#   usage: bash la_cl_gen.sh TAG CHANNEL ESNO SLOTS "GPU:MCS MCS ..." ["GPU:MCS ..." ...]
#   e.g.   bash la_cl_gen.sh high16 DoubleTDLhigh 16 768 "0:13 16" "1:14 11" "2:12 10" "3:15"
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
LARGE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
tag=$1; channel=$2; esno=$3; slots=$4; shift 4
D=$R/dataset_la_$tag
mkdir -p $R/la_cl $D $SLOT_RESULTS/raw $R/logs
for path in $R/dataset_mix/strong_rank2_*; do [ -e $D/$(basename $path) ] || ln -s $(readlink -f $path) $D/$(basename $path); done
stream() {
  local gpu=$1 list=$2 mcs name
  for mcs in $list; do
    name=${tag}_m${mcs}
    [ -f $SLOT_RESULTS/raw/lacl_${name}.json ] && { echo "=== $name (done)"; continue; }
    echo "=== $name $(date +%T)"
    if [ ! -f $R/la_cl/$name.npz ]; then
      bash scripts_for_node/backstop_slot/nv/run_nv_dataset.sh $R/la_cl/$name.npz --base-config nrx_large.cfg --prbs 273 --num-tx 2 \
        --channel $channel --mcs $mcs --esno $esno --slots $slots --batch 10 --gpu $gpu 2>&1 | grep -E 'ebno [-0-9]|rror' | tail -4
    fi
    shifter --image="$AERIAL_IMAGE" python3 scripts_for_node/backstop_slot/nv/nv_build_dataset.py --input $R/la_cl/$name.npz --ebno all \
      --profile nv_mu2_m$mcs --output $D 2>&1 | tail -1
    CUDA_VISIBLE_DEVICES=$gpu shifter_slot python3 /backstop_slot/probe_mu.py --dataset $D --profile nv_mu2_m$mcs --engine $LARGE \
      --conv-iterations 20 --ldpc-iterations 20 --passes 2 --per-slot $SLOT_RESULTS/raw/lacl_${name}_slots.npz \
      --output $SLOT_RESULTS/raw/lacl_${name}.json 2>&1 | grep -E 'rror|Traceback' | head -3
    echo "=== $name done $(date +%T)"
  done
}
for spec in "$@"; do
  stream "${spec%%:*}" "${spec#*:}" > $R/logs/lacl_${tag}_g${spec%%:*}_$(date +%H%M%S).log 2>&1 &
done
wait
echo LA_CL_GEN_DONE
