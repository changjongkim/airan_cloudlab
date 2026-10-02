#!/usr/bin/env bash
# NVlabs NeuralRx vs Aerial conventional receiver on the same received slots.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
R=/pscratch/sd/s/sgkim/kcj/airan_cloudlab/run_state/backstop_slot/nv
S=scripts_for_node/backstop_slot/nv
one() {   # name base-config prbs num_tx channel ebno-list [extra]
  local name=$1 cfg=$2 prbs=$3 ntx=$4 ch=$5 ebno=$6; shift 6
  [ -f $R/$name.json ] && { echo "skip $name"; return; }
  echo "=== $name $(date +%T)"
  bash $S/run_nv_dataset.sh $R/$name.npz --base-config $cfg --prbs $prbs --num-tx $ntx --channel $ch --ebno $ebno --slots ${SLOTS:-100} --batch 10 --gpu 0 "$@" 2>&1 | grep -E 'ebno [-0-9]|rror' 
  bash $S/run_nv_conventional.sh $R/$name.npz $R/$name.json 2>&1 | grep -E '^\|'
}
one rt_2ue_low     nrx_rt.cfg     273 2 DoubleTDLlow  1,1.5,2,2.5,3,3.5
one large_2ue_low  nrx_large.cfg  273 2 DoubleTDLlow  1,1.5,2,2.5,3,3.5
one rt_1ue         nrx_rt.cfg     273 1 TDL-B100      -1,-0.5,0,0.5,1,1.5
one rt_2ue_high    nrx_rt.cfg     273 2 DoubleTDLhigh 2,3,4,5,6,7
one rt_2ue_umi     nrx_rt.cfg     273 2 UMi           1,2,3,4,5,6
one rt64_2ue_low   nrx_rt_64qam.cfg 273 2 DoubleTDLlow 5,6,7,8,9,10
echo SCENARIOS_DONE
