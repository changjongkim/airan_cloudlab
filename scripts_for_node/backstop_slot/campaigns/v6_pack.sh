#!/usr/bin/env bash
# AI packing: requests fill the lowest-numbered GPUs first, leaving the other GPUs' NeuralRx
# lanes and conventional receivers without AI.  16 cells; references: tags x (11.5 ms), t (6.5 ms).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 CELLS=16 KFAIL=20 NRX_BOUND=4.2 TABLE="128:4.4,512:4.4,1024:4.6" REFS=" "
export CONTROLLER=controller4.py EXTRA="--ai-extra $PWD/run_state/backstop_slot/configs/pack.json"
for d2 in ${D2S:-6.5 11.5}; do
  tag=pt; gate="conv=128+yield"
  [ $d2 = 11.5 ] && tag=px && gate="conv=128,512+yield"
  D2=$d2 TAG=$tag GATING="$gate" SHARES="50 70 100" RATES="${RATES:-16 32}" bash run_state/backstop_slot/mu_main.sh
done
echo V6_PACK_DONE
