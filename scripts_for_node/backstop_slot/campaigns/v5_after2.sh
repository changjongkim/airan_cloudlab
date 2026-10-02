#!/usr/bin/env bash
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
for algo in 1 2 0; do
  CUDA_VISIBLE_DEVICES=0 shifter_slot python3 /backstop_slot/probe_eq.py --dataset $SOFTWALL_ROOT/run_state/backstop_slot/dataset_v5 --algo $algo \
    --output $SLOT_RESULTS/raw/eq_check_v5_algo${algo}_j${SLURM_JOB_ID}.json 2>&1 | grep -E '^\{|Error|rror:' | tail -2
done
# Two NeuralRx lanes per GPU, the second used only when both TBs still meet the rescue deadline.
export DATA=$PWD/run_state/backstop_slot/dataset_v5 ITER=20 D2=11.5 CELLS=16 KFAIL=20 NRX_BOUND=5.2 TABLE="128:5.4,512:6.3,1024:6.3"
TAG=l LANES=2 NRXFLAGS="--nrx-bound-busy 5.2,8.0 --nrx-flags second_lane=deadline" REFS="n" RATES="32" SHARES="50 70" GATING="conv=128+yield" bash run_state/backstop_slot/mu_main.sh
echo V5_AFTER2_DONE
