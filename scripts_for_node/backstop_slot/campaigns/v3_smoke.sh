#!/usr/bin/env bash
# Smoke tests of the new v3 components (short runs).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=${PERIODS:-1600} SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
for t in "$@"; do
case $t in
bench)
  shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
    --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=CUDA_VISIBLE_DEVICES=0 \
    python3 /backstop_slot/bench_model_units.py --model Qwen/Qwen2.5-0.5B --output $SLOT_RESULTS/raw/qwen05_units_j${SLURM_JOB_ID}.json 2>&1 | tail -1 ;;
group)
  base="--seed 82000 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms 11.5 --controller controller3.py"
  go sg1 4 "$base --conv-runtime group --group-size 1" "c32:rescue_value:none"
  go sg2 4 "$base --conv-runtime group --group-size 2" "c32:rescue_value:none"
  go sp0 4 "$base" "c32:rescue_value:none" ;;
partial)
  base="--seed 82000 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms 11.5 --controller controller3.py --conv-runtime group"
  go sa64p50 4 "$base --group-size 2 --activity-prob 0.5" "c64:rescue_value:none"
  go sa96p33 4 "$base --group-size 3 --activity-prob 0.33" "c96:rescue_value:none"
  go sa48p50 4 "--seed 82000 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --rescue-deadline-ms 11.5 --controller controller3.py --activity-prob 0.5" "c48:rescue_value:none" ;;
classes)
  base="--seed 82000 --lanes-per-gpu 1 --nrx-bound-ms 2.8 --nrx-bound-corun-ms 3.8 --rescue-deadline-ms 6.5 --ai-admission 1 --ai-chunks 128,512,1024 --ai-max-piece-ms 4.0 --ai-adaptive-bound 1 --static-mps-pct 100"
  go sc3 8 "$base --ai-classes chat:1.5B:8:200,small:0.5B:8:100,batch:1.5B --unit-gating 128:3.8,512:4.5,1024:5.2/conv=128,512,1024" "c24:rescue_value:backstop_units"
  go sc3s 8 "$base --ai-classes chat:1.5B:8:200,small:0.5B:8:100,batch:1.5B --static-mps-pct 50" "c24:rescue_value:static" ;;
lanes)
  base="--seed 82000 --rescue-deadline-ms 11.5 --controller controller3.py --lanes-per-gpu 2 --nrx-bound-ms 2.8 --nrx-bound-busy 2.8,4.5"
  go sl2d 4 "$base --nrx-flags second_lane=deadline" "c40:rescue_value:none"
  go sl2v 4 "$base --nrx-flags second_lane=deadline,rank=value --nrx-max-cb-fail 4" "c40:rescue_value:none"
  go sl2a 4 "$base --nrx-flags second_lane=always" "c40:rescue_value:none" ;;
esac
done
echo SMOKE_DONE
