#!/usr/bin/env bash
# U0: mixed-chunk correctness, then NeuralRx and conventional under a 100% AI share with
# one AI chunk size at a time (rescue deadline 41.5 ms so every NeuralRx runs to the end).
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
  --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
  --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=CUDA_VISIBLE_DEVICES=0 \
  python3 /backstop_slot/bench_units.py --chunks 128,512,1024 --repeats 60 --output $SLOT_RESULTS/raw/qwen_mixed_units_j${SLURM_JOB_ID}.json \
  > $SLOT_RESULTS/raw/qwen_mixed_units_j${SLURM_JOB_ID}.log 2>&1
for cells in 16 32; do
  if [ $cells -eq 16 ]; then lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.5"; else lanes="--lanes-per-gpu 2 --nrx-bound-ms 3.3"; fi
  base="--seed 82000 $lanes --rescue-deadline-ms 41.5 --ai-admission 0 --static-mps-pct 100 --ai-chunks 128,512,1024"
  SLOT_TAG=u0n SLOT_PERIODS=2400 SLOT_AI_RATE=12 SLOT_EXTRA="$base" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
  for c in 128 512 1024; do
    SLOT_TAG=u0f$c SLOT_PERIODS=2400 SLOT_AI_RATE=24 SLOT_EXTRA="$base --ai-force-chunk $c" SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo U0_DONE
