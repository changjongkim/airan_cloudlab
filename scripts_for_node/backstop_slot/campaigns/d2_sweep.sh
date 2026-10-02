#!/usr/bin/env bash
# Rescue-deadline sweep: 16 cells, AI 12 req/s, 2 seeds, Our Scheme vs fixed 50%.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
source scripts_for_node/backstop_slot/slot_env.sh
for c in 512 1024; do
  shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
    --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=CUDA_VISIBLE_DEVICES=0 \
    python3 /backstop_slot/bench_qwen.py --chunk $c --repeats 60 --output $SLOT_RESULTS/raw/qwen_units_c${c}_j${SLURM_JOB_ID}.json > /dev/null 2>&1
done
shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
  --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
  --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=CUDA_VISIBLE_DEVICES=0 \
  python3 /backstop_slot/bench_units.py --chunks 128,512 --output $SLOT_RESULTS/raw/qwen_mixed_units_j${SLURM_JOB_ID}.json > /dev/null 2>&1
for seed in 1 2; do
  for d2 in 4.0 5.0 6.5 11.5 21.5 41.5; do
    tag=$(echo $d2 | tr -d .)
    base="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --rescue-deadline-ms $d2 --lanes-per-gpu 1 --nrx-bound-ms 2.5 --nrx-bound-corun-ms 2.8 --static-mps-pct 50"
    SLOT_TAG=x${seed}d$tag SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$base" \
      SLOT_MATRIX="c16:rescue_value:backstop_corun c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=x${seed}d${tag}p3 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$base --ai-max-piece-ms 3.0" \
      SLOT_MATRIX="c16:rescue_value:backstop_corun" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=x${seed}d${tag}s70 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="${base/--static-mps-pct 50/--static-mps-pct 70}" \
      SLOT_MATRIX="c16:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  done
  SLOT_TAG=x${seed}n SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --rescue-deadline-ms 41.5" \
    SLOT_MATRIX="c16:rescue_value:none" $M 2>&1 | grep -E "^===|FAILED"
done
echo D2_DONE
