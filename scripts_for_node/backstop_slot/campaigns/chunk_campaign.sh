#!/usr/bin/env bash
# Fair AI units: every AI policy uses chunks 128/512/1024 (a fixed share always takes the
# chunk that fits the rest of the prompt). 16/24/32 cells, AI 12 req/s, 2 seeds.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
source scripts_for_node/backstop_slot/slot_env.sh
shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
  --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
  --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=CUDA_VISIBLE_DEVICES=0 \
  python3 /backstop_slot/bench_units.py --chunks 128,512,1024 --repeats 60 --output $SLOT_RESULTS/raw/qwen_mixed_units_j${SLURM_JOB_ID}.json \
  > $SLOT_RESULTS/raw/qwen_mixed_units_j${SLURM_JOB_ID}.log 2>&1
for seed in 1 2; do
  for cells in 16 24 32; do
    if [ $cells -eq 16 ]; then lanes="--lanes-per-gpu 1 --nrx-bound-ms 2.5 --nrx-bound-corun-ms 2.8"
    else lanes="--lanes-per-gpu 2 --nrx-bound-ms 3.3 --nrx-bound-corun-ms 3.7"; fi
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 $lanes --static-mps-pct 50 --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0"
    SLOT_TAG=k${seed}d65 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --rescue-deadline-ms 6.5" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_corun c$cells:rescue_value:static c$cells:parallel_admit:static c$cells:off:static" $M 2>&1 | grep -E "^===|FAILED"
    SLOT_TAG=k${seed}d50 SLOT_PERIODS=4000 SLOT_AI_RATE=12 SLOT_EXTRA="$common --rescue-deadline-ms 5.0" \
      SLOT_MATRIX="c$cells:rescue_value:backstop_corun c$cells:rescue_value:static" $M 2>&1 | grep -E "^===|FAILED"
  done
done
echo CHUNK_DONE
