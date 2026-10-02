#!/usr/bin/env bash
# Capacity region: AI load sweep per policy; the answer is the largest AI throughput a policy
# serves while keeping L1 misses <= 0.05% and rescues >= 99% of no AI (both seeds).
# 1 NeuralRx lane per GPU, fair AI units, token AI admission, 2 seeds.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
run_setting() {   # cells d2 rates...
  local cells=$1 d2=$2; shift 2
  local tag=$(echo $d2 | tr -d .)
  local b bc ours
  if [ $cells -eq 16 ]; then b="128:3.1,512:3.9,1024:4.3"; bc=3.1; ours="$b/conv=128,512"; bound=2.5
  elif [ $cells -eq 24 ]; then b="128:3.8,512:4.5,1024:5.2"; bc=3.8; ours="$b/conv=128"; bound=2.8
  else b="128:4.1,512:4.7,1024:5.3"; bc=4.1; ours="$b/conv=128+yield"; bound=2.8; fi
  for seed in 1 2; do
    local common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms $bound --nrx-bound-corun-ms $bc --ai-chunks 128,512,1024 --ai-max-piece-ms 3.0 --rescue-deadline-ms $d2"
    SLOT_TAG=c${seed}d${tag}n SLOT_PERIODS=4000 SLOT_AI_RATE=4 SLOT_EXTRA="$common" SLOT_MATRIX="c$cells:rescue_value:none" $M 2>&1 | grep -E "FAILED"
    for rate in "$@"; do
      SLOT_TAG=c${seed}d${tag}r${rate}u SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct 100 --unit-gating $ours" \
        SLOT_MATRIX="c$cells:rescue_value:backstop_units" $M 2>&1 | grep -E "FAILED"
      for pct in 30 50 70 100; do
        SLOT_TAG=c${seed}d${tag}r${rate}s$pct SLOT_PERIODS=4000 SLOT_AI_RATE=$rate SLOT_EXTRA="$common --static-mps-pct $pct" \
          SLOT_MATRIX="c$cells:rescue_value:static" $M 2>&1 | grep -E "FAILED"
      done
    done
  done
}
run_setting 24 6.5 4 8 12 16
run_setting 32 11.5 4 8 12 16
run_setting 16 6.5 12 16 20
echo CAP_DONE
