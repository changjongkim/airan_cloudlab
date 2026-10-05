#!/usr/bin/env bash
# Does a second/third request context per GPU help Antiphase?  With one context a request
# whose chunk in progress is larger than the grant allows stalls the GPU's AI; with more
# contexts the worker can run another request's units meanwhile.  Same total arrival rate.
set -uo pipefail
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
go() { SLOT_TAG=$1 SLOT_PERIODS=4000 SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
cells=$1; d2=$2; b=$3; table=$4; gating=$5; shift 5
for seed in 1 2; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) --ai-admission 1 --lanes-per-gpu 1 --nrx-bound-ms $b --ai-chunks 128,512,1024 --rescue-deadline-ms $d2 --static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating $table/$gating"
  go x${seed}n 4 "$common" "c$cells:rescue_value:none"
  for rate in "$@"; do
    h=$(python3 -c "print($rate/2)"); t=$(python3 -c "print($rate/3)"); f=$(python3 -c "print($rate/4)")
    go x${seed}r${rate}k1 $rate "$common --ai-classes a:1.5B:$rate:200" "c$cells:rescue_value:backstop_units"
    go x${seed}r${rate}k2 $rate "$common --ai-classes a:1.5B:$h:200,b:1.5B:$h:200" "c$cells:rescue_value:backstop_units"
    go x${seed}r${rate}k4 $rate "$common --ai-classes a:1.5B:$f:200,b:1.5B:$f:200,c:1.5B:$f:200,d:1.5B:$f:200" "c$cells:rescue_value:backstop_units"
  done
done
echo V3_CONTEXTS_DONE
