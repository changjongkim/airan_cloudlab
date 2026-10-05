#!/usr/bin/env bash
# v15: several models and kinds of AI work (ai_worker5): chat with token generation, a larger
# LLM, text embedding, image classification, and a filler without a time limit.
#   bench   GPU time of every unit alone -> results/backstop_slot/unit_bounds5.json   (GPU 0)
#   smoke   one short run per class mix
#   chat    chat only (prefill + decode)          mix   five kinds of work      mixnf   without the filler
# usage: bash v15.sh STEP...
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
BASE="--weak-profile nv_mu2 --dataset $R/dataset_mix --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --ai-admission 1 --ai-admission-fraction 0.75 --ai-dispatch global --controller controller6.py --lanes-per-gpu 1 --nrx-max-cb-fail 9 --ai-chunks 128,512,1024 --rescue-deadline-ms 11.5 --nrx-bound-ms 7.6 --nrx-bound-corun-ms 10.1 --weak-fraction 0.25 --ring 512"
OURS="--static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512,1024+yield+reserve${RESERVE:-3}"
OURS_NEVER="--static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:99,512:99,1024:99/conv=128,512,1024+yield"
# Antiphase: stoppable pieces, no AI next to a running NeuralRx, the largest unit class not next to the conventional receiver
OURS_FINAL="--static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:99,512:99,1024:99/conv=128,512+yield"
go() { SLOT_TAG=$1 SLOT_PERIODS=$PERIODS SLOT_AI_RATE=4 SLOT_EXTRA="$2" SLOT_MATRIX="$3" $M 2>&1 | grep -E "^===|FAILED"; }
bench_one() {  # model, kind, short name, extra options
  shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
    --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONNOUSERSITE=1 --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
    --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=CUDA_VISIBLE_DEVICES=0 \
    python3 /backstop_slot/bench_units5.py --model $1 --kind $2 $4 --output $SLOT_RESULTS/raw/units5_$3_j${SLURM_JOB_ID}.json 2>&1 | grep -E "^\{|rror|Traceback" | cut -c1-600
}
# cond TAG CELLS MIX "POLICIES" "EXTRA"
cond() {
  local tag=$1 cells=$2 mix=$3 pols=$4 extra=${5:-} seed pol common
  for seed in $SEEDS; do
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) $BASE $extra --ai-classes5 $R/classes5/$mix.json"
    for pol in $pols; do
      case $pol in
        n) go ${tag}${seed}n "$common" "c$cells:rescue_value:none" ;;
        wr) go ${tag}${seed}r32wr "$common $OURS" "c$cells:rescue_value:backstop_units" ;;
        wn) go ${tag}${seed}r32wn "$common $OURS_NEVER" "c$cells:rescue_value:backstop_units" ;;
        wm) go ${tag}${seed}r32wm "$common $OURS_FINAL" "c$cells:rescue_value:backstop_units" ;;
        s*) go ${tag}${seed}r32$pol "$common --static-mps-pct ${pol#s}" "c$cells:rescue_value:static" ;;
        p*) go ${tag}${seed}r32$pol "$common --static-mps-pct ${pol#p} --ai-mps-priority 1" "c$cells:rescue_value:static" ;;
      esac
    done
  done
}
for step in "$@"; do
  case $step in
    bench)
      bench_one Qwen/Qwen2.5-1.5B llm q15 "--chunks 128,512,1024 --decode-groups 10"
      bench_one Qwen/Qwen2.5-0.5B llm q05 "--chunks 128,512,1024 --decode-groups 8"
      bench_one Qwen/Qwen2.5-3B llm q3 "--chunks 128,512,1024 --decode-groups 13"
      bench_one BAAI/bge-base-en-v1.5 encoder bge "--batches 1,4,16"
      bench_one google/vit-base-patch16-224 encoder vit "--batches 1,4,16"
      python3 - <<P
import json, glob
out = {}
for f in sorted(glob.glob("$SLOT_RESULTS/raw/units5_*_j${SLURM_JOB_ID}.json")):
    d = json.load(open(f)); out[d["model"]] = d
json.dump(out, open("$SLOT_RESULTS/unit_bounds5.json", "w"), indent=1)
print("models", list(out))
P
      ;;
    smoke) ( export SEEDS=1 PERIODS=2000; cond ks 16 chat "${SMOKE_CHAT-wr p70}"; cond kt 16 mix "${SMOKE_MIX-wr wn p70}" ) ;;
    chat)  ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000; cond ka 16 chat "${POLS:-n wn wr s10 s30 p30 p70 p100}" ) ;;
    mix)   ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000; cond kb 16 mix "${POLS:-n wn wr s10 s30 p30 p70 p100}" ) ;;
    mixhalf) ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000; cond kd 16 mixhalf "${POLS:-n wn wr s10 s30 p30 p70 p100}" ) ;;
    mixnf) ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000; cond kc 16 mixnf "${POLS:-n wn wr s10 s30 p30 p70 p100}" ) ;;
  esac
done
echo V15_DONE
