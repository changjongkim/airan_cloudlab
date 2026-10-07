#!/usr/bin/env bash
# v21: the SM slice option of the rule.  While the neural receiver of a GPU runs, the AI of that GPU is moved to
# a few SMs of the GPU (a CUDA green context of the AI worker) and is not stopped; with 0 SMs the rule is as before.
#   policies: n, wm (the rule), wsN (the rule with a slice of N SMs), gN (low-priority AI always on N SMs: the
#             green-context baseline), p30 / p70 (MPS share with low-priority AI)
#   smoke     four two-user cells, 10% target, 6 s: n wm ws28 g28                                   (tag lsm)
#   cal       unit times of the AI model on the whole GPU and on slices of 12, 16 and 28 SMs       (slice_units_j<job>.txt)
#   cl SEEDS  closed loop: four two-user cells at 10% and 1%, eight at 1%                            (lsa lsb lsc)
#   fa SEEDS  16 cells at a steady full load and a fixed MCS, 10 s                                   (tag gs)
#   long SEEDS  10% target in runs of 100 s: layer-1 latency and misses in steady state             (tag lsg)
# usage: bash v21_slice.sh STEP [SEEDS]      (env: POLS)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl4.sh
P=${POLS:-"n wm ws12 ws16 ws28 g28 p30 p70"}
step=$1; seeds=${2:-"1 2"}
[ -f run_state/backstop_slot/queue/SKIP_SLICE ] && { echo "the SM slice did not pass its smoke test: step $step skipped"; exit 0; }
case $step in
  cal)   # unit times of the AI model on the whole GPU and on slices of 12, 16 and 28 SMs
         source scripts_for_node/backstop_slot/slot_env.sh
         for n in 12 16 28; do
           shifter --module=gpu --image="$AERIAL_IMAGE" --volume="$SOFTWALL_RUNTIME:/softwall_runtime" --volume="$SLOT_SCRIPTS:/backstop_slot" \
             --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONPATH=/softwall_runtime/python:/backstop_slot \
             --env=HF_HOME=/softwall_runtime/cache/huggingface --env=HF_HUB_OFFLINE=1 --env=PYTHONNOUSERSITE=1 --env=CUDA_VISIBLE_DEVICES=0 \
             python3 /backstop_slot/gc/gc_units.py $n 2>&1 | grep -E "RESULT|rror|Traceback"
         done | tee $SLOT_RESULTS/slice_units_j${SLURM_JOB_ID}.txt ;;
  smoke) PERIODS=2400 bash $C lsm 0.10 "1" "n wm ws28 g28" ;;
  cl)    for seed in $seeds; do
           bash $C lsa 0.10 "$seed" "$P"
           bash $C lsb 0.01 "$seed" "$P"
           WEAK=0.5 bash $C lsc 0.01 "$seed" "$P"
         done ;;
  long)  PERIODS=40000 bash $C lsg 0.10 "${2:-1 2 3}" "${POLS:-n wm ws16 ws28 g28}" ;;
  fa)    source run_state/backstop_slot/v14h.sh none >/dev/null       # go() and BASE of the fixed-MCS campaigns
         export PERIODS=4000
         RULE="--static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --conv-by-active 99:128,512 --unit-gating 128:99,512:99,1024:99/conv=128,512+yield"
         for seed in $seeds; do
           common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) $BASE --weak-fraction 0.25 $RINGOPT"
           for pol in $P; do
             case $pol in
               n)   go gs${seed}n 4 "$common" "c16:rescue_value:none" ;;
               wm)  go gs${seed}r32wd 32 "$common $RULE" "c16:rescue_value:backstop_units" ;;
               ws*) go gs${seed}r32$pol 32 "$common $RULE --ai-slice-sms ${pol#ws}" "c16:rescue_value:backstop_units" ;;
               g*)  go gs${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-mps-priority 1 --ai-green-sms ${pol#g}" "c16:rescue_value:static" ;;
               p*)  go gs${seed}r32$pol 32 "$common --static-mps-pct ${pol#p} --ai-mps-priority 1" "c16:rescue_value:static" ;;
             esac
           done
         done ;;
esac
echo V21_SLICE_DONE $step
