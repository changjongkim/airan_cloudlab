#!/usr/bin/env bash
# Closed-loop link adaptation: the two-user cells pick the MCS of every slot by an outer loop on the TBs that
# need a retransmission (not decoded by the conventional receiver and not recovered before the recovery deadline).
# Every policy shares the GPUs as in v14h.sh; the recovery path, the AI requests and the outer loop are the same.
#   usage: bash la_cl5.sh TAG TARGET "SEEDS" "POLICIES"      (la_cl4.sh with the neural receivers on the SMs that the
#          AI slice leaves: policies wdN, gdN, and gnN.  The runs of 2026-10-07 used a copy of the script directory
#          with the same nrx_lane.py and make_config.py, because another campaign was running from the directory.)
#   env:   DATA (dataset_la_<DATA>, default high16), LEVELS (default 10..16), START (start level index, default 3),
#          DOWN (pointer step down per failed TB, default 0.25), WEAK (share of two-user cells, default 0.25),
#          CELLS (16), PERIODS (8000), DELAY (feedback delay in periods, default 6), EXTRA (e.g. "--gpus 2")
#          STATES (e.g. "high14 high16 high20": the two-user cells move between the datasets dataset_la_<state>,
#          each the same channel at another Es/No, staying PHASE periods (default 800 = 2 s) in a state; la_states.py)
#   policies: x no recovery path, n recovery path without AI, wm the rule, sP fixed share P%, pP share P% with
#             low-priority AI (p100: low priority alone), xpP no recovery path with AI as pP, wrR the rule with AI
#             allowed next to a neural receiver while R neural receivers of the server are free,
#             wsN the rule with the SM slice option: while the neural receiver of a GPU runs, the AI of that GPU moves
#             to N SMs (a CUDA green context) and is not stopped;  gN low-priority AI that always runs on N SMs
#             wdN as wsN with the neural receivers on the other 108-N SMs (the slice belongs to the AI alone)
#             gnN AI at normal priority always on N SMs, receivers on all SMs (a green context for the AI alone)
#             gdN low-priority AI always on N SMs, neural receivers always on the other 108-N SMs (a fixed split)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
tag=$1; target=$2; seeds=$3; pols=$4
cells=${CELLS:-16}
STATEOPT=""
if [ -n "${STATES:-}" ]; then
  dirs=""; for s in $STATES; do dirs="${dirs:+$dirs,}$R/dataset_la_$s"; done
  STATEOPT="--la-states $dirs:${PHASE:-800}"
fi
BASE="--dataset $R/dataset_la_${DATA:-high16} --la ${LEVELS:-10,11,12,13,14,15,16}:$target:${DOWN:-0.25}:${DELAY:-6}:${START:-3} $STATEOPT --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --ai-admission 1 --controller controller5.py --lanes-per-gpu 1 --nrx-max-cb-fail 9 --ai-chunks 128,512,1024 --rescue-deadline-ms 11.5 --ai-dispatch global --ai-admission-fraction 0.75 --nrx-bound-ms ${NRXB:-7.6} --nrx-bound-corun-ms 10.1 --weak-fraction ${WEAK:-0.25} --ring 512 ${EXTRA:-}"
go() { SLOT_TAG=$1 SLOT_PERIODS=${PERIODS:-8000} SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }
for seed in $seeds; do
  common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) $BASE"
  for pol in $pols; do
    case $pol in
      x)  go ${tag}${seed}x 4 "$common" "c$cells:off:none" ;;
      n)  go ${tag}${seed}n 4 "$common" "c$cells:rescue_value:none" ;;
      wm) go ${tag}${seed}r32wm 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:99,512:99,1024:99/conv=128,512+yield" "c$cells:rescue_value:backstop_units" ;;
      xp*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct ${pol#xp} --ai-mps-priority 1" "c$cells:off:static" ;;
      wr*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512+yield+reserve${pol#wr}" "c$cells:rescue_value:backstop_units" ;;
      ws*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:99,512:99,1024:99/conv=128,512+yield --ai-slice-sms ${pol#ws}" "c$cells:rescue_value:backstop_units" ;;
      wd*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating 128:99,512:99,1024:99/conv=128,512+yield --ai-slice-sms ${pol#wd} --nrx-off-slice 1" "c$cells:rescue_value:backstop_units" ;;
      gn*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-green-sms ${pol#gn}" "c$cells:rescue_value:static" ;;
      gd*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-mps-priority 1 --ai-green-sms ${pol#gd} --nrx-off-slice 1" "c$cells:rescue_value:static" ;;
      g*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct 100 --ai-mps-priority 1 --ai-green-sms ${pol#g}" "c$cells:rescue_value:static" ;;
      s*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct ${pol#s}" "c$cells:rescue_value:static" ;;
      p*) go ${tag}${seed}r32$pol 32 "$common --static-mps-pct ${pol#p} --ai-mps-priority 1" "c$cells:rescue_value:static" ;;
    esac
  done
done
echo LA_CL5_DONE
