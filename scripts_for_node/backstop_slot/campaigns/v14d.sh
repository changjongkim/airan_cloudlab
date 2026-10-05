#!/usr/bin/env bash
# v14: Antiphase with stoppable AI pieces and rules chosen by the recovery-loss model, and the
# YinYangRAN-style baseline.  Controller: controller5.py.
# Setting of every run: nrx_large rescue, both receivers 20 LDPC iterations, mixed channels,
# recovery deadline 11.5 ms, 4 A100 GPUs, Qwen2.5-1.5B prefill requests with a 200 ms limit.
#
# Policies (codes in run names):
#   n      no AI (reference of the seed)
#   sP     fixed MPS share P%            (fixed partition, as in the SoftBank/NVIDIA AI-RAN MIG split)
#   i      AI only while the GPU has no radio work, ending before the next arrival (Orion/REEF-style)
#   pP     fixed MPS share P% with the AI at a low MPS priority (p100: priority only, SMEC-style)
#   dAxBlN share follows the radio load: A% when full, B% otherwise, load seen N periods late
#          (YinYangRAN-style; dAxBxClN uses three shares; eAxBlN: the same with low-priority AI)
#   vf     Antiphase: low-priority AI, pieces granted by deadline, every unit size next to the
#          conventional receiver, AI next to a running NeuralRx only while three lanes are free
#   v4     Antiphase of v7-v12 (no priority, no lane reserve)
#   ve/vd/vc/v8  ablations: AI never next to NeuralRx / reserve 2 / 70% cap / 128-token units only
#
# v14d: as v14c (ring 512 in every step) with the conditions of other sizes: eight, two and six
# two-user cells (w8, w26), one and two GPUs (gpu1, gpu2).
# usage: bash v14d.sh STEP...      (STEP: yyrcal full alt steps w8 w26 gpu1 gpu2)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot
M="bash scripts_for_node/backstop_slot/run_matrix.sh"
OUR=${OUR:-wr3}; RINGOPT=${RINGOPT---ring 512}      # every cell replays 512 slots (1.28 s), not 128
BASE="--weak-profile nv_mu2 --dataset $R/dataset_mix --engine /softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt --conv-ldpc-iterations 20 --nrx-ldpc-iterations 20 --ai-admission 1 --controller controller5.py --lanes-per-gpu 1 --nrx-max-cb-fail 9 --ai-chunks 128,512,1024 --rescue-deadline-ms 11.5 --ai-dispatch global --ai-admission-fraction 0.75 --nrx-bound-ms 7.6 --nrx-bound-corun-ms 10.1"
OURS="--static-mps-pct 100 --gated-mps-pct 70 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained --unit-gating 128:10.1,512:9.8,1024:10.3/conv=128+yield"

OURS_COMMON="--static-mps-pct 100 --ai-max-piece-ms 4.0 --ai-chunk-choice sustained"
go() { SLOT_TAG=$1 SLOT_PERIODS=$PERIODS SLOT_AI_RATE=$2 SLOT_EXTRA="$3" SLOT_MATRIX="$4" $M 2>&1 | grep -E "^===|FAILED"; }

# cond TAG CELLS "RATES" "EXTRA" "POLICIES"     (env: SEEDS, PERIODS, LEVELS2, LEVELS3)
cond() {
  local tag=$1 cells=$2 rates=$3 extra=$4 pols=$5 seed rate pol common spec shares lag levels prio pc rs gate est
  for seed in $SEEDS; do
    common="--seed $((81000 + 1000 * seed)) --ring-offset $((50 * (seed - 1))) $BASE $extra"
    case " $pols " in *" n "*) go ${tag}${seed}n 4 "$common" "c$cells:rescue_value:none" ;; esac
    for rate in $rates; do
      for pol in $pols; do
        case $pol in
          n) ;;
          v4) go ${tag}${seed}r${rate}v4 $rate "$common $OURS" "c$cells:rescue_value:backstop_units" ;;
          v4p) go ${tag}${seed}r${rate}v4p $rate "$common $OURS --ai-mps-priority 1" "c$cells:rescue_value:backstop_units" ;;
          v5) go ${tag}${seed}r${rate}v5 $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128,512+yield" "c$cells:rescue_value:backstop_units" ;;
          v6) go ${tag}${seed}r${rate}v6 $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128,512,1024+yield" "c$cells:rescue_value:backstop_units" ;;
          v8) go ${tag}${seed}r${rate}v8 $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128+yield+reserve2" "c$cells:rescue_value:backstop_units" ;;
          v8b) go ${tag}${seed}r${rate}v8b $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128,512+yield+reserve2" "c$cells:rescue_value:backstop_units" ;;
          v9) go ${tag}${seed}r${rate}v9 $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128+yield+reserve1" "c$cells:rescue_value:backstop_units" ;;
          vs) go ${tag}${seed}r${rate}vs $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:99,512:99,1024:99/conv=128+yield" "c$cells:rescue_value:backstop_units" ;;
          va) go ${tag}${seed}r${rate}va $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:99,512:99,1024:99/conv=128,512,1024+yield" "c$cells:rescue_value:backstop_units" ;;
          vb) go ${tag}${seed}r${rate}vb $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128,512,1024+yield+reserve2" "c$cells:rescue_value:backstop_units" ;;
          vc) go ${tag}${seed}r${rate}vc $rate "$common $OURS_COMMON --gated-mps-pct 70 --ai-mps-priority 1 --unit-gating 128:7.8,512:8.2,1024:8.3/conv=128,512,1024+yield+reserve3" "c$cells:rescue_value:backstop_units" ;;
          vd) go ${tag}${seed}r${rate}vd $rate "$common $OURS_COMMON --ai-mps-priority 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512,1024+yield+reserve2" "c$cells:rescue_value:backstop_units" ;;
          ve) go ${tag}${seed}r${rate}ve $rate "$common $OURS_COMMON --ai-mps-priority 1 --unit-gating 128:99,512:99,1024:99/conv=128,512,1024+yield" "c$cells:rescue_value:backstop_units" ;;
          w*) # v14 rules, all with low-priority AI, no cap, every unit size next to the conventional
              # receiver and stoppable pieces:  wn never next to NeuralRx, wrR reserve R,
              # wuR reuse-time rule with reserve R (wu0: reuse time only)
              case $pol in
                wn) gate="128:99,512:99,1024:99/conv=${VF_CONV:-128,512,1024}+yield" ;;
                wr*) gate="${VF_TABLE:-128:8.1,512:8.5,1024:8.7}/conv=${VF_CONV:-128,512,1024}+yield+reserve${pol#wr}" ;;
                wu*) gate="${VF_TABLE:-128:8.1,512:8.5,1024:8.7}/conv=${VF_CONV:-128,512,1024}+yield+reserve${pol#wu}+reuse" ;;
              esac
              go ${tag}${seed}r${rate}$pol $rate "$common --static-mps-pct 100 --ai-max-piece-ms ${PIECE:-4.0} --ai-chunk-choice sustained --ai-mps-priority 1 --ai-stop-check 1 --unit-gating $gate" "c$cells:rescue_value:backstop_units" ;;
          vf) go ${tag}${seed}r${rate}vf $rate "$common $OURS_COMMON --ai-mps-priority 1 --unit-gating ${VF_TABLE:-128:8.1,512:8.5,1024:8.7}/conv=${VF_CONV:-128,512,1024}+yield+reserve${RESERVE:-3}" "c$cells:rescue_value:backstop_units" ;;
          vg|vh|vi|vj) case $pol in vg) pc=2.5; rs=3 ;; vh) pc=1.5; rs=3 ;; vi) pc=2.5; rs=2 ;; vj) pc=1.5; rs=2 ;; esac
              go ${tag}${seed}r${rate}$pol $rate "$common --static-mps-pct 100 --ai-max-piece-ms $pc --ai-chunk-choice sustained --ai-mps-priority 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512,1024+yield+reserve$rs" "c$cells:rescue_value:backstop_units" ;;
          v7) go ${tag}${seed}r${rate}v7 $rate "$common $OURS_COMMON --ai-mps-priority 1 --unit-gating 128:8.1,512:8.5,1024:8.7/conv=128,512+yield" "c$cells:rescue_value:backstop_units" ;;
          s*) go ${tag}${seed}r${rate}$pol $rate "$common --static-mps-pct ${pol#s}" "c$cells:rescue_value:static" ;;
          p*) go ${tag}${seed}r${rate}$pol $rate "$common --static-mps-pct ${pol#p} --ai-mps-priority 1" "c$cells:rescue_value:static" ;;
          i) go ${tag}${seed}r${rate}i $rate "$common --ai-chunk-choice budget" "c$cells:rescue_value:backstop" ;;
          yyr|yyp) # YinYangRAN-style: decision every second from the last second's load and a trained
              # reliability estimator; a change of share stops AI for 0.266 s (yyp: low-priority AI)
              if [ $pol = yyr ]; then shares=${YYR_SHARES:-10,20,30,50,70}; est=$YYR_EST; prio=""
              else shares=${YYP_SHARES:-30,50,70,100}; est=$YYP_EST; prio="--ai-mps-priority 1"; fi
              go ${tag}${seed}r${rate}$pol $rate "$common --static-mps-pct 100 --dynamic-shares $shares --dynamic-lag 0 $prio --yyr $est,${YYR_SPEC:-0.97,0.999,0.25,266,400}" "c$cells:rescue_value:static" ;;
          d*|e*) spec=${pol#?}; prio=""; [ "${pol:0:1}" = e ] && prio="--ai-mps-priority 1"; shares=${spec%%l*}; lag=${spec##*l}; shares=${shares//x/,}
              levels=${LEVELS2:-}
              case $shares in *,*,*) levels=${LEVELS3:-0.875,0.625} ;; esac
              go ${tag}${seed}r${rate}$pol $rate "$common --static-mps-pct 100 --dynamic-shares $shares --dynamic-lag $lag $prio ${levels:+--dynamic-levels $levels}" "c$cells:rescue_value:static" ;;
        esac
      done
    done
  done
}

for step in "$@"; do
  case $step in
    yyrcal)   # calibration of the YinYangRAN-style estimator: steady loads, every share, seed 7
      ( export SEEDS="${SEEDS:-7}" PERIODS=8000
        for item in za:0.25 zb:0.5 zc:0.75; do
          cond ${item%%:*} 16 "32" "--weak-fraction 0.25 --activity-prob ${item##*:} $RINGOPT" "${POLS:-n s10 s20 s30 s50 s70 p30 p50 p70 p100}"
        done
        cond zd 16 "32" "--weak-fraction 0.25 $RINGOPT" "${POLS:-n s10 s20 s30 s50 s70 p30 p50 p70 p100}" ) ;;
    ringtest) # longer slot pool per cell: the pattern of failed TBs repeats every 1.28 s, not 0.32 s
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
        cond rr 16 "32" "--weak-fraction 0.25 --ring ${RING:-512}" "${POLS:-n wr3 wr2 vf p30 p70 s10}" ) ;;
    w8)       # eight two-user cells of sixteen
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
        cond sw 16 "32" "--weak-fraction 0.5 $RINGOPT" "${POLS:-n wn wr3 p30 p70 s10}" ) ;;
    w26)      # two and six two-user cells of sixteen
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
        cond sx 16 "32" "--weak-fraction 0.125 $RINGOPT" "${POLS:-n wn p30 p70 s10}"
        cond sy 16 "32" "--weak-fraction 0.375 $RINGOPT" "${POLS:-n wn p30 p70 s10}" ) ;;
    gpu1)     # one GPU: one and two two-user cells of four
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
        cond sa 4 "32" "--weak-fraction 0.25 --gpus 1 $RINGOPT" "${POLS:-n wn p30 p70 s10}"
        cond sb 4 "32" "--weak-fraction 0.5 --gpus 1 $RINGOPT" "${POLS:-n wn p30 p70 s10}" ) ;;
    gpu2)     # two GPUs: two and four two-user cells of eight
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
        cond sc 8 "32" "--weak-fraction 0.25 --gpus 2 $RINGOPT" "${POLS:-n wn wr1 p30 p70 s10}"
        cond sd 8 "32" "--weak-fraction 0.5 --gpus 2 $RINGOPT" "${POLS:-n wn wr1 p30 p70 s10}" ) ;;
    full)     # headline 1: steady full load
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=4000
        cond fa 16 "32" "--weak-fraction 0.25 $RINGOPT" "${POLS:-n $OUR vf s10 s30 p30 p50 p70 p100 yyr yyp}" ) ;;
    alt)      # headline 2: load alternates between full and half every 2 s
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=8000
        cond fb 16 "32" "--weak-fraction 0.25 --activity-prob 0.5 --activity-mode phased --activity-phase 800 $RINGOPT" "${POLS:-n $OUR vf s10 s30 p30 p50 p70 d10x50l0 e30x70l0 yyr yyp}" ) ;;
    steps)    # headline 3: load jumps between 25/50/75/100% at random times, 40 s
      ( export SEEDS="${SEEDS:-1 2}" PERIODS=16000 LEVELS3=0.875,0.625
        cond fc 16 "32" "--weak-fraction 0.25 --activity-mode steps --activity-phase 400 $RINGOPT" "${POLS:-n $OUR vf s10 p30 p50 p70 d10x30x50l0 e30x50x70l0 yyr yyp}" ) ;;
  esac
done
echo V14D_DONE
