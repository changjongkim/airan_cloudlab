#!/usr/bin/env bash
# v14h: unit sizes next to the conventional receiver by the active cells of the slot, with every size while
# few cells of the GPU are active (policy codes wa: up to two active cells, wb: up to three).
#   pilot   2-s alternation and random steps, seeds 1-2: n, wm, wa, wb in one job      (about 17 min);
#           then 20 cells with the neural receiver bound 7.6 ms (dense76, about 10 min) and 32 / 48 cells with the
#           two smaller unit classes for any number of active cells (wd, we; about 5 min)
#   head    the same two conditions, seeds 3-5: n and the policies of POLS_HEAD        (about 7 min per policy)
#   cells   32 and 48 cells with the policies of POLS_HEAD (n runs of the job are reused)
#   phase   alternation every 0.2 s and 5 s; burst: cell bursts (as v14g_chain b, with POLS_HEAD added)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export YYR_EST=$PWD/results/backstop_slot/yyr_estimator_fixed_j59210955.json
export YYP_EST=$PWD/results/backstop_slot/yyr_estimator_lowprio_j59210955.json
H=${POLS_HEAD:-wa}
for part in "$@"; do
  case $part in
    pilot) SEEDS="1 2" POLS="n wm wa wb" bash run_state/backstop_slot/v14h.sh alt steps
           POLS="n wm wd s10 p30 p70" bash run_state/backstop_slot/v14h.sh dense76
           POLS="wd we" bash run_state/backstop_slot/v14h.sh cells ;;
    head)  SEEDS="3 4 5" POLS="n $H" bash run_state/backstop_slot/v14h.sh alt steps ;;
    cells) POLS="n $H" bash run_state/backstop_slot/v14h.sh cells ;;
    phase) POLS="n wm $H s10 p30 p70 d10x50l0 e30x70l0 yyr yyp" bash run_state/backstop_slot/v14h.sh phase ;;
    burst) POLS="n wm $H s10 p30 p70 e30x70l0" bash run_state/backstop_slot/v14h.sh burst ;;
  esac
done
echo V14H_CHAIN_DONE
