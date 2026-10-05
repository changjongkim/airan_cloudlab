#!/usr/bin/env bash
# v14, final rule (policy code wm: stoppable pieces, no AI next to a running NeuralRx, the largest
# AI units not next to the conventional receiver): the conditions measured with it.
#   headline   full load, load alternating every 2 s, random steps (seeds 1-5; n and wm)
#   sizes      eight / two / six two-user cells, one and two GPUs (seeds 1-2)
#   kinds      five kinds of AI work, chat only (seeds 1-2)
# usage: bash v14_final.sh STEP...
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
for step in "$@"; do
  case $step in
    full|alt|steps) SEEDS="${SEEDS:-1 2 3 4 5}" POLS="n wm" bash run_state/backstop_slot/v14e.sh $step ;;
    sizes) POLS="n wm" bash run_state/backstop_slot/v14e.sh w8 gpu1 gpu2
           POLS="n wm p30 p70 s10" bash run_state/backstop_slot/v14e.sh w26 ;;
    kinds) POLS="n wm" bash run_state/backstop_slot/v15.sh mix chat ;;
  esac
done
echo V14_FINAL_DONE
