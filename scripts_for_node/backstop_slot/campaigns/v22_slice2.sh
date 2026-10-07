#!/usr/bin/env bash
# v22: the SM slice with the neural receivers off the slice.  In v21 the neural receiver of a GPU (higher MPS
# priority, all 108 SMs) also used the SMs of the AI slice, and an AI unit on the slice took 7-8 times as long as on
# the whole GPU (2.2 times when alone on the slice).  Here each lane runs its neural receiver in a green context
# on the SMs that the slice leaves (make_config.py --nrx-off-slice 1, nrx_lane.py).
#   policies: n, wm (the rule), wdN (the rule with a slice of N SMs for the AI and the neural receivers on the other
#             108-N), gnN (AI at normal priority always on N SMs, receivers on all SMs), gdN (low-priority AI always
#             on N SMs, neural receivers always on the other 108-N), wsN / gN / pP as in v21
#   smoke     four two-user cells, 10% target, 6 s: n wm wd28 gd28 gn28                              (tag ltm)
#   cl SEEDS  closed loop: four two-user cells at 10% and 1%, eight at 1%                            (lta ltb ltc)
#   long SEEDS  10% target in runs of 100 s: layer-1 latency and misses in steady state             (tag ltg)
# usage: bash v22_slice2.sh STEP [SEEDS]      (env: POLS, NRXB)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl5.sh
P=${POLS:-"n wm wd16 wd28 gn28 gd28"}
step=$1; seeds=${2:-"1 2"}
[ "$step" != smoke ] && [ -f run_state/backstop_slot/queue_b/SKIP_SLICE2 ] && { echo "the smoke test of the second SM slice campaign failed: step $step skipped"; exit 0; }
case $step in
  smoke) PERIODS=2400 bash $C ltm 0.10 "1" "${POLS:-n wm wd28 gd28 gn28}" ;;
  cl)    for seed in $seeds; do
           bash $C lta 0.10 "$seed" "$P"
           bash $C ltb 0.01 "$seed" "$P"
           WEAK=0.5 bash $C ltc 0.01 "$seed" "$P"
         done ;;
  long)  PERIODS=40000 bash $C ltg 0.10 "${2:-1 2 3}" "${POLS:-n wm wd28 gn28}" ;;
esac
echo V22_SLICE2_DONE $step
