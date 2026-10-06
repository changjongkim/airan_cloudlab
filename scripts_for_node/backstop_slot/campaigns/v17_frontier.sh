#!/usr/bin/env bash
# v17: the trade-off between the AI served and the cell goodput under closed-loop link adaptation, for the caps of
# the low-priority baseline (20-100%) and for the settings of the rule (no AI next to a neural receiver; AI next to
# it while 3 / 2 / 1 neural receivers of the server are free).  Every run is a run of la_cl2.sh; the scheme and its
# options are unchanged.  All policies of a condition run in one job.
#   lfa  four two-user cells, target 10%      lfb  four two-user cells, target 1%
#   lfc  eight two-user cells, target 1%      lfd  eight two-user cells, target 3%
# usage: bash v17_frontier.sh "SEEDS" ["TAGS"]      (env: POLS, DATA)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
C=run_state/backstop_slot/la_cl2.sh
P=${POLS:-"n wm wr3 wr2 wr1 s10 p20 p30 p40 p50 p60 p70 p100"}
for seed in $1; do
  for tag in ${2:-lfa lfb lfc lfd}; do
    case $tag in
      lfa) bash $C lfa 0.10 "$seed" "$P" ;;
      lfb) bash $C lfb 0.01 "$seed" "$P" ;;
      lfc) WEAK=0.5 bash $C lfc 0.01 "$seed" "$P" ;;
      lfd) WEAK=0.5 bash $C lfd 0.03 "$seed" "$P" ;;
    esac
  done
done
echo V17_FRONTIER_DONE
