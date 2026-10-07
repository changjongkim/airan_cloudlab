#!/usr/bin/env bash
# Tables and figures of v14 from the raw runs (login node; no GPU).
#   usage: bash v14_report.sh [STEP...]     (default: all)
#   steps: sweep other model model_new stop mac classes la figs     (not in the default: model_old, the closed form
#   on the two earlier run sets; optimum, the distance to a clairvoyant schedule -> optimum_gap.txt; lacl, closed-loop
#   link adaptation of jobs JL -> la_closed_<tag>.json/.txt and its figure; sched, the scheduling metrics per policy
#   (L1 latency and misses, neural receiver run time, AI request latency, use of the GPU time) -> sched_<tag>_*.json
#   and sched_metrics.txt; levels, the layer-1 latency without a neural receiver, with it, and with AI -> l1_levels.txt,
#   and the busy time of the neural receivers in the closed-loop conditions -> optimum_gap_closed.txt; frontier, AI served
#   against goodput lost for the caps and the settings of the rule -> frontier_closed.txt and eval_frontier.png; esno, the
#   closed-loop runs at other Es/No, on the low-correlation channel and with a changing Es/No -> la_closed_l{v,h,j,w}?.txt,
#   l1_levels_esno.txt, optimum_gap_esno.txt, eval_closed_vary.png; cont, an Es/No that changes from slot to slot, the
#   low-correlation channel in runs of 100 s and mixed channels -> la_closed_l{x,y,z,m}?.txt, la_closed_lgw.txt, la_track.txt,
#   l1_levels_long_low.txt, eval_closed_cont.png; slice, the SM slice option and the green-context baselines ->
#   la_closed_l{s,t}?.txt, slice_time.txt, l1_levels_slice.txt, l1_levels_slice_off.txt, eval_slice.png)
# The figures of the paper: paper/backstop_slot_v14/figures.sh.
# Jobs: J1 seeds 1-2 of the headline conditions and the rule comparison, J2 kinds of AI work (baselines),
# J3 seeds 3-5, other server sizes, link adaptation, the final rule (wm) in the headline conditions,
# J4 the final rule in the other sizes and kinds of AI work, J5 one GPU seeds 3-5 and the conditions of v14g.sh / v14h.sh.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
J1=${J1:-59210955}; J2=${J2:-59225352}; J3=${J3:-59313958}; J4=${J4:-59318975}; J5=${J5:-59321430}; J6=${J6:-59414960}
R=results/backstop_slot; F=docs/current/figures/backstop_v14; S=scripts_for_node/backstop_slot
OUT=$R/v14_report_j${J1}_${J4}.md
OURV=${OUR_V14:-wm}
PY="shifter --image=$AERIAL_IMAGE --env=V14=1 --env=OUR_V14=$OURV python3"
PYD="shifter --image=$AERIAL_IMAGE --env=V14=1 --env=OUR_V14=wd python3"      # more than four cells per GPU: the rule is code wd (v14h.sh)
PLOT="shifter --image=$AERIAL_IMAGE --env=OUR_V14=$OURV --env=SCHEME_NAME=${SCHEME_NAME:-Antiphase} --env=LA_MCS=${LA_MCS:-13,14,15} --env=PYTHONPATH=$PWD/runtime/softwall_same_gpu/sionna_deps --env=MPLCONFIGDIR=$PWD/run_state/backstop_slot/mpl python3 $S/plot_v14.py"
mkdir -p $F run_state/backstop_slot/mpl
steps=${*:-sweep other model model_new stop mac classes la figs}
say() { echo; echo "## $*"; echo; }
cd $S
for step in $steps; do
  case $step in
    sweep)
      { say "Full load (fa), seeds 1-5"; $PY analyze_sweep.py $J1,$J3 fa 16
        say "Load alternates every 2 s (fb), seeds 1-5"; $PY analyze_sweep.py $J1,$J3 fb 16
        say "Load steps at random times (fc), seeds 1-5"; $PY analyze_sweep.py $J1,$J3 fc 16
        say "Random steps by load level (fc)"; $PY analyze_series.py $J1,$J3 fc 16
        say "Eight two-user cells of sixteen (sw)"; $PY analyze_sweep.py $J3,$J4 sw 16
        say "Two two-user cells of sixteen (sx)"; $PY analyze_sweep.py $J4 sx 16
        say "Six two-user cells of sixteen (sy)"; $PY analyze_sweep.py $J4 sy 16
        say "One GPU, four cells, one two-user cell (sa)"; $PY analyze_sweep.py $J3,$J4,$J5 sa 4
        say "One GPU, four cells, two two-user cells (sb)"; $PY analyze_sweep.py $J3,$J4,$J5 sb 4
        say "Two GPUs, eight cells, two two-user cells (sc)"; $PY analyze_sweep.py $J3,$J4 sc 8
        say "Two GPUs, eight cells, four two-user cells (sd)"; $PY analyze_sweep.py $J3,$J4 sd 8
        say "Rule variants against L1 misses, full load (fv): wm = largest units not next to the conventional receiver, ws = only the smallest, wc = 70% cap"; $PY analyze_sweep.py $J3 fv 16
      } > ../../$OUT.sweep 2>&1 ;;
    sched)       # scheduling metrics from the per-process records (README 6.0.1, 6.0.2); J6: the runs of 100 s (v16_long.sh)
      { say "Full load, runs of 100 s without their first 20 s (xa, job $J6), seeds 1-5"
        WARMUP_S=20 $PY analyze_sched.py $J6 xa 16 wd s10 yyr p30 yyp p70 p100
        say "When the TBs past the L1 deadline occur in a run of 100 s: 16 cells (xa), one GPU (xs), 20 cells (xv), 48 cells (xn)"
        $PY analyze_warmup.py $J6 xa 16 wd s10 yyr p30 yyp p70 p100; $PY analyze_warmup.py $J6 xs 4 wd s10 p30
        $PY analyze_warmup.py $J6 xv 20 wd s10 p30; $PY analyze_warmup.py $J6 xn 48 wd s10 p30
        say "One GPU, 4 cells, runs of 100 s without their first 20 s (xs)"; WARMUP_S=20 $PY analyze_sched.py $J6 xs 4 wd s10 p30
        say "20 cells, runs of 100 s without their first 20 s (xv)"; WARMUP_S=20 $PY analyze_sched.py $J6 xv 20 wd s10 p30
        say "48 cells at one third load, runs of 100 s without their first 20 s (xn)"; WARMUP_S=20 $PY analyze_sched.py $J6 xn 48 wd s10 p30
        say "Load alternates every 2 s (fb), runs of 20 s without their first 5 s, seeds 1-5"; WARMUP_S=5 $PY analyze_sched.py $J1,$J3 fb 16 wm s10 yyr p30 yyp e30x70l0 p70
        say "Load steps at random times (fc), runs of 40 s without their first 5 s, seeds 1-5"; WARMUP_S=5 $PY analyze_sched.py $J1,$J3 fc 16 wm s10 yyr p30 yyp e30x50x70l0 p70
        say "Whole runs of 10 s at full load (fa), seeds 1-5: the start of the run is included"; $PY analyze_sched.py $J1,$J3 fa 16 wm s10 yyr p30 yyp p50 p70 p100
        say "Unit sizes next to the conventional receiver (fv): ws = 128 tokens only, wm = 128 and 512, wn = every size"; $PY analyze_sched.py $J3 fv 16 ws wm wn wc
        for r in 8 16 64; do say "AI offered load, rate code $r (ta)"; RATE=$r $PY analyze_sched.py $J5 ta 16 wm s10 p30 p70 p100; done
      } > ../../$R/sched_metrics.txt 2>&1 ;;
    other)       # conditions of v14g.sh / v14h.sh in job J5 (README 6.11)
      { say "AI load 13k / 27k / 106k tokens/s offered (ta)"; $PY analyze_sweep.py $J5 ta 16
        say "AI requests in bursts (tu)"; $PY analyze_sweep.py $J5 tu 16
        say "20 cells, neural receiver bound 7.6 ms (tv)"; $PYD analyze_sweep.py $J5 tv 20
        say "20 cells, neural receiver bound 8.6 ms (to)"; $PY analyze_sweep.py $J5 to 20
        say "32 cells, half load (tm)"; $PYD analyze_sweep.py $J5 tm 32
        say "48 cells, one third load (tn)"; $PYD analyze_sweep.py $J5 tn 48
        say "Load alternates every 0.2 s (tb)"; $PY analyze_sweep.py $J5 tb 16
        say "Load alternates every 5 s (tc)"; $PY analyze_sweep.py $J5 tc 16
        say "Cell bursts, 32 cells, mean busy run 20 ms (tp)"; $PYD analyze_sweep.py $J5 tp 32
        say "Cell bursts, 32 cells, mean busy run 2 s (tq)"; $PYD analyze_sweep.py $J5 tq 32
        say "Cell bursts, 16 cells, mean busy run 20 ms (tr)"; $PY analyze_sweep.py $J5 tr 16
        say "Cell bursts, 16 cells, mean busy run 2 s (ts)"; $PY analyze_sweep.py $J5 ts 16
        say "Unit classes by active cells: 2-s alternation (fb) and random steps (fc), seeds 1-2 of job $J5"; $PY analyze_sweep.py $J5 fb 16; $PY analyze_sweep.py $J5 fc 16
      } > ../../$OUT.other 2>&1 ;;
    model)
      { say "Event simulation against the measured runs (ring 512)"
        $PY analyze_loss_model.py ../../$R/loss_model_ring512_j${J1}_${J4}.json $J1:fa:16 $J3:fa:16 $J4:sx:16 $J4:sy:16 $J3:sw:16 $J4:sw:16 $J3:sa:4 $J4:sa:4 $J3:sb:4 $J4:sb:4 $J3:sc:8 $J4:sc:8 $J3:sd:8 $J4:sd:8
        say "Closed form against the measured runs (ring 512)"
        $PY analyze_chain.py ../../$R/loss_chain_ring512_j${J1}_${J4}.json $J1:fa:16 $J3:fa:16 $J4:sx:16 $J4:sy:16 $J3:sw:16 $J4:sw:16 $J3:sa:4 $J4:sa:4 $J3:sb:4 $J4:sb:4 $J3:sc:8 $J4:sc:8 $J3:sd:8 $J4:sd:8
        say "Closed form without a measured three-slot share (ring 512; c and S from the other seeds)"
        $PY analyze_chain2.py ../../$R/loss_lane_ring512_j${J1}_${J4}.json $J1:fa:16 $J3:fa:16 $J4:sx:16 $J4:sy:16 $J3:sw:16 $J4:sw:16 $J3:sa:4 $J4:sa:4 $J3:sb:4 $J4:sb:4 $J3:sc:8 $J4:sc:8 $J3:sd:8 $J4:sd:8
        say "Lost candidates against the extra run time of the neural receiver (closed form and measured policies, full load)"
        $PY analyze_sensitivity.py ../../$R/sensitivity_fa_j${J1}_${J3}.json $J1,$J3 fa 16 wm wn wr3 wr1 vf s10 s30 p30 p50 p70 p100 yyr yyp
        say "Rules compared by the model (every unit size next to the conventional receiver)"
        $PY choose_rule.py ../../$R/rule_choice_ring512_j${J1}_${J3}.json $J1:fa:16 $J3:sw:16 $J3:sc:8 $J3:sd:8 $J3:sa:4 $J3:sb:4
      } > ../../$OUT.model 2>&1 ;;
    model_old)   # the closed form on the runs of v13 (128-slot replay) and on the 128-slot runs of v14
      { say "Closed form without a measured three-slot share, runs of v13"
        $PY analyze_chain2.py ../../$R/loss_lane_j59192512_59199237.json 59192512:q:16 59199237:wb:16 59199237:wc:16 59199237:wd:16 59199237:ga:4 59199237:gb:8 59199237:da:20 59199237:ab:16
        say "Closed form without a measured three-slot share, 128-slot runs of v14"
        $PY analyze_chain2.py ../../$R/loss_lane_j59210955.json $J1:r:16 $J1:rr:16 $J1:rw:16 $J1:ra:4 $J1:rb:4 $J1:rc:8 $J1:rd:8
      } > ../../$OUT.model_old 2>&1 ;;
    model_new)   # the closed form on the runs of job J5, measured after the model was final (other AI loads, AI bursts,
                 # 20 / 32 / 48 cells, two latest start times, one GPU seeds 3-5, changing loads)
      { say "Closed form without a measured three-slot share, runs of job $J5"
        RATES=8,16,32,64 shifter --image=$AERIAL_IMAGE --env=RATES=8,16,32,64 python3 analyze_chain2.py ../../$R/loss_lane_new_j${J5}.json \
          ${NEW_SPECS:-$J5:ta:16 $J5:tu:16 $J5:to:20 $J5:tv:20 $J5:tm:32 $J5:tn:48 $J5:sa:4 $J5:sb:4 $J5:fb:16 $J5:fc:16 $J5:tb:16 $J5:tc:16 $J5:tp:32 $J5:tq:32 $J5:tr:16 $J5:ts:16}
      } > ../../$OUT.model_new 2>&1 ;;
    stop)
      { say "AI left on after it must stop (full load, seeds 1-2)"; $PY analyze_stop.py $J1 fa 16 vf:3 wr3:3 wr1:1 wn:4
        say "The same, job of seeds 3-5 (wm: all five seeds)"; $PY analyze_stop.py $J3 fa 16 wr3:3 wn:4 wm:4
        say "L1 misses by what else ran on the GPU (full load)"; $PY analyze_l1.py fa 16 $J1,$J3 n wm wn wr3 s10 p30 p70 p100
        say "The same, rule variants of one job (fv)"; $PY analyze_l1.py fv 16 $J3 n wn wm ws wc
      } > ../../$OUT.stop 2>&1 ;;
    mac)
      { say "MAC-level goodput, full load, seeds 1-5"; $PY mac_goodput.py ../../$R/mac_goodput_fa_j${J1}_${J3}.json $J1,$J3:fa:16
        say "MAC-level goodput, 2-s alternation and random steps, seeds 1-5"; $PY mac_goodput.py ../../$R/mac_goodput_fb_fc_j${J1}_${J3}.json $J1,$J3:fb:16 $J1,$J3:fc:16
        say "MAC-level goodput, eight two-user cells and two GPUs"; $PY mac_goodput.py ../../$R/mac_goodput_sw_sd_j${J3}_${J4}.json $J3,$J4:sw:16 $J3,$J4:sd:8
        say "MAC-level goodput, 20 / 32 / 48 cells"; $PYD mac_goodput.py ../../$R/mac_goodput_tv_tm_tn_j${J5}.json $J5:tv:20 $J5:tm:32 $J5:tn:48
        say "Goodput against recoveries kept: cost of one percent of lost recoveries"
        python3.11 analyze_goodput_slope.py ../../$R/goodput_slope.json fa_c16=../../$R/mac_goodput_fa_j${J1}_${J3}.json:../../$R/sweep_fa_c16_j${J1}_${J3}.json \
          fb_c16=../../$R/mac_goodput_fb_fc_j${J1}_${J3}.json:../../$R/sweep_fb_c16_j${J1}_${J3}.json fc_c16=../../$R/mac_goodput_fb_fc_j${J1}_${J3}.json:../../$R/sweep_fc_c16_j${J1}_${J3}.json \
          sw_c16=../../$R/mac_goodput_sw_sd_j${J3}_${J4}.json:../../$R/sweep_sw_c16_j${J3}_${J4}.json sd_c8=../../$R/mac_goodput_sw_sd_j${J3}_${J4}.json:../../$R/sweep_sd_c8_j${J3}_${J4}.json \
          tv_c20=../../$R/mac_goodput_tv_tm_tn_j${J5}.json:../../$R/sweep_tv_c20_j${J5}.json tm_c32=../../$R/mac_goodput_tv_tm_tn_j${J5}.json:../../$R/sweep_tm_c32_j${J5}.json \
          tn_c48=../../$R/mac_goodput_tv_tm_tn_j${J5}.json:../../$R/sweep_tn_c48_j${J5}.json
      } > ../../$OUT.mac 2>&1 ;;
    classes)
      { say "Five kinds of AI work (kb), every policy in job $J4"; $PY analyze_classes5.py $J4 kb 16
        say "Chat only (ka), every policy in job $J4"; $PY analyze_classes5.py $J4 ka 16
        say "Five kinds of AI work, baselines of job $J2 (unit-size boundary 0.90 ms)"; $PY analyze_classes5.py $J2 kb 16
      } > ../../$OUT.classes 2>&1 ;;
    la)
      { say "Link adaptation, DoubleTDLlow"; $PY analyze_la.py DoubleTDLlow; python3.11 table_la.py ../../$R/link_adaptation_DoubleTDLlow.json
        say "Link adaptation, DoubleTDLhigh"; $PY analyze_la.py DoubleTDLhigh; python3.11 table_la.py ../../$R/link_adaptation_DoubleTDLhigh.json
        say "Link adaptation, UMi"; $PY analyze_la.py UMi; python3.11 table_la.py ../../$R/link_adaptation_UMi.json
      } > ../../$OUT.la 2>&1 ;;
    optimum)     # README 6.12: AI time and AI served of an offline schedule that knows every candidate
      shifter --image=$AERIAL_IMAGE --env=V14=1 --env=OTHERS=wn,wr3,wr1,vf,s10,p30,p50,p70,yyr,yyp,d10x50l0,e30x70l0,d10x30x50l0,e30x50x70l0 \
        python3 analyze_optimum.py ../../$R/optimum_gap.json $J1,$J3:fa:16 $J1,$J3:fb:16 $J1,$J3:fc:16 $J4:sx:16 $J4:sy:16 $J3,$J4:sw:16 \
        $J3,$J4:sc:8 $J3,$J4:sd:8 $J3,$J4,$J5:sa:4 $J3,$J4,$J5:sb:4 $J5:tv:20:wd $J5:tm:32:wd $J5:tn:48:wd > ../../$R/optimum_gap.txt 2>&1 ;;
    frontier)    # README 6.13.7: AI served against goodput lost for the caps of the low-priority baseline and the settings
                 # of the rule (v17_frontier.sh, jobs JF)
      for tag in lfa lfb lfc lfd; do
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json ${JF:-59423316,59439086} $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      $PY analyze_frontier.py ../../$R/frontier_closed.json "Four two-user cells, target 10%=../../$R/la_closed_lfa.json" \
        "Four two-user cells, target 1%=../../$R/la_closed_lfb.json" "Eight two-user cells, target 3%=../../$R/la_closed_lfd.json" \
        "Eight two-user cells, target 1%=../../$R/la_closed_lfc.json" > ../../$R/frontier_closed.txt 2>&1
      cd ../..
      ${PLOT/plot_v14.py/plot_eval.py} frontier $F/eval_frontier.png "Target 10%=$R/la_closed_lfa.json" "Target 1%=$R/la_closed_lfb.json" \
        "Target 3%, eight two-user cells=$R/la_closed_lfd.json" "Target 1%, eight two-user cells=$R/la_closed_lfc.json"
      cd $S ;;
    esno)        # README 6.13.8: closed-loop link adaptation at other Es/No, on the low-correlation channel, and with an Es/No
                 # that changes every 2 s (v19_esno.sh; jobs JV and JW)
      JV=${JV:-59429165}; JW=${JW:-59439086,59473418}      # the second job of JW: seed 3 of the low-correlation channel
      for spec in lva:$JV lvb:$JV lvc:$JW lha:$JV lhb:$JV lhc:$JV lja:$JV ljb:$JV ljc:$JV lwa:$JW lwb:$JW lwc:$JW; do
        tag=${spec%%:*}
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json ${spec#*:} $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      $PY analyze_l1_levels.py ../../$R/l1_levels_esno.json $JV,$JW "lva:16:Changing Es/No, target 10%" "lvb:16:Changing Es/No, target 1%" \
        "lvc:16:Changing Es/No, target 1%, eight two-user cells" "lha:16:14 dB, target 10%" "lhb:16:14 dB, target 1%" \
        "lhc:16:14 dB, target 1%, eight two-user cells" "lja:16:20 dB, target 10%" "ljb:16:20 dB, target 1%" \
        "ljc:16:20 dB, target 1%, eight two-user cells" "lwa:16:Low correlation 6 dB, target 10%" "lwb:16:Low correlation 6 dB, target 1%" \
        "lwc:16:Low correlation 6 dB, target 1%, eight two-user cells" > ../../$R/l1_levels_esno.txt 2>&1
      $PY analyze_optimum.py ../../$R/optimum_gap_esno.json $JV:lva:16 $JV:lvb:16 $JW:lvc:16 $JV:lha:16 $JV:lhb:16 $JV:lhc:16 \
        $JV:lja:16 $JV:ljb:16 $JV:ljc:16 $JW:lwa:16 $JW:lwb:16 $JW:lwc:16 > ../../$R/optimum_gap_esno.txt 2>&1
      cd ../..
      E="${PLOT/plot_v14.py/plot_eval.py}"
      $E closed $F/eval_closed_vary.png "Changing Es/No\nTarget 10%=$R/la_closed_lva.json" "Changing Es/No\nTarget 1%=$R/la_closed_lvb.json" \
        "Changing, 8 Cells\nTarget 1%=$R/la_closed_lvc.json" "14 dB, 8 Cells\nTarget 1%=$R/la_closed_lhc.json" \
        "20 dB, 8 Cells\nTarget 1%=$R/la_closed_ljc.json" "Low Correlation\nTarget 10%=$R/la_closed_lwa.json"
      cd $S ;;
    cont)        # README 6.13.9: an Es/No that changes from slot to slot (v20_cont.sh slow / walk / fast; jobs JC), the
                 # low-correlation channel in runs of 100 s and two-user cells of two channels on one server (jobs JD).
                 # SEEDS_SLOW / SEEDS_WALK / SEEDS_FAST keep those seeds while later seeds are still running.
      JC=${JC:-59465023,59472374}; JD=${JD:-59466250,59473418}
      for spec in lxa:SLOW lxb:SLOW lxc:SLOW lza:WALK lzb:WALK lzc:WALK lya:FAST lyb:FAST lyc:FAST; do
        tag=${spec%%:*}; var=SEEDS_${spec#*:}; pick="${!var:+--env=SEEDS=${!var} }"      # shifter rejects an empty value
        ${PY/python3/${pick}python3} analyze_la_closed.py ../../$R/la_closed_$tag.json $JC $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      $PY analyze_la_track.py ../../$R/la_track.json $JC "lxa:16:Slow change (period 5.12 s), target 10%" "lxb:16:Slow change, target 1%" \
        "lya:16:Fast change (period 1.28 s), target 10%" "lyb:16:Fast change, target 1%" "lza:16:Random change, target 10%" \
        "lzb:16:Random change, target 1%" > ../../$R/la_track.txt 2>&1
      for tag in lgw lma lmb lmc lpa lpb lpc lqa lqb lqc; do      # lp?: low-correlation channel at 4.5 dB, lq?: at 7.5 dB
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json $JD $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      for tag in lua lub luc; do                                  # urban micro channel at 10 dB: 10% and 3% targets
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json $JC $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      ${PY/python3/--env=SKIP=8000 --env=LATE=1 python3} analyze_l1_levels.py ../../$R/l1_levels_long_low.json $JD \
        "lgw:16:Low-correlation channel at 6 dB, target 10%, runs of 100 s" > ../../$R/l1_levels_long_low.txt 2>&1
      cd ../..
      E="${PLOT/plot_v14.py/plot_eval.py}"
      $E closed $F/eval_closed_cont.png "Slow Change\nTarget 10%=$R/la_closed_lxa.json" "Slow, 8 Cells\nTarget 1%=$R/la_closed_lxc.json" \
        "Random Change\nTarget 10%=$R/la_closed_lza.json" "Random, 8 Cells\nTarget 1%=$R/la_closed_lzc.json" \
        "Fast, 8 Cells\nTarget 1%=$R/la_closed_lyc.json" "Mixed Channels\n8 Cells, Target 1%=$R/la_closed_lmc.json"
      cd $S ;;
    slice)       # README 6.13.10: the SM slice option and the green-context baselines.  v21_slice.sh: the AI slice shares
                 # its SMs with the neural receiver (jobs JS, seed 1 in the first and seed 2 in the second; runs of 100 s in
                 # job JSL); v22_slice2.sh: the neural receivers run on the other SMs (job JT; runs of 100 s in job JTL).
      JS=${JS:-59472374,59478477}; JSL=${JSL:-59478477}; JT=${JT:-59478478}; JTL=${JTL:-59478478}
      # eight two-user cells at the 1% target with seeds 4-9 (la_cl4.sh lec; job JE): the rule, no AI, the 30% share
      $PY analyze_la_closed.py ../../$R/la_closed_lec.json ${JE:-59482496} lec 16 > ../../$R/la_closed_lec.txt 2>&1
      for tag in lsa lsb lsc; do
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json $JS $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      for tag in lta ltb ltc; do
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json $JT $tag 16 > ../../$R/la_closed_$tag.txt 2>&1
      done
      $PY analyze_slice.py ../../$R/slice_time_s1.json ${JS%%,*} lsa,lsb,lsc 1 > ../../$R/slice_time.txt 2>&1
      $PY analyze_slice.py ../../$R/slice_time_s2.json ${JS##*,} lsa,lsb,lsc 2 >> ../../$R/slice_time.txt 2>&1
      $PY analyze_slice.py ../../$R/slice_time_off.json $JT lta,ltb,ltc 1,2 >> ../../$R/slice_time.txt 2>&1
      $PY analyze_la_closed.py ../../$R/la_closed_lsg.json $JSL lsg 16 > ../../$R/la_closed_lsg.txt 2>&1
      ${PY/python3/--env=SKIP=8000 --env=LATE=1 python3} analyze_l1_levels.py ../../$R/l1_levels_slice.json $JSL \
        "lsg:16:Target 10%, runs of 100 s, AI slice shared with the neural receiver" > ../../$R/l1_levels_slice.txt 2>&1
      $PY analyze_la_closed.py ../../$R/la_closed_ltg.json $JTL ltg 16 > ../../$R/la_closed_ltg.txt 2>&1
      ${PY/python3/--env=SKIP=8000 --env=LATE=1 python3} analyze_l1_levels.py ../../$R/l1_levels_slice_off.json $JTL \
        "ltg:16:Target 10%, runs of 100 s, AI always on 28 SMs" > ../../$R/l1_levels_slice_off.txt 2>&1
      cd ../..
      ${PLOT/plot_v14.py/plot_eval.py} slice $F/eval_slice.png $R
      cd $S ;;
    levels)      # README 6.0.1: what the neural receiver and the AI each add to the layer-1 latency (single-user cells of the
                 # closed-loop runs, which include the server without a neural receiver), and README 6.13.6: the share of the
                 # time in which the neural receivers run in the closed-loop conditions
      $PY analyze_l1_levels.py ../../$R/l1_levels.json ${JL:-59345188,59362400} "laa:16:Target 10%" "lac:16:Target 3%" "lab:16:Target 1%" \
        "laf:16:Target 1%, eight two-user cells" > ../../$R/l1_levels.txt 2>&1
      J=${JL:-59345188,59362400}
      $PY analyze_optimum.py ../../$R/optimum_gap_closed.json $J:laa:16 $J:lac:16 $J:lab:16 $J:laf:16 > ../../$R/optimum_gap_closed.txt 2>&1
      # the 10% target in runs of 100 s (v18_long_cl.sh, job JG): steady state, with the TBs past the deadline
      ${PY/python3/--env=SKIP=8000 --env=LATE=1 python3} analyze_l1_levels.py ../../$R/l1_levels_long_cl.json ${JG:-59423316} \
        "lga:16:Target 10%, runs of 100 s" > ../../$R/l1_levels_long_cl.txt 2>&1
      $PY analyze_la_closed.py ../../$R/la_closed_lga.json ${JG:-59423316} lga 16 > ../../$R/la_closed_lga.txt 2>&1
      # full load at a fixed MCS in runs of 100 s: v16_long.sh (job J6) and the conventional receiver alone (v16_long_x.sh, job JG)
      ${PY/python3/--env=SKIP=8000 --env=ALL_CELLS=1 --env=LATE=1 python3} analyze_l1_levels.py ../../$R/l1_levels_long_full.json $J6,${JG:-59423316} \
        "xa:16:16 cells at full load, fixed MCS, runs of 100 s" > ../../$R/l1_levels_long_full.txt 2>&1 ;;
    lacl)        # README 6.13: closed-loop link adaptation (la_cl.sh, la_cl2.sh), jobs JL (comma-separated)
      for spec in ${LACL_TAGS:-laa:16 lac:16 lab:16 laf:16 lai:16 lae:8 lah:8 lad:8 lag:4 lak:16 lal:16}; do
        tag=${spec%%:*}
        $PY analyze_la_closed.py ../../$R/la_closed_$tag.json ${JL:-59345188,59362400} $tag ${spec#*:} > ../../$R/la_closed_$tag.txt 2>&1
      done
      cd ../..
      $PLOT closedloop $F/closed_loop_link_adaptation.png "Target 10%=$R/la_closed_laa.json" "Target 3%=$R/la_closed_lac.json" \
        "Target 1%=$R/la_closed_lab.json" "Target 1%, eight two-user cells=$R/la_closed_laf.json"
      cd $S ;;
    figs)
      cd ../..
      $PLOT model $F/loss_model.png $R/loss_model_j59192512_59199237.json $R/loss_model_j59210955.json $R/loss_model_ring512_j${J1}_${J4}.json -- \
        $R/loss_lane_j59192512_59199237.json $R/loss_lane_j59210955.json $R/loss_lane_ring512_j${J1}_${J4}.json $R/loss_lane_new_j${J5}.json
      $PLOT sensitivity $F/loss_sensitivity.png $R/sensitivity_fa_j${J1}_${J3}.json
      $PLOT tradeoff $F/tradeoff_full_load.png "Full load, 16 cells, four GPUs" $R/sweep_fa_c16_j${J1}_${J3}.json
      $PLOT headline $F/headline_load_patterns.png "Full load=$R/sweep_fa_c16_j${J1}_${J3}.json" \
        "Load alternates every 2 s=$R/sweep_fb_c16_j${J1}_${J3}.json" "Load steps at random times=$R/sweep_fc_c16_j${J1}_${J3}.json"
      $PLOT gpus $F/server_sizes.png "1 GPU, 4 cells\\n1 two-user cell=$R/sweep_sa_c4_j${J3}_${J4}_${J5}.json" "1 GPU, 4 cells\\n2 two-user cells=$R/sweep_sb_c4_j${J3}_${J4}_${J5}.json" \
        "2 GPUs, 8 cells\\n2 two-user cells=$R/sweep_sc_c8_j${J3}_${J4}.json" "2 GPUs, 8 cells\\n4 two-user cells=$R/sweep_sd_c8_j${J3}_${J4}.json" \
        "4 GPUs, 16 cells\\n4 two-user cells=$R/sweep_fa_c16_j${J1}_${J3}.json" "4 GPUs, 16 cells\\n8 two-user cells=$R/sweep_sw_c16_j${J3}_${J4}.json"
      $PLOT classes $F/kinds_of_ai_work.png $R/classes5_kb_c16_j${J4}.json
      $PLOT la $F/link_adaptation.png $R/link_adaptation_DoubleTDLlow.json
      $PLOT la $F/link_adaptation_high_correlation.png $R/link_adaptation_DoubleTDLhigh.json
      $PLOT timeline $F/timeline_server.png fa1r32wm_c16_rescue_value_backstop_units_j${J3} 10 205
      EVAL="${PLOT/plot_v14.py/plot_eval.py}"         # evaluation figures: one metric per panel, one color per policy
      FA=$R/sweep_fa_c16_j${J1}_${J3}.json
      $EVAL headline $F/eval_headline.png $FA $R/sweep_fb_c16_j${J1}_${J3}.json $R/sweep_fc_c16_j${J1}_${J3}.json
      $EVAL tradeoff $F/eval_tradeoff.png $FA
      $EVAL scale $F/eval_scale.png gpus=$R/sweep_sa_c4_j${J3}_${J4}_${J5}.json,$R/sweep_sc_c8_j${J3}_${J4}.json,$FA \
        cells=$FA,$R/sweep_tv_c20_j${J5}.json,$R/sweep_tm_c32_j${J5}.json,$R/sweep_tn_c48_j${J5}.json \
        demand=$R/sweep_sx_c16_j${J4}.json,$FA,$R/sweep_sy_c16_j${J4}.json,$R/sweep_sw_c16_j${J3}_${J4}.json
      $EVAL use $F/eval_gpu_use.png $R/optimum_gap.json
      $EVAL protect $F/eval_protect.png "Steady Full Load\n(100-s Runs)=$R/sched_xa_c16_j${J6}_w20.json" "Load Changes Every 2 s\n(20-s Runs)=$R/sched_fb_c16_j${J1}_${J3}_w5.json"
      $EVAL closed $F/eval_closed_loop.png "Target 10%=$R/la_closed_laa.json" "Target 3%=$R/la_closed_lac.json" "Target 1%=$R/la_closed_lab.json" \
        "Target 1%\\n2 GPUs=$R/la_closed_lad.json" "Target 1%\\n8 two-user cells=$R/la_closed_laf.json"
      cd $S ;;
  esac
done
cd ../..
cat $OUT.sweep $OUT.other $OUT.model $OUT.model_new $OUT.stop $OUT.mac $OUT.classes $OUT.la 2>/dev/null > $OUT
echo "report: $OUT"
