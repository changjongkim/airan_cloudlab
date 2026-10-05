#!/usr/bin/env bash
# Tables and figures of v14 from the raw runs (login node; no GPU).
#   usage: bash v14_report.sh [STEP...]     (default: all)
#   steps: sweep other model model_new stop mac classes la figs     (not in the default: model_old, the closed form
#   on the two earlier run sets; optimum, the distance to a clairvoyant schedule -> optimum_gap.txt; lacl, closed-loop
#   link adaptation of jobs JL -> la_closed_<tag>.json/.txt and its figure)
# The figures of the paper: paper/backstop_slot_v14/figures.sh.
# Jobs: J1 seeds 1-2 of the headline conditions and the rule comparison, J2 kinds of AI work (baselines),
# J3 seeds 3-5, other server sizes, link adaptation, the final rule (wm) in the headline conditions,
# J4 the final rule in the other sizes and kinds of AI work, J5 one GPU seeds 3-5 and the conditions of v14g.sh / v14h.sh.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
J1=${J1:-59210955}; J2=${J2:-59225352}; J3=${J3:-59313958}; J4=${J4:-59318975}; J5=${J5:-59321430}
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
      cd $S ;;
  esac
done
cd ../..
cat $OUT.sweep $OUT.other $OUT.model $OUT.model_new $OUT.stop $OUT.mac $OUT.classes $OUT.la 2>/dev/null > $OUT
echo "report: $OUT"
