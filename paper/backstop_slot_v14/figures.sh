#!/usr/bin/env bash
# Figures of the paper (login node; no GPU).  usage: bash figures.sh [pdf|png]   (png: previews in run_state)
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh >/dev/null 2>&1
EXT=${1:-pdf}
R=$PWD/results/backstop_slot; S=scripts_for_node/backstop_slot
F=$PWD/paper/backstop_slot_v14/figures; [ $EXT = png ] && F=$PWD/run_state/backstop_slot/paper_preview
J1=59210955; J3=59313958; J4=59318975; J5=59321430
mkdir -p $F run_state/backstop_slot/mpl
run() { local script=$1; shift
  ( cd $S && shifter --image=$AERIAL_IMAGE --env=OUR_V14=${OUR_V14:-wm} --env=SCHEME_NAME=Antiphase --env=FIG_SCALE=${FIG_SCALE:-1} \
    --env=LEGEND_COLS=${LEGEND_COLS:-2} --env=LA_MCS=${LA_MCS:-13,14,15} --env=PYTHONPATH=$SOFTWALL_ROOT/runtime/softwall_same_gpu/sionna_deps \
    --env=MPLCONFIGDIR=$SOFTWALL_ROOT/run_state/backstop_slot/mpl python3 $script "$@" ); }
run plot_paper.py model $F/loss_model.$EXT $R/loss_model_j59192512_59199237.json $R/loss_model_j59210955.json $R/loss_model_ring512_j${J1}_${J4}.json -- \
  $R/loss_lane_j59192512_59199237.json $R/loss_lane_j59210955.json $R/loss_lane_ring512_j${J1}_${J4}.json $R/loss_lane_new_j${J5}.json
run plot_paper.py sensitivity $F/loss_sensitivity.$EXT $R/sensitivity_fa_j${J1}_${J3}.json
run plot_paper.py tradeoff $F/tradeoff_full_load.$EXT $R/sweep_fa_c16_j${J1}_${J3}.json
run plot_paper.py bars $F/headline_load_patterns.$EXT OURS,s10,p30,p70,d,e,yyr,yyp "Steady full load=$R/sweep_fa_c16_j${J1}_${J3}.json" \
  "Full and half load, every 2 s=$R/sweep_fb_c16_j${J1}_${J3}.json" "Random load steps=$R/sweep_fc_c16_j${J1}_${J3}.json"
LEGEND_COLS=3 run plot_paper.py bars $F/server_sizes.$EXT OURS,s10,p30,p70 "1 GPU\\n4 cells\\ntwo-user: 1=$R/sweep_sa_c4_j${J3}_${J4}_${J5}.json" "1 GPU\\n4 cells\\ntwo-user: 2=$R/sweep_sb_c4_j${J3}_${J4}_${J5}.json" \
  "2 GPUs\\n8 cells\\ntwo-user: 2=$R/sweep_sc_c8_j${J3}_${J4}.json" "2 GPUs\\n8 cells\\ntwo-user: 4=$R/sweep_sd_c8_j${J3}_${J4}.json" \
  "4 GPUs\\n16 cells\\ntwo-user: 4=$R/sweep_fa_c16_j${J1}_${J3}.json" "4 GPUs\\n16 cells\\ntwo-user: 8=$R/sweep_sw_c16_j${J3}_${J4}.json"
run plot_paper.py la $F/link_adaptation.$EXT $R/link_adaptation_DoubleTDLlow.json
FIG_SCALE=0.75 run plot_v14.py timeline $F/timeline_server.$EXT fa1r32wm_c16_rescue_value_backstop_units_j${J3} 10 205
# The architecture figure for the README (TikZ source in figures/architecture.tex)
if [ $EXT = pdf ]; then
  B=$SOFTWALL_ROOT/run_state/backstop_slot/figbuild; mkdir -p $B
  ( cd $F && PATH=/global/common/software/nersc9/texlive/2024/bin/x86_64-linux:$PATH pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$B architecture_standalone.tex >/dev/null ) &&
    gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r220 -sOutputFile=$SOFTWALL_ROOT/docs/current/figures/backstop_v14/architecture.png $B/architecture_standalone.pdf
fi
ls $F
