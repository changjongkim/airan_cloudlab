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
run plot_paper.py la $F/link_adaptation.$EXT $R/link_adaptation_DoubleTDLlow.json
FIG_SCALE=0.75 run plot_v14.py timeline $F/timeline_server.$EXT fa1r32wm_c16_rescue_value_backstop_units_j${J3} 10 205
# Evaluation figures (one metric per panel, one color per policy)
EVAL() { ( cd $S && shifter --image=$AERIAL_IMAGE --env=OUR_V14=${OUR_V14:-wm} --env=SCHEME_NAME=Antiphase \
    --env=PYTHONPATH=$SOFTWALL_ROOT/runtime/softwall_same_gpu/sionna_deps --env=MPLCONFIGDIR=$SOFTWALL_ROOT/run_state/backstop_slot/mpl python3 plot_eval.py "$@" ); }
FA=$R/sweep_fa_c16_j${J1}_${J3}.json
EVAL headline $F/eval_headline.$EXT $FA $R/sweep_fb_c16_j${J1}_${J3}.json $R/sweep_fc_c16_j${J1}_${J3}.json
EVAL tradeoff $F/eval_tradeoff.$EXT $FA
EVAL scale $F/eval_scale.$EXT gpus=$R/sweep_sa_c4_j${J3}_${J4}_${J5}.json,$R/sweep_sc_c8_j${J3}_${J4}.json,$FA \
  cells=$FA,$R/sweep_tv_c20_j${J5}.json,$R/sweep_tm_c32_j${J5}.json,$R/sweep_tn_c48_j${J5}.json \
  demand=$R/sweep_sx_c16_j${J4}.json,$FA,$R/sweep_sy_c16_j${J4}.json,$R/sweep_sw_c16_j${J3}_${J4}.json
EVAL use $F/eval_gpu_use.$EXT $R/optimum_gap.json
EVAL protect $F/eval_protect.$EXT "Steady Full Load\n(100-s Runs)=$R/sched_xa_c16_j${J6:-59414960}_w20.json" "Load Changes Every 2 s\n(20-s Runs)=$R/sched_fb_c16_j${J1}_${J3}_w5.json"
EVAL closed $F/eval_closed_loop.$EXT "Target 10%=$R/la_closed_laa.json" "Target 3%=$R/la_closed_lac.json" "Target 1%=$R/la_closed_lab.json" \
  "Target 1%\\n2 GPUs=$R/la_closed_lad.json" "Target 1%\\n8 two-user cells=$R/la_closed_laf.json"
# The design figures for the README (TikZ sources in figures/*.tex)
if [ $EXT = pdf ]; then
  B=$SOFTWALL_ROOT/run_state/backstop_slot/figbuild; mkdir -p $B; D=$SOFTWALL_ROOT/docs/current/figures/backstop_v14
  for pair in architecture:architecture recovery_path:recovery_path hold:loss_two_or_three_slots admission:ai_admission; do
    src=${pair%%:*}; dst=${pair#*:}
    ( cd $F && PATH=/global/common/software/nersc9/texlive/2024/bin/x86_64-linux:$PATH pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory=$B -jobname=$src "\def\figfile{$src.tex}\input{standalone.tex}" >/dev/null ) &&
      gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r1200 -dDownScaleFactor=4 -sOutputFile=$D/$dst.png $B/$src.pdf   # 300 dpi, smooth edges
  done
fi
ls $F
