#!/usr/bin/env bash
# v20: closed-loop link adaptation with an Es/No that changes from slot to slot, and more of the low-correlation
# channel.  The scheme and its options are unchanged; every run is a run of la_cl3.sh.
#   synth     slots at 16 and 14 dB made from the 20 dB slots by added noise (make_esno_dataset.py), and both
#             receivers on them: must match the slots generated at 16 and 14 dB (queue/check_synth.sh)
#   traj      the datasets of three Es/No trajectories between 14 and 20 dB: cslow (sine, period 5.12 s), cfast
#             (sine, period 1.28 s), cwalk (shadow fading: 1.75 dB, correlation time 1 s)
#   slow / walk / fast   SEED: four two-user cells at the 10% and 1% targets, eight at 1% (lxa lxb lxc / lza lzb lzc /
#             lya lyb lyc), runs of 40 s, eight policies
#   long_low  low-correlation channel at 6 dB, 10% target, runs of 100 s (lgw): layer-1 latency in steady state
#   mixed     two-user cells of two channels on one server: high correlation 16 dB and low correlation 6 dB in turn
#             (lma: four cells 10%, lmb: four cells 1%, lmc: eight cells 1%)
#   lowsyn    low-correlation slots at 6, 4.5 and 7.5 dB from the slots generated at 8 dB; receivers on the 6 dB set
#   low45 / low75   closed loop on them (lpa lpb lpc / lqa lqb lqc)
#   umi       urban micro channel at 10 dB (lua: 10%, lub: 3%, luc: eight cells 3%)
#   low6s3    seed 3 of the low-correlation channel at 6 dB (lwa lwb lwc)
# usage: bash v20_cont.sh STEP [SEEDS]
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh
R=$SOFTWALL_ROOT/run_state/backstop_slot; C=$R/la_cl3.sh; Q=$R/queue
LARGE=/softwall_runtime/engines/nv/nrx_large_273prb_2ue.trt
BUILD="shifter --image=$AERIAL_IMAGE python3 scripts_for_node/backstop_slot/make_esno_dataset.py"
P="x n wm s10 p30 p70 p100"; PV="x n wm s10 p30 p50 p70 p100"
step=$1; seeds=${2:-"1 2"}
needs_synth() { [ -f $Q/SKIP_SYNTH ] && { echo "the synthesized slots did not pass their check: step $step skipped"; exit 0; }; }
probe() {      # both receivers on every MCS level of dataset_la_$1
  local tag=$1 mcs n=0
  for mcs in 10 11 12 13 14 15 16; do
    ( shifter_slot python3 /backstop_slot/probe_mu.py --dataset $R/dataset_la_$tag --profile nv_mu2_m$mcs --engine $LARGE \
        --conv-iterations 20 --ldpc-iterations 20 --passes 2 --per-slot $SLOT_RESULTS/raw/lacl_${tag}_m${mcs}_slots.npz \
        --output $SLOT_RESULTS/raw/lacl_${tag}_m${mcs}.json 2>&1 | grep -E 'rror|Traceback' | head -3 ) &
    n=$((n + 1)); [ $((n % 4)) -eq 0 ] && wait
  done
  wait
}
three() {      # three DATA A B C TARGET_AB2 EXTRA : four two-user cells at 10% and at $5, eight at $5
  local data=$1 a=$2 b=$3 c=$4 low=${5:-0.01} seed
  for seed in $seeds; do
    DATA=$data bash $C $a 0.10 "$seed" "$P"
    DATA=$data bash $C $b $low "$seed" "$P"
    DATA=$data WEAK=0.5 bash $C $c $low "$seed" "$P"
  done
}
moving() {     # moving DATA RING A B C : an Es/No trajectory, runs of 40 s
  local data=$1 ring=$2 a=$3 b=$4 c=$5 seed
  for seed in $seeds; do
    DATA=$data EXTRA="--ring $ring" PERIODS=16000 bash $C $a 0.10 "$seed" "$PV"
    DATA=$data EXTRA="--ring $ring" PERIODS=16000 bash $C $b 0.01 "$seed" "$PV"
    DATA=$data EXTRA="--ring $ring" PERIODS=16000 WEAK=0.5 bash $C $c 0.01 "$seed" "$PV"
  done
}
case $step in
  synth)   $BUILD --base $R/dataset_la_high20 --base-esno 20 --output $R/dataset_la_syn16 --trajectory const:16
           $BUILD --base $R/dataset_la_high20 --base-esno 20 --output $R/dataset_la_syn14 --trajectory const:14
           probe syn16; probe syn14 ;;
  traj)    needs_synth
           $BUILD --base $R/dataset_la_high20 --base-esno 20 --output $R/dataset_la_cslow --trajectory sine:14:20 --slots 2048
           $BUILD --base $R/dataset_la_high20 --base-esno 20 --output $R/dataset_la_cfast --trajectory sine:14:20 --slots 512
           $BUILD --base $R/dataset_la_high20 --base-esno 20 --output $R/dataset_la_cwalk --trajectory walk:16.5:1.75:400:13:20 --slots 2048 ;;
  slow)    needs_synth; moving cslow 2048 lxa lxb lxc ;;
  fast)    needs_synth; moving cfast 512 lya lyb lyc ;;
  walk)    needs_synth; moving cwalk 2048 lza lzb lzc ;;
  long_low) DATA=low6 PERIODS=40000 bash $C lgw 0.10 "${2:-1 2 3}" "$P" ;;
  mixed)   for seed in $seeds; do
             STATES="high16 low6" PHASE=0 bash $C lma 0.10 "$seed" "$P"
             STATES="high16 low6" PHASE=0 bash $C lmb 0.01 "$seed" "$P"
             STATES="high16 low6" PHASE=0 WEAK=0.5 bash $C lmc 0.01 "$seed" "$P"
           done ;;
  lowsyn)  needs_synth
           for e in 6:low6s 4.5:low45 7.5:low75; do
             $BUILD --base $R/dataset_la_low8 --base-esno 8 --output $R/dataset_la_${e#*:} --trajectory const:${e%%:*}
           done
           probe low6s ;;
  low45)   needs_synth; three low45 lpa lpb lpc ;;
  low75)   needs_synth; three low75 lqa lqb lqc ;;
  umi)     three umi10 lua lub luc 0.03 ;;
  low6s3)  seeds="3"; three low6 lwa lwb lwc ;;
esac
echo V20_CONT_DONE $step
