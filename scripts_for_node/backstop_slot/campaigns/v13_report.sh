#!/usr/bin/env bash
# v13 report: tables and figures of every sweep.   usage: bash v13_report.sh JOB[,JOB...]
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
source scripts_for_node/backstop_slot/slot_env.sh 2>/dev/null
J=$1; JJ=${J//,/_}
S=${TMPDIR_MPL:-/tmp}
PY="shifter --image=$AERIAL_IMAGE --env=PYTHONPATH=$PWD/runtime/softwall_same_gpu/sionna_deps --env=MPLCONFIGDIR=$S python3"
A=scripts_for_node/backstop_slot
R=results/backstop_slot
have() { local j; for j in ${J//,/ }; do ls $R/raw/${1}[0-9]*_c${2}_*_j$j.json >/dev/null 2>&1 && return 0; done; return 1; }
sweep() { have $1 $2 && { echo "## $3"; $PY $A/analyze_sweep.py $J $1 $2; echo; }; }
sweep q 16 "AI load, 16 cells, steady full load"
sweep ha 16 "load alternates full/half every 0.2 s"
sweep hb 16 "load alternates full/half every 2 s"
sweep hc 16 "load alternates full/half every 5 s"
sweep st 16 "load steps between 25/50/75/100%"
sweep wa 16 "0 weak cells"; sweep wb 16 "2 weak cells"; sweep wc 16 "6 weak cells"; sweep wd 16 "8 weak cells"
sweep ga 4 "1 GPU, 4 cells"; sweep gb 8 "2 GPUs, 8 cells"
sweep ca 32 "32 cells, half load"; sweep cb 48 "48 cells, one-third load"
sweep ba 32 "cell bursts 20 ms"; sweep bb 32 "cell bursts 200 ms"; sweep bc 32 "cell bursts 2 s"
sweep ab 16 "AI arrives in bursts"
sweep da 20 "20 cells, full load"
sweep bd 16 "16 cells, cell bursts 20 ms"; sweep be 16 "16 cells, cell bursts 2 s"
have st 16 && { echo "## step load by level"; $PY $A/analyze_series.py $J st 16 100; }
P="$PY $A/plot_v13.py"
f() { echo $R/sweep_${1}_c${2}_j$JJ.json; }
have q 16 && { $P aiload $(f q 16) $R/v13_aiload_j$JJ.png; $P frontier $(f q 16) 32 $R/v13_frontier_load32_j$JJ.png "16 cells, steady full load, AI offered 53k tokens/s"; $P frontier $(f q 16) 64 $R/v13_frontier_load64_j$JJ.png "16 cells, steady full load, AI offered 106k tokens/s"; }
have hc 16 && $P axis $R/v13_phase_j$JJ.png "Time between load changes (s)" 0.2=$(f ha 16) 2=$(f hb 16) 5=$(f hc 16)
have hb 16 && $P frontier $(f hb 16) 32 $R/v13_frontier_phase2s_j$JJ.png "16 cells, load alternates between full and half every 2 s"
have st 16 && { $P frontier $(f st 16) 32 $R/v13_frontier_steps_j$JJ.png "16 cells, load jumps between 25/50/75/100% at random times"; $P series $R/series_st_c16_j$JJ.json 1 $R/v13_steps_series_j$JJ.png; }
have wd 16 && $P axis $R/v13_nrx_demand_j$JJ.png "Cells that need the neural receiver (of 16)" 0=$(f wa 16) 2=$(f wb 16) 4=$(f q 16) 6=$(f wc 16) 8=$(f wd 16)
have gb 8 && $P axis $R/v13_gpus_j$JJ.png "GPUs (four cells each)" 1=$(f ga 4) 2=$(f gb 8) 4=$(f q 16)
have cb 48 && $P axis $R/v13_cells_j$JJ.png "Cells on four GPUs (same offered radio load)" 16=$(f q 16) 32=$(f ca 32) 48=$(f cb 48)
have bc 32 && $P axis $R/v13_burst32_j$JJ.png "Mean length of a cell's busy run (s), 32 cells" 0.02=$(f ba 32) 2=$(f bc 32)
have be 16 && $P axis $R/v13_burst16_j$JJ.png "Mean length of a cell's busy run (s), 16 cells" 0.02=$(f bd 16) 2=$(f be 16)
