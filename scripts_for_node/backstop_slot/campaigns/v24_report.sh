#!/usr/bin/env bash
# Tables of v24_deadline.sh from the raw runs (login node; no GPU).  -> results/backstop_slot/v24_deadline.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
jobs_of() { local out=""; for n in "$@"; do j=$(awk -F'\t' -v n="$n" '$1==n{print $3}' $Q/done | tail -1); case ",$out," in *",$j,"*) ;; *) [ -n "$j" ] && out="${out:+$out,}$j" ;; esac; done; echo "$out"; }
say() { echo; echo "## $*"; echo; }
cd scripts_for_node/backstop_slot
J=$(jobs_of dl_fixed dl_cl)
{ say "Recovery deadline 12.0 and 12.5 ms, headline condition: predicted before the run against measured (jobs $J)"
  $PY predict_new_conditions.py deadline_compare $R/predicted_deadline_measured.json $R/predicted_deadline.json $J
  say "AI served and recoveries, deadline 12.0 ms (fy)"; V14=1 OUR_V14=wd $PY analyze_sweep.py $J fy 16
  say "AI served and recoveries, deadline 12.5 ms (fz)"; V14=1 OUR_V14=wd $PY analyze_sweep.py $J fz 16
  say "Eight two-user cells, target 1%, deadline 12.0 ms (lby)"; $PY analyze_la_closed.py $R/la_closed_lby.json $J lby 16
  say "Eight two-user cells, target 1%, deadline 12.5 ms (lbz)"; $PY analyze_la_closed.py $R/la_closed_lbz.json $J lbz 16
} > $R/v24_deadline.txt 2>&1
echo V24_REPORT_DONE
