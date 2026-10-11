#!/usr/bin/env bash
# Tables of v31_info.sh and v32_sawtooth.sh (login node; no GPU).
#   -> results/backstop_slot/v31_info.txt, v32_sawtooth.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
jobs_of() { local out=""; for n in "$@"; do j=$(awk -F'\t' -v n="$n" '$1==n{print $3}' $Q/done | tail -1); case ",$out," in *",$j,"*) ;; *) [ -n "$j" ] && out="${out:+$out,}$j" ;; esac; done; echo "$out"; }
cd scripts_for_node/backstop_slot
JI=$(jobs_of info_1 info_2 info_3); JS=$(jobs_of saw_a saw_b)
{ echo; echo "## Clairvoyant controller with errors in its table, headline condition (fk; jobs $JI)"; echo
  [ -n "$JI" ] && $PY analyze_info.py $R/info_value.json $JI fk
} > $R/v31_info.txt 2>&1
{ echo; echo "## Recovery deadline 11.0-14.0 ms: predicted before the runs (commit 8d17f858) / measured (jobs $JS)"; echo
  [ -n "$JS" ] && $PY predict_deadline_sweep.py compare $R/predicted_deadline_sweep_measured.json $R/predicted_deadline_sweep.json $JS
} > $R/v32_sawtooth.txt 2>&1
echo V31_REPORT_DONE
