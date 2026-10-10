#!/usr/bin/env bash
# Tables of v23_review.sh from the raw runs (login node; no GPU).  The jobs are read from queue/done.
#   usage: bash v23_report.sh [seeds pre base]      (default: all)   -> results/backstop_slot/v23_*.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
jobs_of() { local out=""; for n in "$@"; do j=$(awk -F'\t' -v n="$n" '$1==n{print $3}' $Q/done | tail -1); case ",$out," in *",$j,"*) ;; *) [ -n "$j" ] && out="${out:+$out,}$j" ;; esac; done; echo "$out"; }
say() { echo; echo "## $*"; echo; }
cd scripts_for_node/backstop_slot
for step in ${*:-seeds pre base}; do
  case $step in
    seeds) J=$(jobs_of seeds_a seeds_b)
      { say "Eight two-user cells, target 1%, seeds 10-33 (led; jobs $J)"; $PY analyze_la_closed.py $R/la_closed_led.json $J led 16
        say "All seeds of this condition (earlier measurements laf, lfc, lsc, ltc, lec and the new seeds led)"
        $PY analyze_seed_interval.py $R/seed_interval_8cells_1pct.json wm $R/la_closed_laf.json $R/la_closed_lfc.json $R/la_closed_lsc.json $R/la_closed_ltc.json $R/la_closed_lec.json $R/la_closed_led.json
      } > $R/v23_seeds.txt 2>&1 ;;
    pre) J=$(jobs_of pre_a pre_b)
      { say "Conditions never run before: predicted (commit 3b915a8a) against measured (jobs $J)"
        $PY predict_new_conditions.py compare $R/predicted_new_conditions_measured.json $R/predicted_new_conditions.json $J
      } > $R/v23_predicted.txt 2>&1 ;;
    base) J=$(jobs_of cl_a cl_b cl_c fixed)
      { say "Baselines one rule away, four two-user cells, target 10% (lba; jobs $J)"; $PY analyze_la_closed.py $R/la_closed_lba.json $J lba 16
        say "Baselines one rule away, four two-user cells, target 1% (lbb)"; $PY analyze_la_closed.py $R/la_closed_lbb.json $J lbb 16
        say "Baselines one rule away, eight two-user cells, target 1% (lbc)"; $PY analyze_la_closed.py $R/la_closed_lbc.json $J lbc 16
        say "Own deadline only at full load, fixed MCS (fx)"; V14=1 OUR_V14=wd $PY analyze_sweep.py $J fx 16
        say "Three-slot runs (fx)"; $PY analyze_slack.py $R/slack_fx.json fx 16 n,wr0,wd $J
        for t in lba lbb lbc; do say "Three-slot runs ($t)"; $PY analyze_slack.py $R/slack_$t.json $t 16 n,p100,wn,wr0,wm $J; done
      } > $R/v23_baselines.txt 2>&1 ;;
  esac
done
echo V23_REPORT_DONE
