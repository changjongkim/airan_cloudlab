#!/usr/bin/env bash
# Tables of v29_followup.sh (login node; no GPU).  -> results/backstop_slot/v29_cap_deadline.txt, v29_long.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
jobs_of() { local out=""; for n in "$@"; do j=$(awk -F'\t' -v n="$n" '$1==n{print $3}' $Q/done | tail -1); case ",$out," in *",$j,"*) ;; *) [ -n "$j" ] && out="${out:+$out,}$j" ;; esac; done; echo "$out"; }
say() { echo; echo "## $*"; echo; }
cd scripts_for_node/backstop_slot
for step in ${*:-cap long oracle2}; do
  case $step in
    cap) J=$(jobs_of cap_a cap_b)
      { for item in "lcy:eight two-user cells, target 1%, deadline 12.0 ms" "lcz:eight two-user cells, target 1%, deadline 12.5 ms" "ldb:four two-user cells, target 1%, deadline 12.5 ms" "lda:four two-user cells, target 10%, deadline 12.5 ms"; do
          say "70% cap with a later recovery deadline: ${item#*:} (${item%%:*}; jobs $J)"; $PY analyze_la_closed.py $R/la_closed_${item%%:*}.json $J ${item%%:*} 16
        done
      } > $R/v29_cap_deadline.txt 2>&1 ;;
    long) J20=$(jobs_of long20_a long20_b); J32=$(jobs_of long32)
      { say "20 cells, runs of 100 s without their first 20 s, seeds 1-8 (xv; jobs 59642023,$J20): wd = the rule, wn = stop on NeuralRx only"
        V14=1 OUR_V14=wd WARMUP_S=20 $PY analyze_sched.py 59642023,$J20 xv 20 wd wn
        say "32 cells at half load, runs of 100 s without their first 20 s, seeds 1-3 (xm; job $J32)"
        [ -n "$J32" ] && V14=1 OUR_V14=wd WARMUP_S=20 $PY analyze_sched.py $J32 xm 32 wd wn
      } > $R/v29_long.txt 2>&1 ;;
    oracle2) J=$(jobs_of or2_full)
      { say "Clairvoyant controller with the reuse-time rule, headline condition, seeds 1-5 (ft; job $J): wo1 = AI for a whole run or not, wq1 / wq0 = also next to needed runs while the reuse time holds (one / no receiver kept free)"
        V14=1 OUR_V14=wd $PY analyze_sweep.py $J ft 16
      } > $R/v30_oracle2.txt 2>&1 ;;
  esac
done
echo V29_REPORT_DONE
