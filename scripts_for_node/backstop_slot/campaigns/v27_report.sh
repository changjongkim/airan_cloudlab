#!/usr/bin/env bash
# Tables of v27_oracle_more.sh and of the second long-run check (login node; no GPU).
#   -> results/backstop_slot/v27_oracle_more.txt, v26_long_wn20.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
job_of() { awk -F'\t' -v n="$1" '$1==n{print $3}' $Q/done | tail -1; }
cd scripts_for_node/backstop_slot
JD=$(job_of or_demand); JL=$(job_of or_load); J20=$(job_of long_wn20)
{ for item in "ox:two two-user cells:$JD" "oy:six two-user cells:$JD" "ow:eight two-user cells:$JD" "og:load alternates every 2 s:$JL" "oh:load steps at random times:$JL"; do
    IFS=: read -r tag label job <<<"$item"
    echo; echo "## Clairvoyant controller, $label ($tag; job $job)"; echo
    [ -n "$job" ] && V14=1 OUR_V14=wd $PY analyze_sweep.py $job $tag 16
  done
} > $R/v27_oracle_more.txt 2>&1
{ echo; echo "## 20 cells, runs of 100 s without their first 20 s, seeds 1-3 (xv; job $J20): wd = the rule, wn = stop on NeuralRx only"; echo
  [ -n "$J20" ] && V14=1 OUR_V14=wd WARMUP_S=20 $PY analyze_sched.py $J20 xv 20 wd wn
  echo; echo "## The same runs with their first 20 s"; echo
  [ -n "$J20" ] && V14=1 OUR_V14=wd $PY analyze_sched.py $J20 xv 20 wd wn
} > $R/v26_long_wn20.txt 2>&1
JR=$(job_of ring1000)
{ echo; echo "## Replay of 1,000 slots per cell, headline condition, seeds 1-3, 20 s (fr; job $JR)"; echo
  [ -n "$JR" ] && V14=1 OUR_V14=wd $PY analyze_sweep.py $JR fr 16
  echo; echo "## Replay of 512 slots, the same condition, seeds 1-5 of the earlier jobs (fa, 10 s)"; echo
  V14=1 OUR_V14=wm $PY analyze_sweep.py 59313958,59210955 fa 16
} > $R/v28_ring1000.txt 2>&1
echo V27_REPORT_DONE
