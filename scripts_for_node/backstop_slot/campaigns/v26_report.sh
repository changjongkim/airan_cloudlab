#!/usr/bin/env bash
# Runs of 100 s for the unit-size rule: the rule (wd) against stop-on-NeuralRx only (wn: every unit size next to the
# conventional receiver), full load, seeds 1-3 (v16_long.sh long16 with POLS="n wn wd").  Login node; no GPU.
#   -> results/backstop_slot/v26_long_wn.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
J=$(awk -F'\t' '$1=="long_wn"{print $3}' $Q/done | tail -1)
cd scripts_for_node/backstop_slot
{ echo; echo "## Runs of 100 s without their first 20 s, full load, seeds 1-3 (xa; job $J): wd = the rule, wn = stop on NeuralRx only"; echo
  V14=1 OUR_V14=wd WARMUP_S=20 $PY analyze_sched.py $J xa 16 wd wn
  echo; echo "## The same runs with their first 20 s"; echo
  V14=1 OUR_V14=wd $PY analyze_sched.py $J xa 16 wd wn
} > $R/v26_long_wn.txt 2>&1
echo V26_REPORT_DONE
