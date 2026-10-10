#!/usr/bin/env bash
# Tables of v25_oracle.sh from the raw runs (login node; no GPU).  -> results/backstop_slot/v25_oracle.txt
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/global/common/software/nersc/pe/conda-envs/26.8.1/python-3.13/nersc-python/bin/python3
Q=$PWD/run_state/backstop_slot/queue; R=$PWD/results/backstop_slot
J=$(awk -F'\t' '$1=="or_full"{print $3}' $Q/done | tail -1)
cd scripts_for_node/backstop_slot
{ echo; echo "## Clairvoyant controller, headline condition, seeds 1-5 (fo; job $J): wo0 = no receiver kept free beyond the known demand, wo1 = one"; echo
  V14=1 OUR_V14=wd $PY analyze_sweep.py $J fo 16
  echo; echo "## Neural receiver runs and the AI beside them (fo)"; echo
  $PY analyze_slack.py $R/slack_fo.json fo 16 n,wd,wo0,wo1,p100 $J
  echo; echo "## Controller counters (runs marked needed / not needed)"; echo
  $PY - <<PYEOF
import json, glob
for code in ("wo0", "wo1"):
    need = free = 0
    for p in sorted(glob.glob("$R/raw/fo[0-9]r32%s_c16_*_j$J.json" % code)):
        c = json.load(open(p))["controller"]["counters"]
        need += c.get("oracle_needed", 0); free += c.get("oracle_free", 0)
    if need + free:
        print(f"{code}: AI allowed next to {100 * free / (need + free):.1f}% of the neural receiver runs ({free} of {need + free})")
PYEOF
} > $R/v25_oracle.txt 2>&1
echo V25_REPORT_DONE
