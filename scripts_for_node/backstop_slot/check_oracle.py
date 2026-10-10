#!/usr/bin/env python3
"""Smoke check of the clairvoyant baseline: exit code 0 if the run with the oracle finished and used the table.

usage: check_oracle.py TAG JOB      (runs TAG1n, TAG1r32wd, TAG1r32wo0 of v25_oracle.sh smoke)
"""
import json
import sys
from pathlib import Path

from analyze_sweep import metrics
from loss_trace import RAW

tag, job = sys.argv[1], sys.argv[2]
rows, ok = {}, True
for code in ("n", "r32wd", "r32wo0"):
    found = sorted(RAW.glob(f"{tag}1{code}_c16_*_j{job}.json"))
    if len(found) != 1:
        print(f"{code}: no result"); ok = False
        continue
    m = metrics(found[0])
    c = json.loads(found[0].read_text())["controller"]["counters"]
    rows[code] = (m, c)
    print(f"{code}: AI {m['ai_slo'] / 1e3:.1f}k tokens/s, recovered {m['recovered_per_s']:.0f}/s, L1 late {m['l1_late_pct']:.3f}%, "
          f"runs {c['assigned']}, dropped {c['dropped']}, oracle needed {c.get('oracle_needed', '-')}, not needed {c.get('oracle_free', '-')}")
if "r32wo0" in rows:
    m, c = rows["r32wo0"]
    if not c.get("oracle_free"):
        print("the oracle marked no run as not needed"); ok = False
    if "n" in rows and m["recovered_per_s"] < 0.9 * rows["n"][0]["recovered_per_s"]:
        print("the oracle lost more than 10% of the recoveries"); ok = False
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
