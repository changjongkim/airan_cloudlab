#!/usr/bin/env python3
"""Variant runs against the no-AI references of another tag.

usage: analyze_variants.py JOB REF_TAG CELLS TAG [TAG ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
job, ref_tag, cells = sys.argv[1], sys.argv[2], sys.argv[3]
refs = {}
for path in glob.glob(str(ROOT / f"{ref_tag}[0-9][nm]_c{cells}_*_j{job}.json")):
    seed = int(Path(path).name[len(ref_tag)])
    refs.setdefault(seed, []).append(json.loads(Path(path).read_text())["headline"]["nrx_rescues_on_time"])
refs = {seed: float(np.mean(v)) for seed, v in refs.items()}
print("| run | AI load | tokens/s within limit | rescues (per seed) | L1 late (per seed) | NRx late |")
print("|---|---|---|---|---|---|")
for tag in sys.argv[4:]:
    rows = {}
    for path in sorted(glob.glob(str(ROOT / f"{tag}[0-9]r*_c{cells}_*_j{job}.json"))):
        m = re.match(rf"{tag}(\d)r(\d+)(\w+?)_c", Path(path).name)
        h = json.loads(Path(path).read_text())["headline"]
        rows.setdefault((int(m.group(2)), m.group(3)), []).append((int(m.group(1)), h))
    for (rate, policy), items in sorted(rows.items()):
        slo = np.mean([(h.get("ai_total") or {}).get("tokens_within_slo_per_s", 0.0) for _, h in items])
        ratio = ", ".join(f"{100 * h['nrx_rescues_on_time'] / refs[s]:.1f}%" for s, h in items if s in refs)
        late = ", ".join(f"{100 * h['late_tbs'] / h['tbs']:.3f}%" for _, h in items)
        nrx_late = np.mean([h.get("nrx_late") or 0 for _, h in items])
        print(f"| {tag} {policy} | {rate} | {slo:.0f} | {ratio} | {late} | {nrx_late:.0f} |")
