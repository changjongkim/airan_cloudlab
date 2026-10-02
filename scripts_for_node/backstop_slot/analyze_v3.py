#!/usr/bin/env python3
"""Step 3 table: per density and policy, radio targets and AI served, plus compliant capacity.

Tags: {prefix}{seed}n|m (no AI), {prefix}{seed}r{rate}{policy} with policy v3, v2, s{pct}, o1, o2.
A load is compliant when, on every seed, L1 misses <= 0.05% of TBs and NeuralRx rescues
>= 99% of the mean of that seed's no-AI runs.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "h"
L1_TARGET, RESCUE_TARGET = 0.0005, 0.99
NAMES = {"v3": "Our Scheme v3", "v2": "Our Scheme v2", "v3b": "Our Scheme v3b", "v4": "Our Scheme", "v3c": "Our Scheme v3c", "v3d": "Our Scheme v3d", "v2b": "Our Scheme v2b", "o1": "AI only while radio idle",
         "o2": "smallest AI unit always"}


def load(path):
    d = json.loads(Path(path).read_text())
    h = d["headline"]
    ai = h.get("ai_total") or {}
    seconds = h["tbs"] / h["cells"] * 2.5e-3 if h["tbs"] else 10.0
    periods = int(d["config"]["periods"])
    seconds = periods * float(d["config"]["period_ms"]) / 1e3
    return {"cells": h["cells"], "tbs": h["tbs"], "late": h["late_tbs"], "rescues": h["nrx_rescues_on_time"],
            "nrx_late": h.get("nrx_late") or 0, "slo": ai.get("tokens_within_slo_per_s", 0.0),
            "served": ai.get("tokens_per_s", 0.0),
            "busy": ai.get("gpu_busy_ms", 0.0) / 1e3 / seconds / d["config"]["num_gpus"],
            "overruns": ai.get("piece_overruns", 0) / max(1, ai.get("pieces", 0))}


def main() -> None:
    runs, refs = {}, {}
    for path in glob.glob(str(ROOT / "raw" / f"{PREFIX}[0-9]*_j{JOB}.json")):
        m = re.match(rf"{PREFIX}(\d)(n|m|r(\d+)(v4|v3e\d+|v3d|v3c|v3b|v2b|v3|v2|s\d+|o1|o2))_c(\d+)_", Path(path).name)
        if not m:
            continue
        seed, cells = int(m.group(1)), int(m.group(5))
        r = load(path)
        if m.group(2) in ("n", "m"):
            refs.setdefault((cells, seed), []).append(r)
        else:
            runs.setdefault((cells, m.group(4), int(m.group(3))), {})[seed] = r
    out = {"targets": {"l1": L1_TARGET, "rescue": RESCUE_TARGET}, "rows": [], "capacity": []}
    for cells in sorted({k[0] for k in runs}):
        ref = {seed: float(np.mean([r["rescues"] for r in rs])) for (c, seed), rs in refs.items() if c == cells}
        ref_l1 = {seed: [round(100 * r["late"] / r["tbs"], 3) for r in rs] for (c, seed), rs in refs.items() if c == cells}
        print(f"\n## {cells} cells ({cells // 4} per GPU): no-AI rescues {ref}, no-AI L1 late % {ref_l1}")
        print("| policy | AI load | SLO tokens/s | AI GPU busy | rescues min (mean) | L1 late max | NRx late | compliant |")
        print("|---|---|---|---|---|---|---|---|")
        cap = {}
        keys = sorted((k for k in runs if k[0] == cells), key=lambda k: (not k[1].startswith("v"), k[1], k[2]))
        for key in keys:
            by_seed = runs[key]
            seeds = [x for x in sorted(by_seed) if x in ref]
            if not seeds:
                continue
            ratio = [by_seed[x]["rescues"] / max(1.0, ref[x]) for x in seeds]
            late = [by_seed[x]["late"] / by_seed[x]["tbs"] for x in seeds]
            row = {"cells": cells, "policy": NAMES.get(key[1], f"Our Scheme {key[1]}" if key[1].startswith("v") else f"fixed {key[1][1:]}% share"), "key": key[1],
                   "ai_rate": key[2], "seeds": len(seeds),
                   "slo": float(np.mean([by_seed[x]["slo"] for x in seeds])),
                   "busy": float(np.mean([by_seed[x]["busy"] for x in seeds])),
                   "rescue_min": min(ratio), "rescue_mean": float(np.mean(ratio)), "l1_max": max(late),
                   "l1_mean": float(np.mean(late)),
                   "nrx_late": float(np.mean([by_seed[x]["nrx_late"] for x in seeds])),
                   "compliant": all(a >= RESCUE_TARGET for a in ratio) and all(b <= L1_TARGET for b in late)}
            out["rows"].append(row)
            print(f"| {row['policy']} | {row['ai_rate']} | {row['slo']:.0f} | {100*row['busy']:.0f}% | "
                  f"{100*row['rescue_min']:.1f}% ({100*row['rescue_mean']:.1f}%) | {100*row['l1_max']:.3f}% | "
                  f"{row['nrx_late']:.0f} | {'yes' if row['compliant'] else 'no'}{'' if row['seeds'] > 1 else ' (1 seed)'} |")
            c = cap.setdefault(row["policy"], {"cells": cells, "policy": row["policy"], "best": 0.0, "at": None,
                                               "max_any": 0.0})
            c["max_any"] = max(c["max_any"], row["slo"])
            if row["compliant"] and row["slo"] > c["best"]:
                c["best"], c["at"] = row["slo"], row["ai_rate"]
        for c in cap.values():
            out["capacity"].append(c)
            print(f"  capacity {c['policy']}: {c['best']:.0f} (load {c['at']}); highest served regardless of targets {c['max_any']:.0f}")
    (ROOT / f"v3_density_{PREFIX}_j{JOB}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
