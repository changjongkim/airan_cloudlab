#!/usr/bin/env python3
"""Compliant AI capacity per density setting (tags e{seed}{setting}...).

A load is compliant when, on every seed, L1 misses are at most 0.05% of TBs and NeuralRx
rescues reach 99% of the mean of that seed's two no-AI runs.
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
SETTINGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["a", "b"]
L1_TARGET, RESCUE_TARGET = 0.0005, 0.99
NAMES = {"u": "Our Scheme", "s30": "fixed 30% share", "s50": "fixed 50% share", "s70": "fixed 70% share",
         "s100": "fixed 100% share"}


def load(path):
    d = json.loads(Path(path).read_text())
    h = d["headline"]
    ai = h.get("ai_total") or {}
    return {"cells": h["cells"], "tbs": h["tbs"], "late": h["late_tbs"], "rescues": h["nrx_rescues_on_time"],
            "nrx_late": h.get("nrx_late") or 0, "slo": ai.get("tokens_within_slo_per_s", 0.0),
            "conv_p999": d["all"]["conv_done_ms"]["p999"]}


def main() -> None:
    out = {"targets": {"l1": L1_TARGET, "rescue": RESCUE_TARGET}, "settings": {}}
    for s in SETTINGS:
        refs, runs = {}, {}
        for path in glob.glob(str(ROOT / "raw" / f"e[0-9]{s}*_j{JOB}.json")):
            m = re.match(rf"e(\d){s}(n|m|r(\d+)(u|s\d+))_c\d+_", Path(path).name)
            if not m:
                continue
            seed = int(m.group(1))
            r = load(path)
            if m.group(2) in ("n", "m"):
                refs.setdefault(seed, []).append(r)
            else:
                runs.setdefault((m.group(4), int(m.group(3))), {})[seed] = r
        ref = {seed: float(np.mean([r["rescues"] for r in rs])) for seed, rs in refs.items()}
        ref_l1 = {seed: [r["late"] / r["tbs"] for r in rs] for seed, rs in refs.items()}
        rows = []
        for (key, rate), by_seed in sorted(runs.items(), key=lambda kv: (kv[0][0] != "u", kv[0][0], kv[0][1])):
            seeds = [x for x in sorted(by_seed) if x in ref]
            if not seeds:
                continue
            ratio = [by_seed[x]["rescues"] / max(1.0, ref[x]) for x in seeds]
            late = [by_seed[x]["late"] / by_seed[x]["tbs"] for x in seeds]
            rows.append({"policy": NAMES.get(key, key), "ai_rate": rate, "seeds": len(seeds),
                         "slo": float(np.mean([by_seed[x]["slo"] for x in seeds])),
                         "rescue_min": min(ratio), "rescue_mean": float(np.mean(ratio)),
                         "l1_max": max(late), "nrx_late": float(np.mean([by_seed[x]["nrx_late"] for x in seeds])),
                         "compliant": len(seeds) >= 2 and all(a >= RESCUE_TARGET for a in ratio)
                                      and all(b <= L1_TARGET for b in late)})
        cap = {}
        for r in rows:
            c = cap.setdefault(r["policy"], {"best": 0.0, "at": None, "first_fail": None})
            if r["compliant"] and r["slo"] > c["best"]:
                c["best"], c["at"] = r["slo"], r["ai_rate"]
        for policy, c in cap.items():
            fails = [r for r in rows if r["policy"] == policy and not r["compliant"]
                     and (c["at"] is None or r["ai_rate"] > c["at"])]
            if fails:
                first = min(fails, key=lambda r: r["ai_rate"])
                c["first_fail"] = {"ai_rate": first["ai_rate"], "slo": first["slo"]}
        out["settings"][s] = {"no_ai_rescues": ref, "no_ai_l1": ref_l1, "rows": rows, "capacity": cap}
        print(f"\n## setting {s}: no-AI rescues per seed {ref}, no-AI L1 late {ref_l1}")
        print("| policy | AI load | SLO tokens/s | rescues min (mean) | L1 late max | NRx late | compliant |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['policy']} | {r['ai_rate']} | {r['slo']:.0f} | {100*r['rescue_min']:.1f}% ({100*r['rescue_mean']:.1f}%) | "
                  f"{100*r['l1_max']:.3f}% | {r['nrx_late']:.0f} | {'yes' if r['compliant'] else 'no'} |")
        for policy, c in cap.items():
            print(f"  capacity {policy}: {c['best']:.0f} (load {c['at']}), first failing load after it: {c['first_fail']}")
    (ROOT / f"density_capacity_j{JOB}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
