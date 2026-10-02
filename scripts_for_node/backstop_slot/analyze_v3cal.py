#!/usr/bin/env python3
"""Calibration table: conventional completion and L1 misses vs AI overlap per period."""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1]
PERIOD = 2_500_000


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "raw" / f"k*_j{JOB}.json"))):
        name = Path(path).name
        m = re.match(r"k(n|x(\d+)c(\d+))_c(\d+)_", name)
        if not m:
            continue
        d = json.loads(Path(path).read_text())
        h, a = d["headline"], d["all"]
        ai = h.get("ai_total") or {}
        cross = pieces = 0
        busy = 0.0
        scales = []
        for f in glob.glob(path.replace(".json", "_work") + "/ai*.json"):
            w = json.loads(Path(f).read_text())
            ep = w["epoch_ns"]
            scales.append(w["summary"].get("unit_bound_scale", {}))
            for s, e, _, _, g in w["pieces"]:
                pieces += 1
                busy += g
                cross += e > ep + ((s - ep) // PERIOD + 1) * PERIOD
        seconds = h["tbs"] / h["cells"] * 2.5e-3
        x = None if m.group(1) == "n" else {"0": 0.0, "05": 0.5, "10": 1.0, "20": 2.0}[m.group(2)]
        rows.append({
            "cells": int(m.group(4)), "chunk": int(m.group(3)) if m.group(3) else None, "overlap_ms": x,
            "conv_p50": a["conv_done_ms"]["p50"], "conv_p99": a["conv_done_ms"]["p99"],
            "conv_p999": a["conv_done_ms"]["p999"], "l1_late": h["late_tbs"],
            "l1_late_pct": 100 * h["late_tbs"] / h["tbs"],
            "ai_tokens_per_s": ai.get("tokens_per_s", 0.0),
            "ai_busy_frac": busy / 1000 / seconds / d["config"]["num_gpus"] if seconds else 0.0,
            "pieces": pieces, "cross_pct": 100 * cross / max(1, pieces),
            "scale": {c: round(float(np.mean([s[c] for s in scales if c in s])), 2)
                      for c in ("128", "512", "1024")} if scales else {},
        })
    (ROOT / f"v3_calibration_j{JOB}.json").write_text(json.dumps(rows, indent=2))
    print("| cells | AI unit | overlap/period | conv p50/p99/p99.9 ms | L1 late (%) | AI tokens/s | AI busy | pieces crossing arrival | bound scale |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["cells"], r["chunk"] or 0, -1 if r["overlap_ms"] is None else r["overlap_ms"])):
        print(f"| {r['cells']} | {r['chunk'] or '-'} | {'no AI' if r['overlap_ms'] is None else r['overlap_ms']} | "
              f"{r['conv_p50']:.2f}/{r['conv_p99']:.2f}/{r['conv_p999']:.2f} | {r['l1_late']} ({r['l1_late_pct']:.3f}) | "
              f"{r['ai_tokens_per_s']:.0f} | {100*r['ai_busy_frac']:.0f}% | {r['cross_pct']:.1f}% | {r['scale']} |")


if __name__ == "__main__":
    main()
