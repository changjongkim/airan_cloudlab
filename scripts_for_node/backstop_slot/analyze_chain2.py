#!/usr/bin/env python3
"""Closed form without a measured three-slot share (loss_model.lane_chain) against the runs.

usage: analyze_chain2.py OUT.json JOB:TAG:CELLS [JOB:TAG:CELLS ...]      (env RATES=8,16,32,64: AI loads to include; default 32)
Inputs of the closed form for a run of a policy:
  N   number of neural receivers
  T   transition matrix of the candidates per slot, from the no-AI run of the same seed
  c   times at which candidates became known, S run lengths of the neural receiver: samples from
      the runs of the same condition and policy with OTHER seeds (the run itself if it is the
      only seed; column "seeds for c, S" says which)
Output per condition and policy: lost candidates and three-slot share, measured and predicted.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from loss_model import arrival_transitions, lane_chain
from loss_trace import RAW, trace


def main() -> None:
    out_path = Path(sys.argv[1])
    rates = {int(r) for r in os.environ.get("RATES", "32").split(",")}
    rows = []
    for spec in sys.argv[2:]:
        job, tag, cells = spec.split(":")
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
        names = sorted(p.stem for p in RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json"))
        refs, groups = {}, {}
        for name in names:
            m = pattern.match(name)
            if not m or (m.group(2) is not None and int(m.group(2)) not in rates):
                continue
            t = trace(name)
            if m.group(2) is None:
                refs[int(m.group(1))] = t
            policy = m.group(3) or "n"
            if m.group(2) is not None and int(m.group(2)) != 32:
                policy = f"{policy}@{m.group(2)}"            # another AI load of the same policy
            groups.setdefault(policy, {})[int(m.group(1))] = t
        for policy, by_seed in groups.items():
            for seed, run in by_seed.items():
                if seed not in refs:
                    continue
                ref = refs[seed]
                others = [t for s, t in by_seed.items() if s != seed] or [run]
                known = np.concatenate([t["known_ms"] for t in others])
                length = np.concatenate([t["run_ms"][t["start_ms"] >= 0] for t in others])
                lanes, weak, periods = int(ref["lanes"]), int(ref["weak_cells"]), int(ref["periods"])
                latest = float(ref["rescue_ms"]) - float(ref["bound_ms"])
                model = lane_chain(lanes, arrival_transitions(ref["period"], periods, weak), known, length,
                                   latest_start_ms=latest)
                ran = run["start_ms"] >= 0
                rows.append({"condition": f"{tag} c{cells}", "seed": seed, "policy": policy, "lanes": lanes, "weak_cells": weak,
                             "held_out": len(by_seed) > 1,
                             "lost_measured": float((~ran).sum() / len(ref["period"])), "lost_model": model["lost_share"],
                             "three_slot_measured": float((run["start_ms"][ran] + run["run_ms"][ran] > 5.0 + latest).mean()),
                             "three_slot_model": model["three_slot_share"], "waited_model": model["waited_share"]})
    out_path.write_text(json.dumps({"rows": rows}, indent=1))
    print("| condition | policy | seeds | c, S from other seeds | lost: measured | closed form | three-slot runs: measured | closed form |")
    print("|---|---|---|---|---|---|---|---|")
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["policy"]), []).append(r)
    for (condition, policy), items in groups.items():
        mean = lambda key: float(np.mean([i[key] for i in items]))
        print(f"| {condition} | {policy} | {len(items)} | {'yes' if items[0]['held_out'] else 'no'} | {100 * mean('lost_measured'):.1f}% | "
              f"{100 * mean('lost_model'):.1f}% | {100 * mean('three_slot_measured'):.0f}% | {100 * mean('three_slot_model'):.0f}% |")
    for label, pick in (("all runs", rows), ("runs with c, S from other seeds", [r for r in rows if r["held_out"]])):
        if not pick:
            continue
        a = np.array([100 * r["lost_measured"] for r in pick])
        b = np.array([100 * r["lost_model"] for r in pick])
        fa_ = np.array([100 * r["three_slot_measured"] for r in pick])
        fb_ = np.array([100 * r["three_slot_model"] for r in pick])
        print(f"\n{label}: {len(pick)}; lost candidates: correlation {np.corrcoef(a, b)[0, 1]:.3f}, mean absolute error "
              f"{np.abs(a - b).mean():.2f} points, mean error {np.mean(b - a):+.2f} points; three-slot share: mean absolute error "
              f"{np.abs(fa_ - fb_).mean():.1f} points")


if __name__ == "__main__":
    main()
