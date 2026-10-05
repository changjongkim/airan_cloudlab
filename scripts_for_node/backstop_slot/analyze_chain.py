#!/usr/bin/env python3
"""Closed-form recovery loss (loss_model.chain_markov) against the measured runs.

usage: analyze_chain.py OUT.json JOB:TAG:CELLS [JOB:TAG:CELLS ...]
Inputs of the closed form per run:
  T   transition matrix of the number of candidates per slot, from the no-AI run of the seed
  f   share of the neural receiver runs of the run that hold their receiver for three slots,
      h = ceil((start offset + run length - latest start) / period) >= 3
  N   number of neural receivers
Output: share of the candidates that find no free neural receiver, measured and closed form;
also the closed form with independent Binomial arrivals (loss_model.chain).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from loss_model import arrival_transitions, chain, chain_markov
from loss_trace import RAW, trace


def main() -> None:
    out_path = Path(sys.argv[1])
    rows = []
    for spec in sys.argv[2:]:
        job, tag, cells = spec.split(":")
        pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
        names = sorted(p.stem for p in RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json"))
        refs = {}
        for name in names:
            m = pattern.match(name)
            if m and m.group(2) is None:
                refs[int(m.group(1))] = trace(name)
        for name in names:
            m = pattern.match(name)
            if not m or int(m.group(1)) not in refs or (m.group(2) is not None and int(m.group(2)) != 32):
                continue
            seed, policy = int(m.group(1)), (m.group(3) or "n")
            ref, run = refs[seed], trace(name)
            lanes, weak, periods = int(ref["lanes"]), int(ref["weak_cells"]), int(ref["periods"])
            ran = run["start_ms"] >= 0
            latest = float(ref["rescue_ms"]) - float(ref["bound_ms"])
            held = np.ceil((run["start_ms"][ran] + run["run_ms"][ran] - latest) / 2.5)
            f = float((held >= 3).mean()) if ran.any() else 0.0
            transitions = arrival_transitions(ref["period"], periods, weak)
            q = len(ref["period"]) / (periods * weak)
            rows.append({"condition": f"{tag} c{cells}", "seed": seed, "policy": policy, "lanes": lanes, "weak_cells": weak,
                         "three_slot_share": f, "candidates": int(len(ref["period"])),
                         "lost_measured": float((~ran).sum() / len(ref["period"])),
                         "lost_chain_markov": chain_markov(lanes, transitions, f),
                         "lost_chain_independent": chain(lanes, weak, q, f)})
    out_path.write_text(json.dumps({"rows": rows}, indent=1))
    print("| condition | policy | seeds | three-slot runs | lost: measured | closed form, dependent arrivals | closed form, independent arrivals |")
    print("|---|---|---|---|---|---|---|")
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["policy"]), []).append(r)
    for (condition, policy), items in groups.items():
        mean = lambda key: float(np.mean([i[key] for i in items]))
        print(f"| {condition} | {policy} | {len(items)} | {100 * mean('three_slot_share'):.0f}% | {100 * mean('lost_measured'):.1f}% | "
              f"{100 * mean('lost_chain_markov'):.1f}% | {100 * mean('lost_chain_independent'):.1f}% |")
    measured = np.array([100 * r["lost_measured"] for r in rows])
    for key, label in (("lost_chain_markov", "dependent arrivals"), ("lost_chain_independent", "independent arrivals")):
        model = np.array([100 * r[key] for r in rows])
        print(f"\n{label}: runs {len(rows)}, correlation {np.corrcoef(measured, model)[0, 1]:.3f}, "
              f"mean absolute error {np.abs(measured - model).mean():.2f} points, mean error {np.mean(model - measured):+.2f} points")


if __name__ == "__main__":
    main()
