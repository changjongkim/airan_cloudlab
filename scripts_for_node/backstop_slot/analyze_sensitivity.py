#!/usr/bin/env python3
"""How much a longer neural receiver run costs: closed form against the measured policies.

usage: analyze_sensitivity.py OUT.json JOB[,JOB] TAG CELLS POLICY ...
Curve: loss_model.lane_chain with the candidates of the no-AI runs (arrival transitions, times
known) and the run lengths of the no-AI runs plus a constant delay of 0 .. 2.4 ms; a second
curve adds the delay of the conventional result that AI causes (median over the policies).
Points: per policy, the median run length of the neural receiver minus the median without AI,
and the measured share of candidates that got no neural receiver.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from loss_model import arrival_transitions, lane_chain
from loss_trace import RAW, trace


def main() -> None:
    out_path, jobs, tag, cells = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3], sys.argv[4]
    policies = sys.argv[5:]
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r(\d+)([a-z0-9]+))_c{cells}_")
    refs, runs = {}, {}
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if not m or (m.group(2) is not None and int(m.group(2)) != 32):
                continue
            t = trace(path.stem)
            if m.group(2) is None:
                refs[(job, int(m.group(1)))] = t
            elif m.group(3) in policies:
                runs.setdefault(m.group(3), []).append((job, int(m.group(1)), t))
    known = np.concatenate([t["known_ms"] for t in refs.values()])
    alone = np.concatenate([t["run_ms"][t["start_ms"] >= 0] for t in refs.values()])
    first = next(iter(refs.values()))
    lanes, weak = int(first["lanes"]), int(first["weak_cells"])
    latest = float(first["rescue_ms"]) - float(first["bound_ms"])
    transitions = np.mean([arrival_transitions(t["period"], int(t["periods"]), weak) for t in refs.values()], axis=0)
    base = float(np.median(alone))
    points = []
    for policy, items in runs.items():
        lost = float(np.mean([(t["start_ms"] < 0).sum() / len(refs[(job, seed)]["period"]) for job, seed, t in items]))
        length = float(np.median(np.concatenate([t["run_ms"][t["start_ms"] >= 0] for _, _, t in items])))
        shift = float(np.median(np.concatenate([t["known_ms"] for _, _, t in items])) - np.median(known))
        three = float(np.mean([(t["start_ms"][t["start_ms"] >= 0] + t["run_ms"][t["start_ms"] >= 0] > 5.0 + latest).mean() for _, _, t in items]))
        points.append({"policy": policy, "seeds": len(items), "extra_run_ms": length - base, "known_shift_ms": shift, "lost": lost,
                       "three_slot": three})
    lost_ref = float(np.mean([(t["start_ms"] < 0).mean() for t in refs.values()]))
    three_ref = float(np.mean([(t["start_ms"][t["start_ms"] >= 0] + t["run_ms"][t["start_ms"] >= 0] > 5.0 + latest).mean() for t in refs.values()]))
    shift = float(np.median([p["known_shift_ms"] for p in points])) if points else 0.0
    curve = []
    for delay in np.arange(0.0, 2.41, 0.1):
        a = lane_chain(lanes, transitions, known, alone + delay, latest_start_ms=latest)
        b = lane_chain(lanes, transitions, known + shift, alone + delay, latest_start_ms=latest)
        curve.append({"extra_run_ms": float(delay), "lost": a["lost_share"], "three_slot": a["three_slot_share"],
                      "lost_with_known_shift": b["lost_share"], "three_slot_with_known_shift": b["three_slot_share"]})
        print(f"delay {delay:.1f} ms: lost {100 * a['lost_share']:.2f}% (three-slot runs {100 * a['three_slot_share']:.0f}%), "
              f"with the conventional result {shift:.2f} ms later: {100 * b['lost_share']:.2f}%")
    out_path.write_text(json.dumps({"lanes": lanes, "weak_cells": weak, "run_alone_ms": base, "known_shift_ms": shift,
                                    "lost_no_ai": lost_ref, "three_slot_no_ai": three_ref, "curve": curve, "points": points}, indent=1))
    for p in sorted(points, key=lambda p: p["extra_run_ms"]):
        print(f"{p['policy']}: run +{p['extra_run_ms']:.2f} ms, conventional result +{p['known_shift_ms']:.2f} ms, lost {100 * p['lost']:.2f}%, three-slot runs {100 * p['three_slot']:.0f}% ({p['seeds']} seeds)")
    print(f"no AI: lost {100 * lost_ref:.2f}%")


if __name__ == "__main__":
    main()
